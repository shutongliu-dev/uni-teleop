"""Trusted practice -> independent validation -> immutable competition scheduler.

The scheduler supplies no robot strategy. Each learner owns an empty library,
model contexts, publications and public feedback. Bench remains the physical
authority. Unresolved infrastructure/accounting stops expenditure and is retained.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time

from .assessment import (FreezeManifest, PublishedCandidate, ValidationSlot,
    assess_validation, content_sha256, public_validation_feedback, select_candidate,
    select_budget_fallback, sum_effort, validate_freeze)
from .bindings import SkillBinding
from .curriculum import StudyBudget, bind_continuation, prepare_study_plan
from .journal import atomic_write_json, canonical_json, read_json
from .skills import SkillLibrary


@dataclass
class StudyConfig:
    gym_url: str
    formal_url: str
    gym_release: Path
    gym_episode_root: Path
    formal_episode_root: Path
    output: Path
    player_models: Path = field(repr=False)
    model_slot: str = 'player_1'
    parent_task_id: str = 'polyomino_territory'
    budget: StudyBudget = field(default_factory=StudyBudget)
    full_curriculum: bool = False
    model_retry_policy: str | None = None
    model_retries: int | None = None
    continuation: Path | None = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value.model_dump(mode='json') if hasattr(value, 'model_dump') else value,
                      replace=False)


def public_model_condition(binding, *, retry_policy=None, retries=None):
    """Pin only public routing; credentials and their file paths stay private."""
    from ..transport import ModelGatewayConfig
    transport = ModelGatewayConfig.from_file(binding.credentials).public_metadata()
    if binding.provider in ('micu', 'qwen'):
        transport.update(endpoint=transport['base_url']+'/responses', protocol='responses')
    elif binding.provider != 'gateway':
        raise ValueError('Unknown model provider')
    condition = {**binding.public_metadata(), **transport}
    if retry_policy is not None:
        from ..transport import GATEWAY_RETRY_POLICY, gateway_retry_metadata
        if binding.provider != 'gateway' or retry_policy != GATEWAY_RETRY_POLICY:
            raise ValueError('Unsupported study retry policy for this provider')
        condition.update(gateway_retry_metadata(retries))
    elif retries is not None:
        raise ValueError('Explicit study retries require an explicit retry policy')
    return condition


def publications(library):
    result = []
    for event in library.journal.read():
        if event['kind'] != 'candidate_committed':
            continue
        payload = event['payload']
        library.verify(payload['bundle_hash'])
        catalog = read_json(library.bundles / payload['bundle_hash'] / 'payload' / 'catalog.json')
        result.append(PublishedCandidate(candidate_event_id=event['event_id'],
            bundle_hash=payload['bundle_hash'], publication_sequence=event['sequence'],
            is_empty=not catalog['skills'],
            author='inherited_model' if payload.get('import') else 'model'))
    return result


def _needs_stop(report):
    result = report.get('result') or {}
    physical = result.get('episode', result)
    return bool(report.get('run_error') or report.get('finalization_error') or report.get('result') is None
                or report.get('action_recovery_pending')
                or physical.get('outcome') in ('policy_error', 'infrastructure_error', 'evaluator_invalid')
                or physical.get('physics_valid') is False
                or report.get('run_status') in ('infra_error', 'configuration_error', 'protocol_invalid'))


def recent_public_executions(folder):
    """Re-present bounded own public feedback, never host/private/native files."""
    result = []
    for path in reversed(sorted((folder / 'single' / 'decisions').glob('*/execution.json'))):
        feedback = read_json(path)
        if feedback.get('function') in ('skill_write', 'skill_read', 'skill_status'):
            continue
        if len(canonical_json(feedback)) <= 12288:
            result.append(feedback)
        if len(result) == 4:
            break
    return list(reversed(result))


def freeze_library(library, plan, assessments, validation_events, effort, ledger, *, learner_id,
                   prior_effort=()):
    """Qualification and budget freeze are distinct, explicit host decisions."""
    selected = select_candidate(assessments)
    if selected is not None:
        candidate_id, bundle = selected.candidate_event_id, selected.bundle_hash
        ids = validation_events[candidate_id]
        event = library.select(candidate_id, validation_event_ids=ids,
            gate={'approved': True, 'protocol_id': 'card_rsi_evaluation_plan.v1',
                  'assessment_sha256': content_sha256(selected)}, transaction_id='study-freeze')
        event_id, mode = event['selection_event_id'], 'validated_rank'
    else:
        candidate = select_budget_fallback(plan.validation, publications(library), budget_exhausted=True)
        if candidate is None:
            return None
        candidate_id, bundle = candidate.candidate_event_id, candidate.bundle_hash
        selected = next((a for a in assessments if a.candidate_event_id == candidate_id), None)
        ids = validation_events.get(candidate_id, [])
        event = library.journal.append('budget_frozen', {
            'candidate_event_id': candidate_id, 'bundle_hash': bundle,
            'validation_plan_sha256': content_sha256(plan.validation),
            'qualification': False, 'budget_fallback': plan.validation.budget_fallback})
        event_id, mode = event['event_id'], 'budget_fallback'
    reasons = list(selected.reasons) if selected else ['no_validation_assessment']
    if mode == 'budget_fallback':
        reasons.append('declared_budget_fallback')
    evaluation = plan.validation
    frozen = FreezeManifest(learner_id=learner_id, learning_run_id='study',
        **{key: getattr(evaluation, key) for key in ('gym_id', 'parent_task_id',
           'parent_release_sha256', 'public_rules_sha256', 'condition_sha256', 'source_manifest_sha256')},
        candidate_event_id=candidate_id, bundle_hash=bundle, selection_event_id=event_id,
        validation_plan_sha256=content_sha256(evaluation), selection_mode=mode,
        validation_assessment_sha256=content_sha256(selected) if selected else None,
        validation_event_ids=ids, validation_complete=bool(selected and selected.all_slots_resolved),
        immutable_validation_complete=bool(selected and selected.immutable_execution_complete),
        qualification_reasons=reasons,
        qualified_for_declared_curriculum=bool(mode == 'validated_rank' and selected.qualified_for_declared_curriculum),
        freeze_sequence=len(ledger), development_cost=sum_effort([*prior_effort, *effort]),
        development_ledger_sha256=content_sha256(ledger))
    validate_freeze(frozen, evaluation, selected)
    return frozen


def run_study(config: StudyConfig, *, source_verification, episode_runner=None,
              receipt_collector=None, effort_collector=None, feedback_projector=None, match_effort_collector=None):
    from ..function_runner import RunConfig, run
    from ..player_models import load_player_models
    from .receipts import (collect_effort, collect_match_effort, collect_trial_record,
                           collect_revision_receipt, public_practice_feedback)
    episode_runner = episode_runner or run
    receipt_collector = receipt_collector or collect_trial_record
    effort_collector = effort_collector or collect_effort
    feedback_projector = feedback_projector or public_practice_feedback
    match_effort_collector = match_effort_collector or collect_match_effort
    binding = load_player_models(config.player_models)[config.model_slot]
    model_condition = public_model_condition(binding, retry_policy=config.model_retry_policy,
                                             retries=config.model_retries)
    source_sha = source_verification['manifest_sha256']
    plan = prepare_study_plan(config.gym_release, parent_task_id=config.parent_task_id,
        model=model_condition, source_manifest_sha256=source_sha,
        budget=config.budget, full_curriculum=config.full_curriculum)
    continuation = None
    if config.continuation is not None:
        from .continuation import load_continuation
        continuation = load_continuation(config.continuation, plan)
        plan = bind_continuation(plan, continuation.checkpoint_sha256)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    if continuation is not None:
        write(output / 'continuation_request.json', read_json(config.continuation))
        write(output / 'continuation.json', continuation.manifest)
    write(output / 'study_plan.json', plan)
    write(output / 'source_verification.json', source_verification)
    # Both players use the requested provider, with separate libraries and histories.
    write(output / 'model_condition.json', {'model': model_condition,
        'learners': ['player_0', 'player_1'], 'independent_libraries': True,
        'same_model_self_play': True, 'operator_manipulation_recipe': False})
    budget = plan.budget
    libraries, freezes, ledger, efforts = {}, {}, [], []
    unaccounted_attempts = []
    summary = {'schema': 'card_rsi_study_report.v1', 'status': 'running',
        'model': model_condition, 'learners': {},
        'closed_loop_verified': False, 'formal_match_completed': False}
    inherited_efforts = ([] if continuation is None else
        [cost for player in ('player_0', 'player_1') for cost in continuation.prior_efforts(player)])
    if continuation is not None:
        summary.update(continuation_sha256=continuation.checkpoint_sha256,
            inherited_cost=sum_effort(inherited_efforts).model_dump(mode='json'),
            inherited_qualification=False)

    def receipt_event(kind, value):
        entry = {'sequence': len(ledger), 'kind': kind, 'value': value}
        write(output / 'ledger' / f'{len(ledger):06d}.json', entry)
        ledger.append(entry)
        atomic_write_json(output / 'progress.json', {'stage': kind, 'sequence': entry['sequence'],
            'model': model_condition, 'value': value})

    def verify_model_condition():
        if public_model_condition(binding, retry_policy=config.model_retry_policy,
                                  retries=config.model_retries) != model_condition:
            raise ValueError('Public model transport changed after study conditions were frozen')

    def run_trial(player, library, phase, selection, attempt_id, candidate=None):
        folder = output / player / attempt_id
        pin = SkillBinding(library, phase,
            candidate.candidate_event_id if candidate else None,
            candidate.bundle_hash if candidate else None)
        previous = summary['learners'][player]['public_feedback'] if phase == 'practice' else None
        cfg = RunConfig(bench_url=config.gym_url, family='card_room', task=plan.gym_id,
            output=folder, practice=selection, provider=binding.provider, model=binding.model,
            reasoning_effort=binding.reasoning_effort, credentials=binding.credentials,
            model_retry_policy=config.model_retry_policy, model_retries=config.model_retries,
            max_decisions=budget.max_decisions_per_episode, max_actions=budget.max_actions_per_episode,
            max_function_calls=budget.max_function_calls_per_episode,
            max_action_sim_s=budget.max_action_sim_seconds, sim_budget=budget.sim_seconds_per_episode,
            wall_budget=budget.wall_seconds_per_episode, max_output_tokens=budget.max_output_tokens,
            model_timeout=None, http_timeout=None, history_mode='current_with_recall', geometry_math=True)
        trial_binding = {'schema': 'card_rsi_trial_binding.v1', 'phase': phase,
            'attempt_id': attempt_id, 'condition_sha256': plan.validation.condition_sha256,
            'source_manifest_sha256': source_sha, 'candidate_event_id': pin.candidate_event_id,
            'bundle_hash': pin.bundle_sha256}
        verify_model_condition()
        receipt_event('trial_start', {'learner': player, 'attempt_id': attempt_id,
            'phase': phase, 'selection': selection, 'candidate_event_id': pin.candidate_event_id})
        accounting = {'attempt_id': attempt_id, 'phase': phase, 'learners': [player],
                      'output_relative': folder.relative_to(output).as_posix()}
        unaccounted_attempts.append(accounting)
        started, cpu = time.monotonic(), time.process_time()
        report = episode_runner(cfg, source_verification=source_verification,
            learning_bindings={'single': pin}, learning_feedback={'single': previous} if previous else None,
            learning_trial=trial_binding)
        wall_s, cpu_s = time.monotonic() - started, time.process_time() - cpu
        accounting['episode_id'] = report.get('episode_id')
        cost = effort_collector(folder, wall_s=wall_s, cpu_s=cpu_s)
        efforts.append(cost)
        unaccounted_attempts.remove(accounting)
        receipt_event('trial_end', {'learner': player, 'attempt_id': attempt_id,
            'episode_id': report.get('episode_id'), 'cost': cost.model_dump(mode='json')})
        if _needs_stop(report):
            raise RuntimeError('Unresolved episode or finalization; attempt retained, automatic retry disabled')
        episode_root = config.gym_episode_root / report['episode_id']
        return report, folder, episode_root, wall_s, cpu_s

    try:
        for player in ('player_0', 'player_1'):
            library = SkillLibrary.create(player, learning_root=output / 'libraries', contract={
                'dependencies': {}, 'compatibility': {'task_ids': [plan.gym_id, plan.parent_task_id],
                    'task_releases': {plan.gym_id: plan.gym_release_sha256,
                                      plan.parent_task_id: plan.parent_release_sha256}},
                'condition_sha256': plan.validation.condition_sha256,
                'source_manifest_sha256': source_sha, 'model': model_condition})
            libraries[player] = library
            completed_slots, prior_efforts, inherited_feedback = (), [], []
            if continuation is not None:
                from .continuation import seed_library
                seeded = seed_library(continuation, player, library)
                completed_slots = continuation.completed_slots(player)
                prior_efforts = continuation.prior_efforts(player)
                inherited_feedback = continuation.public_feedback(player)
                receipt_event('library_continued', {'learner': player,
                    'continuation_sha256': continuation.checkpoint_sha256,
                    'seed_receipt_sha256': content_sha256(seeded)})
            info = summary['learners'][player] = {'public_feedback': [], 'validation': [], 'freeze': None,
                'revision_receipts': [], 'autonomous_revision_verified': False,
                'independent_validation_verified': False}
            info['public_feedback'] = inherited_feedback
            if continuation is not None:
                info['completed_in_predecessor'] = list(completed_slots)
            assessments, validation_events, validated = [], {}, set()
            practice_records = []
            effort_start = len(efforts)
            # Distribute practice across predeclared batches; feedback is disclosed
            # only when a complete validation batch has closed.
            for batch in range(budget.validation_batches):
                begin = batch * len(plan.practice_slots) // budget.validation_batches
                end = (batch + 1) * len(plan.practice_slots) // budget.validation_batches
                for index in range(begin, end):
                    slot = plan.practice_slots[index]
                    attempt = f'practice_{index:04d}'
                    if slot.slot_id in completed_slots:
                        receipt_event('practice_slot_inherited', {'learner': player,
                            'attempt_id': attempt, 'selection': slot.selection,
                            'continuation_sha256': continuation.checkpoint_sha256})
                        continue
                    report, folder, episode_root, wall_s, cpu_s = run_trial(
                        player, library, 'practice', slot.selection, attempt)
                    published = publications(library)
                    record = None
                    if published:
                        candidate = published[-1]
                        public = slot.case['public']
                        case_slot = ValidationSlot(slot_id=slot.slot_id,
                            **{k: public[k] for k in ('active_side', 'stratum', 'case_id', 'case_sha256')},
                            case_fingerprint=slot.case['case_fingerprint'],
                            private_case_sha256=slot.case['private_case_sha256'])
                        record = receipt_collector(plan.validation, case_slot,
                            candidate_event_id=candidate.candidate_event_id, bundle_hash=candidate.bundle_hash,
                            agent_output=folder, episode_root=episode_root, attempt_id=attempt,
                            attempt_index=0, receipt_sequence=len(ledger), phase='practice',
                            wall_s=wall_s, cpu_s=cpu_s)
                        write(folder / 'host' / 'trial_record.json', record)
                    feedback = feedback_projector(record, report)
                    info['public_feedback'].append({'assessment': feedback,
                        'recent_public_executions': recent_public_executions(folder)})
                    write(folder / 'host' / 'public_learning_feedback.json', feedback)
                    receipt_event('practice_feedback', {'learner': player, 'attempt_id': attempt,
                        'public_feedback_sha256': content_sha256(feedback),
                        'publications': [p.model_dump(mode='json') for p in publications(library)]})
                    if published:
                        candidate = published[-1]
                        previous = {r.candidate_event_id for r in practice_records if r.native_result()
                                    and r.candidate_event_id != candidate.candidate_event_id}
                        for previous_id in sorted(previous):
                            diagnostic = {}
                            revision = collect_revision_receipt(library_root=library.path,
                                previous_candidate_event_id=previous_id,
                                candidate_event_id=candidate.candidate_event_id, agent_output=folder,
                                practice_records=practice_records, receipt_sequence=len(ledger), diagnostics=diagnostic)
                            write(folder / 'host' / ('revision_' + previous_id + '.json'), {
                                'diagnostics': diagnostic,
                                'receipt': revision.model_dump(mode='json') if revision else None})
                            if revision is not None:
                                info['revision_receipts'].append(revision.model_dump(mode='json'))
                                info['autonomous_revision_verified'] = True
                                receipt_event('model_revision_verified', {'learner': player,
                                    'candidate_event_id': candidate.candidate_event_id,
                                    'receipt_sha256': content_sha256(revision)})
                    if record is not None:
                        practice_records.append(record)
                published = publications(library)
                if not published or published[-1].candidate_event_id in validated:
                    continue
                candidate = published[-1]
                records, event_ids = [], []
                for index, slot in enumerate(plan.validation.slots):
                    attempt = f'validation_{batch:02d}_{index:03d}'
                    selection = {'active_side': slot.active_side, 'case_id': slot.case_id, 'split': 'validation'}
                    report, folder, episode_root, wall_s, cpu_s = run_trial(
                        player, library, 'validation', selection, attempt, candidate)
                    record = receipt_collector(plan.validation, slot,
                        candidate_event_id=candidate.candidate_event_id, bundle_hash=candidate.bundle_hash,
                        agent_output=folder, episode_root=episode_root, attempt_id=attempt,
                        attempt_index=0, receipt_sequence=len(ledger), phase='validation',
                        wall_s=wall_s, cpu_s=cpu_s)
                    write(folder / 'host' / 'trial_record.json', record)
                    records.append(record)
                    event = library.record_validation(candidate.candidate_event_id, {
                        'source': 'bench', 'protocol_id': 'card_rsi_evaluation_plan.v1',
                        'slot_id': slot.slot_id, 'attempt_id': attempt, 'task_outcome': record.outcome,
                        'valid': record.native_result() and record.immutable_skill_execution(),
                        'execution_status': record.execution_status, 'cost': record.cost.model_dump(mode='json'),
                        'receipt_sha256': content_sha256(record)}, transaction_id=attempt)
                    event_ids.append(event['event_id'])
                    receipt_event('validation_receipt', {'learner': player, 'attempt_id': attempt,
                        'receipt_sha256': content_sha256(record)})
                assessment = assess_validation(plan.validation, records,
                    candidate_event_id=candidate.candidate_event_id, bundle_hash=candidate.bundle_hash,
                    condition_sha256=plan.validation.condition_sha256)
                assessments.append(assessment)
                validation_events[candidate.candidate_event_id] = event_ids
                validated.add(candidate.candidate_event_id)
                write(output / player / f'assessment_{batch:02d}.json', assessment)
                info['validation'].append(assessment.model_dump(mode='json'))
                info['independent_validation_verified'] |= bool(
                    assessment.all_slots_resolved and assessment.immutable_execution_complete)
                info['public_feedback'].append(public_validation_feedback(assessment, batch_closed=True))
            frozen = freeze_library(library, plan, assessments, validation_events, efforts[effort_start:],
                                    ledger, learner_id=player, prior_effort=prior_efforts)
            if frozen is None:
                summary['status'] = 'no_published_candidate'
                return summary
            write(output / player / 'freeze_manifest.json', frozen)
            freezes[player] = frozen
            info['freeze'] = frozen.model_dump(mode='json')
            receipt_event('learner_frozen', {'learner': player, 'bundle_hash': frozen.bundle_hash,
                'selection_mode': frozen.selection_mode,
                'qualified_for_declared_curriculum': frozen.qualified_for_declared_curriculum})
        # The same model provider is intentionally used for both independent
        # learners. No cross-provider config or private context is reused here.
        cfg = RunConfig(bench_url=config.formal_url, family='card_room', task=plan.parent_task_id,
            output=output / 'match', provider=binding.provider, model=binding.model,
            reasoning_effort=binding.reasoning_effort, credentials=binding.credentials,
            model_retry_policy=config.model_retry_policy, model_retries=config.model_retries,
            max_decisions=budget.match_max_decisions_per_player, max_actions=budget.match_max_actions,
            max_function_calls=budget.max_function_calls_per_episode,
            max_action_sim_s=budget.max_action_sim_seconds, sim_budget=budget.match_sim_seconds,
            wall_budget=budget.match_wall_seconds, model_timeout=None, http_timeout=None,
            max_output_tokens=budget.max_output_tokens, history_mode='current_with_recall', geometry_math=True,
            stop_after_commits_per_player=budget.match_commits_per_player)
        match_bindings = {p: SkillBinding(libraries[p], 'evaluation', f.candidate_event_id, f.bundle_hash)
                          for p, f in freezes.items()}
        author_before = {p: library.author_revision() for p, library in libraries.items()}
        verify_model_condition()
        receipt_event('frozen_match_start', {'bundles': {p: f.bundle_hash for p, f in freezes.items()},
            'target_commits_per_player': budget.match_commits_per_player})
        accounting = {'attempt_id': 'match', 'phase': 'evaluation',
                      'learners': ['player_0', 'player_1'], 'output_relative': 'match'}
        unaccounted_attempts.append(accounting)
        started, cpu = time.monotonic(), time.process_time()
        report = episode_runner(cfg, source_verification=source_verification, learning_bindings=match_bindings)
        accounting['episode_id'] = report.get('episode_id')
        match_cost = match_effort_collector(cfg.output, wall_s=time.monotonic()-started, cpu_s=time.process_time()-cpu)
        efforts.append(match_cost)
        unaccounted_attempts.remove(accounting)
        summary['match_cost'] = match_cost.model_dump(mode='json')
        if author_before != {p: library.author_revision() for p, library in libraries.items()}:
            raise RuntimeError('Author content changed during frozen match')
        summary['match_report'] = 'match/run_report.json'
        summary['status'] = 'stages_executed' if not _needs_stop(report) else 'match_unresolved'
        summary['formal_match_completed'] = bool(report.get('result_valid_for_scoring'))
        summary['qualification_scope'] = 'See independent native receipts; stage scheduling alone is not RSI or gameplay success'
        receipt_event('frozen_match_end', {'episode_id': report.get('episode_id'), 'status': summary['status']})
        return summary
    except Exception as error:
        summary['status'] = 'stopped_error'
        summary['error_type'] = type(error).__name__
        # No exception text: transports/configuration may contain private values.
        raise
    finally:
        summary['cost_accounting_complete'] = not unaccounted_attempts
        summary['unaccounted_attempts'] = unaccounted_attempts
        summary['cost_scope'] = ('all_started_attempts' if not unaccounted_attempts
                                 else 'accounted_attempts_only; unaccounted costs are unknown, not zero')
        summary['cost'] = sum_effort(efforts).model_dump(mode='json')
        if continuation is not None:
            summary['cumulative_cost'] = sum_effort([*inherited_efforts, *efforts]).model_dump(mode='json')
            summary['cumulative_cost_scope'] = 'predecessor attempts plus this segment; unknown provider usage remains unknown'
        write(output / 'study_report.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('gym-url', 'formal-url'):
        parser.add_argument('--' + name, required=True)
    for name in ('gym-release', 'gym-episode-root', 'formal-episode-root', 'output', 'player-models'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task', default='polyomino_territory')
    parser.add_argument('--model-slot', choices=('player_0', 'player_1'), default='player_1')
    parser.add_argument('--budget', type=Path)
    parser.add_argument('--full-curriculum', action='store_true')
    parser.add_argument('--model-retry-policy', choices=('gateway_transient_backoff_v1',))
    parser.add_argument('--model-retries', type=int, help='Retry limit; omitted with transient policy means continue waiting')
    parser.add_argument('--continuation', type=Path, help='Explicit practice-only predecessor descriptor; output must be new')
    args = parser.parse_args()
    from .. import function_runner
    from . import runtime, receipts
    from scripts.maintenance.freeze_project_sources import verify_runtime_imports
    root = Path(__file__).resolve().parents[3]
    verification = verify_runtime_imports(root, required_modules=('agent.model_client.learning.curriculum',
        'agent.model_client.function_runner', 'agent.model_client.learning.runtime',
        'agent.model_client.learning.receipts'), entrypoint_file=Path(__file__))
    verification['manifest_sha256'] = hashlib.sha256((root / 'source_snapshot_manifest.json').read_bytes()).hexdigest()
    config = StudyConfig(gym_url=args.gym_url, formal_url=args.formal_url,
        gym_release=args.gym_release, gym_episode_root=args.gym_episode_root,
        formal_episode_root=args.formal_episode_root, output=args.output,
        player_models=args.player_models, model_slot=args.model_slot, parent_task_id=args.task,
        budget=StudyBudget.model_validate(read_json(args.budget)) if args.budget else StudyBudget(),
        full_curriculum=args.full_curriculum, model_retry_policy=args.model_retry_policy,
        model_retries=args.model_retries, continuation=args.continuation)
    run_study(config, source_verification=verification)


if __name__ == '__main__':
    main()
