"""Read-only posthoc verification of a completed RSI study and frozen match.

The runtime snapshot remains the source that actually ran. This auditor has its
own separately verified snapshot. It neither starts nor retries models/physics,
repairs evidence, nor treats archived qualification flags as native proof.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import re
import time

from .assessment import (Effort, FreezeManifest, TrialRecord, ValidationSlot,
    assess_validation, audit_closed_loop, content_sha256, sum_effort, validate_freeze)
from .curriculum import StudyPlan
from .journal import atomic_write_json, checked_directory, read_bytes, read_json
from .receipts import (_Reader, ReceiptError, _require, _source_record,
                       collect_effort, collect_match_effort, collect_revision_receipt, collect_trial_record)
from bench.protocol.schemas import Budgets

PLAYERS = ('player_0', 'player_1')
TOKEN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$')


def _token(value):
    _require(isinstance(value, str) and TOKEN.fullmatch(value) is not None, 'invalid_artifact_identity')
    return value


def _reason(error):
    # Only fixed internal ReceiptError codes cross the audit output boundary.
    return str(error) if isinstance(error, ReceiptError) else type(error).__name__


def _completion(job):
    if job is None or not (job/'completion.json').exists():
        return None
    value = read_json(job/'completion.json')
    _require(value.get('state') in ('command_exited', 'launch_failed'), 'unknown_job_completion')
    if value['state'] == 'command_exited':
        _require(type(value.get('exit_code')) is int, 'unknown_job_exit_code')
    return {k: value.get(k) for k in ('state', 'exit_code', 'completed_at_utc', 'elapsed_wall_s')}


def await_study_report(study, study_job, *, sleep=time.sleep):
    """Poll only local artifacts every 30 s; a completed job cannot wait forever."""
    study, job = checked_directory(study), checked_directory(study_job)
    while True:
        completion = _completion(job)
        if (study/'study_report.json').exists() and completion is not None:
            return {'report_available': True, 'job_completion': completion}
        if completion is not None:
            return {'report_available': False, 'job_completion': completion,
                    'reason': 'study_job_finished_without_report'}
        sleep(30)


def _ledger(reader):
    directory = checked_directory(reader.root/'ledger')
    paths = sorted(directory.iterdir())
    _require([p.name for p in paths] == [f'{i:06d}.json' for i in range(len(paths))], 'study_ledger_gap')
    result = []
    for index, path in enumerate(paths):
        row = reader.json('ledger/'+path.name)
        _require(set(row) == {'sequence', 'kind', 'value'} and type(row['sequence']) is int
            and row['sequence'] == index and isinstance(row['kind'], str)
            and isinstance(row['value'], dict), 'study_ledger_schema')
        result.append(row)
    return result


def _one(rows, kind, *, learner=None, attempt=None):
    matches = [r for r in rows if r['kind'] == kind
        and (learner is None or r['value'].get('learner') == learner)
        and (attempt is None or r['value'].get('attempt_id') == attempt)]
    _require(len(matches) == 1, 'missing_or_duplicate_'+kind)
    return matches[0]


def _accounting_complete(summary, rows):
    if summary.get('cost_accounting_complete') is not True or summary.get('unaccounted_attempts') != []:
        return False
    try:
        starts = [(r['value']['learner'], r['value']['attempt_id']) for r in rows if r['kind'] == 'trial_start']
        ends = [r for r in rows if r['kind'] == 'trial_end']
        identities = [(r['value']['learner'], r['value']['attempt_id']) for r in ends]
        if len(set(starts)) != len(starts) or sorted(starts) != sorted(identities):
            return False
        costs = [Effort.model_validate(r['value']['cost']) for r in ends]
        match_starts = [r for r in rows if r['kind'] == 'frozen_match_start']
        match_ends = [r for r in rows if r['kind'] == 'frozen_match_end']
        if len(match_starts) != len(match_ends) or len(match_starts) > 1:
            return False
        if match_starts:
            costs.append(Effort.model_validate(summary['match_cost']))
        return content_sha256(sum_effort(costs)) == content_sha256(summary['cost'])
    except (ValueError, TypeError, KeyError):
        return False


def _recheck_effort(reader, summary, rows, *, trial_collector, match_collector):
    """Recount API/action receipts even without a candidate or native success."""
    audit = {'complete': True, 'trials': [], 'match': None,
             'clock_scope': 'wall_s/cpu_s are inherited host measurements, not independently remeasured; all other effort fields are recounted from actual receipts.'}
    for start in (r for r in rows if r['kind'] == 'trial_start'):
        value = start['value']
        row = {'learner': value.get('learner'), 'attempt_id': value.get('attempt_id'), 'verified': False}
        audit['trials'].append(row)
        try:
            player, attempt = value['learner'], _token(value['attempt_id'])
            _require(player in PLAYERS, 'unknown_cost_learner')
            end = _one(rows, 'trial_end', learner=player, attempt=attempt)
            expected = Effort.model_validate(end['value']['cost'])
            actual = Effort.model_validate(trial_collector(reader.root/player/attempt,
                wall_s=expected.wall_s, cpu_s=expected.cpu_s))
            row.update(recorded=expected.model_dump(mode='json'), actual=actual.model_dump(mode='json'))
            _require(content_sha256(actual) == content_sha256(expected), 'actual_trial_effort_differs_from_ledger')
            # A missing terminal response can leave the original counter at
            # zero while native execution was real. Counter agreement alone
            # cannot certify that interval or a late worker finalization.
            if end['value'].get('episode_id') is not None:
                row['native_cost_verified'] = False
                run = reader.json(player+'/'+attempt+'/run_report.json')
                result = run.get('result')
                _require(isinstance(result, dict) and not run.get('finalization_error'),
                         'native_effort_unavailable_without_terminal_result')
                elapsed = result.get('elapsed_sim_s')
                _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                         'native_effort_terminal_clock_missing')
                _require(math.isclose(actual.native_sim_s, elapsed, rel_tol=0, abs_tol=1e-8),
                         'actual_native_effort_differs_from_terminal_result')
                row['native_cost_verified'] = True
            row['verified'] = True
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
            row['issue'] = _reason(error)
            audit['complete'] = False
    if any(r['kind'] == 'frozen_match_start' for r in rows):
        row = audit['match'] = {'verified': False}
        try:
            _one(rows, 'frozen_match_start')
            _one(rows, 'frozen_match_end')
            expected = Effort.model_validate(summary['match_cost'])
            actual = Effort.model_validate(match_collector(reader.root/'match',
                wall_s=expected.wall_s, cpu_s=expected.cpu_s))
            row.update(recorded=expected.model_dump(mode='json'), actual=actual.model_dump(mode='json'))
            _require(content_sha256(actual) == content_sha256(expected), 'actual_match_effort_differs_from_summary')
            row['verified'] = True
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
            row['issue'] = _reason(error)
            audit['complete'] = False
    return audit


def _slot(plan, start):
    value = start['value']
    if value['phase'] == 'practice':
        matches = [s for s in plan.practice_slots if s.slot_id == value['attempt_id']]
        _require(len(matches) == 1 and matches[0].selection == value['selection'], 'practice_slot_changed')
        case = matches[0].case
        return ValidationSlot(slot_id=matches[0].slot_id,
            **{k: case['public'][k] for k in ('active_side', 'stratum', 'case_id', 'case_sha256')},
            case_fingerprint=case['case_fingerprint'], private_case_sha256=case['private_case_sha256'])
    _require(value['phase'] == 'validation', 'unknown_trial_phase')
    matches = [s for s in plan.validation.slots if value['selection'] == {
        'active_side': s.active_side, 'case_id': s.case_id, 'split': 'validation'}]
    _require(len(matches) == 1, 'validation_slot_changed')
    return matches[0]


def _run_condition(reader, relative, plan, *, match=False):
    config = reader.json(relative+'/agent_config.json')
    budget = plan.budget
    decisions = budget.match_max_decisions_per_player if match else budget.max_decisions_per_episode
    expected = Budgets(sim_time_s=budget.match_sim_seconds if match else budget.sim_seconds_per_episode,
        wall_time_s=budget.match_wall_seconds if match else budget.wall_seconds_per_episode,
        max_actions=budget.match_max_actions if match else budget.max_actions_per_episode,
        max_model_calls=decisions*(2 if match else 1), max_action_sim_s=budget.max_action_sim_seconds).model_dump(mode='json')
    _require(config.get('budgets') == expected and config.get('max_decisions_per_player') == decisions
        and config.get('max_function_calls_per_player') == budget.max_function_calls_per_episode
        and config.get('history_mode') == 'current_with_recall'
        and config.get('local_geometry_math', {}).get('enabled') is True
        and config.get('visual_input') == 'rgbd', 'run_conditions_differ_from_plan')
    keys = PLAYERS if match else ('single',)
    _require(set(config.get('models', {})) == set(keys), 'run_model_players_changed')
    from .model_conditions import check_model_condition, check_retry_receipts
    for model in config['models'].values():
        check_model_condition(plan.model, model, reason='run_model_condition_changed')
    if match:
        target = config.get('validation_target')
        _require((target is None if budget.match_commits_per_player is None else target == {
            'commits_per_player': budget.match_commits_per_player,
            'source': 'verified_public_operation_committed_events', 'pass_counts': False,
            'game_completion_claimed': False}), 'formal_stop_target_changed')
    # Successful request receipts expose the actual per-request output bound.
    for player in keys:
        check_retry_receipts(reader, relative, player, plan.model)
        decisions_root = reader.root/relative/player/'decisions'
        started_calls = [folder for folder in decisions_root.glob('*') if folder.is_dir() and folder.name.isdigit()
                         and any((folder/name).exists() for name in ('model_request.json', 'model_response.json',
                                                                    'model_error.json', 'model_api_events.jsonl'))]
        _require(len(started_calls) <= decisions, 'model_logical_budget_exceeded')
        for path in sorted((reader.root/relative/player/'decisions').glob('*/model_response.json')):
            response = reader.json(str(path.relative_to(reader.root)))
            metadata = response['metadata']
            _require('max_output_tokens' in metadata and metadata['max_output_tokens'] == budget.max_output_tokens,
                     'model_output_budget_changed')


def _continuation(reader, plan, summary, rows, report):
    """Revalidate parent provenance and budget credit without inheriting success."""
    checkpoint = getattr(plan, 'continuation_sha256', None)
    if checkpoint is not None or 'retry_policy' in plan.model:
        from .curriculum import study_condition_sha256
        _require(plan.validation.condition_sha256 == study_condition_sha256(plan), 'study_condition_hash_changed')
    if checkpoint is None:
        _require(not (reader.root/'continuation.json').exists()
                 and not (reader.root/'continuation_request.json').exists()
                 and not any(r['kind'] in ('library_continued', 'practice_slot_inherited') for r in rows),
                 'undeclared_study_continuation')
        return None
    report['continuation'] = dict(checkpoint_sha256=checkpoint, manifest_verified=False,
        inherited_cost_verified=False, cumulative_cost_verified=False, inherited_qualification=False,
        required_libraries=[], libraries={}, verified=False)
    from .continuation import load_continuation
    reader.json('continuation_request.json')  # Retain the exact request hash in the audit inventory.
    manifest = reader.json('continuation.json')
    cont = load_continuation(reader.path('continuation_request.json'), plan)
    _require(content_sha256(manifest) == checkpoint == cont.checkpoint_sha256
             and content_sha256(cont.manifest) == checkpoint, 'continuation_checkpoint_changed')
    _require(summary.get('continuation_sha256') == checkpoint
             and summary.get('inherited_qualification') is False, 'continuation_summary_changed')
    prior = [effort for player in PLAYERS for effort in cont.prior_efforts(player)]
    _require(content_sha256(summary.get('inherited_cost')) == content_sha256(sum_effort(prior)),
             'inherited_cost_mismatch')
    current = [Effort.model_validate(r['value']['cost']) for r in rows if r['kind'] == 'trial_end']
    if any(r['kind'] == 'frozen_match_end' for r in rows):
        current.append(Effort.model_validate(summary['match_cost']))
    _require(content_sha256(summary.get('cumulative_cost')) == content_sha256(sum_effort(prior+current)),
             'cumulative_cost_mismatch')
    slots = {slot.slot_id: slot.selection for slot in plan.practice_slots}
    inherited = [r['value'] for r in rows if r['kind'] == 'practice_slot_inherited']
    _require(all(v.get('learner') in PLAYERS for v in inherited), 'inherited_slot_learner_changed')
    seeded = [r for r in rows if r['kind'] == 'library_continued']
    _require(all(r['value'].get('learner') in PLAYERS for r in seeded), 'continuation_seed_learner_changed')
    # A stopped segment may not have reached the second learner. Any credits
    # that it did write must be a complete, ordered prefix for that learner.
    for player in PLAYERS:
        started_library = [r for r in rows if r['kind'] == 'library_continued' and r['value'].get('learner') == player]
        _require(len(started_library) <= 1, 'duplicate_library_continuation')
        actual = [v for v in inherited if v.get('learner') == player]
        expected = [dict(learner=player, attempt_id=slot, selection=slots[slot], continuation_sha256=checkpoint)
                    for slot in cont.completed_slots(player)]
        _require(actual == expected if started_library else not actual, 'inherited_practice_prefix_changed')
        _require(not started_library or all(r['sequence'] > started_library[0]['sequence'] for r in rows
                 if r['kind'] == 'practice_slot_inherited' and r['value'].get('learner') == player),
                 'inherited_credit_precedes_seed')
        credited = {v['attempt_id'] for v in actual}
        _require(not any(r['kind'] == 'trial_start' and r['value'].get('learner') == player
                         and r['value'].get('attempt_id') in credited for r in rows),
                 'inherited_practice_slot_reexecuted')
    required = sorted({r['value']['learner'] for r in seeded}
                      | {p for p in PLAYERS if (reader.root/'libraries'/p).exists()})
    report['continuation'].update(manifest_verified=True, inherited_cost_verified=True,
        cumulative_cost_verified=True, required_libraries=required, verified=not required)
    return cont


def _continuation_library(reader, player, rows, cont, report):
    if cont is None:
        return
    from .continuation import verify_seed_library
    receipt = verify_seed_library(cont, player, reader.root/'libraries'/player)
    reader.json('libraries/'+player+'/host/continuation.json')
    event = _one(rows, 'library_continued', learner=player)
    _require(event['value'] == dict(learner=player, continuation_sha256=cont.checkpoint_sha256,
                                  seed_receipt_sha256=content_sha256(receipt)), 'continuation_seed_ledger_changed')
    report['continuation']['libraries'][player] = dict(verified=True, seed_receipt_sha256=content_sha256(receipt))
    report['continuation']['verified'] = (set(report['continuation']['libraries'])
                                         == set(report['continuation']['required_libraries']))


def _freeze(reader, player, plan, rows, records, *, diagnostics=None, prior_effort=()):
    freeze = FreezeManifest.model_validate(reader.json(player+'/freeze_manifest.json'))
    _require(freeze.learner_id == player and freeze.learning_run_id == 'study', 'freeze_learner_changed')
    n = freeze.freeze_sequence
    _require(n < len(rows) and content_sha256(rows[:n]) == freeze.development_ledger_sha256,
             'frozen_development_ledger_mismatch')
    event = rows[n]
    _require(event['kind'] == 'learner_frozen' and event['value'].get('learner') == player
        and event['value'].get('bundle_hash') == freeze.bundle_hash
        and event['value'].get('selection_mode') == freeze.selection_mode
        and event['value'].get('qualified_for_declared_curriculum') is freeze.qualified_for_declared_curriculum,
        'freeze_ledger_boundary_mismatch')
    costs = [Effort.model_validate(r['value']['cost']) for r in rows[:n]
             if r['kind'] == 'trial_end' and r['value'].get('learner') == player]
    _require(content_sha256(sum_effort(list(prior_effort)+costs)) == content_sha256(freeze.development_cost), 'freeze_cost_mismatch')
    if diagnostics is not None:
        diagnostics['archived_freeze'] = {
            'manifest': freeze.model_dump(mode='json'),
            'file_sha256': reader.hashes['study/'+player+'/freeze_manifest.json'],
            'development_ledger_verified': True,
            'scope': 'Original manifest and ledger binding; does not replace its recorded qualification or establish corrected validation.'}
    validation = [r for r in records if r.phase == 'validation' and r.candidate_event_id == freeze.candidate_event_id]
    assessment = (assess_validation(plan.validation, validation, candidate_event_id=freeze.candidate_event_id,
        bundle_hash=freeze.bundle_hash, condition_sha256=plan.validation.condition_sha256) if validation else None)
    validate_freeze(freeze, plan.validation, assessment)
    return freeze, assessment, validation


def auditor_source_identity():
    """Report this reader's source separately; editable source cannot qualify."""
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    # Load the bottom-level reader before recording the auditor import closure.
    from . import match_receipts
    root = Path(__file__).absolute().parents[3]
    record = verify_if_snapshot(root, required_modules=(
        'agent.model_client.learning.receipts', 'agent.model_client.learning.match_receipts'),
        entrypoint_file=Path(__file__))
    return {'verified': record is not None, 'snapshot_root': str(root),
        'manifest_sha256': hashlib.sha256(read_bytes(root/'source_snapshot_manifest.json')).hexdigest()
                             if record is not None else None,
        'module_sha256': hashlib.sha256(read_bytes(Path(__file__))).hexdigest(),
        'verification': record,
        'scope': 'Posthoc auditor source; never substituted for the runtime source.'}


def audit_study(study, *, practice_episodes, formal_episodes, auditor_identity=None,
                study_job=None, trial_collector=None, revision_collector=None, match_collector=None,
                effort_collector=None, match_effort_collector=None):
    """Recompute native/skill evidence; injected collectors are CPU test scope only."""
    injected = any(v is not None for v in (trial_collector, revision_collector, match_collector,
                                         effort_collector, match_effort_collector))
    trial_collector = trial_collector or collect_trial_record
    revision_collector = revision_collector or collect_revision_receipt
    effort_collector = effort_collector or collect_effort
    match_effort_collector = match_effort_collector or collect_match_effort
    report = {'schema': 'card_rsi_study_audit.v1', 'status': 'incomplete', 'issues': [], 'players': {},
        'closed_loop_verified': False, 'formal_match_completed': False,
        'cost_accounting_complete': False, 'runtime_source': None,
        'auditor_source': auditor_identity, 'injected_test_collectors': injected,
        'scope': 'Read-only posthoc evidence audit; no new model/native execution or performance-improvement claim.'}
    hashes = {}
    plan = None
    try:
        reader = _Reader(study, 'study', hashes)
        practice_root = checked_directory(practice_episodes)
        job = checked_directory(study_job) if study_job is not None else None
        report['job_completion'] = _completion(job)
        if not (reader.root/'study_report.json').exists():
            raise ReceiptError('study_report_missing')
        archived_summary = reader.json('study_report.json')
        report['study_status'] = archived_summary.get('status')
        _require(archived_summary.get('schema') == 'card_rsi_study_report.v1', 'unknown_study_report')
        missing = archived_summary.get('unaccounted_attempts')
        report['reported_cost_accounting_complete'] = archived_summary.get('cost_accounting_complete')
        report['cost_accounting_complete'] = archived_summary.get('cost_accounting_complete') is True and missing == []
        report['unaccounted_attempts'] = missing
        report['reported_cost'] = archived_summary.get('cost')
        if not report['cost_accounting_complete']:
            report['issues'].append('study_cost_accounting_incomplete')
        plan = StudyPlan.model_validate(reader.json('study_plan.json'))
        report['runtime_source'] = {'verified': False, 'manifest_sha256': plan.source_manifest_sha256}
        source = _source_record(reader, 'source_verification.json', plan.source_manifest_sha256,
                               ['agent.model_client.function_runner'])
        report['runtime_source'] = {'verified': True, 'manifest_sha256': plan.source_manifest_sha256,
                                   'snapshot_root': source['snapshot_root']}
        rows = _ledger(reader)
        continuation = _continuation(reader, plan, archived_summary, rows, report)
        report['effort_audit'] = _recheck_effort(reader, archived_summary, rows,
            trial_collector=effort_collector, match_collector=match_effort_collector)
        report['cost_accounting_complete'] = (_accounting_complete(archived_summary, rows)
                                              and report['effort_audit']['complete'])
        if not report['cost_accounting_complete'] and 'study_cost_accounting_incomplete' not in report['issues']:
            report['issues'].append('study_cost_accounting_incomplete')
        all_records, freezes, assessments, validations, all_revisions = {}, {}, {}, {}, {}
        for player in PLAYERS:
            detail = report['players'][player] = {'issues': [], 'trials': [], 'revisions': [],
                                                  'closed_loop': None}
            records, revisions = [], []
            all_records[player], all_revisions[player] = records, revisions
            try:
                library = _Reader(reader.root/'libraries'/player, 'library_'+player, hashes)
                experiment = library.json('host/experiment.json')
                journal = library.journal('host/journal')
                _require(journal and journal[0]['payload'].get('experiment_sha256') == content_sha256(experiment),
                         'library_origin_changed')
                contract = experiment['contract']
                _require(contract.get('model') == plan.model and contract.get('condition_sha256') == plan.validation.condition_sha256
                    and contract.get('source_manifest_sha256') == plan.source_manifest_sha256, 'library_condition_changed')
                _continuation_library(reader, player, rows, continuation, report)
                candidates = {e['event_id']: e['payload']['bundle_hash'] for e in journal if e['kind'] == 'candidate_committed'}
                starts = [r for r in rows if r['kind'] == 'trial_start' and r['value'].get('learner') == player]
                _require(len({r['value']['attempt_id'] for r in starts}) == len(starts), 'duplicate_trial_attempt')
                for start in starts:
                    attempt = _token(start['value']['attempt_id'])
                    phase = start['value']['phase']
                    trial = {'attempt_id': attempt, 'phase': phase, 'native_rechecked': False}
                    detail['trials'].append(trial)
                    try:
                        end = _one(rows, 'trial_end', learner=player, attempt=attempt)
                        _require(end['sequence'] > start['sequence'], 'trial_end_precedes_start')
                        folder = reader.root/player/attempt
                        _run_condition(reader, player+'/'+attempt, plan)
                        path = player+'/'+attempt+'/host/trial_record.json'
                        stamp = _one(rows, 'practice_feedback' if phase == 'practice' else 'validation_receipt',
                                     learner=player, attempt=attempt)
                        if not (reader.root/path).exists():
                            if phase == 'practice' and stamp['value'].get('publications') == []:
                                trial['scope'] = 'No published candidate; no skill/native qualification receipt claimed.'
                                continue
                            raise ReceiptError('native_trial_record_missing')
                        archived = TrialRecord.model_validate(reader.json(path))
                        _require(archived.receipt_sequence == end['sequence']+1 == stamp['sequence']
                            and archived.attempt_id == attempt and archived.phase == phase and archived.attempt_index == 0
                            and archived.episode_id == end['value']['episode_id'], 'trial_receipt_order_mismatch')
                        if phase == 'practice':
                            pubs = stamp['value']['publications']
                            _require(pubs and archived.candidate_event_id == pubs[-1]['candidate_event_id'], 'practice_candidate_changed')
                        else:
                            _require(archived.candidate_event_id == start['value']['candidate_event_id'], 'validation_candidate_changed')
                        _require(candidates.get(archived.candidate_event_id) == archived.bundle_hash, 'trial_bundle_not_published')
                        expected = stamp['value'].get('public_feedback_sha256' if phase == 'practice' else 'receipt_sha256')
                        _require(expected == (archived.public_feedback_sha256 if phase == 'practice' else content_sha256(archived)),
                                 'trial_ledger_receipt_mismatch')
                        costs = Effort.model_validate(end['value']['cost'])
                        record = trial_collector(plan.validation, _slot(plan, start),
                            candidate_event_id=archived.candidate_event_id, bundle_hash=archived.bundle_hash,
                            agent_output=folder, episode_root=practice_root/_token(archived.episode_id),
                            attempt_id=attempt, attempt_index=0, receipt_sequence=stamp['sequence'], phase=phase,
                            wall_s=costs.wall_s, cpu_s=costs.cpu_s)
                        record = TrialRecord.model_validate(record)
                        trial.update(native_rechecked=record.native_result(), record=record.model_dump(mode='json'))
                        if content_sha256(record.cost) != content_sha256(costs):
                            report['cost_accounting_complete'] = False
                            raise ReceiptError('trial_record_cost_differs_from_ledger')
                        if content_sha256(record) != content_sha256(archived):
                            old, new = archived.model_dump(mode='json'), record.model_dump(mode='json')
                            fields = ('outcome', 'result_valid', 'physics_valid', 'evidence_source',
                                      'native_substeps', 'verification_issues')
                            trial['corrected_recheck'] = {
                                'archived_file_sha256': hashes['study/'+path],
                                'archived_record_sha256': content_sha256(archived),
                                'rechecked_record_sha256': content_sha256(record),
                                'archived_classification': {key: old[key] for key in fields},
                                'rechecked_classification': {key: new[key] for key in fields},
                                'changed_fields': sorted(key for key in old if content_sha256(old[key]) != content_sha256(new[key])),
                                'accepted_for_original_qualification': False,
                                'scope': 'Posthoc recheck differs from the preserved archive; original feedback, ledger and freeze are unchanged.'}
                            raise ReceiptError('rechecked_trial_differs_from_archive')
                        if phase == 'practice':
                            previous_records = [r for r in records if r.phase == 'practice']
                            previous_ids = sorted({r.candidate_event_id for r in previous_records if r.native_result()
                                                  and r.candidate_event_id != record.candidate_event_id})
                            next_sequence = stamp['sequence']+1
                            for previous_id in previous_ids:
                                archived_revision = reader.json(player+'/'+attempt+'/host/revision_'+_token(previous_id)+'.json')
                                diagnostic = {}
                                revision = revision_collector(library_root=library.root, previous_candidate_event_id=previous_id,
                                    candidate_event_id=record.candidate_event_id, agent_output=folder,
                                    practice_records=previous_records, receipt_sequence=next_sequence, diagnostics=diagnostic)
                                actual = revision.model_dump(mode='json') if revision else None
                                detail['revisions'].append({'attempt_id': attempt, 'previous_candidate_event_id': previous_id,
                                    'receipt': actual, 'diagnostics': diagnostic})
                                _require(actual == archived_revision.get('receipt'), 'rechecked_revision_differs_from_archive')
                                if revision is not None:
                                    _require(next_sequence < len(rows), 'revision_missing_study_event')
                                    event = rows[next_sequence]
                                    _require(event['kind'] == 'model_revision_verified' and event['value'].get('learner') == player
                                        and event['value'].get('candidate_event_id') == revision.candidate_event_id
                                        and event['value'].get('receipt_sha256') == content_sha256(revision), 'revision_study_event_mismatch')
                                    revisions.append(revision)
                                    next_sequence += 1
                        records.append(record)
                    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
                        trial['issue'] = _reason(error)
                        detail['issues'].append(attempt+':'+_reason(error))
                freeze, assessment, validation = _freeze(reader, player, plan, rows, records, diagnostics=detail,
                    prior_effort=continuation.prior_efforts(player) if continuation else ())
                freezes[player], assessments[player], validations[player] = freeze, assessment, validation
                detail.update(freeze=freeze.model_dump(mode='json'), development_ledger_verified=True,
                              validation_assessment=assessment.model_dump(mode='json') if assessment else None)
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
                detail['issues'].append(_reason(error))
        _require(set(freezes) == set(PLAYERS), 'both_verified_freezes_required')
        match_end = _one(rows, 'frozen_match_end')
        match_start = _one(rows, 'frozen_match_start')
        _require(match_end['sequence'] > match_start['sequence'] > max(f.freeze_sequence for f in freezes.values())
            and match_start['value']['bundles'] == {p: f.bundle_hash for p, f in freezes.items()}, 'formal_match_order_mismatch')
        formal_root = checked_directory(formal_episodes)
        _run_condition(reader, 'match', plan, match=True)
        if match_collector is None:
            from .match_receipts import collect_frozen_match_receipts
            match_collector = collect_frozen_match_receipts
        diagnostic = {}
        matches = match_collector(freezes, library_roots={p: reader.root/'libraries'/p for p in PLAYERS},
            agent_output=reader.root/'match', episode_root=formal_root/_token(match_end['value']['episode_id']),
            receipt_sequence=match_end['sequence'], diagnostics=diagnostic)
        report['match_diagnostics'] = diagnostic
        _require(matches is not None and set(matches) == set(PLAYERS), 'native_frozen_match_not_verified')
        actual_match_cost = diagnostic.get('match_effort')
        expected_match_cost = Effort.model_validate(archived_summary['match_cost']).model_dump(mode='json')
        fields = set(expected_match_cost)-{'wall_s', 'cpu_s'}
        if (not isinstance(actual_match_cost, dict) or not fields <= set(actual_match_cost)
                or content_sha256({k: actual_match_cost[k] for k in fields}) != content_sha256(
                    {k: expected_match_cost[k] for k in fields})):
            report['cost_accounting_complete'] = False
            raise ReceiptError('formal_native_audit_effort_differs_from_summary')
        for player in PLAYERS:
            detail = report['players'][player]
            detail['match_receipt'] = matches[player].model_dump(mode='json')
            candidates = [r for r in all_revisions[player] if r.candidate_event_id == freezes[player].candidate_event_id
                          and r.bundle_hash == freezes[player].bundle_hash]
            if not candidates or assessments[player] is None:
                detail['issues'].append('no_rechecked_revision_and_validation_for_frozen_candidate')
                continue
            outcomes = [audit_closed_loop(plan=plan.validation,
                practice=[r for r in all_records[player] if r.phase == 'practice'], revision=revision,
                validation=validations[player], assessment=assessments[player], freeze=freezes[player], match=matches[player])
                for revision in candidates]
            outcome = next((r for r in outcomes if r.closed_loop_executed), outcomes[0])
            detail['closed_loop'] = outcome.model_dump(mode='json')
        completion = _completion(job)
        if completion is not None:
            report['job_completion'] = completion
        job_ok = (job is None or completion is not None) and (completion is None or (
            completion['state'] == 'command_exited' and completion['exit_code'] == 0))
        if not job_ok:
            report['issues'].append('study_job_exit_not_confirmed_successful')
        report['closed_loop_verified'] = bool(not injected and auditor_identity and auditor_identity.get('verified') is True
            and report['cost_accounting_complete'] and archived_summary.get('status') == 'stages_executed' and job_ok
            and all(not d['issues'] and d['closed_loop'] and d['closed_loop']['closed_loop_executed']
                    for d in report['players'].values()))
        report['formal_match_completed'] = report['closed_loop_verified'] and all(
            d['closed_loop']['formal_match_completed'] for d in report['players'].values())
        report['status'] = 'verified' if report['closed_loop_verified'] else 'incomplete'
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as error:
        report['issues'].append(_reason(error))
    if getattr(plan, 'continuation_sha256', None) is not None:
        if report.get('continuation', {}).get('verified') is not True:
            report['cost_accounting_complete'] = False
            report['closed_loop_verified'] = False
            report['formal_match_completed'] = False
    report['artifact_sha256'] = hashes
    if report.get('study_status') in ('stopped_error', 'no_published_candidate', 'match_unresolved'):
        report['status'] = 'study_stopped'
    completion = report.get('job_completion')
    if completion and (completion['state'] != 'command_exited' or completion['exit_code'] != 0):
        report['status'] = 'study_failed_exit'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'practice-episodes', 'formal-episodes', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--study-job', type=Path)
    parser.add_argument('--wait-for-study-report', action='store_true')
    args = parser.parse_args()
    if args.wait_for_study_report and args.study_job is None:
        parser.error('--wait-for-study-report requires --study-job')
    output = Path(args.output)
    absolute_output = output.absolute()
    if any(absolute_output.is_relative_to(Path(root).absolute()) for root in (
            args.study, args.practice_episodes, args.formal_episodes, Path(__file__).absolute().parents[3])):
        parser.error('--output must be outside the study, episode roots and auditor source')
    if output.exists():
        parser.error('--output must be a new audit directory')
    output = checked_directory(output, create=True)
    identity = None
    try:
        identity = auditor_source_identity()
        wait = await_study_report(args.study, args.study_job) if args.wait_for_study_report else None
        result = audit_study(args.study, practice_episodes=args.practice_episodes,
            formal_episodes=args.formal_episodes, auditor_identity=identity, study_job=args.study_job)
        if wait is not None:
            result['wait_result'] = wait
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError, KeyboardInterrupt) as error:
        result = {'schema': 'card_rsi_study_audit.v1', 'status': 'incomplete',
            'issues': [_reason(error)], 'auditor_source': identity, 'closed_loop_verified': False,
            'formal_match_completed': False, 'cost_accounting_complete': False}
    atomic_write_json(output/'audit_report.json', result, replace=False)
    print('Audit recorded: '+result['status'])


if __name__ == '__main__':
    main()
