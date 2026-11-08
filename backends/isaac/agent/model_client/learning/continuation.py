"""Explicit practice-only continuation across independent study/source identities.

Historical trials supply budget credit and public experience, never qualification
under a new condition. Native streams are not replayed here: this loader binds the
already sealed independent audit, its receipts, source and closed processes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import stat

from pydantic import BaseModel, ConfigDict

from .assessment import Effort, TrialRecord, content_sha256, sum_effort
from .curriculum import StudyPlan, study_condition_sha256
from .journal import (atomic_write_bytes, atomic_write_json, checked_directory,
                      read_bytes, read_json)
from .receipts import _Reader, _source_record, collect_effort
from .skills import SkillLibrary, _file_map, _inventory, _revision

PLAYERS = ('player_0', 'player_1')
ROUTING_KEYS = ('provider', 'model', 'base_url', 'endpoint', 'protocol', 'reasoning_effort')


class ContinuationError(ValueError):
    """A parent run or migration is not a complete, compatible checkpoint."""


def _require(ok, reason):
    if not ok:
        raise ContinuationError(reason)


class _Request(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    schema: str
    study: str
    study_job: str
    audit_report: str
    practice_episodes: str


def _absolute(value, *, directory=False):
    path = Path(value)
    _require(path.is_absolute() and str(path) == value and '..' not in path.parts,
             'absolute_canonical_path_required')
    checked_directory(path if directory else path.parent)
    if not directory:
        read_bytes(path)  # Reject links, hard links, devices and oversized files.
    return path


def _ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(read_bytes(path)).hexdigest()}


def _tree_inventory(root):
    """Small library state only; no episode/media/native-stream tree traversal."""
    checked_directory(root)
    result = {}
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        _require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'library_link_or_special_file')
        if stat.S_ISREG(mode):
            result[path.relative_to(root).as_posix()] = hashlib.sha256(read_bytes(path)).hexdigest()
    return result


def _closed_job(path, *, expected_exit=None):
    from scripts.maintenance.background_job import status
    value = status(path)
    _require(value.get('state') == 'command_exited' and type(value.get('exit_code')) is int
             and not value.get('live_session_members'), 'parent_process_not_closed')
    if expected_exit is not None:
        _require(value['exit_code'] == expected_exit, 'parent_process_exit_mismatch')
    return read_json(path/'completion.json')


def _parent_sources(study, plan, audit):
    reader = _Reader(study, 'study', {})
    source = _source_record(reader, 'source_verification.json', plan.source_manifest_sha256,
                            ['agent.model_client.function_runner'])
    runtime = audit.get('runtime_source', {})
    _require(runtime.get('verified') is True and runtime.get('manifest_sha256') == plan.source_manifest_sha256
             and runtime.get('snapshot_root') == source['snapshot_root'], 'parent_audit_runtime_source_mismatch')
    auditor = audit.get('auditor_source', {})
    _require(auditor.get('verified') is True, 'parent_auditor_not_verified')
    snapshot = _absolute(auditor['snapshot_root'], directory=True)
    from scripts.maintenance.freeze_project_sources import verify
    verify(snapshot)
    _require(_ref(snapshot/'source_snapshot_manifest.json')['sha256'] == auditor['manifest_sha256']
             and _ref(snapshot/'agent/model_client/learning/audit_study.py')['sha256'] == auditor['module_sha256'],
             'parent_auditor_source_changed')
    # Bind every artifact directly recorded by the parent auditor. Its nested
    # native receipts remain historical evidence; no new native claim is made.
    roots = {'study': study, 'source': Path(source['snapshot_root']),
             **{'library_'+p: study/'libraries'/p for p in PLAYERS}}
    pins = audit.get('artifact_sha256')
    _require(isinstance(pins, dict) and pins, 'parent_audit_artifact_pins_missing')
    for key, digest in pins.items():
        prefix, relative = key.split('/', 1)
        _require(prefix in roots and not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                 'parent_audit_artifact_path_invalid')
        _require(_ref(roots[prefix]/relative)['sha256'] == digest, 'parent_audit_artifact_changed')
    for relative in ('study_plan.json', 'study_report.json', 'source_verification.json'):
        _require('study/'+relative in pins, 'parent_audit_required_artifact_missing')
    return source


def _compatible(old, new):
    _require(all(old.model.get(k) == new.model.get(k) for k in ROUTING_KEYS), 'model_routing_changed')
    _require(old.budget == new.budget, 'practice_budget_changed')
    for key in ('parent_task_id', 'gym_id', 'gym_release_sha256', 'parent_release_sha256',
                'practice_budget_fallback', 'operator_manipulation_recipe', 'practice_slots'):
        _require(getattr(old, key) == getattr(new, key), 'parent_task_or_cases_changed')
    for key in ('gym_id', 'parent_task_id', 'parent_release_sha256', 'public_rules_sha256',
                'scope', 'slots', 'required_coverage', 'practice_case_fingerprints', 'budget_fallback',
                'minimum_success_rate', 'minimum_successes_per_coverage',
                'maximum_rule_violations', 'selection_policy'):
        _require(getattr(old.validation, key) == getattr(new.validation, key), 'validation_cases_changed')


@dataclass(frozen=True)
class Continuation:
    manifest: dict
    _completed: dict = field(repr=False)
    _feedback: dict = field(repr=False)
    _efforts: dict = field(repr=False)
    _libraries: dict = field(repr=False)

    @property
    def checkpoint_sha256(self):
        return content_sha256(self.manifest)

    def completed_slots(self, player):
        _require(player in PLAYERS, 'unknown_learner')
        return tuple(self._completed[player])

    def public_feedback(self, player):
        _require(player in PLAYERS, 'unknown_learner')
        return deepcopy(self._feedback[player])

    def prior_efforts(self, player):
        _require(player in PLAYERS, 'unknown_learner')
        return [Effort.model_validate(deepcopy(value)) for value in self._efforts[player]]


def load_continuation(request_path, new_base_plan):
    """Read a sealed 429 parent; no files, clients, processes or physics mutate.

    A condition-bound plan is accepted too. New condition/continuation hashes
    are deliberately absent from this manifest to avoid a digest cycle.
    """
    request_path = _absolute(str(request_path))
    request = _Request.model_validate(read_json(request_path))
    _require(request.schema == 'card_rsi_continuation_request.v1', 'unknown_continuation_request')
    study = _absolute(request.study, directory=True)
    job = _absolute(request.study_job, directory=True)
    audit_path = _absolute(request.audit_report)
    episodes = _absolute(request.practice_episodes, directory=True)
    old = StudyPlan.model_validate(read_json(study/'study_plan.json'))
    new = StudyPlan.model_validate(new_base_plan)
    _require(old.validation.condition_sha256 == study_condition_sha256(old), 'parent_condition_digest_changed')
    _compatible(old, new)
    _require(not getattr(old, 'continuation_sha256', None), 'nested_continuation_requires_explicit_lineage_support')
    summary = read_json(study/'study_report.json')
    audit = read_json(audit_path)
    completion = _closed_job(job)
    _require(completion['exit_code'] != 0 and summary.get('status') == 'stopped_error'
             and summary.get('cost_accounting_complete') is True and summary.get('unaccounted_attempts') == [],
             'parent_not_sealed_interruption')
    _require(audit.get('schema') == 'card_rsi_study_audit.v1'
             and audit.get('status') == 'study_failed_exit' and audit.get('injected_test_collectors') is False
             and audit.get('cost_accounting_complete') is True and audit.get('unaccounted_attempts') == []
             and audit.get('effort_audit', {}).get('complete') is True, 'parent_audit_incomplete')
    _require(audit.get('job_completion') == {k: completion.get(k) for k in
             ('state', 'exit_code', 'completed_at_utc', 'elapsed_wall_s')}, 'parent_audit_job_mismatch')
    source = _parent_sources(study, old, audit)
    closure_path = study.parent/'closure01.json'
    closure = read_json(closure_path)
    _require(closure.get('complete') is True, 'parent_native_cleanup_incomplete')
    closed_ids = set()
    for kind in ('gym', 'formal'):
        row = closure['servers'][kind]
        _require(row.get('complete') is True, 'parent_server_cleanup_incomplete')
        _closed_job(study.parent/(kind+'_server_job'), expected_exit=0)
        shutdown_path = _absolute(row['shutdown']['path'])
        _require(shutdown_path.is_relative_to(study.parent)
                 and _ref(shutdown_path)['sha256'] == row['shutdown']['sha256'], 'parent_shutdown_changed')
        sealed = read_json(shutdown_path)['episodes']
        _require(sealed == row['episodes'] and all(v.get('finished') is True and v.get('worker_exit_code') == 0
                                                 for v in sealed.values()), 'parent_worker_not_closed')
        if kind == 'formal':
            _require(not sealed, 'formal_experience_cannot_resume_practice')
        else:
            closed_ids = set(sealed)
    ledger_paths = sorted((study/'ledger').iterdir())
    _require([p.name for p in ledger_paths] == [f'{i:06d}.json' for i in range(len(ledger_paths))], 'parent_ledger_gap')
    ledger = [read_json(p) for p in ledger_paths]
    allowed = {'trial_start', 'trial_end', 'practice_feedback', 'model_revision_verified'}
    _require(all(row.get('sequence') == i and row.get('kind') in allowed for i, row in enumerate(ledger)),
             'only_unfrozen_practice_can_continue')
    starts = [r for r in ledger if r['kind'] == 'trial_start']
    ends = [r for r in ledger if r['kind'] == 'trial_end']
    _require(starts and len(starts) == len(ends), 'parent_unclosed_attempt')
    completed, feedback, efforts = ({p: [] for p in PLAYERS} for _ in range(3))
    libraries = {}; attempts = []; episode_ids = set(); terminal429 = None
    all_costs = []
    seen_players = []
    for number, start in enumerate(starts):
        value = start['value']; player = value['learner']; attempt = value['attempt_id']
        _require(player in PLAYERS and value['phase'] == 'practice', 'nonpractice_attempt')
        seen_players.append(player)
        index = sum(r['value']['learner'] == player for r in starts[:number])
        _require(index < len(old.practice_slots), 'parent_practice_budget_exceeded')
        slot = old.practice_slots[index]
        _require(attempt == slot.slot_id and value['selection'] == slot.selection, 'parent_practice_prefix_changed')
        matches = [r for r in ends if r['value']['learner'] == player and r['value']['attempt_id'] == attempt]
        _require(len(matches) == 1 and matches[0]['sequence'] > start['sequence'], 'parent_attempt_end_mismatch')
        end = matches[0]; folder = study/player/attempt; report = read_json(folder/'run_report.json')
        result = report.get('result') or {}; eid = report.get('episode_id')
        _require(isinstance(eid, str) and eid in closed_ids and eid not in episode_ids
                 and end['value']['episode_id'] == eid, 'parent_episode_identity_mismatch')
        episode_ids.add(eid)
        _require(not report.get('finalization_error') and not report.get('action_recovery_pending')
                 and result.get('physics_valid') is True and result == read_json(episodes/eid/'agent/result.json'),
                 'parent_terminal_result_not_closed')
        physical = read_json(episodes/eid/'private/physics_audit.json')
        _require(physical.get('physics_valid') is True, 'parent_physics_not_valid')
        expected = Effort.model_validate(end['value']['cost'])
        actual = collect_effort(folder, wall_s=expected.wall_s, cpu_s=expected.cpu_s)
        _require(actual == expected, 'parent_effort_changed')
        audit_efforts = [r for r in audit['effort_audit']['trials']
                         if r['learner'] == player and r['attempt_id'] == attempt]
        _require(len(audit_efforts) == 1 and audit_efforts[0].get('verified') is True
                 and audit_efforts[0].get('native_cost_verified') is True
                 and audit_efforts[0]['actual'] == expected.model_dump(mode='json'), 'parent_effort_audit_mismatch')
        efforts[player].append(actual.model_dump(mode='json')); all_costs.append(actual)
        feedback_rows = [r for r in ledger if r['kind'] == 'practice_feedback'
                         and r['value']['learner'] == player and r['value']['attempt_id'] == attempt]
        error = report.get('run_error')
        if error:
            _require(number == len(starts)-1 and not feedback_rows
                     and error == {'error_type': 'ModelGatewayError', 'error_category': 'http_status', 'http_status': 429}
                     and report.get('run_status') == 'infra_error' and report.get('task_outcome') == 'unknown'
                     and result.get('outcome') == 'cancelled', 'only_final_sealed_429_may_resume')
            terminal429 = {'learner': player, 'slot_id': attempt, 'episode_id': eid,
                           'run_report': _ref(folder/'run_report.json'), 'cost': actual.model_dump(mode='json')}
        else:
            _require(report.get('run_status') in ('completed', 'agent_budget')
                     and report.get('task_outcome') in ('success', 'failure') and len(feedback_rows) == 1,
                     'parent_completed_slot_not_valid')
            stamp = feedback_rows[0]; payload = read_json(folder/'host/public_learning_feedback.json')
            _require(stamp['sequence'] > end['sequence']
                     and content_sha256(payload) == stamp['value']['public_feedback_sha256'], 'parent_public_feedback_changed')
            checked = [r for r in audit['players'][player]['trials'] if r['attempt_id'] == attempt]
            _require(len(checked) == 1 and not checked[0].get('issue'), 'parent_completed_trial_audit_missing')
            if (folder/'host/trial_record.json').exists():
                record = TrialRecord.model_validate(read_json(folder/'host/trial_record.json'))
                _require(record.native_result() and not record.verification_issues
                         and checked[0].get('native_rechecked') is True
                         and checked[0].get('record') == record.model_dump(mode='json'), 'parent_native_receipt_not_verified')
            else:
                _require(stamp['value']['publications'] == [], 'parent_native_receipt_missing')
            prior_feedback = summary['learners'][player]['public_feedback']
            _require(len(prior_feedback) > len(completed[player])
                     and prior_feedback[len(completed[player])]['assessment'] == payload, 'parent_feedback_summary_mismatch')
            feedback[player].append(deepcopy(prior_feedback[len(completed[player])]))
            completed[player].append(attempt)
        attempts.append({'learner': player, 'slot_id': attempt, 'episode_id': eid,
                         'run_report': _ref(folder/'run_report.json'), 'cost': actual.model_dump(mode='json'),
                         'completed_slot_credit': not bool(error)})
    _require(terminal429 is not None and episode_ids == closed_ids, 'parent_not_exactly_a_429_checkpoint')
    _require(seen_players == sorted(seen_players), 'parent_learner_order_changed')
    _require(audit.get('issues') == ['both_verified_freezes_required'], 'parent_audit_unexpected_issue')
    _require(len(audit['effort_audit']['trials']) == len(starts), 'parent_effort_audit_extra_trial')
    for player in PLAYERS:
        detail = audit.get('players', {}).get(player, {})
        player_attempts = [a['slot_id'] for a in attempts if a['learner'] == player]
        allowed_issues = ['FileNotFoundError']  # The unfrozen learner has no freeze manifest.
        if terminal429['learner'] == player:
            allowed_issues.insert(0, terminal429['slot_id']+':missing_or_duplicate_practice_feedback')
        _require(detail.get('issues') == allowed_issues
                 and [r['attempt_id'] for r in detail.get('trials', [])] == player_attempts,
                 'parent_audit_trial_or_issue_mismatch')
    _require(content_sha256(sum_effort(all_costs)) == content_sha256(summary['cost']), 'parent_total_effort_mismatch')
    compatibility = {'task_ids': [new.gym_id, new.parent_task_id],
                     'task_releases': {new.gym_id: new.gym_release_sha256,
                                       new.parent_task_id: new.parent_release_sha256}}
    library_records = {}
    for player in PLAYERS:
        path = study/'libraries'/player
        _require(not (study/player/'freeze_manifest.json').exists(), 'frozen_practice_cannot_continue')
        if not path.exists():
            _require(not efforts[player], 'parent_learner_library_missing')
            library_records[player] = None
            continue
        library = SkillLibrary(path)
        contract = library.contract
        _require(contract.get('source_manifest_sha256') == old.source_manifest_sha256
                 and contract.get('condition_sha256') == old.validation.condition_sha256
                 and contract.get('model') == old.model
                 and contract.get('compatibility') == compatibility, 'parent_library_contract_changed')
        events = library.journal.read()
        _require(not any(e['kind'] in ('validation_recorded', 'candidate_tested', 'candidate_selected', 'budget_frozen')
                         for e in events), 'validation_or_selection_cannot_resume_practice')
        prepared = {e['transaction_id'] for e in events if e['kind'] == 'publication_prepared'}
        candidates = [e for e in events if e['kind'] == 'candidate_committed']
        _require(prepared == {e['transaction_id'] for e in candidates}, 'parent_publication_unresolved')
        for candidate in candidates:
            library._candidate(candidate['event_id']); library.verify(candidate['payload']['bundle_hash'])
        author = _file_map(library.author)
        inventory = _tree_inventory(path)
        record = {'path': str(path), 'files_sha256': inventory,
                  'experiment_sha256': content_sha256(library.experiment),
                  'journal_sha256': content_sha256(events), 'journal_last_hash': events[-1]['event_hash'],
                  'author_inventory': _inventory(author), 'author_revision': _revision(author),
                  'candidates': [{'event_id': e['event_id'], 'transaction_id': e['transaction_id'],
                                  'bundle_hash': e['payload']['bundle_hash']} for e in candidates]}
        libraries[player] = record; library_records[player] = record
    manifest = {'schema': 'card_rsi_continuation_checkpoint.v1', 'request': request.model_dump(mode='json'),
        'parent_study_plan': _ref(study/'study_plan.json'), 'parent_study_report': _ref(study/'study_report.json'),
        'parent_audit_report': _ref(audit_path), 'parent_completion': _ref(job/'completion.json'),
        'parent_closure': _ref(closure_path), 'parent_ledger_sha256': content_sha256(ledger),
        'parent_source_manifest_sha256': old.source_manifest_sha256, 'parent_source_root': source['snapshot_root'],
        'new_source_manifest_sha256': new.source_manifest_sha256, 'new_model': deepcopy(new.model),
        'new_base_condition_sha256': study_condition_sha256(new.model_copy(update={'continuation_sha256': None})),
        'compatibility': compatibility,
        'budget': new.budget.model_dump(mode='json'), 'gym_release_sha256': new.gym_release_sha256,
        'parent_release_sha256': new.parent_release_sha256, 'public_rules_sha256': new.validation.public_rules_sha256,
        'practice_cases_sha256': content_sha256([s.model_dump(mode='json') for s in new.practice_slots]),
        'validation_cases_sha256': content_sha256([s.model_dump(mode='json') for s in new.validation.slots]),
        'completed_slots': completed, 'public_feedback': feedback, 'prior_efforts': efforts,
        'prior_attempts': attempts, 'interrupted_attempt': terminal429, 'libraries': library_records,
        'historical_receipts_qualify_new_condition': False, 'restores_native_state': False}
    return Continuation(manifest, completed, feedback, efforts, libraries)


def _new_library_contract(continuation, player, library):
    manifest = continuation.manifest
    condition = content_sha256({'base_condition_sha256': manifest['new_base_condition_sha256'],
                                'continuation_sha256': continuation.checkpoint_sha256})
    _require(library.experiment['experiment_id'] == player
             and library.contract.get('source_manifest_sha256') == manifest['new_source_manifest_sha256']
             and library.contract.get('model') == manifest['new_model']
             and library.contract.get('condition_sha256') == condition
             and library.contract.get('compatibility') == manifest['compatibility'],
             'new_library_condition_mismatch')


def seed_library(continuation, player, new_library):
    """Migrate own model content with explicit host provenance, never old credit."""
    _require(player in PLAYERS, 'unknown_learner')
    library = new_library if isinstance(new_library, SkillLibrary) else SkillLibrary(new_library)
    manifest = continuation.manifest
    _new_library_contract(continuation, player, library)
    events = library.journal.read()
    _require(len(events) == 1 and events[0]['kind'] == 'experiment_created'
             and read_json(library.author/'catalog.json') == {'skills': []}
             and len(_file_map(library.author)) == 1 and not list(library.bundles.iterdir()), 'new_library_not_empty')
    old = continuation._libraries.get(player); imports = []
    if old is not None:
        source = SkillLibrary(Path(old['path']))
        _require(_tree_inventory(source.path) == old['files_sha256'], 'parent_library_changed_since_checkpoint')
        _require(source.path != library.path and not library.path.is_relative_to(source.path), 'migration_would_modify_parent')
        for candidate in old['candidates']:
            imported = library.import_candidate(source, candidate['event_id'], transaction_id=candidate['transaction_id'],
                provenance={'scenes': [library.contract['compatibility']['task_ids'][0],
                                       library.contract['compatibility']['task_ids'][1]], 'phase': 'practice',
                    'cost': {'checkpoint_sha256': continuation.checkpoint_sha256,
                             'scope': 'Parent attempts counted once, not once per imported bundle'},
                    'licenses': [], 'authorship': 'Inherited model-authored content; no new license declaration',
                    'source_study': manifest['request']['study'], 'checkpoint_sha256': continuation.checkpoint_sha256})
            _require(imported['event_id'] == candidate['event_id'], 'candidate_identifier_not_preserved')
            imports.append({'candidate_event_id': imported['event_id'], 'bundle_hash': imported['payload']['bundle_hash'],
                            'origin_sha256': imported['payload']['import']['origin_sha256']})
        files = _file_map(source.author)
    else:
        files = _file_map(library.author)
    archive = checked_directory(library.host/'continuation_author', create=True)
    for name, data in sorted(files.items()):
        for root in (archive, library.author):
            path = root/name; checked_directory(path.parent, create=True)
            atomic_write_bytes(path, data, replace=root == library.author, mode=0o400 if root == archive else 0o600)
    payload = {'checkpoint_sha256': continuation.checkpoint_sha256, 'learner': player,
        'parent_library': deepcopy(old), 'imported_candidates': imports,
        'initial_author_inventory': _inventory(files), 'initial_author_revision': _revision(files),
        'new_experiment_sha256': content_sha256(library.experiment),
        'authorship': 'host migration of inherited model-authored content; not a new model revision',
        'inherited_qualification': False}
    library.journal.append('continuation_seeded', payload, event_id='continuation-seeded')
    prefix = library.journal.read()
    receipt = {'schema': 'card_rsi_continuation_library_seed.v1', **payload,
        'seed_journal_length': len(prefix), 'seed_journal_last_hash': prefix[-1]['event_hash'],
        'seed_journal_sha256': content_sha256(prefix)}
    if old is not None:
        _require(_tree_inventory(Path(old['path'])) == old['files_sha256'], 'parent_library_changed_during_migration')
    atomic_write_json(library.host/'continuation.json', receipt, replace=False)
    return verify_seed_library(continuation, player, library)


def verify_seed_library(continuation, player, new_library):
    """Check sealed migration history without comparing subsequently edited drafts."""
    _require(player in PLAYERS, 'unknown_learner')
    library = new_library if isinstance(new_library, SkillLibrary) else SkillLibrary(new_library)
    _new_library_contract(continuation, player, library)
    receipt = read_json(library.host/'continuation.json')
    _require(receipt.get('schema') == 'card_rsi_continuation_library_seed.v1'
             and receipt.get('checkpoint_sha256') == continuation.checkpoint_sha256
             and receipt.get('learner') == player and receipt.get('inherited_qualification') is False
             and receipt.get('new_experiment_sha256') == content_sha256(library.experiment)
             and receipt.get('parent_library') == continuation._libraries.get(player), 'continuation_seed_binding_changed')
    events = library.journal.read(); n = receipt['seed_journal_length']
    _require(type(n) is int and 1 < n <= len(events), 'continuation_seed_prefix_missing')
    prefix = events[:n]
    _require(prefix[-1]['kind'] == 'continuation_seeded'
             and prefix[-1]['event_hash'] == receipt['seed_journal_last_hash']
             and content_sha256(prefix) == receipt['seed_journal_sha256'], 'continuation_seed_journal_changed')
    payload = {k: v for k, v in receipt.items() if k not in
               ('schema', 'seed_journal_length', 'seed_journal_last_hash', 'seed_journal_sha256')}
    _require(prefix[-1]['payload'] == payload, 'continuation_seed_receipt_changed')
    files = _file_map(library.host/'continuation_author')
    _require(_inventory(files) == receipt['initial_author_inventory']
             and _revision(files) == receipt['initial_author_revision'], 'continuation_initial_author_changed')
    old = continuation._libraries.get(player)
    if old is not None:
        _require(receipt['initial_author_inventory'] == old['author_inventory']
                 and receipt['initial_author_revision'] == old['author_revision'], 'continuation_author_origin_changed')
    candidates = [e for e in prefix if e['kind'] == 'candidate_committed']
    _require(len(candidates) == len(receipt['imported_candidates'])
             and len(candidates) == len(old['candidates'] if old else []), 'continuation_candidate_inventory_changed')
    for event, expected, original in zip(candidates, receipt['imported_candidates'], old['candidates'] if old else []):
        library._candidate(event['event_id']); library.verify(event['payload']['bundle_hash'])
        _require(event['event_id'] == expected['candidate_event_id']
                 and event['payload']['bundle_hash'] == expected['bundle_hash']
                 and event['payload']['import']['origin_sha256'] == expected['origin_sha256']
                 and event['event_id'] == original['event_id']
                 and event['transaction_id'] == original['transaction_id']
                 and event['payload']['bundle_hash'] == original['bundle_hash'], 'continuation_import_changed')
        origin = read_json(library.host/'origins'/(expected['origin_sha256']+'.json'))
        _require(content_sha256(origin['experiment']) == old['experiment_sha256']
                 and content_sha256(origin['journal']) == old['journal_sha256'], 'continuation_import_origin_changed')
    return receipt
