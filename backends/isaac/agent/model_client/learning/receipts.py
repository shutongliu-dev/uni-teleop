"""Read-only, trusted-host adaptation of real runner/native RSI artifacts.

No supplied outcome or model-written text is authoritative. Qualification needs
the actual episode result, source pins, closed native frame stream and original
host journals. These files are not signatures against a hostile host: this
adapter must run outside the model's writable namespace. It never sends private
records to a model, starts physics, or repairs incomplete evidence.
"""
from __future__ import annotations

import hashlib
import json
import ast
import base64
import math
import os
from pathlib import Path
import re
import stat
from collections import Counter
from types import SimpleNamespace

from bench.protocol.functions import ACTION_FUNCTIONS, FUNCTION_INPUTS

from .assessment import (Effort, EvaluationPlan, RevisionReceipt, TrialRecord,
                         ValidationSlot, content_sha256)
from .journal import canonical_json, checked_directory, read_bytes, strict_json_loads
from .robot_api import RobotAPI


class ReceiptError(ValueError):
    pass


READ_ONLY_FUNCTIONS = (set(FUNCTION_INPUTS)-set(ACTION_FUNCTIONS)-{'cancel_action', 'finish_episode'}) | {
    'read_decision', 'skill_read', 'skill_status', 'skill_trace', 'compute_rotation', 'compute_vector',
    'compute_pose_delta', 'compute_transform_points', 'compute_frame_alignment'}


def _require(condition, code):
    if not condition:
        raise ReceiptError(code)


def _require_task_outcome(result):
    # A closed, physically valid stream can still end in a controller/host
    # failure. Such an attempt consumed resources but did not establish a
    # model task outcome. Keep the original EpisodeResult untouched.
    outcome = result.get('outcome')
    if outcome in ('policy_error', 'infrastructure_error', 'evaluator_invalid'):
        raise ReceiptError('episode_execution_error_'+outcome)


def _count(value):
    return type(value) is int and value >= 0


def _scene_hash(value):
    # Frozen scene releases predate the learning journal and use escaped-ASCII
    # JSON. Their Chinese instructions must retain that exact hash convention.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class _Reader:
    def __init__(self, root, label, hashes):
        self.root = checked_directory(root)
        self.label, self.hashes = label, hashes

    def path(self, relative):
        path = Path(relative)
        _require(not path.is_absolute() and '..' not in path.parts and str(path) == relative,
                 'invalid_artifact_path')
        return self.root/path

    def raw(self, relative):
        value = read_bytes(self.path(relative))
        self.hashes[self.label+'/'+relative] = hashlib.sha256(value).hexdigest()
        return value

    def json(self, relative):
        return strict_json_loads(self.raw(relative))

    def journal(self, relative):
        """Reuse the Journal format without creating a writer lock or directory."""
        directory = checked_directory(self.path(relative))
        paths = []
        for path in directory.iterdir():
            if path.name == '.writer.lock' or path.name.startswith('.pending-'):
                continue
            _require(re.fullmatch(r'\d{20}\.json', path.name), 'unexpected_journal_entry')
            paths.append(path)
        previous, identities, events = None, set(), []
        for index, path in enumerate(sorted(paths), 1):
            event = self.json(relative+'/'+path.name)
            _require(isinstance(event, dict) and set(event) == {
                'sequence', 'event_id', 'kind', 'transaction_id', 'payload', 'previous_hash', 'event_hash'},
                'invalid_journal_schema')
            _require(type(event['sequence']) is int and event['sequence'] == index
                and path.name == f'{index:020d}.json' and event['previous_hash'] == previous,
                'journal_chain_discontinuity')
            _require(isinstance(event['event_id'], str) and event['event_id']
                and event['event_id'] not in identities and isinstance(event['kind'], str)
                and isinstance(event['payload'], dict), 'invalid_journal_event')
            body = {k: v for k, v in event.items() if k != 'event_hash'}
            expected = hashlib.sha256(b'agent-journal-v1\0'+canonical_json(body)).hexdigest()
            _require(event['event_hash'] == expected, 'journal_checksum_mismatch')
            previous = expected
            identities.add(event['event_id'])
            events.append(event)
        return events


def _actions(reader, episode_id, *, relative='single/host/action_journal', player='single'):
    events = reader.journal(relative)
    _require(events and events[0]['kind'] == 'action_ledger_bound', 'missing_action_binding')
    binding = events[0]['payload']
    _require(binding.get('protocol') == 'agent_action_recovery_v1'
        and binding.get('episode_id') == episode_id and binding.get('trial_id') == player
        and binding.get('contract', {}).get('interface') == 'public_functions_v1', 'action_binding_mismatch')
    owner = binding['contract'].get('player')
    _require(owner is None if player == 'single' else isinstance(owner, dict) and owner.get('player_id') == player,
             'action_player_binding_mismatch')
    # The existing reducer validates identities, payload hashes, terminal
    # observations and sequence/clock correspondence. Its constructor writes a
    # binding, so deliberately use only the pure reducer with a read-only view.
    reducer = object.__new__(RobotAPI)
    reducer.journal = SimpleNamespace(read=lambda: events)
    reducer.binding, reducer.episode_id = binding, episode_id
    reducer.max_status_queries = binding['recovery_policy']['max_status_queries']
    actions, pending = reducer._state()
    accepted = completed = 0
    for row in actions.values():
        receipt = row.get('receipt') or {}
        accepted += int(bool(receipt) and receipt.get('termination_reason') != 'begin_rejected')
        completed += int(row.get('settled', {}).get('receipt', {}).get('status') == 'completed')
    return {'events': events, 'actions': actions, 'pending': pending,
            'accepted': accepted, 'completed': completed, 'binding': binding}


def _model_effort(reader, report, *, player='single', include_pre=True):
    metadata, logical, actual, incomplete_usage = [], 0, 0, False
    folders = sorted((reader.root/player/'decisions').glob('*'))
    response_metadata = []
    for folder in folders:
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        relative = str(folder.relative_to(reader.root))
        response = folder/'model_response.json'
        error = folder/'model_error.json'
        if not response.exists() and not error.exists():
            # Preparation alone does not prove a send, but an interrupted
            # request cannot be silently certified as zero consumption either.
            _require(not (folder/'model_request.json').exists()
                and not (folder/'model_api_events.jsonl').exists(),
                'model_request_completion_unavailable')
            continue
        _require(not (response.exists() and error.exists()), 'ambiguous_model_completion')
        value = reader.json(relative+('/model_response.json' if response.exists() else '/model_error.json'))
        receipt = value['metadata'] if response.exists() else value
        attempts = receipt.get('attempts')
        _require(_count(attempts), 'unknown_model_attempt_count')
        logical += 1
        actual += attempts
        if response.exists():
            response_metadata.append(receipt)
            metadata.append(receipt.get('usage', {}))
            incomplete_usage |= attempts != 1
        else:
            incomplete_usage |= attempts > 0
        events_file = folder/'model_api_events.jsonl'
        if events_file.exists():
            events = [strict_json_loads(line) for line in reader.raw(relative+'/model_api_events.jsonl').splitlines()]
            sent = sum(e.get('event') == 'model_request_started' for e in events)
            sent -= sum(e.get('event') == 'model_request_not_sent' for e in events)
            _require(sent == attempts, 'model_event_attempt_mismatch')
    decisions = report.get('decisions', {}).get(player)
    if decisions is not None:
        _require(_count(decisions) and decisions == logical, 'model_decision_receipt_count_mismatch')
    # Configuration selection is a separate pre-episode request, and also
    # appears at the front of the report's single-player response list.
    pre = report.get('pre_episode_model_requests', [])
    _require(isinstance(pre, list), 'invalid_pre_episode_requests')
    for row in pre if include_pre else []:
        _require(_count(row.get('attempts')), 'unknown_model_attempt_count')
        logical += 1
        actual += row['attempts']
        metadata.append(row.get('usage', {}))
        incomplete_usage |= row['attempts'] != 1
    _require(report.get('model_requests', {}).get(player, []) == (pre if player == 'single' else [])+response_metadata,
             'model_report_receipts_mismatch')

    aliases = {'input_tokens': [('input_tokens',), ('prompt_tokens',)],
        'output_tokens': [('output_tokens',), ('completion_tokens',)],
        'reasoning_tokens': [('output_tokens_details', 'reasoning_tokens'), ('completion_tokens_details', 'reasoning_tokens')],
        'cached_input_tokens': [('input_tokens_details', 'cached_tokens'), ('prompt_tokens_details', 'cached_tokens')]}
    usage = {}
    for name, paths in aliases.items():
        numbers = []
        for row in metadata:
            found = []
            for path in paths:
                value = row
                for part in path:
                    value = value.get(part) if isinstance(value, dict) else None
                if value is not None:
                    _require(_count(value), 'invalid_provider_token_count')
                    found.append(value)
            _require(not found or len(set(found)) == 1, 'conflicting_provider_token_aliases')
            numbers.append(found[0] if found else None)
        usage[name] = (None if incomplete_usage or any(v is None for v in numbers)
                       else sum(numbers))
    return dict(model_requests=actual, model_logical_calls=logical, **usage)


def _closed_native_elapsed(report, *, match=False, creation_attempted=False):
    """A created episode needs its own closed result before cost is complete.

    A lost final response says nothing about native consumption. Preserve its
    receipts, but let the scheduler retain the attempt as unaccounted instead
    of substituting a zero clock. Before an episode exists, zero is legitimate.
    """
    episode_id = report.get('episode_id')
    result = report.get('result')
    if episode_id is None:
        _require(result is None and not match, 'native_result_without_episode')
        _require(not creation_attempted, 'native_effort_unavailable_without_episode_identity')
        return 0.
    _require(isinstance(result, dict) and not report.get('finalization_error'),
             'native_effort_unavailable_without_terminal_result')
    episode = result.get('episode') if match else result
    _require(isinstance(episode, dict) and isinstance(episode.get('spec'), dict)
        and episode['spec'].get('episode_id') == episode_id,
        'match_result_episode_mismatch' if match else 'trial_result_episode_mismatch')
    elapsed = episode.get('elapsed_sim_s')
    _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
             'invalid_elapsed_sim_time')
    return float(elapsed)


def _effort_details(agent_output, *, wall_s, cpu_s, hashes):
    reader = _Reader(agent_output, 'agent', hashes)
    report = reader.json('run_report.json')
    model = _model_effort(reader, report)
    calls = []
    directory = reader.root/'single/calls'
    if directory.exists():
        files = sorted(directory.glob('*.json'))
        _require([p.name for p in files] == [f'{n:06d}.json' for n in range(1, len(files)+1)],
                 'function_call_sequence_gap')
        calls = [reader.json('single/calls/'+p.name) for p in files]
    physical = sum(row.get('function') in ACTION_FUNCTIONS for row in calls)
    _require(report.get('function_calls', {}).get('single', 0) == len(calls), 'function_counter_mismatch')
    accepted = completed = 0
    episode_id = report.get('episode_id')
    action_info = None
    if episode_id:
        action_info = _actions(reader, episode_id)
        accepted, completed = action_info['accepted'], action_info['completed']
        _require(len(action_info['actions']) <= physical, 'more_actions_than_attempts')
        maximum = action_info['binding']['contract']['max_actions']
        _require(report.get('physical_attempts', {}).get('single', 0) == min(physical, maximum),
                 'physical_attempt_counter_mismatch')
    else:
        _require(physical == 0, 'physical_calls_without_episode')
    elapsed = _closed_native_elapsed(report, creation_attempted=(reader.root/'create_request.json').exists())
    return Effort(**model, physical_attempts=physical, accepted_actions=accepted, completed_actions=completed,
        read_only_calls=sum(row.get('function') in READ_ONLY_FUNCTIONS for row in calls),
        reset_attempts=int((reader.root/'create_request.json').exists()), native_sim_s=float(elapsed),
        wall_s=wall_s, cpu_s=cpu_s, provider_cost_usd=None), report, action_info


def collect_effort(agent_output: Path, *, wall_s=0., cpu_s=0.) -> Effort:
    """Count original actual receipts even if no skill was ever published.

    Missing/corrupt accounting raises: callers must preserve an unknown-cost
    infrastructure attempt, never catch this and replace its cost with zero.
    Unknown provider usage is null, including interrupted retry token costs.
    """
    return _effort_details(agent_output, wall_s=wall_s, cpu_s=cpu_s, hashes={})[0]


def collect_match_effort(agent_output: Path, *, wall_s=0., cpu_s=0.) -> Effort:
    """Sum both model/action streams, counting the shared episode clock once."""
    reader = _Reader(agent_output, 'agent', {})
    report = reader.json('run_report.json')
    players = ('player_0', 'player_1')
    _require(set(report.get('model_requests', {})) == set(players)
        and set(report.get('function_calls', {})) == set(players)
        and set(report.get('physical_attempts', {})) == set(players), 'incomplete_match_accounting')
    models, physical, accepted, completed, queries = [], 0, 0, 0, 0
    for index, player in enumerate(players):
        models.append(_model_effort(reader, report, player=player, include_pre=index == 0))
        files = sorted((reader.root/player/'calls').glob('*.json'))
        _require([p.name for p in files] == [f'{n:06d}.json' for n in range(1, len(files)+1)],
                 'function_call_sequence_gap')
        calls = [reader.json(str(p.relative_to(reader.root))) for p in files]
        _require(report['function_calls'][player] == len(calls), 'function_counter_mismatch')
        attempts = sum(row.get('function') in ACTION_FUNCTIONS for row in calls)
        actions = _actions(reader, report['episode_id'], relative=player+'/host/action_journal', player=player)
        _require(report['physical_attempts'][player] == min(attempts, actions['binding']['contract']['max_actions'])
            and len(actions['actions']) <= attempts, 'physical_attempt_counter_mismatch')
        physical += attempts
        accepted += actions['accepted']
        completed += actions['completed']
        queries += sum(row.get('function') in READ_ONLY_FUNCTIONS for row in calls)
    model = {}
    for key in models[0]:
        values = [row[key] for row in models]
        model[key] = None if any(v is None for v in values) else sum(values)
    elapsed = _closed_native_elapsed(report, match=True)
    return Effort(**model, physical_attempts=physical, accepted_actions=accepted, completed_actions=completed,
        read_only_calls=queries, reset_attempts=int((reader.root/'create_request.json').exists()),
        native_sim_s=float(elapsed), wall_s=wall_s, cpu_s=cpu_s, provider_cost_usd=None)


def _source_record(reader, relative, expected_manifest, required):
    record = reader.json(relative)
    _require(record.get('verified') is True and record.get('runtime_imports_verified') is True
        and record.get('file_modes_verified') is True, 'source_verification_missing')
    snapshot = checked_directory(record['snapshot_root'])
    manifest_bytes = read_bytes(snapshot/'source_snapshot_manifest.json')
    _require(hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest, 'source_manifest_mismatch')
    manifest = strict_json_loads(manifest_bytes)
    from scripts.maintenance.freeze_project_sources import verify
    verify(snapshot)
    imported = record['imported_project_modules']
    _require(set(required).issubset(imported), 'required_native_import_missing')
    for value in [*imported.values(), record.get('entrypoint')]:
        if value is None:
            continue
        path = Path(value['path'])
        _require(path.is_relative_to(snapshot), 'runtime_import_outside_snapshot')
        digest = hashlib.sha256(read_bytes(path)).hexdigest()
        _require(value['sha256'] == manifest['files_sha256'].get(str(path.relative_to(snapshot))) == digest,
                 'runtime_import_hash_mismatch')
    for paths in record.get('project_package_paths', {}).values():
        _require(all(Path(p).is_relative_to(snapshot) for p in paths), 'package_outside_snapshot')
    reader.hashes['source/source_snapshot_manifest.json'] = expected_manifest
    return record


def _skills(reader, binding, *, phase, episode_id, candidate_event_id, bundle_hash, player='single'):
    relative = binding['skill_runtime_journal']
    events = reader.journal(relative)
    _require(events and events[0]['kind'] == 'skill_runtime_bound', 'missing_skill_runtime_binding')
    bound = events[0]['payload']
    _require(bound.get('protocol') == 'agent_skill_runtime.v1'
        and bound.get('language') == 'agent_json_skill_ast.v1' and bound.get('episode_id') == episode_id
        and bound.get('trial_id') == player and bound.get('player_id') == player
        and bound.get('phase') == ('evaluation' if phase == 'heldout' else phase), 'skill_runtime_binding_mismatch')
    if phase != 'practice':
        _require(bound.get('frozen_candidate_event_id') == candidate_event_id
            and bound.get('frozen_bundle_hash') == bundle_hash, 'skill_frozen_pin_mismatch')
    root = str(Path(relative).parent/'skill_execution_records')
    actual_feedback = Counter(content_sha256(reader.json(str(p.relative_to(reader.root))))
                              for p in sorted((reader.root/player/'calls').glob('*.json')))
    executions, bundles, active, temporary, language_entries = {}, [], None, False, 0
    for event in events[1:]:
        kind, payload = event['kind'], event['payload']
        if kind == 'skill_execution_started':
            identifier = payload['execution_id']
            _require(active is None and identifier not in executions and re.fullmatch(r'[a-zA-Z0-9_-]+', identifier),
                     'overlapping_skill_execution')
            _require(reader.json(root+'/'+identifier+'/input.json') == payload, 'skill_input_mismatch')
            executions[identifier] = dict(binding=payload, calls=[], prepared=None, epoch=None, changed_epoch=False)
            active = identifier
            if payload['bundle_hash'] not in bundles:
                bundles.append(payload['bundle_hash'])
            if phase != 'practice':
                _require(payload['candidate_event_id'] == candidate_event_id and payload['bundle_hash'] == bundle_hash,
                         'executed_skill_pin_mismatch')
        elif kind == 'skill_api_prepared':
            _require(active == payload['execution_id'], 'skill_api_without_execution')
            current = executions[active]
            _require(current['prepared'] is None and payload['call_index'] == len(current['calls'])
                and not current['changed_epoch'], 'skill_api_sequence_or_epoch_mismatch')
            epoch = payload['authorization_epoch']
            _require(isinstance(epoch, dict) and set(epoch) == {'turn_index', 'current_player', 'terminal'}
                and epoch['terminal'] is False and (
                    epoch['turn_index'] is None and epoch['current_player'] is None if player == 'single'
                    else _count(epoch['turn_index']) and epoch['current_player'] == player),
                'invalid_practice_authorization_epoch')
            if current['epoch'] is None:
                current['epoch'] = epoch
            _require(epoch == current['epoch'], 'skill_authorization_epoch_changed')
            current['prepared'] = payload
        elif kind == 'skill_api_completed':
            _require(active == payload['execution_id'], 'skill_api_without_execution')
            current = executions[active]
            prepared = current['prepared']
            _require(prepared is not None and payload['call_index'] == prepared['call_index'], 'skill_api_completion_mismatch')
            record = reader.json(root+'/'+active+f"/call_{payload['call_index']:06d}.json")
            _require(content_sha256(record) == payload['record_sha256'] and
                all(record.get(k) == v for k, v in prepared.items()), 'skill_api_record_mismatch')
            summary = payload['summary']
            function = record['decision'].get('function', 'intent')
            primitive = record['feedback'].get('result') or {}
            primitive = primitive if isinstance(primitive, dict) else {}
            feedback_hash = content_sha256(record['feedback'])
            _require(actual_feedback[feedback_hash] > 0, 'skill_api_not_in_function_session')
            actual_feedback[feedback_hash] -= 1
            _require(summary.get('record_sha256') == payload['record_sha256']
                and summary.get('is_action') is (function in ACTION_FUNCTIONS)
                and summary.get('function') == function and summary.get('ok') == record['feedback'].get('ok')
                and summary.get('call_index') == payload['call_index'] and summary.get('status') == 'completed'
                and summary.get('primitive_status') == primitive.get('status')
                and summary.get('termination_reason') == primitive.get('termination_reason'),
                'skill_api_summary_mismatch')
            current['changed_epoch'] = record['after_epoch'] != current['epoch']
            current['calls'].append(summary)
            current['prepared'] = None
        elif kind == 'skill_execution_finished':
            _require(active == payload['execution_id'], 'skill_execution_completion_mismatch')
            current = executions[active]
            result = reader.json(root+'/'+active+'/result.json')
            _require(current['prepared'] is None and content_sha256(result) == payload['result_sha256']
                and result.get('calls') == current['calls'] and result.get('status') == payload['status']
                and all(result.get(k) == current['binding'][k] for k in
                        ('execution_id', 'skill_id', 'candidate_event_id', 'bundle_hash', 'source_sha256')),
                'skill_result_mismatch')
            _require(result.get('physical_action_pending') is False, 'skill_action_unresolved')
            _require(type(payload.get('language_entered')) is bool
                and result.get('language_entered') is payload['language_entered'], 'unknown_skill_language_entry')
            _require(payload['language_entered'] or not current['calls'], 'api_calls_before_language_entry')
            language_entries += int(payload['language_entered'])
            if current['changed_epoch']:
                _require(result.get('stop_reason') == 'authorization_epoch_changed', 'skill_failed_to_stop_at_epoch')
            active = None
        elif kind == 'skill_tool_finished':
            # A rejected write request is not a source mutation. Validation
            # runtime must refuse write/publish; conservatively retain any
            # successful one rather than hiding it as a no-op.
            temporary |= phase != 'practice' and payload.get('function') in ('skill_write', 'skill_publish') and payload.get('ok') is True
        elif kind != 'skill_tool_requested':
            raise ReceiptError('unknown_skill_journal_event')
    _require(active is None, 'skill_execution_unclosed')
    return len(executions), language_entries, bundles, temporary


def _native_frames(reader, inputs, observer, evidence, *, runtime='task_runtime', formal=False,
                   frame_audit=None, practice_stages=None, practice_thresholds=None):
    relative = 'private/'+runtime+'/native_frames.jsonl'
    path = reader.path(relative)
    checked_directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    digest, count, last_time = hashlib.sha256(), 0, evidence['native_start_time_s']
    last_step, dt = inputs['native_step'], inputs['physics_dt_s']
    actors = set(inputs['mass_properties_verified'])
    confirmations = []
    stage, terminal_success, stable_start, stable_positions = 0, False, None, None
    if not formal:
        _require(type(practice_stages) is int and practice_stages > 0
            and isinstance(practice_thresholds, dict), 'missing_practice_stage_contract')
        dwell, drift = (practice_thresholds.get(k) for k in ('dwell_s', 'maximum_drift_m'))
        _require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (dwell, drift)),
                 'invalid_practice_stability_contract')
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'invalid_native_stream_file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            for line in stream:
                _require(line.endswith(b'\n') and len(line) <= 8*1024*1024, 'native_stream_truncated_or_oversized')
                digest.update(line)
                row = strict_json_loads(line)
                frame = row['frame']
                _require(set(row) == {'frame', 'raw_contacts', 'result'} and isinstance(row['raw_contacts'], list), 'invalid_native_row')
                _require(frame.get('source') == 'native_solver_substeps' and frame.get('contacts_complete') is True,
                         'non_native_or_incomplete_frame')
                _require(type(frame['substep_index']) is int and frame['substep_index'] == last_step+1
                    and frame['dt_s'] == dt and math.isclose(frame['simulation_time_s'], last_time+dt, abs_tol=1e-8, rel_tol=0),
                    'native_substep_or_clock_gap')
                _require(set(frame['objects']) == actors, 'native_actor_inventory_changed')
                for body in frame['objects'].values():
                    for name, size in [('position_m', 3), ('quaternion_wxyz', 4), ('linear_velocity_m_s', 3), ('angular_velocity_rad_s', 3)]:
                        _require(isinstance(body[name], list) and len(body[name]) == size
                            and all(type(v) in (int, float) and math.isfinite(v) for v in body[name]), 'invalid_native_object_state')
                    _require(abs(sum(v*v for v in body['quaternion_wxyz'])-1.) <= .001, 'invalid_native_quaternion')
                result = row['result']
                if formal:
                    _require(type(result.get('committed')) is bool, 'missing_formal_commit_flag')
                    if result['committed']:
                        confirmations.append((None, frame['substep_index'], frame['simulation_time_s']))
                else:
                    _require(result.get('committed') is False, 'practice_forged_formal_commit')
                    index, reason, success = (result.get(k) for k in
                        ('practice_stage_index', 'reason', 'practice_success'))
                    _require(_count(index) and index <= practice_stages and type(success) is bool,
                             'invalid_practice_stage_result')
                    if terminal_success:
                        # A success latch is repeated for the remaining native
                        # substeps of the current ACK. It is not a new event.
                        _require(index == stage and success and reason == 'success',
                                 'practice_terminal_success_changed')
                    else:
                        _require(index in (stage, stage+1), 'practice_stage_jump_or_backtrack')
                        advanced = index == stage+1
                        if advanced:
                            final = index == practice_stages
                            _require(reason == ('success' if final else 'stage_confirmed') and success is final,
                                     'practice_stage_transition_reason_mismatch')
                            _require(stable_start is not None and
                                frame['simulation_time_s']-stable_start+1.e-10 >= dwell,
                                'practice_stage_without_stability_window')
                        else:
                            _require(not success and reason not in ('stage_confirmed', 'success'),
                                     'practice_confirmation_without_transition')
                        if reason == 'settling' and stable_start is None:
                            stable_start = frame['simulation_time_s']
                            stable_positions = {n: obj['position_m'] for n, obj in frame['objects'].items()}
                        if reason == 'settling' or advanced:
                            _require(all(math.dist(frame['objects'][n]['position_m'], pos) <= drift
                                for n, pos in stable_positions.items()), 'practice_stability_drift_mismatch')
                        else:
                            stable_start, stable_positions = None, None
                        if advanced:
                            confirmations.append((index, frame['substep_index'], frame['simulation_time_s'], stable_start))
                            stage, terminal_success = index, success
                            stable_start, stable_positions = None, None
                if frame_audit is not None:
                    frame_audit(row)
                count += 1
                last_step, last_time = frame['substep_index'], frame['simulation_time_s']
        after = os.fstat(fd)
        _require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                 (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'native_stream_changed_during_read')
    finally:
        os.close(fd)
    _require(count > 0 and count == observer['samples'] and last_step == observer['last_native_step'], 'native_closure_count_mismatch')
    if not formal:
        progress = evidence['score']['practice_progress']
        _require(progress == dict(stage_index=stage, stage_count=practice_stages, confirmed_stages=len(confirmations)),
                 'practice_progress_differs_from_native_transitions')
    reader.hashes['episode/'+relative] = digest.hexdigest()
    return count, last_time, confirmations


def _native(reader, plan, slot, report, agent, hashes, phase):
    episode_id = report['episode_id']
    _require(reader.root.name == episode_id, 'episode_directory_mismatch')
    spec, result = reader.json('agent/spec.json'), reader.json('agent/result.json')
    _require(spec.get('episode_id') == episode_id and spec.get('task_id') == plan.gym_id
        and result.get('spec') == spec and report.get('result') == result, 'episode_result_binding_mismatch')
    _require_task_outcome(result)
    ready = agent.json('episode_ready.json')
    config = agent.json('agent_config.json')
    _require(ready.get('spec') == spec and spec.get('agent_config_sha256') == content_sha256(config)
        and config.get('task_id') == plan.gym_id and report.get('ready_validated') is True, 'agent_ready_binding_mismatch')
    source = _source_record(agent, 'source_verification.json', plan.source_manifest_sha256,
                            ['agent.model_client.function_runner'])
    _require(config.get('source_verification_sha256') == content_sha256(source), 'agent_source_receipt_mismatch')
    _source_record(reader, 'private/runtime_imports.json', plan.source_manifest_sha256, ['bench.runtime.worker'])
    _source_record(reader, 'private/task_runtime/runtime_imports_final.json', plan.source_manifest_sha256,
        ['scenes.workrooms.card_room.practice.runtime', 'scenes.workrooms.card_room.practice.referee',
         'scenes.workrooms.card_room.practice.cases', 'scenes.board_games.expansion_native_observer',
         'scenes.workrooms.card_room.expansion_referee'])
    worker = reader.json('private/worker_config.json')
    _require(worker.get('output') == str(reader.root/'private') and worker.get('task_id') == plan.gym_id,
             'worker_episode_binding_mismatch')
    from bench.isaacsim.release import task_inputs
    release = checked_directory(worker['release'])
    manifest_bytes = read_bytes(release/'release_manifest.json')
    _require(hashlib.sha256(manifest_bytes).hexdigest() == spec['task_release_sha256'], 'gym_release_pin_mismatch')
    manifest = strict_json_loads(manifest_bytes)
    revised_practice = manifest.get('schema') == 'workroom_robot_revision_release.v1'
    if revised_practice:
        from bench.isaacsim.release import validate
        validated = validate(release)
        _require(validated == manifest and validated.get('selection_kind') == 'practice'
            and validated.get('parent_schema') == 'card_rsi_practice_release.v1', 'non_practice_release')
        _source_record(reader, 'private/task_runtime/runtime_imports_final.json', plan.source_manifest_sha256,
            ['scenes.workrooms.robot_revision.practice_runtime', 'scenes.workrooms.robot_revision.practice',
             'scenes.workrooms.robot_revision.inputs', 'scenes.workrooms.robot_revision.release'])
    else:
        _require(manifest.get('schema') == 'card_rsi_practice_release.v1', 'non_practice_release')
    hashes['release/release_manifest.json'] = hashlib.sha256(manifest_bytes).hexdigest()
    task = reader.json('private/task.json')
    _, _, expected_task, _ = task_inputs(release, plan.gym_id, seed=worker.get('seed', 0), practice=worker.get('practice'))
    _require(task == expected_task, 'worker_task_differs_from_frozen_release')
    case, public = task['practice_case'], task['practice_case']['public']
    _require(public.get('split') == ('train' if phase == 'practice' else 'validation'), 'case_learning_split_mismatch')
    _require(case['private_case_sha256'] == _scene_hash({k: v for k, v in case.items() if k != 'private_case_sha256'})
        and public['case_sha256'] == _scene_hash({k: v for k, v in public.items() if k != 'case_sha256'}), 'case_content_hash_mismatch')
    for key in ('gym_id', 'parent_task_id', 'parent_release_sha256', 'public_rules_sha256'):
        _require(public.get(key) == getattr(plan, key), 'case_plan_identity_mismatch')
    for key in ('active_side', 'stratum', 'case_id', 'case_sha256'):
        _require(public.get(key) == getattr(slot, key), 'case_slot_identity_mismatch')
    _require(case['private_case_sha256'] == slot.private_case_sha256
        and case['case_fingerprint'] == slot.case_fingerprint, 'private_case_identity_mismatch')
    if revised_practice:
        from scenes.workrooms.robot_revision.practice import verified_legacy_view, derive_case
        legacy = verified_legacy_view(task)
        derived = derive_case(legacy['practice_case'], task['asset_metadata'],
            revision_sha256=manifest['robot_revision_sha256'],
            formal_release_sha256=manifest['parent_release']['manifest_sha256'])
        _require(derived == case, 'robot_revision_case_derivation_mismatch')
        expected_fingerprint = derived['case_fingerprint']
    else:
        expected_fingerprint = _scene_hash({'parent_release_sha256': public['parent_release_sha256'],
            'reset': case['reset'], 'active_side': public['active_side'], 'goal': public['goal'],
            'private_pairs': task['asset_metadata'].get('private_reset_pairs')})
    _require(expected_fingerprint == slot.case_fingerprint, 'reset_goal_fingerprint_mismatch')
    evidence = reader.json('private/task_runtime/evidence.json')
    _require(evidence.get('schema') == 'card_rsi_practice_native_evidence.v1'
        and evidence.get('episode_id') == episode_id and evidence.get('task_id') == plan.gym_id
        and evidence.get('active_side') == slot.active_side and evidence.get('case_id') == slot.case_id
        and evidence.get('split') == public['split']
        and evidence.get('case_sha256') == slot.case_sha256
        and evidence.get('private_case_sha256') == slot.private_case_sha256
        and evidence.get('case_fingerprint') == slot.case_fingerprint
        and evidence.get('release_manifest_sha256') == spec['task_release_sha256']
        and evidence.get('parent_release_sha256') == plan.parent_release_sha256
        and evidence.get('public_rules_sha256') == plan.public_rules_sha256
        and evidence.get('evaluator_sources_sha256') == content_sha256(task['sources_sha256'])
        and evidence.get('inactive_robot_drive_locked') is True
        and all(evidence.get(k) is False for k in ('object_state_writes', 'robot_target_overrides', 'formal_game_result')),
        'native_evidence_binding_mismatch')
    observer, inputs = evidence['observer'], reader.json('private/task_runtime/observer_inputs.json')
    _require(evidence.get('observer_inputs_sha256') == hashes['episode/private/task_runtime/observer_inputs.json']
        and evidence.get('runtime_imports_final_sha256') == hashes['episode/private/task_runtime/runtime_imports_final.json']
        and evidence.get('native_start_step') == inputs['native_step']
        and evidence.get('physics_dt_s') == inputs['physics_dt_s'], 'native_input_hash_mismatch')
    _require(observer.get('schema') == 'card_game_expansion_native_observer.v1'
        and observer.get('failed') is False and observer.get('unsupported_samples') == 0
        and observer.get('final_context_verified') is True, 'native_observer_not_closed')
    closure = observer.get('closure_checks', {})
    required = {'same_reset_epoch', 'same_contact_view', 'same_contact_mapping', 'same_body_mapping',
                'same_physics_dt', 'no_unobserved_native_step', 'same_public_time', 'same_native_time', 'acquisition_ends_at_last_step'}
    _require(required.issubset(closure) and all(v is True for v in closure.values())
        and all(observer.get(k) == 0 for k in ('physical_steps_added', 'additional_contact_acquisitions', 'physical_state_setters_called'))
        and observer.get('reset_epoch') == inputs['reset_epoch'] and inputs.get('physical_state_setters_called') == 0
        and inputs['mass_properties_verified'] and all(v is True for v in inputs['mass_properties_verified'].values()),
        'native_closure_or_mass_invalid')
    metadata, audit = reader.json('private/metadata.json'), reader.json('private/physics_audit.json')
    _require(metadata.get('native_isaacsim_app') is True
        and metadata.get('native_physics_tensor_pipeline', {}).get('cuda_context_available') is True
        and metadata.get('native_readback_configuration', {}).get('gpu_dynamics_enabled') is True,
        'not_native_gpu_runtime')
    _require(audit.get('schema') == 'native_agent_physics_audit.v1' and audit.get('physics_valid') is True
        and result.get('physics_valid') is True, 'native_physics_not_valid')
    count, elapsed, confirmations = _native_frames(reader, inputs, observer, evidence,
        practice_stages=len(public['goal']['stages']),
        practice_thresholds=task['public_rules']['parent_game_rules']['thresholds'])
    _require(evidence.get('native_frames_sha256') == hashes['episode/private/task_runtime/native_frames.jsonl'],
             'native_stream_hash_mismatch')
    _require(math.isclose(elapsed, result['elapsed_sim_s'], abs_tol=1e-8, rel_tol=0), 'native_result_clock_mismatch')
    _require(reader.json('private/final_score.json') == evidence['score'], 'native_final_score_mismatch')
    evaluator = reader.json('private/evaluator_evidence.json')
    _require(hashlib.sha256(read_bytes(reader.path('private/evaluator_evidence.json'))).hexdigest()
        == result['evaluator_evidence_sha256'], 'terminal_evaluator_hash_mismatch')
    # Server seals a score immediately before closing an ongoing practice.
    # Its success bit and completed-stage count must still match native close.
    _require(evaluator.get('success') is evidence['score'].get('success')
        and evaluator.get('practice_progress') == evidence['score'].get('practice_progress'), 'terminal_score_mismatch')
    recorded = evidence['confirmations']
    _require(len(recorded) == len(confirmations) and all(
        row['stage_index']+1 == actual[0] and row['native_substep'] == actual[1]
        and math.isclose(row['simulation_time_s'], actual[2], rel_tol=0, abs_tol=1e-8)
        and math.isclose(row['stable_start_s'], actual[3], rel_tol=0, abs_tol=1e-8)
        for row, actual in zip(recorded, confirmations)), 'stage_confirmation_mismatch')
    success = evidence['score'].get('success') is True
    _require(success == (result.get('outcome') == 'success'), 'official_native_success_mismatch')
    outcome = 'success' if success else report.get('task_outcome')
    _require(outcome in ('success', 'failure'), 'no_authoritative_native_task_outcome')
    digest = content_sha256({k: v for k, v in hashes.items() if k.startswith(('episode/', 'source/', 'release/'))})
    violations = int(evidence['score'].get('reason') == 'rule_violation')
    return outcome, count, elapsed, digest, violations


def collect_trial_record(plan, slot, *, candidate_event_id, bundle_hash, agent_output: Path,
                         episode_root: Path | None, attempt_id, attempt_index, receipt_sequence,
                         phase='validation', wall_s, cpu_s) -> TrialRecord:
    plan, slot = EvaluationPlan.model_validate(plan), ValidationSlot.model_validate(slot)
    hashes, issues = {}, []
    # Cost cannot silently become zero after corrupt/missing receipts. This is
    # intentionally a hard accounting error rather than an invented TrialRecord.
    cost, report, actions = _effort_details(agent_output, wall_s=wall_s, cpu_s=cpu_s, hashes=hashes)
    episode_id = report.get('episode_id')
    invocations, language_entries, bundles, temporary, fresh, history = 0, 0, [], False, False, True
    binding = None
    outcome, count, native_digest, violations = 'unknown', 0, None, 0
    reader = _Reader(agent_output, 'agent', hashes)
    for stage in ('binding', 'skills', 'native'):
        try:
            if stage == 'binding':
                binding = reader.json('host/rsi_trial_binding.json')
                _require(binding.get('schema') == 'card_rsi_trial_binding.v1'
                    and binding.get('phase') == phase and binding.get('episode_id') == episode_id
                    and binding.get('attempt_id') == attempt_id and binding.get('trial_id') == 'single'
                    and binding.get('condition_sha256') == plan.condition_sha256
                    and binding.get('source_manifest_sha256') == plan.source_manifest_sha256
                    and binding.get('skill_runtime_journal') == 'single/host/skill_runtime_journal'
                    and binding.get('action_journal') == 'single/host/action_journal', 'trial_binding_mismatch')
                if phase != 'practice':
                    _require(binding.get('candidate_event_id') == candidate_event_id
                        and binding.get('bundle_hash') == bundle_hash, 'trial_candidate_pin_mismatch')
                _require(type(binding.get('fresh_context')) is bool and type(binding.get('author_history_visible')) is bool,
                         'unknown_context_independence')
                fresh, history = binding['fresh_context'], binding['author_history_visible']
                _require(actions is not None and actions['pending'] is None and not report.get('action_recovery_pending'),
                         'unresolved_original_action')
                _require(actions['binding']['contract'].get('spec_sha256') == content_sha256(
                    (report.get('result') or {}).get('spec')), 'action_spec_binding_mismatch')
            elif stage == 'skills':
                if any(issue.startswith('binding:') for issue in issues):
                    continue
                invocations, language_entries, bundles, temporary = _skills(reader, binding, phase=phase, episode_id=episode_id,
                    candidate_event_id=candidate_event_id, bundle_hash=bundle_hash)
            else:
                _require(episode_root is not None, 'native_episode_directory_missing')
                outcome, count, elapsed, native_digest, violations = _native(
                    _Reader(episode_root, 'episode', hashes), plan, slot, report, reader, hashes, phase)
                cost.native_sim_s = elapsed
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
            issues.append(stage+':'+(str(error) if isinstance(error, ReceiptError) else type(error).__name__))
            if stage == 'skills' and binding is not None:
                # A durable started event still consumed an invocation even
                # when its result was lost. Keep that cost/count, while no
                # unverified entry into the language earns execution credit.
                try:
                    events = reader.journal(binding['skill_runtime_journal'])
                    starts = [e['payload'] for e in events if e['kind'] == 'skill_execution_started']
                    _require(len({s['execution_id'] for s in starts}) == len(starts)
                        and all(re.fullmatch('[0-9a-f]{64}', s['bundle_hash']) for s in starts),
                        'invalid_skill_start_receipts')
                    invocations = len(starts)
                    bundles = list(dict.fromkeys(s['bundle_hash'] for s in starts))
                except (ValueError, TypeError, KeyError, OSError):
                    pass
    verified = not issues and native_digest is not None
    status = report.get('run_status', 'unknown')
    if status not in ('completed', 'agent_budget', 'agent_error', 'infra_error'):
        status = 'unknown'
    result = report.get('result')
    record = TrialRecord(**slot.model_dump(), phase=phase, gym_id=plan.gym_id, parent_task_id=plan.parent_task_id,
        parent_release_sha256=plan.parent_release_sha256, public_rules_sha256=plan.public_rules_sha256,
        condition_sha256=plan.condition_sha256, source_manifest_sha256=plan.source_manifest_sha256,
        candidate_event_id=candidate_event_id, bundle_hash=bundle_hash, attempt_id=attempt_id,
        attempt_index=attempt_index, receipt_sequence=receipt_sequence, episode_id=episode_id,
        execution_status=status, outcome=outcome if verified else 'unknown', physics_valid=True if verified else None,
        result_valid=verified, evidence_source='native_solver_substeps' if verified else 'unverified',
        native_substeps=count, native_evidence_sha256=native_digest if verified else None,
        result_sha256=content_sha256(result) if result is not None else None,
        public_feedback_sha256=None, skill_invocations=invocations, skill_language_entries=language_entries,
        executed_bundle_hashes=bundles,
        temporary_skill_changes=temporary, fresh_context=fresh, author_history_visible=history,
        rule_violations=violations, cost=cost, verification_issues=issues, artifact_sha256=hashes)
    record.public_feedback_sha256 = content_sha256(public_practice_feedback(record, report))
    return record


def public_practice_feedback(record: TrialRecord | None, report: dict):
    """Small next-practice projection; no arbitrary strings, hashes or raw state."""
    termination = report.get('termination')
    allowed_termination = {'client_wall_deadline', 'max_decisions', 'server_finished', 'run_error',
                          'interrupted', 'agent_finished', 'server_finished_during_model_call'}
    error = report.get('run_error') or {}
    category = error.get('error_category')
    allowed_error = {'transport_timeout', 'transport_error', 'http_status', 'invalid_response',
                     'request_budget_exhausted', 'invalid_json', 'empty_response', 'action_recovery_error'}
    return {'schema': 'card_rsi_public_practice_feedback.v1', 'phase': 'practice',
        'attempt_id': record.attempt_id if record is not None else None,
        'outcome': record.outcome if record is not None else 'unknown',
        'termination': termination if termination in allowed_termination else 'other',
        'last_public_error': category if category in allowed_error else None,
        'candidate_event_id': record.candidate_event_id if record is not None else None,
        'bundle_hash': record.bundle_hash if record is not None else None,
        'native_evidence_verified': record.native_result() if record is not None else False,
        'formal_game_result': False}


def _verified_model_request(reader, relative):
    """Reconstruct the recorded public image references to check actual wire hash."""
    request = reader.json(relative+'/model_request.json')
    messages = []
    contexts = []
    for original in request['messages']:
        message = dict(original)
        content = message['content']
        if isinstance(content, list):
            expanded = []
            for item in content:
                if item.get('type') == 'recorded_image':
                    raw = reader.raw(item['path'])
                    _require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'model_image_hash_mismatch')
                    if 'source_path' in item:
                        _require(hashlib.sha256(reader.raw(item['source_path'])).hexdigest() == item['source_sha256'],
                                 'model_depth_source_hash_mismatch')
                    expanded.append({'type': 'image_url', 'image_url': {
                        'url': 'data:image/png;base64,'+base64.b64encode(raw).decode()}})
                else:
                    expanded.append(item)
            message['content'] = expanded
            texts = [item['text'] for item in content if item.get('type') == 'text']
        else:
            texts = [content] if isinstance(content, str) else []
        if message.get('role') == 'user':
            for text in texts:
                try:
                    value = strict_json_loads(text)
                except ValueError:
                    continue
                if isinstance(value, dict) and isinstance(value.get('prior_practice_feedback'), list):
                    contexts.extend(value['prior_practice_feedback'])
        messages.append(message)
    _require(content_sha256(messages) == request['expanded_messages_sha256'], 'expanded_model_request_hash_mismatch')
    feedbacks = []
    for value in contexts:
        if isinstance(value, dict):
            candidate = value.get('assessment', value)
            if isinstance(candidate, dict) and candidate.get('schema') == 'card_rsi_public_practice_feedback.v1':
                feedbacks.append(candidate)
    return feedbacks


def collect_revision_receipt(*, library_root: Path, previous_candidate_event_id: str,
                             candidate_event_id: str, agent_output: Path,
                             practice_records: list[TrialRecord], receipt_sequence: int,
                             diagnostics: dict | None = None) -> RevisionReceipt | None:
    """Prove a cross-episode model code revision after actual native feedback.

    A revision is not a performance improvement. Within-episode unsealed
    feedback and ambiguous mixed-bundle practice are deliberately not promoted
    by this first verifier. Missing evidence yields None plus a fixed diagnostic
    code; it never invents an authorship/feedback edge.
    """
    diagnostic = diagnostics if diagnostics is not None else {}
    diagnostic.clear()
    hashes = {}
    try:
        library, agent = _Reader(library_root, 'library', hashes), _Reader(agent_output, 'agent', hashes)
        events = library.journal('host/journal')
        experiment = library.json('host/experiment.json')
        _require(events and events[0]['kind'] == 'experiment_created'
            and events[0]['payload'].get('experiment_sha256') == content_sha256(experiment), 'library_origin_mismatch')
        candidates = {e['event_id']: e for e in events if e['kind'] == 'candidate_committed'}
        previous, current = candidates[previous_candidate_event_id], candidates[candidate_event_id]
        _require(previous['sequence'] < current['sequence']
            and previous['payload'].get('import') is None and current['payload'].get('import') is None,
            'revision_not_local_forward_publication')
        before, after = previous['payload']['bundle_hash'], current['payload']['bundle_hash']
        _require(before != after, 'unchanged_bundle')
        from .skills import SkillLibrary
        # Verify immutable content without opening a writer transaction.
        checker = object.__new__(SkillLibrary)
        checker.bundles = checked_directory(library.root/'bundles')
        old_manifest, new_manifest = checker.verify(before), checker.verify(after)
        old_code = {r['path']: r['sha256'] for r in old_manifest['files'] if r['path'].endswith('.py')}
        new_code = {r['path']: r['sha256'] for r in new_manifest['files'] if r['path'].endswith('.py')}
        _require(set(old_code).issubset(new_code), 'source_deletion_has_no_model_write_receipt')
        changed = {name for name, digest in new_code.items() if old_code.get(name) != digest}
        semantic_changes = set()
        for name in changed:
            source = library.raw('bundles/'+after+'/payload/'+name).decode('utf-8')
            old = library.raw('bundles/'+before+'/payload/'+name).decode('utf-8') if name in old_code else None
            if old is None or ast.dump(ast.parse(source), include_attributes=False) != ast.dump(ast.parse(old), include_attributes=False):
                semantic_changes.add(name)
        _require(semantic_changes, 'no_executable_source_change')
        practice = [TrialRecord.model_validate(r) for r in practice_records]
        eligible = [r for r in practice if r.phase == 'practice' and r.native_result()
            and r.bundle_hash == before and r.executed_bundle_hashes == [before]
            and r.skill_language_entries > 0 and not r.temporary_skill_changes
            and r.receipt_sequence < receipt_sequence and r.public_feedback_sha256]
        _require(eligible, 'no_unambiguous_prior_native_practice')
        sources = {r.source_manifest_sha256 for r in eligible}
        conditions = {r.condition_sha256 for r in eligible}
        _require(len(sources) == len(conditions) == 1, 'practice_condition_changed')
        source = _source_record(agent, 'source_verification.json', next(iter(sources)),
            ['agent.model_client.function_runner'])
        config = agent.json('agent_config.json')
        binding = agent.json('host/rsi_trial_binding.json')
        _require(config.get('source_verification_sha256') == content_sha256(source)
            and binding.get('phase') == 'practice' and binding.get('condition_sha256') in conditions
            and binding.get('source_manifest_sha256') in sources, 'revision_trial_condition_mismatch')
        runtime = agent.journal('single/host/skill_runtime_journal')
        _require(runtime and runtime[0]['kind'] == 'skill_runtime_bound'
            and runtime[0]['payload'].get('protocol') == 'agent_skill_runtime.v1'
            and runtime[0]['payload'].get('library_experiment_sha256') == content_sha256(experiment)
            and runtime[0]['payload'].get('phase') == 'practice'
            and runtime[0]['payload'].get('episode_id') == binding.get('episode_id'), 'revision_runtime_origin_mismatch')
        _model_effort(agent, agent.json('run_report.json'))
        tool_pairs, pending = [], None
        for event in runtime[1:]:
            if event['kind'] == 'skill_tool_requested':
                _require(pending is None, 'overlapping_skill_tool_requests')
                pending = event
            elif event['kind'] == 'skill_tool_finished':
                if pending is not None:
                    _require(pending['payload']['function'] == event['payload']['function'], 'skill_tool_pair_mismatch')
                    tool_pairs.append((pending, event))
                    pending = None
        writes, publications = {}, []
        model_calls = []
        from ..actions import parse_json_object
        for folder in sorted((agent.root/'single/decisions').glob('*')):
            if not folder.is_dir() or not (folder/'model_response.json').is_file() or not (folder/'execution.json').is_file():
                continue
            relative = str(folder.relative_to(agent.root))
            response = agent.json(relative+'/model_response.json')
            try:
                decision = parse_json_object(response['text'])
            except ValueError:
                continue
            if set(decision) != {'function', 'arguments'} or decision['function'] not in ('skill_write', 'skill_publish'):
                continue
            metadata = response['metadata']
            _require(_count(metadata.get('attempts')) and metadata['attempts'] > 0, 'no_actual_model_request')
            expected_model = config.get('models', {}).get('single', {}).get('model')
            _require(expected_model and metadata.get('model') == metadata.get('requested_model') == expected_model,
                     'revision_model_identity_mismatch')
            feedback = agent.json(relative+'/execution.json')
            if feedback.get('ok') is not True or feedback.get('function') != decision['function']:
                continue
            matches = [(a, b) for a, b in tool_pairs if a['payload'] == decision
                and b['payload'].get('ok') is True and b['payload'].get('feedback_sha256') == content_sha256(feedback)]
            _require(len(matches) == 1, 'model_tool_execution_not_unique')
            request_feedback = _verified_model_request(agent, relative)
            linked = [r for r in eligible if r.public_feedback_sha256 in {content_sha256(v) for v in request_feedback}]
            row = dict(decision=decision, feedback=feedback, folder=relative, pair=matches[0], linked=linked, attempts=metadata['attempts'])
            model_calls.append(row)
            if decision['function'] == 'skill_write':
                name = 'skills/'+decision['arguments']['skill_id']+'.py'
                source_sha = hashlib.sha256(decision['arguments']['source'].encode()).hexdigest()
                if name in changed and source_sha == new_code[name] and feedback['result'].get('source_sha256') == source_sha:
                    changes = [e for e in events if previous['sequence'] < e['sequence'] < current['sequence']
                        and e['kind'] == 'author_changed' and e['payload'].get('path') == name
                        and e['payload'].get('sha256') == source_sha and e['payload'].get('operation') == 'write'
                        and e['payload'].get('previous_revision') == decision['arguments']['expected_revision']]
                    _require(len(changes) == 1, 'model_source_write_not_in_library_journal')
                    write_event = changes[0]
                    next_event = next((e for e in events if e['sequence'] == write_event['sequence']+1), None)
                    _require(next_event and next_event['kind'] == 'author_changed'
                        and next_event['payload'].get('path') == 'catalog.json'
                        and next_event['payload'].get('previous_revision') == write_event['payload']['revision']
                        and next_event['payload'].get('revision') == feedback['result'].get('author_revision'),
                        'model_catalog_write_not_bound')
                    writes[name] = row
            elif feedback.get('result', {}).get('candidate_event_id') == candidate_event_id:
                _require(feedback['result'].get('bundle_hash') == after
                    and decision['arguments'].get('transaction_id') == current['transaction_id']
                    and decision['arguments'].get('expected_revision') == current['payload']['author_revision'],
                    'model_publication_differs_from_candidate')
                publications.append(row)
        _require(set(writes) == changed, 'changed_source_lacks_model_authorship')
        _require(len(publications) == 1, 'candidate_not_published_by_this_model_run')
        publish = publications[0]
        _require(all(w['pair'][1]['sequence'] < publish['pair'][0]['sequence'] for w in writes.values()),
                 'publication_precedes_model_source_write')
        witnessed = [writes[name] for name in semantic_changes if writes[name]['linked']]
        _require(witnessed, 'native_feedback_not_in_actual_revision_request')
        witness = min(witnessed, key=lambda row: row['pair'][0]['sequence'])
        receipt = RevisionReceipt(receipt_sequence=receipt_sequence, author='model', previous_bundle_hash=before,
            candidate_event_id=candidate_event_id, bundle_hash=after, changed_executable_content=True,
            model_request_sha256=hashes['agent/'+witness['folder']+'/model_request.json'],
            model_response_sha256=hashes['agent/'+witness['folder']+'/model_response.json'],
            actual_model_requests=sum(row['attempts'] for row in model_calls),
            practice_attempt_ids=[r.attempt_id for r in witness['linked']],
            public_feedback_sha256=list(dict.fromkeys(r.public_feedback_sha256 for r in witness['linked'])))
        diagnostic.update(verified=True, reason='model_revision_after_native_practice', artifact_sha256=hashes)
        return receipt
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, SyntaxError) as error:
        diagnostic.update(verified=False, reason=str(error) if isinstance(error, ReceiptError) else type(error).__name__,
                          artifact_sha256=hashes)
        return None
