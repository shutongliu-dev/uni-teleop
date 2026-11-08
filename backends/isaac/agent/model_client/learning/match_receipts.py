"""Posthoc frozen-match evidence verification; no native or model execution.

Runtime artifacts retain their actual source snapshot. The auditor's source is
recorded separately; hashes below bind files at audit time, not at original
write time. This does not independently replay all game rules or calibrate
physical forces: the unchanged native referee remains the scoring authority.
"""
from __future__ import annotations

import hashlib
import argparse
import math
import os
from pathlib import Path
import stat

import numpy as np

from bench.protocol.card_games import CardGameState, EVENT_ADAPTER
from bench.protocol.board_games import PlayerBinding
from .assessment import FreezeManifest, FrozenMatchReceipt, content_sha256
from .journal import atomic_write_json, checked_directory, read_bytes, strict_json_loads
from .model_conditions import check_model_condition, check_retry_receipts
from .receipts import (_Reader, ReceiptError, _actions, _native_frames, _require,
                       _require_task_outcome, _skills, _source_record, collect_match_effort)


def _first_line_and_hash(reader, relative):
    path = reader.path(relative)
    checked_directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'invalid_reset_stream')
        digest = hashlib.sha256()
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            first = stream.readline(8*1024*1024)
            _require(first.endswith(b'\n'), 'missing_reset_record')
            digest.update(first)
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        after = os.fstat(fd)
        _require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                 (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'reset_stream_changed')
    finally:
        os.close(fd)
    reader.hashes[reader.label+'/'+relative] = digest.hexdigest()
    return strict_json_loads(first)


def _same_numbers(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            _same_numbers(actual[k], v) for k, v in expected.items())
    return (type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1.e-10, abs_tol=1.e-10))


def collect_frozen_match_receipts(freezes, *, library_roots, agent_output: Path,
                                  episode_root: Path, receipt_sequence: int,
                                  diagnostics: dict | None = None):
    """Return both actual player receipts, or None with a fixed failure reason.

    Zero skill invocations remains zero: a physically valid incomplete episode
    can be recorded without certifying that both frozen skills were exercised.
    ``assessment.audit_closed_loop`` applies that additional execution gate.
    """
    diagnostic = diagnostics if diagnostics is not None else {}
    diagnostic.clear()
    hashes = {}
    try:
        players = ('player_0', 'player_1')
        _require(set(freezes) == set(library_roots) == set(players), 'both_frozen_players_required')
        freezes = {p: FreezeManifest.model_validate(freezes[p]) for p in players}
        identities = {(f.parent_task_id, f.parent_release_sha256, f.source_manifest_sha256) for f in freezes.values()}
        _require(len(identities) == 1, 'frozen_players_disagree_on_runtime')
        task_id, release_sha, source_sha = next(iter(identities))
        agent, episode = _Reader(agent_output, 'agent', hashes), _Reader(episode_root, 'episode', hashes)
        report = agent.json('run_report.json')
        episode_id = report.get('episode_id')
        _require(episode.root.name == episode_id and report.get('ready_validated') is True
            and not report.get('action_recovery_pending'), 'match_episode_not_closed')
        spec = episode.json('agent/spec.json')
        result = episode.json('agent/board_game_result.json')
        physical_result = episode.json('agent/result.json')
        _require(report.get('result') == result and result.get('episode') == physical_result
            and physical_result.get('spec') == spec and spec.get('episode_id') == episode_id
            and spec.get('task_id') == task_id and spec.get('task_release_sha256') == release_sha,
            'formal_result_binding_mismatch')
        _require_task_outcome(physical_result)
        state = CardGameState.model_validate(result['board_game']).model_dump(mode='json')
        _require(state['terminal'] and state['task_id'] == task_id, 'formal_game_not_terminal')
        _require((state['status']['outcome'] in ('win', 'draw')) == (physical_result.get('outcome') == 'success'),
                 'formal_result_invents_completion')
        config = agent.json('agent_config.json')
        ready = agent.json('episode_ready.json')
        _require(config.get('task_id') == task_id and spec.get('agent_config_sha256') == content_sha256(config)
            and ready.get('spec') == spec
            and 'prior_practice_feedback_sha256' not in config, 'frozen_match_context_mismatch')
        bindings = [PlayerBinding.model_validate(value).model_dump(mode='json') for value in ready['players']]
        _require(len(bindings) == 2 and {v['player_id'] for v in bindings} == set(players)
            and not set(bindings[0]['joint_names']) & set(bindings[1]['joint_names']), 'invalid_frozen_player_partition')
        source = _source_record(agent, 'source_verification.json', source_sha, ['agent.model_client.function_runner'])
        _require(config.get('source_verification_sha256') == content_sha256(source), 'match_agent_source_mismatch')
        _source_record(episode, 'private/runtime_imports.json', source_sha, ['bench.runtime.worker'])
        _source_record(episode, 'private/runtime_imports_final.json', source_sha, [
            'bench.runtime.worker', 'bench.runtime.card_game_worker',
            'scenes.board_games.expansion_native_observer', 'scenes.workrooms.card_room.expansion_referee',
            'scenes.workrooms.card_room.expansion_rules'])
        worker = episode.json('private/worker_config.json')
        _require(worker.get('task_id') == task_id and worker.get('output') == str(episode.root/'private'),
                 'match_worker_binding_mismatch')
        release = checked_directory(worker['release'])
        release_bytes = read_bytes(release/'release_manifest.json')
        _require(hashlib.sha256(release_bytes).hexdigest() == release_sha, 'formal_release_hash_mismatch')
        hashes['release/release_manifest.json'] = release_sha
        from bench.isaacsim.release import task_inputs
        _, _, expected_task, _ = task_inputs(release, task_id, seed=worker.get('seed', 0))
        task = episode.json('private/task.json')
        _require(task == expected_task, 'formal_task_differs_from_release')
        evidence = episode.json('private/board_game/evidence.json')
        _require(evidence.get('schema') == 'card_game_expansion_worker_evidence.v1'
            and evidence.get('object_state_writes') is False and evidence.get('robot_target_overrides') is False
            and evidence.get('public_state') == state, 'formal_native_evidence_mismatch')
        inputs = episode.json('private/board_game/observer_inputs.json')
        observer = evidence['observer']
        closure = observer.get('closure_checks', {})
        required_closure = {'same_reset_epoch', 'same_contact_view', 'same_contact_mapping', 'same_body_mapping',
            'same_physics_dt', 'no_unobserved_native_step', 'same_public_time', 'same_native_time', 'acquisition_ends_at_last_step'}
        _require(observer.get('schema') == 'card_game_expansion_native_observer.v1'
            and observer.get('failed') is False and observer.get('unsupported_samples') == 0
            and observer.get('final_context_verified') is True and required_closure.issubset(closure)
            and all(v is True for v in closure.values()) and inputs['mass_properties_verified']
            and all(v is True for v in inputs['mass_properties_verified'].values())
            and observer.get('reset_epoch') == inputs.get('reset_epoch')
            and inputs.get('physical_state_setters_called') == 0
            and all(observer.get(k) == 0 for k in ('physical_steps_added', 'additional_contact_acquisitions', 'physical_state_setters_called')),
            'formal_native_closure_invalid')
        metadata, audit = episode.json('private/metadata.json'), episode.json('private/physics_audit.json')
        _require(metadata.get('native_isaacsim_app') is True
            and metadata.get('native_physics_tensor_pipeline', {}).get('cuda_context_available') is True
            and metadata.get('native_readback_configuration', {}).get('gpu_dynamics_enabled') is True
            and audit.get('schema') == 'native_agent_physics_audit.v1' and audit.get('physics_valid') is True
            and physical_result.get('physics_valid') is True, 'formal_native_physics_invalid')
        _require(type(inputs.get('native_step')) is int and inputs['native_step'] >= 0
            and type(inputs.get('reset_epoch')) is int and inputs['reset_epoch'] >= 0
            and type(inputs.get('physics_dt_s')) in (int, float) and inputs['physics_dt_s'] > 0
            and metadata.get('physics_dt_s') == inputs['physics_dt_s'], 'formal_native_clock_input_mismatch')
        reset = _first_line_and_hash(episode, 'private/physics.jsonl')
        _require(reset.get('kind') == 'reset', 'native_reset_prefix_missing')
        start_time = reset['observation']['sim_time_s']
        _require(type(start_time) in (int, float) and math.isfinite(start_time) and start_time >= 0, 'invalid_native_start_clock')
        archives = [agent.json(p+'/public_game_events.json') for p in players]
        _require(archives[0] == archives[1] and archives[0].get('cursor') == state['event_cursor']
            and archives[0].get('protocol') == 'bench_card_game_state_v1', 'players_missed_public_events')
        events = [EVENT_ADAPTER.validate_python(e).model_dump(mode='json') for e in archives[0]['events']]
        _require([e['sequence'] for e in events] == list(range(state['event_cursor']+1))
            and all(e['task_id'] == task_id for e in events), 'public_event_sequence_mismatch')
        event_steps = {e['native_substep_index']: [] for e in events}
        for event in events:
            event_steps[event['native_substep_index']].append(event)
        operations = [e for e in events if e['type'] == 'operation_committed']
        _require([e['turn_index'] for e in operations] == list(range(1, state['turn_index']+1)), 'public_operation_epoch_gap')
        from scenes.board_games import expansion_native_observer as aggregation
        module_sha = hashlib.sha256(read_bytes(Path(aggregation.__file__))).hexdigest()
        _require(module_sha == task['sources_sha256']['scenes/board_games/expansion_native_observer.py'],
                 'auditor_aggregation_differs_from_pinned_rule_source')
        paths = {path: name for name, path in task['asset_metadata']['actors'].items()}
        physical = task['asset_metadata']['mass_properties']
        body_paths = inputs['body_paths']
        _require(len(body_paths) == len(paths) and set(body_paths) == set(paths)
            and set(inputs['mass_properties_verified']) == set(physical) == set(paths.values())
            and inputs['expected_mass_properties'] == physical, 'native_inventory_or_mass_source_mismatch')
        masses = np.asarray(inputs['masses_kg'], dtype=float)
        coms = np.asarray(inputs['coms'], dtype=float)
        inertias = np.asarray(inputs['inertias_kg_m2'], dtype=float)
        _require(masses.shape == (len(paths),) and coms.shape == (len(paths), 7)
            and inertias.shape == (len(paths), 3, 3) and np.isfinite(masses).all()
            and np.isfinite(coms).all() and np.isfinite(inertias).all(), 'native_mass_readback_shape_invalid')
        for i, body_path in enumerate(body_paths):
            expected = physical[paths[body_path]]
            _require(np.isclose(masses[i], expected['mass_kg'], rtol=1e-5, atol=1e-8)
                and np.allclose(coms[i, :3], expected['center_of_mass_m'], rtol=0, atol=2e-7)
                and np.allclose(inertias[i], expected['inertia_tensor_kg_m2'], rtol=2e-4, atol=1e-10),
                'native_mass_readback_differs_from_frozen_asset')
        scene = metadata['scene_info']
        owners = aggregation.robot_ownership(scene['robot_link_paths'])
        decks = [scene['collider_actor_paths'].get(p, p) for p in task['asset_metadata']['fixture_collision_paths']]
        statics = (set(scene['support_paths']) | {p for v in scene['fixtures'].values()
                    for p in v.get('contact_filter_paths', v.get('collision_paths', []))}) - set(decks) - set(paths) - set(owners)
        _require(inputs['robot_contact_ownership'] == observer['robot_contact_ownership'] == owners
            and inputs['deck_contact_paths'] == observer['deck_contact_paths'] == decks
            and set(inputs['known_static_contact_paths']) == statics
            and all(scene['objects'].get(name, {}).get('path') == p for p, name in paths.items()),
            'native_contact_mapping_changed')
        seen_events = set()
        contacts = 0
        def audit_frame(row):
            nonlocal contacts
            frame = row['frame']
            contacts += len(row['raw_contacts'])
            values = aggregation.aggregate_contacts(row['raw_contacts'], now=frame['simulation_time_s'], paths=paths,
                deck_paths=inputs['deck_contact_paths'], owners=inputs['robot_contact_ownership'],
                static_paths=inputs['known_static_contact_paths'])
            for name, body in frame['objects'].items():
                expected = dict(deck_support_impulse_ns=values[0][name], robot_contact_impulses_ns=values[1][name],
                    robot_nonhand_contact_impulses_ns=values[2][name], object_support_impulses_ns=values[3][name],
                    maximum_penetration_m=values[4][name], other_static_support_impulse_ns=values[5][name])
                _require(all(_same_numbers(body.get(k), v) for k, v in expected.items()), 'raw_contact_projection_mismatch')
            for event in event_steps.get(frame['substep_index'], []):
                _require(math.isclose(event['simulation_time_s'], frame['simulation_time_s'], abs_tol=1e-8, rel_tol=0),
                         'public_event_native_clock_mismatch')
                if event['type'] == 'operation_committed':
                    _require(row['result']['committed'] is True, 'public_commit_without_native_commit')
                seen_events.add(event['sequence'])
        count, elapsed, commits = _native_frames(episode, inputs, observer, {'native_start_time_s': start_time},
            runtime='board_game', formal=True, frame_audit=audit_frame)
        _require(len(commits) == state['commit_count'] == len(operations)
            and seen_events == set(range(len(events)))
            and math.isclose(elapsed, physical_result['elapsed_sim_s'], rel_tol=0, abs_tol=1.e-8),
            'formal_result_native_trace_mismatch')
        expected_score = dict(state, schema='card_game_expansion_worker_score.v1',
            success=state['status']['outcome'] in ('win', 'draw'), outcome=state['status']['outcome'],
            reason=state['status']['reason'], end_reason=state['status']['end_reason'],
            native_observer_samples=count, native_observer_unsupported_samples=0)
        _require(evidence['score'] == expected_score, 'formal_native_score_disagrees_with_state')
        _require(episode.json('private/final_score.json') == evidence['score'], 'formal_final_score_mismatch')
        _require(episode.json('private/evaluator_evidence.json') == evidence['score'], 'formal_evaluator_score_mismatch')
        _require(hashes['episode/private/evaluator_evidence.json'] == physical_result['evaluator_evidence_sha256'],
                 'formal_evaluator_evidence_hash_mismatch')
        accounting = collect_match_effort(agent_output)
        receipts = {}
        from .skills import SkillLibrary
        for player in players:
            freeze = freezes[player]
            _require(receipt_sequence > freeze.freeze_sequence, 'formal_match_precedes_freeze')
            library = _Reader(library_roots[player], 'library_'+player, hashes)
            experiment = library.json('host/experiment.json')
            journal = library.journal('host/journal')
            _require(journal and journal[0]['payload'].get('experiment_sha256') == content_sha256(experiment),
                     'frozen_library_origin_mismatch')
            contract = experiment['contract']
            _require(contract.get('condition_sha256') == freeze.condition_sha256
                and contract.get('source_manifest_sha256') == source_sha
                and contract.get('compatibility', {}).get('task_releases', {}).get(task_id) == release_sha,
                'frozen_library_condition_changed')
            model = config['models'][player]
            expected_model = contract['model']
            check_model_condition(expected_model, model, reason='frozen_player_model_changed')
            check_retry_receipts(agent, '', player, expected_model)
            _require(all(row.get('attempts') == 0 or (
                row.get('model') == row.get('requested_model') == expected_model['model']
                and row.get('reasoning_effort') == row.get('requested_reasoning_effort') == expected_model['reasoning_effort']
                and row.get('protocol') == expected_model['protocol']) for row in report['model_requests'][player]),
                'frozen_model_response_identity_changed')
            pin = config.get('skill_bindings', {}).get(player, {})
            _require(pin.get('phase') == 'evaluation' and pin.get('candidate_event_id') == freeze.candidate_event_id
                and pin.get('bundle_sha256') == freeze.bundle_hash and pin.get('skill_mutation_tools_available') is False
                and pin.get('library_contract_sha256') == content_sha256(contract), 'frozen_player_skill_binding_changed')
            candidates = [e for e in journal if e['event_id'] == freeze.candidate_event_id and e['kind'] == 'candidate_committed']
            _require(len(candidates) == 1 and candidates[0]['payload']['bundle_hash'] == freeze.bundle_hash,
                     'frozen_publication_missing')
            selections = [e for e in journal if e['event_id'] == freeze.selection_event_id]
            selection_kind = 'budget_frozen' if freeze.selection_mode == 'budget_fallback' else 'candidate_selected'
            _require(len(selections) == 1 and selections[0]['kind'] == selection_kind
                and selections[0]['sequence'] > candidates[0]['sequence']
                and selections[0]['payload'].get('candidate_event_id') == freeze.candidate_event_id
                and selections[0]['payload'].get('bundle_hash') == freeze.bundle_hash,
                'frozen_selection_not_in_library')
            selection = selections[0]['payload']
            _require(not any(e['sequence'] > selections[0]['sequence'] and e['kind'] in
                             ('author_changed', 'catalog_changed', 'candidate_committed') for e in journal),
                     'author_library_changed_after_freeze')
            if freeze.selection_mode == 'budget_fallback':
                _require(selection.get('qualification') is False and not freeze.qualified_for_declared_curriculum
                    and selection.get('validation_plan_sha256') == freeze.validation_plan_sha256,
                    'fallback_freeze_invents_qualification')
            else:
                _require(selection.get('validation_event_ids') == freeze.validation_event_ids
                    and selection.get('gate', {}).get('assessment_sha256') == freeze.validation_assessment_sha256,
                    'frozen_validation_selection_changed')
            verifier = object.__new__(SkillLibrary)
            verifier.bundles = checked_directory(library.root/'bundles')
            manifest = verifier.verify(freeze.bundle_hash)
            source_hashes = {row['path']: row['sha256'] for row in manifest['files']}
            skill_sources = {row['skill_id']: source_hashes[row['module']] for row in manifest['catalog']['skills']}
            actions = _actions(agent, episode_id, relative=player+'/host/action_journal', player=player)
            _require(actions['pending'] is None and actions['binding']['contract']['spec_sha256'] == content_sha256(spec),
                     'frozen_action_stream_unresolved')
            _require(actions['binding']['contract']['player'] == next(v for v in bindings if v['player_id'] == player),
                     'frozen_action_player_partition_changed')
            binding = {'skill_runtime_journal': player+'/host/skill_runtime_journal'}
            runtime_events = agent.journal(binding['skill_runtime_journal'])
            _require(runtime_events[0]['payload'].get('library_experiment_sha256') == content_sha256(experiment),
                     'frozen_runtime_library_changed')
            _require(all(e['payload'].get('source_sha256') == skill_sources.get(e['payload'].get('skill_id'))
                         and e['payload'].get('skill_id') in skill_sources
                         for e in runtime_events if e['kind'] == 'skill_execution_started'),
                     'frozen_execution_source_differs_from_bundle')
            invocations, entered, bundles, temporary = _skills(agent, binding, phase='heldout', episode_id=episode_id,
                candidate_event_id=freeze.candidate_event_id, bundle_hash=freeze.bundle_hash, player=player)
            _require(not temporary, 'frozen_strategy_modified')
            outcome = 'draw' if state['draw'] else ('win' if state['winner'] == player else 'loss') if state['winner'] else 'incomplete'
            receipts[player] = FrozenMatchReceipt(receipt_sequence=receipt_sequence, episode_id=episode_id,
                parent_task_id=task_id, parent_release_sha256=release_sha, bundle_hash=freeze.bundle_hash,
                condition_sha256=freeze.condition_sha256, source_manifest_sha256=source_sha,
                result_sha256=content_sha256(result), outcome=outcome, physics_valid=True,
                evidence_source='native_solver_substeps', native_substeps=count, native_evidence_sha256='0'*64,
                skill_invocations=invocations, skill_language_entries=entered, executed_bundle_hashes=bundles,
                temporary_skill_changes=False, fresh_context=True, author_history_visible=False)
        native_hash = content_sha256({k: v for k, v in hashes.items() if k.startswith(('episode/', 'source/', 'release/'))})
        for receipt in receipts.values():
            receipt.native_evidence_sha256 = native_hash
        diagnostic.update(verified=True, reason='native_frozen_match_artifacts_verified', artifact_sha256=hashes,
            hash_scope='posthoc_read_of_complete_closed_artifacts_not_original_write_time',
            runtime_source_manifest_sha256=source_sha,
            auditor_module_sha256=hashlib.sha256(read_bytes(Path(__file__))).hexdigest(),
            native_frames=count, raw_contact_rows=contacts, commits=len(commits),
            non_pass_commits_by_player={p: sum(e['player_id'] == p and e['details']['operation'] != 'pass' for e in operations) for p in players},
            match_effort=accounting.model_dump(mode='json') | {'wall_s': None, 'cpu_s': None},
            match_effort_clock_scope='Shared native episode clock counted once; host wall/CPU unavailable here, consult scheduler accounting.',
            scope='Original native referee outcome, complete frame/contact continuity, public events and unchanged skill pins; not independent full rule replay or force calibration.')
        return receipts
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        diagnostic.update(verified=False, reason=str(error) if isinstance(error, ReceiptError) else type(error).__name__,
                          artifact_sha256=hashes)
        return None


def main(argv=None):
    """Host-only offline CLI. Paths in the request resolve against its directory.

    Request keys: schema=card_rsi_frozen_match_audit_request.v1, freezes
    (player -> freeze JSON path), library_roots (player -> library directory),
    agent_output, episode_root, receipt_sequence. No files are overwritten.
    Execute from a complete frozen audit snapshot; the recorded runtime source
    may be an earlier snapshot and is verified separately.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    request_path = args.request.absolute()
    raw = read_bytes(request_path)
    request = strict_json_loads(raw)
    _require(isinstance(request, dict) and set(request) == {
        'schema', 'freezes', 'library_roots', 'agent_output', 'episode_root', 'receipt_sequence'}
        and request['schema'] == 'card_rsi_frozen_match_audit_request.v1', 'invalid_match_audit_request')
    _require(type(request['receipt_sequence']) is int and request['receipt_sequence'] >= 0,
             'invalid_match_receipt_sequence')
    def path(value):
        _require(isinstance(value, str) and bool(value), 'invalid_match_audit_path')
        value = Path(value)
        return value if value.is_absolute() else request_path.parent/value
    _require(isinstance(request['freezes'], dict) and isinstance(request['library_roots'], dict),
             'invalid_match_player_paths')
    freeze_raw = {player: read_bytes(path(value)) for player, value in request['freezes'].items()}
    freezes = {player: FreezeManifest.model_validate(strict_json_loads(value)) for player, value in freeze_raw.items()}
    diagnostic = {}
    receipts = collect_frozen_match_receipts(freezes,
        library_roots={p: path(value) for p, value in request['library_roots'].items()},
        agent_output=path(request['agent_output']), episode_root=path(request['episode_root']),
        receipt_sequence=request['receipt_sequence'], diagnostics=diagnostic)
    from scripts.maintenance.freeze_project_sources import verify_runtime_imports
    root = Path(__file__).absolute().parents[3]
    auditor_source = verify_runtime_imports(root,
        required_modules=('agent.model_client.learning.match_receipts', 'agent.model_client.learning.receipts'),
        entrypoint_file=Path(__file__))
    auditor_source_sha = hashlib.sha256(read_bytes(root/'source_snapshot_manifest.json')).hexdigest()
    output = dict(schema='card_rsi_frozen_match_audit.v1', verified=receipts is not None,
        request_sha256=hashlib.sha256(raw).hexdigest(),
        freeze_files_sha256={p: hashlib.sha256(value).hexdigest() for p, value in freeze_raw.items()},
        auditor_source_manifest_sha256=auditor_source_sha, auditor_source_verification=auditor_source,
        receipts={p: value.model_dump(mode='json') for p, value in receipts.items()} if receipts else None,
        diagnostics=diagnostic)
    checked_directory(args.output.absolute().parent, create=True)
    atomic_write_json(args.output.absolute(), output, replace=False)
    print('verified' if receipts is not None else 'unverified: '+diagnostic['reason'])
    return 0 if receipts is not None else 1


if __name__ == '__main__':
    raise SystemExit(main())
