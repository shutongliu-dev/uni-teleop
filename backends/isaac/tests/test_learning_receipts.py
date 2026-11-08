"""CPU evidence-parser fixtures only; no native or model success is claimed.

The source inventory/release builders are stubbed, while their identities,
individual file hashes, journal chains and every frame are still checked. The
real frozen-source/release checkers have their own independent test suites.
"""
from copy import deepcopy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.model_client.learning.assessment import (
    Coverage, EvaluationPlan, ValidationSlot, assess_validation, content_sha256,
)
from agent.model_client.learning.journal import Journal, canonical_json
from agent.model_client.learning.receipts import (
    ReceiptError, collect_effort, collect_match_effort, collect_revision_receipt, collect_trial_record, public_practice_feedback,
)
from agent.model_client.learning.robot_api import RobotAPI
from bench.paths import load_json


def h(value):
    return content_sha256(value)


def scene_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update(path, **values):
    data = load_json(path)
    data.update(values)
    return save(path, data)


def rewrite_journal(path, change):
    events = Journal(path).read()
    for p in path.glob('*.json'):
        p.unlink()
    journal = Journal(path)
    for event in events:
        kind, payload = change(event['kind'], deepcopy(event['payload']))
        journal.append(kind, payload)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    agent, episode, snapshot, release = (tmp_path/n for n in ('agent', 'ep1', 'snapshot', 'release'))
    for p in (agent, episode, snapshot, release):
        p.mkdir()
    modules = ['agent.model_client.function_runner', 'bench.runtime.worker',
        'scenes.workrooms.card_room.practice.runtime', 'scenes.workrooms.card_room.practice.referee',
        'scenes.workrooms.card_room.practice.cases', 'scenes.board_games.expansion_native_observer',
        'scenes.workrooms.card_room.expansion_referee']
    imported = {}
    for name in modules:
        path = snapshot/(name.replace('.', '/')+'.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# CPU receipt parser fixture; no native execution\n')
        imported[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    source_manifest = {'files_sha256': {str(Path(v['path']).relative_to(snapshot)): v['sha256'] for v in imported.values()}}
    source_sha = save(snapshot/'source_snapshot_manifest.json', source_manifest)
    source = dict(verified=True, runtime_imports_verified=True, file_modes_verified=True,
        snapshot_root=str(snapshot), imported_project_modules=imported, project_package_paths={}, entrypoint=None)
    def verify_source(path):
        assert Path(path) == snapshot
        for relative, digest in source_manifest['files_sha256'].items():
            if hashlib.sha256((snapshot/relative).read_bytes()).hexdigest() != digest:
                raise ValueError('source changed')
        return {'verified': True}
    monkeypatch.setattr('scripts.maintenance.freeze_project_sources.verify', verify_source)
    save(agent/'source_verification.json', source)
    save(episode/'private/runtime_imports.json', source)
    final_imports_sha = save(episode/'private/task_runtime/runtime_imports_final.json', source)
    config = dict(task_id='polyomino_territory_practice', source_verification_sha256=h(source))
    save(agent/'agent_config.json', config)
    release_sha = save(release/'release_manifest.json', {'schema': 'card_rsi_practice_release.v1'})
    public = dict(schema='card_rsi_practice_case.v1', gym_id=config['task_id'], parent_task_id='polyomino_territory',
        active_side='left', stratum='mono_translate', case_id='mono_translate_04', split='validation',
        parent_release_sha256=h('parent'), public_rules_sha256=h('rules'),
        goal={'stages': [{'operation': 'place'}], 'description': '真实目标，与学习日志使用不同的既有哈希格式。'})
    public['case_sha256'] = scene_hash(public)
    reset = {'black_mono': {'position_m': [.1, .2, .3]}}
    fingerprint = scene_hash(dict(parent_release_sha256=h('parent'), reset=reset, active_side='left', goal=public['goal'], private_pairs=None))
    case = dict(schema='card_rsi_private_case.v1', public=public, reset=reset, case_fingerprint=fingerprint)
    case['private_case_sha256'] = scene_hash(case)
    task = dict(id=config['task_id'], practice_case=case, asset_metadata={}, sources_sha256={'referee': h('source_referee')},
        public_rules={'parent_game_rules': {'thresholds': {'dwell_s': .3, 'maximum_drift_m': .003}}})
    save(episode/'private/task.json', task)
    monkeypatch.setattr('bench.isaacsim.release.task_inputs', lambda *a, **kw: ({}, {}, deepcopy(task), {}))
    slot = ValidationSlot(slot_id='left_mono', active_side='left', stratum='mono_translate', case_id=public['case_id'],
        case_sha256=public['case_sha256'], case_fingerprint=fingerprint, private_case_sha256=case['private_case_sha256'])
    plan = EvaluationPlan(plan_id='fixture', gym_id=config['task_id'], parent_task_id='polyomino_territory',
        parent_release_sha256=h('parent'), public_rules_sha256=h('rules'), condition_sha256=h('condition'),
        source_manifest_sha256=source_sha, scope='integration_smoke', slots=[slot],
        required_coverage=[Coverage(active_side='left', stratum='mono_translate')])
    spec = dict(episode_id='ep1', task_id=config['task_id'], agent_config_sha256=h(config), task_release_sha256=release_sha)
    save(episode/'agent/spec.json', spec)
    save(agent/'episode_ready.json', dict(spec=spec))
    save(agent/'create_request.json', dict(task_id=config['task_id']))
    save(episode/'private/worker_config.json', dict(output=str(episode/'private'), task_id=config['task_id'],
        release=str(release), seed=0, practice=dict(active_side='left', case_id=public['case_id'], split='validation')))
    score = dict(success=False, reason='interrupted', practice_progress=dict(stage_index=0, stage_count=1, confirmed_stages=0))
    evaluator_sha = save(episode/'private/evaluator_evidence.json', score)
    save(episode/'private/final_score.json', score)
    result = dict(spec=spec, outcome='cancelled', physics_valid=True, elapsed_sim_s=.001, evaluator_evidence_sha256=evaluator_sha)
    save(episode/'agent/result.json', result)
    closure = {k: True for k in ('same_reset_epoch', 'same_contact_view', 'same_contact_mapping', 'same_body_mapping',
        'same_physics_dt', 'no_unobserved_native_step', 'same_public_time', 'same_native_time', 'acquisition_ends_at_last_step')}
    observer = dict(schema='card_game_expansion_native_observer.v1', failed=False, unsupported_samples=0,
        final_context_verified=True, closure_checks=closure, reset_epoch=2, samples=2, last_native_step=4,
        physical_steps_added=0, additional_contact_acquisitions=0, physical_state_setters_called=0)
    inputs = dict(native_step=2, physics_dt_s=.0005, reset_epoch=2, physical_state_setters_called=0,
        mass_properties_verified={'black_mono': True})
    inputs_sha = save(episode/'private/task_runtime/observer_inputs.json', inputs)
    frames = []
    for i in range(2):
        frames.append(dict(frame=dict(substep_index=3+i, simulation_time_s=(i+1)*.0005, dt_s=.0005,
            contacts_complete=True, source='native_solver_substeps', objects={'black_mono': dict(position_m=[0., 0., 1.],
            quaternion_wxyz=[1., 0., 0., 0.], linear_velocity_m_s=[0., 0., 0.], angular_velocity_rad_s=[0., 0., 0.])}),
            raw_contacts=[], result=dict(committed=False, reason='target_required', practice_stage_index=0, practice_success=False)))
    frame_path = episode/'private/task_runtime/native_frames.jsonl'
    frame_path.write_text(''.join(json.dumps(row)+'\n' for row in frames))
    evidence = dict(schema='card_rsi_practice_native_evidence.v1', observer=observer, episode_id='ep1',
        task_id=config['task_id'], active_side='left', case_id=public['case_id'], split='validation',
        native_start_step=2, native_start_time_s=0., physics_dt_s=.0005,
        native_frames_sha256=hashlib.sha256(frame_path.read_bytes()).hexdigest(), observer_inputs_sha256=inputs_sha,
        runtime_imports_final_sha256=final_imports_sha, release_manifest_sha256=release_sha,
        parent_release_sha256=h('parent'), public_rules_sha256=h('rules'), case_fingerprint=fingerprint,
        evaluator_sources_sha256=h(task['sources_sha256']), score=score, confirmations=[],
        case_sha256=public['case_sha256'], private_case_sha256=case['private_case_sha256'], inactive_robot_drive_locked=True,
        object_state_writes=False, robot_target_overrides=False, formal_game_result=False)
    save(episode/'private/task_runtime/evidence.json', evidence)
    save(episode/'private/metadata.json', dict(native_isaacsim_app=True,
        native_physics_tensor_pipeline=dict(cuda_context_available=True), native_readback_configuration=dict(gpu_dynamics_enabled=True)))
    save(episode/'private/physics_audit.json', dict(schema='native_agent_physics_audit.v1', physics_valid=True))
    binding = dict(schema='card_rsi_trial_binding.v1', phase='validation', episode_id='ep1', trial_id='single',
        attempt_id='attempt0', candidate_event_id='candidate', bundle_hash=h('bundle'), condition_sha256=h('condition'),
        source_manifest_sha256=source_sha, fresh_context=True, author_history_visible=False,
        skill_runtime_journal='single/host/skill_runtime_journal', action_journal='single/host/action_journal')
    save(agent/'host/rsi_trial_binding.json', binding)

    observation = dict(episode_id='ep1', sequence=0, sim_time_s=0., instruction='fixture', profile='vision_proprio',
        proprioception=dict(joint_positions_rad={'joint': 0.}, joint_velocities_rad_s={'joint': 0.}),
        cameras=[], last_action_status='idle')
    terminal = dict(action_id='a1', status='completed', sim_time_s=.001)
    final_observation = observation | dict(sequence=1, sim_time_s=.001, last_action_id='a1', last_action_status='completed')
    bench = SimpleNamespace(base_url='http://fixture.invalid', submit=lambda *_: dict(action_id='a1', status='accepted', sim_time_s=0.),
        get_action=lambda *_: terminal, get_observation=lambda *_: final_observation)
    api = RobotAPI(bench, agent/'single/host/action_journal', episode_id='ep1', trial_id='single',
        binding=dict(interface='public_functions_v1', player=None, spec_sha256=h(spec), max_actions=10, max_calls=20))
    decision = dict(function='hold', arguments={'duration_s': .001})
    api.submit(dict(function='hold', arguments=dict(action_id='a1', observation_sequence=0, duration_s=.001),
                    action_id='a1', observation_sequence=0), observation)
    api.reconcile()
    feedback = dict(ok=True, function='hold', result=terminal, observation_sequence=1, pending_action=None)
    save(agent/'single/calls/000001.json', feedback)
    runtime = Journal(agent/'single/host/skill_runtime_journal')
    runtime.append('skill_runtime_bound', dict(protocol='agent_skill_runtime.v1', language='agent_json_skill_ast.v1',
        phase='validation', episode_id='ep1', trial_id='single', player_id='single', frozen_candidate_event_id='candidate', frozen_bundle_hash=h('bundle')))
    execution = dict(execution_id='execute1', skill_id='hold_skill', candidate_event_id='candidate',
        bundle_hash=h('bundle'), source_sha256=h('code'), args={}, state={}, limits={})
    records = agent/'single/host/skill_execution_records/execute1'
    save(records/'input.json', execution)
    runtime.append('skill_execution_started', execution)
    epoch = dict(turn_index=None, current_player=None, terminal=False)
    prepared = dict(execution_id='execute1', call_index=0, decision=decision, authorization_epoch=epoch)
    runtime.append('skill_api_prepared', prepared)
    call = prepared | dict(feedback=feedback, after_epoch=epoch)
    save(records/'call_000000.json', call)
    summary = dict(call_index=0, function='hold', is_action=True, status='completed', record_sha256=h(call),
                   ok=True, primitive_status='completed', termination_reason=None, error=None)
    runtime.append('skill_api_completed', dict(execution_id='execute1', call_index=0, record_sha256=h(call), summary=summary))
    skill_result = execution | dict(status='completed', calls=[summary], language_entered=True,
                                   physical_action_pending=False, authorization_epoch=epoch)
    save(records/'result.json', skill_result)
    runtime.append('skill_execution_finished', dict(execution_id='execute1', status='completed',
        language_entered=True, result_sha256=h(skill_result), counts={}))
    metadata = dict(attempts=1, usage={'input_tokens': 10, 'output_tokens': 20,
        'output_tokens_details': {'reasoning_tokens': 15}, 'input_tokens_details': {'cached_tokens': 0}})
    save(agent/'single/decisions/000000/model_response.json', dict(text='public fixture', metadata=metadata))
    report = dict(episode_id='ep1', result=result, ready_validated=True, run_status='completed', task_outcome='failure',
        termination='agent_finished', function_calls={'single': 1}, physical_attempts={'single': 1},
        model_requests={'single': [metadata]}, pre_episode_model_requests=[], action_recovery_pending=None)
    save(agent/'run_report.json', report)
    return SimpleNamespace(agent=agent, episode=episode, snapshot=snapshot, release=release, plan=plan, slot=slot,
        report=report, evidence=evidence, frames=frames, runtime=runtime, records=records, metadata=metadata)


def collect(f, **kwargs):
    return collect_trial_record(f.plan, f.slot, candidate_event_id='candidate', bundle_hash=h('bundle'),
        agent_output=f.agent, episode_root=f.episode, attempt_id='attempt0', attempt_index=0,
        receipt_sequence=1, wall_s=1., cpu_s=.1, **kwargs)


def test_parser_accepts_complete_fixture_graph_without_claiming_cpu_physics(fixture):
    r = collect(fixture)
    assert r.verification_issues == []
    assert r.outcome == 'failure' and r.native_result()
    assert r.skill_invocations == r.skill_language_entries == 1
    assert r.executed_bundle_hashes == [h('bundle')]
    assert r.cost.model_requests == r.cost.model_logical_calls == 1
    assert r.cost.input_tokens == 10 and r.cost.output_tokens == 20 and r.cost.reasoning_tokens == 15
    assert (r.cost.physical_attempts, r.cost.accepted_actions, r.cost.completed_actions) == (1, 1, 1)
    assert r.public_feedback_sha256 == h(public_practice_feedback(r, fixture.report))
    assessment = assess_validation(fixture.plan, [r], candidate_event_id='candidate', bundle_hash=h('bundle'),
                                   condition_sha256=fixture.plan.condition_sha256)
    assert assessment.failures == 1 and not assessment.qualified_for_declared_curriculum


@pytest.mark.parametrize('relative', ['private/task_runtime/evidence.json', 'private/task_runtime/native_frames.jsonl',
    'private/task_runtime/runtime_imports_final.json', 'private/task_runtime/observer_inputs.json', 'private/runtime_imports.json'])
def test_missing_native_artifacts_stay_unknown(fixture, relative):
    (fixture.episode/relative).unlink()
    record = collect(fixture)
    assert record.outcome == 'unknown' and record.evidence_source == 'unverified' and record.verification_issues


@pytest.mark.parametrize('field,value', [('episode_id', 'another'), ('active_side', 'right'),
    ('native_frames_sha256', h('wrong')), ('case_fingerprint', h('wrong')), ('runtime_imports_final_sha256', h('wrong')),
    ('object_state_writes', True), ('inactive_robot_drive_locked', False)])
def test_native_binding_or_integrity_substitution_is_rejected(fixture, field, value):
    update(fixture.episode/'private/task_runtime/evidence.json', **{field: value})
    record = collect(fixture)
    assert record.outcome == 'unknown' and not record.native_result()


@pytest.mark.parametrize('kind', ['synthetic', 'step_gap', 'clock_gap', 'contacts_missing', 'truncated'])
def test_raw_stream_cannot_be_relabelled_native(fixture, kind):
    rows = deepcopy(fixture.frames)
    if kind == 'synthetic': rows[0]['frame']['source'] = 'synthetic'
    if kind == 'step_gap': rows[1]['frame']['substep_index'] += 1
    if kind == 'clock_gap': rows[1]['frame']['simulation_time_s'] += .1
    if kind == 'contacts_missing': rows[0]['frame']['contacts_complete'] = False
    path = fixture.episode/'private/task_runtime/native_frames.jsonl'
    text = ''.join(json.dumps(row)+'\n' for row in rows)
    path.write_text(text[:-1] if kind == 'truncated' else text)
    update(fixture.episode/'private/task_runtime/evidence.json', native_frames_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert collect(fixture).evidence_source == 'unverified'


def test_closure_source_and_result_checks_are_independent(fixture):
    evidence = fixture.evidence
    evidence['observer']['closure_checks']['no_unobserved_native_step'] = False
    save(fixture.episode/'private/task_runtime/evidence.json', evidence)
    assert collect(fixture).evidence_source == 'unverified'
    evidence['observer']['closure_checks']['no_unobserved_native_step'] = True
    save(fixture.episode/'private/task_runtime/evidence.json', evidence)
    module = fixture.snapshot/'bench/runtime/worker.py'
    module.write_text('# altered\n')
    assert collect(fixture).evidence_source == 'unverified'


def test_agent_and_episode_result_must_be_identical(fixture):
    update(fixture.episode/'agent/result.json', outcome='success')
    assert collect(fixture).outcome == 'unknown'


@pytest.mark.parametrize('relative', ['single/host/skill_execution_records/execute1/input.json',
    'single/host/skill_execution_records/execute1/call_000000.json', 'single/host/skill_execution_records/execute1/result.json'])
def test_skill_record_tampering_is_not_independent_execution(fixture, relative):
    update(fixture.agent/relative, unexpected='changed')
    record = collect(fixture)
    assert record.outcome == 'unknown' and not record.immutable_skill_execution()
    assert record.skill_invocations == 1 and record.skill_language_entries == 0


def test_call_ok_is_distinct_from_primitive_completion(fixture):
    feedback_path = fixture.agent/'single/calls/000001.json'
    feedback = load_json(feedback_path)
    feedback['result']['status'] = 'failed'
    feedback['result']['termination_reason'] = 'begin_rejected'
    save(feedback_path, feedback)
    # Independent action journal remains completed: mismatch must not acquire
    # a false skill success simply because top-level ok is still true.
    assert collect(fixture).evidence_source == 'unverified'


def test_action_ledger_tamper_is_a_hard_unknown_cost_error(fixture):
    path = next((fixture.agent/'single/host/action_journal').glob('*.json'))
    update(path, event_hash=h('tampered'))
    with pytest.raises(ReceiptError, match='checksum'):
        collect_effort(fixture.agent)


def test_function_counters_cannot_hide_physical_attempts(fixture):
    report = fixture.report | {'physical_attempts': {'single': 0}}
    save(fixture.agent/'run_report.json', report)
    with pytest.raises(ReceiptError, match='physical_attempt'):
        collect_effort(fixture.agent)


def test_provider_usage_unknown_on_retries_and_failure(fixture):
    metadata = deepcopy(fixture.metadata)
    metadata['attempts'] = 2
    save(fixture.agent/'single/decisions/000000/model_response.json', dict(text='fixture', metadata=metadata))
    save(fixture.agent/'run_report.json', fixture.report | {'model_requests': {'single': [metadata]}})
    effort = collect_effort(fixture.agent)
    assert effort.model_requests == 2 and effort.output_tokens is None and effort.input_tokens is None
    (fixture.agent/'single/decisions/000000/model_response.json').unlink()
    save(fixture.agent/'single/decisions/000000/model_error.json', dict(attempts=0, error_category='request_budget_exhausted'))
    save(fixture.agent/'run_report.json', fixture.report | {'model_requests': {'single': []}})
    effort = collect_effort(fixture.agent)
    assert effort.model_logical_calls == 1 and effort.model_requests == 0


def test_public_projection_does_not_copy_private_fields_or_error_text(fixture):
    record = collect(fixture)
    report = fixture.report | {'run_error': {'error_category': 'transport_timeout', 'message': 'secret-private-value'},
                              'private_case_sha256': 'secret-private-value'}
    text = json.dumps(public_practice_feedback(record, report))
    for forbidden in ['secret-private-value', record.case_fingerprint, record.private_case_sha256, record.source_manifest_sha256]:
        assert forbidden not in text
    assert public_practice_feedback(None, report)['outcome'] == 'unknown'


def test_no_episode_no_candidate_still_accounts_real_model_cost(tmp_path):
    agent = tmp_path/'agent'
    agent.mkdir()
    metadata = dict(attempts=1, usage={})
    save(agent/'single/decisions/000000/model_response.json', dict(metadata=metadata))
    save(agent/'run_report.json', dict(episode_id=None, model_requests={'single': [metadata]}, pre_episode_model_requests=[],
         physical_attempts={}, function_calls={}, result=None))
    effort = collect_effort(agent, wall_s=2., cpu_s=.2)
    assert effort.model_requests == 1 and effort.output_tokens is None and effort.reset_attempts == 0
    assert effort.wall_s == 2.


def test_wrong_authorization_epoch_is_rejected_even_after_rehashing_journal(fixture):
    def change(kind, payload):
        if kind == 'skill_api_prepared':
            payload['authorization_epoch']['turn_index'] = 7
        return kind, payload
    rewrite_journal(fixture.agent/'single/host/skill_runtime_journal', change)
    record = collect(fixture)
    assert record.evidence_source == 'unverified'
    assert any('authorization_epoch' in issue for issue in record.verification_issues)


def test_skill_summary_must_preserve_actual_primitive_status(fixture):
    def change(kind, payload):
        if kind == 'skill_api_completed':
            payload['summary']['primitive_status'] = 'failed'
        return kind, payload
    rewrite_journal(fixture.agent/'single/host/skill_runtime_journal', change)
    assert any('summary_mismatch' in issue for issue in collect(fixture).verification_issues)


def test_wrong_action_episode_is_not_silently_rebound(fixture):
    def change(kind, payload):
        if kind == 'action_ledger_bound':
            payload['episode_id'] = 'ep2'
        return kind, payload
    rewrite_journal(fixture.agent/'single/host/action_journal', change)
    with pytest.raises(ReceiptError, match='binding_mismatch'):
        collect_effort(fixture.agent)


def test_primitive_budget_rejection_is_still_a_counted_attempt(fixture):
    def change(kind, payload):
        if kind == 'action_ledger_bound':
            payload['contract']['max_actions'] = 1
        return kind, payload
    rewrite_journal(fixture.agent/'single/host/action_journal', change)
    save(fixture.agent/'single/calls/000002.json', dict(ok=False, function='hold',
        error=dict(code='budget_exhausted', message='Client action-attempt budget exhausted')))
    save(fixture.agent/'run_report.json', fixture.report | {'function_calls': {'single': 2}})
    effort = collect_effort(fixture.agent)
    assert (effort.physical_attempts, effort.accepted_actions, effort.completed_actions) == (2, 1, 1)


def test_invocation_blocked_before_language_entry_is_not_skill_execution(fixture):
    journal_path = fixture.agent/'single/host/skill_runtime_journal'
    events = Journal(journal_path).read()
    for p in journal_path.glob('*.json'):
        p.unlink()
    journal = Journal(journal_path)
    for event in events[:2]:
        journal.append(event['kind'], event['payload'])
    result = load_json(fixture.records/'result.json')
    result.update(status='stopped', stop_reason='episode_terminal', calls=[], language_entered=False)
    save(fixture.records/'result.json', result)
    journal.append('skill_execution_finished', dict(execution_id='execute1', status='stopped',
        language_entered=False, result_sha256=h(result), counts={}))
    record = collect(fixture)
    assert record.native_result() and record.skill_invocations == 1 and record.skill_language_entries == 0
    assert not record.immutable_skill_execution()


def test_practice_keeps_actual_older_bundle_when_passed_last_candidate_differs(fixture):
    update(fixture.agent/'host/rsi_trial_binding.json', phase='practice', candidate_event_id=None, bundle_hash=None)
    def change(kind, payload):
        if kind == 'skill_runtime_bound':
            payload.update(phase='practice', frozen_candidate_event_id=None, frozen_bundle_hash=None)
        return kind, payload
    rewrite_journal(fixture.agent/'single/host/skill_runtime_journal', change)
    record = collect_trial_record(fixture.plan, fixture.slot, candidate_event_id='last_candidate', bundle_hash=h('latest'),
        agent_output=fixture.agent, episode_root=fixture.episode, attempt_id='attempt0', attempt_index=0,
        receipt_sequence=1, phase='practice', wall_s=1., cpu_s=.1)
    assert record.executed_bundle_hashes == [h('bundle')] and record.bundle_hash == h('latest')
    assert not record.immutable_skill_execution()


def test_collected_failure_is_not_replaced_by_successful_same_slot_retry(fixture):
    failed = collect(fixture)
    # Assessment reduction is exercised on the immutable first collector
    # receipt; a later fixture success cannot erase that failure.
    retry = failed.model_copy(update=dict(attempt_id='retry', attempt_index=1, receipt_sequence=2, outcome='success'))
    result = assess_validation(fixture.plan, [failed, retry], candidate_event_id='candidate', bundle_hash=h('bundle'),
        condition_sha256=fixture.plan.condition_sha256)
    assert result.failures == 1 and result.successes == 0 and result.cost.attempts == 2


def match_fixture(f):
    for player in ('player_0', 'player_1'):
        shutil.copytree(f.agent/'single', f.agent/player)
        def change(kind, payload):
            if kind == 'action_ledger_bound':
                payload['trial_id'] = player
                payload['contract']['player'] = {'player_id': player}
            return kind, payload
        rewrite_journal(f.agent/player/'host/action_journal', change)
    report = f.report | dict(model_requests={p: [f.metadata] for p in ('player_0', 'player_1')},
        function_calls={'player_0': 1, 'player_1': 1}, physical_attempts={'player_0': 1, 'player_1': 1},
        result={'episode': f.report['result']})
    save(f.agent/'run_report.json', report)
    return report


def test_match_cost_adds_both_streams_but_only_one_episode_clock(fixture):
    match_fixture(fixture)
    effort = collect_match_effort(fixture.agent, wall_s=3., cpu_s=.2)
    assert effort.model_requests == effort.model_logical_calls == 2
    assert effort.input_tokens == 20 and effort.output_tokens == 40 and effort.reasoning_tokens == 30
    assert (effort.physical_attempts, effort.accepted_actions, effort.completed_actions) == (2, 2, 2)
    assert effort.native_sim_s == .001 and effort.reset_attempts == 1
    assert effort.wall_s == 3. and effort.cpu_s == .2


def test_match_stream_cannot_reuse_other_players_binding(fixture):
    match_fixture(fixture)
    def change(kind, payload):
        if kind == 'action_ledger_bound': payload['contract']['player']['player_id'] = 'player_0'
        return kind, payload
    rewrite_journal(fixture.agent/'player_1/host/action_journal', change)
    with pytest.raises(ReceiptError, match='player_binding'):
        collect_match_effort(fixture.agent)


def test_match_unknown_usage_and_pre_episode_request_are_not_double_counted(fixture):
    report = match_fixture(fixture)
    pre = dict(attempts=1, usage={'input_tokens': 1, 'output_tokens': 2})
    report['pre_episode_model_requests'] = [pre]
    save(fixture.agent/'run_report.json', report)
    effort = collect_match_effort(fixture.agent)
    assert effort.model_requests == 3 and effort.input_tokens == 21 and effort.output_tokens == 42
    assert effort.reasoning_tokens is None


def test_cancel_and_intent_are_not_reported_as_read_only(fixture):
    save(fixture.agent/'single/calls/000002.json', {'function': 'cancel_action', 'ok': False})
    save(fixture.agent/'single/calls/000003.json', {'function': 'intent', 'ok': False})
    save(fixture.agent/'run_report.json', fixture.report | {'function_calls': {'single': 3}})
    assert collect_effort(fixture.agent).read_only_calls == 0


def revision_fixture(f, *, comment_only=False):
    from agent.model_client.learning.skills import SkillLibrary
    # Native parser is tested separately above. This is a trusted-host prior
    # receipt stub, not a claim that this CPU fixture executed either program.
    native = collect(f)
    library = SkillLibrary.create('learner', learning_root=f.agent.parent/'libraries', contract={'dependencies': {}})
    old_source = 'def run(api, args, state):\n    return 0\n'
    new_source = ('# comment only\n'+old_source if comment_only else 'def run(api, args, state):\n    return 1\n')
    catalog = {'skills': [dict(skill_id='plan', module='skills/plan.py', entrypoint='run', description='model plan',
        tags=[], input_schema={}, output_schema={})]}
    revision = library.write_author('skills/plan.py', old_source, expected_revision=library.author_revision())
    revision = library.write_author('catalog.json', canonical_json(catalog), expected_revision=revision)
    old = library.publish(revision, transaction_id='old', reason='first model program')
    prior = native.model_copy(update=dict(phase='practice', candidate_event_id=old['event_id'],
        bundle_hash=old['payload']['bundle_hash'], executed_bundle_hashes=[old['payload']['bundle_hash']], receipt_sequence=1))
    feedback = public_practice_feedback(prior, f.report)
    prior.public_feedback_sha256 = h(feedback)
    expected = library.author_revision()
    revision = library.write_author('skills/plan.py', new_source, expected_revision=expected)
    revision = library.write_author('catalog.json', canonical_json(catalog), expected_revision=revision)
    new = library.publish(revision, transaction_id='new', reason='revise after observed failure')
    config = load_json(f.agent/'agent_config.json')
    config['models'] = {'single': {'model': 'model-fixture'}}
    save(f.agent/'agent_config.json', config)
    update(f.agent/'host/rsi_trial_binding.json', phase='practice')
    journal_path = f.agent/'single/host/skill_runtime_journal'
    for path in journal_path.glob('*.json'):
        path.unlink()
    journal = Journal(journal_path)
    journal.append('skill_runtime_bound', dict(protocol='agent_skill_runtime.v1', phase='practice', episode_id='ep1',
        library_experiment_sha256=h(library.experiment)))
    write_decision = dict(function='skill_write', arguments=dict(skill_id='plan', source=new_source,
        description='model plan', expected_revision=expected))
    write_feedback = dict(ok=True, function='skill_write', result=dict(skill_id='plan', author_revision=revision,
        source_sha256=hashlib.sha256(new_source.encode()).hexdigest(), published=False))
    publish_decision = dict(function='skill_publish', arguments=dict(expected_revision=revision, transaction_id='new',
        reason='revise after observed failure'))
    publish_feedback = dict(ok=True, function='skill_publish', result=dict(candidate_event_id=new['event_id'],
        bundle_hash=new['payload']['bundle_hash'], author_revision=revision, validated=False, selected=False))
    metadata = f.metadata | dict(model='model-fixture', requested_model='model-fixture')
    for index, (decision, execution) in enumerate(((write_decision, write_feedback), (publish_decision, publish_feedback))):
        relative = f'single/decisions/{index:06d}'
        request = [{'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
            'prior_practice_feedback': [{'assessment': feedback, 'recent_public_executions': [{'function': 'hold', 'ok': True}]}]})}]}]
        save(f.agent/relative/'model_request.json', dict(system_prompt='fixture', messages=request, expanded_messages_sha256=h(request)))
        save(f.agent/relative/'model_response.json', dict(text=json.dumps(decision), metadata=metadata))
        save(f.agent/relative/'execution.json', execution)
        journal.append('skill_tool_requested', decision)
        journal.append('skill_tool_finished', dict(function=decision['function'], feedback_sha256=h(execution), ok=True, counts={}))
    save(f.agent/'run_report.json', f.report | {'model_requests': {'single': [metadata, metadata]}})
    return dict(library_root=library.path, previous_candidate_event_id=old['event_id'], candidate_event_id=new['event_id'],
        agent_output=f.agent, practice_records=[prior], receipt_sequence=5)


def test_revision_requires_model_code_and_actual_native_feedback_in_request(fixture):
    args = revision_fixture(fixture)
    diagnostics = {}
    revision = collect_revision_receipt(**args, diagnostics=diagnostics)
    assert revision is not None and diagnostics['verified']
    assert revision.actual_model_requests == 2 and revision.changed_executable_content
    assert revision.practice_attempt_ids == ['attempt0']
    assert revision.public_feedback_sha256 == [args['practice_records'][0].public_feedback_sha256]


def test_comment_or_notes_changes_do_not_establish_executable_revision(fixture):
    args = revision_fixture(fixture, comment_only=True)
    diagnostics = {}
    assert collect_revision_receipt(**args, diagnostics=diagnostics) is None
    assert diagnostics['reason'] == 'no_executable_source_change'


def test_feedback_mentioned_only_by_assistant_does_not_count_as_delivered(fixture):
    args = revision_fixture(fixture)
    for folder in (fixture.agent/'single/decisions').iterdir():
        path = folder/'model_request.json'
        value = load_json(path)
        value['messages'][0]['role'] = 'assistant'
        value['expanded_messages_sha256'] = h(value['messages'])
        save(path, value)
    diagnostic = {}
    assert collect_revision_receipt(**args, diagnostics=diagnostic) is None
    assert diagnostic['reason'] == 'native_feedback_not_in_actual_revision_request'


def test_fake_model_requests_do_not_establish_revision(fixture):
    args = revision_fixture(fixture)
    report = load_json(fixture.agent/'run_report.json')
    for index, row in enumerate(report['model_requests']['single']):
        row['attempts'] = 0
        path = fixture.agent/f'single/decisions/{index:06d}/model_response.json'
        value = load_json(path)
        value['metadata'] = row
        save(path, value)
    save(fixture.agent/'run_report.json', report)
    diagnostic = {}
    assert collect_revision_receipt(**args, diagnostics=diagnostic) is None
    assert diagnostic['reason'] == 'no_actual_model_request'


def test_revision_request_hash_and_journal_are_both_required(fixture):
    args = revision_fixture(fixture)
    update(fixture.agent/'single/decisions/000000/model_request.json', expanded_messages_sha256=h('wrong'))
    diagnostic = {}
    assert collect_revision_receipt(**args, diagnostics=diagnostic) is None
    assert diagnostic['reason'] == 'expanded_model_request_hash_mismatch'


def test_unverified_or_mixed_prior_practice_cannot_establish_rsi(fixture):
    args = revision_fixture(fixture)
    args['practice_records'][0].evidence_source = 'synthetic'
    diagnostic = {}
    assert collect_revision_receipt(**args, diagnostics=diagnostic) is None
    assert diagnostic['reason'] == 'no_unambiguous_prior_native_practice'


def formal_fixture(f, monkeypatch):
    """Native-shaped parser fixture, not a physical gameplay simulation."""
    from agent.model_client.learning.assessment import FreezeManifest, sum_effort
    from agent.model_client.learning.skills import SkillLibrary
    from scenes.board_games import expansion_native_observer as aggregation
    report = match_fixture(f)
    source = load_json(f.agent/'source_verification.json')
    for name in ('bench.runtime.card_game_worker', 'scenes.workrooms.card_room.expansion_rules'):
        path = f.snapshot/(name.replace('.', '/')+'.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# CPU parser fixture\n')
        source['imported_project_modules'][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = {'files_sha256': {str(Path(v['path']).relative_to(f.snapshot)): v['sha256']
        for v in source['imported_project_modules'].values()}}
    source_sha = save(f.snapshot/'source_snapshot_manifest.json', manifest)
    def verify_source(path):
        assert Path(path) == f.snapshot
        assert all(hashlib.sha256((f.snapshot/k).read_bytes()).hexdigest() == v for k,v in manifest['files_sha256'].items())
        return {'verified': True}
    monkeypatch.setattr('scripts.maintenance.freeze_project_sources.verify', verify_source)
    save(f.agent/'source_verification.json', source)
    save(f.episode/'private/runtime_imports.json', source)
    save(f.episode/'private/runtime_imports_final.json', source)
    task_id = 'polyomino_territory'
    release_sha = hashlib.sha256((f.release/'release_manifest.json').read_bytes()).hexdigest()
    model = dict(provider='gateway', model='fixture-model', reasoning_effort='max',
        protocol='chat/completions', base_url='https://fixture.invalid/v1', endpoint='https://fixture.invalid/v1/chat/completions')
    config = dict(task_id=task_id, source_verification_sha256=h(source), models={}, skill_bindings={})
    freezes, libraries, published, codes = {}, {}, {}, {}
    for player in ('player_0', 'player_1'):
        contract = dict(condition_sha256=h('condition'), source_manifest_sha256=source_sha, model=model,
            dependencies={}, compatibility={'task_ids': [task_id], 'task_releases': {task_id: release_sha}})
        library = SkillLibrary.create(player, learning_root=f.agent.parent/'libraries', contract=contract)
        code = 'def run(api, args, state):\n    return api.call("hold", {"duration_s": 0.001})\n'
        revision = library.write_author('skills/hold_skill.py', code, expected_revision=library.author_revision())
        catalog = {'skills': [dict(skill_id='hold_skill', module='skills/hold_skill.py', entrypoint='run',
            description='CPU fixture', tags=[], input_schema={}, output_schema={})]}
        revision = library.write_author('catalog.json', canonical_json(catalog), expected_revision=revision)
        candidate = library.publish(revision, transaction_id='published', reason='fixture')
        bundle = candidate['payload']['bundle_hash']
        selection = library.journal.append('budget_frozen', dict(candidate_event_id=candidate['event_id'], bundle_hash=bundle,
            qualification=False, validation_plan_sha256=h('plan'), budget_fallback='last_published'))
        freezes[player] = FreezeManifest(learner_id=player, learning_run_id='fixture', gym_id=task_id+'_practice',
            parent_task_id=task_id, parent_release_sha256=release_sha, public_rules_sha256=h('rules'),
            condition_sha256=h('condition'), source_manifest_sha256=source_sha,
            candidate_event_id=candidate['event_id'], bundle_hash=bundle, selection_event_id=selection['event_id'],
            validation_plan_sha256=h('plan'), selection_mode='budget_fallback', validation_assessment_sha256=None,
            validation_event_ids=[], validation_complete=False, immutable_validation_complete=False,
            qualification_reasons=['declared_budget_fallback', 'no_validation_assessment'],
            qualified_for_declared_curriculum=False, freeze_sequence=1, development_cost=sum_effort([]), development_ledger_sha256=h('ledger'))
        libraries[player], published[player], codes[player] = library, candidate, hashlib.sha256(code.encode()).hexdigest()
        # Real gateway metadata intentionally omits provider.
        config['models'][player] = {k:v for k,v in model.items() if k != 'provider'}
        config['skill_bindings'][player] = dict(phase='evaluation', candidate_event_id=candidate['event_id'],
            bundle_sha256=bundle, skill_mutation_tools_available=False, library_contract_sha256=h(contract))
    save(f.agent/'agent_config.json', config)
    spec = load_json(f.episode/'agent/spec.json') | dict(task_id=task_id, agent_config_sha256=h(config))
    save(f.episode/'agent/spec.json', spec)
    players = [dict(player_id=p, color=c, hand=s, robot_id=s, joint_names=[s+'_joint'])
        for p,c,s in [('player_0','black','left'),('player_1','white','right')]]
    save(f.agent/'episode_ready.json', dict(spec=spec, players=players))
    physical = f.report['result'] | dict(spec=spec)
    save(f.episode/'agent/result.json', physical)
    state = dict(protocol='bench_card_game_state_v1', task_id=task_id, turn_index=0, round_index=0,
        current_player=None, terminal=True, winner=None, draw=False, pending_move=None,
        status=dict(outcome='aborted',end_reason='cancelled',recovery_required=False,evidence_unavailable=False,reason='cancelled',retryable=False),
        settlement_dwell_s=0., commit_count=0, phase='terminal', event_cursor=-1)
    score=dict(state,schema='card_game_expansion_worker_score.v1',success=False,outcome='aborted',reason='cancelled',
        end_reason='cancelled',native_observer_samples=2,native_observer_unsupported_samples=0)
    physical['evaluator_evidence_sha256']=save(f.episode/'private/evaluator_evidence.json',score)
    save(f.episode/'private/final_score.json',score)
    save(f.episode/'agent/result.json',physical)
    result = dict(board_game=state, episode=physical)
    save(f.episode/'agent/board_game_result.json', result)
    report['result'] = result
    metadata = f.metadata | dict(model=model['model'], requested_model=model['model'],
        reasoning_effort='max', requested_reasoning_effort='max', protocol=model['protocol'])
    report['model_requests'] = {p: [metadata] for p in libraries}
    save(f.agent/'run_report.json', report)
    for player, library in libraries.items():
        candidate = published[player]
        save(f.agent/player/'decisions/000000/model_response.json', dict(text='CPU parser fixture', metadata=metadata))
        save(f.agent/player/'public_game_events.json', dict(protocol=state['protocol'], cursor=-1, events=[]))
        def change_action(kind, payload):
            if kind == 'action_ledger_bound':
                payload['contract']['spec_sha256'] = h(spec)
                payload['contract']['player'] = next(p for p in players if p['player_id'] == player)
            return kind,payload
        rewrite_journal(f.agent/player/'host/action_journal', change_action)
        folder = f.agent/player/'host/skill_execution_records/execute1'
        epoch = dict(turn_index=0, current_player=player, terminal=False)
        new_input = load_json(folder/'input.json') | dict(candidate_event_id=candidate['event_id'],
            bundle_hash=candidate['payload']['bundle_hash'], source_sha256=codes[player])
        call = load_json(folder/'call_000000.json') | dict(authorization_epoch=epoch, after_epoch=epoch)
        skill_result = load_json(folder/'result.json') | new_input | dict(authorization_epoch=epoch)
        skill_result['calls'][0]['record_sha256'] = h(call)
        save(folder/'input.json', new_input);save(folder/'call_000000.json',call);save(folder/'result.json',skill_result)
        def change_runtime(kind,payload):
            if kind == 'skill_runtime_bound':
                payload.update(phase='evaluation',trial_id=player,player_id=player,
                    library_experiment_sha256=h(library.experiment),frozen_candidate_event_id=candidate['event_id'],frozen_bundle_hash=candidate['payload']['bundle_hash'])
            elif kind == 'skill_execution_started': payload = new_input
            elif kind == 'skill_api_prepared': payload['authorization_epoch'] = epoch
            elif kind == 'skill_api_completed':
                payload.update(record_sha256=h(call),summary=skill_result['calls'][0])
            elif kind == 'skill_execution_finished': payload['result_sha256'] = h(skill_result)
            return kind,payload
        rewrite_journal(f.agent/player/'host/skill_runtime_journal',change_runtime)
    asset = dict(actors={'black_mono':'/World/CardGame/Actors/black_mono'}, fixture_collision_paths=['/World/CardGame/Fixtures/Deck'],
        mass_properties={'black_mono':dict(mass_kg=.05,center_of_mass_m=[0.,0.,0.],inertia_tensor_kg_m2=[[1e-5,0.,0.],[0.,1e-5,0.],[0.,0.,1e-5]])})
    task = dict(task_id=task_id, asset_metadata=asset,
        sources_sha256={'scenes/board_games/expansion_native_observer.py':hashlib.sha256(Path(aggregation.__file__).read_bytes()).hexdigest()})
    save(f.episode/'private/task.json',task)
    monkeypatch.setattr('bench.isaacsim.release.task_inputs',lambda *a,**kw: ({},{},deepcopy(task),{}))
    update(f.episode/'private/worker_config.json',task_id=task_id)
    inputs = load_json(f.episode/'private/task_runtime/observer_inputs.json')
    links = dict(left_tip='/World/Robot/Left/tip',right_tip='/World/Robot/Right/tip')
    owners = aggregation.robot_ownership(links)
    inputs.update(body_paths=list(asset['actors'].values()),masses_kg=[.05],coms=[[0.,0.,0.,0.,0.,0.,1.]],
        inertias_kg_m2=[asset['mass_properties']['black_mono']['inertia_tensor_kg_m2']],expected_mass_properties=asset['mass_properties'],
        deck_contact_paths=asset['fixture_collision_paths'],robot_contact_ownership=owners,known_static_contact_paths=[])
    save(f.episode/'private/board_game/observer_inputs.json', inputs)
    observer = f.evidence['observer'] | dict(deck_contact_paths=inputs['deck_contact_paths'],robot_contact_ownership=owners)
    save(f.episode/'private/board_game/evidence.json', dict(schema='card_game_expansion_worker_evidence.v1',
        observer=observer,score=score,public_state=state,object_state_writes=False,robot_target_overrides=False))
    update(f.episode/'private/metadata.json',physics_dt_s=.0005,scene_info=dict(robot_link_paths=links,collider_actor_paths={},
        support_paths=[],fixtures={},objects={'black_mono':{'path':asset['actors']['black_mono']}}))
    frames = deepcopy(f.frames)
    for row in frames:
        row['result'] = dict(committed=False,reason='target_required')
        row['frame']['objects']['black_mono'].update(deck_support_impulse_ns=0.,robot_contact_impulses_ns={'player_0':0.,'player_1':0.},
            robot_nonhand_contact_impulses_ns={'player_0':0.,'player_1':0.},object_support_impulses_ns={},maximum_penetration_m=0.,other_static_support_impulse_ns=0.)
    (f.episode/'private/board_game/native_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in frames))
    (f.episode/'private/physics.jsonl').write_text(json.dumps({'kind':'reset','observation':{'sim_time_s':0.}})+'\n')
    return dict(freezes=freezes,library_roots={p:v.path for p,v in libraries.items()},
        agent_output=f.agent,episode_root=f.episode,receipt_sequence=10)


def test_formal_parser_accepts_complete_closed_fixture_without_inventing_game_completion(fixture,monkeypatch):
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    args = formal_fixture(fixture,monkeypatch)
    diagnostic = {}
    receipts = collect_frozen_match_receipts(**args,diagnostics=diagnostic)
    assert receipts is not None, diagnostic
    assert all(r.outcome == 'incomplete' and r.skill_language_entries == 1 for r in receipts.values())
    assert diagnostic['commits'] == 0 and diagnostic['native_frames'] == 2
    assert diagnostic['match_effort']['wall_s'] is None and diagnostic['match_effort']['cpu_s'] is None
    assert diagnostic['runtime_source_manifest_sha256'] == args['freezes']['player_0'].source_manifest_sha256


@pytest.mark.parametrize('mutation,reason',[
    ('missing_final','FileNotFoundError'),('source','source_manifest_mismatch'),
    ('ownership','frozen_action_player_partition_changed'),('routing','frozen_player_model_changed'),
    ('private_history','frozen_match_context_mismatch'),('inventory','native_inventory_or_mass_source_mismatch'),
    ('mass','native_mass_readback_differs_from_frozen_asset'),('contacts','raw_contact_projection_mismatch'),
    ('synthetic','non_native_or_incomplete_frame'),('gap','native_substep_or_clock_gap'),
    ('missing_public_event','players_missed_public_events'),('skill_pin','executed_skill_pin_mismatch'),
    ('source_code','frozen_execution_source_differs_from_bundle'),('selection','frozen_selection_not_in_library'),
])
def test_formal_parser_rejects_missing_or_mismatched_evidence(fixture,monkeypatch,mutation,reason):
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    args = formal_fixture(fixture,monkeypatch)
    episode,agent = fixture.episode,fixture.agent
    if mutation == 'missing_final': (episode/'private/runtime_imports_final.json').unlink()
    elif mutation == 'source': args['freezes']['player_0'].source_manifest_sha256=args['freezes']['player_1'].source_manifest_sha256=h('different')
    elif mutation == 'ownership':
        rewrite_journal(agent/'player_1/host/action_journal',lambda kind,payload:(kind,payload if kind!='action_ledger_bound' else payload|{'contract':payload['contract']|{'player':payload['contract']['player']|{'joint_names':['different_joint']}}}))
    elif mutation in ('routing','private_history'):
        config=load_json(agent/'agent_config.json')
        if mutation=='routing': config['models']['player_1']['endpoint']='https://different.invalid'
        else: config['prior_practice_feedback_sha256']=h('history')
        # Config must retain its original signed spec hash; deliberately do not
        # reseal unrelated artifacts for the history/config test.
        save(agent/'agent_config.json',config)
        if mutation=='routing': reason='frozen_match_context_mismatch'
    elif mutation in ('inventory','mass'):
        path=episode/'private/board_game/observer_inputs.json';value=load_json(path)
        if mutation=='inventory': value['body_paths']=[]
        else:value['masses_kg']=[.06]
        save(path,value)
    elif mutation in ('contacts','synthetic','gap'):
        path=episode/'private/board_game/native_frames.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        if mutation=='contacts':rows[0]['frame']['objects']['black_mono']['deck_support_impulse_ns']=1.
        elif mutation=='synthetic':rows[0]['frame']['source']='synthetic'
        else:rows[1]['frame']['substep_index']+=1
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif mutation=='missing_public_event':update(agent/'player_1/public_game_events.json',cursor=0)
    elif mutation in ('skill_pin','source_code'):
        player='player_0';folder=agent/player/'host/skill_execution_records/execute1'
        key='bundle_hash' if mutation=='skill_pin' else 'source_sha256'
        value=load_json(folder/'input.json')|{key:h('changed')};save(folder/'input.json',value)
        rewrite_journal(agent/player/'host/skill_runtime_journal',lambda kind,payload:(kind,value if kind=='skill_execution_started' else payload))
    elif mutation=='selection':args['freezes']['player_0'].selection_event_id='not-a-selection'
    diagnostic={}
    assert collect_frozen_match_receipts(**args,diagnostics=diagnostic) is None
    assert diagnostic['reason']==reason


def reseal_formal_result(f, *, state=None, config=None, physical=None):
    """Keep fixture cross-file identities coherent when testing deeper checks."""
    report=load_json(f.agent/'run_report.json')
    result=report['result']
    if config is not None:
        save(f.agent/'agent_config.json',config)
        result['episode']['spec']['agent_config_sha256']=h(config)
    if physical is not None: result['episode'].update(physical)
    if state is not None:
        result['board_game']=state
        score=dict(state,schema='card_game_expansion_worker_score.v1',success=state['status']['outcome'] in ('win','draw'),
            outcome=state['status']['outcome'],reason=state['status']['reason'],end_reason=state['status']['end_reason'],
            native_observer_samples=2,native_observer_unsupported_samples=0)
        update(f.episode/'private/board_game/evidence.json',public_state=state,score=score)
        save(f.episode/'private/final_score.json',score)
        result['episode']['evaluator_evidence_sha256']=save(f.episode/'private/evaluator_evidence.json',score)
    spec=result['episode']['spec']
    save(f.episode/'agent/spec.json',spec)
    update(f.agent/'episode_ready.json',spec=spec)
    save(f.episode/'agent/board_game_result.json',result)
    save(f.episode/'agent/result.json',result['episode'])
    save(f.agent/'run_report.json',report)
    for player in ('player_0','player_1'):
        def change(kind,payload):
            if kind=='action_ledger_bound':payload['contract']['spec_sha256']=h(spec)
            return kind,payload
        rewrite_journal(f.agent/player/'host/action_journal',change)


def test_formal_routing_is_compared_even_with_consistently_resealed_config(fixture,monkeypatch):
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    args=formal_fixture(fixture,monkeypatch)
    config=load_json(fixture.agent/'agent_config.json')
    config['models']['player_1']['endpoint']='https://different.invalid'
    reseal_formal_result(fixture,config=config)
    diagnostic={}
    assert collect_frozen_match_receipts(**args,diagnostics=diagnostic) is None
    assert diagnostic['reason']=='frozen_player_model_changed'


def test_formal_successful_model_response_must_preserve_exact_requested_identity(fixture,monkeypatch):
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    args=formal_fixture(fixture,monkeypatch)
    report=load_json(fixture.agent/'run_report.json')
    report['model_requests']['player_0'][0]['model']='other-model'
    save(fixture.agent/'run_report.json',report)
    path=fixture.agent/'player_0/decisions/000000/model_response.json'
    update(path,metadata=report['model_requests']['player_0'][0])
    diagnostic={}
    assert collect_frozen_match_receipts(**args,diagnostics=diagnostic) is None
    assert diagnostic['reason']=='frozen_model_response_identity_changed'


@pytest.mark.parametrize('invoked',[False,True])
def test_formal_absent_or_blocked_invocation_cannot_invent_language_execution(fixture,monkeypatch,invoked):
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    args=formal_fixture(fixture,monkeypatch)
    path=fixture.agent/'player_1/host/skill_runtime_journal'
    events=Journal(path).read()
    for item in path.glob('*.json'):item.unlink()
    journal=Journal(path)
    journal.append(events[0]['kind'],events[0]['payload'])
    if invoked:
        journal.append(events[1]['kind'],events[1]['payload'])
        folder=fixture.agent/'player_1/host/skill_execution_records/execute1'
        result=load_json(folder/'result.json')|dict(calls=[],language_entered=False,status='failed')
        save(folder/'result.json',result)
        finished=events[-1]['payload']|dict(language_entered=False,result_sha256=h(result),status='failed')
        journal.append('skill_execution_finished',finished)
    diagnostic={}
    receipts=collect_frozen_match_receipts(**args,diagnostics=diagnostic)
    assert receipts is not None,diagnostic
    assert receipts['player_1'].skill_invocations==int(invoked)
    assert receipts['player_1'].skill_language_entries==0


def test_formal_skill_cannot_reuse_opponents_authorization_epoch(fixture,monkeypatch):
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    args=formal_fixture(fixture,monkeypatch)
    path=fixture.agent/'player_1/host/skill_runtime_journal'
    def change(kind,payload):
        if kind=='skill_api_prepared':payload['authorization_epoch']['current_player']='player_0'
        return kind,payload
    rewrite_journal(path,change)
    diagnostic={}
    assert collect_frozen_match_receipts(**args,diagnostics=diagnostic) is None
    assert diagnostic['reason']=='invalid_practice_authorization_epoch'


@pytest.mark.parametrize('omit_native_commit',[False,True])
def test_formal_event_commits_require_same_actual_native_step(fixture,monkeypatch,omit_native_commit):
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    args=formal_fixture(fixture,monkeypatch)
    state=load_json(fixture.episode/'agent/board_game_result.json')['board_game']
    state.update(turn_index=1,round_index=1,commit_count=1,event_cursor=0)
    reseal_formal_result(fixture,state=state)
    event=dict(sequence=0,type='operation_committed',task_id='polyomino_territory',player_id='player_0',
        turn_index=1,native_substep_index=3,simulation_time_s=.0005,details=dict(operation='place_piece',piece_ids=['black_mono']))
    for player in ('player_0','player_1'):
        save(fixture.agent/player/'public_game_events.json',dict(protocol=state['protocol'],cursor=0,events=[event]))
    if not omit_native_commit:
        path=fixture.episode/'private/board_game/native_frames.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['result']['committed']=True
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    diagnostic={}
    receipts=collect_frozen_match_receipts(**args,diagnostics=diagnostic)
    if omit_native_commit:
        assert receipts is None and diagnostic['reason']=='public_commit_without_native_commit'
    else:
        assert receipts is not None,diagnostic
        assert diagnostic['non_pass_commits_by_player']=={'player_0':1,'player_1':0}


def test_formal_audit_cli_keeps_auditor_and_runtime_source_separate_and_never_overwrites(fixture,monkeypatch,capsys):
    from agent.model_client.learning import match_receipts
    args=formal_fixture(fixture,monkeypatch)
    auditor=fixture.agent.parent/'auditor'
    module=auditor/'agent/model_client/learning/match_receipts.py'
    module.parent.mkdir(parents=True)
    module.write_text('# separate frozen auditor CPU fixture\n')
    auditor_sha=save(auditor/'source_snapshot_manifest.json',{'files_sha256':{}})
    monkeypatch.setattr(match_receipts,'__file__',str(module))
    def verify(path,**kwargs):
        assert path==auditor
        return {'verified':True,'snapshot_root':str(auditor)}
    monkeypatch.setattr('scripts.maintenance.freeze_project_sources.verify_runtime_imports',verify)
    request=dict(schema='card_rsi_frozen_match_audit_request.v1',freezes={},
        library_roots={p:str(path) for p,path in args['library_roots'].items()},
        agent_output='agent',episode_root='ep1',receipt_sequence=10)
    for p,freeze in args['freezes'].items():
        path=fixture.agent.parent/(p+'_freeze.json');save(path,freeze.model_dump(mode='json'));request['freezes'][p]=path.name
    request_path=fixture.agent.parent/'request.json';save(request_path,request)
    output=fixture.agent.parent/'audit.json'
    assert match_receipts.main(['--request',str(request_path),'--output',str(output)])==0
    actual=load_json(output)
    assert actual['verified'] and actual['auditor_source_manifest_sha256']==auditor_sha
    assert actual['diagnostics']['runtime_source_manifest_sha256']!=auditor_sha
    previous=output.read_bytes()
    with pytest.raises((ValueError,FileExistsError)):
        match_receipts.main(['--request',str(request_path),'--output',str(output)])
    assert output.read_bytes()==previous


@pytest.mark.parametrize('outcome',['policy_error','infrastructure_error','evaluator_invalid'])
def test_closed_native_practice_execution_error_is_unknown_not_model_task_failure(fixture,outcome):
    # All native/source/skill artifacts remain complete. Only the actual
    # EpisodeResult changes, matching the source02 ACK policy_error incident.
    assert collect(fixture).native_result()
    result=fixture.report['result']|dict(outcome=outcome,physics_valid=True)
    save(fixture.episode/'agent/result.json',result)
    report=fixture.report|dict(result=result,task_outcome='failure')
    save(fixture.agent/'run_report.json',report)
    record=collect(fixture)
    assert record.outcome=='unknown' and record.evidence_source=='unverified'
    assert record.result_valid is False and not record.native_result()
    assert record.verification_issues==['native:episode_execution_error_'+outcome]
    assert record.cost.model_requests==1 and record.cost.native_sim_s==.001
    assert record.result_sha256==h(result)
    assessment=assess_validation(fixture.plan,[record],candidate_event_id='candidate',bundle_hash=h('bundle'),
        condition_sha256=fixture.plan.condition_sha256)
    assert assessment.failures==assessment.successes==0 and assessment.unknown==1
    feedback=public_practice_feedback(record,report)
    assert feedback['outcome']=='unknown' and feedback['native_evidence_verified'] is False
    assert load_json(fixture.episode/'agent/result.json')==result


@pytest.mark.parametrize('outcome',['policy_error','infrastructure_error','evaluator_invalid'])
def test_closed_formal_execution_error_cannot_qualify_frozen_match(fixture,monkeypatch,outcome):
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    args=formal_fixture(fixture,monkeypatch)
    assert collect_frozen_match_receipts(**args) is not None
    reseal_formal_result(fixture,physical=dict(outcome=outcome,physics_valid=True))
    diagnostic={}
    assert collect_frozen_match_receipts(**args,diagnostics=diagnostic) is None
    assert diagnostic['verified'] is False and diagnostic['reason']=='episode_execution_error_'+outcome
