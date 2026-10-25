"""Callable/legacy transport equivalence with CPU workers, not native qualification."""
from copy import deepcopy
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from bench.protocol import Budgets, PolicyRegistry
from bench.runtime.server import create_app
from test_agent_service import (runner, frozen_inputs, session_for, action, join_action,
    publish_targets, teacher_session, TimedHoldTeacher, AcknowledgedTeacher, skill_action)


def call(session, name, arguments=None, request_id='test/request'):
    return session.functions.invoke({'request_id': request_id, 'function': name,
                                     'arguments': arguments or {}})


def steps(worker):
    return [payload['command'] for op, payload in worker.calls if op == 'step']


@pytest.mark.parametrize('function_first', [True, False])
def test_joint_calls_share_legacy_action_identity_budget_and_recording(runner, function_first):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True, on_step=publish_targets))
    payload = action(session, name='move_1', duration=.4)
    arguments = {key: value for key, value in payload.items() if key != 'kind'}
    if function_first:
        response = call(session, 'move_joints', arguments)
        assert response.ok and response.result.status == 'accepted'
    else:
        session.submit(payload)
    join_action(session)
    count = len(steps(workers[-1]))
    assert session.submit(payload)['status'] == 'completed'
    response = call(session, 'move_joints', arguments, 'retry')
    assert response.ok and response.result.status == 'completed'
    assert response.context.budget.action_attempts == 1
    assert len(steps(workers[-1])) == count == 20
    assert response.result.robot_execution_state == session.status('move_1')['robot_execution_state']
    rows = [json.loads(line) for line in (session.recorder.directory/'events.jsonl').read_text().splitlines()]
    assert sum(row['kind'] == 'agent_action' for row in rows) == 1
    assert any(row['kind'] == 'bench_function' for row in rows)


@pytest.mark.parametrize('action_id', ['move/1', '/World/private_canary'])
@pytest.mark.parametrize('transport', ['function', 'legacy'])
@pytest.mark.parametrize('kind', ['direct_joints', 'system0'])
def test_http_action_ids_reject_paths_before_admission(runner, action_id, transport, kind):
    make, _, workers = runner
    if kind == 'direct_joints':
        api = make()
        session, _ = session_for(api)
        payload = action(session, name=action_id)
        function = 'move_joints'
        arguments = {key: value for key, value in payload.items() if key != 'kind'}
    else:
        registry = PolicyRegistry()
        registry.register(TimedHoldTeacher())
        api = make(registry=registry)
        session, _ = session_for(api, track='system0', policy_id='test_teacher', policy_version='v1')
        payload = {'kind': 'system0', 'skill': 'hold', 'action_id': action_id,
            'observation_sequence': 0, 'timeout_sim_s': .04, 'hold_duration_s': .04}
        function = 'hold'
        arguments = {'action_id': action_id, 'observation_sequence': 0, 'duration_s': .04}
    with TestClient(create_app(service=api)) as client:
        endpoint = f'/episodes/{session.spec.episode_id}'
        if transport == 'function':
            response = client.post(endpoint + '/call', json={
                'function': function, 'request_id': 'test/request', 'arguments': arguments})
            assert response.json()['error']['code'] == 'invalid_arguments'
            assert response.json()['request_id'] == 'test/request'
        else:
            response = client.post(endpoint + '/actions', json=payload)
        assert response.status_code == 422
        assert session.clock.action_attempts == 1
        assert session.requests == {} and session.direct_hold_replays == {}
        assert session.life.events == [] and session.life.active_action_id is None
        assert session.thread is None and session.last_action_id is None
        assert session.result is None and not steps(workers[-1])
        attempts = (session.directory/'private/action_attempts.jsonl').read_text().splitlines()
        assert len(attempts) == 1
        rows = [json.loads(line) for line in (session.recorder.directory/'events.jsonl').read_text().splitlines()]
        assert all(row['kind'] not in ('agent_action', 'action_event') for row in rows)
        # Request IDs are correlation names, not action-route path segments.
        observation = client.post(endpoint + '/call', json={
            'function': 'observe', 'request_id': 'test/request', 'arguments': {}})
        assert observation.status_code == 200
        assert observation.json()['ok'] and observation.json()['request_id'] == 'test/request'
        assert session.clock.action_attempts == 1


def test_typed_motion_error_is_charged_once_over_http_and_shares_legacy_budget(runner):
    make, _, workers = runner
    api = make()
    session, _ = session_for(api, budgets=Budgets(max_actions=2))
    with TestClient(create_app(service=api)) as client:
        endpoint = f'/episodes/{session.spec.episode_id}/call'
        result = client.post(endpoint, json={'function': 'move_joints', 'request_id': 'bad/request',
                                            'arguments': {'action_id': 'broken'}})
        assert result.status_code == 422
        assert result.json()['error']['code'] == 'invalid_arguments'
        assert result.json()['request_id'] == 'bad/request'
        assert result.json()['context']['budget']['action_attempts'] == 1
        response = call(session, 'move_wrist', {'action_id': 'unsupported', 'observation_sequence': 0,
            'hand': 'right', 'timeout_sim_s': .2, 'wrist': {'frame_id': 'world',
            'position_m': [0., 0., 1.], 'quaternion_wxyz': [1., 0., 0., 0.]}})
        assert response.error.code == 'unsupported_capability'
        assert response.context.budget.action_attempts == 2
        with pytest.raises(Exception, match='action_budget'):
            session.submit(action(session))
        assert session.result.outcome == 'action_budget'
        assert not steps(workers[-1])


def test_rejected_motion_and_unknown_queries_have_distinct_costs(runner):
    make, _, _ = runner
    session, _ = session_for(make())
    assert call(session, 'made_up').error.code == 'unsupported_function'
    assert call(session, 'query_ik', {'objects': {}}).error.code == 'invalid_arguments'
    assert session.clock.action_attempts == 0
    payload = action(session, sequence=99)
    response = call(session, 'move_joints', {k:v for k,v in payload.items() if k != 'kind'})
    assert response.error.code == 'stale_observation'
    assert session.clock.action_attempts == 1


def test_failed_duration_preflight_is_action_failure_not_function_failure(runner):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True, on_step=publish_targets))
    payload = action(session, duration=.02)
    response = call(session, 'move_joints', {k:v for k,v in payload.items() if k != 'kind'})
    assert response.ok
    join_action(session)
    response = call(session, 'get_action', {'action_id': 'move_1'})
    assert response.ok and response.result.status == 'failed'
    assert response.result.diagnostic.code == 'insufficient_duration'
    assert response.result.elapsed_sim_s == 0
    assert not steps(workers[-1])
    assert session.result is None


def test_hold_uses_ack_targets_and_retries_original_command_after_another_move(runner):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True, on_step=publish_targets))
    # Contact lag is deliberately different from the drive reference.
    session.raw['joint_positions_rad'] = dict.fromkeys(session.raw['joint_positions_rad'], .5)
    args = {'action_id': 'hold_1', 'observation_sequence': 0, 'duration_s': .04}
    assert call(session, 'hold', args).ok
    join_action(session)
    assert all(value == 0. for command in steps(workers[-1]) for value in command['joint_positions_rad'].values())
    session.submit(action(session, name='move_2', duration=.4))
    join_action(session)
    before = len(steps(workers[-1]))
    assert call(session, 'hold', args).result.status == 'completed'
    assert len(steps(workers[-1])) == before
    assert session.clock.action_attempts == 2
    conflict = call(session, 'hold', {**args, 'duration_s': .06})
    assert conflict.error.code == 'action_conflict'
    assert session.clock.action_attempts == 3


@pytest.mark.parametrize('track',['direct_joints','system0'])
def test_fixed_hold_rejects_executor_without_stationary_ack_capability(runner,track):
    make, _, workers = runner
    if track == 'system0':
        registry = PolicyRegistry()
        registry.register(TimedHoldTeacher())
        session, _ = session_for(make(registry=registry),track=track,
                                 policy_id='test_teacher',policy_version='v1')
    else:
        session, _ = session_for(make(acknowledged_targets=True,on_step=publish_targets))
    args = {'action_id':'fixed','observation_sequence':0,'duration_s':.04,'mode':'acknowledged_drive'}
    result = call(session,'hold',args)
    assert not result.ok and result.error.code == 'unsupported_capability'
    assert session.clock.action_attempts == 1 and not steps(workers[-1])
    assert not session.requests and session.thread is None
    if track == 'system0':
        with pytest.raises(ValueError,match='advertised'):
            from bench.protocol.validation import validate_action
            from bench.protocol.schemas import System0Action
            validate_action(System0Action(action_id='legacy',observation_sequence=0,skill='hold',
                timeout_sim_s=.04,hold_duration_s=.04,hold_mode='acknowledged_drive'),
                session.capabilities,session.spec)


def test_legacy_hold_replays_through_function_after_reference_changes(runner):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True, on_step=publish_targets))
    payload = action(session, name='legacy_hold', duration=.04)
    payload['arm_joint_positions_rad'] = dict.fromkeys(session.capabilities.arm_joints, 0.)
    payload['hand_joint_positions_rad'] = dict.fromkeys(session.capabilities.hand_joints, 0.)
    session.submit(payload)
    join_action(session)
    session.submit(action(session, name='move_2', duration=.4))
    join_action(session)
    before = len(steps(workers[-1]))
    response = call(session, 'hold', {'action_id': 'legacy_hold', 'duration_s': .04, 'observation_sequence': 0})
    assert response.ok and response.result.status == 'completed'
    assert session.clock.action_attempts == 2 and len(steps(workers[-1])) == before


def test_system0_hold_preserves_defaults_for_legacy_retries(runner):
    make, _, workers = runner
    session = teacher_session(make, TimedHoldTeacher())
    arguments = {'action_id': 'timed_hold', 'observation_sequence': 0, 'duration_s': .04}
    assert call(session, 'hold', arguments).ok
    join_action(session)
    reply = session.submit({'kind': 'system0', 'skill': 'hold', 'action_id': 'timed_hold',
        'observation_sequence': 0, 'hold_duration_s': .04, 'timeout_sim_s': .04})
    assert reply['status'] == 'completed'
    assert session.clock.action_attempts == 1 and len(steps(workers[-1])) == 2
    from bench.runtime.service import canonical_hash
    rows = [json.loads(line) for line in (session.directory/'private/action_attempts.jsonl').read_text().splitlines()]
    assert all(row['payload_sha256'] == canonical_hash(row['payload']) for row in rows)
    assert rows[0]['canonical_action_sha256'] == session.requests['timed_hold']


def test_cancel_flag_belongs_to_requested_action_not_global_event(runner):
    make, _, _ = runner
    teacher = AcknowledgedTeacher()
    session = teacher_session(make, teacher)
    first = skill_action(session)
    session.submit(first)
    join_action(session)
    teacher.block = True
    teacher.started.clear()
    second = {**skill_action(session), 'action_id': 'skill_2'}
    session.submit(second)
    assert teacher.started.wait(timeout=1)
    response = call(session, 'cancel_action', {'action_id': 'skill_2'})
    assert response.ok and response.result.cancellation_requested
    assert not call(session, 'cancel_action', {'action_id': 'skill_1'}).result.cancellation_requested
    teacher.release.set()
    join_action(session)
    assert call(session, 'cancel_action', {'action_id': 'skill_2'}).result.cancellation_requested
    assert session.status('skill_2')['status'] == 'cancelled'


def test_queries_use_wall_budget_without_advancing_physics_and_sealed_results_stay_immutable(runner):
    make, clock, workers = runner
    session, _ = session_for(make(depth=True), budgets=Budgets(wall_time_s=1.))
    response = call(session, 'read_camera', {'observation_sequence': 0, 'camera_id': 'overview'})
    assert response.ok
    response.result.camera.camera_id = 'changed_by_client'
    response.context.budget.limits.max_actions = 1
    catalog = session.functions.catalog()
    catalog.robot_capabilities.arm_joints.clear()
    assert session.snapshot.cameras[0].camera_id == 'overview'
    assert session.spec.budgets.max_actions == 6000
    assert len(session.capabilities.arm_joints) == 7
    assert not steps(workers[-1]) and session.clock.action_attempts == 0
    clock.value = 2.
    assert call(session, 'get_robot').error.code == 'budget_exhausted'
    assert session.result.outcome == 'wall_timeout'
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in session.recorder.directory.iterdir() if p.is_file()}
    response = call(session, 'get_result')
    assert response.ok and response.context.request_recorded is False
    response.result.outcome = 'cancelled'
    assert session.result.outcome == 'wall_timeout'
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in session.recorder.directory.iterdir() if p.is_file()}
    assert before == after


def test_invalid_ack_does_not_leave_a_half_admitted_action(runner):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True))
    session.raw['robot']['joint_targets_rad']['fr3_joint1'] = float('nan')
    response = call(session, 'move_joints', {k:v for k,v in action(session, duration=.4).items() if k != 'kind'})
    assert response.ok and response.result.status == 'accepted'
    join_action(session)
    assert session.status('move_1')['status'] == 'failed'
    assert session.result.outcome == 'infrastructure_error'
    assert not steps(workers[-1])
