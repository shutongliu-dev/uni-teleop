"""Callable execution with real URDF controllers and an ideal CPU worker.

These tests exercise the complete episode/action/ACK path, not native dynamics,
contact, collision safety or grasp qualification. The worker instantly tracks
joint targets; no controller planning or execution method is replaced.
"""
from copy import deepcopy

import numpy as np
import pytest

from bench.control.bimanual import BimanualFingertipPolicy
from bench.control.fingertip import ClassicalFingertipPolicy
from bench.geometry import quaternion_matrix, rotation_distance
from bench.paths import DATA_ROOT
from bench.protocol import PolicyRegistry
import test_agent_service as service_tests
from test_agent_service import frozen_inputs, runner, join_action, session_for


@pytest.fixture(params=['single_right', 'dual_left', 'dual_right'])
def controlled_session(request, runner, monkeypatch):
    side = request.param.rsplit('_', 1)[1]
    cls = ClassicalFingertipPolicy if request.param.startswith('single') else BimanualFingertipPolicy
    policy = cls(f'{DATA_ROOT}/assets', [0., 0., .77])
    geometry = policy.geometry
    models = geometry.sides if hasattr(geometry, 'sides') else {'right': geometry}
    initial = {joint.name: float(np.clip(0., joint.lower, joint.upper)) for joint in geometry.joints}
    for model in models.values():
        initial.update(zip(model.arm.names, [0., -.785, 0., -2.356, 0., 1.57, .785]))

    class GeometryWorker(service_tests.FakeWorker):
        def __init__(self, *, hand_id, **kwargs):
            assert hand_id == geometry.robot_id
            super().__init__(hand_id='sharpa_w01_right', **kwargs)
            self.arm = [name for model in models.values() for name in model.arm.names]
            self.hand = [name for model in models.values() for name in model.hand.names]
            self.active = self.arm + self.hand
            self.raw['joint_positions_rad'] = dict(initial)
            self.raw['joint_velocities_rad_s'] = dict.fromkeys(self.active, 0.)
            self.raw['robot'] = {'joint_targets_rad': dict(initial)}
            self.initial['observation'] = deepcopy(self.raw)
            self.initial['metadata']['action_spec'] = {
                'joint_names': self.active, 'arm_joint_names': self.arm, 'hand_joint_names': self.hand,
                'joint_limits_rad': {j.name: [j.lower, j.upper] for j in geometry.joints}}
            self.initial['metadata']['native_joint_max_velocities'] = {
                j.name: j.velocity for j in geometry.joints}

    monkeypatch.setattr(service_tests, 'FakeWorker', GeometryWorker)
    registry = PolicyRegistry()
    registry.register(policy)
    make, _, workers = runner
    session, _ = session_for(make(registry=registry, on_step=service_tests.publish_targets),
        robot_id=geometry.robot_id, track='system0', policy_id=policy.descriptor.policy_id,
        policy_version=policy.descriptor.version)
    assert session.policy is policy and session.robot_geometry is geometry
    assert len(session.capabilities.joint_limits) == (58 if len(models) == 2 else 29)
    return session, workers[-1], side, models


def invoke(session, function, arguments, request_id=None):
    return session.functions.invoke({'function': function, 'arguments': arguments,
        'request_id': request_id or function}).model_dump(mode='json')


def steps(worker):
    return [payload['command'] for operation, payload in worker.calls if operation == 'step']


def wrist_target(session, model):
    q = {name: session.snapshot.proprioception.joint_positions_rad[name] for name in model.names}
    q[model.arm.names[0]] += .04
    return model.public_geometry(q)['wrist']


def geometry_arguments(session, side, models, objective, action_id):
    target = ({'hand_geometries': {side: objective}} if len(models) == 2 else {'geometry': objective})
    return {'action_id': action_id, 'observation_sequence': session.sequence,
            'timeout_sim_s': 3., **target}


def execute(session, worker, function, arguments):
    first_step = len(steps(worker))
    previous = dict(worker.raw['robot']['joint_targets_rad'])
    attempts = session.clock.action_attempts
    result = invoke(session, function, arguments)
    assert result['ok'], result
    assert result['result']['status'] in ('accepted', 'running', 'completed')
    join_action(session)
    result = invoke(session, 'get_action', {'action_id': arguments['action_id']})
    assert result['ok'], result
    status = result['result']
    assert status['status'] == 'completed', status
    assert session.result is None  # Completing a skill does not complete the task.
    assert session.clock.action_attempts == attempts + 1
    commands = steps(worker)[first_step:]
    assert commands and all(command['duration_s'] == .02 for command in commands)
    for command in commands:
        assert set(command['joint_positions_rad']) == set(session.capabilities.joint_limits)
        for name, value in command['joint_positions_rad'].items():
            limit = session.capabilities.joint_limits[name]
            assert limit.lower_rad <= value <= limit.upper_rad
            assert abs(value - previous[name]) <= limit.max_velocity_rad_s * .02 + 1e-12
        previous = command['joint_positions_rad']
    assert session.raw['robot']['joint_targets_rad'] == commands[-1]['joint_positions_rad']
    assert session.snapshot.proprioception.commanded_joint_positions_rad == commands[-1]['joint_positions_rad']
    return status, commands


def assert_untargeted_side_held(commands, models, side, previous):
    for other, model in models.items():
        if other != side:
            for command in commands:
                assert {name: command['joint_positions_rad'][name] for name in model.names} == {
                    name: previous[name] for name in model.names}


def assert_wrist_reached(session, model, target):
    q = {name: session.snapshot.proprioception.joint_positions_rad[name] for name in model.names}
    actual = model.public_geometry(q)['wrist']
    assert np.linalg.norm(np.asarray(actual['position_m']) - target['position_m']) <= .003
    assert rotation_distance(quaternion_matrix(actual['quaternion_wxyz']),
                             quaternion_matrix(target['quaternion_wxyz'])) <= .03


def test_callable_motion_functions_share_real_controller_and_hold_omitted_hand(controlled_session):
    session, worker, side, models = controlled_session
    policy = session.policy
    model = models[side]
    before = dict(session.snapshot.proprioception.joint_positions_rad)
    target = wrist_target(session, model)
    status, commands = execute(session, worker, 'move_wrist', {
        'action_id': 'wrist', 'observation_sequence': session.sequence, 'hand': side,
        'wrist': target, 'timeout_sim_s': 3.})
    assert status['termination_reason'] == 'controller_converged'
    assert status['robot_execution_state']['objective_status'] == 'reached'
    assert_wrist_reached(session, model, target)
    assert any(abs(commands[-1]['joint_positions_rad'][name] - before[name]) > .005
               for name in model.arm.names)
    assert_untargeted_side_held(commands, models, side, before)

    before = dict(session.snapshot.proprioception.joint_positions_rad)
    posture = {f'{side}_index_PIP': .12}
    status, commands = execute(session, worker, 'set_hand_targets', {
        'action_id': 'fingers', 'observation_sequence': session.sequence, 'hand': side,
        'hand_posture_rad': posture, 'timeout_sim_s': 3.})
    assert status['robot_execution_state']['objective_status'] == 'reached'
    assert abs(session.snapshot.proprioception.joint_positions_rad[f'{side}_index_PIP'] - .12) < .01
    assert_wrist_reached(session, model, target)
    assert_untargeted_side_held(commands, models, side, before)

    before = dict(session.snapshot.proprioception.joint_positions_rad)
    target = wrist_target(session, model)
    objective = {'wrist': target, 'fingertip_frame_id': model.frame_id,
                 'hand_posture_rad': {f'{side}_index_PIP': .20}}
    status, commands = execute(session, worker, 'move_geometry',
        geometry_arguments(session, side, models, objective, 'combined'))
    assert status['robot_execution_state']['objective_status'] == 'reached'
    assert_wrist_reached(session, model, target)
    assert abs(session.snapshot.proprioception.joint_positions_rad[f'{side}_index_PIP'] - .20) < .01
    assert_untargeted_side_held(commands, models, side, before)

    before = dict(session.raw['robot']['joint_targets_rad'])
    controllers = policy.sides if hasattr(policy, 'sides') else {'right': policy}
    nominal = {hand: controller.goal.copy() for hand, controller in controllers.items()}
    start = session.clock.sim_time_s
    status, commands = execute(session, worker, 'hold', {
        'action_id': 'hold', 'observation_sequence': session.sequence, 'duration_s': .08})
    assert status['termination_reason'] == 'time_hold_completed'
    assert status['robot_execution_state']['completion_scope'] == 'timed_reference_maintenance'
    assert session.clock.sim_time_s - start == pytest.approx(.08)
    # Normal hold maintains the preceding nominal goal with bounded feedback;
    # it need not freeze the last drive target inside the convergence tolerance.
    for hand, controller in controllers.items():
        np.testing.assert_array_equal(controller.goal, nominal[hand])
    assert status['robot_execution_state']['objective_status'] == 'not_checked'
    assert status['robot_execution_state']['reference_stationary'] is True
    assert_untargeted_side_held(commands, models, side, before)
    assert_wrist_reached(session, model, target)
    error = abs(session.snapshot.proprioception.joint_positions_rad[f'{side}_index_PIP'] - .20)
    assert error <= controllers[side].limits.joint_tracking_tolerance_rad
    assert status['robot_execution_state']['maintained_joint_error_rad'][f'{side}_index_PIP'] == pytest.approx(error)
    assert session.policy is policy


def test_equivalent_cross_function_action_retry_does_not_execute_again(controlled_session):
    session, worker, side, models = controlled_session
    model = models[side]
    target = wrist_target(session, model)
    sequence = session.sequence
    arguments = {'action_id': 'same_motion', 'observation_sequence': sequence,
                 'hand': side, 'wrist': target, 'timeout_sim_s': 3.}
    status, _ = execute(session, worker, 'move_wrist', arguments)
    attempts, time, count = session.clock.action_attempts, session.clock.sim_time_s, len(steps(worker))
    objective = {'wrist': target, 'fingertip_frame_id': model.frame_id,
                 'position_tolerance_m': .01, 'orientation_tolerance_rad': .1}
    equivalent = geometry_arguments(session, side, models, objective, 'same_motion')
    equivalent['observation_sequence'] = sequence  # Replay retains the original decision snapshot.
    replay = invoke(session, 'move_geometry', equivalent, request_id='equivalent_retry')
    assert replay['ok'], replay.get('error', replay)
    assert replay['result'] == status
    assert session.clock.action_attempts == attempts
    assert session.clock.sim_time_s == time and len(steps(worker)) == count
    assert session.life.active_action_id is None
