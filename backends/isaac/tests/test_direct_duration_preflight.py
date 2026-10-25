"""Direct command timing with a CPU worker; no native-physics qualification."""
from copy import deepcopy
from types import SimpleNamespace
import math

import pytest

from bench.protocol import PolicyRegistry
from bench.protocol.schemas import DirectControlHandoff, PolicyDescriptor
from bench.runtime.service import EpisodeSession
from test_agent_service import (Teacher, action, frozen_inputs, join_action,
                                publish_targets, runner, session_for)


@pytest.mark.parametrize('joint,target,speed', [
    ('fr3_joint1', .2, .65), ('active_finger_joint_0', .3, 1.),
])
def test_short_direct_rejects_without_steps_and_retry_uses_same_ack(runner, joint, target, speed):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True, on_step=publish_targets))
    worker = workers[-1]
    request = action(session, duration=.02)
    request['arm_joint_positions_rad'] = dict.fromkeys(session.capabilities.arm_joints, 0.)
    request['hand_joint_positions_rad'] = dict.fromkeys(session.capabilities.hand_joints, 0.)
    request['arm_joint_positions_rad' if joint in session.capabilities.arm_joints
            else 'hand_joint_positions_rad'][joint] = target
    raw = deepcopy(session.raw)
    cameras = session.observe()['cameras']
    worker.calls.clear()
    session.submit(request); join_action(session)
    feedback = session.status(request['action_id'])
    assert feedback['status'] == 'failed' and feedback['termination_reason'] == 'begin_rejected'
    assert feedback['elapsed_sim_s'] == 0. and session.sequence == 0 and session.result is None
    assert not worker.calls and session.raw == raw
    assert session.observe()['cameras'] == cameras
    timing = feedback['diagnostic']['duration_preflight']
    minimum = math.ceil(1.875*target/speed/.02-1e-9)
    assert timing['skill'] == 'direct_joints' and timing['minimum_steps'] == minimum
    assert timing['requested_duration_s'] == .02 and timing['minimum_duration_s'] == minimum*.02
    assert timing['braking_steps'] == timing['settling_steps'] == 0
    assert timing['trajectory_steps'] == minimum
    assert session.submit(request) == feedback  # Same-ID retry never reexecutes.
    assert not worker.calls
    request.update(action_id='retry', duration_s=timing['minimum_duration_s'])
    session.submit(request); join_action(session)
    assert session.status('retry')['status'] == 'completed'
    commands = [payload['command']['joint_positions_rad'][joint]
                for operation, payload in worker.calls if operation == 'step']
    assert len(commands) == minimum and commands[-1] == target
    assert max(abs(b-a)/.02 for a,b in zip([0.]+commands,commands)) <= speed+1e-12
    assert session.clock.sim_time_s == minimum*.02


def test_native_velocity_bound_is_published_and_used_for_preflight(runner):
    make, _, workers = runner
    session, reply = session_for(make(acknowledged_targets=True,
                                     native_velocity_caps={'fr3_joint1': .1}))
    assert reply['capabilities']['joint_limits']['fr3_joint1']['max_velocity_rad_s'] == .1
    request = action(session, duration=.4)
    session.submit(request); join_action(session)
    status = session.status(request['action_id'])
    assert status['diagnostic']['duration_preflight']['minimum_duration_s'] == pytest.approx(1.88)
    assert not [row for row in workers[-1].calls if row[0] == 'step']


def test_velocity_sources_use_stricter_urdf_native_and_reference_caps():
    session = SimpleNamespace(robot_geometry=SimpleNamespace(by_name={
        'joint': SimpleNamespace(velocity=.2)}), metadata={'native_joint_max_velocities': {'joint': .3}})
    assert EpisodeSession._direct_velocity_limit(session, 'joint', True) == .2
    session.robot_geometry.by_name['joint'].velocity = 5.
    assert EpisodeSession._direct_velocity_limit(session, 'joint', True) == .3
    session.metadata = {}
    assert EpisodeSession._direct_velocity_limit(session, 'joint', True) == .65


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, 0., -1.])
def test_invalid_native_velocity_limit_is_rejected(runner, bad):
    make, _, workers = runner
    with pytest.raises(RuntimeError, match='Invalid direct joint velocity limit'):
        session_for(make(acknowledged_targets=True, native_velocity_caps={'fr3_joint1': bad}))
    assert not [row for row in workers[-1].calls if row[0] == 'step']


def test_native_worker_without_acknowledged_targets_cannot_advertise_a_ramp(runner, monkeypatch):
    from test_agent_service import FakeWorker
    from bench.runtime import controllers
    original = FakeWorker.__init__

    def initialize(self, **kwargs):
        original(self, **kwargs)
        self.initial['metadata']['native_isaacsim_app'] = True

    monkeypatch.setattr(FakeWorker, '__init__', initialize)
    # This test isolates startup's missing-ACK branch, not the URDF/FK audit.
    monkeypatch.setattr(controllers, 'public_robot_geometry', lambda configuration: None)
    make, _, workers = runner
    with pytest.raises(RuntimeError, match='requires acknowledged drive targets'):
        session_for(make())
    assert not [row for row in workers[-1].calls if row[0] == 'step']


def test_unchanged_acknowledged_targets_allow_one_tick_without_new_velocity_claim(runner):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True, on_step=publish_targets))
    request = action(session, duration=.02)
    request['arm_joint_positions_rad'] = dict.fromkeys(session.capabilities.arm_joints, 0.)
    request['hand_joint_positions_rad'] = dict.fromkeys(session.capabilities.hand_joints, 0.)
    session.submit(request); join_action(session)
    assert session.status(request['action_id'])['status'] == 'completed'
    assert len([row for row in workers[-1].calls if row[0] == 'step']) == 1


def test_direct_timing_uses_acknowledged_drive_not_loaded_measured_position(runner):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True, on_step=publish_targets))
    # A loaded encoder may be far from the retained drive. A direct hold must
    # neither re-anchor to it nor add the encoder error to its timing delta.
    session.raw['joint_positions_rad']['fr3_joint1'] = .8
    session.raw['robot']['joint_targets_rad']['fr3_joint1'] = .1
    request = action(session, duration=.02)
    request['arm_joint_positions_rad'] = dict(session.raw['robot']['joint_targets_rad'])
    request['arm_joint_positions_rad'] = {n:request['arm_joint_positions_rad'][n]
                                         for n in session.capabilities.arm_joints}
    request['hand_joint_positions_rad'] = dict.fromkeys(session.capabilities.hand_joints, 0.)
    session.submit(request); join_action(session)
    commands = [payload['command']['joint_positions_rad'] for operation,payload in workers[-1].calls
                if operation == 'step']
    assert session.status(request['action_id'])['status'] == 'completed'
    assert len(commands) == 1 and commands[0]['fr3_joint1'] == .1


@pytest.mark.parametrize('bad', [None, float('nan'), 1.01])
def test_lost_or_invalid_ack_cannot_fall_back_to_step_target(runner, bad):
    make, _, workers = runner
    session, _ = session_for(make(acknowledged_targets=True))
    if bad is None:
        session.raw['robot'].pop('joint_targets_rad')
    else:
        session.raw['robot']['joint_targets_rad']['fr3_joint1'] = bad
    session.submit(action(session, duration=.5)); join_action(session)
    assert session.result.outcome == 'infrastructure_error'
    assert not [row for row in workers[-1].calls if row[0] == 'step']


class HandoffPolicy(Teacher):
    """Explicit protocol double: distinguish checking a handoff from ACK adoption."""
    def __init__(self):
        super().__init__()
        self.descriptor = PolicyDescriptor(policy_id='classical_fingertip',version='test',
            implementation_sha256='c'*64,trained=False,skills=['move_geometry','hold'],
            camera_observation_mode='decision_only',direct_control_handoff=DirectControlHandoff())
        self.checked = 0
        self.adopted = []
        self._pending_step_state = None

    def check_direct_handoff(self):
        self.checked += 1

    def adopt_direct_reference(self, command, **fields):
        self.adopted.append((command.model_dump(), deepcopy(fields)))

    def commit_pending_step(self):
        raise AssertionError('Direct execution must not commit a classical step')

    def discard_pending_step(self):
        pass


def test_free_track_rejection_does_not_change_handoff_and_retry_records_real_ack(runner):
    make, _, workers = runner
    policy = HandoffPolicy()
    registry = PolicyRegistry(); registry.register(policy)
    session, _ = session_for(make(registry=registry, acknowledged_targets=True, on_step=publish_targets),
        track='free_tool_agent',policy_id='classical_fingertip',policy_version='test')
    worker = workers[-1]
    request = action(session, duration=.02)
    session.submit(request); join_action(session)
    assert session.status(request['action_id'])['diagnostic']['code'] == 'insufficient_duration'
    assert policy.checked == 0 and policy.adopted == []
    assert session.free_handoff.command is None and session.free_handoff.sequence == 0
    assert not (session.directory/'private/control_handoffs.jsonl').exists()
    request.update(action_id='retry',duration_s=.4)
    session.submit(request); join_action(session)
    assert session.status('retry')['status'] == 'completed'
    assert policy.checked == 1 and len(policy.adopted) == 20
    assert policy.adopted[0][1]['velocity_rad_s'] is None  # One ACK cannot establish velocity.
    previous = policy.adopted[-2][0]['joint_positions_rad']
    last, fields = policy.adopted[-1]
    assert fields['velocity_rad_s'] == {name:(value-previous[name])/.02
        for name,value in last['joint_positions_rad'].items()}
    assert session.free_handoff.sequence == 20
    assert len([row for row in worker.calls if row[0] == 'step']) == 20
