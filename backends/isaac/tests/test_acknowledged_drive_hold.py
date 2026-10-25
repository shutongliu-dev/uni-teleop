"""Fixed ACK maintenance with real controllers and CPU feedback, not physics."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.control.bimanual import BimanualFingertipPolicy
from bench.control.factory import robot_configuration
from bench.control.fingertip import GeometryIKError
from bench.protocol.functions import HoldArguments
from bench.protocol.schemas import AgentObservation, JointLimit, RobotJointCommand, System0Action
from bench.runtime.board_game_control import BoardGameFingertipPolicy
from test_classical_fingertip_control import geometry, home, ready_policy
from test_robot_instances import robot


def hold(sequence=0, mode='acknowledged_drive', duration=.06):
    return System0Action(action_id='hold', observation_sequence=sequence, skill='hold',
        timeout_sim_s=duration, hold_duration_s=duration, hold_mode=mode)


def observe(q, ack, sequence=0, episode='public_test'):
    return AgentObservation(episode_id=episode, sequence=sequence, sim_time_s=sequence*.02,
        profile='vision_proprio', instruction='Public robot maintenance', cameras=[], proprioception={
            'joint_positions_rad':q, 'joint_velocities_rad_s':dict.fromkeys(q, 0.),
            'commanded_joint_positions_rad':ack})


def state(policy):
    return BimanualFingertipPolicy._state(policy)


def same_state(before, after):
    assert before.keys() == after.keys()
    for key in before:
        if isinstance(before[key], np.ndarray): np.testing.assert_array_equal(before[key], after[key])
        else: assert before[key] == after[key], key


def test_legacy_hold_payload_identity_and_explicit_mode_are_distinct():
    args = {'action_id':'hold','observation_sequence':0,'duration_s':.06}
    assert HoldArguments(**args).model_dump(mode='json') == args
    assert HoldArguments(**args,mode='reference').model_dump(mode='json') == args
    assert HoldArguments(**args,mode='acknowledged_drive').model_dump()['mode'] == 'acknowledged_drive'
    assert 'hold_mode' not in hold(mode='reference').model_dump(mode='json')
    assert hold().model_dump(mode='json')['hold_mode'] == 'acknowledged_drive'


def test_fixed_hold_anchors_ack_with_drift_then_unlocks_only_selected_chains(geometry):
    policy = ready_policy(geometry)
    ack = {n:float(np.float32(v)) for n,v in home(geometry).items()}
    q = dict(ack)
    q['fr3_joint1'] -= .001
    q['right_index_PIP'] += .03
    policy.integral[:] = .01
    policy._arm_rest_integral[:] = .007
    policy.begin(hold(), observe(q, ack))
    assert policy._fixed_drive_mask.all()
    assert not policy.integral.any() and not policy._arm_rest_integral.any()
    for sequence in range(3):
        result = policy.step(observe(q, ack, sequence))
        assert result.command.joint_positions_rad == ack
        policy.commit_pending_step()
    assert policy.step(observe(q, ack, 3)).status == 'completed'
    target = geometry.public_geometry(q)['wrist']
    policy.begin(System0Action(action_id='move', observation_sequence=3, skill='move_geometry',
        timeout_sim_s=1.,geometry={'wrist':target,'hand_posture_rad':{'right_thumb_IP':.05}}), observe(q, ack, 3))
    for index,name in enumerate(geometry.names):
        selected = name in geometry.arm.names or name in [j.name for j in geometry.finger_joints['thumb']]
        assert bool(policy._fixed_drive_mask[index]) == (not selected)
    result = policy.step(observe(q, ack, 3))
    assert result.status == 'running'
    for name in geometry.hand.names:
        if name not in [j.name for j in geometry.finger_joints['thumb']]:
            assert result.command.joint_positions_rad[name] == ack[name]
    policy.discard_pending_step()


def test_default_hold_still_tracks_nominal_with_pi(geometry):
    policy = ready_policy(geometry)
    ack = home(geometry)
    policy.previous = np.array([ack[n] for n in geometry.names])
    policy.velocity = np.zeros(len(ack))
    policy.goal = policy.previous.copy()
    policy._has_goal = True
    q = dict(ack, fr3_joint1=-.0001)
    policy.begin(hold(mode='reference'), observe(q, ack))
    result = policy.step(observe(q, ack))
    assert result.command.joint_positions_rad['fr3_joint1'] > ack['fr3_joint1']
    assert not policy._fixed_drive_mask.any()


@pytest.mark.parametrize('bound',['lower','upper'])
def test_native_float32_boundary_ack_keeps_identical_drive_bits(geometry,bound):
    policy = ready_policy(geometry)
    exact = np.array([getattr(j,bound) for j in geometry.joints])
    native = exact.astype(np.float32)
    ack = dict(zip(geometry.names,native.astype(float).tolist()))
    policy.previous = exact.copy()
    policy.velocity = np.zeros(len(exact))
    policy.begin(hold(),observe(ack,ack))
    for sequence in range(3):
        result = policy.step(observe(ack,ack,sequence))
        target = np.array([result.command.joint_positions_rad[n] for n in geometry.names])
        assert np.all(target >= geometry.lower) and np.all(target <= geometry.upper)
        np.testing.assert_array_equal(target.astype(np.float32),native)
        policy.commit_pending_step()
    assert policy.step(observe(ack,ack,3)).status == 'completed'


@pytest.mark.parametrize('damage', ['missing','inventory','nonfinite','bounds','moving','unknown_velocity','different_ack','late_unknown'])
def test_bad_or_moving_ack_rejects_without_mutating_policy(geometry, damage):
    policy = ready_policy(geometry)
    q = home(geometry)
    ack = dict(q)
    sequence = 0
    policy.previous = np.array([q[n] for n in geometry.names])
    policy.velocity = np.zeros(len(q))
    if damage == 'missing': ack = None
    elif damage == 'inventory': ack.pop('fr3_joint1')
    elif damage == 'nonfinite': ack['fr3_joint1'] = float('nan')
    elif damage == 'bounds': ack['fr3_joint1'] = 10.
    elif damage == 'moving': policy.velocity[0] = .01
    elif damage == 'unknown_velocity': policy.velocity = None
    elif damage == 'different_ack': ack['fr3_joint1'] += .001
    elif damage == 'late_unknown': policy.previous = policy.velocity = None; sequence = 1
    # Construct malformed sensor data without letting schema validation hide the controller check.
    obs = observe(q,q,sequence)
    obs = obs.model_copy(update={'proprioception':obs.proprioception.model_copy(
        update={'commanded_joint_positions_rad':ack})})
    before = state(policy)
    with pytest.raises(GeometryIKError, match='acknowledged'):
        policy.begin(hold(sequence), obs)
    same_state(before,state(policy))


def test_discard_cancel_and_changed_ack_never_emit_an_extra_command(geometry):
    policy = ready_policy(geometry)
    ack = home(geometry)
    q = dict(ack,fr3_joint1=.001)
    obs = observe(q,ack)
    policy.begin(hold(),obs)
    before = state(policy)
    first = policy.step(obs)
    policy.discard_pending_step()
    # step diagnostics are lazily created; compare all pre-step fields.
    after = state(policy)
    same_state(before,{key:after[key] for key in before})
    assert policy.step(obs).command == first.command
    policy.cancel('user_cancel')
    np.testing.assert_array_equal(policy.previous,[ack[n] for n in geometry.names])
    assert policy._pending_step_state is None and not policy._active
    with pytest.raises(ValueError,match='No active'):
        policy.step(obs)
    policy.begin(hold(),obs)
    policy.step(obs)
    policy.commit_pending_step()
    altered = dict(ack,fr3_joint1=.002)
    result = policy.step(observe(q,altered,1))
    assert result.status == 'failed' and result.command is None
    assert result.diagnostic.code == 'control_handoff_unverified'
    np.testing.assert_array_equal(policy.previous,[ack[n] for n in geometry.names])


@pytest.mark.parametrize('velocity',[None, .01, 0.])
def test_free_track_direct_handoff_requires_verified_stationary_velocity(geometry,velocity):
    policy = ready_policy(geometry)
    ack = home(geometry)
    policy.adopt_direct_reference(RobotJointCommand(duration_s=.02,joint_positions_rad=ack),
        velocity_rad_s=None if velocity is None else dict.fromkeys(ack,velocity),
        observation_sequence=2,sim_time_s=.04)
    obs = observe(ack,ack,2)
    before = state(policy)
    if velocity != 0.:
        with pytest.raises(GeometryIKError): policy.begin(hold(2),obs)
        same_state(before,state(policy))
    else:
        policy.begin(hold(2),obs)
        result = policy.step(obs)
        assert result.command.joint_positions_rad == ack
        policy.discard_pending_step()
        assert policy._direct_handoff_pending
        policy.step(obs)
        policy.commit_pending_step()
        assert not policy._direct_handoff_pending


@pytest.mark.parametrize('player', [False,True])
@pytest.mark.parametrize('names', [('front','rear'),('one','two','three')])
def test_all_namespaced_ack_targets_stay_exact_with_measured_drift(player,names):
    klass = BoardGameFingertipPolicy if player else BimanualFingertipPolicy
    policy = klass(**robot_configuration({'robot':robot(names)}))
    model = policy.geometry
    policy.reset(SimpleNamespace(episode_id='instances',observation_profile='vision_proprio'),
        SimpleNamespace(control_period_s=.02,joint_limits={j.name:JointLimit(lower_rad=j.lower,upper_rad=j.upper) for j in model.joints}))
    ack = {j.name:float(np.clip(0.,j.lower,j.upper)) for j in model.joints}
    for member in model.sides.values():
        ack.update(zip(member.arm.names,[.2,-.5,0.,-2.3,0.,1.8,.8]))
    ack = {n:float(np.float32(v)) for n,v in ack.items()}
    q = dict(ack)
    for member in model.sides.values(): q[member.arm.names[0]] -= .0001
    selected = list(policy.sides)[0]
    if player: policy.set_player_hand(selected,ack)
    policy.begin(hold(),observe(q,ack,episode='instances'))
    for sequence in range(3):
        result = policy.step(observe(q,ack,sequence,episode='instances'))
        assert result.command.joint_positions_rad == ack
        policy.commit_pending_step()
    assert policy.step(observe(q,ack,3,episode='instances')).status == 'completed'
    assert 'acknowledged_drive' in policy.descriptor.hold_modes


def test_composite_rejection_restores_preceding_instance_plan():
    policy = BimanualFingertipPolicy(**robot_configuration({'robot':robot()}))
    model = policy.geometry
    policy.reset(SimpleNamespace(episode_id='instances',observation_profile='vision_proprio'),
        SimpleNamespace(control_period_s=.02,joint_limits={j.name:JointLimit(lower_rad=j.lower,upper_rad=j.upper) for j in model.joints}))
    ack = {j.name:float(np.clip(0.,j.lower,j.upper)) for j in model.joints}
    rear = policy.sides['rear']
    rear.previous = np.array([ack[n] for n in rear.geometry.names])
    rear.velocity = np.full(len(rear.previous),.01)
    before = {side:state(member) for side,member in policy.sides.items()}
    with pytest.raises(GeometryIKError,match='stationary'):
        policy.begin(hold(),observe(ack,ack,episode='instances'))
    assert policy._active is None
    for side in policy.sides: same_state(before[side],state(policy.sides[side]))
