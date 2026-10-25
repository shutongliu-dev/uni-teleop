"""Real URDF/controller CPU checks; ideal tracking is not native qualification."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.control.bimanual import BimanualFingertipPolicy
from bench.paths import DATA_ROOT
from bench.protocol.schemas import AgentObservation, JointLimit, System0Action
from bench.runtime.board_game_control import BoardGameFingertipPolicy, VERSION


@pytest.fixture
def controlled():
    policy = BoardGameFingertipPolicy(f'{DATA_ROOT}/assets', [0., 0., .77])
    geometry = policy.geometry
    policy.reset(SimpleNamespace(episode_id='board', observation_profile='vision_proprio'),
        SimpleNamespace(control_period_s=.02, joint_limits={joint.name:
            JointLimit(lower_rad=joint.lower, upper_rad=joint.upper) for joint in geometry.joints}))
    q = {j.name: float(np.clip(0., j.lower, j.upper)) for j in geometry.joints}
    for side, sign in [('left', -1), ('right', 1)]:
        q.update(zip(geometry.sides[side].arm.names, [sign*.4, -.5, 0., -2.3, 0., 1.8, sign*.8]))
    state = dict(q=dict(q), targets=dict(q), seq=0)
    def observation():
        return AgentObservation(episode_id='board', sequence=state['seq'], sim_time_s=.02*state['seq'],
            instruction='Play', profile='vision_proprio', cameras=[], proprioception={
                'joint_positions_rad': state['q'], 'joint_velocities_rad_s': dict.fromkeys(q, 0.),
                'commanded_joint_positions_rad': state['targets']})
    def ack(result, owned):
        assert result.status == 'running'
        state['targets'] = dict(result.command.joint_positions_rad)
        state['q'].update({n: state['targets'][n] for n in geometry.sides[owned].names})
        state['seq'] += 1
        policy.commit_pending_step()
    return policy, state, observation, ack


def hold(state, name='hold', duration=.10):
    return System0Action(action_id=name, observation_sequence=state['seq'], skill='hold',
        timeout_sim_s=duration, hold_duration_s=duration)


def forbidden(*args, **kwargs):
    raise AssertionError('Inactive executor must never be called')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_inactive_drift_never_advances_peer_executor_or_changes_its_drive(controlled, monkeypatch, side):
    policy, state, obs, ack = controlled
    peer = 'right' if side == 'left' else 'left'
    peer_names = policy.geometry.sides[peer].names
    frozen = {n: state['targets'][n] for n in peer_names}
    state['q'][policy.geometry.sides[peer].arm.names[0]] += .03
    state['q'][f'{peer}_index_PIP'] += .08
    for method in ('begin', 'step', 'commit_pending_step', 'discard_pending_step', 'cancel', 'execution_state'):
        monkeypatch.setattr(policy.sides[peer], method, forbidden)
    policy.set_player_hand(side, state['targets'])
    policy.begin(hold(state), obs())
    for _ in range(6):
        step = policy.step(obs())
        if step.status != 'running':
            break
        assert {n: step.command.joint_positions_rad[n] for n in peer_names} == frozen
        assert step.robot_execution_state.maintained_joint_error_rad[f'{peer}_index_PIP'] == pytest.approx(.08)
        ack(step, side)
    assert step.status == 'completed'
    policy.execution_state(obs())
    policy.cancel('done')


@pytest.mark.parametrize('side', ['left', 'right'])
def test_owned_geometry_keeps_classical_motion_and_completes_with_peer_fixed(controlled, side):
    policy, state, obs, ack = controlled
    peer = 'right' if side == 'left' else 'left'
    peer_names = policy.geometry.sides[peer].names
    frozen = {n: state['targets'][n] for n in peer_names}
    own = policy.geometry.sides[side]
    q_goal = {n: state['q'][n] for n in own.names}
    q_goal[own.arm.names[0]] += .04
    target = own.public_geometry(q_goal)['wrist']
    before_fk = policy.geometry.public_hand_geometries(state['q'])
    policy.set_player_hand(side, state['targets'])
    action = System0Action(action_id='wrist', observation_sequence=state['seq'], skill='move_geometry',
        timeout_sim_s=3., hand_geometries={side: {'wrist': target, 'fingertip_frame_id': own.frame_id}})
    policy.begin(action, obs())
    assert policy.geometry.public_hand_geometries(state['q']) == before_fk
    for _ in range(151):
        step = policy.step(obs())
        if step.status != 'running':
            break
        assert {n: step.command.joint_positions_rad[n] for n in peer_names} == frozen
        ack(step, side)
    assert step.status == 'completed' and step.robot_execution_state.objective_status == 'reached'
    actual = own.public_geometry({n: state['q'][n] for n in own.names})['wrist']['position_m']
    assert np.linalg.norm(np.array(actual) - np.array(target['position_m'])) <= .003


def test_switch_players_only_after_ack_and_reuse_each_side_without_stale_clock(controlled):
    policy, state, obs, ack = controlled
    assert isinstance(policy, BimanualFingertipPolicy)
    assert policy.descriptor.version == VERSION
    for turn, side in enumerate(('left', 'right', 'left')):
        policy.set_player_hand(side, state['targets'])
        policy.begin(hold(state, str(turn)), obs())
        while True:
            step = policy.step(obs())
            if step.status != 'running':
                break
            with pytest.raises(ValueError, match='switch players'):
                policy.set_player_hand('right' if side == 'left' else 'left', state['targets'])
            ack(step, side)
        assert step.status == 'completed'
    assert state['seq'] == 15


def test_discarded_candidate_does_not_advance_reference_and_cancel_preserves_ack(controlled):
    policy, state, obs, ack = controlled
    policy.set_player_hand('left', state['targets'])
    policy.begin(hold(state), obs())
    previous = policy.sides['left'].previous.copy()
    first = policy.step(obs())
    policy.discard_pending_step()
    np.testing.assert_array_equal(policy.sides['left'].previous, previous)
    assert policy._acknowledged_targets == state['targets']
    repeated = policy.step(obs())
    assert repeated.command == first.command
    ack(repeated, 'left')
    policy.cancel('turn_changed')
    assert policy._pending_command is None and policy._active is None
    assert policy._acknowledged_targets == state['targets']
    policy.set_player_hand('right', state['targets'])
    policy.begin(hold(state), obs())
    ack(policy.step(obs()), 'right')
    np.testing.assert_array_equal(policy.sides['left'].velocity, 0.)


def test_wrong_owner_begin_and_unknown_ack_are_transactional(controlled):
    policy, state, obs, ack = controlled
    with pytest.raises(ValueError, match='Bind'):
        policy.begin(hold(state), obs())
    policy.set_player_hand('left', state['targets'])
    before = dict(policy._acknowledged_targets)
    for targets in ({}, {**state['targets'], 'unknown': 0.},
                    {**state['targets'], 'left_index_PIP': float('nan')}):
        with pytest.raises(ValueError):
            policy.set_player_hand('right', targets)
        assert policy._player_hand == 'left' and policy._acknowledged_targets == before
    action = System0Action(action_id='wrong', observation_sequence=0, skill='move_geometry',
        timeout_sim_s=1., hand_geometries={'right': {'hand_posture_rad': {'right_index_PIP': .1},
                                                   'fingertip_frame_id': 'right_hand_flange'}})
    with pytest.raises(ValueError, match='exactly'):
        policy.begin(action, obs())
    assert policy.sides['left'].previous is None and policy._active is None


def test_failed_ik_does_not_replace_initial_ack_or_mutate_peer(controlled):
    policy, state, obs, ack = controlled
    policy.set_player_hand('left', state['targets'])
    peer = deepcopy(policy._state(policy.sides['right']))
    action = System0Action(action_id='unreachable', observation_sequence=0, skill='move_geometry',
        timeout_sim_s=1., hand_geometries={'left': {
            'wrist': {'frame_id': 'world', 'position_m': [100., 0., 0.], 'quaternion_wxyz': [1., 0., 0., 0.]},
            'fingertip_frame_id': 'left_hand_flange'}})
    with pytest.raises(ValueError):
        policy.begin(action, obs())
    assert policy.sides['left'].previous is None and policy._active is None
    assert policy.sides['right'].previous is peer['previous'] is None
    np.testing.assert_array_equal(policy.sides['right'].integral, peer['integral'])
