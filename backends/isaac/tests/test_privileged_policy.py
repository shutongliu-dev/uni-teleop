"""Privileged planning boundaries and real parser/controller ingress, no physics."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from bench.protocol.schemas import AgentObservation, JointLimit, System0Action
from bench.control.fingertip import ClassicalFingertipPolicy
from bench.geometry import matrix_quaternion, rotvec_matrix
from scenes.manipulation import DEFAULT_TASK_RELEASE
from scenes.manipulation.privileged_policy import PrivilegedManipulationPolicy
from agent.model_client.geometry_actions import build_geometry_action, parse_geometry_action


@pytest.fixture
def inputs(request):
    task_id = getattr(request, 'param', '01_cube_push')
    root = Path(DEFAULT_TASK_RELEASE) / 'tasks' / task_id
    if not root.is_dir():
        pytest.skip('Pinned shared task and robot assets unavailable')
    scene = load_json(root / 'scene.json')
    task = load_json(root / 'task.json')
    controller = ClassicalFingertipPolicy(scene['robot']['assets_root'], scene['robot']['base_position_m'])
    geometry = controller.geometry
    q = {j.name: float(np.clip(0., j.lower, j.upper)) for j in geometry.joints}
    q.update(zip(geometry.arm.names, scene['robot']['arm_home_rad']))
    observation = {'episode_id': 'test', 'sequence': 0, 'sim_time_s': 0., 'profile': 'vision_proprio',
                   'instruction': 'Push the cube.', 'cameras': [], 'last_action_status': 'idle',
                   'proprioception': {'joint_positions_rad': q, 'joint_velocities_rad_s': dict.fromkeys(q, 0.),
                                      'geometry': geometry.public_geometry(q)}}
    obj = next(x for x in scene['objects'] if x['id'] == 'cube')
    private = {'sequence': 0, 'sim_time_s': 0., 'objects': {'cube': {
        'position_m': deepcopy(obj['position_m']), 'quaternion_wxyz': deepcopy(obj['quaternion_wxyz'])}}}
    return scene, task, observation, private, controller


def advance(observation, private, amount=.2):
    observation['sequence'] += round(amount / .02)
    observation['sim_time_s'] += amount
    private['sequence'] = observation['sequence']
    private['sim_time_s'] = observation['sim_time_s']
    observation['last_action_status'] = 'completed'


def test_emits_bounded_real_model_json_without_mutating_inputs(inputs):
    scene, task, obs, state, _ = inputs
    original = deepcopy((scene, task, obs, state))
    policy = PrivilegedManipulationPolicy(scene, task)
    decision = parse_geometry_action(policy.next_decision(obs, state))
    assert decision['action'] == 'move_geometry'
    assert decision['reason'] == 'privileged_push_pre_shape'
    assert all(np.linalg.norm(x) <= .01 for x in decision['fingertip_delta_m'].values())
    assert np.linalg.norm(decision['wrist_delta_m']) <= .05
    assert np.linalg.norm(decision['wrist_rotation_delta_rad']) <= .15
    assert (scene, task, obs, state) == original
    assert policy.diagnostic()['score_inputs'] is False
    json.dumps(policy.diagnostic(), allow_nan=False)


@pytest.mark.parametrize('inputs', ['01_cube_push', '02_cube_lift_hold', '03_cube_pick_place', '04_cube_grasp_hold'], indirect=True)
def test_first_decision_passes_actual_parser_and_classical_controller(inputs):
    scene, task, obs, state, controller = inputs
    strategy = PrivilegedManipulationPolicy(scene, task)
    decision = strategy.next_decision(obs, state)
    sha = 'a' * 64
    g = controller.geometry
    descriptor = controller.descriptor.model_dump(mode='json')
    caps = {'robot_id': 'sharpa_w01_right', 'robot_asset_sha256': sha,
            'arm_joints': g.arm.names, 'hand_joints': g.hand.names,
            'finger_names': ['thumb', 'index', 'middle', 'ring', 'little'],
            'joint_limits': {j.name: {'lower_rad': j.lower, 'upper_rad': j.upper} for j in g.joints},
            'tracks': ['direct_joints', 'system0'], 'system0': descriptor,
            'frame_ids': ['world', 'robot_base', 'right_hand_flange']}
    spec = {'episode_id': 'test', 'agent_id': 'privileged', 'agent_version': 'v1', 'agent_config_sha256': sha,
            'task_id': task['id'], 'task_release_sha256': sha, 'evaluator_sha256': sha,
            'robot_id': 'sharpa_w01_right', 'robot_asset_sha256': sha, 'physics_profile_sha256': sha,
            'seed': 0, 'track': 'system0', 'system0': descriptor, 'instruction': obs['instruction'],
            'budgets': {'max_actions': 128, 'max_model_calls': 1, 'max_action_sim_s': 5.}}
    action = build_geometry_action(decision, obs, caps, spec, action_id='privileged_0', expected_sequence=0)
    assert action['kind'] == 'system0' and action['object_effect'] == 'none' and action['target'] is None
    controller.reset(SimpleNamespace(episode_id='test', observation_profile='vision_proprio'),
                     SimpleNamespace(control_period_s=.02, arm_joints=g.arm.names, hand_joints=g.hand.names,
                         joint_limits={j.name: JointLimit(lower_rad=j.lower, upper_rad=j.upper) for j in g.joints}))
    public = AgentObservation.model_validate(obs)
    controller.begin(System0Action.model_validate(action), public)
    step = controller.step(public)
    assert step.status == 'running' and step.command is not None
    assert all(g.by_name[n].lower <= q <= g.by_name[n].upper for n, q in step.command.joint_positions_rad.items())
    controller.commit_pending_step()


def test_rotated_object_true_back_plane_and_historical_contact_height(inputs):
    scene, task, obs, state, _ = inputs
    state['objects']['cube']['quaternion_wxyz'] = matrix_quaternion(rotvec_matrix([0., 0., np.pi / 2])).tolist()
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    plan = policy.diagnostic()['plans'][0]
    assert plan['index_world_z_m'] == pytest.approx(.7736470318610988, abs=1e-7)
    # A quarter turn changes the box's width along y from 40 mm to 32 mm.
    assert plan['object_back_plane_xy_m'] == pytest.approx([.45, .1 - .016], abs=1e-8)


@pytest.mark.parametrize('change', [
    lambda state: state.update(sequence=1),
    lambda state: state.update(sim_time_s=.02),
    lambda state: state['objects']['cube'].update(position_m=[float('nan'), 0., 0.]),
    lambda state: state['objects']['cube'].update(quaternion_wxyz=[2., 0., 0., 0.]),
    lambda state: state['objects']['cube'].update(quaternion_wxyz=[0., 1., 0., 0.]),
])
def test_rejects_stale_nonfinite_or_overturned_privileged_state(inputs, change):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    change(state)
    with pytest.raises(ValueError):
        policy.next_decision(obs, state)
    assert policy.decisions == 0


def test_near_goal_releases_from_measured_wrist_before_vertical_withdrawal(inputs):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    policy.phase_index = 5
    phase = policy.phases[5]
    wrist = obs['proprioception']['geometry']['wrist']
    wrist['position_m'] = (phase['position'] - [0., .02, 0.]).tolist()
    wrist['quaternion_wxyz'] = matrix_quaternion(phase['rotation']).tolist()
    obs['proprioception']['geometry']['fingertip_positions_m'] = {f: v.tolist() for f, v in policy.tips.items()}
    state['objects']['cube']['position_m'][:2] = [.45, .18]
    decision = policy.next_decision(obs, state, {'status': 'completed', 'elapsed_sim_s': .2})
    assert decision['reason'] == 'privileged_push_unload'
    assert decision['wrist_delta_m'] == pytest.approx([0., -.008, 0.])
    assert policy.phases[7]['position'][2] == pytest.approx(wrist['position_m'][2] + .12)


def test_one_limit_recovery_hold_then_fail_closed_if_controller_still_rejects(inputs):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    failure = {'status': 'failed', 'diagnostic': {'code': 'invalid_robot_state'}, 'elapsed_sim_s': 0.}
    decision = policy.next_decision(obs, state, failure)
    assert decision['action'] == 'hold' and decision['duration_s'] == .5
    assert policy.next_decision(obs, state, failure)['action'] == 'stop'


def hold_recovery_boundary(inputs):
    """Use the real service's begin rejection and timed-hold feedback fields."""
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    obs['last_action_id'] = 'move_0'
    select_phase(policy, 'establish_contact_hold', obs)
    first = policy.next_decision(obs, state, {'action_id': 'move_0', 'status': 'completed',
        'sim_time_s': obs['sim_time_s'], 'elapsed_sim_s': .2,
        'termination_reason': 'controller_converged', 'diagnostic': None, 'episode_finished': False})
    assert first['duration_s'] == .4
    obs.update(last_action_id='phase_hold_1', last_action_status='failed')
    rejected = {'action_id': 'phase_hold_1', 'status': 'failed', 'elapsed_sim_s': 0.,
        'sim_time_s': obs['sim_time_s'], 'termination_reason': 'begin_rejected',
        'diagnostic': {'code': 'invalid_robot_state'}, 'episode_finished': False}
    recovery = policy.next_decision(obs, state, rejected)
    assert recovery == {'action': 'hold', 'duration_s': .5, 'reason': 'controller_limit_recovery'}
    assert policy.pending_hold is None and policy.phase_index == 6
    return policy, obs, state


def completed_recovery_ack(obs):
    obs.update(last_action_id='recovery_2', last_action_status='completed')
    return {'action_id': 'recovery_2', 'status': 'completed', 'elapsed_sim_s': .5,
        'sim_time_s': obs['sim_time_s'], 'termination_reason': 'time_hold_completed',
        'diagnostic': None, 'episode_finished': False}


@pytest.mark.parametrize('inputs', ['02_cube_lift_hold'], indirect=True)
def test_recovery_ack_reissues_failed_phase_hold_and_restarts_its_clock(inputs):
    policy, obs, state = hold_recovery_boundary(inputs)
    advance(obs, state, .5)
    restarted = policy.next_decision(obs, state, completed_recovery_ack(obs))
    assert restarted == {'action': 'hold', 'duration_s': .4,
                         'reason': 'privileged_lift_hold_establish_contact_hold'}
    assert policy.phase_index == 6 and policy.pending_recovery is None
    assert policy.pending_hold['start_sim_s'] == pytest.approx(.7)
    assert not any(row['phase'] == 'establish_contact_hold' for row in policy.trace)
    assert policy.trace[-1]['phase'] == 'controller_limit_recovery'
    advance(obs, state, .4)
    obs['last_action_id'] = 'phase_hold_retry_3'
    following = policy.next_decision(obs, state, {'action_id': 'phase_hold_retry_3', 'status': 'completed',
        'elapsed_sim_s': .4, 'sim_time_s': obs['sim_time_s'], 'termination_reason': 'time_hold_completed',
        'diagnostic': None, 'episode_finished': False})
    assert following['reason'] == 'privileged_lift_hold_maintain_contact_hold'
    assert following['duration_s'] == .6 and policy.phase_index == 7
    assert policy.trace[-1]['phase'] == 'establish_contact_hold'
    assert policy.trace[-1]['acknowledged_hold_s'] == .4


@pytest.mark.parametrize('inputs', ['02_cube_lift_hold'], indirect=True)
def test_recovery_elapsed_cannot_count_toward_restarted_hold(inputs):
    policy, obs, state = hold_recovery_boundary(inputs)
    advance(obs, state, .5)
    policy.next_decision(obs, state, completed_recovery_ack(obs))
    advance(obs, state, .2)
    obs['last_action_id'] = 'phase_hold_retry_3'
    decision = policy.next_decision(obs, state, {'action_id': 'phase_hold_retry_3', 'status': 'completed',
        'elapsed_sim_s': .2, 'sim_time_s': obs['sim_time_s'], 'termination_reason': 'task_terminated',
        'diagnostic': None, 'episode_finished': True})
    assert decision == {'action': 'stop', 'reason': 'establish_contact_hold_hold_not_completed'}
    assert policy.phase_index == 6


@pytest.mark.parametrize('inputs', ['02_cube_lift_hold'], indirect=True)
@pytest.mark.parametrize('mutation', ['short_hold', 'claimed_full_but_observed_short', 'nan_elapsed',
    'bool_elapsed', 'nan_ack_time', 'stale_sequence', 'wrong_action_id', 'replayed_failed_action_id',
    'missing_action_id', 'public_failed', 'recovery_failed', 'wrong_termination'])
def test_invalid_recovery_ack_stops_without_advancing_interrupted_phase(inputs, mutation):
    policy, obs, state = hold_recovery_boundary(inputs)
    old_sequence = obs['sequence']
    advance(obs, state, .2 if mutation in ('short_hold', 'claimed_full_but_observed_short') else .5)
    feedback = completed_recovery_ack(obs)
    if mutation == 'short_hold': feedback['elapsed_sim_s'] = .2
    if mutation == 'nan_elapsed': feedback['elapsed_sim_s'] = float('nan')
    if mutation == 'bool_elapsed': feedback['elapsed_sim_s'] = True
    if mutation == 'nan_ack_time': feedback['sim_time_s'] = float('nan')
    if mutation == 'stale_sequence':
        obs['sequence'] = state['sequence'] = old_sequence
    if mutation == 'wrong_action_id': feedback['action_id'] = 'unrelated_action'
    if mutation == 'replayed_failed_action_id':
        obs['last_action_id'] = feedback['action_id'] = 'phase_hold_1'
    if mutation == 'missing_action_id':
        obs['last_action_id'] = feedback['action_id'] = None
    if mutation == 'public_failed': obs['last_action_status'] = 'failed'
    if mutation == 'recovery_failed':
        feedback.update(status='failed', diagnostic={'code': 'convergence_timeout'})
    if mutation == 'wrong_termination': feedback['termination_reason'] = 'controller_converged'
    decision = policy.next_decision(obs, state, feedback)
    assert decision == {'action': 'stop', 'reason': 'controller_recovery_hold_not_completed'}
    assert policy.phase_index == 6 and policy.pending_hold is None
    assert not any(row['phase'] == 'establish_contact_hold' for row in policy.trace)


def test_no_phase_advance_without_completed_full_settle_hold(inputs):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    policy.phase_index = 8
    advance(obs, state)
    assert policy.next_decision(obs, state, {'status': 'completed'})['action'] == 'hold'
    advance(obs, state, .5)
    stop = policy.next_decision(obs, state, {'status': 'completed', 'elapsed_sim_s': .5})
    assert stop['reason'] == 'settle_hold_not_completed'


def test_does_not_read_private_evaluator_fields(inputs):
    scene, task, obs, state, _ = inputs
    decision = PrivilegedManipulationPolicy(scene, task).next_decision(obs, state)
    task['evaluation'] = {'success': 'forged'}
    state.update(score={'success': True}, evaluator={'done': True})
    assert PrivilegedManipulationPolicy(scene, task).next_decision(obs, state) == decision


def select_phase(policy, name, obs, posture=None):
    """Synthetic decision-boundary geometry, not a physical rollout."""
    policy.phase_index = next(i for i, p in enumerate(policy.phases) if p['name'] == name)
    policy.phase_attempts = 0
    phase = policy.phases[policy.phase_index]
    if 'position' in phase:
        obs['proprioception']['geometry']['wrist'].update(
            position_m=phase['position'].tolist(), quaternion_wxyz=matrix_quaternion(phase['rotation']).tolist())
    if posture is not None:
        obs['proprioception']['joint_positions_rad'].update(policy.hand_goals[posture])
        obs['proprioception']['geometry']['fingertip_positions_m'] = {
            f: value.tolist() for f, value in policy.grasp_tips[posture].items()}
    return phase


def contact_observation(obs):
    obs['tactile'] = {'status': 'available', 'sim_time_s': obs['sim_time_s'], 'fingers': {
        f: {'status': 'valid', 'rigid_contact_any': True} for f in ('thumb', 'index')}}


@pytest.mark.parametrize('inputs', ['03_cube_pick_place'], indirect=True)
@pytest.mark.parametrize('phase_name,posture,target', [('close', 'open', 'closed'), ('release', 'closed', 'open')])
def test_close_and_release_use_opposite_postures_with_four_mm_bound(inputs, phase_name, posture, target):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    select_phase(policy, phase_name, obs, posture)
    contact_observation(obs)
    original = deepcopy((obs, state))
    decision = parse_geometry_action(policy.next_decision(obs, state, {'status': 'completed'}))
    # Contact alone cannot finish a closing phase before any closing attempt.
    assert decision['action'] == 'move_geometry'
    assert decision['reason'].endswith(phase_name)
    assert max(np.linalg.norm(x) for x in decision['fingertip_delta_m'].values()) <= .004 + 1e-12
    for finger in ('thumb', 'index'):
        current = policy.grasp_tips[posture][finger]
        endpoint = policy.grasp_tips[target][finger]
        waypoint = current + decision['fingertip_delta_m'][finger]
        assert np.linalg.norm(endpoint - waypoint) < np.linalg.norm(endpoint - current)
    assert (obs, state) == original


@pytest.mark.parametrize('inputs', ['02_cube_lift_hold'], indirect=True)
def test_loaded_timeout_ends_close_without_squeezing_again_and_requires_real_hold_ack(inputs):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    select_phase(policy, 'close', obs, 'open')
    close = policy.next_decision(obs, state, {'status': 'completed'})
    assert close['reason'] == 'privileged_lift_hold_close'
    advance(obs, state, 5.)
    contact_observation(obs)
    failure = {'status': 'failed', 'diagnostic': {'code': 'convergence_timeout'}, 'elapsed_sim_s': 5.}
    decision = policy.next_decision(obs, state, failure)
    assert decision['action'] == 'hold' and decision['duration_s'] == .4
    trace = next(p for p in policy.trace if p['phase'] == 'close')
    assert trace['contact_proxy_phase_end'] is True and trace['last_primitive_status'] == 'failed'
    advance(obs, state, .4)
    contact_observation(obs)
    decision = policy.next_decision(obs, state, {'status': 'completed', 'elapsed_sim_s': .4})
    assert decision['action'] == 'hold' and decision['duration_s'] == .6
    advance(obs, state, .6)
    contact_observation(obs)
    decision = parse_geometry_action(policy.next_decision(obs, state, {'status': 'completed', 'elapsed_sim_s': .6}))
    assert decision['reason'] == 'privileged_lift_hold_lift'
    assert 'fingertip_delta_m' not in decision and 'fingertip_targets_m' not in decision
    assert decision['wrist_delta_m'] == pytest.approx([0., 0., .015])


@pytest.mark.parametrize('inputs', ['02_cube_lift_hold'], indirect=True)
@pytest.mark.parametrize('contact', [True, False])
def test_third_close_stall_with_contact_finishes_the_close_not_the_episode(inputs, contact):
    # mount_v2_baseline_01/sharpa_lift (1.1.6): the thumb stalled on the cube three times;
    # the index pad touched on the third, but the stall counter stopped the episode first.
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    select_phase(policy, 'close', obs, 'open')
    policy.next_decision(obs, state, {'status': 'completed'})
    stall = {'status': 'failed', 'diagnostic': {'code': 'tracking_stalled'}, 'elapsed_sim_s': 1.}
    for _ in range(2):
        advance(obs, state, 1.)
        assert policy.next_decision(obs, state, stall)['action'] == 'move_geometry'      # thumb alone: keep closing
    advance(obs, state, 1.)
    if contact:
        contact_observation(obs)
    decision = policy.next_decision(obs, state, stall)
    if contact:
        assert decision['action'] == 'hold' and decision['duration_s'] == .4
        assert next(p for p in policy.trace if p['phase'] == 'close')['contact_proxy_phase_end'] is True
    else:
        assert decision == {'action': 'stop', 'reason': 'repeated_controller_convergence_timeout'}


@pytest.mark.parametrize('inputs', ['02_cube_lift_hold'], indirect=True)
@pytest.mark.parametrize('contact_kind', ['missing', 'stale', 'unavailable', 'one_finger'])
def test_lift_requires_current_two_finger_contact_proxy(inputs, contact_kind):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    select_phase(policy, 'lift', obs)
    if contact_kind != 'missing':
        contact_observation(obs)
        if contact_kind == 'stale':
            obs['tactile']['sim_time_s'] -= .02
        elif contact_kind == 'unavailable':
            obs['tactile']['status'] = 'unavailable'
        else:
            obs['tactile']['fingers']['index']['rigid_contact_any'] = False
    decision = policy.next_decision(obs, state, {'status': 'completed'})
    assert decision['action'] == 'stop'
    assert decision['reason'] == 'thumb_index_contact_proxy_missing_before_lift'


@pytest.mark.parametrize('inputs', ['03_cube_pick_place'], indirect=True)
@pytest.mark.parametrize('phase_name', ['lift', 'transfer', 'lower'])
def test_carry_phases_preserve_hand_by_omitting_hand_commands(inputs, phase_name):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    select_phase(policy, phase_name, obs, 'closed')
    obs['proprioception']['geometry']['wrist']['position_m'][2] -= .02
    contact_observation(obs)
    decision = parse_geometry_action(policy.next_decision(obs, state, {'status': 'completed'}))
    assert decision['reason'].endswith(phase_name)
    assert 'fingertip_delta_m' not in decision
    assert 'fingertip_targets_m' not in decision
    assert decision['wrist_delta_m'] == pytest.approx([0., 0., .015])


@pytest.mark.parametrize('inputs', ['03_cube_pick_place'], indirect=True)
def test_recorded_load_tilt_preserves_existing_lift_hold_and_transfer(inputs):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    select_phase(policy, 'lift', obs, 'closed')
    contact_observation(obs)
    # Native place_01 decision 0058: both pads stayed in contact, but the
    # 0.363 rad load tilt previously failed an initial-planning-only check.
    state['objects']['cube'].update(
        position_m=[.4434684216976166, .09425622224807739, .891335129737854],
        quaternion_wxyz=[.9823896884918213, -.09059357643127441,
                         .1559666395187378, -.048759825527668])
    original = deepcopy(state)
    hold = policy.next_decision(obs, state, {'status': 'completed'})
    assert hold == {'action': 'hold', 'duration_s': 1.,
                    'reason': 'privileged_pick_place_lift_hold'}
    assert state == original
    advance(obs, state, 1.)
    decision = parse_geometry_action(policy.next_decision(
        obs, state, {'status': 'completed', 'elapsed_sim_s': 1.}))
    assert decision['reason'] == 'privileged_pick_place_transfer'
    assert 'fingertip_delta_m' not in decision
    assert np.linalg.norm(decision['wrist_delta_m']) <= .015 + 1e-12
    assert len(policy.plans) == 1


@pytest.mark.parametrize('inputs', ['01_cube_push', '03_cube_pick_place'], indirect=True)
def test_upright_prior_still_required_for_each_new_plan(inputs):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    state['objects']['cube']['quaternion_wxyz'] = matrix_quaternion(rotvec_matrix([.36, 0., 0.])).tolist()
    center, rotation, top = policy._object(obs, state)
    with pytest.raises(ValueError, match='upright manipulation planning envelope'):
        policy._plan(obs, center, rotation, top)
    assert len(policy.plans) == 1


@pytest.mark.parametrize('inputs', ['03_cube_pick_place'], indirect=True)
def test_existing_plan_still_rejects_invalid_object_measurements(inputs):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    state['objects']['cube']['quaternion_wxyz'] = [float('nan'), 0., 0., 0.]
    with pytest.raises(ValueError, match='finite coordinates'):
        policy.next_decision(obs, state, {'status': 'completed'})


@pytest.mark.parametrize('inputs', ['04_cube_grasp_hold'], indirect=True)
@pytest.mark.parametrize('observed_duration,reported_duration', [(0.2, .4), (.4, .2), (.4, float('nan'))])
def test_grasp_dwell_needs_both_full_elapsed_physics_time_and_completed_feedback(inputs, observed_duration, reported_duration):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    advance(obs, state)
    select_phase(policy, 'establish_contact_hold', obs)
    assert policy.next_decision(obs, state, {'status': 'completed'})['duration_s'] == .4
    advance(obs, state, observed_duration)
    decision = policy.next_decision(obs, state, {'status': 'completed', 'elapsed_sim_s': reported_duration})
    assert decision['action'] == 'stop'
    assert policy.phase_index == 6


@pytest.mark.parametrize('inputs', ['04_cube_grasp_hold'], indirect=True)
def test_grasp_plan_uses_true_pose_without_reading_success_and_never_lifts(inputs):
    scene, task, obs, state, _ = inputs
    state['objects']['cube']['position_m'][0] += .01
    state['objects']['cube']['quaternion_wxyz'] = matrix_quaternion(rotvec_matrix([0., 0., .2])).tolist()
    policy = PrivilegedManipulationPolicy(scene, task)
    first = policy.next_decision(obs, state)
    plan = policy.diagnostic()['plans'][0]
    assert plan['initial_object_position_m'][:2] == pytest.approx([.46, .1])
    assert plan['object_yaw_rad'] == pytest.approx(.2)
    assert not any(p['name'] in ('lift', 'transfer', 'lower', 'release') for p in policy.phases)
    # Establish (0.4 s) plus the public terminal hold with a 0.6 s margin.
    terminal = task['public_goal']['success_contract']['terminal']['hold_s']
    assert sum(p.get('hold', 0.) for p in policy.phases) == pytest.approx(1. + terminal)
    state.update(score={'success': True}, evaluator={'done': True})
    task['evaluation'] = {'success': 'forged'}
    assert PrivilegedManipulationPolicy(scene, task).next_decision(obs, state) == first
    json.dumps(policy.diagnostic(), allow_nan=False)


@pytest.mark.parametrize('inputs', ['02_cube_lift_hold'], indirect=True)
def test_grasp_approach_requests_tight_public_wrist_tolerances(inputs):
    scene, task, obs, state, _ = inputs
    policy = PrivilegedManipulationPolicy(scene, task)
    policy.next_decision(obs, state)
    precise = {p['name'] for p in policy.phases if p.get('precise')}
    assert precise == {'orient', 'above', 'descend'}
    advance(obs, state)
    phase = select_phase(policy, 'descend', obs)
    # Start the descend 10 mm above its endpoint so a wrist move is required.
    obs['proprioception']['geometry']['wrist']['position_m'] = (phase['position'] + [0., 0., .01]).tolist()
    decision = parse_geometry_action(policy.next_decision(obs, state, {'status': 'completed'}))
    assert decision['reason'].endswith('descend')
    assert decision['position_tolerance_m'] == .001 and decision['orientation_tolerance_rad'] == .01
    advance(obs, state)
    select_phase(policy, 'close', obs, 'open')
    decision = parse_geometry_action(policy.next_decision(obs, state, {'status': 'completed'}))
    assert decision['reason'].endswith('close') and 'position_tolerance_m' not in decision
