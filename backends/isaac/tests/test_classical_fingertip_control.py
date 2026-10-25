"""FK/IK, rejected geometry and bounded public-feedback control; no simulator."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import DATA_ROOT
from bench.protocol.schemas import AgentObservation, JointLimit, System0Action
from bench.control.fingertip import (
    ClassicalFingertipPolicy, ExecutorLimits, FINGERTIP_LINKS, Fr3SharpaGeometry, GeometryIKError,
    MEASURED_JOINT_LIMIT_TOLERANCE_RAD,
)
from bench.geometry import quaternion_matrix, rotation_distance, transform


ASSETS = Path(f'{DATA_ROOT}/assets')


@pytest.fixture(scope='module')
def geometry():
    if not (ASSETS / 'robots/franka/fr3/fr3_no_hand.absolute.urdf').is_file():
        pytest.skip('Pinned shared FR3 and SHARPA URDF assets are unavailable')
    return Fr3SharpaGeometry(ASSETS, [0., 0., .77])


def home(geometry):
    result = {joint.name: float(np.clip(0., joint.lower, joint.upper)) for joint in geometry.joints}
    result.update(dict(zip(geometry.arm.names, [0., -.785, 0., -2.356, 0., 1.57, .785])))
    return result


def observation(q, time=0., *, instruction='An arbitrary task unrelated to geometry.'):
    return AgentObservation(episode_id='public_test', sequence=1 + round(time / .02), sim_time_s=time,
        instruction=instruction, profile='vision_proprio',
        proprioception={'joint_positions_rad': q, 'joint_velocities_rad_s': {name: 0. for name in q}}, cameras=[])


def ready_policy(geometry, *, limits=None):
    policy = ClassicalFingertipPolicy(ASSETS, [0., 0., .77], limits=limits)
    capabilities = SimpleNamespace(control_period_s=.02,
        arm_joints=geometry.arm.names, hand_joints=geometry.hand.names,
        joint_limits={j.name: JointLimit(lower_rad=j.lower, upper_rad=j.upper) for j in geometry.joints})
    policy.reset(SimpleNamespace(episode_id='public_test', observation_profile='vision_proprio'), capabilities)
    return policy


def executed_step(policy, obs):
    result = policy.step(obs)
    if result.command is not None:
        policy.commit_pending_step()
    return result


def action(geometry, **kwargs):
    return System0Action(action_id='geometry1', observation_sequence=1, skill='move_geometry',
                         timeout_sim_s=kwargs.pop('timeout_sim_s', 5.), geometry=geometry, **kwargs)


def test_public_frames_match_robot_fk_and_actual_pinky_frame(geometry):
    q = home(geometry)
    public = geometry.public_geometry(q)
    wrist = public['wrist']
    expected = geometry.base @ geometry.arm.forward(q)['fr3_link8']
    np.testing.assert_allclose(wrist['position_m'], expected[:3, 3], atol=1e-12)
    np.testing.assert_allclose(quaternion_matrix(wrist['quaternion_wxyz']), expected[:3, :3], atol=1e-12)
    assert public['fingertip_frame_id'] == 'right_hand_flange'
    assert FINGERTIP_LINKS['little'] == 'right_pinky_fingertip'
    hand_fk = geometry.hand.forward(q)
    np.testing.assert_allclose(hand_fk['right_hand_C_MC'][:3, 3], [0., 0., .0295], atol=1e-12)
    for finger, link in FINGERTIP_LINKS.items():
        np.testing.assert_allclose(public['fingertip_positions_m'][finger], hand_fk[link][:3, 3], atol=1e-12)


@pytest.mark.parametrize('selected', [('thumb', 'index'), ('thumb', 'middle', 'little'), tuple(FINGERTIP_LINKS)])
def test_subset_finger_ik_reaches_fk_targets_and_preserves_other_joints(geometry, selected):
    initial = home(geometry)
    reference = dict(initial)
    reference['right_ring_PIP'] = .12
    desired = dict(reference)
    for finger in selected:
        for joint in geometry.finger_joints[finger]:
            desired[joint.name] = float(np.clip(reference[joint.name] + .15, joint.lower, joint.upper))
    target = geometry.public_geometry(desired)['fingertip_positions_m']
    solved, report = geometry.solve(initial, fingertip_positions_m={f: target[f] for f in selected},
                                    hand_posture_rad={'right_ring_PIP': .12}, fingertip_tolerance_m=.001)
    result = geometry.public_geometry(solved)['fingertip_positions_m']
    selected_joints = {j.name for f in selected for j in geometry.finger_joints[f]}
    for finger in selected:
        assert np.linalg.norm(np.asarray(result[finger]) - target[finger]) <= .001
        assert report['fingertip_errors_m'][finger] <= .001
    for name in set(geometry.names) - selected_joints:
        assert solved[name] == reference[name]
    for joint in geometry.joints:
        assert joint.lower <= solved[joint.name] <= joint.upper


def test_fk_preserves_world_base_rotation_and_ik_recovers_world_target(geometry):
    rotated = Fr3SharpaGeometry(ASSETS, [.1, -.2, .8], [np.sqrt(.5), 0., 0., np.sqrt(.5)])
    initial = home(rotated)
    desired = dict(initial)
    desired['fr3_joint1'] += .03
    desired['fr3_joint3'] += .02
    target = rotated.public_geometry(desired)['wrist']
    solved, _ = rotated.solve(initial, wrist=target, position_tolerance_m=.001, rotation_tolerance_rad=.01)
    actual = rotated.public_geometry(solved)['wrist']
    assert np.linalg.norm(np.asarray(actual['position_m']) - target['position_m']) <= .001
    assert rotation_distance(quaternion_matrix(actual['quaternion_wxyz']), quaternion_matrix(target['quaternion_wxyz'])) <= .01
    assert all(solved[name] == initial[name] for name in rotated.hand.names)


@pytest.mark.parametrize('finger', ['thumb', 'index', 'little'])
def test_unreachable_fingertip_fails_without_projecting_or_mutating_input(geometry, finger):
    initial = home(geometry)
    before = dict(initial)
    with pytest.raises(GeometryIKError, match='no verified IK solution'):
        geometry.solve(initial, fingertip_positions_m={finger: [4., 4., 4.]})
    assert initial == before


def test_unreachable_wrist_fails_instead_of_returning_best_effort_target(geometry):
    with pytest.raises(GeometryIKError, match='Wrist target'):
        geometry.solve(home(geometry), wrist={'frame_id':'world', 'position_m':[9., 9., 9.],
                                              'quaternion_wxyz':[1., 0., 0., 0.]})


@pytest.mark.parametrize('kwargs', [
    {'fingertip_positions_m': {'pinky':[.1, .1, .1]}},
    {'fingertip_positions_m': {'index':[np.nan, .1, .1]}},
    {'hand_posture_rad': {'fr3_joint1':0.}},
    {'hand_posture_rad': {'right_index_PIP':100.}},
    {'wrist': {'frame_id':'palm', 'position_m':[0., 0., 0.], 'quaternion_wxyz':[1., 0., 0., 0.]}},
])
def test_invalid_geometry_is_rejected(geometry, kwargs):
    with pytest.raises(ValueError):
        geometry.solve(home(geometry), **kwargs)


def test_classical_policy_is_untrained_public_and_does_not_need_per_step_images(geometry):
    policy = ready_policy(geometry)
    assert policy.descriptor.trained is False and policy.descriptor.weights_sha256 is None
    assert policy.descriptor.observation_profile == 'vision_proprio'
    assert policy.descriptor.camera_observation_mode == 'decision_only'
    assert policy.descriptor.supported_robot_ids == ['sharpa_w01_right']
    assert policy.descriptor.force_objectives is False
    assert policy.provenance['collision_planning'] is False


def test_policy_limits_joint_targets_and_requires_actual_feedback(geometry):
    policy = ready_policy(geometry)
    q = home(geometry)
    desired = dict(q)
    desired['fr3_joint1'] += .05
    for finger in ('thumb', 'index'):
        for joint in geometry.finger_joints[finger]:
            desired[joint.name] = float(np.clip(q[joint.name] + .1, joint.lower, joint.upper))
    targets = geometry.public_geometry(desired)
    policy.begin(action({'wrist': targets['wrist'],
        'fingertip_positions_m': {f: targets['fingertip_positions_m'][f] for f in ('thumb', 'index')}}), observation(q))
    previous = np.array([q[name] for name in geometry.names])
    previous_velocity = np.zeros(len(previous))
    result = None
    for index in range(251):
        result = executed_step(policy, observation(q, index * .02))
        if result.status != 'running':
            break
        assert result.command.duration_s == .02
        target = np.array([result.command.joint_positions_rad[name] for name in geometry.names])
        velocity = (target - previous) / .02
        assert np.all(np.abs(velocity) <= policy.max_velocity + 1e-10)
        assert np.all(np.abs(velocity - previous_velocity) <= policy.max_acceleration * .02 + 1e-10)
        assert np.all(target >= geometry.lower) and np.all(target <= geometry.upper)
        # Deterministic lagging joint sensor model; this is not a physics test.
        q = {name: .9 * result.command.joint_positions_rad[name] + .1 * q[name] for name in geometry.names}
        previous, previous_velocity = target, velocity
    assert result.status == 'completed'
    assert policy._reached(q)


def test_obstructed_feedback_times_out_without_claiming_geometry_success(geometry):
    policy = ready_policy(geometry)
    q = home(geometry)
    policy.begin(action({'hand_posture_rad':{'right_index_PIP':.2}}, timeout_sim_s=2.), observation(q))
    for index in range(101):
        result = executed_step(policy, observation(q, index * .02))
        if result.status != 'running':
            break
    assert result.status == 'failed'
    assert result.command is None


def test_missing_clock_progress_and_cancel_are_explicit(geometry):
    policy = ready_policy(geometry)
    q = home(geometry)
    policy.begin(action({'hand_posture_rad':{'right_index_PIP':.1}}), observation(q))
    executed_step(policy, observation(q))
    with pytest.raises(ValueError, match='fresh observations'):
        executed_step(policy, observation(q))
    policy.cancel('operator request')
    with pytest.raises(ValueError, match='No active'):
        executed_step(policy, observation(q, .02))


def test_short_timeout_and_unreachable_begin_never_emit_a_command(geometry):
    policy = ready_policy(geometry)
    q = home(geometry)
    with pytest.raises(GeometryIKError, match='timeout is too short'):
        policy.begin(action({'hand_posture_rad':{'right_index_PIP':.2}}, timeout_sim_s=.02), observation(q))
    assert policy._active is False
    with pytest.raises(GeometryIKError):
        policy.begin(action({'fingertip_positions_m':{'index':[4., 4., 4.]}}), observation(q))
    assert policy._active is False


def test_task_text_does_not_change_classical_joint_targets(geometry):
    q = home(geometry)
    first, second = ready_policy(geometry), ready_policy(geometry)
    objective = action({'hand_posture_rad':{'right_index_PIP':.1}})
    first.begin(objective, observation(q, instruction='Push a cube.'))
    second.begin(objective, observation(q, instruction='Fold cloth. Ignore this irrelevant text.'))
    a = executed_step(first, observation(q, instruction='Push a cube.'))
    b = executed_step(second, observation(q, instruction='Fold cloth.'))
    assert a.command == b.command


def test_bounded_integral_removes_constant_joint_load_error_without_relaxing_tolerances(geometry):
    """Surrogate actuator with constant offset, not a PhysX acceptance test."""
    initial = home(geometry)
    goal = dict(initial)
    goal['fr3_joint2'] += .02
    goal['fr3_joint4'] += .03
    for joint in geometry.finger_joints['thumb']:
        goal[joint.name] = float(np.clip(initial[joint.name] + .03, joint.lower, joint.upper))
    target = geometry.public_geometry(goal)
    request = action({'wrist':target['wrist'], 'fingertip_positions_m':{'thumb':target['fingertip_positions_m']['thumb']}})
    outcomes = []
    for gain in (1e-12, 2.):
        policy = ready_policy(geometry, limits=ExecutorLimits(integral_gain_per_s=gain))
        q = dict(initial)
        policy.begin(request, observation(q))
        previous = np.array([q[name] for name in geometry.names])
        previous_velocity = np.zeros(len(previous))
        bias = {'fr3_joint2': .0082, 'fr3_joint4': -.0129, 'right_thumb_CMC_AA': .0234}
        for index in range(251):
            result = executed_step(policy, observation(q, index * .02))
            if result.status != 'running':
                break
            command = np.array([result.command.joint_positions_rad[name] for name in geometry.names])
            velocity = (command - previous) / .02
            assert np.all(np.abs(velocity) <= policy.max_velocity + 1e-10)
            assert np.all(np.abs(velocity - previous_velocity) <= policy.max_acceleration * .02 + 1e-10)
            assert np.max(np.abs(policy.integral)) <= policy.limits.maximum_tracking_offset_rad
            assert policy.feedback_diagnostic['maximum_abs_desired_tracking_offset_rad'] <= .04 + 1e-12
            q = {name: .8 * (result.command.joint_positions_rad[name] + bias.get(name, 0.)) + .2*q[name]
                 for name in geometry.names}
            previous, previous_velocity = command, velocity
        outcomes.append(result.status)
        if gain == 2.:
            assert policy._reached(q)
            assert abs(policy.integral[geometry.names.index('fr3_joint4')]) > .002
            # Default wrist tolerance scales with the commanded displacement (cap 3 mm).
            assert .0005 <= policy.position_tolerance <= .003 and policy.limits.fingertip_tolerance_m == .0015
    assert outcomes == ['failed', 'completed']


def test_integral_does_not_wind_up_into_saturation_and_is_reset_per_episode(geometry):
    policy = ready_policy(geometry)
    q = home(geometry)
    policy.begin(action({'hand_posture_rad':{'right_index_PIP':.4}}, timeout_sim_s=5.), observation(q))
    integrals, blocked = [], []
    for index in range(251):
        result = executed_step(policy, observation(q, index * .02))
        integrals.append(policy.integral.copy())
        blocked.append(policy.feedback_diagnostic['blocked_integrator_joint_names'])
        if result.status != 'running':
            break
    # The obstructed joint cannot move: the move stalls well before its timeout.
    assert result.status == 'failed' and result.diagnostic.code == 'tracking_stalled'
    assert result.diagnostic.elapsed_sim_s < 5.
    assert np.max(np.abs(integrals)) <= .04
    np.testing.assert_array_equal(integrals[-1], integrals[-25])
    assert any('right_index_PIP' in names for names in blocked)
    # A retained nonzero actuator bias survives begin; it is not a task phase.
    policy.integral[geometry.names.index('fr3_joint4')] = .01
    retained = policy.integral.copy()
    policy.begin(action({'hand_posture_rad':{'right_index_PIP':.1}}), observation(q))
    np.testing.assert_array_equal(policy.integral, retained)
    caps = SimpleNamespace(control_period_s=.02, arm_joints=geometry.arm.names, hand_joints=geometry.hand.names,
        joint_limits={j.name:JointLimit(lower_rad=j.lower, upper_rad=j.upper) for j in geometry.joints})
    policy.reset(SimpleNamespace(episode_id='public_test', observation_profile='vision_proprio'), caps)
    np.testing.assert_array_equal(policy.integral, np.zeros(29))
    policy.integral[0] = .02
    policy.cancel('operator interrupted contact')
    # Cancel itself does not reset actuator compensation or command state.
    assert policy.integral[0] == .02


def test_episode_and_sequence_boundary_rejects_stale_plugin_observations(geometry):
    policy = ready_policy(geometry)
    q = home(geometry)
    request = action({'hand_posture_rad':{'right_index_PIP':.1}})
    wrong_episode = observation(q).model_copy(update={'episode_id':'other_episode'})
    with pytest.raises(ValueError, match='different episode'):
        policy.begin(request, wrong_episode)
    with pytest.raises(ValueError, match='stale observation sequence'):
        policy.begin(request.model_copy(update={'observation_sequence':0}), observation(q))
    policy.begin(request, observation(q))
    executed_step(policy, observation(q))
    with pytest.raises(ValueError, match='consecutive physics sequences'):
        executed_step(policy, observation(q, .02).model_copy(update={'sequence':1}))
    with pytest.raises(ValueError, match='different episode'):
        executed_step(policy, observation(q, .02).model_copy(update={'episode_id':'other_episode'}))


@pytest.mark.parametrize('side', ['lower', 'upper'])
@pytest.mark.parametrize('excursion', [8.678383665028377e-7, 1.e-5, 1.e-3, 1.8969556549564004e-3, .01, .0299])
def test_measured_excursion_inside_safety_band_only_projects_internal_reference(geometry, side, excursion):
    q = home(geometry)
    joint = geometry.by_name['right_pinky_PIP']
    boundary = joint.lower if side == 'lower' else joint.upper
    q[joint.name] = boundary + (-excursion if side == 'lower' else excursion)
    original = dict(q)
    solved, _ = geometry.solve(q)
    assert solved[joint.name] == boundary
    assert q == original
    policy = ready_policy(geometry)
    measured = observation(q)
    policy.begin(action({'wrist':geometry.public_geometry(q)['wrist']}), measured)
    index = geometry.names.index(joint.name)
    assert policy.initial[index] == policy.previous[index] == boundary
    emitted = executed_step(policy, measured)
    assert emitted.status == 'running'
    for name, target in emitted.command.joint_positions_rad.items():
        assert geometry.by_name[name].lower <= target <= geometry.by_name[name].upper
    assert measured.proprioception.joint_positions_rad[joint.name] == original[joint.name]
    assert policy.descriptor.version == '1.9.0'
    assert policy.provenance['measured_reference_limit_tolerance_rad'] == MEASURED_JOINT_LIMIT_TOLERANCE_RAD == .03


@pytest.mark.parametrize('side', ['lower', 'upper'])
@pytest.mark.parametrize('excursion', [.0300001, .05])
def test_excursions_beyond_the_safety_band_are_invalid_robot_state(geometry, side, excursion):
    q = home(geometry)
    joint = geometry.by_name['right_pinky_PIP']
    q[joint.name] = joint.lower-excursion if side == 'lower' else joint.upper+excursion
    with pytest.raises(GeometryIKError, match='joint-limit safety band'):
        geometry.solve(q)
    policy = ready_policy(geometry)
    with pytest.raises(GeometryIKError, match='joint-limit safety band'):
        policy.begin(action({'wrist':geometry.public_geometry(q)['wrist']}), observation(q))
    assert policy._active is False


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_explicit_requested_posture_is_strict_even_inside_measured_allowance(geometry, side):
    q = home(geometry)
    joint = geometry.by_name['right_pinky_PIP']
    target = joint.lower-1.e-9 if side == 'lower' else joint.upper+1.e-9
    with pytest.raises(ValueError, match='outside URDF limits'):
        geometry.solve(q, hand_posture_rad={joint.name:target})


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_measured_projection_rejects_nonfinite_values(geometry, bad):
    q = home(geometry)
    q['right_pinky_PIP'] = bad
    with pytest.raises(ValueError, match='finite'):
        geometry.measured_reference(q)


def test_recorded_zero_boundary_roundoff_emits_legal_target(geometry):
    policy = ready_policy(geometry)
    q = home(geometry)
    q['right_pinky_CMC'] = 0.0004986763815395534
    policy.begin(action({'hand_posture_rad':{'right_pinky_CMC':0.}}), observation(q))
    index = geometry.names.index('right_pinky_CMC')
    # Recorded predecessor from the first real action-mode failure. The desired
    # boundary is zero; x + ((0-x)/dt)*dt evaluates to -3.388e-21 in float64.
    policy.initial[index] = policy.goal[index] = 0.
    policy.previous[index] = 2.2488954921051628e-05
    policy.velocity[index] = 4.1088388691428053e-07
    previous = policy.previous.copy()
    old_velocity = policy.velocity.copy()
    result = executed_step(policy, observation(q))
    assert result.status == 'running'
    # The finite-response reference approaches this bound rather than jumping
    # to zero in one tick. It still must never emit the old negative roundoff.
    assert 0. <= result.command.joint_positions_rad['right_pinky_CMC'] < previous[index]
    target = np.array([result.command.joint_positions_rad[name] for name in geometry.names])
    velocity = (target-previous)/.02
    assert np.all(np.abs(velocity) <= policy.max_velocity+1.e-12)
    assert np.all(np.abs(velocity-old_velocity) <= policy.max_acceleration*.02+1.e-12)


@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_real_braking_overshoot_still_fails_instead_of_clipping(geometry, side):
    policy = ready_policy(geometry)
    q = home(geometry)
    policy.begin(action({'hand_posture_rad':{'right_pinky_CMC':0.}}), observation(q))
    index = geometry.names.index('right_pinky_CMC')
    if side == 'lower':
        policy.previous[index] = geometry.lower[index]+.0001
        policy.velocity[index] = -.2
    else:
        policy.previous[index] = geometry.upper[index]-.0001
        policy.velocity[index] = .2
    result = executed_step(policy, observation(q))
    assert result.status == 'failed' and result.command is None
    assert result.detail == 'Bounded target trajectory would cross a joint limit'


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_internal_command_is_rejected_before_emission(geometry, bad):
    policy = ready_policy(geometry)
    q = home(geometry)
    policy.begin(action({'hand_posture_rad':{'right_pinky_CMC':0.}}), observation(q))
    policy.previous[geometry.names.index('right_pinky_CMC')] = bad
    with np.errstate(invalid='ignore'):
        result = executed_step(policy, observation(q))
    assert result.status == 'failed' and result.command is None
    assert result.detail == 'Bounded target trajectory is nonfinite'


def timed_hold(obs, duration):
    return System0Action(action_id='timed_hold', observation_sequence=obs.sequence,
                         skill='hold', hold_duration_s=duration, timeout_sim_s=duration)


def assert_continuous_command(policy, command, previous, velocity):
    current = np.array([command.joint_positions_rad[name] for name in policy.geometry.names])
    next_velocity = (current - previous) / .02
    assert np.all(current >= policy.geometry.lower) and np.all(current <= policy.geometry.upper)
    assert np.all(abs(next_velocity) <= policy.max_velocity + 1e-10)
    assert np.all(abs(next_velocity - velocity) <= policy.max_acceleration * .02 + 1e-10)
    policy.commit_pending_step()  # surrogate below actually executes this command
    return current, next_velocity


@pytest.mark.parametrize('duration', [.02, .2, 1.])
def test_identical_reference_hold_executes_exact_duration_with_no_early_completion(geometry, duration):
    policy, q = ready_policy(geometry), home(geometry)
    obs = observation(q)
    policy.begin(timed_hold(obs, duration), obs)
    commands = []
    for tick in range(round(duration / .02) + 1):
        result = executed_step(policy, observation(q, tick * .02))
        if tick * .02 < duration - 1e-9:
            assert result.status == 'running'
            assert result.command.joint_positions_rad == q
            commands.append(result.command)
        else:
            assert result.status == 'completed' and result.command is None
    assert len(commands) == round(duration / .02)
    assert sum(x.duration_s for x in commands) == pytest.approx(duration)
    assert 'hold' in policy.descriptor.skills


def test_loaded_move_to_hold_preserves_goal_compensation_and_command_continuity(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    desired = dict(q)
    desired['fr3_joint4'] += .03
    target = geometry.public_geometry(desired)
    policy.begin(action({'wrist':target['wrist']}), observation(q))
    previous = np.array([q[name] for name in geometry.names])
    velocity = np.zeros(29)
    time = 0.
    for _ in range(251):
        result = executed_step(policy, observation(q, time))
        if result.status != 'running':
            break
        previous, velocity = assert_continuous_command(policy, result.command, previous, velocity)
        q = {name: .8 * (result.command.joint_positions_rad[name] + (-.0129 if name == 'fr3_joint4' else 0.))
                   + .2 * q[name] for name in geometry.names}
        time += .02
    assert result.status == 'completed'
    assert np.all(abs(velocity) < 1e-10)
    assert abs(policy.integral[geometry.names.index('fr3_joint4')]) > .002
    goal, integral, reference = policy.goal.copy(), policy.integral.copy(), policy.previous.copy()
    obs = observation(q, time)
    assert np.max(abs(reference - np.array([q[name] for name in geometry.names]))) > .01
    policy.begin(timed_hold(obs, .4), obs)
    np.testing.assert_array_equal(policy.goal, goal)
    np.testing.assert_array_equal(policy.integral, integral)
    np.testing.assert_array_equal(policy.previous, reference)
    for tick in range(21):
        result = executed_step(policy, observation(q, time + tick * .02))
        if tick == 20:
            assert result.status == 'completed'
        else:
            assert result.status == 'running'
            previous, velocity = assert_continuous_command(policy, result.command, previous, velocity)
            q = {name: .8 * (result.command.joint_positions_rad[name] + (-.0129 if name == 'fr3_joint4' else 0.))
                       + .2 * q[name] for name in geometry.names}
    assert policy._reached(q)
    assert np.all(abs(velocity) < 1e-10)


def test_cancel_brakes_continuously_then_discards_failed_goal_integral(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    policy.begin(action({'hand_posture_rad':{'right_index_PIP':.8}}), observation(q))
    previous = np.array([q[name] for name in geometry.names])
    velocity = np.zeros(29)
    for tick in range(18):
        result = executed_step(policy, observation(q, tick * .02))
        assert result.status == 'running'
        previous, velocity = assert_continuous_command(policy, result.command, previous, velocity)
        # Contact-like tracking lag; does not claim native contact validation.
        q = {name: .3 * result.command.joint_positions_rad[name] + .7 * q[name] for name in geometry.names}
    assert max(abs(velocity)) > .1
    policy.integral[geometry.names.index('right_ring_PIP')] = -.03
    integral = policy.integral.copy()
    policy.cancel('explicit operator cancel')
    np.testing.assert_array_equal(policy.previous, previous)
    np.testing.assert_allclose(policy.velocity, velocity, atol=1e-12)
    np.testing.assert_array_equal(policy.integral, integral)
    obs = observation(q, .36)
    # Rejection must not erase the reference or silently extend a short hold.
    with pytest.raises(GeometryIKError) as caught:
        policy.begin(timed_hold(obs, .02), obs)
    assert caught.value.public_diagnostic.code == 'insufficient_duration'
    np.testing.assert_array_equal(policy.previous, previous)
    np.testing.assert_array_equal(policy.integral, integral)
    assert not policy._active
    policy.begin(timed_hold(obs, .4), obs)
    stop_tick = policy._recovery_steps
    safe_goal = policy.goal.copy()
    assert stop_tick > 0
    for tick in range(21):
        result = executed_step(policy, observation(q, .36 + tick * .02))
        if tick == 20:
            assert result.status == 'completed'
        else:
            assert result.status == 'running'
            previous, velocity = assert_continuous_command(policy, result.command, previous, velocity)
            q = dict(result.command.joint_positions_rad)
            if tick < stop_tick:
                np.testing.assert_array_equal(policy.integral, integral)
            if tick == stop_tick:
                # No old ring task error can push a wrist/hold-only recovery.
                assert policy.integral[geometry.names.index('right_ring_PIP')] == 0.
    np.testing.assert_array_equal(policy.goal, safe_goal)
    assert np.all(abs(velocity) < 1e-10)


@pytest.mark.parametrize('boundary', ['lower', 'upper'])
def test_obstructed_limit_goal_times_out_stationary_and_recovers_without_guard(geometry, boundary):
    policy, q = ready_policy(geometry), home(geometry)
    name = 'right_index_PIP'
    joint = geometry.by_name[name]
    q[name] = (joint.lower + joint.upper) / 2.
    policy.begin(action({'hand_posture_rad':{name:getattr(joint, boundary)}}), observation(q))
    previous = np.array([q[n] for n in geometry.names])
    velocity = np.zeros(29)
    for tick in range(251):
        result = executed_step(policy, observation(q, tick * .02))
        if result.status != 'running':
            break
        previous, velocity = assert_continuous_command(policy, result.command, previous, velocity)
        # The contact-blocked measurement stays fixed throughout the action.
    # A blocked objective ends as soon as it stalls, not at the 5 s timeout.
    assert result.status == 'failed' and result.diagnostic.code == 'tracking_stalled'
    stop = result.diagnostic.elapsed_sim_s
    assert stop < 5.
    assert result.diagnostic.joint_name == name
    assert np.all(abs(velocity) < 1e-10)
    obs = observation(q, stop)
    wrist_only = action({'wrist':geometry.public_geometry(q)['wrist']}).model_copy(
        update={'observation_sequence':obs.sequence})
    policy.begin(wrist_only, obs)
    np.testing.assert_array_equal(policy.previous, previous)
    for tick in range(251):
        result = executed_step(policy, observation(q, stop + tick * .02))
        if result.status != 'running':
            break
        previous, velocity = assert_continuous_command(policy, result.command, previous, velocity)
        q = dict(result.command.joint_positions_rad)
    assert result.status == 'completed'
    assert np.all(abs(velocity) < 1e-10)


def test_ik_rejection_reports_measured_residual_without_replacing_reference(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    obs = observation(q)
    policy.begin(timed_hold(obs, .02), obs)
    assert executed_step(policy, obs).status == 'running'
    assert executed_step(policy, observation(q, .02)).status == 'completed'
    reference, velocity, integral = policy.previous.copy(), policy.velocity.copy(), policy.integral.copy()
    obs = observation(q, .02)
    bad = action({'fingertip_positions_m':{'thumb':[4., 4., 4.]}}).model_copy(
        update={'observation_sequence':obs.sequence})
    with pytest.raises(GeometryIKError) as caught:
        policy.begin(bad, obs)
    assert caught.value.public_diagnostic.code == 'ik_solution_unverified'
    assert caught.value.public_diagnostic.fingertip_errors_m['thumb'] > 1.
    np.testing.assert_array_equal(policy.previous, reference)
    np.testing.assert_array_equal(policy.velocity, velocity)
    np.testing.assert_array_equal(policy.integral, integral)
    assert not policy._active


def test_unexecuted_candidate_is_discarded_before_cancel_and_recovery(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    policy.begin(action({'hand_posture_rad':{'right_index_PIP':.4}}), observation(q))
    first = policy.step(observation(q))
    policy.commit_pending_step()
    q = dict(first.command.joint_positions_rad)
    previous, velocity, integral = policy.previous.copy(), policy.velocity.copy(), policy.integral.copy()
    candidate = policy.step(observation(q, .02))
    assert candidate.status == 'running'
    assert not np.array_equal(policy.previous, previous)
    # An accepted cancellation races the candidate before native worker.step.
    policy.cancel('cancel before execution')
    np.testing.assert_array_equal(policy.previous, previous)
    np.testing.assert_array_equal(policy.velocity, velocity)
    np.testing.assert_array_equal(policy.integral, integral)
    obs = observation(q, .02)
    policy.begin(timed_hold(obs, .2), obs)
    next_step = policy.step(obs)
    assert_continuous_command(policy, next_step.command, previous, velocity)


def test_final_only_candidate_rolls_back_clock_reference_and_integral(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    obs = observation(q)
    policy.begin(action({'hand_posture_rad':{'right_index_PIP':.2}}), obs)
    previous, velocity, integral = policy.previous.copy(), policy.velocity.copy(), policy.integral.copy()
    candidate = policy.step(obs)
    assert candidate.status == 'running'
    policy.discard_pending_step()
    np.testing.assert_array_equal(policy.previous, previous)
    np.testing.assert_array_equal(policy.velocity, velocity)
    np.testing.assert_array_equal(policy.integral, integral)
    # The same observation can execute after explicit non-execution rollback.
    repeated = policy.step(obs)
    assert repeated.command == candidate.command
    policy.commit_pending_step()


def test_hold_rejects_unadvanced_sequence_with_advanced_first_clock(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    obs = observation(q)
    policy.begin(timed_hold(obs, .2), obs)
    with pytest.raises(ValueError, match='spaced by one control period'):
        policy.step(obs.model_copy(update={'sim_time_s':.02}))


def test_unacknowledged_command_cannot_be_assumed_executed_from_new_observation(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    obs = observation(q)
    policy.begin(timed_hold(obs, .2), obs)
    policy.step(obs)
    with pytest.raises(ValueError, match='Commit or discard'):
        policy.step(observation(q, .02))
    assert policy._pending_step_state is not None
    policy.discard_pending_step()


@pytest.mark.parametrize('gap', [.04, .019])
def test_prior_keeps_recorded_grasp_endpoints_geometrically_verified_and_natural(geometry, gap):
    measured = home(geometry)
    target = {'thumb':[.07, .05 + gap, .15], 'index':[.07, .05 - gap, .15]}
    solved, report = geometry.solve(measured, fingertip_positions_m=target, fingertip_tolerance_m=.00075)
    assert solved['right_index_DIP'] - solved['right_index_PIP'] <= .5
    actual = geometry.public_geometry(solved)['fingertip_positions_m']
    for finger, point in target.items():
        assert np.linalg.norm(np.asarray(actual[finger]) - point) <= .00075
        assert report['fingertip_errors_m'][finger] <= .00075
    assert measured == home(geometry)


def test_every_ik_seed_and_trial_is_feasible_before_fk(geometry, monkeypatch):
    measured = home(geometry)
    measured['right_index_DIP'], measured['right_index_PIP'] = 1.37, .68
    calls = []
    original = geometry.hand.forward
    def checked_forward(q):
        assert q['right_index_DIP'] - q['right_index_PIP'] <= .5
        for joint in geometry.finger_joints['index']:
            assert joint.lower <= q[joint.name] <= joint.upper
        calls.append(dict(q))
        return original(q)
    monkeypatch.setattr(geometry.hand, 'forward', checked_forward)
    solved, _ = geometry.solve(measured, fingertip_positions_m={'index':[.07, .031, .15]},
                               fingertip_tolerance_m=.00075)
    assert len(calls) > 1
    assert solved['right_index_DIP'] - solved['right_index_PIP'] <= .5
    assert measured['right_index_DIP'] == 1.37


@pytest.mark.parametrize('posture', [
    {'right_index_DIP':.8},
    {'right_index_DIP':1.2, 'right_index_PIP':.6},
])
def test_explicit_unnatural_posture_is_rejected_without_silent_projection(geometry, posture):
    measured = home(geometry)
    original = dict(measured)
    with pytest.raises(GeometryIKError) as caught:
        geometry.solve(measured, hand_posture_rad=posture)
    diagnostic = caught.value.public_diagnostic
    assert diagnostic.code == 'posture_constraint_violation'
    assert diagnostic.joint_name == 'right_index_DIP' and diagnostic.joint_error_rad > 0
    assert measured == original


def test_untargeted_unnatural_finger_and_timed_hold_are_not_silently_repaired(geometry):
    measured = home(geometry)
    measured['right_index_DIP'], measured['right_index_PIP'] = 1.2, .1
    solved, _ = geometry.solve(measured, wrist=geometry.public_geometry(measured)['wrist'])
    for name in geometry.hand.names:
        assert solved[name] == measured[name]
    policy = ready_policy(geometry)
    obs = observation(measured)
    policy.begin(timed_hold(obs, .04), obs)
    for tick in range(2):
        result = executed_step(policy, observation(measured, tick * .02))
        assert result.command.joint_positions_rad == measured
    profile = policy.descriptor.robot_posture_profile
    assert profile.scope == 'requested_finger_goals_only'
    assert profile.measured_state_projected is False and profile.unselected_fingers == 'preserve'
    assert profile.grasp_success_guarantee is False
    assert profile.max_distal_minus_proximal_rad == .5
    assert len(profile.joint_pairs) == 4


def test_noncompliant_selected_finger_recovers_continuously_not_by_measurement_clip(geometry):
    policy, measured = ready_policy(geometry), home(geometry)
    measured['right_index_DIP'], measured['right_index_PIP'] = 1.37, .68
    obs = observation(measured)
    policy.begin(action({'fingertip_positions_m':{'index':[.07, .031, .15]}}), obs)
    assert policy.goal[geometry.names.index('right_index_DIP')] - policy.goal[geometry.names.index('right_index_PIP')] <= .5
    original = np.array([measured[name] for name in geometry.names])
    np.testing.assert_array_equal(policy.previous, original)
    velocity = np.zeros(29)
    result = executed_step(policy, obs)
    previous, velocity = assert_continuous_command(policy, result.command, original, velocity)
    assert result.command.joint_positions_rad['right_index_DIP'] - result.command.joint_positions_rad['right_index_PIP'] > .5
    assert obs.proprioception.joint_positions_rad['right_index_DIP'] == 1.37
    for tick in range(1, 251):
        measured = dict(zip(geometry.names, previous.tolist()))
        result = executed_step(policy, observation(measured, tick * .02))
        if result.status != 'running':
            break
        previous, velocity = assert_continuous_command(policy, result.command, previous, velocity)
    assert result.status == 'completed'
    assert measured['right_index_DIP'] - measured['right_index_PIP'] <= .5 + .03


def test_entered_stop_finishes_before_resuming_after_measured_pose_drift(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    desired = dict(q)
    desired['right_index_PIP'] = .1
    policy.begin(action({'hand_posture_rad': {'right_index_PIP': .1}}), observation(q))
    # A real observation can enter tolerance while its submitted reference is
    # still moving. Reproduce this controller boundary without a simulator.
    index = geometry.names.index('right_index_PIP')
    policy.previous[index] = .1
    policy.velocity[index] = .3
    policy.duration = .02
    first = executed_step(policy, observation(desired))
    assert first.status == 'running'
    second = executed_step(policy, observation(desired, .02))
    assert second.status == 'running' and policy._settling
    old_velocity = policy.velocity.copy()
    previous = policy.previous.copy()
    # Loaded measured joint moves outside tolerance during the stop. Continue
    # braking until zero; a fresh pursuit may begin only from stationary qref.
    drifted = dict(desired)
    drifted['right_index_PIP'] = -.0
    for tick in range(2, 30):
        assert np.max(abs(old_velocity)) > 1e-10
        result = executed_step(policy, observation(drifted, tick * .02))
        assert result.status == 'running'
        assert policy.feedback_diagnostic['reference_phase'] == 'settling_stop'
        previous, velocity = assert_continuous_command(policy, result.command, previous, old_velocity)
        assert np.all(abs(velocity) <= abs(old_velocity) + 1e-10)
        old_velocity = velocity
        if np.max(abs(velocity)) <= 1e-10:
            break
    assert np.max(abs(old_velocity)) <= 1e-10
    resumed = executed_step(policy, observation(drifted, (tick + 1) * .02))
    assert resumed.status == 'running'
    assert policy.feedback_diagnostic['reference_phase'] == 'tracking'
    assert_continuous_command(policy, resumed.command, previous, old_velocity)


@pytest.mark.parametrize('response_gain', [8., 1.e6])
def test_loaded_inertial_wrist_reversals_preserve_bounds_and_reach_targets(geometry, response_gain):
    """Local empirical CPU plant, not native/physics/task-success acceptance.

    Joint4 coefficients fit 245 executed commands of boundary_01's other nine
    actions, excluding both failing actions. Held-out open-loop q RMSE was
    .279/.345 mrad. Evidence: development/bench_capability_repair_20261012T054520Z/
    native_analysis/boundary_tracking_diagnostic/mechanism.json. Other joints
    use a first-order sensor/actuator lag, so this is deliberately not a full
    reconstructed robot. Both configured and effectively one-tick reference
    response exercise the current nominal-goal handoff. High gain alone no
    longer reconstructs the old drive-as-nominal double-bias limit cycle.
    """
    coefficients = np.array([
        [.8612315885536982, -9.829714370716596],
        [.009561716621750272, .17044628547223797],
        [.13641925344100964, 9.805368781220588],
        [-.0018333393440899777, -.12113895473733882],
    ])
    policy = ready_policy(geometry, limits=ExecutorLimits(reference_tracking_gain_per_s=response_gain))
    q = home(geometry)
    origin = q['fr3_joint4'] = -2.299999952316284
    state, time = np.zeros(2), 0.
    center = geometry.public_geometry(q)['wrist']
    previous, velocity = np.array([q[n] for n in geometry.names]), np.zeros(29)
    failures, successful_settling_entries = 0, []
    for offset in (.01, -.01, 0., .01, -.01, 0.):
        target = dict(center, position_m=list(center['position_m']))
        target['position_m'][0] += offset
        obs = observation(q, time)
        request = action({'wrist':target}).model_copy(update={'observation_sequence':obs.sequence})
        policy.begin(request, obs)
        np.testing.assert_array_equal(policy.previous, previous)
        phase, entries = None, 0
        for tick in range(251):
            result = executed_step(policy, observation(q, time))
            if result.status != 'running':
                break
            current_phase = policy.feedback_diagnostic['reference_phase']
            entries += current_phase == 'settling_stop' and phase != current_phase
            phase = current_phase
            previous, velocity = assert_continuous_command(policy, result.command, previous, velocity)
            cmd = result.command.joint_positions_rad
            state = state @ coefficients[:2] + (cmd['fr3_joint4'] - origin) * coefficients[2] + coefficients[3]
            q = {name:.8 * cmd[name] + .2 * q[name] for name in geometry.names}
            q['fr3_joint4'] = origin + state[0]
            time += .02
        assert np.max(abs(velocity)) < 1e-10
        if result.status == 'failed':
            failures += 1
            assert result.diagnostic.code == 'convergence_timeout'
            assert tick == 250
        else:
            assert result.status == 'completed' and policy._reached(q)
            successful_settling_entries.append(entries)
    assert failures == 0
    if response_gain == 8.:
        # The settle gate may re-enter braking once the wrist has crept out of and
        # back into tolerance, but never oscillates in a limit cycle.
        assert all(1 <= entries <= 3 for entries in successful_settling_entries)


def _finish_ideal_primitive(policy, q, time, geometry_request, *, load_joint=None, load=.0):
    """CPU actuator surrogate; never a physical/contact success certificate."""
    obs = observation(q, time)
    request = action(geometry_request).model_copy(update={'observation_sequence': obs.sequence})
    policy.begin(request, obs)
    for _ in range(251):
        result = executed_step(policy, observation(q, time))
        if result.status != 'running':
            assert result.status == 'completed'
            return q, time
        q = {name: float(np.clip(.8 * (result.command.joint_positions_rad[name]
                         - (load if name == load_joint else 0.)) + .2 * q[name],
                         policy.geometry.by_name[name].lower, policy.geometry.by_name[name].upper))
             for name in q}
        time += .02
    raise AssertionError('CPU primitive did not complete')


def _controller_state_bytes(policy):
    import pickle
    # Include goal/has_goal/recovery and pending snapshots, not only q/v/PI.
    return pickle.dumps({k: v for k, v in policy.__dict__.items()
                         if k not in {'geometry', 'descriptor', 'provenance', 'limits'}})


def test_wrist_only_retains_loaded_hand_goal_and_committed_reference(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    name = 'right_index_PIP'; index = geometry.names.index(name)
    q, time = _finish_ideal_primitive(policy, q, 0., {'hand_posture_rad': {name: .35}},
                                     load_joint=name, load=.0129)
    old_goal, previous, velocity, integral = [x.copy() for x in
                                            (policy.goal, policy.previous, policy.velocity, policy.integral)]
    # The integral only accumulates at a stationary nominal, so it is smaller but present.
    assert abs(old_goal[index] - q[name]) > 1e-4 and abs(integral[index]) > 1e-4
    wrist = geometry.public_geometry(q)['wrist']; wrist['position_m'][0] += .006
    obs = observation(q, time)
    policy.begin(action({'wrist': wrist}).model_copy(update={'observation_sequence': obs.sequence}), obs)
    np.testing.assert_array_equal(policy.goal[7:], old_goal[7:])
    np.testing.assert_array_equal(policy.previous, previous)
    np.testing.assert_array_equal(policy.velocity, velocity)
    np.testing.assert_array_equal(policy.integral, integral)
    first = policy.step(obs)
    assert_continuous_command(policy, first.command, previous, velocity)
    assert policy.descriptor.hand_goal_semantics.model_dump() == policy.provenance['hand_goal_semantics']
    assert policy.descriptor.version == '1.9.0'


@pytest.mark.parametrize('selected_input', ['tip', 'posture'])
def test_selected_chain_uses_explicit_input_unselected_retains_and_wrist_is_measured(geometry, selected_input):
    policy, q = ready_policy(geometry), home(geometry)
    q, time = _finish_ideal_primitive(policy, q, 0., {'hand_posture_rad': {
        'right_index_PIP': .35, 'right_index_DIP': .2, 'right_ring_PIP': .25}})
    old_goal = dict(zip(geometry.names, policy.goal))
    measured = dict(q); measured['right_ring_PIP'] -= .02; measured['right_index_DIP'] -= .015
    measured['fr3_joint4'] += .01  # omitted wrist must follow current measurement
    if selected_input == 'posture':
        request = {'hand_posture_rad': {'right_index_PIP': .3}}
    else:
        selected_q = dict(measured); selected_q['right_index_PIP'] += .03
        request = {'fingertip_positions_m': {'index': geometry.public_geometry(selected_q)['fingertip_positions_m']['index']}}
    obs = observation(measured, time)
    policy.begin(action(request).model_copy(update={'observation_sequence': obs.sequence}), obs)
    goal = dict(zip(geometry.names, policy.goal))
    for finger, joints in geometry.finger_joints.items():
        if finger != 'index':
            for joint in joints:
                assert goal[joint.name] == old_goal[joint.name]
    assert goal['right_ring_PIP'] != measured['right_ring_PIP']
    for name in geometry.arm.names:
        assert goal[name] == measured[name]
    np.testing.assert_allclose(policy.requested_wrist[:3, 3], geometry.public_geometry(measured)['wrist']['position_m'])
    if selected_input == 'posture':
        assert goal['right_index_PIP'] == .3
        assert goal['right_index_DIP'] == measured['right_index_DIP']
    else:
        error = np.linalg.norm(np.array(geometry.public_geometry(goal)['fingertip_positions_m']['index'])
                               - request['fingertip_positions_m']['index'])
        assert error <= policy.limits.fingertip_tolerance_m * .5


def test_initial_wrist_only_uses_measured_hand_reference(geometry):
    q = home(geometry); q['right_middle_PIP'] = .2
    policy = ready_policy(geometry); obs = observation(q)
    policy.begin(action({'wrist': geometry.public_geometry(q)['wrist']}), obs)
    assert dict(zip(geometry.names, policy.goal)) == q


def test_cancelled_pending_wrist_only_uses_committed_stop_not_unexecuted_or_old_goal(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    policy.begin(action({'hand_posture_rad': {'right_index_PIP': .8}}), observation(q))
    for tick in range(18):
        result = executed_step(policy, observation(q, tick * .02))
        q = dict(result.command.joint_positions_rad)
    previous, velocity, integral = [x.copy() for x in (policy.previous, policy.velocity, policy.integral)]
    obs = observation(q, .36)
    pending = policy.step(obs)
    assert pending.command is not None and np.max(abs(policy.previous - previous)) > 1e-4
    active_state = _controller_state_bytes(policy)
    with pytest.raises(ValueError, match='active action'):
        policy.begin(action({'wrist': geometry.public_geometry(q)['wrist']}).model_copy(
            update={'observation_sequence': obs.sequence}), obs)
    assert _controller_state_bytes(policy) == active_state
    policy.cancel('pending command never executed')
    np.testing.assert_array_equal(policy.previous, previous)
    np.testing.assert_array_equal(policy.velocity, velocity)
    amax = policy.max_acceleration
    counts = np.ceil(np.abs(velocity) / (amax * .02) - 1e-10).astype(int)
    expected = previous.copy()
    for k in range(1, int(counts.max()) + 1):
        expected += .02 * np.sign(velocity) * np.maximum(abs(velocity) - k * amax * .02, 0.)
    policy.begin(action({'wrist': geometry.public_geometry(q)['wrist']}).model_copy(
        update={'observation_sequence': obs.sequence}), obs)
    np.testing.assert_allclose(policy.goal[7:], expected[7:], atol=1e-12)
    assert abs(policy.goal[geometry.names.index('right_index_PIP')] - .8) > .1
    np.testing.assert_array_equal(policy.integral, integral)
    first = policy.step(obs)
    assert_continuous_command(policy, first.command, previous, velocity)


def test_failed_then_rejected_requests_do_not_revive_failed_or_historic_grip(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    q, time = _finish_ideal_primitive(policy, q, 0., {'hand_posture_rad': {'right_index_PIP': .1}})
    obs = observation(q, time)
    policy.begin(action({'hand_posture_rad': {'right_index_PIP': .8}}).model_copy(
        update={'observation_sequence': obs.sequence}), obs)
    for tick in range(251):
        result = executed_step(policy, observation(q, time + tick * .02))
        if result.status != 'running':
            break  # frozen measured hand represents an obstruction, not native physics
    assert result.status == 'failed' and result.diagnostic.code == 'tracking_stalled'
    assert policy._recovery_required and not policy._active
    obs = observation(q, time + tick * .02)
    original = _controller_state_bytes(policy)
    for geometry_request, timeout in [({'fingertip_positions_m': {'index': [4., 4., 4.]}}, 5.),
                                      ({'hand_posture_rad': {'right_index_PIP': .4}}, .02)]:
        with pytest.raises(GeometryIKError):
            policy.begin(action(geometry_request, timeout_sim_s=timeout).model_copy(
                update={'observation_sequence': obs.sequence}), obs)
        assert _controller_state_bytes(policy) == original
    stopped = policy.previous.copy()
    assert np.max(abs(policy.velocity)) < 1e-10
    policy.begin(action({'wrist': geometry.public_geometry(q)['wrist']}).model_copy(
        update={'observation_sequence': obs.sequence}), obs)
    np.testing.assert_allclose(policy.goal[7:], stopped[7:], atol=1e-12)
    index = geometry.names.index('right_index_PIP')
    assert abs(policy.goal[index] - .8) > .02 and abs(policy.goal[index] - .1) > .1


def test_accepted_omitted_hand_goal_reports_load_without_blocking_wrist_completion(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    q, time = _finish_ideal_primitive(policy, q, 0., {'hand_posture_rad': {'right_index_PIP': .3}})
    q['right_index_PIP'] = .25
    obs = observation(q, time)
    policy.begin(action({'wrist': geometry.public_geometry(q)['wrist']}).model_copy(
        update={'observation_sequence': obs.sequence}), obs)
    for tick in range(251):
        result = executed_step(policy, observation(q, time + tick * .02))
        if result.status != 'running':
            break
    assert result.status == 'completed' and result.diagnostic is None
    state = result.robot_execution_state
    assert state.objective_status == 'reached' and state.reference_stationary is True
    assert state.completion_scope == 'requested_geometry'
    assert state.maintained_joint_error_rad['right_index_PIP'] == pytest.approx(.05)
    # Explicitly reselecting that loaded posture still requires its joint error.
    obs = observation(q, time + tick * .02)
    policy.begin(action({'hand_posture_rad': {'right_index_PIP': .3}}).model_copy(
        update={'observation_sequence': obs.sequence}), obs)
    assert not policy._reached(q)


def test_begin_final_fk_failure_is_transactional_for_all_control_state(geometry, monkeypatch):
    policy, q = ready_policy(geometry), home(geometry)
    q, time = _finish_ideal_primitive(policy, q, 0., {'hand_posture_rad': {'right_index_PIP': .3}})
    obs = observation(q, time); wrist = geometry.public_geometry(q)['wrist']
    state = _controller_state_bytes(policy)
    def broken_fk(_):
        raise ValueError('FK unavailable before plan commit')
    monkeypatch.setattr(policy.geometry, 'public_geometry', broken_fk)
    with pytest.raises(ValueError, match='FK unavailable'):
        policy.begin(action({'wrist': wrist}).model_copy(update={'observation_sequence': obs.sequence}), obs)
    assert _controller_state_bytes(policy) == state


def test_closest_ik_solution_within_one_millimetre_executes_and_reports_the_boundary(geometry):
    q = home(geometry)
    tip = np.array(geometry.public_geometry(q)['fingertip_positions_m']['index'])
    # Push the request beyond reach along the extended finger until the best
    # solution lies just inside, then just outside, the 1 mm acceptance.
    direction = tip - np.array(geometry.hand.forward(q)['right_index_DP'][:3, 3])
    direction /= np.linalg.norm(direction)
    policy = ready_policy(geometry)
    accepted = rejected = None
    for extra in np.arange(0., .01, .0002):
        request = action({'fingertip_positions_m': {'index': (tip + extra * direction).tolist()}})
        try:
            ready_policy(geometry).begin(request, observation(q))
            accepted = extra
        except GeometryIKError as error:
            rejected = error.public_diagnostic
            break
    assert accepted is not None and rejected is not None
    assert rejected.code == 'ik_solution_unverified'
    assert 'index' in rejected.best_found_fingertips_m
    residual = rejected.fingertip_errors_m['index']
    assert .001 < residual
    assert rejected.geometry_tolerances.fingertips_m['index'] == .001


def test_arm_integral_is_frozen_while_the_nominal_moves_and_hand_integral_is_not(geometry):
    policy, q = ready_policy(geometry), home(geometry)
    wrist = geometry.public_geometry(q)['wrist']; wrist['position_m'][0] += .03
    goal = dict(q)
    policy.begin(action({'wrist': wrist, 'hand_posture_rad': {'right_index_PIP': .3}}), observation(q))
    arm = [geometry.names.index(n) for n in geometry.arm.names]
    pip = geometry.names.index('right_index_PIP')
    moving = []
    for tick in range(10):
        result = executed_step(policy, observation(q, tick * .02))
        moving.append(policy.integral.copy())
        q = {name: .5 * result.command.joint_positions_rad[name] + .5 * q[name] for name in geometry.names}
    assert all(np.all(row[arm] == 0.) for row in moving)
    assert any(row[pip] != 0. for row in moving)
