"""Synthetic feedback checks for the cloth planner; not physical acceptance."""
from copy import deepcopy

import numpy as np
import pytest

from scenes.cloth.privileged_policy import (
    DEFAULT_CLOTH_TASK_RELEASE, TASK_ID, PrivilegedClothDragPolicy, HandCollisionPrior,
)
from bench.isaacsim.release import task_inputs
from agent.model_client.geometry_actions import parse_geometry_action, build_geometry_action
from test_model_geometry_actions import public_geometry_inputs


class GeometricPrior:
    sources = {'synthetic_geometry': 'not_native'}
    def measure(self, observation, *, wrist=None):
        wrist = wrist or observation['proprioception']['geometry']['wrist']
        low = np.asarray(wrist['position_m'], float)
        return low, float(low[2] + .0025)


@pytest.fixture
def inputs(public_geometry_inputs):
    _, scene, task, _ = task_inputs(DEFAULT_CLOTH_TASK_RELEASE, TASK_ID)
    observation, capabilities, spec = deepcopy(public_geometry_inputs)
    observation.update(sequence=0, sim_time_s=0., last_action_status='idle', last_action_id=None)
    observation['proprioception']['geometry']['wrist'].update(
        position_m=[.34, -.08, .7622], quaternion_wxyz=[1., 0., 0., 0.])
    spec['budgets'].update(sim_time_s=None, wall_time_s=None, max_actions=128)
    vertices = np.asarray(task['evaluation']['expected_reset_vertices_m'])
    oracle = {'sequence': 0, 'sim_time_s': 0., 'cloth': {
        'vertices_m': vertices.tolist(), 'velocities_m_s': np.zeros_like(vertices).tolist(),
        'triangles': task['evaluation']['triangles']}, 'objects': {}}
    return scene, task, observation, capabilities, spec, oracle


def roll(inputs, *, gain=.8, budget=128, retain_completion_residual=False):
    scene, task, obs, caps, spec, oracle = deepcopy(inputs)
    policy = PrivilegedClothDragPolicy(scene, task, prior=GeometricPrior())
    policy.configure_budget(budget)
    feedback, actions = None, []
    for index in range(budget + 1):
        original = deepcopy((obs, oracle, feedback))
        action = policy.next_decision(obs, oracle, feedback)
        assert (obs, oracle, feedback) == original
        parse_geometry_action(action)
        if action['action'] == 'stop':
            return policy, actions, obs, oracle
        build_geometry_action(action, obs, caps, spec, action_id=f'cloth_{index}',
                              expected_sequence=obs['sequence'], actions_used=index)
        actions.append(action)
        elapsed = action['duration_s'] if action['action'] == 'hold' else .2
        if action['action'] == 'move_geometry':
            delta = np.asarray(action['wrist_delta_m'])
            if retain_completion_residual:
                tolerance = action['position_tolerance_m']
                distance = np.linalg.norm(delta)
                assert distance > tolerance, 'Planner emitted motion already inside requested completion tolerance'
                delta = delta * (1 - .95*tolerance/distance)
            wrist = obs['proprioception']['geometry']['wrist']
            wrist['position_m'] = (np.asarray(wrist['position_m']) + delta).tolist()
            if action['reason'] == 'privileged_cloth_drag':
                vertices = np.asarray(oracle['cloth']['vertices_m'])
                vertices[:, :2] += gain * delta[:2]
                oracle['cloth']['vertices_m'] = vertices.tolist()
        if index == 0:
            vertices = np.asarray(oracle['cloth']['vertices_m'])
            vertices[:, 2] = .7515
            oracle['cloth']['vertices_m'] = vertices.tolist()
        obs.update(sequence=obs['sequence'] + round(elapsed / .02),
                   sim_time_s=round(obs['sim_time_s'] + elapsed, 8),
                   last_action_status='completed', last_action_id=f'cloth_{index}')
        oracle.update(sequence=obs['sequence'], sim_time_s=obs['sim_time_s'])
        feedback = {'action_id': obs['last_action_id'], 'status': 'completed',
                    'elapsed_sim_s': elapsed, 'sim_time_s': obs['sim_time_s']}
    pytest.fail('Planner exceeded its action budget without a bounded release')


def test_continuous_vertex_feedback_reaches_geometric_goal_and_releases_within128(inputs):
    policy, actions, _, _ = roll(inputs)
    assert len(actions) < 128
    assert policy.last_metrics['minimum_vertex_translation_y_m'] >= .055
    assert policy.last_metrics['centroid_goal_error_m'] <= .006
    assert policy.last_metrics['pad_table_gap_m'] >= .012
    assert policy.finished_reason == 'geometric_target_reached_independent_score_required'
    assert policy.pressure_count == policy.MAX_PRESSURE_STEPS
    dragging = [a for a in actions if a['reason'] == 'privileged_cloth_drag']
    assert len(dragging) > 10
    assert all(a['wrist_delta_m'][2] == 0. for a in dragging)
    assert any(np.linalg.norm(a['wrist_delta_m']) > .0015 for a in dragging)
    phases = [a['reason'] for a in actions]
    first, last = phases.index('privileged_cloth_drag'), len(phases) - 1 - phases[::-1].index('privileged_cloth_drag')
    assert set(phases[first:last + 1]) == {'privileged_cloth_drag'}
    assert actions[-1]['action'] == 'hold' and actions[-1]['duration_s'] == .4


def test_lost_actual_cloth_motion_releases_without_repeated_pressure(inputs):
    policy, actions, _, _ = roll(inputs, gain=0.)
    assert policy.finished_reason == 'cloth_traction_progress_lost'
    assert policy.drag_steps == 2 and policy.pressure_count == policy.MAX_PRESSURE_STEPS
    assert policy.last_metrics['pad_table_gap_m'] >= .012


@pytest.mark.parametrize('budget', [12, 32, 48])
def test_short_diagnostic_preserves_action_reserve_and_final_release(inputs, budget):
    policy, actions, _, _ = roll(inputs, budget=budget)
    assert len(actions) <= budget
    assert policy.finished_reason == 'diagnostic_action_budget_release_reserve'
    assert policy.last_metrics['pad_table_gap_m'] >= .012
    assert actions[-1]['action'] == 'hold'


@pytest.mark.parametrize('mutation', [
    lambda obs, oracle: oracle.update(sequence=1),
    lambda obs, oracle: oracle['cloth']['vertices_m'][0].__setitem__(0, float('nan')),
    lambda obs, oracle: oracle['cloth']['triangles'][0].__setitem__(0, 9),
    lambda obs, oracle: oracle['cloth']['vertices_m'][0].__setitem__(1, -.1),
    lambda obs, oracle: obs['proprioception']['geometry']['wrist']['position_m'].__setitem__(2, .752),
])
def test_invalid_oracle_or_clearance_cannot_authorize_motion(inputs, mutation):
    scene, task, obs, _, _, oracle = deepcopy(inputs)
    policy = PrivilegedClothDragPolicy(scene, task, prior=GeometricPrior())
    mutation(obs, oracle)
    with pytest.raises(ValueError):
        policy.next_decision(obs, oracle)


def test_real_robot_collision_prior_reads_original_reset_only(inputs):
    from bench.control.fingertip import Fr3SharpaGeometry
    from bench.runtime.controllers import robot_configuration
    scene, task, obs, _, _, oracle = deepcopy(inputs)
    geometry = Fr3SharpaGeometry(**robot_configuration(scene))
    q = task['initial_state']['joint_positions_rad']
    obs['proprioception']['joint_positions_rad'] = q
    obs['proprioception']['joint_velocities_rad_s'] = dict.fromkeys(q, 0.)
    obs['proprioception']['geometry'] = geometry.public_geometry(q)
    policy = PrivilegedClothDragPolicy(scene, task)
    action = policy.next_decision(obs, oracle)
    assert action['action'] == 'hold'
    assert len(policy.prior.sources) > 10
    assert policy.last_metrics['pad_table_gap_m'] > .012


@pytest.mark.parametrize('field,value', [
    ('sim_time_s', float('nan')), ('sim_time_s', float('inf')), ('sim_time_s', True),
    ('elapsed_sim_s', float('nan')), ('elapsed_sim_s', float('inf')), ('elapsed_sim_s', True),
    ('elapsed_sim_s', -.1), ('elapsed_sim_s', '.4'), ('action_id', None), ('action_id', ''),
    ('action_id', 'contains space'),
])
def test_invalid_ack_cannot_complete_hold_and_retry_cannot_skip_it(inputs, field, value):
    scene, task, obs, _, _, oracle = deepcopy(inputs)
    policy = PrivilegedClothDragPolicy(scene, task, prior=GeometricPrior())
    action = policy.next_decision(obs, oracle)
    assert action['action'] == 'hold' and action['duration_s'] == .4
    # A deliberately shortened actual hold must not be authorized by NaN.
    obs.update(sequence=1, sim_time_s=.02, last_action_id='ack', last_action_status='completed')
    oracle.update(sequence=1, sim_time_s=.02)
    feedback = {'action_id': 'ack', 'status': 'completed', 'sim_time_s': .02, 'elapsed_sim_s': .02}
    feedback[field] = value
    if field == 'action_id':
        obs['last_action_id'] = value
    for _ in range(2):
        with pytest.raises(ValueError):
            policy.next_decision(obs, oracle, feedback)
        assert policy.pending is not None and policy.actions == 1


@pytest.mark.parametrize('field,value', [('sim_time_s', float('inf')), ('sim_time_s', True), ('sequence', True), ('sequence', 0.)])
def test_matching_but_invalid_public_and_private_boundaries_reject(inputs, field, value):
    scene, task, obs, _, _, oracle = deepcopy(inputs)
    obs[field] = oracle[field] = value
    policy = PrivilegedClothDragPolicy(scene, task, prior=GeometricPrior())
    with pytest.raises(ValueError):
        policy.next_decision(obs, oracle)


def test_phase_precision_remains_inside_transition_radius_without_empty_moves(inputs):
    policy, actions, _, _ = roll(inputs, retain_completion_residual=True)
    assert len(actions) < 128
    assert policy.finished_reason == 'geometric_target_reached_independent_score_required'
    assert policy.last_metrics['pad_table_gap_m'] >= .012
    assert policy.pressure_count == policy.MAX_PRESSURE_STEPS
    assert policy.last_metrics['minimum_vertex_translation_y_m'] >= .055
    phases = {}
    for action in actions:
        if action['action'] == 'move_geometry':
            phases.setdefault(action['reason'], set()).add(action['position_tolerance_m'])
    assert phases['privileged_cloth_align'] == {.0005}
    assert phases['privileged_cloth_approach'] == {.00025, .00005}
    assert phases['privileged_cloth_pressure'] == {.000025}
    assert policy.ALIGN_HEIGHT_M - policy.ALIGN_PHASE_RADIUS_M > .012
    assert policy.ALIGN_COMPLETION_M < policy.ALIGN_PHASE_RADIUS_M
    assert policy.APPROACH_NEAR_COMPLETION_M < policy.APPROACH_PHASE_RADIUS_M


def test_48_action_diagnostic_releases_with_nonzero_completion_residual(inputs):
    policy, actions, _, _ = roll(inputs, budget=48, retain_completion_residual=True)
    assert len(actions) <= 48 and policy.last_metrics['pad_table_gap_m'] >= .012
    assert policy.finished_reason == 'diagnostic_action_budget_release_reserve'
    assert actions[-1]['action'] == 'hold' and actions[-1]['duration_s'] == .4


def pressure_boundary(inputs, *, gap, count=0):
    scene, task, obs, _, _, oracle = deepcopy(inputs)
    policy = PrivilegedClothDragPolicy(scene, task, prior=GeometricPrior())
    policy.next_decision(obs, oracle)
    obs.update(sequence=20, sim_time_s=.4, last_action_status='completed', last_action_id='initial')
    oracle.update(sequence=20, sim_time_s=.4)
    obs['proprioception']['geometry']['wrist']['position_m'][2] = .75 + gap
    policy.phase = 'pressure'
    policy.pressure_count = count
    policy.pressure_progress = [.0001] if count else []
    feedback = {'action_id': 'initial', 'status': 'completed', 'sim_time_s': .4, 'elapsed_sim_s': .4}
    return policy, obs, oracle, feedback


def acknowledge_pressure(obs, oracle, *, down, action_id):
    obs['proprioception']['geometry']['wrist']['position_m'][2] -= down
    obs.update(sequence=obs['sequence'] + 10, sim_time_s=round(obs['sim_time_s'] + .2, 8),
               last_action_status='completed', last_action_id=action_id)
    oracle.update(sequence=obs['sequence'], sim_time_s=obs['sim_time_s'])
    return {'action_id': action_id, 'status': 'completed', 'sim_time_s': obs['sim_time_s'], 'elapsed_sim_s': .2}


@pytest.mark.parametrize('remainder', [.000025 + 1e-10, .00004, .000099999999, .0001, .000100000001])
def test_pressure_clips_to_unchanged_floor_without_undersized_request(inputs, remainder):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0026 + remainder, count=2)
    before_z = obs['proprioception']['geometry']['wrist']['position_m'][2]
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_pressure'
    requested_down = -action['wrist_delta_m'][2]
    assert .000025 < requested_down <= .0001
    assert requested_down == pytest.approx(min(remainder, .0001), abs=1e-15)
    assert before_z - requested_down >= .7526 - 1e-15
    assert action['position_tolerance_m'] == .000025


@pytest.mark.parametrize('remaining', [0., .000024999999, .000025, .000025 + 1e-13])
def test_pressure_near_floor_does_not_emit_zero_or_below_tolerance_move(inputs, remaining):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0026 + remaining, count=2)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['action'] == 'hold' and action['reason'] == 'privileged_cloth_contact_settle'
    assert policy.pressure_count == 2


def test_pressure_count_limit_is_five_even_when_extra_depth_remains(inputs):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0038)
    emitted = []
    for step in range(15):
        action = policy.next_decision(obs, oracle, feedback)
        if action['reason'] == 'privileged_cloth_contact_settle':
            break
        emitted.append(action)
        if action['action'] == 'move_geometry':
            feedback = acknowledge_pressure(obs, oracle, down=-action['wrist_delta_m'][2], action_id=f'p{step}')
        else:
            assert action['duration_s'] == .08
            obs.update(sequence=obs['sequence']+4, sim_time_s=round(obs['sim_time_s']+.08,8), last_action_id=f'h{step}')
            oracle.update(sequence=obs['sequence'], sim_time_s=obs['sim_time_s'])
            feedback={'action_id':f'h{step}', 'status':'completed', 'sim_time_s':obs['sim_time_s'], 'elapsed_sim_s':.08}
    else:
        pytest.fail('Pressure phase failed to respect its finite step limit')
    pressure = [a for a in emitted if a['action'] == 'move_geometry']
    assert len(pressure) == policy.pressure_count == 5
    assert all(-a['wrist_delta_m'][2] <= .0001 for a in pressure)
    assert obs['proprioception']['geometry']['wrist']['position_m'][2] > .7532


@pytest.mark.parametrize('prior_steps', [0, 2, 4])
@pytest.mark.parametrize('actual_down', [0., -.00001, .000005])
def test_each_pressure_requires_real_forward_progress_before_any_further_pressure(inputs, prior_steps, actual_down):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0031, count=prior_steps)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_pressure'
    feedback = acknowledge_pressure(obs, oracle, down=actual_down, action_id='pressure')
    result = policy.next_decision(obs, oracle, feedback)
    assert policy.finished_reason == 'pressure_progress_unverified'
    assert result['reason'] == 'privileged_cloth_release' and result['wrist_delta_m'][2] > 0
    assert policy.pressure_count == prior_steps + 1


def test_completed_pressure_floor_overshoot_releases_with_unchanged_measurement(inputs):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.00268, count=4)
    requested = policy.next_decision(obs, oracle, feedback)
    assert requested['wrist_delta_m'][2] == pytest.approx(-.00008, abs=1e-15)
    feedback = acknowledge_pressure(obs, oracle, down=.00009, action_id='last_pressure')
    original = deepcopy((obs, oracle, feedback))
    result = policy.next_decision(obs, oracle, feedback)
    assert (obs, oracle, feedback) == original
    assert policy.finished_reason == 'pressure_endpoint_margin_lost'
    assert result['reason'] == 'privileged_cloth_release' and result['wrist_delta_m'][2] > 0
    assert policy.last_metrics['pad_table_gap_m'] == pytest.approx(.00259, abs=1e-15)


def test_pressure_hold_downward_drift_releases_before_contact_settle(inputs):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.00268, count=4)
    policy.next_decision(obs, oracle, feedback)
    feedback = acknowledge_pressure(obs, oracle, down=.00007, action_id='last_pressure')
    hold = policy.next_decision(obs, oracle, feedback)
    assert hold['action'] == 'hold' and hold['duration_s'] == .08
    obs['proprioception']['geometry']['wrist']['position_m'][2] = .75258
    obs.update(sequence=obs['sequence']+4, sim_time_s=round(obs['sim_time_s']+.08,8),last_action_id='hold')
    oracle.update(sequence=obs['sequence'],sim_time_s=obs['sim_time_s'])
    feedback={'action_id':'hold','status':'completed','sim_time_s':obs['sim_time_s'],'elapsed_sim_s':.08}
    result=policy.next_decision(obs,oracle,feedback)
    assert policy.finished_reason == 'pressure_endpoint_margin_lost'
    assert result['reason'] == 'privileged_cloth_release'


@pytest.mark.parametrize('phase',['pressure','pressure_hold','contact_settle','drag'])
@pytest.mark.parametrize('below_floor',[2e-12,.00001,.000299999])
def test_legal_measured_gap_below_endpoint_releases_in_every_contact_phase(inputs,phase,below_floor):
    policy,obs,oracle,feedback=pressure_boundary(inputs,gap=.0026-below_floor,count=2)
    policy.phase=phase
    result=policy.next_decision(obs,oracle,feedback)
    assert result['reason']=='privileged_cloth_release'
    assert policy.finished_reason=='pressure_endpoint_margin_lost'


@pytest.mark.parametrize('below_floor',[0.,5e-13])
def test_endpoint_roundoff_allowance_does_not_create_false_release(inputs,below_floor):
    policy,obs,oracle,feedback=pressure_boundary(inputs,gap=.0026-below_floor,count=2)
    policy.phase='drag'
    policy.drag_height_anchor_m = obs['proprioception']['geometry']['wrist']['position_m'][2]
    result=policy.next_decision(obs,oracle,feedback)
    assert result['reason']=='privileged_cloth_drag'
    assert policy.finished_reason is None


def test_existing_release_reason_is_preserved_below_endpoint(inputs):
    policy,obs,oracle,feedback=pressure_boundary(inputs,gap=.00259,count=2)
    policy.phase='release';policy.finished_reason='cloth_traction_progress_lost'
    result=policy.next_decision(obs,oracle,feedback)
    assert result['reason']=='privileged_cloth_release'
    assert policy.finished_reason=='cloth_traction_progress_lost'


def test_endpoint_release_does_not_bypass_raw_measured_abort_floor(inputs):
    policy,obs,oracle,feedback=pressure_boundary(inputs,gap=.002299,count=2)
    with pytest.raises(ValueError,match='Measured robot violates'):
        policy.next_decision(obs,oracle,feedback)
    assert policy.pending is not None


class CompliantPrior(GeometricPrior):
    """Synthetic hand compliance separates measured pad and wrist movement."""
    def measure(self, observation, *, wrist=None):
        wrist = wrist or observation['proprioception']['geometry']['wrist']
        low = np.asarray(wrist['position_m'], float).copy()
        low[2] += observation.get('synthetic_compliance_m', 0.)
        return low, float(low[2] + .0025)


def plateau_boundary(inputs, *, prior_steps=2, pad_down=.0000035, wrist_down=.000086,
                     gap=.0031, lateral_error=0.):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=gap, count=prior_steps)
    policy.prior = CompliantPrior()
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_pressure'
    feedback = acknowledge_pressure(obs, oracle, down=wrist_down, action_id='plateau_pressure')
    obs['synthetic_compliance_m'] = wrist_down - pad_down
    obs['proprioception']['geometry']['wrist']['position_m'][0] += lateral_error
    return policy, obs, oracle, feedback


def acknowledge_synthetic(action, obs, oracle, *, gain=0.):
    elapsed = action['duration_s'] if action['action'] == 'hold' else .2
    if action['action'] == 'move_geometry':
        delta = np.asarray(action['wrist_delta_m'])
        wrist = obs['proprioception']['geometry']['wrist']
        wrist['position_m'] = (np.asarray(wrist['position_m']) + delta).tolist()
        if action['reason'] == 'privileged_cloth_drag':
            vertices = np.asarray(oracle['cloth']['vertices_m'])
            vertices[:, :2] += gain * delta[:2]
            oracle['cloth']['vertices_m'] = vertices.tolist()
    sequence = obs['sequence'] + round(elapsed / .02)
    obs.update(sequence=sequence, sim_time_s=round(obs['sim_time_s']+elapsed, 8),
               last_action_id=f'synthetic_{sequence}', last_action_status='completed')
    oracle.update(sequence=sequence, sim_time_s=obs['sim_time_s'])
    return {'action_id': obs['last_action_id'], 'status': 'completed',
            'elapsed_sim_s': elapsed, 'sim_time_s': obs['sim_time_s']}


@pytest.mark.parametrize('pad_down', [0., .0000035, .0000099])
def test_reached_wrist_plateau_after_verified_pressure_settles_before_bounded_probe(inputs, pad_down):
    policy, obs, oracle, feedback = plateau_boundary(inputs, pad_down=pad_down)
    original = deepcopy((obs, oracle, feedback))
    action = policy.next_decision(obs, oracle, feedback)
    assert (obs, oracle, feedback) == original
    assert action['reason'] == 'privileged_cloth_contact_settle' and action['duration_s'] == .3
    assert policy.pressure_count == 3 and policy.pressure_plateau
    assert policy.diagnostic()['plateau_is_contact_certificate'] is False
    feedback = acknowledge_synthetic(action, obs, oracle)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_drag'
    assert np.linalg.norm(action['wrist_delta_m']) <= .001 + 1e-12
    assert action['wrist_delta_m'][2] == 0.


@pytest.mark.parametrize('kwargs', [
    {'prior_steps': 0}, {'pad_down': -.0000001}, {'wrist_down': 0.},
    {'wrist_down': .000005}, {'lateral_error': .000026}, {'wrist_down': .00005},
])
def test_plateau_without_prior_progress_or_reached_forward_wrist_releases(inputs, kwargs):
    policy, obs, oracle, feedback = plateau_boundary(inputs, **kwargs)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_release'
    assert policy.finished_reason == 'pressure_progress_unverified'
    assert not policy.plateau_probe_active


@pytest.mark.parametrize('mutation', [
    lambda f: f.update(status='failed'), lambda f: f.update(action_id='wrong'),
    lambda f: f.update(elapsed_sim_s=float('nan')), lambda f: f.update(sim_time_s=0.),
])
def test_plateau_cannot_bypass_matching_completed_ack(inputs, mutation):
    policy, obs, oracle, feedback = plateau_boundary(inputs)
    mutation(feedback)
    with pytest.raises(ValueError):
        policy.next_decision(obs, oracle, feedback)
    assert not policy.pressure_plateau and not policy.plateau_probe_active
    assert policy.pending['phase'] == 'pressure'


@pytest.mark.parametrize('gains', [(0., 0.), (.8, 0.), (0., .8), (-.2, -.2)])
def test_plateau_has_at_most_two_one_mm_probes_without_two_consecutive_useful_responses(inputs, gains):
    policy, obs, oracle, feedback = plateau_boundary(inputs)
    action = policy.next_decision(obs, oracle, feedback)
    feedback = acknowledge_synthetic(action, obs, oracle)
    for gain in gains:
        action = policy.next_decision(obs, oracle, feedback)
        assert action['reason'] == 'privileged_cloth_drag'
        assert np.linalg.norm(action['wrist_delta_m']) == pytest.approx(.001)
        feedback = acknowledge_synthetic(action, obs, oracle, gain=gain)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_release'
    assert policy.plateau_probe_steps == policy.drag_steps == 2
    assert policy.pressure_count == 3


def test_two_useful_plateau_probes_return_to_normal_growth_only_after_third_useful_drag(inputs):
    policy, obs, oracle, feedback = plateau_boundary(inputs)
    action = policy.next_decision(obs, oracle, feedback)
    feedback = acknowledge_synthetic(action, obs, oracle)
    for _ in range(3):
        action = policy.next_decision(obs, oracle, feedback)
        assert action['reason'] == 'privileged_cloth_drag'
        assert np.linalg.norm(action['wrist_delta_m']) == pytest.approx(.001)
        feedback = acknowledge_synthetic(action, obs, oracle, gain=.8)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_drag'
    assert np.linalg.norm(action['wrist_delta_m']) == pytest.approx(.002)
    assert not policy.plateau_probe_active and policy.plateau_probe_steps == 2
    assert policy.pressure_count == 3


@pytest.mark.parametrize('start_actions', [3, 39, 40])
def test_plateau_still_releases_within_48_actions_including_hold(inputs, start_actions):
    policy, obs, oracle, feedback = plateau_boundary(inputs)
    policy.configure_budget(48)
    policy.actions = start_actions
    for _ in range(49):
        action = policy.next_decision(obs, oracle, feedback)
        if action['action'] == 'stop':
            break
        feedback = acknowledge_synthetic(action, obs, oracle, gain=.8)
    else:
        pytest.fail('Plateau exhausted the release reserve')
    assert policy.actions <= 48 and policy.phase == 'done'
    assert policy.finished_reason == 'diagnostic_action_budget_release_reserve'
    assert policy.last_metrics['pad_table_gap_m'] >= .012


def test_plateau_settle_drift_below_endpoint_floor_releases_before_probe(inputs):
    policy, obs, oracle, feedback = plateau_boundary(inputs, gap=.00265, wrist_down=.00004)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_contact_settle'
    feedback = acknowledge_synthetic(action, obs, oracle)
    obs['synthetic_compliance_m'] -= .00007
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_release'
    assert policy.finished_reason == 'pressure_endpoint_margin_lost'
    assert policy.drag_steps == 0


def completed_settle_boundary(inputs):
    policy, obs, oracle, feedback = plateau_boundary(inputs)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_contact_settle'
    assert policy.drag_height_anchor_m is None
    feedback = acknowledge_synthetic(action, obs, oracle)
    return policy, obs, oracle, feedback


def test_height_anchor_uses_actual_completed_settle_boundary_only_once(inputs):
    policy, obs, oracle, feedback = completed_settle_boundary(inputs)
    obs['synthetic_compliance_m'] += .00002
    low, _ = policy.prior.measure(obs)
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_drag'
    assert policy.drag_height_anchor_m == low[2]
    assert action['wrist_delta_m'][2] == 0. and action['position_tolerance_m'] == .0001
    anchor = policy.drag_height_anchor_m
    feedback = acknowledge_synthetic(action, obs, oracle, gain=.8)
    obs['synthetic_compliance_m'] += .00003
    policy.next_decision(obs, oracle, feedback)
    assert policy.drag_height_anchor_m == anchor
    assert sum(x['event'] == 'drag_height_anchor_acknowledged' for x in policy.trace) == 1


@pytest.mark.parametrize('mutation', [
    lambda f: f.update(status='failed'), lambda f: f.update(action_id='wrong'),
    lambda f: f.update(elapsed_sim_s=.02), lambda f: f.update(sim_time_s=float('nan')),
])
def test_height_anchor_cannot_be_captured_from_missing_or_incomplete_settle_ack(inputs, mutation):
    policy, obs, oracle, feedback = completed_settle_boundary(inputs)
    mutation(feedback)
    for _ in range(2):
        with pytest.raises(ValueError):
            policy.next_decision(obs, oracle, feedback)
        assert policy.drag_height_anchor_m is None and policy.pending['phase'] == 'contact_settle'


def test_drag_phase_cannot_invent_an_anchor_without_settle_ack(inputs):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0031, count=2)
    policy.phase = 'drag'
    with pytest.raises(ValueError, match='acknowledged contact-settle'):
        policy.next_decision(obs, oracle, feedback)
    assert policy.drag_height_anchor_m is None


@pytest.mark.parametrize('step,probe', [(.001, True), (.001, False), (.002, False)])
@pytest.mark.parametrize('deficit', [-.0001, 0., .000007, .00015, .0003])
def test_compensation_is_up_only_and_preserves_total_step_bound_and_hand_goals(inputs, step, probe, deficit):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0031, count=2)
    policy.phase = 'drag'
    policy.drag_step_m, policy.plateau_probe_active = step, probe
    policy.drag_height_anchor_m = obs['proprioception']['geometry']['wrist']['position_m'][2] + deficit
    original = deepcopy((obs, oracle, feedback))
    action = policy.next_decision(obs, oracle, feedback)
    assert (obs, oracle, feedback) == original
    assert action['reason'] == 'privileged_cloth_drag'
    expected_up = max(0., deficit)
    delta = np.asarray(action['wrist_delta_m'])
    assert delta[2] == pytest.approx(expected_up, abs=1e-15)
    assert np.linalg.norm(delta) <= step + 1e-12
    assert np.linalg.norm(delta[:2]) == pytest.approx(np.sqrt(step**2 - expected_up**2), abs=1e-15)
    assert action['position_tolerance_m'] == (.000025 if expected_up > .000025 + 1e-12 else .0001)
    assert action['wrist_rotation_delta_rad'] == [0., 0., 0.]
    assert set(action) == {'action', 'duration_s', 'wrist_delta_m', 'wrist_rotation_delta_rad',
                           'position_tolerance_m', 'orientation_tolerance_rad', 'reason'}


@pytest.mark.parametrize('step', [.001, .002])
@pytest.mark.parametrize('extra', [0., .00001])
def test_height_only_request_cannot_be_emitted_or_counted_as_useful_drag(inputs, step, extra):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0031, count=2)
    policy.phase = 'drag'
    policy.drag_step_m = step
    # Use exact binary subtraction outcome to exercise the >= step branch.
    policy.drag_height_anchor_m = obs['proprioception']['geometry']['wrist']['position_m'][2] + step + extra + 1e-12
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_release'
    assert policy.finished_reason == 'cloth_height_compensation_exceeds_step_bound'
    assert policy.drag_steps == policy.good_drag_steps == 0


def test_floor_guard_precedes_height_compensation(inputs):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.00259, count=2)
    policy.phase = 'drag'
    policy.drag_height_anchor_m = .75284
    action = policy.next_decision(obs, oracle, feedback)
    assert action['reason'] == 'privileged_cloth_release'
    assert policy.finished_reason == 'pressure_endpoint_margin_lost'


def test_pure_vertical_actual_motion_does_not_fabricate_useful_traction(inputs):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0031, count=2)
    policy.phase = 'drag'
    policy.drag_height_anchor_m = .75325
    action = policy.next_decision(obs, oracle, feedback)
    obs['proprioception']['geometry']['wrist']['position_m'][2] += action['wrist_delta_m'][2]
    obs.update(sequence=obs['sequence']+10, sim_time_s=.6, last_action_id='only_up')
    oracle.update(sequence=obs['sequence'], sim_time_s=.6)
    feedback = {'action_id': 'only_up', 'status': 'completed', 'elapsed_sim_s': .2, 'sim_time_s': .6}
    policy.next_decision(obs, oracle, feedback)
    assert policy.good_drag_steps == 0 and policy.bad_drag_steps == 1
    drag = [x for x in policy.trace if x['event'] == 'drag'][-1]
    assert drag['observed_response_ratio'] is None and drag['mean_node_progress_m'] == 0.


@pytest.mark.parametrize('step,probe', [(.001, True), (.001, False), (.002, False)])
@pytest.mark.parametrize('deficit,precision', [
    (-.00001, .0001), (0., .0001), (.000003, .0001),
    (.000025 - 2e-12, .0001), (.000025, .0001),
    (.000025 + .5e-12, .0001), (.000025 + 1.5e-12, .000025),
    (.000025 + 2e-12, .000025), (.00005, .000025),
])
def test_precision_deadband_preserves_full_correction_and_norm_at_real_height_boundary(
        inputs, step, probe, deficit, precision):
    policy, obs, oracle, feedback = pressure_boundary(inputs, gap=.0031, count=2)
    policy.phase = 'drag'
    policy.drag_step_m, policy.plateau_probe_active = step, probe
    low, _ = policy.prior.measure(obs)
    policy.drag_height_anchor_m = float(low[2] + deficit)
    computed_up = max(0., policy.drag_height_anchor_m - float(low[2]))
    before = deepcopy((obs, oracle, feedback))
    action = policy.next_decision(obs, oracle, feedback)
    assert (obs, oracle, feedback) == before
    assert action['position_tolerance_m'] == precision
    assert action['wrist_delta_m'][2] == computed_up
    assert computed_up >= 0.
    assert np.linalg.norm(action['wrist_delta_m']) <= step + 1e-12
    assert np.linalg.norm(action['wrist_delta_m'][:2]) == pytest.approx(
        np.sqrt(step**2 - computed_up**2), abs=1e-15)
    assert action['wrist_rotation_delta_rad'] == [0., 0., 0.]
    assert set(action) == {'action', 'duration_s', 'wrist_delta_m', 'wrist_rotation_delta_rad',
                           'position_tolerance_m', 'orientation_tolerance_rad', 'reason'}
    assert policy.drag_steps == policy.good_drag_steps == policy.bad_drag_steps == 0
    assert policy.pressure_count == 2 and policy.phase == 'drag'
