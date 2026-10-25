"""Bimanual (dual FR3 + SHARPA) contracts on CPU; no native physics claim."""
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import DATA_ROOT
from bench.protocol.schemas import AgentObservation, JointLimit, System0Action
from scenes.manipulation.evaluator import BimanualTaskEvaluator, SequenceEvaluator, SimpleTaskEvaluator

ASSETS = f'{DATA_ROOT}/assets'
COMMON = {'initial_tolerance_m': .005, 'minimum_lift_m': .08, 'hold_s': .3, 'maximum_speed_m_s': .04,
          'maximum_angular_speed_rad_s': .3, 'maximum_motion_speed_m_s': 1.5, 'minimum_contact_duration_s': .04,
          'max_observation_gap_s': .041, 'maximum_tilt_rad': .35, 'physics_dt_s': .0005, 'control_period_s': .02,
          'hand_sides': {'left': ['left_'], 'right': ['right_']}}
INITIAL = {'carry_box': [.55, 0., .811], 'cube_left': [.45, .2, .781], 'cube_right': [.45, -.2, .781]}


def bimanual_task(mode):
    evaluation = ({'mode': 'bimanual_lift', 'object': 'carry_box', 'initial_position_m': INITIAL['carry_box']}
                  if mode == 'bimanual_lift' else
                  {'mode': 'dual_lift', 'objects': {'cube_left': 'left', 'cube_right': 'right'},
                   'initial_positions_m': {k: INITIAL[k] for k in ('cube_left', 'cube_right')}})
    return {'id': '05_' + mode, 'steps': [{'id': mode, 'instruction_zh': 'x'}], 'evaluation': {**evaluation, **COMMON}}


class Run:
    def __init__(self, mode):
        self.evaluator = SequenceEvaluator(bimanual_task(mode))
        self.time = -.02
        self.objects = list(self.evaluator.objects)
        self.positions = {name: np.array(INITIAL[name]) for name in self.objects}

    def observe(self, lift=0., contacts=()):
        """contacts: (object, actor_link) pairs touching in every substep of this window."""
        self.time = round(self.time + .02, 10)
        rows = []
        if self.time:
            for name, link in contacts:
                actor = '/World/Table' if link == 'table' else f'/World/Robot/Side/{link}'
                for tick in range(1, 41):
                    rows.append({'actor0': f'/World/Objects/{name}', 'actor1': actor, 'time_s': self.time - .02 + tick * .0005,
                                 'point_count': 1, 'impulse_magnitude_sum_ns': .002, 'min_separation_m': -.0001,
                                 'impulse_vector_ns': [0., 0., .002], 'nonzero_impulse_normal_count': 1,
                                 'normal_length_max_abs_error': 0., 'impulse_components': 'normal_only'})
        start = self.time - .02 if self.time else 0.
        objects = {name: {'position_m': (self.positions[name] + [0., 0., lift]).tolist(), 'quaternion_wxyz': [1, 0, 0, 0],
                          'linear_velocity_m_s': [0, 0, 0], 'angular_velocity_rad_s': [0, 0, 0]} for name in self.objects}
        obs = {'time_s': self.time, 'last_duration_s': .02 if self.time else 0., 'metrics': {'state_finite': True},
               'objects': objects,
               'contacts': {'source': 'PhysX stage-bound rigid contact tensor; normal impulses', 'rigid': rows, 'errors': [],
                            'sampling': {'protocol': 'native_rigid_contact_sampling_v1', 'physics_dt_s': .0005,
                                         'start_sim_time_s': start, 'end_sim_time_s': self.time,
                                         'native_start_time_s': start, 'native_end_time_s': self.time,
                                         'native_start_step_index': round(start / .0005),
                                         'native_end_step_index': round(self.time / .0005),
                                         'sampled_substeps': 40 if self.time else 0, 'complete': True}}}
        return self.evaluator.update(obs)

    def raise_to(self, lift, contacts, step=.002):
        """Lift gradually (motion-speed bound), keeping the given contacts."""
        current = getattr(self, 'lift', 0.)
        while current < lift - 1e-12:
            current = min(lift, current + step)
            result = self.observe(lift=current, contacts=contacts)
        self.lift = current
        return result


def test_dispatcher_keeps_single_arm_modes():
    from test_simple_tasks import task
    assert type(SequenceEvaluator(task('push'))) is SimpleTaskEvaluator
    assert type(SequenceEvaluator(bimanual_task('dual_lift'))) is BimanualTaskEvaluator


def test_bimanual_lift_needs_both_hands_off_table_for_the_hold():
    run = Run('bimanual_lift')
    run.observe(contacts=[('carry_box', 'table')])
    both = [('carry_box', 'left_index_elastomer'), ('carry_box', 'right_middle_elastomer')]
    run.raise_to(.09, [('carry_box', 'left_index_elastomer')])
    for _ in range(30):
        result = run.observe(lift=.09, contacts=[('carry_box', 'left_index_elastomer')])
    assert not result['success'] and result['objects']['carry_box']['holding_hands'] == ['left']
    for _ in range(14):
        result = run.observe(lift=.09, contacts=both)
    assert not result['success'] and result['reason'] == 'holding'
    for _ in range(3):
        result = run.observe(lift=.09, contacts=both)
    assert result['success']


def test_bimanual_lift_rejects_table_support_and_low_lift():
    run = Run('bimanual_lift')
    run.observe()
    both = [('carry_box', 'left_index_elastomer'), ('carry_box', 'right_index_elastomer')]
    run.raise_to(.05, both)
    for _ in range(40):
        low = run.observe(lift=.05, contacts=both)
    assert not low['success']
    run.raise_to(.09, both + [('carry_box', 'table')])
    for _ in range(40):
        supported = run.observe(lift=.09, contacts=both + [('carry_box', 'table')])
    assert not supported['success'] and supported['objects']['carry_box']['table_support_contact']


def test_arm_contact_is_irrecoverable():
    run = Run('bimanual_lift')
    run.observe()
    result = run.observe(contacts=[('carry_box', 'fr3_left_link7')])
    assert result['non_hand_robot_contact_violated'] and result['irrecoverable_reason'] == 'non_hand_robot_contact_violated'
    run.raise_to(.09, [('carry_box', 'left_index_elastomer'), ('carry_box', 'right_index_elastomer')])
    for _ in range(40):
        result = run.observe(lift=.09, contacts=[('carry_box', 'left_index_elastomer'), ('carry_box', 'right_index_elastomer')])
    assert not result['success']


def test_dual_lift_requires_each_cube_in_its_own_hand_only():
    run = Run('dual_lift')
    run.observe()
    swapped = [('cube_left', 'right_thumb_elastomer'), ('cube_right', 'left_thumb_elastomer')]
    run.raise_to(.09, swapped)
    for _ in range(40):
        result = run.observe(lift=.09, contacts=swapped)
    assert not result['success']
    shared = [('cube_left', 'left_thumb_elastomer'), ('cube_left', 'right_index_elastomer'), ('cube_right', 'right_thumb_elastomer')]
    for _ in range(40):
        result = run.observe(lift=.09, contacts=shared)
    assert not result['success'] and not result['objects']['cube_left']['held']
    own = [('cube_left', 'left_thumb_elastomer'), ('cube_right', 'right_index_elastomer')]
    for _ in range(17):
        result = run.observe(lift=.09, contacts=own)
    assert result['success']


def test_initial_state_is_checked_per_object():
    run = Run('dual_lift')
    run.positions['cube_left'] = run.positions['cube_left'] + [.02, 0., 0.]
    assert run.observe()['reason'].startswith('invalid_observation:initial state')


@pytest.fixture(scope='module')
def bimanual_policy():
    from bench.control.bimanual import BimanualFingertipPolicy
    return BimanualFingertipPolicy(ASSETS, [0., 0., .77])


def test_bimanual_geometry_is_an_exact_mirror(bimanual_policy):
    g = bimanual_policy.geometry
    assert g.frame_ids == ['left_hand_flange', 'right_hand_flange'] and len(g.names) == 58
    q = {j.name: 0. for j in g.joints}
    q.update({f'fr3_right_joint{i}': v for i, v in enumerate([.4, -.5, .1, -2.3, .2, 1.8, .8], 1)})
    q.update({f'fr3_left_joint{i}': v for i, v in enumerate([-.4, -.5, -.1, -2.3, -.2, 1.8, -.8], 1)})
    q = {n: float(np.clip(v, g.by_name[n].lower, g.by_name[n].upper)) for n, v in q.items()}
    hands = g.public_hand_geometries(q)
    left, right = (np.asarray(hands[s]['wrist']['position_m']) for s in ('left', 'right'))
    assert np.allclose(left * [1, -1, 1], right, atol=1e-9)
    tips = {s: np.asarray(hands[s]['fingertip_positions_m']['index']) for s in ('left', 'right')}
    assert np.allclose(tips['left'] * [1, -1, 1], tips['right'], atol=2e-6)


def test_bimanual_action_moves_one_hand_and_holds_the_other(bimanual_policy):
    p, g = bimanual_policy, bimanual_policy.geometry
    caps = SimpleNamespace(control_period_s=.02, joint_limits={j.name: JointLimit(lower_rad=j.lower, upper_rad=j.upper) for j in g.joints})
    p.reset(SimpleNamespace(episode_id='bi', observation_profile='vision_proprio'), caps)
    q = {j.name: float(np.clip(0., j.lower, j.upper)) for j in g.joints}
    q.update({f'fr3_right_joint{i}': v for i, v in enumerate([.4, -.5, 0., -2.3, 0., 1.8, .8], 1)})
    q.update({f'fr3_left_joint{i}': v for i, v in enumerate([-.4, -.5, 0., -2.3, 0., 1.8, -.8], 1)})
    state = {'q': dict(q), 'qd': dict.fromkeys(q, 0.), 'seq': 1}
    obs = lambda: AgentObservation(episode_id='bi', sequence=state['seq'], sim_time_s=.02 * (state['seq'] - 1), instruction='x',
                                   profile='vision_proprio', cameras=[],
                                   proprioception={'joint_positions_rad': state['q'], 'joint_velocities_rad_s': state['qd']})
    wrist = g.public_hand_geometries(q)['right']['wrist']
    target = {**wrist, 'position_m': [wrist['position_m'][0] + .03, wrist['position_m'][1], wrist['position_m'][2] - .02]}
    action = System0Action(action_id='a', observation_sequence=1, skill='move_geometry', timeout_sim_s=5.,
                           hand_geometries={'right': {'wrist': target, 'fingertip_frame_id': 'right_hand_flange'}})
    p.begin(action, obs())
    left_before = {n: q[n] for n in g.sides['left'].names}
    for _ in range(250):
        result = p.step(obs())
        if result.status != 'running':
            break
        p.commit_pending_step()
        command = result.command.joint_positions_rad
        assert set(command) == set(g.names)
        new = {n: state['q'][n] + .45 * (command[n] - state['q'][n]) for n in state['q']}
        state['qd'] = {n: (new[n] - state['q'][n]) / .02 for n in new}
        state['q'], state['seq'] = new, state['seq'] + 1
    assert result.status == 'completed' and result.robot_execution_state.objective_status == 'reached'
    assert max(abs(state['q'][n] - left_before[n]) for n in left_before) < 1e-6
    # A hand objective in the other hand's frame is rejected before execution.
    with pytest.raises(ValueError):
        System0Action(action_id='b', observation_sequence=1, skill='move_geometry', timeout_sim_s=5.,
                      hand_geometries={'left': {'wrist': target, 'fingertip_frame_id': 'right_hand_flange'}})


def test_bimanual_geometry_decision_builds_per_hand_objectives(bimanual_policy):
    from bench.protocol.schemas import EpisodeSpec, RobotCapabilities
    from agent.model_client.geometry_actions import build_geometry_action
    p, g = bimanual_policy, bimanual_policy.geometry
    q = {j.name: float(np.clip(0., j.lower, j.upper)) for j in g.joints}
    caps = RobotCapabilities(robot_id='dual_fr3_sharpa_w01', robot_asset_sha256='0' * 64, arm_joints=g.arm_names,
                             hand_joints=[n for n in g.names if n not in g.arm_names],
                             joint_limits={j.name: JointLimit(lower_rad=j.lower, upper_rad=j.upper) for j in g.joints},
                             tracks=['system0'], system0=p.descriptor, frame_ids=['world', 'robot_base', *g.frame_ids],
                             finger_names=['thumb', 'index', 'middle', 'ring', 'little'])
    spec = EpisodeSpec(episode_id='bi', agent_id='a', agent_version='v', agent_config_sha256='0' * 64, task_id='05',
                       task_release_sha256='0' * 64, evaluator_sha256='0' * 64, robot_id='dual_fr3_sharpa_w01',
                       robot_asset_sha256='0' * 64, physics_profile_sha256='0' * 64, scene_variant='original_v1',
                       scene_sha256='0' * 64, seed=0, track='system0', observation_profile='vision_proprio',
                       instruction='x', budgets={'sim_time_s': 60., 'max_actions': 10, 'max_model_calls': 1},
                       system0=p.descriptor)
    observation = AgentObservation(episode_id='bi', sequence=3, sim_time_s=.04, instruction='x', profile='vision_proprio',
                                   cameras=[], proprioception={'joint_positions_rad': q, 'joint_velocities_rad_s': dict.fromkeys(q, 0.),
                                                               'hand_geometries': g.public_hand_geometries(q)}).model_dump(mode='json', exclude_none=True)
    decision = {'action': 'move_geometry', 'duration_s': 2., 'hands': {
        'left': {'wrist_delta_m': [0., 0., .01]}, 'right': {'fingertip_delta_m': {'index': [.002, 0., 0.]}}}}
    action = build_geometry_action(decision, observation, caps.model_dump(mode='json'), spec.model_dump(mode='json'),
                                   action_id='x1', expected_sequence=3)
    objectives = action['hand_geometries']
    assert set(objectives) == {'left', 'right'} and action.get('geometry') is None
    assert objectives['left']['fingertip_frame_id'] == 'left_hand_flange'
    left_wrist = g.public_hand_geometries(q)['left']['wrist']['position_m']
    assert np.allclose(objectives['left']['wrist']['position_m'], np.asarray(left_wrist) + [0., 0., .01])
    from agent.model_client.actions import ActionValidationError
    with pytest.raises(ActionValidationError):
        build_geometry_action({**decision, 'wrist_delta_m': [0., 0., .01]}, observation, caps.model_dump(mode='json'),
                              spec.model_dump(mode='json'), action_id='x2', expected_sequence=3)


# --- soft-body (sponge) evaluator ------------------------------------------------
SOFT = {'initial_tolerance_m': .005, 'nominal_height_m': .060, 'hold_s': .3, 'maximum_rest_speed_m_s': .02,
        'max_observation_gap_s': .041, 'maximum_motion_speed_m_s': 1.5, 'maximum_lift_m': .005,
        'recovered_height_fraction': .9, 'table_top_z_m': .75, 'physics_dt_s': .0005, 'control_period_s': .02}


def soft_task(mode):
    base = {'soft_body': 'sponge', 'initial_centroid_m': [.45, .1, .7815], **SOFT}
    if mode == 'soft_press':
        base.update(mode=mode, press_height_fraction=.75, release_hold_s=.2, maximum_displacement_m=.02)
    else:
        base.update(mode=mode, goal_position_m=[.45, .2, .7815], goal_xy_radius_m=.025)
    return {'id': '07_' + mode, 'evaluation': base}


class SoftRun:
    def __init__(self, mode):
        self.evaluator = SequenceEvaluator(soft_task(mode))
        self.time = -.02

    def observe(self, xy=(.45, .1), height=.060, speed=0., lift=0.):
        self.time = round(self.time + .02, 10)
        low = [xy[0] - .016, xy[1] - .02, .75 + lift]
        high = [xy[0] + .016, xy[1] + .02, .75 + lift + height]
        centroid = [xy[0], xy[1], .7815 if height == .060 and not lift else (low[2] + high[2]) / 2]
        return self.evaluator.update({'time_s': self.time, 'metrics': {'state_finite': True}, 'soft_bodies': {'sponge': {
            'centroid_m': centroid, 'bounds_min_m': low, 'bounds_max_m': high, 'max_node_speed_m_s': speed}}})


def test_soft_press_needs_a_held_press_then_a_release():
    from scenes.manipulation.evaluator import SoftBodyTaskEvaluator
    run = SoftRun('soft_press')
    assert isinstance(run.evaluator, SoftBodyTaskEvaluator)
    run.observe()
    for _ in range(10):
        result = run.observe(height=.040)
    assert not result['press_completed']                      # 0.18 s < 0.3 s hold
    run.observe(height=.050)                                  # released too early: dwell resets
    for _ in range(16):
        result = run.observe(height=.040)
    assert result['press_completed'] and not result['success']
    for _ in range(12):
        result = run.observe(height=.058)
    assert result['success'] and result['contact_evidence'].startswith('unavailable')


def test_soft_press_latches_displacement_and_lift():
    run = SoftRun('soft_press')
    run.observe()
    xs = np.linspace(.45, .475, 6)
    for x in xs:
        result = run.observe(xy=(float(x), .1), height=.040)
    assert result['irrecoverable_reason'] == 'episode_displacement_limit_exceeded'
    lifted = SoftRun('soft_press')
    lifted.observe()
    for k in range(5):
        result = lifted.observe(lift=.002 * (k + 1))
    assert result['irrecoverable_reason'] == 'soft_body_lifted_off_table'


def test_soft_push_needs_goal_rest_and_no_compression():
    run = SoftRun('soft_push')
    run.observe()
    for y in np.linspace(.1, .2, 26)[1:]:
        run.observe(xy=(.45, float(y)), speed=.05)
    for _ in range(20):
        squashed = run.observe(xy=(.45, .2), height=.045)
    assert not squashed['success']
    for _ in range(14):
        moving = run.observe(xy=(.45, .2), speed=.05)
    assert not moving['success']
    for _ in range(16):
        result = run.observe(xy=(.45, .2))
    assert result['success'] and result['goal_xy_error_m'] < 1e-9


def test_soft_windowed_rest_ignores_node_jitter_but_not_motion():
    # PhysX reports 0.04-0.09 m/s nodal speeds for an untouched sponge (contact
    # jitter); v6 rest (instantaneous) then never holds. v7 opt-in rest uses the
    # body state's change over a trailing window.
    task = soft_task('soft_push')
    task['evaluation'].update(rest_speed_source='windowed_body_state_v1', rest_window_s=.1)

    def pushed(evaluator=None):
        run = SoftRun('soft_push')
        if evaluator is not None:
            run.evaluator = evaluator
        first = run.observe()
        for y in np.linspace(.1, .2, 26)[1:]:
            run.observe(xy=(.45, float(y)), speed=.05)
        return run, first
    jittery, first = pushed(SequenceEvaluator(task))
    assert first['rest_speed_m_s'] is None                    # no full window yet: not at rest
    for _ in range(60):
        result = jittery.observe(xy=(.45, .2), speed=.07)
    assert result['success'] and result['rest_speed_m_s'] == 0. and result['max_node_speed_m_s'] == .07
    instantaneous, _ = pushed()
    for _ in range(60):
        legacy = instantaneous.observe(xy=(.45, .2), speed=.07)
    assert not legacy['success'] and legacy['rest_speed_source'] == 'instantaneous_max_node_speed'
    sliding, _ = pushed(SequenceEvaluator(task))
    for k in range(60):                                       # 3 cm/s, inside the goal disc
        result = sliding.observe(xy=(.45 + .0006 * k, .2))
    assert not result['success'] and abs(result['rest_speed_m_s'] - .03) < 1e-6
    task['evaluation'].pop('rest_window_s')
    with pytest.raises(ValueError, match='rest_window_s'):
        SequenceEvaluator(task)


def test_client_prompt_is_embodiment_aware_and_sharpa_unchanged():
    from agent.model_client.runner import model_system_prompt
    for control, interaction in (('fingertip', 'action'), ('fingertip', 'tools'), ('hybrid', 'free')):
        assert model_system_prompt(control, interaction) == model_system_prompt(control, interaction, robot_id='sharpa_w01_right')
    allegro = model_system_prompt('fingertip', 'action', robot_id='allegro_v5_right')
    assert 'Allegro' in allegro and 'right SHARPA W01 hand.' not in allegro
    assert 'Available fingers: index, middle, ring, thumb' in allegro and '10 mm aluminium adapter' in allegro
    shadow = model_system_prompt('fingertip', 'action', robot_id='shadow_hand_e_right')
    assert 'directly on the FR3 tool flange' in shadow
    dual = model_system_prompt('fingertip', 'action', robot_id='dual_fr3_sharpa_w01')
    assert '"hands":{"left"' in dual and 'pair of Franka FR3 arms' in dual
    assert model_system_prompt('fingertip', 'action', robot_id='unregistered_fixture') == model_system_prompt('fingertip', 'action')


def test_bimanual_tactile_is_projected_per_hand():
    from bench.protocol.schemas import AgentObservation, TactileObservation
    from bench.sensors.tactile import FINGER_LINKS, NATIVE_SOURCE, SOURCE, project_tactile, tactile_configuration
    links = [f'{side}_' + link.split('_', 1)[1] for side in ('left', 'right') for link in FINGER_LINKS.values()]
    paths = {link: '/World/Robot/' + link for link in links}
    metadata = {'native_isaacsim_app': True, 'physics_dt_s': .0005, 'control_dt_s': .02,
                'hand_spec': {'hand_id': 'dual_fr3_sharpa_w01'}, 'scene_info': {'robot_link_paths': paths},
                'body_names': list(paths),
                'contact_evidence': {'source': NATIVE_SOURCE, 'impulse_components': 'normal_only',
                                     'aggregation': {'version': 'float64_point_normal_impulses_v1'},
                                     'sensor_paths': list(paths.values())}}
    configs = {side: tactile_configuration(metadata, 'dual_fr3_sharpa_w01', side) for side in ('left', 'right')}
    assert all(c['available'] and c['side'] == side for side, c in configs.items())
    assert configs['left']['sensor_paths']['little'] == '/World/Robot/left_pinky_elastomer'
    assert tactile_configuration(metadata, 'dual_fr3_sharpa_w01')['available'] is False   # no side, no projection
    row = {'time_s': .0005, 'actor0': '/World/Robot/left_index_elastomer', 'actor1': '/World/Objects/x',
           'point_count': 1, 'nonzero_impulse_normal_count': 1, 'normal_length_max_abs_error': 0.,
           'impulse_components': 'normal_only', 'impulse_vector_ns': [0., 0., .001], 'impulse_magnitude_sum_ns': .001,
           'min_separation_m': -.001}
    raw = {'sim_time_s': .02, 'last_duration_s': .02, 'contacts': {'source': SOURCE, 'errors': [], 'rigid': [row]}}
    hands = {side: project_tactile(raw, configs[side]) for side in ('left', 'right')}
    assert hands['left']['fingers']['index']['rigid_contact_any'] is True
    assert hands['right']['fingers']['index']['rigid_contact_any'] is False
    q = {'fr3_left_joint1': 0.}
    observation = AgentObservation(episode_id='bi', sequence=1, sim_time_s=.02, instruction='x', cameras=[],
                                   proprioception={'joint_positions_rad': q, 'joint_velocities_rad_s': q}, hand_tactile=hands)
    assert set(observation.model_dump(mode='json')['hand_tactile']) == {'left', 'right'}
    single = AgentObservation(episode_id='bi', sequence=1, sim_time_s=.02, instruction='x', cameras=[],
                              proprioception={'joint_positions_rad': q, 'joint_velocities_rad_s': q})
    assert 'hand_tactile' not in single.model_dump(mode='json')
    with pytest.raises(ValueError):
        AgentObservation(episode_id='bi', sequence=1, sim_time_s=.02, instruction='x', cameras=[],
                         proprioception={'joint_positions_rad': q, 'joint_velocities_rad_s': q},
                         hand_tactile=hands, tactile=TactileObservation.model_validate(hands['left']))


def test_audit_labels_every_registered_hand_as_hand_not_arm():
    """The episode audit's pair types follow the evaluator's hand rule (left SHARPA,
    Allegro, LEAP and Shadow hand bodies were reported as arm contacts before)."""
    from bench.isaacsim.audit import pair_labels
    box = '/World/Objects/carry_box'
    hands = ['/World/Robot/Geometry/base/fr3_link0/fr3_link7/fr3_link8/right_hand_flange/right_pinky_elastomer',
             '/World/Robot/Left/Geometry/fr3_left_base/fr3_left_link7/fr3_left_link8/left_hand_flange/left_pinky_elastomer',
             '/World/Robot/Geometry/base/fr3_link7/fr3_link8/fr3_hand_adapter/allegro_palm_link/allegro_link_3_0_tip',
             '/World/Robot/Geometry/base/fr3_link7/fr3_link8/fr3_hand_adapter/leap_palm_lower/leap_fingertip',
             '/World/Robot/Geometry/base/fr3_link7/rh_wrist/rh_palm/rh_ffdistal']
    arms = ['/World/Robot/Right/Geometry/fr3_right_base/fr3_right_link0/fr3_right_link3',
            '/World/Robot/Geometry/base/fr3_link0/fr3_link7', '/World/Robot/Geometry/base/fr3_link7/fr3_link8/fr3_hand_adapter']
    for path in hands:
        _, robot, hand, objects = pair_labels({'actor0': path, 'actor1': box, 'collider0': None, 'collider1': None})
        assert robot == [True, False] and hand == [True, False] and objects == [False, True], path
    for path in arms:
        _, robot, hand, _ = pair_labels({'actor0': path, 'actor1': box, 'collider0': None, 'collider1': None})
        assert robot[0] and not hand[0], path


def test_compliant_squeeze_steps_land_within_the_target_band():
    # 0.1.6: fingertip steps below the 1.5 mm tolerance complete without moving, so the last
    # step is >= 1.6 mm and lands on the target when it can (pad depth from measured joints).
    from scenes.manipulation.bimanual_policy import (SQUEEZE_DEPTH_M, SQUEEZE_DONE_GAP_M, SQUEEZE_MAX_STEP_M,
                                                    SQUEEZE_MIN_STEP_M, squeeze_step)
    assert squeeze_step(SQUEEZE_DONE_GAP_M) == 0. and squeeze_step(-.001) == 0.
    assert squeeze_step(.001) == SQUEEZE_MIN_STEP_M
    assert squeeze_step(.0025) == .0025
    assert squeeze_step(.010) == SQUEEZE_MAX_STEP_M and squeeze_step(.005) == pytest.approx(.005 - SQUEEZE_MIN_STEP_M)
    for gap in np.linspace(0., .03, 301):
        remaining, steps = gap, 0
        while (step := squeeze_step(remaining)):
            assert SQUEEZE_MIN_STEP_M <= step <= SQUEEZE_MAX_STEP_M
            remaining -= step
            steps += 1
        assert -(SQUEEZE_MIN_STEP_M - SQUEEZE_DONE_GAP_M) - 1e-12 <= remaining <= SQUEEZE_DONE_GAP_M + 1e-12
        assert SQUEEZE_DEPTH_M - remaining <= .002   # final overlap stays inside the 2 mm penetration gate
