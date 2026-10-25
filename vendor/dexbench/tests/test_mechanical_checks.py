"""Reject false support/clearance acceptance using recorded-state fixtures."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from scripts.validation.mechanical_checks import (
    assess_ready_pose, assess_robot_clearance, assess_support, contact_filter_coverage,
)


OBJECT = '/World/Objects/tool'
TABLE = '/World/Room/Geometry/worktop/Collider'
FLOOR = '/World/Room/Geometry/room_floor/Collider'
BASE = '/World/Robot/Geometry/base/fr3_link0'
ARM = BASE+'/fr3_link1'
MOUNT = '/World/Fixtures/robot_stand/collision_002'


def ready_fixture():
    from scenes.workrooms.mechanical_assembly.robot_layout import READY_FINGER_POSTURE_RAD
    joints = {**{f'fr3_joint{i}': 0. for i in range(1, 8)}, **READY_FINGER_POSTURE_RAD}
    wrist = {'frame_id': 'world', 'position_m': [.1, 0., 1.4],
             'quaternion_wxyz': [0., 1., 0., 0.]}
    rows = [{'phase': phase, 'observation': {
                'joint_positions_rad': deepcopy(joints),
                'robot': {'right_hand_flange': deepcopy(wrist)},
                'terminated': False, 'metrics': {'state_finite': True}}}
            for phase in ('settle', 'post_motion', 'repeat_settle') for _ in range(21)]
    return rows, joints, wrist


def test_ready_checks_all_three_terminal_windows_and_ignores_prior_transient():
    rows, joints, wrist = ready_fixture()
    for index in (0, 21, 42):
        rows[index]['observation']['joint_positions_rad']['fr3_joint2'] += .5
    result = assess_ready_pose(rows, joints, wrist)
    assert result['passed']
    assert set(result['phases']) == {'settle', 'post_motion', 'repeat_settle'}
    assert all(phase['checked_control_windows'] == 20 for phase in result['phases'].values())


@pytest.mark.parametrize('phase_index', [1, 22, 43])
@pytest.mark.parametrize('failure', ['other_arm', 'other_finger', 'wrist_position',
                                    'wrist_orientation', 'terminated'])
def test_ready_catches_uncommanded_joints_and_wrist_drift_in_every_phase(phase_index, failure):
    rows, joints, wrist = ready_fixture()
    obs = rows[phase_index]['observation']
    if failure == 'other_arm':
        obs['joint_positions_rad']['fr3_joint2'] += .011
    elif failure == 'other_finger':
        obs['joint_positions_rad']['right_ring_DIP'] += .021
    elif failure == 'wrist_position':
        obs['robot']['right_hand_flange']['position_m'][1] += .006
    elif failure == 'wrist_orientation':
        obs['robot']['right_hand_flange']['quaternion_wxyz'] = [-math.sin(.031/2), math.cos(.031/2), 0., 0.]
    else:
        obs['terminated'] = True
    assert not assess_ready_pose(rows, joints, wrist)['passed']


def test_ready_rejects_missing_phase_incomplete_inventory_and_nonfinite_data():
    rows, joints, wrist = ready_fixture()
    with pytest.raises(ValueError, match='repeat_settle'):
        assess_ready_pose(rows[:42], joints, wrist)
    rows[-1]['observation']['joint_positions_rad'].pop('fr3_joint7')
    with pytest.raises(ValueError, match='inventory'):
        assess_ready_pose(rows, joints, wrist)
    rows, joints, wrist = ready_fixture()
    rows[-1]['observation']['joint_positions_rad']['fr3_joint7'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        assess_ready_pose(rows, joints, wrist)


def contact(first=OBJECT, second=TABLE, impulse=.001, separation=-.0001, time=.02):
    return {'actor0': first, 'actor1': second, 'impulse_magnitude_sum_ns': impulse,
            'min_separation_m': separation, 'time_s': time}


def observation(state=None, contacts=()):
    return {'objects': {} if state is None else {'tool': state},
            'contacts': {'rigid': list(contacts), 'errors': [], 'sampling': {'complete': True}}}


def support_fixture(*, rotated=False):
    # The object origin is not its AABB centre.  At 90 degrees about +Y,
    # the local +X end is lowest, so world bottom must use x_max=.30.
    spec = {'bounds_min_m': [.10, -.02, .03], 'bounds_max_m': [.30, .02, .07],
            'support_top_z_m': .78, 'expected_support_path': TABLE, 'release_height_m': .03}
    quat = [math.sqrt(.5), 0., math.sqrt(.5), 0.] if rotated else [1., 0., 0., 0.]
    rest_z = 1.08 if rotated else .75
    state = {'position_m': [.35, .0, rest_z], 'quaternion_wxyz': quat,
             'linear_velocity_m_s': [0., 0., 0.], 'angular_velocity_rad_s': [0., 0., 0.]}
    initial = deepcopy(state)
    initial['position_m'][2] += .03
    falling = deepcopy(state)
    falling['position_m'][2] += .015
    falling['linear_velocity_m_s'][2] = -.1
    rows = [{'observation': observation(falling)}]
    rows += [{'observation': observation(deepcopy(state), [contact(time=(i+2)*.02)])}
             for i in range(20)]
    return observation(initial), rows, {'tool': spec}, {'tool': OBJECT}


@pytest.mark.parametrize('rotated', [False, True])
def test_offset_origin_and_actual_rotation_support_a_short_drop(rotated):
    result = assess_support(*support_fixture(rotated=rotated))
    assert result['passed']
    tool = result['objects']['tool']
    assert tool['initial_support_gap_m'] == pytest.approx(.03)
    assert tool['maximum_descent_m'] == pytest.approx(.03)
    assert tool['rest_bottom_gap_range_m'] == pytest.approx([0., 0.], abs=1.e-12)
    assert tool['supported_rest_windows'] == 20


@pytest.mark.parametrize('failure', ['wrong_surface', 'no_contacts', 'intermittent',
                                    'penetration', 'hover', 'moving', 'drift',
                                    'no_fall_velocity', 'incomplete', 'wrong_release'])
def test_support_requires_gravity_geometry_settling_and_true_support(failure):
    args = support_fixture()
    _, rows, specs, _ = args
    for index, row in enumerate(rows):
        obs = row['observation']
        if failure == 'wrong_surface':
            for item in obs['contacts']['rigid']:
                item['actor1'] = FLOOR
        elif failure == 'no_contacts' or (failure == 'intermittent' and index > 17):
            obs['contacts']['rigid'] = []
        elif failure == 'penetration' and index:
            obs['objects']['tool']['position_m'][2] -= .003
        elif failure == 'hover' and index:
            obs['objects']['tool']['position_m'][2] += .006
        elif failure == 'moving' and index:
            obs['objects']['tool']['linear_velocity_m_s'][0] = .021
        elif failure == 'drift' and index == 20:
            obs['objects']['tool']['position_m'][0] += .003
        elif failure == 'no_fall_velocity':
            obs['objects']['tool']['linear_velocity_m_s'] = [0., 0., 0.]
        elif failure == 'incomplete' and index == 20:
            obs['contacts']['sampling']['complete'] = False
    if failure == 'wrong_release':
        specs['tool']['release_height_m'] = .22
    assert not assess_support(*args)['passed']


def test_solver_penetration_is_not_hidden_by_correct_visual_bottom():
    args = support_fixture()
    args[1][-1]['observation']['contacts']['rigid'][0]['min_separation_m'] = -.003
    result = assess_support(*args)
    assert result['objects']['tool']['checks']['geometry_penetration']
    assert not result['objects']['tool']['checks']['solver_penetration']
    assert not result['passed']


def test_support_rejects_missing_inventory_and_invalid_bounds():
    args = support_fixture()
    args[2]['tool']['bounds_max_m'][0] = args[2]['tool']['bounds_min_m'][0]
    with pytest.raises(ValueError, match='positive extents'):
        assess_support(*args)
    args = support_fixture()
    args[1][-1]['observation']['objects'] = {}
    with pytest.raises(ValueError, match='inventory changed'):
        assess_support(*args)


def clearance_rows(*contacts):
    return [{'observation': observation(contacts=contacts)}]


def test_clearance_separates_exact_base_mount_contact_from_illegal_contact():
    allowed = [(BASE, MOUNT)]
    result = assess_robot_clearance(clearance_rows(contact(BASE, MOUNT, 3.)), [BASE, ARM], allowed)
    assert result['passed']
    assert result['allowed_contacts'][0]['maximum_substep_impulse_ns'] == 3.
    for other in (TABLE, OBJECT, MOUNT, BASE):
        result = assess_robot_clearance(clearance_rows(contact(ARM, other)), [BASE, ARM], allowed)
        assert not result['passed'], other
        assert len(result['rejected_contacts']) == 1


def test_clearance_uses_fixed_noise_floor_and_keeps_zero_impulse_geometry():
    result = assess_robot_clearance(clearance_rows(contact(ARM, TABLE, 1.e-9, -.003)), [BASE, ARM])
    assert result['passed']
    assert result['observed_robot_pairs'][0]['minimum_separation_m'] == -.003
    result = assess_robot_clearance(clearance_rows(contact(ARM, TABLE, 1.01e-8)), [BASE, ARM])
    assert not result['passed']


def test_clearance_rejects_incomplete_or_nonfinite_evidence_and_broad_exemptions():
    rows = clearance_rows()
    rows[0]['observation']['contacts']['sampling']['complete'] = False
    assert not assess_robot_clearance(rows, [BASE, ARM])['passed']
    with pytest.raises(ValueError, match='Invalid native'):
        assess_robot_clearance(clearance_rows(contact(ARM, TABLE, float('nan'))), [BASE, ARM])
    with pytest.raises(ValueError, match='Only explicit fixed-base'):
        assess_robot_clearance(clearance_rows(), [BASE, ARM], [(ARM, MOUNT)])


def coverage_env():
    mesh = '/World/Room/Geometry/cabinet/Collider'
    actor = '/World/Room/Geometry/cabinet/Actor'
    sensors = [BASE, ARM, OBJECT]
    filters = sensors+[TABLE, FLOOR, actor, MOUNT]
    return SimpleNamespace(
        scene_info={'robot_link_paths': {'fr3_link0': BASE, 'fr3_link1': ARM},
            'objects': {'tool': {'path': OBJECT}}, 'support_paths': [TABLE, FLOOR, actor],
            'fixtures': {'robot_stand': {'collision_paths': [MOUNT]}},
            'collider_actor_paths': {mesh: actor},
            'workbench': {'solid_paths': [TABLE, FLOOR, mesh],
                          'supports': {'table': TABLE, 'floor': FLOOR},
                          'contact_unobserved_paths': []}},
        _contact_sensor_paths=sensors, _contact_filter_paths=[filters.copy() for _ in sensors])


def test_coverage_uses_actual_actor_remapping_for_every_sensor():
    env = coverage_env()
    result = contact_filter_coverage(env)
    assert result['passed']
    assert result['robot_sensor_count'] == 2 and result['object_sensor_count'] == 1
    assert result['fixture_actor_paths'] == [MOUNT]


@pytest.mark.parametrize('failure', ['missing_sensor', 'missing_mount', 'missing_object',
                                    'missing_self_pair', 'unobserved_mesh', 'duplicate_sensor'])
def test_coverage_does_not_accept_silently_unobserved_collisions(failure):
    env = coverage_env()
    if failure == 'missing_sensor':
        env._contact_sensor_paths.pop()
        env._contact_filter_paths.pop()
    elif failure == 'missing_mount':
        env._contact_filter_paths[1].remove(MOUNT)
    elif failure == 'missing_object':
        env._contact_filter_paths[0].remove(OBJECT)
    elif failure == 'missing_self_pair':
        env._contact_filter_paths[1].remove(BASE)
    elif failure == 'unobserved_mesh':
        env.scene_info['workbench']['contact_unobserved_paths'] = ['/World/Room/hidden_mesh']
    else:
        env._contact_sensor_paths.append(BASE)
        env._contact_filter_paths.append(env._contact_filter_paths[0])
    assert not contact_filter_coverage(env)['passed']
