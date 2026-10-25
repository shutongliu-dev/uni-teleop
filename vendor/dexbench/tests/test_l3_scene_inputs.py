from copy import deepcopy
import math

import pytest

from scripts.assets.prepare_l3_scene_inputs import build_cases, prepare
from scripts.validation.validate_assembly_contacts import assess_case


@pytest.fixture
def inputs():
    heights = {'spacer': .030, 'wide_ring': .020, 'top_ring': .025}
    supplies = {n: {'position_m': [.25, y, .786], 'quaternion_wxyz': [1., 0., 0., 0.]}
                for n, y in zip(heights, [-.27, 0., .27])}
    scene = {'physics': {'gravity_m_s2': [0., 0., -9.81]}}
    task = {'evaluation': {
        'part_heights_m': heights, 'initial_objects': supplies,
        'target_bottom_centers_m': {'spacer': [.4, 0., .788],
            'wide_ring': [.4, 0., .818], 'top_ring': [.4, 0., .838]},
        'maximum_axis_offset_m': .00425, 'maximum_bottom_height_error_m': .0005,
        'maximum_axis_tilt_rad': math.radians(5), 'maximum_object_linear_speed_m_s': 1.,
    }}
    geometry = {'shaft_tip_world_z_m': .888, 'shoulder_world_z_m': .788,
        'shaft_center_xy_m': [.4, 0.], 'shaft_radius_m': .012,
        'top_ring_outer_radius_m': .035, 'shoulder_size_xy_m': [.120, .120]}
    return scene, task, geometry


def test_energy_budget_counts_the_full_shaft_travel(inputs):
    cases, energy = build_cases(*inputs)
    scene, task, geometry = inputs
    release_z = cases[0]['reset_objects']['top_ring']['position_m'][2]
    assert release_z - geometry['shaft_tip_world_z_m'] == pytest.approx(.0002)
    fall = release_z - task['evaluation']['target_bottom_centers_m']['top_ring'][2]
    assert .05 < fall < .051
    assert math.sqrt(2 * 9.81 * fall) < 1.
    assert energy['spacer_canonical_above_tip_ballistic_speed_m_s'] > 1.4
    too_low = deepcopy(task)
    too_low['evaluation']['target_bottom_centers_m']['top_ring'][2] -= .001
    with pytest.raises(ValueError, match='speed budget'):
        build_cases(scene, too_low, geometry)


def test_source_inputs_and_thresholds_stay_unchanged(inputs):
    before = deepcopy(inputs)
    cases, _ = build_cases(*inputs)
    assert inputs == before
    for case in cases:
        assert set(case['reset_objects']) == set(case['assertions']) == {'spacer', 'wide_ring', 'top_ring'}
        assert 'robot_phases' not in case
        assert case['duration_s'] > .5
        assert case['duration_s'] / .02 == round(case['duration_s'] / .02)
        assert all(row['quaternion_wxyz'] == [1., 0., 0., 0.]
                   for row in case['reset_objects'].values())


def test_positive_resets_are_separated_and_guidance_is_required(inputs):
    cases, _ = build_cases(*inputs)
    e = inputs[1]['evaluation']
    for case in cases[:2]:
        reset = case['reset_objects']
        assert reset['spacer']['position_m'][2] > .788
        assert reset['wide_ring']['position_m'][2] > (
            reset['spacer']['position_m'][2] + e['part_heights_m']['spacer'])
        assert reset['top_ring']['position_m'][2] > .888
        assert case['assertions']['wide_ring']['support_path_prefix'] == '/World/Objects/spacer'
        assert case['assertions']['top_ring']['support_path_prefix'] == '/World/Objects/wide_ring'
    guided = cases[1]
    assert guided['reset_objects']['top_ring']['position_m'][0] == pytest.approx(.407)
    assert guided['assertions']['top_ring']['require_guidance_contact'] is True


def test_outside_case_has_no_reset_overlap_with_shaft_or_lower_rings(inputs):
    case = build_cases(*inputs)[0][-1]
    g = inputs[2]
    p = case['reset_objects']['top_ring']['position_m']
    separation = p[0] - .4 - g['shaft_radius_m'] - g['top_ring_outer_radius_m']
    assert separation == pytest.approx(.001)
    assert .4 < p[0] < .4 + g['shoulder_size_xy_m'][0] / 2
    assert p[2] - .788 == pytest.approx(.0015)
    for name in ('spacer', 'wide_ring'):
        assert case['reset_objects'][name] == inputs[1]['evaluation']['initial_objects'][name]
    assert case['expected_functional_success'] is False
    assert case['negative_control_objects'] == ['top_ring']


def test_negative_subset_preserves_inventory_and_legacy_gate(inputs):
    case = build_cases(*inputs)[0][-1]
    paths = {n: {'path': '/World/Objects/' + n} for n in case['reset_objects']}
    rows = []
    for i in range(400):
        poses, contacts = {}, []
        for name, reset in case['reset_objects'].items():
            position = list(reset['position_m'])
            if name == 'top_ring':
                position[2] -= .0015 * min(i / 20, 1.)
            poses[name] = position + [0., 0., 0., 1.]
            contacts.append({'actor0': paths[name]['path'],
                'actor1': case['assertions'][name]['support_path_prefix'] + '/collision_000',
                'normal_impulses_ns': [.001], 'normals': [[0., 0., 1.]],
                'impulse_magnitude_sum_ns': .001})
        rows.append({'native_step': i + 1, 'time_s': (i + 1) * .005,
            'poses_xyzw': poses, 'velocities': {n: [0.] * 6 for n in poses},
            'robot_interference': [], 'maximum_penetration_m': 0., 'raw_contacts': contacts})
    result = assess_case(rows, case, {'objects': paths}, .005)
    assert result['passed']
    assert result['checks']['negative_control_is_settled_wrong_pose']
    assert result['checks']['negative_control_inactive_inventory']
    for invalid in ([], 'top_ring', ['top_ring', 'top_ring'], ['missing'], [1], [{}]):
        invalid_case = deepcopy(case)
        invalid_case['negative_control_objects'] = invalid
        with pytest.raises(ValueError, match='nonempty unique subset'):
            assess_case(rows, invalid_case, {'objects': paths}, .005)
    legacy = deepcopy(case)
    legacy.pop('negative_control_objects')
    assert not assess_case(rows, legacy, {'objects': paths}, .005)['passed']
    # Inactive supply contacts cannot establish active-object rejection.
    without_active_contacts = deepcopy(rows)
    for row in without_active_contacts:
        row['raw_contacts'] = [c for c in row['raw_contacts'] if c['actor0'] != paths['top_ring']['path']]
    assert not assess_case(without_active_contacts, case, {'objects': paths}, .005)['passed']
    without_supply_contacts = deepcopy(rows)
    for row in without_supply_contacts:
        row['raw_contacts'] = [c for c in row['raw_contacts'] if c['actor0'] == paths['top_ring']['path']]
    assert not assess_case(without_supply_contacts, case, {'objects': paths}, .005)['passed']
    for row in rows:
        row['raw_contacts'] = []
    assert not assess_case(rows, case, {'objects': paths}, .005)['passed']


def test_changed_release_is_rejected_before_loading_assets(tmp_path):
    (tmp_path / 'release_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='exact pinned L3'):
        prepare(tmp_path)
