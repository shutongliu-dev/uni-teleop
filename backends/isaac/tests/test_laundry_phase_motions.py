from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace
import builtins
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.validation.laundry_captured_garment_fold import prepare_phase, planned_duration
from scripts.validation.laundry_capture_options import validate_hand_options
from scripts.validation.laundry_phase_motions import (
    RELEASE_DEFAULTS, CartesianParkTransition, phase_release_settings,
    phase_park_transition, validate_phase_motions, observed_corner_bottom)
from scripts.validation.laundry_phase_motions import cuff_boundary_descriptions


def options():
    return {'phases': {'0': {'capture_order': ['right', 'left']}, '1': {
        'folded_cuff_boundary': 'left_cuff_minimum', 'sides': ['left'],
        'hands': {'left': {'corner_relative_bottom_m': -.0042761213302612}},
        'release_settings': {'release_profile': 'cooked_hand_clearance_v2',
            'release_hand_profile': 'full_spread_extend_v2', 'retreat_profile': 'finger_axis_exit_v2'},
        'park_transition': {'profile': 'cartesian_to_park_v1', 'side': 'left',
            'duration_s': 3., 'peak_height_m': .12}}}}


def test_later_release_and_transition_do_not_leak_to_original_first_phase():
    value = options(); before = deepcopy(value)
    validate_hand_options(value)
    validate_phase_motions(value, 'trousers_folding')
    assert phase_release_settings(value, 0) == RELEASE_DEFAULTS
    assert phase_park_transition(value, 0) is None
    assert phase_release_settings(value, 1) == value['phases']['1']['release_settings']
    assert planned_duration(1, value, 2.) == planned_duration(1, {'phases': {'0': value['phases']['0']}}, 2.)
    assert planned_duration(2, value, 2.) == 47.
    assert value == before
    value['release_profile'] = 'cooked_hand_clearance_v2'
    value['release_hand_profile'] = 'distal_extend_v1'
    assert phase_release_settings(value, 0)['release_hand_profile'] == 'distal_extend_v1'
    assert phase_release_settings(value, 1)['release_hand_profile'] == 'full_spread_extend_v2'


@pytest.mark.parametrize('change', [
    {'release_settings': {'release_hand_profile': 'full_spread_extend_v2'}},
    {'release_settings': {'typo': 3}},
    {'folded_cuff_boundary': 'invented_interior_material'},
    {'folded_cuff_boundary': []},
    {'park_transition': {'profile': 'teleport'}},
    {'hands': {'left': {'corner_relative_bottom_m': -.011}}},
    {'hands': {'left': {'corner_relative_bottom_m': float('nan')}}},
    {'hands': {'left': {'corner_relative_bottom_m': True}}},
    {'hands': {'left': {'corner_relative_bottom_m': -.004, 'bottom_z_m': .914}}},
])
def test_unused_later_invalid_motion_is_rejected_before_geometry(change):
    value = options(); value['phases']['1'].update(change)
    with pytest.raises(ValueError):
        # Deliberately invalid task geometry must never be reached.
        prepare_phase(None, {'id': 'trousers_folding'}, {}, None, 0, value)


@pytest.mark.parametrize('key,value', [
    ('duration_s', 0.), ('duration_s', 2.01), ('duration_s', True),
    ('peak_height_m', .3), ('peak_height_m', float('nan')), ('side', 'third'),
])
def test_transition_limits_are_explicit_and_no_first_phase_transition(key, value):
    settings = options(); settings['phases']['1']['park_transition'][key] = value
    with pytest.raises(ValueError):validate_phase_motions(settings, 'trousers_folding')
    settings = options(); settings['phases']['0']['park_transition'] = settings['phases']['1']['park_transition']
    with pytest.raises(ValueError):validate_phase_motions(settings, 'trousers_folding')


def test_corner_hand_floor_follows_new_observation_and_clamps_to_real_table():
    pick = {'natural_corner_vertex_id': 2056, 'edge_world_m': [1.15, -.53, .9185761213302612]}
    setting = {'corner_relative_bottom_m': -.0042761213302612}
    bottom, receipt = observed_corner_bottom(pick, setting, .9)
    assert bottom == pytest.approx(.9143, abs=1.e-14)
    assert receipt['natural_corner_vertex_id'] == 2056 and not receipt['cloth_state_written']
    pick['edge_world_m'][2] += .003
    assert observed_corner_bottom(pick, setting, .9)[0] == pytest.approx(.9173)
    pick['edge_world_m'][2] = .901
    bottom, receipt = observed_corner_bottom(pick, setting, .9)
    assert bottom == .9003 and receipt['table_floor_clamped']
    pick['edge_world_m'][2] = 1.05
    with pytest.raises(ValueError):observed_corner_bottom(pick, setting, .9)
    assert observed_corner_bottom({}, {}, .9) == (.9003, None)


def task_fixture():
    rest = np.array([[-.16, -.37, .003], [-.16, -.37, -.003],
                     [-.15, -.37, .003], [.16, -.37, .003], [.16, -.37, -.003]])
    q = rest+np.array([.8, -.1, .9]); q[1, 2] = .9185761213302612
    task = {'id': 'trousers_folding', 'asset_metadata': {'design': {
        'fold_sequence_axis_hinge_sign_radius_m': [[0, .8, -1, .007], [1, -.55, -1, .01]]},
        'cloths': [{'rest_vertices_m': rest.tolist(), 'physical_thickness_m': .0018,
            'boundary_features': {'left_cuff': [0, 1, 2], 'right_cuff': [3, 4], 'waistband': [0, 3]}}]}}
    scene = {'table': {'position_m': [0, 0, .875], 'half_extents_m': [1, 1, .025]}}
    return task, scene, q


def test_boundary_is_original_material_only_fresh_height_and_default_prepare_exact():
    task, scene, q = task_fixture(); original = q.copy(); value = options(); calls = []
    def fake_plan(*args):
        calls.append(args)
        return {'fixed_fixture': 'unchanged'}
    with patch('scripts.validation.laundry_captured_garment_fold.pinch_plan', side_effect=fake_plan):
        baseline = prepare_phase(None, task, scene, q, 0, {})
        modified = prepare_phase(None, task, scene, q, 0, value)
        second = prepare_phase(None, task, scene, q, 1, value)
    assert baseline == modified
    edge = second['edges'][0]
    assert edge['feature'] == 'left_cuff' and edge['natural_corner_vertex_id'] == 1
    assert set(edge['allowed_material_vertex_ids']) <= {0, 1, 2}
    assert calls[-1][5] == pytest.approx(.9143)
    assert edge['contact_plan']['observed_corner_hand_height']['natural_corner_vertex_id'] == 1
    np.testing.assert_array_equal(q, original)
    for invalid_task in ('shirt_folding', 'towel_folding'):
        with pytest.raises(ValueError):validate_phase_motions(value, invalid_task)


def test_opposite_sleeve_corner_retains_real_boundary_and_inward_direction():
    value = {'pickup_profile': 'shirt_cuff_corners', 'phases': {
        '0': {'sleeve_cuff_corner': 'maximum'}, '1': {'sleeve_cuff_corner': 'maximum'}}}
    validate_phase_motions(value, 'shirt_folding')
    original = [{'feature': 'right_cuff', 'material_y_corner': 'minimum', 'inward_xy': [-1., 1.]}]
    assert cuff_boundary_descriptions('shirt_folding', 0, original, {}, pickup_profile='shirt_cuff_corners') is original
    for index in (0, 1):
        row = cuff_boundary_descriptions('shirt_folding', index, original, value['phases'][str(index)],
                                         pickup_profile='shirt_cuff_corners')[0]
        assert row['material_y_corner'] == 'maximum'
        assert row['feature'] == ('right_cuff' if index == 0 else 'left_cuff')
        assert row['inward_xy'] == ([-1., -1.] if index == 0 else [1., -1.])
    for bad in ('center', [], True):
        changed = deepcopy(value); changed['phases']['0']['sleeve_cuff_corner'] = bad
        with pytest.raises(ValueError):validate_phase_motions(changed, 'shirt_folding')
    for task in ('trousers_folding', 'towel_folding'):
        with pytest.raises(ValueError):validate_phase_motions(value, task)
    with pytest.raises(ValueError):validate_phase_motions({'phases': value['phases']}, 'shirt_folding')


def test_actual_rigid_wrist_and_hand_arc_reaches_original_park_without_state_writes():
    start = np.eye(4); start[:3, 3] = [.4, -.2, 1.1]
    end = np.eye(4); end[:3, :3] = Rotation.from_euler('z', 100, degrees=True).as_matrix()
    end[:3, 3] = [.7, .1, 1.05]
    source_start, source_end = start.copy(), end.copy()
    path = CartesianParkTransition(start, end, {'finger': .5}, {'finger': .1}, peak_height_m=.12)
    for u in np.linspace(0, 1, 151):
        matrix, hand = path.targets(u)
        np.testing.assert_allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1.e-14)
        assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.)
        assert hand['finger'] == pytest.approx(.5-.4*u)
    np.testing.assert_allclose(path.targets(0)[0], start, atol=1.e-14)
    np.testing.assert_allclose(path.targets(1)[0], end, atol=1.e-14)
    assert path.targets(.5)[0][2, 3] > max(start[2, 3], end[2, 3])+.09
    assert not path.receipt()['cloth_state_written']
    np.testing.assert_array_equal(start, source_start); np.testing.assert_array_equal(end, source_end)
    for invalid in (float('nan'), -.01, 1.01, True):
        with pytest.raises(ValueError):path.targets(invalid)
    bad = start.copy(); bad[0, 0] = -1
    with pytest.raises(ValueError):CartesianParkTransition(bad, end, {'a': 0}, {'a': 1}, peak_height_m=.12)
    with pytest.raises(ValueError):CartesianParkTransition(start, end, {'a': 0}, {'b': 1}, peak_height_m=.12)


@pytest.mark.parametrize('bad', [
    {'release_settings': {'release_hand_profile': 'full_spread_extend_v2'}},
    {'park_transition': {'profile': 'cartesian_to_park_v1', 'side': 'left',
        'duration_s': 3., 'peak_height_m': float('nan')}},
])
def test_later_invalid_phase_rejected_before_simulator_and_output(tmp_path, monkeypatch, bad):
    from scripts.validation import laundry_captured_garment_fold as runner
    import scripts.maintenance.freeze_project_sources as freeze
    import bench.isaacsim.release as releases
    release = tmp_path/'release'; release.mkdir()
    for name, value in {
        'task.json': {'id': 'trousers_folding'},
        'scene.json': {'named_cloth_initialization': 'author-only', 'cloths': [{'id': 'towel'}],
                       'robot': {'hand_id': 'dual_fr3_sharpa_w01'}},
        'initial_state.json': {},
    }.items():
        (release/name).write_text(json.dumps(value))
    option_file = tmp_path/'options.json'
    option_file.write_text(json.dumps({'phases': {'1': bad}}))
    monkeypatch.setattr(freeze, 'verify_if_snapshot', lambda *a, **k: {'verified': True})
    monkeypatch.setattr(releases, 'validate', lambda _: {'task_id': 'trousers_folding'})
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == 'bench.simulation.isaacsim_backend.env':
            raise AssertionError('An invalid later phase must not initialize a simulator')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    output = tmp_path/'output'
    with pytest.raises(ValueError):
        runner.native(SimpleNamespace(release=release, options=option_file, output=output, max_phases=1))
    assert not output.exists()
