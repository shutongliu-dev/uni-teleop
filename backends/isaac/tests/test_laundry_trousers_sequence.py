from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace
import builtins
import json

import numpy as np
import pytest

from scripts.validation.laundry_captured_garment_fold import (
    execution_phase_edges, phase_edges, prepare_phase, planned_duration)
from scripts.validation.laundry_phase_motions import (
    phase_released_dwell, validate_phase_motions, phase_release_settings)
from scripts.validation.laundry_trousers_sequence import (
    SEQUENTIAL_CUFFS, reference_phase_indices, reference_phase_index)


def original_options():
    return {'phases': {'0': {'capture_order': ['right', 'left']}, '1': {
        'folded_cuff_boundary': 'left_cuff_minimum', 'sides': ['left'],
        'hands': {'left': {'corner_relative_bottom_m': -.0042761213302612}},
        'release_settings': {'release_profile': 'cooked_hand_clearance_v2',
            'release_hand_profile': 'full_spread_extend_v2', 'retreat_profile': 'finger_axis_exit_v2'},
        'park_transition': {'profile': 'cartesian_to_park_v1', 'side': 'left',
            'duration_s': 3., 'peak_height_m': .12}}}}


def sequential_options():
    value = original_options()
    value['trousers_phase_sequence'] = SEQUENTIAL_CUFFS
    value['phases']['1']['additional_released_dwell_s'] = 2.
    value['phases']['2'] = {
        'folded_cuff_boundary': 'right_cuff_minimum', 'sides': ['right'],
        'hands': {'right': {'corner_relative_bottom_m': -.0047565361976623}},
        'release_settings': {'release_profile': 'cooked_opening_lift_v3',
            'release_hand_profile': 'full_spread_extend_v2', 'retreat_profile': 'finger_axis_exit_v2'}}
    return value


def geometry():
    rest = np.array([[-.16, -.37, .003], [-.16, -.37, -.003],
                     [-.15, -.37, .003], [.16, -.37, .003], [.16, -.37, -.003]])
    q = rest + [.8, -.1, .9]
    q[1, 2] = .9185761213302612
    q[3, 2] = .9050565361976624
    task = {'id': 'trousers_folding', 'asset_metadata': {'design': {
        'fold_sequence_axis_hinge_sign_radius_m': [[0, .8, -1, .007], [1, -.55, -1, .01]]},
        'cloths': [{'rest_vertices_m': rest.tolist(), 'physical_thickness_m': .0018,
            'boundary_features': {'left_cuff': [0, 1, 2], 'right_cuff': [3, 4], 'waistband': [0, 3]}}]}}
    scene = {'table': {'position_m': [0, 0, .875], 'half_extents_m': [1, 1, .025]}}
    return task, scene, q


def test_default_edges_metadata_and_first_two_geometries_remain_exact():
    task, scene, q = geometry()
    original = original_options(); changed = sequential_options()
    old_input, new_input, old_q = deepcopy(original), deepcopy(changed), q.copy()
    assert reference_phase_indices(original, task['id']) is None
    assert execution_phase_edges(task, original) == phase_edges(task)
    assert reference_phase_indices(changed, task['id']) == (0, 1, 1)
    assert [reference_phase_index(changed, i, task['id']) for i in range(3)] == [0, 1, 1]
    metadata = {'execution_phase_index', 'original_reference_phase_index',
                'trousers_phase_sequence', 'public_fold_goal_changed'}
    with patch('scripts.validation.laundry_captured_garment_fold.pinch_plan',
               return_value={'fixture': 'original source geometry'}):
        for i in (0, 1):
            before = prepare_phase(None, task, scene, q, i, original)
            after = prepare_phase(None, task, scene, q, i, changed)
            assert {k: v for k, v in after.items() if k not in metadata} == before
        third = prepare_phase(None, task, scene, q, 2, changed)
        ordinary = {'phases': {'1': deepcopy(changed['phases']['2'])}}
        reference = prepare_phase(None, task, scene, q, 1, ordinary)
    assert third['phase_index'] == third['execution_phase_index'] == 2
    assert third['original_reference_phase_index'] == 1
    assert third['reference_fold'] == reference['reference_fold']
    assert third['edges'] == reference['edges']
    edge = third['edges'][0]
    assert edge['feature'] == 'right_cuff' and edge['natural_corner_vertex_id'] == 3
    assert set(edge['allowed_material_vertex_ids']) <= {3, 4}
    assert edge['contact_plan']['observed_corner_hand_height']['commanded_hand_bottom_z_m'] == pytest.approx(.9003)
    np.testing.assert_array_equal(q, old_q)
    assert original == old_input and changed == new_input


def test_released_dwell_preserves_old_planned_prefix_without_checkpoint_claim():
    old, new = original_options(), sequential_options()
    assert planned_duration(2, old, 2.) == 47.
    # Two original terminal-free seconds become explicit phase1 dwell before
    # the third new regrasp; the new run still starts from its original reset.
    assert planned_duration(2, new, 0.) == 47.
    assert planned_duration(3, new, 2.) == 67.5
    assert phase_released_dwell(old, 1) == 0.
    assert phase_released_dwell(new, 0) == phase_released_dwell(new, 2) == 0.
    assert phase_released_dwell(new, 1) == 2.
    assert phase_release_settings(new, 0) == phase_release_settings(old, 0)
    assert phase_release_settings(new, 1) == phase_release_settings(old, 1)
    assert phase_release_settings(new, 2)['release_profile'] == 'cooked_opening_lift_v3'


@pytest.mark.parametrize('mutate', [
    lambda v: v.update(trousers_phase_sequence='invented'),
    lambda v: v.update(trousers_phase_sequence=True),
    lambda v: v.update(transport_mode='capture_lift_probe'),
    lambda v: v['phases'].pop('2'),
    lambda v: v['phases'].update({'3': {}}),
    lambda v: v['phases']['0'].update(edge_indices=[1]),
    lambda v: v['phases']['0'].update(edge_indices=[True, 0]),
    lambda v: v['phases']['1'].update(folded_cuff_boundary='right_cuff_minimum'),
    lambda v: v['phases']['2'].update(folded_cuff_boundary='left_cuff_minimum'),
    lambda v: v['phases']['2'].update(sides=['left', 'right']),
    lambda v: v['phases']['2'].update(capture_order=['left']),
    lambda v: v['phases']['2'].update(edge_indices=[1]),
    lambda v: v['phases']['2'].update(edge_indices=[False]),
    lambda v: v['phases']['2'].update(edge_indices=[0.0]),
])
def test_all_later_sequence_settings_rejected_before_geometry(mutate):
    value = sequential_options(); mutate(value)
    with pytest.raises(ValueError):
        prepare_phase(None, {'id': 'trousers_folding'}, {}, None, 0, value)


@pytest.mark.parametrize('dwell', [-.02, 5.02, .01, True, float('nan'), '2'])
def test_invalid_later_free_window_is_rejected_even_for_first_phase(dwell):
    value = sequential_options(); value['phases']['2']['additional_released_dwell_s'] = dwell
    with pytest.raises(ValueError):
        prepare_phase(None, {'id': 'trousers_folding'}, {}, None, 0, value)


def test_no_cross_task_or_implicit_third_boundary():
    for task in ('shirt_folding', 'towel_folding'):
        with pytest.raises(ValueError):validate_phase_motions(sequential_options(), task)
    with pytest.raises(ValueError):validate_phase_motions(sequential_options(), 'trousers_folding', towel=True)
    value = sequential_options(); value.pop('trousers_phase_sequence')
    with pytest.raises(ValueError):validate_phase_motions(value, 'trousers_folding')
    for index in (True, -1, 3):
        with pytest.raises(ValueError):reference_phase_index(sequential_options(), index, 'trousers_folding')


@pytest.mark.parametrize('bad', [
    {'folded_cuff_boundary': 'left_cuff_minimum'},
    {'additional_released_dwell_s': .01},
    {'capture_order': ['left']},
    {'edge_indices': [False]},
    {'edge_indices': [0.0]},
])
def test_unused_invalid_third_operation_never_starts_native(tmp_path, monkeypatch, bad):
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
    value = sequential_options(); value['phases']['2'].update(bad)
    option_file = tmp_path/'options.json'; option_file.write_text(json.dumps(value))
    monkeypatch.setattr(freeze, 'verify_if_snapshot', lambda *a, **k: {'verified': True})
    monkeypatch.setattr(releases, 'validate', lambda _: {'task_id': 'trousers_folding'})
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name == 'bench.simulation.isaacsim_backend.env':
            raise AssertionError('Invalid later operations must not initialize native physics')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    output = tmp_path/'output'
    with pytest.raises(ValueError):
        runner.native(SimpleNamespace(release=release, options=option_file, output=output, max_phases=1))
    assert not output.exists()
