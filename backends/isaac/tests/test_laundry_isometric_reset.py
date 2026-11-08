from copy import deepcopy

import numpy as np
import pytest

from scenes.workrooms.laundry_room.assets import candidate_metadata, validate_metadata


def candidate(**changes):
    options = dict(cells_xy=(60, 40), material_profile='towel_4mm_density_v1',
                   reset_profile='towel_isometric_u_fold_v1')
    options.update(changes)
    return candidate_metadata('towel_stacking', **options)


def test_discrete_fold_preserves_material_edges_areas_and_current_physics():
    value = candidate()
    baseline = candidate(reset_profile=None)
    assert value['physics_options'] == baseline['physics_options']
    assert value['material_prior'] == baseline['material_prior']
    assert value['target_center_xy_m'] == baseline['target_center_xy_m']
    for cloth, original in zip(value['cloths'], baseline['cloths']):
        copied = deepcopy(cloth)
        copied['initial_vertices_m'] = original['initial_vertices_m']
        assert copied == original
        rest, q, tri = [np.asarray(cloth[k]) for k in
                        ('rest_vertices_m', 'initial_vertices_m', 'triangles')]
        edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        length = lambda p: np.linalg.norm(p[edges[:, 0]]-p[edges[:, 1]], axis=1)
        area = lambda p: np.linalg.norm(np.cross(p[tri[:, 1]]-p[tri[:, 0]],
                                                 p[tri[:, 2]]-p[tri[:, 0]]), axis=1)/2
        np.testing.assert_allclose(length(q), length(rest), rtol=0, atol=3.e-16)
        np.testing.assert_allclose(area(q), area(rest), rtol=0, atol=3.e-18)
        assert q[:, 2].min()-cloth['physical_thickness_m']/2 >= .9-1.e-14
        assert np.ptp(q[:, 2]) == pytest.approx(np.sqrt(2)*.010)
        assert cloth['fixed_vertices'] == []
    assert value['native_physics_accepted'] is False
    assert value['material_calibrated'] is False
    validate_metadata(value)


@pytest.mark.parametrize('changes', [
    {'cells_xy': (120, 40)}, {'material_profile': None},
    {'reset_profile': 'unrecognized_fold'},
])
def test_fold_profile_does_not_silently_change_grid_or_material(changes):
    with pytest.raises(ValueError):
        candidate(**changes)


def test_only_stacking_can_select_this_initial_fold_and_tampering_is_rejected():
    with pytest.raises(ValueError):
        candidate_metadata('towel_folding', cells_xy=(60, 40),
                           material_profile='towel_4mm_density_v1',
                           reset_profile='towel_isometric_u_fold_v1')
    value = candidate()
    value['cloths'][0]['initial_vertices_m'][30][2] += .001
    with pytest.raises(ValueError):
        validate_metadata(value)
