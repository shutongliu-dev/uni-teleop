"""CPU metric checks; these never stand in for native material testing."""
import numpy as np
import pytest

from scenes.workrooms.laundry_room import assets, garment_folding_assets
from scripts.validation.laundry_material_settle import mesh_data, metrics, validate_probe_scene


def test_probe_accepts_public_implicit_enabled_robot_but_rejects_disabled_scene():
    scene = {'cloths': [{'id': 'towel'}], 'robot': {'hand_id': 'dual_fr3_sharpa_w01'},
             'named_cloth_initialization': 'author-only'}
    assert validate_probe_scene(scene) == scene['cloths']
    scene['robot']['enabled'] = False
    with pytest.raises(ValueError, match='actual supported robot'):
        validate_probe_scene(scene)


@pytest.mark.parametrize('task,cells', [('towel_flattening', (30, 20)),
    ('towel_folding', (60, 40)), ('towel_stacking', (60, 40))])
def test_towel_probe_uses_material_edge_lengths_and_independent_fold_pairs(task, cells):
    meta = assets.candidate_metadata(task, cells_xy=cells, material_profile='towel_4mm_density_v1')
    for row in meta['cloths']:
        spec = {key: row[key] for key in ('id', 'cells_xy', 'size_m')}
        mesh = mesh_data(spec, {'id': task, 'asset_metadata': meta})
        q = np.asarray(row['initial_vertices_m']);qd = np.zeros_like(q)
        result = metrics(q, qd, mesh, .004, .90, meta['clear_rectangle_xy_m'])
        assert result['edge_ratio_max'] < 1.05
        assert result['table_penetration_max_m'] < 1.e-8
        assert result['raw_speed_p95_m_s'] == 0.
        if task == 'towel_stacking':
            assert result['pair_gap_median_m'] == pytest.approx(.008)
        else:
            assert 'pair_gap_min_m' not in result
        q[0, 2] = .89
        bad = metrics(q, qd, mesh, .004, .9, meta['clear_rectangle_xy_m'])
        assert bad['table_penetration_nodes_over_2mm'] == 1
        assert bad['table_penetration_max_m'] == pytest.approx(.012)


def test_sewn_wall_probe_detects_collapsed_shell_separately_from_stable_speed():
    meta = garment_folding_assets.candidate_metadata('shirt_folding',
        mesh_profile=garment_folding_assets.FLAT_SHIRT_PROFILE, material_profile='shirt_16mm_density_v1')
    row = meta['cloths'][0]
    mesh = mesh_data(row, {'id': 'shirt_folding', 'asset_metadata': meta})
    q = np.asarray(row['initial_vertices_m']);qd = np.zeros_like(q)
    normal = metrics(q, qd, mesh, .0016, .9, meta['clear_rectangle_xy_m'])
    assert normal['pair_gap_median_m'] == pytest.approx(.003)
    half = len(q)//2
    q[:half, 2] = q[half:, 2]+.0001
    collapsed = metrics(q, qd, mesh, .0016, .9, meta['clear_rectangle_xy_m'])
    assert collapsed['raw_speed_p95_m_s'] == 0.
    assert collapsed['pair_gap_fraction_above_45pct_thickness'] == 0.
