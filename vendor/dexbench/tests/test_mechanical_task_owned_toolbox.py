"""Independent CPU geometry checks for the toolbox candidate."""
from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.mechanical_assembly import task_owned_toolbox as toolbox


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp('toolbox') / 'assets'
    toolbox.prepare_assets(root)
    return root


def _top_at(meshes, point_xy):
    tops = []
    for mesh in meshes:
        lower, upper = -np.inf, np.inf
        valid = True
        for normal_x, normal_y, normal_z, offset in ConvexHull(mesh.vertices).equations:
            remaining = normal_x*point_xy[0]+normal_y*point_xy[1]+offset
            if abs(normal_z) < 1e-10:
                if remaining > 1e-10:
                    valid = False
                    break
            elif normal_z > 0:
                upper = min(upper, -remaining/normal_z)
            else:
                lower = max(lower, -remaining/normal_z)
        if valid and lower <= upper and np.isfinite(upper):
            tops.append(upper)
    return max(tops) if tops else None


def test_collision_solids_do_not_duplicate_mass():
    for _, asset in toolbox.make_assets():
        for first, second in combinations([mesh for _, mesh, _ in asset.collision], 2):
            if np.any(np.minimum(first.bounds[1], second.bounds[1]) -
                      np.maximum(first.bounds[0], second.bounds[0]) <= 1e-10):
                continue
            planes = np.concatenate([ConvexHull(first.vertices).equations,
                                     ConvexHull(second.vertices).equations])
            result = linprog([0., 0., 0., -1.],
                             A_ub=np.column_stack([planes[:, :3], np.ones(len(planes))]),
                             b_ub=-planes[:, 3], bounds=[(None, None)]*3+[(0., None)],
                             method='highs')
            assert not result.success or result.x[3] <= 1e-9


def test_lid_and_button_have_actual_support_surfaces():
    assets = dict(toolbox.make_assets())
    specs = toolbox.presentation_specs(task_id='fastener_assembly')
    rows = {row['asset_id']: row for row in specs['fixtures']+specs['objects']}
    body = assets[toolbox.TOOLBOX_ID]
    lid = assets[toolbox.LID_ID]
    button = assets[toolbox.BUTTON_ID]
    body_meshes = [mesh for _, mesh, _ in body.collision]
    lid_meshes = [mesh for _, mesh, _ in lid.collision]
    assert _top_at(body_meshes, [0., 0.]) == pytest.approx(.012)
    for point in [[0., -.085], [0., .085], [-.125, 0.], [.125, 0.]]:
        surface = _top_at(body_meshes, point)+rows[toolbox.TOOLBOX_ID]['position_m'][2]
        assert rows[toolbox.LID_ID]['position_m'][2]-surface == pytest.approx(.0002)
    surface = _top_at(lid_meshes, [0., 0.])+rows[toolbox.LID_ID]['position_m'][2]
    assert rows[toolbox.BUTTON_ID]['position_m'][2]-surface == pytest.approx(.0002)
    assert len(button.visual) == 3
    assert all(color[0] > color[1] for _, _, color in button.visual)
    assert lid.dynamic and button.dynamic


def test_specs_resolve_all_assets_and_do_not_claim_opening(library):
    for task_id in ('fastener_assembly', 'gear_train_assembly'):
        specs = toolbox.presentation_specs(task_id=task_id, asset_root=library)
        for row in specs['fixtures']+specs['objects']:
            asset = load_task_asset(row['asset_id'], row['asset_root'], verify_hashes=True)
            assert asset['collision_parts'] and asset['visual_parts']
        assert specs['stored_task_items'] == []
    contract = toolbox.opening_contract(asset_root=library)
    assert contract['hinge_authored'] is False
    assert contract['latch_authored'] is False
    assert contract['room_release_modified'] is False
    assert contract['task_success_claimed'] is False


def test_layout_rejects_overlap_and_out_of_table_placement(library, monkeypatch):
    specs = toolbox.presentation_specs(task_id='fastener_assembly', asset_root=library)
    blocker = dict(specs['objects'][-1], id='blocking_task_object')
    scene = {'objects': [blocker]}
    report = toolbox.audit_placement(scene, task_id='fastener_assembly', asset_root=library)
    assert report['checks']['inside_fixed_table']
    assert not report['checks']['external_clearance']
    assert not report['passed']
    monkeypatch.setitem(toolbox.PARAMETERS, 'placement_xy_m', [.94, .55])
    report = toolbox.audit_placement(scene, task_id='fastener_assembly', asset_root=library)
    assert not report['checks']['inside_fixed_table']
