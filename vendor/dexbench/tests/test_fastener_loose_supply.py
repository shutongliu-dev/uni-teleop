"""Source-mesh supply clearances; no native support or manipulation claim."""
from pathlib import Path

import numpy as np
import pytest

from bench.geometry import quaternion_matrix
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly import loose_assets, task


OLD_ASSETS = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/'
                  'fastener_assembly/loose_bolt_v4')
THREADED_ASSETS = Path(f'{DATA_ROOT}/assets/task_parts/threaded_sharpa/'
                       'sdf1024_convex_partition_candidate02_20261018')


@pytest.fixture(scope='module')
def supply(tmp_path_factory):
    root = tmp_path_factory.mktemp('loose_supply')/'assets'
    loose_assets.prepare_assets(root)
    values = task.build_loose_task_inputs(guide_root=root, threaded_root=THREADED_ASSETS)
    return root, values


def collision_vertices(asset):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(asset['usd_path'])
    cache = UsdGeom.XformCache()
    points = []
    for part in asset['collision_parts']:
        mesh = stage.GetPrimAtPath(part['usd_prim_path'])
        vertices = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(cache.GetLocalToWorldTransform(mesh), dtype=float)
        points.append(vertices @ transform[:3, :3] + transform[3, :3])
    return np.concatenate(points)


def test_bolt_nest_removes_only_its_two_intruding_tool_rails(supply):
    root, _ = supply
    before = load_json(OLD_ASSETS/'manifest.json')['assets']
    after = load_json(root/'manifest.json')['assets']
    for role in ('entry_guide', 'tool_rest'):
        asset_id = loose_assets.ASSET_IDS[role]
        assert after[asset_id] == before[asset_id]
    asset_id = loose_assets.ASSET_IDS['supply_rest']
    old_rest, new_rest = before[asset_id], after[asset_id]
    assert [part['label'] for part in old_rest['collision_parts'][64:]] == [
        'tool_side_rail_left', 'tool_side_rail_right']
    assert new_rest['collision_parts'] == old_rest['collision_parts'][:64]
    assert new_rest['visual_parts'] == old_rest['visual_parts'][:1]
    assert new_rest['geometry_contract']['vertical_support'] == 'original_workroom_table'
    assert new_rest['cpu_geometry_audit']['passed']
    for key in ('material_prior', 'friction', 'contact_offset_m', 'rest_offset_m'):
        assert new_rest[key] == old_rest[key]


def test_actual_scaled_bolt_mesh_clears_annular_nest_and_starts_above_table(supply):
    root, values = supply
    scene = values['scene']
    bolt = next(row for row in scene['objects'] if row['id'] == 'bolt')
    rest = next(row for row in scene['fixtures'] if row['id'] == 'bolt_supply_rest')
    asset = load_task_asset(bolt['asset_id'], bolt['asset_root'], verify_hashes=True)
    points = collision_vertices(asset)
    # Compute from all actual nested collider vertices, including the 1.5 source
    # scale and inverse cooking scales, rather than trusting a visual AABB.
    world = points @ quaternion_matrix(bolt['quaternion_wxyz']).T + bolt['position_m']
    table_top = rest['position_m'][2]
    assert float(world[:, 2].min()-table_top) == pytest.approx(.0002, abs=5.e-9)
    assert float(np.linalg.norm(world[:, :2]-rest['position_m'][:2], axis=1).max()) == pytest.approx(.0225, abs=5.e-9)
    nest = load_task_asset(rest['asset_id'], root, verify_hashes=True)
    inner_radius = nest['geometry_contract']['bolt_nest_inner_radius_m']
    # The whole body is inside the guaranteed open bore at reset. This excludes
    # intersection with every convex annular segment, including between vertices.
    assert inner_radius-np.linalg.norm(world[:, :2]-rest['position_m'][:2], axis=1).max() >= .00049999
    assert values['task']['reset_conditions']['bolt_supply_position_m'] == bolt['position_m']
    assert scene['mechanical_task_provenance']['bolt_supply_layout']['vertical_support'] == 'original_workroom_table'


def test_old_rails_geometrically_explain_the_recorded_initial_penetration(supply):
    _, values = supply
    bolt = next(row for row in values['scene']['objects'] if row['id'] == 'bolt')
    asset = load_task_asset(bolt['asset_id'], bolt['asset_root'], verify_hashes=True)
    points = collision_vertices(asset)
    head_radius = float(np.linalg.norm(points[:, :2], axis=1).max())
    rest = load_json(OLD_ASSETS/'manifest.json')['assets'][loose_assets.ASSET_IDS['supply_rest']]
    left, right = rest['collision_parts'][64:]
    assert head_radius-abs(left['aabb_m'][1][0]) == pytest.approx(.0025, abs=5.e-9)
    assert head_radius-right['aabb_m'][0][0] == pytest.approx(.0025, abs=5.e-9)
    # The old 4 mm + 2 mm offset placed the flipped head at the rail bottom;
    # the rails overlap the genuine head floor/walls, rather than just an AABB.
    flipped = points @ quaternion_matrix([0., 1., 0., 0.]).T
    old_head_bottom = .006
    head = flipped[points[:, 2] >= .0248*1.5-1.e-8]
    old_origin = float(asset['bounds_max_m'][2]) + old_head_bottom
    head_z = head[:, 2] + old_origin
    assert head_z.min() == pytest.approx(.006, abs=5.e-9)
    assert head_z.max() > right['aabb_m'][1][2]
    assert right['aabb_m'][0][2] == pytest.approx(.006)
