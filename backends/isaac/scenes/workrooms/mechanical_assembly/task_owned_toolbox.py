"""Task-owned toolbox geometry with explicit support and placement audits.

The movable lid and rotary cap are separate rigid bodies. Their eventual hinge
and latch need a native mechanism layer; geometry alone does not open the lid.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from bench.geometry import quaternion_matrix
from bench.paths import DATA_ROOT
from bench.simulation.assets import scene_parts
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from . import assembly_geometry as geometry
from .scene import SCENE


ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/toolbox/v7')
TOOLBOX_ID = 'mechanical_task_toolbox'
LID_ID = 'mechanical_task_toolbox_lid'
BUTTON_ID = 'mechanical_task_rotary_button'
ASSET_IDS = {'toolbox': TOOLBOX_ID, 'lid': LID_ID, 'button': BUTTON_ID}
PARAMETERS = {
    'body_size_m': [.260, .180, .012],
    'wall_thickness_m': .010,
    'wall_height_m': .048,
    'lid_thickness_m': .006,
    'lid_grip_overhang_m': .002,
    'button_radius_m': .014,
    'button_stem_radius_m': .009,
    'button_stem_height_m': .003,
    'button_cap_height_m': .009,
    'support_gap_m': .0002,
    'placement_xy_m': [.94, -.38],
    'minimum_external_gap_m': .020,
}


def _toolbox():
    asset = geometry.make_asset(TOOLBOX_ID, dynamic=False, density_kg_m3=1150.)
    width, depth, bottom = PARAMETERS['body_size_m']
    wall, height = PARAMETERS['wall_thickness_m'], PARAMETERS['wall_height_m']
    geometry.add_box(asset, 'base', [width, depth, bottom],
                     center=[0., 0., bottom/2], color='plastic_blue')
    for side, sign in [('front', -1.), ('rear', 1.)]:
        geometry.add_box(asset, 'wall_'+side, [width, wall, height],
                         center=[0., sign*(depth-wall)/2, bottom+height/2], color='plastic_blue')
    for side, sign in [('left', -1.), ('right', 1.)]:
        geometry.add_box(asset, 'wall_'+side, [wall, depth-2*wall, height],
                         center=[sign*(width-wall)/2, 0., bottom+height/2], color='plastic_blue')
    asset.data['geometry_contract'] = {
        'kind': 'open_toolbox_body', 'room_owned': False,
        'interior_size_m': [width-2*wall, depth-2*wall, height],
        'support_surface': 'four wall top faces',
    }
    return asset


def _lid():
    asset = geometry.make_asset(LID_ID, dynamic=True, density_kg_m3=1150.)
    width, depth, _ = PARAMETERS['body_size_m']
    thickness, overhang = PARAMETERS['lid_thickness_m'], PARAMETERS['lid_grip_overhang_m']
    geometry.add_box(asset, 'lid_panel', [width+2*overhang, depth+2*overhang, thickness],
                     center=[0., 0., thickness/2], color='plastic_blue')
    asset.data['geometry_contract'] = {
        'kind': 'loose_lid_on_rim', 'grip_overhang_m': overhang,
        'hinge_authored': False, 'support_surface': 'panel underside on box walls',
    }
    return asset


def _button():
    asset = geometry.make_asset(BUTTON_ID, dynamic=True, density_kg_m3=1150.)
    stem_height = PARAMETERS['button_stem_height_m']
    stem_radius = PARAMETERS['button_stem_radius_m']
    cap_top = stem_height+PARAMETERS['button_cap_height_m']
    cap_radius = PARAMETERS['button_radius_m']
    geometry.add_revolved(asset, 'stem', [[0., 0., stem_radius], [stem_height, 0., stem_radius]],
                          color='plastic_red', segments=48)
    geometry.add_revolved(asset, 'turning_cap', [[stem_height, 0., cap_radius], [cap_top, 0., cap_radius]],
                          color='plastic_red', segments=48)
    geometry.add_box(asset, 'turning_tab', [.020, .008, .005],
                     center=[0., 0., cap_top+.0025], color='plastic_red')
    asset.data['geometry_contract'] = {
        'kind': 'stepped_rotary_cap_with_turning_tab', 'axis': 'Z',
        'support_surface': 'stem bottom on lid top', 'latch_authored': False,
    }
    return asset


def make_assets():
    return [(TOOLBOX_ID, _toolbox()), (LID_ID, _lid()), (BUTTON_ID, _button())]


def opening_contract(*, task_ids=('fastener_assembly', 'gear_train_assembly'), asset_root=ASSET_ROOT):
    return {
        'schema': 'mechanical_task_toolbox_opening_candidate.v1',
        'asset_root': str(Path(asset_root).resolve()), 'asset_ids': dict(ASSET_IDS),
        'task_ids': list(task_ids),
        'control': {'kind': 'rotary_button', 'button_axis': 'Z', 'lid_hinge_axis': 'X'},
        'hinge_authored': False, 'latch_authored': False,
        'shortcuts_forbidden': ['object_pose_writeback', 'rigid_weld', 'magnetic_snap',
                               'direct_lid_torque', 'direct_button_pose_writeback'],
        'status': 'prepared_unvalidated',
        'native_mechanism_status': 'hinge_and_latch_not_implemented',
        'room_release_modified': False, 'ordinary_agent_validation': False,
        'model_calls': 0, 'task_success_claimed': False,
    }


def presentation_specs(*, task_id, asset_root=ASSET_ROOT):
    """Compose closed geometry; actual hinge/latch operation remains unimplemented."""
    if task_id not in ('fastener_assembly', 'gear_train_assembly'):
        raise ValueError('Toolbox detail is scoped to L4/L5')
    root = str(Path(asset_root).resolve())
    position_xy = PARAMETERS['placement_xy_m']
    table_top = SCENE['coordinates']['table_top_z_m']
    gap = PARAMETERS['support_gap_m']
    lid_bottom = table_top+PARAMETERS['body_size_m'][2]+PARAMETERS['wall_height_m']+gap
    button_bottom = lid_bottom+PARAMETERS['lid_thickness_m']+gap
    specs = [geometry.physical_spec(asset_id, root, name, [*position_xy, bottom])
             for asset_id, name, bottom in (
                 (TOOLBOX_ID, 'task_toolbox', table_top),
                 (LID_ID, 'task_toolbox_lid', lid_bottom),
                 (BUTTON_ID, 'task_rotary_button', button_bottom))]
    return {
        'task_id': task_id, 'asset_root': root,
        'fixtures': specs[:1], 'objects': specs[1:],
        'stored_task_items': [],
        'placement_scope': 'Source geometry support and clearance only; native settling and opening unvalidated.',
    }


def _world_bounds(spec):
    asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
    bounds = asset.get('aabb_m')
    if bounds is None:
        bounds = [asset['bounds_min_m'], asset['bounds_max_m']]
    lower, upper = np.asarray(bounds, dtype=float)
    corners = np.asarray(list(product(*zip(lower, upper))))
    points = corners @ quaternion_matrix(spec.get('quaternion_wxyz', [1., 0., 0., 0.])).T
    points += np.asarray(spec['position_m'], dtype=float)
    return np.array([points.min(axis=0), points.max(axis=0)])


def audit_placement(scene, *, task_id, asset_root=ASSET_ROOT):
    """Check transformed catalog bounds against the fixed table and all other props."""
    specs = presentation_specs(task_id=task_id, asset_root=asset_root)
    bounds = {row['id']: _world_bounds(row) for row in specs['fixtures']+specs['objects']}
    coordinates = SCENE['coordinates']
    center, size = np.asarray(coordinates['table_center_xy_m']), np.asarray(coordinates['table_size_xy_m'])
    table_lower, table_upper = center-size/2, center+size/2
    margins = {name: float(min(np.min(box[0, :2]-table_lower), np.min(table_upper-box[1, :2])))
               for name, box in bounds.items()}
    external_gaps = {}
    for other in scene.get('objects', [])+scene.get('fixtures', []):
        other_bounds = _world_bounds(other)
        external_gaps[other['id']] = min(float(np.linalg.norm(np.maximum(np.maximum(
            other_bounds[0, :2]-box[1, :2], box[0, :2]-other_bounds[1, :2]), 0.)))
            for box in bounds.values())
    checks = {
        'inside_fixed_table': all(margin >= PARAMETERS['minimum_external_gap_m'] for margin in margins.values()),
        'external_clearance': bool(external_gaps) and all(
            gap >= PARAMETERS['minimum_external_gap_m'] for gap in external_gaps.values()),
    }
    return {
        'passed': all(checks.values()), 'checks': checks,
        'bounds_world_m': {name: box.tolist() for name, box in bounds.items()},
        'table_edge_margin_m': margins, 'external_xy_gap_m': external_gaps,
        'scope': 'Conservative transformed source bounds; no camera visibility, robot path or native contact qualification.',
    }


def prepare_assets(output=ASSET_ROOT):
    return geometry.write_assets(
        output, make_assets(), generator=__name__+'.prepare_assets',
        parameters={**PARAMETERS, 'opening_contract': opening_contract(asset_root=output)},
        source_paths=[__file__, geometry.__file__, scene_parts.__file__],
    )
