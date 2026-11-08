"""Versioned laundry installations; no runtime robot swaps or controller edits.

These are authoring candidates. Native contact, endpoint reachability and task
qualification belong to each immutable release and are never inferred here.
"""
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT, load_json

CONFIGURATION_IDS = ('single_arm', 'dual_same_side', 'dual_opposed')
SCHEMA = 'laundry_robot_installation.v1'
ASSET_ID = 'laundry_configured_robot_stand'
CURRENT_LAYOUT_VERSION = 2


def active_sides(configuration_id):
    if configuration_id not in CONFIGURATION_IDS:
        raise ValueError('Unknown laundry robot configuration')
    return ['right'] if configuration_id == 'single_arm' else ['left', 'right']


def layout_id(configuration_id, *, version=CURRENT_LAYOUT_VERSION):
    active_sides(configuration_id)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Unknown laundry installation version')
    return 'laundry_'+configuration_id+'_sharpa_v'+str(version)


def placements(configuration_id, *, version=CURRENT_LAYOUT_VERSION):
    sides = active_sides(configuration_id)
    layout_id(configuration_id, version=version)
    south_y = -1.15 if version == 2 else -1.25
    raw = {'single_arm': ((.75, south_y, .90, math.pi/2),),
           'dual_same_side': ((.45, south_y, .90, math.pi/2), (1.05, south_y, .90, math.pi/2)),
           'dual_opposed': ((.75, .15, .90, -math.pi/2), (.75, -1.25, .90, math.pi/2))}[configuration_id]
    return {side: {'base_position_m': list(p[:3]),
        'base_quaternion_wxyz': [math.cos(p[3]/2), 0., 0., math.sin(p[3]/2)],
        'base_yaw_rad': p[3], 'hand_chirality': side}
        for side, p in zip(sides, raw)}


@lru_cache(maxsize=1)
def _source_single():
    from scenes.workrooms.mechanical_assembly.robot_layout import robot_configuration, ready_configuration
    robot = robot_configuration()
    return robot, ready_configuration(robot)[0]


def robot_inputs(configuration_id, *, version=CURRENT_LAYOUT_VERSION):
    from bench.control.factory import public_robot_geometry, robot_configuration
    poses = placements(configuration_id, version=version)
    if configuration_id == 'single_arm':
        robot, initial = deepcopy(_source_single())
        for key in ('base_position_m', 'base_quaternion_wxyz'):
            robot[key] = deepcopy(poses['right'][key])
    else:
        from .robot_layout import robot_inputs as original_inputs
        robot, initial, _ = original_inputs()
        center = np.mean([p['base_position_m'] for p in poses.values()], axis=0)
        robot.update(base_position_m=center.tolist(), base_quaternion_wxyz=[1., 0., 0., 0.],
            arm_base_poses={side: {'position_m': (np.asarray(p['base_position_m'])-center).tolist(),
                                  'quaternion_wxyz': p['base_quaternion_wxyz']} for side, p in poses.items()})
    model = public_robot_geometry(robot_configuration({'robot': robot}))
    models = {'right': model} if configuration_id == 'single_arm' else model.sides
    ready = {}
    for side, own in models.items():
        q = own.checked_joints({k: initial['joint_positions_rad'][k] for k in own.names}, enforce_limits=True)
        wrist = own.arm.forward(q, own.arm_base)[own.arm_tip] @ own.wrist_offset
        ready[side] = {**poses[side], 'layout_id': layout_id(configuration_id, version=version),
            'ready_wrist_position_m': wrist[:3, 3].tolist(), 'wrist_fk_matrix': wrist.tolist(),
            'source_ready_joint_positions_preserved': True, 'source_hand_chirality': side,
            'native_qualified': False, 'manipulation_qualified': False}
    return robot, initial, ready


def stand_geometry(configuration_id, *, version=CURRENT_LAYOUT_VERSION):
    result = {}
    for side, pose in placements(configuration_id, version=version).items():
        x, y, top = pose['base_position_m']
        result[side] = {'layout_id': layout_id(configuration_id, version=version), 'owner': 'task', 'side': side,
            'ground_datum_m': 0., 'mount_top_z_m': top,
            'boxes': [
                {'name': 'foot', 'position_m': [x, y, .025], 'size_m': [.40, .36, .05]},
                {'name': 'column', 'position_m': [x, y, (.05+top-.025)/2], 'size_m': [.22, .18, top-.075]},
                {'name': 'mount', 'position_m': [x, y, top-.0125], 'size_m': [.30, .28, .025]}]}
    return result


def prepare_stands(configuration_id, output, *, version=CURRENT_LAYOUT_VERSION):
    from bench.simulation.assets.scene_parts import Asset, write_library
    output = Path(output).resolve()
    if not output.is_relative_to(f'{DATA_ROOT}/assets') or output.exists():
        raise ValueError('Choose a new shared-assets installation directory')
    asset = Asset(ASSET_ID, dynamic=False, mass_kg=0)
    stands = stand_geometry(configuration_id, version=version)
    for side, stand in stands.items():
        for box in stand['boxes']:
            asset.add_part({'geometry': 'box', 'size_m': box['size_m'], 'color': 'steel'},
                label=side+'_'+box['name'], offset=box['position_m'])
    return write_library(output, [(ASSET_ID, asset)], generator=__name__+'.prepare_stands', extra={
        'installation_schema': SCHEMA, 'configuration_id': configuration_id,
        **({'layout_version': version} if version != 1 else {}),
        'layout_id': layout_id(configuration_id, version=version), 'stand_layout': stands,
        'source': {'kind': 'original_procedural', 'license': 'CC0-1.0'},
        'robot_assets_modified': False, 'parent_environment_modified': False})


def installation(configuration_id, stand_manifest):
    from bench.isaacsim.release import digest
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    path = Path(stand_manifest).resolve()
    if not path.is_relative_to(f'{DATA_ROOT}/assets') or path.name != 'manifest.json':
        raise ValueError('A shared immutable stand manifest is required')
    data = load_json(path)
    version = data.get('layout_version', 1)
    expected = {'installation_schema': SCHEMA, 'configuration_id': configuration_id,
                'layout_id': layout_id(configuration_id, version=version),
                'stand_layout': stand_geometry(configuration_id, version=version),
                'robot_assets_modified': False, 'parent_environment_modified': False}
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError('Stand asset differs from the selected installation')
    load_task_asset(ASSET_ID, path.parent, verify_hashes=True)
    return {'schema': SCHEMA, 'configuration_id': configuration_id,
        **({'layout_version': version} if version != 1 else {}),
        'layout_id': layout_id(configuration_id, version=version), 'active_sides': active_sides(configuration_id),
        'stand_manifest': {'path': str(path), 'sha256': digest(path)},
        'fixture': {'id': 'robot_stand', 'asset_id': ASSET_ID, 'asset_root': str(path.parent),
                    'position_m': [0., 0., 0.]}}


def public_layout(configuration_id, *, version=CURRENT_LAYOUT_VERSION):
    return {'robot_layout_id': layout_id(configuration_id, version=version),
        'placements': placements(configuration_id, version=version),
        'reference_surface': 'unchanged laundry folding table', 'frame': 'world', 'units': 'metres',
        'control_scope': 'one_agent_29_dof' if configuration_id == 'single_arm' else 'one_agent_58_dof'}


def workspace_contract(configuration_id, *, version=CURRENT_LAYOUT_VERSION):
    from .robot_layout import TABLE_BOUNDS_XY_M, TABLE_TOP_Z_M
    return {**public_layout(configuration_id, version=version), 'layout_id': layout_id(configuration_id, version=version),
        'configuration_id': configuration_id, 'active_sides': active_sides(configuration_id),
        'table_bounds_xy_m': [list(v) for v in TABLE_BOUNDS_XY_M], 'table_top_z_m': TABLE_TOP_Z_M,
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Candidate installation; actual fingertips, grasp frames and paths require separate evidence.'}
