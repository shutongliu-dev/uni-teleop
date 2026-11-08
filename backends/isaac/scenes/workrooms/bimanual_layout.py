"""Task-owned official left/right FR3 installations for optional embodiments.

The fixed room, task objects and shared robot assets stay unchanged.  Positions
refer to real room worktops, not the legacy fallback ``scene.table`` box.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path

import numpy as np

from bench.paths import DATA_ROOT

CONFIGURATION_IDS = ('dual_same_side', 'dual_opposed')
SIDES = ('left', 'right')
ELIGIBLE_TASKS = {
    'mechanical_assembly': ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
        'fastener_assembly', 'gear_train_assembly', 'nist_connector_insertion',
        'nist_belt_drive_assembly', 'nist_cable_routing', 'nist_wire_harness'),
    'laboratory': ('sample_sorting', 'vial_racking', 'cap_placement', 'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer'),
    'study': ('stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving', 'drawer_retrieval', 'reading_stand'),
    'living_room': ('remote_operation', 'water_pouring', 'tower_building', 'storage_box'),
    'card_room': ('task0',),
}

# Explicit world (x,y,z,yaw) placements.  Laboratory's island and rear counter
# share an aisle: same-side means both bases on the island's north side;
# opposed means one island-north floor base and one south-half tabletop mount,
# inside the service spine.  A south floor base cannot cross that fixed spine.
_PLACEMENTS = {
    'mechanical_assembly': {
        'dual_same_side': ((-.10, .30, .78, 0.), (-.10, -.30, .78, 0.)),
        'dual_opposed': ((.42, .76, .78, -math.pi/2), (.42, -.76, .78, math.pi/2)),
    },
    'study': {
        'dual_same_side': ((-.23, .30, .78, 0.), (-.23, -.30, .78, 0.)),
        'dual_opposed': ((1.35, 0., .78, math.pi), (-.23, 0., .78, 0.)),
    },
    'living_room': {
        'dual_same_side': ((.35, -.68, .78, math.pi/2), (.95, -.68, .78, math.pi/2)),
        'dual_opposed': ((.65, .76, .78, -math.pi/2), (.65, -.76, .78, math.pi/2)),
    },
    'card_room': {
        'dual_same_side': ((.35, -.66, .90, math.pi/2), (.95, -.66, .90, math.pi/2)),
        'dual_opposed': ((.65, .72, .90, -math.pi/2), (.65, -.72, .90, math.pi/2)),
    },
    'laboratory': {
        'dual_same_side': ((1.00, 1.10, .90, -math.pi/2), (.40, 1.10, .90, -math.pi/2)),
        'dual_opposed': ((.70, 1.10, .90, -math.pi/2), (.70, -.12, .925, math.pi/2)),
    },
}


def layout_id(family, task_id, configuration_id):
    _checked(family, task_id, configuration_id)
    return f'{family}_{configuration_id}_left_right_sharpa_v1'


def _checked(family, task_id, configuration_id):
    if family not in ELIGIBLE_TASKS or task_id not in ELIGIBLE_TASKS[family]:
        raise ValueError('Task has no optional bimanual layout')
    if configuration_id not in CONFIGURATION_IDS:
        raise ValueError('Unknown bimanual configuration')


def placements(family, task_id, configuration_id):
    _checked(family, task_id, configuration_id)
    return {side: {'base_position_m': list(pose[:3]),
        'base_quaternion_wxyz': [math.cos(pose[3]/2), 0., 0., math.sin(pose[3]/2)],
        'base_yaw_rad': pose[3], 'hand_chirality': side}
        for side, pose in zip(SIDES, _PLACEMENTS[family][configuration_id])}


@lru_cache(maxsize=1)
def _source_inputs():
    from .mechanical_assembly.cooperative_robot_layout import MIRRORED_LAYOUT_ID, robot_inputs
    return robot_inputs(MIRRORED_LAYOUT_ID)


def robot_inputs(family, task_id, configuration_id):
    """One Agent's 58 DOF, original mirrored waiting joints, explicit bases."""
    from bench.control.factory import public_robot_geometry, robot_configuration
    poses = placements(family, task_id, configuration_id)
    robot, initial, source_ready = deepcopy(_source_inputs())
    center = np.mean([p['base_position_m'] for p in poses.values()], axis=0)
    robot['base_position_m'] = center.tolist()
    robot['base_quaternion_wxyz'] = [1., 0., 0., 0.]
    robot['arm_base_poses'] = {side: {
        'position_m': (np.asarray(pose['base_position_m'])-center).tolist(),
        'quaternion_wxyz': pose['base_quaternion_wxyz']} for side, pose in poses.items()}
    models = public_robot_geometry(robot_configuration({'robot': robot})).sides
    ready = {}
    for side, model in models.items():
        own = model.checked_joints({name: initial['joint_positions_rad'][name]
            for name in model.names}, enforce_limits=True)
        wrist = model.arm.forward(own, model.arm_base)[model.arm_tip] @ model.wrist_offset
        ready[side] = {**poses[side], 'layout_id': layout_id(family, task_id, configuration_id),
            'ready_wrist_position_m': wrist[:3, 3].tolist(), 'wrist_fk_matrix': wrist.tolist(),
            'source_layout_id': source_ready[side]['layout_id'],
            'source_ready_joint_positions_preserved': True,
            'source_hand_chirality': side, 'native_qualified': False,
            'manipulation_qualified': False,
            'scope': 'Reset placement and source FK only; workspace/clearance evidence is recorded separately.'}
    return robot, initial, ready


def stand_geometry(family, task_id, configuration_id):
    """Floor stands or an explicitly supported task-owned tabletop plate."""
    result = {}
    for side, pose in placements(family, task_id, configuration_id).items():
        x, y, top = pose['base_position_m']
        if family=='laboratory' and configuration_id=='dual_opposed' and side=='right':
            result[side]={'layout_id':layout_id(family,task_id,configuration_id),
                'owner':'task','side':side,'installation':'tabletop_bolted_plate',
                'support_surface':'main island south half, inside the fixed service spine',
                'ground_datum_m':.90,'mount_top_z_m':top,
                'boxes':[{'name':'robot_tabletop_mount','position_m':[x,y,.9125],
                    'size_m':[.32,.28,.025],'material':'brushed_steel'}]}
            continue
        ground = (.005 if y < 0 else .012) if family == 'living_room' else 0.
        quarter = abs(math.sin(pose['base_yaw_rad'])) > .5
        def xy(a, b): return [b, a] if quarter else [a, b]
        result[side] = {'layout_id': layout_id(family, task_id, configuration_id),
            'owner': 'task', 'side': side, 'ground_datum_m': ground,
            'mount_top_z_m': top, 'boxes': [
                {'name': 'robot_stand_foot', 'position_m': [x, y, ground+.025],
                 'size_m': [*xy(.36, .40), .05], 'material': 'powder_coated_steel'},
                {'name': 'robot_stand_column', 'position_m': [x, y, (ground+.05+top-.025)/2],
                 'size_m': [*xy(.18, .22), top-.025-ground-.05], 'material': 'powder_coated_steel'},
                {'name': 'robot_stand_mount', 'position_m': [x, y, top-.0125],
                 'size_m': [*xy(.28, .30), .025], 'material': 'brushed_steel'}]}
    return result


def prepare_stands(family, task_id, configuration_id, output):
    """Small reusable procedural asset, original robots remain referenced."""
    from bench.simulation.assets.scene_parts import Asset, write_library
    output = Path(output).resolve()
    if not output.is_relative_to(f'{DATA_ROOT}/assets'):
        raise ValueError('Reusable stands belong under shared assets')
    if output.exists():
        raise FileExistsError('Never overwrite a robot installation')
    asset = Asset('bimanual_option_stand', dynamic=False, mass_kg=0)
    stands = stand_geometry(family, task_id, configuration_id)
    for side, stand in stands.items():
        for box in stand['boxes']:
            asset.add_part({'geometry': 'box', 'size_m': box['size_m'], 'color': 'steel'},
                           label=side+'_'+box['name'], offset=box['position_m'])
    return write_library(output, [('bimanual_option_stand', asset)],
        generator='scenes.workrooms.bimanual_layout.prepare_stands',
        extra={'layout_id': layout_id(family, task_id, configuration_id), 'stand_layout': stands,
            'source': {'kind': 'original_procedural', 'license': 'CC0-1.0'},
            'robot_assets_modified': False, 'parent_environment_modified': False})


def workspace_contract(family, task_id, configuration_id):
    poses = placements(family, task_id, configuration_id)
    return {'layout_id': layout_id(family, task_id, configuration_id),
        'configuration_id': configuration_id, 'family': family, 'task_id': task_id,
        'frame': 'world', 'units': 'metres', 'active_sides': list(SIDES),
        'control_scope': 'one_agent_58_dof', 'placements': poses,
        'reference_surface': 'main laboratory island; rear counter remains a separate target surface'
            if family == 'laboratory' else 'fixed room task table',
        'installation': {side:row.get('installation','floor_pedestal')
                         for side,row in stand_geometry(family,task_id,configuration_id).items()},
        'native_qualified': False, 'manipulation_qualified': False}
