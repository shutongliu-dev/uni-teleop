"""Task-owned dual FR3 installation around the unchanged laundry island.

One Agent owns the original official left and right SHARPA hands (58 DOF).
The arms stand on opposite long edges so both can follow their corresponding
towel corners across the existing fold and stacking workspaces. Nothing in
this module changes shared robot assets, drives, controllers or room geometry.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from bench.geometry import rpy_matrix
from bench.paths import DATA_ROOT

LAYOUT_ID = 'laundry_opposed_left_right_sharpa_v1'
SIDES = ('left', 'right')
BASE_CENTER_M = (.75, -.55, .90)
BASE_HALF_SPACING_M = .70
TABLE_BOUNDS_XY_M = ((-.25, -1.), (1.55, -.10))
TABLE_TOP_Z_M = .90
STAND_ROOT = Path(f'{DATA_ROOT}/assets/task_assets/laundry_room/robot_installation/opposed_left_right_v1')
STAND_MANIFEST_SHA256 = '71b3da7e463e72265ec16e7cc1fa206dbecdb44c0206b3986a3e1f82c1c75262'


def placements():
    """World base poses; north left hand and south right hand face the island."""
    result = {}
    for side, sign in (('left', 1), ('right', -1)):
        yaw = -sign * math.pi/2
        result[side] = {
            'base_position_m': [BASE_CENTER_M[0], BASE_CENTER_M[1] + sign*BASE_HALF_SPACING_M, BASE_CENTER_M[2]],
            'base_quaternion_wxyz': [math.cos(yaw/2), 0., 0., math.sin(yaw/2)],
            'base_yaw_rad': yaw,
            'wrist_probe_rpy_rad': [math.pi, 0., yaw + (-1 if side == 'left' else 1)*.75*math.pi],
            'assigned_cloth_edge_y_m': -.37 if side == 'left' else -.73,
        }
    return result


def robot_inputs():
    """Return ``(robot, initial_state, ready_design)`` in the Task10 convention."""
    from bench.control.factory import public_robot_geometry, robot_configuration
    from scenes.workrooms.mechanical_assembly.cooperative_robot_layout import (
        MIRRORED_LAYOUT_ID, robot_inputs as source_inputs)

    robot, initial, source_ready = source_inputs(MIRRORED_LAYOUT_ID)
    robot = deepcopy(robot)
    robot['base_position_m'] = list(BASE_CENTER_M)
    robot['base_quaternion_wxyz'] = [1., 0., 0., 0.]
    robot['arm_base_poses'] = {
        side: {'position_m': [0., sign*BASE_HALF_SPACING_M, 0.],
               'quaternion_wxyz': deepcopy(placements()[side]['base_quaternion_wxyz'])}
        for side, sign in (('left', 1), ('right', -1))}
    models = public_robot_geometry(robot_configuration({'robot': robot})).sides
    ready = {}
    for side, model in models.items():
        own = model.checked_joints({name: initial['joint_positions_rad'][name]
                                    for name in model.names}, enforce_limits=True)
        wrist = model.arm.forward(own, model.arm_base)[model.arm_tip] @ model.wrist_offset
        ready[side] = {
            'layout_id': LAYOUT_ID, 'side': side, **placements()[side],
            'ready_wrist_position_m': wrist[:3, 3].tolist(),
            'wrist_fk_matrix': wrist.tolist(),
            'source_layout_id': MIRRORED_LAYOUT_ID,
            'source_ready_joint_positions_preserved': True,
            'source_hand_chirality': side,
            'minimum_arm_joint_margin_rad': source_ready[side]['minimum_arm_joint_margin_rad'],
            'native_qualified': False, 'manipulation_qualified': False,
            'scope': 'One Agent, official left/right SHARPA, unchanged joints and controls; reset installation only.',
        }
    return robot, deepcopy(initial), ready


def stand_geometry():
    """Three stationary boxes per arm; world boxes with floor-to-mount support.

    Boxes use the same keys as mechanical Task10. Their XY dimensions account
    for each arm's quarter-turn, so callers need no implicit orientation.
    """
    result = {}
    for side, pose in placements().items():
        x, y, top = pose['base_position_m']
        result[side] = {
            'layout_id': LAYOUT_ID, 'owner': 'task', 'side': side,
            'ground_datum_m': 0., 'mount_top_z_m': top,
            'boxes': [
                {'name': 'robot_stand_foot', 'position_m': [x, y, .025],
                 'size_m': [.40, .36, .05], 'material': 'powder_coated_steel'},
                {'name': 'robot_stand_column', 'position_m': [x, y, (.05+top-.025)/2],
                 'size_m': [.22, .18, top-.025-.05], 'material': 'powder_coated_steel'},
                {'name': 'robot_stand_mount', 'position_m': [x, y, top-.0125],
                 'size_m': [.30, .28, .025], 'material': 'brushed_steel'},
            ],
            'scope': 'Task-owned fixed stand; intended floor and robot mounting contact only.',
        }
    return result


def hand_probe_transform(side, cloth_point_m, *, fingertip_clearance_m=.015):
    """Downward near-cloth wrist from the real soft-curl middle fingertip.

    A wrist target alone can be 20 cm away from the cloth. This mapping places
    the real middle fingertip above the corner with an explicit clearance.
    Finger posture is the preserved waiting curl; it is not a pinch grasp.
    """
    if side not in SIDES:
        raise ValueError('Expected left or right')
    point = np.asarray(cloth_point_m, dtype=float)
    if (point.shape != (3,) or not np.isfinite(point).all()
            or not math.isfinite(fingertip_clearance_m) or fingertip_clearance_m < 0):
        raise ValueError('Finite cloth point and nonnegative fingertip clearance required')
    # Source public FK, same values for both official hands except lateral sign.
    tip = np.array([.06385159639490733, -.009999969577734058 if side == 'left' else .010000030422265943,
                    .2005338090649819])
    rotation = rpy_matrix(placements()[side]['wrist_probe_rpy_rad'])
    position = point + [0., 0., fingertip_clearance_m] - rotation @ tip
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def workspace_contract():
    return {
        'layout_id': LAYOUT_ID, 'frame': 'world', 'units': 'metres',
        'active_sides': list(SIDES), 'control_scope': 'one_agent_58_dof',
        'table_bounds_xy_m': [list(value) for value in TABLE_BOUNDS_XY_M],
        'table_top_z_m': TABLE_TOP_Z_M,
        'cloth_corner_ranges_m': {'x': [.439, 1.091], 'y': [-.731, -.369], 'z': [.901, 1.065]},
        'hand_mapping': {'left': 'north towel corner', 'right': 'south towel corner'},
        'probe_fingertip_clearance_m': .015,
        'orientation_policy': 'Downward fixed hand orientations; actual grasp/regrasp and fold-orientation planning remain separate.',
        'native_qualified': False, 'manipulation_qualified': False,
        'scope': 'Discrete actual-hand operational-space endpoint checks, not swept-path or grasp qualification.',
    }


def prepare_stands(output):
    """Author a reusable six-collider fixed asset without touching the room."""
    from pathlib import Path
    from bench.simulation.assets.scene_parts import Asset, write_library

    output = Path(output).resolve()
    if not output.is_relative_to(f'{DATA_ROOT}/assets'):
        raise ValueError('Reusable robot stands belong under shared assets')
    if output.exists():
        raise FileExistsError('Never overwrite an existing robot installation: '+str(output))
    asset = Asset('laundry_dual_robot_stand', dynamic=False, mass_kg=0)
    stands = stand_geometry()
    for side, stand in stands.items():
        for box in stand['boxes']:
            asset.add_part({'geometry': 'box', 'size_m': box['size_m'], 'color': 'steel'},
                           label=side+'_'+box['name'], offset=box['position_m'])
    return write_library(output, [('laundry_dual_robot_stand', asset)],
                         generator='scenes.workrooms.laundry_room.robot_layout.prepare_stands',
                         extra={'layout_id': LAYOUT_ID, 'stand_layout': stands,
                                'source': {'kind': 'original_procedural', 'license': 'CC0-1.0'},
                                'robot_assets_modified': False, 'parent_environment_modified': False})
