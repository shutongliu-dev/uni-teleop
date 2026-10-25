"""Task-owned L3 tabletop layout and manipulation boundaries.

The parent workshop remains the pinned empty room.  This module records the
task's supply lane, shaft station, visual/grasp cues, and recovery contract so
the native privileged validator can check the reset without guessing from
asset geometry or a renderer preview.
"""
from __future__ import annotations

from ...robot_layout import TABLE_BOUNDS_XY_M, TABLE_TOP_Z_M
from .assets import PARAMETERS


LAYOUT_ID = 'mechanical_l3_table_layout_v1'
SHAFT_POSITION_M = (.40, 0., .78)
SUPPLY_REST_POSITIONS_M = {
    'spacer': (.25, -.27, .78),
    'wide_ring': (.25, 0., .78),
    'top_ring': (.25, .27, .78),
}
SUPPLY_POSITIONS_M = {
    name: (position[0], position[1], position[2] + PARAMETERS['supply_rest_height'])
    for name, position in SUPPLY_REST_POSITIONS_M.items()
}
PART_ORDER = ('spacer', 'wide_ring', 'top_ring')
SUPPLY_REST_MARGIN_M = .006
TARGET_HAND_CLEARANCE_M = .018


def _bounds(position, size):
    half = [float(value) / 2. for value in size]
    return ([float(position[index]) - half[index] for index in range(3)],
            [float(position[index]) + half[index] for index in range(3)])


def _overlap_xy(first, second):
    return all(first[0][index] < second[1][index]
               and second[0][index] < first[1][index] for index in (0, 1))


def layout_contract():
    """Return the immutable CPU contract for task-owned L3 placement.

    Positions intentionally match the canonical supply reset.  The contract
    makes the visual and grasp boundary explicit while keeping all physics
    tolerances in the task evaluator.
    """
    p = PARAMETERS
    return {
        'schema': 'mechanical_l3_table_layout.v1',
        'layout_id': LAYOUT_ID,
        'owner': 'task',
        'frame': 'world',
        'units': 'metres',
        'parent_scene_unchanged': True,
        'table_top_z_m': TABLE_TOP_Z_M,
        'table_bounds_xy_m': [list(value) for value in TABLE_BOUNDS_XY_M],
        'shaft_station': {
            'center_xy_m': list(SHAFT_POSITION_M[:2]),
            'fixture_position_m': list(SHAFT_POSITION_M),
            'axis': 'world_z',
            'entry_tip_z_m': TABLE_TOP_Z_M + p['workplate_height'] + p['shaft_exposed_height'],
            'target_hand_clearance_m': TARGET_HAND_CLEARANCE_M,
        },
        'supply_lane': {
            'axis': 'world_y',
            'rest_size_xy_m': list(p['supply_rest_size_xy']),
            'rest_height_m': p['supply_rest_height'],
            'positions_m': {name: list(value) for name, value in SUPPLY_REST_POSITIONS_M.items()},
            'part_positions_m': {name: list(value) for name, value in SUPPLY_POSITIONS_M.items()},
            'part_order': list(PART_ORDER),
            'separate_rest_per_part': True,
            'minimum_rest_edge_margin_m': SUPPLY_REST_MARGIN_M,
        },
        'visual_cues': {
            'part_labels': {'spacer': 'SPACER', 'wide_ring': 'WIDE RING', 'top_ring': 'TOP RING'},
            'part_colors': {'spacer': 'plastic_blue', 'wide_ring': 'plastic_yellow', 'top_ring': 'plastic_red'},
            'bore_visible_at_reset': True,
            'lower_entry_chamfer_visible': True,
            'shaft_tip_taper_visible': True,
            'visual_only': True,
        },
        'grasp_boundary': {
            'preferred_axis': 'world_z',
            'approach': 'top_down',
            'grasp_outside_bore': True,
            'minimum_radial_finger_clearance_m': TARGET_HAND_CLEARANCE_M,
            'part_zones': {
                name: {
                    'center_xy_m': list(SUPPLY_POSITIONS_M[name][:2]),
                    'outer_radius_m': p[name + '_outer_diameter_height'][0] / 2.,
                    'bore_radius_m': p['part_bore_diameter'] / 2.,
                    'inner_exclusion_radius_m': p['part_bore_diameter'] / 2.,
                    'minimum_finger_radius_m': p['part_bore_diameter'] / 2. + TARGET_HAND_CLEARANCE_M,
                    'outer_contact_radius_m': p[name + '_outer_diameter_height'][0] / 2.,
                    'outer_clearance_radius_m': p[name + '_outer_diameter_height'][0] / 2. + TARGET_HAND_CLEARANCE_M,
                    'release_height_above_target_m': .010,
                } for name in PART_ORDER
            },
            'allowed_part_reorientation': 'in_hand_only',
            'object_pose_writeback_forbidden': True,
            'grasp_weld_forbidden': True,
            'magnetic_snap_forbidden': True,
            'swept_path_native_validation_required': True,
            'endpoint_ik_is_not_reachability_proof': True,
        },
        'recovery': {
            'wrong_order': 'withdraw_the_top_part, restore_the_supply_lane, rebuild_bottom_to_top',
            'misalignment': 'lift_clear, regrasp, realign_with_shaft_tip, retry',
            'reset_is_in_episode_action': False,
            'allowed_operations': ['withdraw', 'regrasp', 'realign', 'retry'],
            'forbidden_operations': ['object_pose_playback', 'direct_object_force_or_torque', 'reset_mid_episode'],
        },
        'qualification': {
            'cpu_contract': True,
            'native_privileged_validation_required': True,
            'ordinary_agent_validation': False,
            'manipulation_qualified': False,
        },
    }


def validate_layout(scene, task):
    """Validate the composed reset against task-owned layout invariants.

    This is deliberately CPU-only.  It checks table bounds, non-overlapping
    supply rests, reset identity and contract consistency; it does not infer
    collision-free paths or native contact success.
    """
    contract = layout_contract()
    if task.get('task_id') != 'shaft_spacer_assembly':
        raise ValueError('L3 layout contract requires shaft_spacer_assembly')
    if scene.get('task_layout') != contract or task.get('task_layout') != contract:
        raise ValueError('L3 scene/task layout contracts differ from the pinned layout')
    if scene.get('workroom', {}).get('family') != 'mechanical_assembly':
        raise ValueError('L3 layout must retain the mechanical workshop parent')
    if scene.get('workroom', {}).get('version') != 'mechanical_workshop_20261017_v3':
        raise ValueError('L3 layout must retain the pinned workshop release')
    objects = {row.get('id'): row for row in scene.get('objects', [])}
    fixtures = {row.get('id'): row for row in scene.get('fixtures', [])}
    if set(objects) != set(PART_ORDER):
        raise ValueError('L3 layout requires exactly one object per ordered ring')
    expected_rests = {name + '_supply_rest' for name in PART_ORDER}
    if not expected_rests <= fixtures.keys():
        raise ValueError('L3 layout is missing one or more supply rests')
    if 'shaft_fixture' not in fixtures:
        raise ValueError('L3 layout is missing the fixed shaft fixture')
    rest_bounds = {}
    edge_margins = {}
    for name in PART_ORDER:
        rest = fixtures[name + '_supply_rest']
        part = objects[name]
        expected_rest = list(SUPPLY_REST_POSITIONS_M[name])
        expected_part = list(SUPPLY_POSITIONS_M[name])
        if rest.get('position_m') != expected_rest or part.get('position_m') != expected_part:
            raise ValueError('L3 supply reset differs from its task-owned layout')
        # Supply positions are the tabletop contact plane, while bounds use a
        # centered box. Keep the XY contract unchanged and place the box center
        # halfway above that plane so its reported bottom is the table top.
        rest_center = [expected_rest[0], expected_rest[1],
                       expected_rest[2] + PARAMETERS['supply_rest_height'] / 2.]
        bounds = _bounds(rest_center, [*PARAMETERS['supply_rest_size_xy'], PARAMETERS['supply_rest_height']])
        rest_bounds[name] = bounds
        table_bounds = contract['table_bounds_xy_m']
        if (bounds[0][0] < table_bounds[0][0] or bounds[1][0] > table_bounds[1][0]
                or bounds[0][1] < table_bounds[0][1] or bounds[1][1] > table_bounds[1][1]):
            raise ValueError('L3 supply rest leaves the pinned tabletop bounds')
        margins = [bounds[0][0] - table_bounds[0][0], table_bounds[1][0] - bounds[1][0],
                   bounds[0][1] - table_bounds[0][1], table_bounds[1][1] - bounds[1][1]]
        if min(margins) < SUPPLY_REST_MARGIN_M:
            raise ValueError('L3 supply rest is too close to a tabletop edge')
        edge_margins[name] = margins
    for index, first in enumerate(PART_ORDER):
        for second in PART_ORDER[index + 1:]:
            if _overlap_xy(rest_bounds[first], rest_bounds[second]):
                raise ValueError('L3 supply rests overlap: '+first+' / '+second)
    shaft = fixtures['shaft_fixture']
    if shaft.get('position_m') != list(SHAFT_POSITION_M):
        raise ValueError('L3 shaft station moved outside its task-owned position')
    evaluation = task.get('evaluation', {})
    if evaluation.get('part_order') != list(PART_ORDER):
        raise ValueError('L3 evaluation order differs from the physical order')
    expected_pairs = [['spacer', 'fixture'], ['wide_ring', 'spacer'], ['top_ring', 'wide_ring']]
    if evaluation.get('required_support_pairs') != expected_pairs:
        raise ValueError('L3 support chain differs from the physical order')
    return {
        'passed': True,
        'layout_id': LAYOUT_ID,
        'supply_rest_bounds_m': {name: [list(low), list(high)] for name, (low, high) in rest_bounds.items()},
        'supply_edge_margins_m': edge_margins,
        'supply_rests_non_overlapping': True,
        'all_supply_rests_inside_table': True,
        'part_order': list(PART_ORDER),
        'support_chain': expected_pairs,
        'scope': 'CPU reset/layout contract only; no native cooking, swept-path, grasp or contact qualification.',
    }


def privileged_phase_contract():
    """Return phases for the zero-model native privileged validation harness."""
    return {
        'schema': 'mechanical_l3_privileged_validation.v1',
        'model_calls': 0,
        'ordinary_agent_validation': False,
        'object_state_writeback': False,
        'external_object_forces': False,
        'phases': [
            {'id': 'reset', 'kind': 'native_reset_readback', 'writes_object_pose': False},
            {'id': 'ready', 'kind': 'privileged_robot_move', 'writes_object_pose': False},
            {'id': 'place_spacer', 'part': 'spacer', 'support': 'fixture', 'release': True},
            {'id': 'place_wide_ring', 'part': 'wide_ring', 'support': 'spacer', 'release': True},
            {'id': 'place_top_ring', 'part': 'top_ring', 'support': 'wide_ring', 'release': True},
            {'id': 'stability', 'kind': 'native_stability_window', 'duration_s': .5},
            {'id': 'recovery_wrong_order', 'kind': 'withdraw_realign_retry',
             'order': ['top_ring', 'wide_ring', 'spacer'], 'task_success_credit': False},
        ],
        'required_native_observations': [
            'complete_native_contacts', 'robot_reset_joints', 'object_reset_poses',
            'support_contacts', 'released_parts', 'stable_linear_and_angular_velocity',
            'fixture_static', 'robot_task_clearance',
        ],
        'forbidden_shortcuts': [
            'object_pose_playback', 'direct_object_force_or_torque', 'grasp_weld',
            'magnetic_snap', 'reset_mid_episode',
        ],
        'scope': 'Privileged native engineering validation entry; result is not an ordinary Agent score.',
    }


def cpu_reachability_targets():
    """Endpoint targets for the shared robot layout's CPU IK contract."""
    return {
        'layout_id': LAYOUT_ID,
        'approach_points_m': {
            name: [position[0], position[1], TABLE_TOP_Z_M + .10]
            for name, position in SUPPLY_POSITIONS_M.items()
        },
        'shaft_approach_m': [SHAFT_POSITION_M[0], SHAFT_POSITION_M[1], TABLE_TOP_Z_M + .18],
        'shaft_release_m': [SHAFT_POSITION_M[0], SHAFT_POSITION_M[1], TABLE_TOP_Z_M + .05],
        'wrist_orientation': 'shared mechanical_single_arm_v1 ready orientation',
        'scope': 'Discrete endpoint targets for CPU IK only; no swept-path or grasp qualification.',
    }


def validate_privileged_contract():
    """Validate static privileged phases before a native run is launched."""
    phases = privileged_phase_contract()
    ids = [phase['id'] for phase in phases['phases']]
    required = ['reset', 'ready', 'place_spacer', 'place_wide_ring', 'place_top_ring', 'stability']
    if ids[:len(required)] != required:
        raise ValueError('L3 privileged phases must follow reset, ready, and part order')
    if phases['model_calls'] != 0 or phases['ordinary_agent_validation']:
        raise ValueError('L3 privileged validation must not construct a model run')
    if any(phase.get('writes_object_pose') for phase in phases['phases']):
        raise ValueError('L3 privileged validation cannot write object poses')
    return {'passed': True, 'phase_ids': ids, 'model_calls': 0,
            'object_state_writeback': False, 'ordinary_agent_validation': False,
            'scope': phases['scope']}

