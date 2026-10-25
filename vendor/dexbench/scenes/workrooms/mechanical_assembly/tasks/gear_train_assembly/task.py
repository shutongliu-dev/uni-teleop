"""Task-owned L5 physical scene and separately labelled contact-test reset."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from itertools import product

import numpy as np

from bench.geometry import quaternion_matrix
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
from scenes.workrooms.mechanical_assembly.task_scene import empty_task_scene, DEFAULT_STAND_ROOT
from scenes.workrooms.mechanical_assembly.threaded_sharpa_catalog import ASSET_IDS
from .assets import ASSET_ROOT, TRAIN_GEOMETRY, GEAR_TEETH, gear_geometry
from scenes.workrooms.mechanical_assembly.referee import bind_task_rules
from scenes.workrooms.mechanical_assembly.scene import SCENE
from .rules import GEARS, public_rules, thresholds, geometry_contract


TASK_ID = 'gear_train_assembly'
THREADED_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/threaded_sharpa/sdf1024_convex_partition_candidate02_20261018')
THREADED_MANIFEST_SHA256 = 'b81a9ea7e4003c72f8bb0f71ef607cb10235e68fb92df7732c42c98a42fa7850'
SUPPLY_LAYOUT_REVISION = 'table_inset_v1'
SUPPLY_LAYOUTS = {
    SUPPLY_LAYOUT_REVISION: [[.225, -.20], [.28, .34], [.225, .10]],
    'legacy_v3_4': deepcopy(TRAIN_GEOMETRY['gear_supply_positions_xy_m']),
}


def yaw_quaternion(degrees):
    half = math.radians(degrees)/2
    return [math.cos(half), 0., 0., math.sin(half)]


def tabletop_position(asset, xy, quaternion):
    """Put the rotated asset's lower bound 0.2 mm above the original table.

    For the selected upright gears/tools and 90-degree sideways screwdriver,
    rotating these source AABB corners gives the actual lowest mesh height.
    No display pad or hidden support is introduced.
    """
    bounds = np.asarray(asset.get('aabb_m', [asset.get('bounds_min_m'), asset.get('bounds_max_m')]), float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
        raise ValueError('Finite ordered source bounds are required for tabletop placement')
    corners = np.array(list(product(*bounds.T)))
    bottom = float((corners @ quaternion_matrix(quaternion).T)[:, 2].min())
    return [*xy, .7802-bottom]


def _world_aabb(asset, position, quaternion):
    bounds = np.asarray(asset.get('aabb_m', [asset.get('bounds_min_m'), asset.get('bounds_max_m')]), float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
        raise ValueError('Finite ordered source bounds are required for reset auditing')
    corners = np.array(list(product(*bounds.T)))
    world = corners @ quaternion_matrix(quaternion).T + np.asarray(position, float)
    return world.min(axis=0), world.max(axis=0)


def reset_audit(scene, *, table_top_z=.78, minimum_gap_m=.0002,
                minimum_table_edge_clearance_m=.02, minimum_fixture_clearance_m=.02):
    """Validate the unassembled tabletop reset before publishing task inputs."""
    objects = scene.get('objects', [])
    ids = [row.get('id') for row in objects]
    if len(ids) != 6 or len(set(ids)) != len(ids):
        raise ValueError('L5 reset requires six uniquely identified free objects')
    table_center = np.asarray(SCENE['coordinates']['table_center_xy_m'], float)
    table_half_size = np.asarray(SCENE['coordinates']['table_size_xy_m'], float)/2
    table_low, table_high = table_center-table_half_size, table_center+table_half_size
    fixtures = {}
    for fixture in scene.get('fixtures', []):
        asset = load_task_asset(fixture['asset_id'], fixture['asset_root'], verify_hashes=True)
        fixtures[fixture['id']] = _world_aabb(asset, fixture['position_m'],
            fixture.get('quaternion_wxyz', [1., 0., 0., 0.]))
    if 'gear_workplate' not in fixtures:
        raise ValueError('L5 reset requires its fixed gear workplate')
    rows, failures = {}, []
    for row in objects:
        asset = load_task_asset(row['asset_id'], row['asset_root'], verify_hashes=True)
        low, high = _world_aabb(asset, row['position_m'], row['quaternion_wxyz'])
        gap = float(low[2] - table_top_z)
        edge_clearances = np.concatenate((low[:2]-table_low, table_high-high[:2]))
        edge_clearance = float(edge_clearances.min())
        inside_table = bool(edge_clearance >= -1.e-12)
        edge_safe = edge_clearance >= minimum_table_edge_clearance_m-1.e-12
        fixture_clearances = {name: float(np.linalg.norm(np.maximum(np.maximum(
            fixture_low[:2]-high[:2], low[:2]-fixture_high[:2]), 0.)))
            for name, (fixture_low, fixture_high) in fixtures.items()}
        fixture_safe = all(value >= minimum_fixture_clearance_m-1.e-12
                           for value in fixture_clearances.values())
        stable_gap = abs(gap - minimum_gap_m) <= 1.e-9
        finite = bool(np.isfinite(low).all() and np.isfinite(high).all())
        rows[row['id']] = {'world_bounds_m': [low.tolist(), high.tolist()],
                           'table_gap_m': gap, 'finite': finite,
                           'table_bounds': inside_table, 'exact_initial_gap': stable_gap,
                           'minimum_table_edge_clearance_m': edge_clearance,
                           'table_edge_clearance': bool(edge_safe),
                           'fixture_xy_clearances_m': fixture_clearances,
                           'fixture_clearance': bool(fixture_safe)}
        if not (finite and inside_table and stable_gap and edge_safe and fixture_safe):
            failures.append(row['id'])
    overlaps = []
    for i, first in enumerate(ids):
        low, high = np.asarray(rows[first]['world_bounds_m'])
        for second in ids[i+1:]:
            other_low, other_high = np.asarray(rows[second]['world_bounds_m'])
            if np.all(high > other_low) and np.all(other_high > low):
                overlaps.append([first, second])
    passed = not failures and not overlaps
    return {'schema': 'mechanical_l5_reset_audit.v2', 'passed': passed,
            'table_top_z_m': table_top_z, 'minimum_gap_m': minimum_gap_m,
            'table_bounds_xy_m': [table_low.tolist(), table_high.tolist()],
            'required_table_edge_clearance_m': minimum_table_edge_clearance_m,
            'required_fixture_clearance_m': minimum_fixture_clearance_m,
            'bounds_method': 'Conservative transformed source AABBs; fixture distances are in the XY plane.',
            'objects': rows, 'object_aabb_overlaps': overlaps,
            'failures': failures,
            'scope': 'CPU reset bounds only; no native settling, contact or velocity qualification.'}


def contact_probe_reset():
    """Preassembled diagnostic only: passive gears driven through a free tool.

    These poses may be selected at reset. They never award assembly credit and
    are not a motion trajectory or a post-reset pose/velocity writeback.
    """
    return {'scope': 'Privileged scene contact diagnostic; no robot assembly, grasp, tool choice or task-success credit',
        'object_overrides': {
            'input_gear': {'position_m': [.40, -.13, .8021], 'quaternion_wxyz': yaw_quaternion(0.)},
            'idler_gear': {'position_m': [.40, 0., .8021], 'quaternion_wxyz': yaw_quaternion(180./32)},
            'output_gear': {'position_m': [.40, .13, .8021], 'quaternion_wxyz': yaw_quaternion(0.)},
            'socket_crank': {'position_m': [.40, -.13, .8202], 'quaternion_wxyz': yaw_quaternion(0.)},
        },
        'loaded_body': 'socket_crank', 'all_gears_are_passive': True,
        'permitted_diagnostic_actuation': 'Bounded recorded external wrench on the actual free socket_crank only; no gear wrench or pose target',
        'expected_contact_pairs': [['socket_crank', 'input_gear'], ['input_gear', 'idler_gear'],
            ['idler_gear', 'output_gear'], *[[name, 'gear_workplate'] for name in ('input_gear', 'idler_gear', 'output_gear')]],
        'gear_teeth': deepcopy(GEAR_TEETH),
        'expected_signed_angle_ratios': deepcopy(TRAIN_GEOMETRY['signed_angle_ratios']),
        'nominal_pair_tangential_backlash_m': .0008,
        'nominal_pair_reversal_backlash_degrees': math.degrees(.0008/.05),
        'nominal_two_mesh_reversal_backlash_degrees': math.degrees(2*.0008/.05),
        'nominal_socket_hub_half_free_angle_degrees': 30.-math.degrees(math.acos(.029/(.027/math.cos(math.pi/6)))),
        'socket_backlash_scope': 'Independent centered 54 mm AF hex inside 58 mm AF socket; twice the half free angle on reversal. Not gear mesh backlash.',
        'backlash_scope': 'Normal fit clearance at nominal shaft centers; angles above are input/output-equivalent (20 teeth). The 32-tooth idler angular free play is 20/32 of that value. Native tolerance must separately account for measured bore/shaft displacement, contact discretization and elastic-free numerical behavior.',
        'prohibited': ['gear_ratio_constraint', 'hinge_constraint', 'gear_force_or_torque', 'post_reset_object_pose_writeback'],
        'native_physics_validated': False}


def build_task_inputs(*, asset_root=ASSET_ROOT, stand_root=DEFAULT_STAND_ROOT,
                      threaded_root=THREADED_ROOT, solver_type=None,
                      supply_layout_revision=SUPPLY_LAYOUT_REVISION):
    """Return scene/task/initial_state using the shared fixed robot and room."""
    asset_root, threaded_root = Path(asset_root).resolve(), Path(threaded_root).resolve()
    manifest_path = asset_root/'manifest.json'
    manifest = load_json(manifest_path)
    if manifest.get('source', {}).get('schema') != 'mechanical_gear_geometry_source.v1':
        raise ValueError('Expected the versioned L5 physical asset library')
    if manifest['source'].get('train_geometry') != TRAIN_GEOMETRY:
        raise ValueError('Expected the frozen v3 20/32/20 geometry and layout')
    if supply_layout_revision not in SUPPLY_LAYOUTS:
        raise ValueError('Unknown L5 task-owned supply layout revision')
    threaded_manifest = threaded_root/'manifest.json'
    if hashlib.sha256(threaded_manifest.read_bytes()).hexdigest() != THREADED_MANIFEST_SHA256:
        raise ValueError('Pinned source-equivalent SHARPA screwdriver catalog changed')
    threaded = load_json(threaded_manifest)
    scene, initial, ready = empty_task_scene(stand_root, solver_type=solver_type)
    common = {'asset_root': str(asset_root), 'friction': .3,
              'contact_offset_m': .0001, 'rest_offset_m': 0.,
              'collision_mesh_coordinate_scale': 1000.}
    def add(group, name, asset_id, position, quaternion=None, *, on_table=False):
        asset = load_task_asset(asset_id, asset_root, verify_hashes=True)
        quaternion = quaternion or [1., 0., 0., 0.]
        if on_table:
            position = tabletop_position(asset, position, quaternion)
        scene[group].append({**common, 'id': name, 'asset_id': asset_id, 'position_m': position,
                             'quaternion_wxyz': quaternion})
    add('fixtures', 'gear_workplate', 'gear_workplate', [.40, 0., .78])
    supplies = SUPPLY_LAYOUTS[supply_layout_revision]
    for i, (name, xy) in enumerate(zip(('input_gear', 'idler_gear', 'output_gear'), supplies)):
        add('objects', name, name, xy, yaw_quaternion((0., 180./32, 0.)[i]), on_table=True)
    add('objects', 'socket_crank', 'socket_crank', [.65, -.28], on_table=True)
    add('objects', 'open_end_wrench', 'open_end_wrench', [.80, .29], on_table=True)
    screwdriver = load_task_asset(ASSET_IDS['screwdriver'], threaded_root, verify_hashes=True)
    screwdriver_q = [2**-.5, 2**-.5, 0., 0.]
    scene['objects'].append({'id': 'straight_hex_screwdriver', 'asset_id': ASSET_IDS['screwdriver'],
        'asset_root': str(threaded_root), 'position_m': tabletop_position(screwdriver, [.68, .10], screwdriver_q),
        'quaternion_wxyz': screwdriver_q,
        **deepcopy(threaded['per_role_spec_options']['screwdriver'])})
    scene.update(description='L5: three separate bore-preserving involute gears, fixed guide shafts and three independent free tools.',
        scenario={'family': 'mechanical_assembly', 'task_id': TASK_ID, 'seed': 0,
                  'scope': 'Task-owned unassembled reset; native contact qualification pending'})
    scene['mechanical_task_provenance'].update(task_id=TASK_ID,
        asset_manifest_path=str(manifest_path), asset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        screwdriver_manifest_sha256=THREADED_MANIFEST_SHA256,
        screwdriver_collision_scope='Original v3 visual/solid union, mass, COM, inertia and material; shaft/handle convex partition and mesh-coordinate conditioning only. L5 native qualification required.',
        asset_geometry_modified_after_reset=False, no_task_joints=True,
        supply_layout_revision=supply_layout_revision,
        supply_layout_asset_geometry_unchanged=True)
    clearance = .02 if supply_layout_revision == SUPPLY_LAYOUT_REVISION else 0.
    audit = reset_audit(scene, minimum_table_edge_clearance_m=clearance,
                        minimum_fixture_clearance_m=clearance)
    if not audit['passed']:
        raise ValueError('L5 reset audit failed: '+json.dumps(audit, sort_keys=True))
    scene['mechanical_task_provenance']['reset_audit'] = audit
    task = {
        'schema_version': 'mechanical-gear-train-task-v1', 'id': TASK_ID, 'task_id': TASK_ID,
        'title_zh': '三齿轮装配与自主选工具', 'difficulty': 5, 'cloth': False,
        'steps': [{'id': 'assemble', 'instruction_zh': '把三个齿轮装到标记的轴位上'},
                  {'id': 'transmit', 'instruction_zh': '自行选择工具，从输入端缓慢驱动，让输出标记转到目标区域'}],
        'public_goal': {'object_id': 'output_gear', 'operation': TASK_ID, 'frame_id': 'world',
            'description': '完成三齿轮装配，用自主选择的工具从输入端驱动，令输出齿轮标记旋转约30度。'},
        'physical_geometry': {**deepcopy(TRAIN_GEOMETRY),
            'gear_supply_positions_xy_m': deepcopy(supplies),
            'gears': {name: gear_geometry(teeth) for name, teeth in GEAR_TEETH.items()}},
        'supply_layout': {'support': 'original_table', 'table_top_z_m': .78,
            'initial_bottom_gap_m': .0002, 'display_pads_or_supply_rests': False,
            'revision': supply_layout_revision, 'gear_supply_positions_xy_m': deepcopy(supplies),
            'minimum_table_edge_clearance_m': clearance, 'minimum_fixture_clearance_m': clearance,
            'source_asset_geometry_and_manifest_unchanged': True},
        'public_rules': public_rules(),
        'evaluation': {**thresholds(), **geometry_contract(), 'actors': {row['id']: '/World/Objects/'+row['id'] for row in scene['objects']},
            'fixture_actor': '/World/Fixtures/gear_workplate',
            'robot_actor_prefix': '/World/Robot',
            'initial_objects': {row['id']: {k: deepcopy(row[k]) for k in ('position_m', 'quaternion_wxyz')}
                for row in scene['objects']},
            'target_bottom_centers_m': {name: [*xy, .802] for name, xy in zip(GEARS, geometry_contract()['shaft_centers_world_xy_m'])},
            'nominal_output_turn_degrees': 30., 'candidate_target_half_width_degrees': 5.,
            'candidate_max_drive_speed_degrees_per_second': 15.,
            'seated_gear_bottom_z_m': .802, 'physics_dt_s': scene['isaacsim']['physics_dt_s'],
            'native_criteria_status': 'Pending independent native scene evidence and frozen evaluator release'},
        'initial_state': deepcopy(initial), 'dynamics_probe': contact_probe_reset(),
        'recommended_budgets': {'sim_time_s': 120., 'wall_time_s': 3600., 'max_actions': 6000},
        'ready_design': ready, 'physics_accepted': False, 'task_policy_success_claimed': False,
        'qualification_scope': 'New physical task candidate. CPU geometry and historical screwdriver evidence do not establish gear contact dynamics or robot assembly.'}
    if screwdriver.get('placement_form') != 'root':
        task['evaluation']['actors']['straight_hex_screwdriver'] += screwdriver['source_rigid_body_prim']
    task['evaluation']['original_screwdriver_actor'] = task['evaluation']['actors']['straight_hex_screwdriver']
    bind_task_rules(task)
    return {'scene': scene, 'task': task, 'initial_state': initial}
