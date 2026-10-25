"""Task-owned fastener resets and static fastener intent.

The historical preinserted reset remains the default.  ``loose_bolt`` is a
separate task-owned candidate with an independent bolt supply rest, shallow
entry funnel and independent tool pose; it never edits the fixed room or the
source threaded USDs.
"""
from copy import deepcopy
import math
from pathlib import Path

from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.referee import bind_task_rules
from .rules import public_rules, thresholds

TASK_ID = 'fastener_assembly'
TASK_SCHEMA = 'mechanical-fastener-task-v1'
STAND_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_workshop_supports/v1')
THREADED_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_workshop/threaded_v2')
LOOSE_ASSET_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/'
                        'fastener_assembly/loose_bolt_v5')
LOOSE_ASSET_IDS = {
    'entry_guide': 'mechanical_l4_thread_entry_guide',
    'supply_rest': 'mechanical_l4_bolt_supply_rest',
    'tool_rest': 'mechanical_l4_tool_supply_rest',
}
FIXTURE_TOP_OFFSET_M = .057
LOOSE_TOOL_POSITION_XY = [.29, .36]
LOOSE_TOOL_REST_POSITION_XY = [.29, .25]
LOOSE_BOLT_TABLE_CLEARANCE_M = .0002


def build_task_inputs(*, stand_root=STAND_ROOT, threaded_root=THREADED_ROOT,
                      target_turn_degrees=30., tool_clocking_degrees=0.,
                      reset_variant='preinserted', guide_root=LOOSE_ASSET_ROOT):
    """Return scene/task/initial_state; no simulator, source write or acceptance."""
    if reset_variant == 'loose_bolt':
        return build_loose_task_inputs(
            stand_root=stand_root, threaded_root=threaded_root,
            target_turn_degrees=target_turn_degrees, guide_root=guide_root)
    if reset_variant != 'preinserted':
        raise ValueError('Unknown fastener reset variant: '+str(reset_variant))
    from scenes.workrooms.mechanical_assembly.dynamics import preinserted_scene_v3
    from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset

    if (type(tool_clocking_degrees) not in (int, float)
            or not math.isfinite(tool_clocking_degrees) or not -30. <= tool_clocking_degrees <= 0.):
        raise ValueError('Reset tool clocking must be finite and in [-30, 0] degrees')
    scene, initial, _, _ = preinserted_scene_v3(stand_root, threaded_root)
    scene['objects'] = [row for row in scene['objects'] if row['id'] in ('bolt', 'screwdriver')]
    scene['description'] = 'Single-arm short fastener advance; task-owned pre-engaged/preinserted reset.'
    provenance = scene.pop('mechanical_diagnostic_provenance')
    scene['mechanical_task_provenance'] = {**provenance, 'task_id': TASK_ID,
        'removed_diagnostic_objects': ['cube_table', 'cube_floor'],
        'new_task_initialization_requires_native_validation': True}
    if tool_clocking_degrees:
        from bench.geometry import matrix_quaternion, quaternion_matrix, rotvec_matrix
        tool = next(row for row in scene['objects'] if row['id'] == 'screwdriver')
        original = deepcopy(tool['quaternion_wxyz'])
        tool['quaternion_wxyz'] = matrix_quaternion(quaternion_matrix(original)
            @ rotvec_matrix([0., 0., math.radians(tool_clocking_degrees)])).tolist()
        scene['mechanical_task_provenance']['tool_clocking_at_reset'] = {
            'degrees_about_tool_local_z': float(tool_clocking_degrees),
            'original_quaternion_wxyz': original,
            'reset_quaternion_wxyz': deepcopy(tool['quaternion_wxyz']),
            'tool_origin_and_axis_unchanged': True, 'bolt_reset_unchanged': True,
            'reason': 'Reduce the preinserted hex coupling free play for the short-turn task.',
            'task_credit': 0, 'new_native_reset_validation_required': True}
    # Pin the seed and variant at the ordinary service boundary.
    scene['scenario'] = {'family': 'mechanical_assembly', 'task_id': TASK_ID, 'seed': 0,
                         'scope': 'Frozen task initialization; no native qualification.'}
    fixture = next(row for row in scene['fixtures'] if row['id'] == 'threaded_fixture')
    actors = {}
    for spec in [*scene['objects'], fixture]:
        asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        wrapper = '/World/' + ('Fixtures/' if spec is fixture else 'Objects/') + spec['id']
        actors[spec['id']] = (wrapper if asset['placement_form'] == 'root'
                             else wrapper + asset['source_rigid_body_prim'])
    model = geometry_model(scene['robot'])
    joint_limits = {joint.name: [joint.lower, joint.upper] for joint in model.joints}
    rules = public_rules(target_turn_degrees)
    config = {**thresholds(target_turn_degrees), 'actors': actors,
        'fixture_position_m': fixture['position_m'],
        'fixture_quaternion_wxyz': fixture.get('quaternion_wxyz', [1., 0., 0., 0.]),
        'initial_objects': {row['id']: {key: deepcopy(row[key]) for key in ('position_m', 'quaternion_wxyz')}
                            for row in scene['objects']},
        'joint_limits_rad': joint_limits,
        'robot_actor_prefix': '/World/Robot/', 'hand_link_prefix': 'right_'}
    task = {'schema_version': TASK_SCHEMA, 'id': TASK_ID, 'task_id': TASK_ID,
        'title_zh': '螺丝刀短行程旋拧', 'difficulty': 'guided_single_arm', 'cloth': False,
        'steps': [{'id': 'settle', 'instruction_zh': '保持待命至少0.5秒，让预插组合稳定'},
                  {'id': 'advance', 'instruction_zh': f'握住工具，俯视顺时针缓慢转约{target_turn_degrees:g}度，使螺丝额外下行约{rules["success"]["nominal_additional_advance_m"]*1000:g}毫米'},
                  {'id': 'release', 'instruction_zh': '松开工具和螺丝，保持稳定至少0.5秒'}],
        'public_goal': {'object_id': 'bolt', 'operation': TASK_ID, 'frame_id': 'world',
                        'description': rules['instruction_zh']},
        'public_rules': rules, 'evaluation': config, 'initial_state': deepcopy(initial),
        'recommended_budgets': {'sim_time_s': 90., 'wall_time_s': 1800., 'max_actions': 4500},
        'evaluation_scope': rules['evidence_scope'],
        'training_policy_included': False, 'physics_accepted': False,
        'task_policy_success_claimed': False,
        'qualification_scope': 'Prepared task implementation; no public-tool/native grasp or screw-turn success established.'}
    if tool_clocking_degrees:
        task['reset_conditions'] = deepcopy(scene['mechanical_task_provenance']['tool_clocking_at_reset'])
    bind_task_rules(task)
    return {'scene': scene, 'task': task, 'initial_state': initial}


def build_loose_task_inputs(*, stand_root=STAND_ROOT, threaded_root=THREADED_ROOT,
                            target_turn_degrees=30., guide_root=LOOSE_ASSET_ROOT):
    """Compose the loose-bolt task candidate from unchanged source threaded assets.

    The bolt and driver begin as independent bodies on task-owned desktop
    rests.  A separate static funnel is placed over the fixture mouth; it is a
    coarse visual/physical lead-in and does not create a joint or snap.
    """
    from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from .loose_assets import ASSET_IDS as LOOSE_IDS
    from .rules import public_rules

    guide_root = Path(guide_root).resolve()
    guide_catalog = guide_root / 'manifest.json'
    if not guide_catalog.exists():
        raise FileNotFoundError('Loose fastener asset catalog is missing: '+str(guide_catalog))
    for asset_id in LOOSE_IDS.values():
        load_task_asset(asset_id, guide_root, verify_hashes=True)

    # Compose the historical room/robot/fixture exactly once, then replace
    # only the task-owned body poses and append task-owned static aids.
    result = build_task_inputs(stand_root=stand_root, threaded_root=threaded_root,
                               target_turn_degrees=target_turn_degrees,
                               tool_clocking_degrees=0., reset_variant='preinserted')
    scene, task, initial = (result[key] for key in ('scene', 'task', 'initial_state'))
    objects = {row['id']: row for row in scene['objects']}
    fixture = next(row for row in scene['fixtures'] if row['id'] == 'threaded_fixture')
    bolt_asset = load_task_asset(objects['bolt']['asset_id'], threaded_root, verify_hashes=True)
    tool_asset = load_task_asset(objects['screwdriver']['asset_id'], threaded_root, verify_hashes=True)
    supply_position = [.56, -.16, .78]
    # The bolt origin is the thread tip and its 180-degree X rotation places
    # the broad head over the table inside the open annular nest. The ring
    # provides lateral guidance; it is not a platform under the smaller head.
    bolt = objects['bolt']
    bolt['quaternion_wxyz'] = [0., 1., 0., 0.]
    bolt['position_m'] = [supply_position[0], supply_position[1],
                          supply_position[2] + float(bolt_asset['bounds_max_m'][2])
                          + LOOSE_BOLT_TABLE_CLEARANCE_M]
    # The tool lies independently on the table with its driver axis horizontal.
    tool = objects['screwdriver']
    tool['quaternion_wxyz'] = [2**-.5, 2**-.5, 0., 0.]
    tool['position_m'] = [*LOOSE_TOOL_POSITION_XY,
                          .78 - float(tool_asset['bounds_min_m'][1]) + .008]

    from scenes.workrooms.mechanical_assembly.assembly_geometry import physical_spec
    guide_position = [fixture['position_m'][0], fixture['position_m'][1],
                      fixture['position_m'][2] + FIXTURE_TOP_OFFSET_M]
    scene['fixtures'].append(physical_spec(
        LOOSE_IDS['entry_guide'], guide_root, 'thread_entry_guide', guide_position))
    scene['fixtures'].append(physical_spec(
        LOOSE_IDS['supply_rest'], guide_root, 'bolt_supply_rest', supply_position))
    scene['fixtures'].append(physical_spec(
        LOOSE_IDS['tool_rest'], guide_root, 'tool_supply_rest',
        [*LOOSE_TOOL_REST_POSITION_XY, .78]))
    scene['description'] = ('L4 loose-bolt candidate: independent desktop bolt and tool, '
                            'task-owned low supply rest and shallow threaded entry funnel.')
    provenance = scene.pop('mechanical_task_provenance')
    scene['mechanical_task_provenance'] = {
        **provenance, 'task_id': TASK_ID, 'reset_variant': 'loose_bolt',
        'removed_diagnostic_objects': ['cube_table', 'cube_floor'],
        'source_threaded_assets_unchanged': True,
        'task_owned_assets': ['thread_entry_guide', 'bolt_supply_rest', 'tool_supply_rest'],
        'bolt_supply_layout': {
            'vertical_support': 'original_workroom_table',
            'table_top_z_m': supply_position[2],
            'initial_head_bottom_clearance_m': LOOSE_BOLT_TABLE_CLEARANCE_M,
            'nest_role': 'passive lateral guidance only; open centre, no tool rails',
        },
        'tool_supply_layout': {
            'tool_origin_xy_m': deepcopy(LOOSE_TOOL_POSITION_XY),
            'tray_center_xy_m': deepcopy(LOOSE_TOOL_REST_POSITION_XY),
            'source_length_m': float(tool_asset['dimensions_m'][2]),
            'fixture_clearance_y_m': float(LOOSE_TOOL_POSITION_XY[1]
                                            - tool_asset['dimensions_m'][2]
                                            - fixture['position_m'][1]
                                            - .075),
            'support_contract': 'tool body spans the existing side rails; no pose lock or weld',
        },
        'entry_guide': {
            'kind': 'shallow_annular_thread_entry_funnel',
            'fixture_top_offset_m': FIXTURE_TOP_OFFSET_M,
            'pose_snap': False, 'thread_joint': False,
        },
        'new_task_initialization_requires_native_validation': True,
    }
    scene['scenario'] = {'family': 'mechanical_assembly', 'task_id': TASK_ID,
                         'seed': 0, 'reset_variant': 'loose_bolt',
                         'scope': 'Frozen loose-bolt candidate; no native qualification.'}
    actors = {}
    for spec in [*scene['objects'], fixture]:
        asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        wrapper = '/World/' + ('Fixtures/' if spec is fixture else 'Objects/') + spec['id']
        actors[spec['id']] = (wrapper if asset['placement_form'] == 'root'
                              else wrapper + asset['source_rigid_body_prim'])
    model = geometry_model(scene['robot'])
    joint_limits = {joint.name: [joint.lower, joint.upper] for joint in model.joints}
    rules = public_rules(target_turn_degrees, reset_variant='loose_bolt')
    config = {**thresholds(target_turn_degrees), 'actors': actors,
              'fixture_position_m': fixture['position_m'],
              'fixture_quaternion_wxyz': fixture.get('quaternion_wxyz', [1., 0., 0., 0.]),
              'initial_objects': {row['id']: {key: deepcopy(row[key])
                                              for key in ('position_m', 'quaternion_wxyz')}
                                 for row in scene['objects']},
              'joint_limits_rad': joint_limits, 'robot_actor_prefix': '/World/Robot/',
              'hand_link_prefix': 'right_', 'reset_variant': 'loose_bolt',
              'entry_guide_actor': '/World/Fixtures/thread_entry_guide',
              'supply_rest_actor': '/World/Fixtures/bolt_supply_rest',
              'tool_rest_actor': '/World/Fixtures/tool_supply_rest',
              'engagement_requires_post_reset_contacts': True}
    task = {'schema_version': TASK_SCHEMA, 'id': TASK_ID, 'task_id': TASK_ID,
            'title_zh': '螺丝桌面取放与导向旋拧', 'difficulty': 'guided_single_arm',
            'cloth': False,
            'steps': [
                {'id': 'pickup_bolt', 'instruction_zh': '从桌面供料托盘取起独立螺丝。'},
                {'id': 'start_thread', 'instruction_zh': '将螺丝对准导向口并完成起牙。'},
                {'id': 'insert_tool', 'instruction_zh': '取起螺丝刀并插入螺丝头六角孔。'},
                {'id': 'settle', 'instruction_zh': '保持接合至少0.5秒，让螺纹组合稳定。'},
                {'id': 'advance', 'instruction_zh': f'沿旋入方向缓慢转约{target_turn_degrees:g}度，'
                 f'使螺丝额外下行约{rules["success"]["nominal_additional_advance_m"]*1000:g}毫米。'},
                {'id': 'release', 'instruction_zh': '松开工具并保持稳定至少0.5秒。'},
            ],
            'public_goal': {'object_id': 'bolt', 'operation': TASK_ID, 'frame_id': 'world',
                            'description': rules['instruction_zh']},
            'public_rules': rules, 'evaluation': config,
            'initial_state': deepcopy(initial),
            'recommended_budgets': {'sim_time_s': 150., 'wall_time_s': 2400., 'max_actions': 7000},
            'evaluation_scope': rules['evidence_scope'], 'training_policy_included': False,
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'qualification_scope': 'Prepared loose-bolt candidate; native pickup, guidance and screw-turn validation required.'}
    task['reset_conditions'] = {
        'kind': 'loose_desktop_supply', 'reset_variant': 'loose_bolt',
        'bolt_supply_position_m': deepcopy(bolt['position_m']),
        'tool_supply_position_m': deepcopy(tool['position_m']),
        'entry_guide_id': 'thread_entry_guide', 'task_credit': 0,
        'pose_snap': False, 'object_pose_playback': False,
    }
    bind_task_rules(task)
    return {'scene': scene, 'task': task, 'initial_state': initial}
