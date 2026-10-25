"""L5: three gears and open tool selection, with actual contact transmission."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.task_design import planned_task
from scenes.workrooms.mechanical_assembly.threaded_sharpa_catalog import threaded_sharpa_asset_bindings
from .rules import public_rules, thresholds


INSET_RELEASE = {
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/practical_rules_v4/release',
    'manifest_sha256': '62bd44368d8d42a36885d29272d7ef0dae2582acf84da2f8faee6d6fe37c96f5',
}


TASK = planned_task(
    task_id='gear_train_assembly', title='L5 · 多齿轮装配与自主选工具', difficulty=5,
    intent='把三个齿轮装到标记的轴位上，再从桌上工具中自行选择并使用工具，'
           '从输入端缓慢驱动，让输出齿轮的标记转到目标区域。',
    objects={
        'intended_roles': ['input_gear', 'idler_gear', 'output_gear', 'fixed_three_shaft_workplate',
                           'socket_crank', 'open_end_wrench', 'straight_hex_screwdriver'],
        'inventory_counts': {'gear': 3, 'tool': 3},
        'asset_bindings': {'screwdriver': threaded_sharpa_asset_bindings()['screwdriver'],
                           'gears_fixture_and_other_tools': None},
    },
    design={
        'skills': ['assemble_multiple_parts', 'recover_tooth_phase', 'compare_tool_affordances',
                   'grasp_and_engage_tool', 'slow_short_arc_drive', 'inspect_transmission'],
        'candidate_gear_geometry': {
            'type': 'external_involute_spur', 'count': 3, 'teeth_in_order': [20, 32, 20],
            'module_m': .005, 'pressure_angle_degrees': 20., 'profile_shift': 0.,
            'pitch_diameters_m': [.100, .160, .100], 'outside_diameters_m': [.110, .170, .110],
            'root_diameters_m': [.0875, .1475, .0875], 'adjacent_axis_distance_m': .130,
            'face_width_m': .018, 'bore_diameter_m': .024, 'shaft_diameter_m': .0234,
            'bore_entry_diameter_m': .038, 'entry_chamfer_depth_m': .010,
            'shaft_tip_diameter_m': .010, 'shaft_tip_taper_height_m': .016,
            'candidate_pair_tangential_backlash_m': .0008,
            'raised_grip_hub_diameter_m': .050, 'raised_hub_height_above_gear_m': .022,
            'input_drive_hex_across_flats_m': .054,
            'derivation_source': 'https://khkgears.net/gear-knowledge/gear-technical-reference/calculation-gear-dimensions/',
            'scope': 'Standard size relations only. Custom bore fit, backlash, mass and contact geometry are unvalidated design choices.',
        },
        'candidate_layout': {
            'revision': 'table_inset_v1',
            'shaft_centers_xy_m': [[.40, -.13], [.40, 0.], [.40, .13]],
            'fixture_footprint_xy_m': [.200, .410],
            'supply_centers_xy_m': [[.225, -.20], [.28, .34], [.225, .10]],
            'minimum_table_edge_clearance_m': .020,
            'minimum_fixture_clearance_m': .020,
            'tool_layout': 'Separate free tools directly on the table, with a 0.2 mm initial mesh-bottom gap; no supply pads.',
        },
        'supply_layout_candidate': {
            'revision': 'table_inset_v1',
            'release': dict(INSET_RELEASE),
            'cpu_audit': {
                'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261019/gear/supply_layout_inset01/report.json',
                'sha256': '7c246f7a66fb8dd6f44b699442109e1044d47f60ea1da052d7366a677d2ba82b'},
            'source_geometry': 'Unchanged v3_4 assets; only three gear supply positions change.',
            'status': 'prepared_unvalidated',
            'scope': 'CPU source bounds, table/fixture margins and wrist endpoint IK only; native support and manipulation remain unqualified.'},
        'reset_design': 'All three gears and tools start as separate free bodies directly on the original table, with 0.2 mm initial bottom clearance. Only the functional shaft workplate and robot stand remain; no gear is pre-mounted.',
        'assembly_design': [
            'Fixed upward shafts and broad entry chamfers help initial centering; final running clearance stays narrow enough to preserve mesh.',
            'Place one gear, then lower the next with small slow yaw adjustments or withdraw/retry if teeth meet tip-to-tip.',
            'Allow neighboring gears to rotate through real tooth contact; no phase snapping or motorized alignment.',
            'Support each gear on a fixed thrust surface. Keep bores, tooth flanks and root relief in the actual collision model.',
        ],
        'tools': {
            'selection_mode': 'open_effect_based', 'all_tools_free_rigid_bodies': True,
            'agent_goal_names_correct_tool': False,
            'options': [
                {'role': 'socket_crank', 'asset_status': 'authored_v3_4_candidate',
                 'affordance': 'Deep chamfered socket fits the input hex hub; a broad vertical handle permits a slow short arc.',
                 'candidate_socket_hub_af_m': [.058, .054], 'candidate_handle_diameter_m': .034,
                 'candidate_lever_radius_m': .070, 'candidate_socket_through_bore_diameter_m': .026,
                 'shaft_clearance': 'Open-center socket passes over the protruding fixed guide shaft; no closed cap that bottoms on the shaft.',
                 'drive_posture': 'Start with the lever toward the robot (world -X); follow a short position/yaw arc without regrasping.'},
                {'role': 'open_end_wrench', 'asset_status': 'authored_v3_4_candidate',
                 'affordance': 'Matching open jaws may drive the same hub, but lateral engagement and maintaining contact are harder.'},
                {'role': 'straight_hex_screwdriver', 'asset_status': 'pinned_threaded_sharpa_v3',
                 'existing_geometry_m': {'length': .211, 'handle_diameter': .044,
                                         'hex_bit_across_flats': .01875},
                 'affordance': 'This hex bit does not directly mate with the large external hex hub.',
                 'qualification': 'Existing scripted screw-contact tests do not establish robot grasp or gear-tool use.'},
            ],
            'engineering_hypothesis': 'The broad socket crank is the easiest candidate; this is a design hypothesis, not a scored tool-ID answer or an Agent instruction.',
            'alternative_tools': 'Any available tool may succeed if held and actually transmitting contact torque at the input hub.',
            'optional_later_distractor': 'A rubber mallet only if a separately checked layout provides enough free space; striking is not required.',
        },
        'success_design': {
            'all_of': ['All three gears are fully supported on the intended shafts without hand support.',
                       'A held free tool contacts the input drive feature and causes the motion.',
                       'Two simultaneous physical gear meshes transmit the motion: idler/input = -20/32, output/input = +1.',
                       'Output marker reaches the target sector through continuous observed transmission, without directly driving output or idler.'],
            'candidate_output_turn_degrees': 30., 'candidate_target_half_width_degrees': 5.,
            'candidate_max_drive_speed_degrees_per_second': 15.,
            'tool_id_affects_score': False,
            'reject': ['An assembly-only final pose.', 'Merely choosing or naming a tool.',
                       'Turning only the input while other gears slip or skip.', 'Directly pushing the output into the target sector.'],
        },
        'failure_design': ['Unrecoverable part/tool drop.', 'Budget exhausted without contact-backed transmission.'],
        'difficulty_control': '20/32/20 teeth at unchanged module, with a visibly larger middle gear; no keyed shaft, retaining ring, fastening, powered tool, continuous full turn or torque target. Slow phase correction is allowed.',
        'remaining_difficulty': 'Three placements plus real tool pickup, socket engagement and a constrained arc remain harder than L1–L4; broad entries do not eliminate contact/control difficulty.',
        'next_validation': ['Native tooth/bore cooking and center-distance/clearance/backlash compatibility.',
                            'No skipped teeth, penetration or binding over a short arc and small parameter perturbations.',
                            'Full hand/tool collision clearance, real tool grasp and hub contact at every arc point.',
                            'Independent tool-contact and pairwise angle/contact evidence; no kinematic gear relation.'],
    },
)

# Runtime availability is independent of native contact and robot task
# qualification. The task still starts with all six free bodies unassembled.
TASK['rules'] = {'intent': public_rules()['instruction_zh'],
    'implementation': 'scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.rules',
    'contract': public_rules()}
TASK['candidate_referee'] = {
    'implementation': 'scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.evaluator.SequenceEvaluator',
    'thresholds': thresholds(),
    'release': dict(INSET_RELEASE),
    'scope': 'The current inset-layout frozen referee; availability does not claim native transmission or robot task qualification.'}
TASK.update(
    status='implemented',
    entrypoint='scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.task',
    runtime_entrypoint='bench.runtime.service:EpisodeSession',
    release_schema='mechanical_assembly_task_release.v1',
    release_schema_version='isaacsim-mechanical-assembly-release-v1',
    source_release=dict(INSET_RELEASE),
    evaluator={
        'implementation': 'scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.evaluator.SequenceEvaluator',
        'frozen_path': 'evaluator/sequence_evaluator.py', 'thresholds': thresholds()},
    qualification={
        'release_status': 'prepared_unvalidated', 'native_preflight': False,
        'manipulation': False, 'physics_accepted': False, 'task_policy_success_claimed': False,
        'scene_contact_evidence': [{
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/qualification01.json',
            'sha256': '47b87baa65630b4314b18820877dd60516643f1f5945416618bb9c19b4412fab'}],
        'engineering_goal_validation_status': 'passed_with_operator_tool_boundary',
        'engineering_goal_evidence': {
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_gear_scene_validation_20261020/native09/result/report.json',
            'sha256': '8cad058c04210b137941e6482c7b314cc570ab414db31e35b145bea3342a7c19'},
        'free_dynamic_tool_goal': 'not_reached_in_bounded_load_probes',
        'scope': 'Controlled native socket motion transmitted through three passive gears to 27.314 degrees; actual tool withdrawal and passive gear rest passed. Rules v4 retain identical physics/reset/assets and allow normal backlash in final net ratios. No robot assembly, grasp, formal hand-held scoring or dynamic-tool release is qualified.'},
)
TASK.pop('cannot_run_reason')
TASK['objects']['asset_bindings'].update({
    'gears_fixture_and_other_tools': {
        'asset_root': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/v3_4',
        'manifest_sha256': 'f5ce40c0229d72b90834e3831fcead0ef1d56b826623c01afd66acbc606613f7'},
    'frozen_scene': INSET_RELEASE['path']+'/scene.json',
    'frozen_inventory': INSET_RELEASE['path']+'/asset_file_hashes.json'})
TASK['layout']['initialization'] = 'Frozen scene.json plus complete initial_state.json; reset only.'
TASK['design']['status'] = 'frozen_candidate_with_separate_scene_evidence'
TASK['design']['dimensions_and_thresholds'] = 'Frozen scoring contract; practical engineering scene evidence and robot task qualification are separate.'
TASK['validation']['privileged_entrypoint'] = 'scripts.validation.validate_gear_scene:run'
TASK['validation']['operator_motion_entrypoint'] = 'scripts.validation.validate_gear_driven_tool:run'
TASK['validation']['tool_contact_entrypoint'] = 'scripts.validation.validate_gear_tool_function:run'
TASK['validation']['hand_object_interaction_owner'] = 'data_collection; outside scene transmission acceptance'
TASK['qualification']['tool_contact_validation_status'] = 'passed_under_operator_tool_motion'
TASK['qualification']['tool_contact_evidence'] = [{
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_gear_tools_validation_20261020/transmission_qualification01.json',
    'sha256': '0a8aa97cd38c1e6c1bb221464bac690f57f5900b7f5ca586891f9bf0617f903b'}]
TASK['design']['failure_design'] = [
    'Invalid evidence and physical safety stops are reported separately.',
    'Budget exhaustion records an unfinished episode, not an invented irreversible task violation.']
