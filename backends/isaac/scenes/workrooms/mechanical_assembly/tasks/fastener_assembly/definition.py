"""L4: loose-bolt assembly with an immutable historical preinserted release."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.task_design import planned_task
from scenes.workrooms.mechanical_assembly.threaded_sharpa_catalog import threaded_sharpa_asset_bindings
from .rules import public_rules, thresholds


LOOSE_BOLT_RELEASE = {
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/releases/loose_bolt_candidate_v10_pgs',
    'manifest_sha256': 'e3fdb6d0ce747bcc0f4b4786dbe9fe4ad11cfd69ab4266919551820357e40a44',
}


TASK = planned_task(
    task_id='fastener_assembly', title='L4 · 散装螺栓装配', difficulty=4,
    intent='取起桌面螺栓并对准导向口起牙，再取工具插入六角孔，缓慢旋拧后松手稳定。',
    objects={'intended_roles': ['straight_hex_screwdriver', 'socket_head_bolt', 'fixed_threaded_fixture'],
             'inventory_counts': {'screwdriver': 1, 'bolt': 1, 'fixture': 1},
             'source_asset_bindings': threaded_sharpa_asset_bindings(), 'asset_bindings': None},
    design={
        'skills': ['bolt_pickup', 'thread_start', 'tool_pickup', 'tool_insertion',
                   'handle_grasp', 'slow_wrist_rotation', 'axial_following', 'inspect_thread_progress'],
        'existing_asset_geometry_m': {
            'screwdriver_length': .211, 'handle_diameter': .044, 'handle_grip_length': .110,
            'hex_bit_across_flats': .01875, 'bolt_socket_across_flats': .021,
            'per_side_geometric_clearance': .001125, 'socket_depth': .01095,
            'nominal_thread_lead': .003,
        },
        'historical_contact_qualification': {
            'asset_bundle': 'threaded_sharpa/v3',
            'report_path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/threaded_asset_native_20261016/screw_contact_acceptance.json',
            'report_sha256': '726025545af41c53f08ef62493ac7baa7721530540c0f9e2b08a0a8c25c8b3aa',
            'scope': 'Bounded external tool-wrench approach, thread advance, reverse and release with negative controls and 480/960 Hz comparison; no robot present.',
            'measured_drive_lead_m_per_revolution': .002999856770744052,
            'robot_grasp_or_task_accepted': False,
        },
        'reset_design': {
            'variant': 'loose_bolt',
            'bolt': 'Independent gravity-enabled rigid body supported by the original table inside a task-owned annular nest; not engaged.',
            'tool': 'Independent gravity-enabled rigid body on its separate desktop tool rest; not inserted.',
            'hand': 'Starts at the shared ready pose; approach and opening use robot joint actions, never a weld or initial attachment.',
            'validation_gate': 'Pickup, thread start and tool insertion must establish a settled contact-backed engaged baseline before advance can score.',
            'privilege': 'The shallow entry funnel only guides centering; no pre-engagement, preinsertion, snap or screw joint.',
        },
        'sequence': ['Pick up the loose bolt from its desktop supply rest.',
                     'Align it with the entry guide and start the thread.',
                     'Pick up the independent tool and insert it into the bolt socket.',
                     'Establish the settled engaged baseline, then turn slowly while following axial advance.',
                     'Release and retain the verified additional advance; no full tightening required.'],
        'success_design': {
            'all_of': ['Post-reset pickup, thread start and tool insertion establish a settled contact-backed baseline.',
                       'Held tool transfers torque to the bolt through hex contact.',
                       'Bolt rotation and downward travel agree with the measured thread handedness and lead.',
                       'The additional progress occurs after reset; no initial-state completion credit.'],
            'candidate_turn_degrees': 30.,
            'advance_target': 'Frozen engineering candidate: 0.250 mm for 30 degrees at 3 mm/revolution; native task success remains unvalidated.',
            'final_torque_or_clamping_force_required': False,
        },
        'failure_design': ['Unrecoverable tool drop or cross-thread escape.', 'Budget exhausted without additional valid advance.'],
        'difficulty_control': 'Loose bolt and tool start separately; the shallow entry guide assists alignment without starting the thread or inserting the tool.',
        'later_extension': 'Full tightening and repeated finger gaiting are outside the short-advance goal; 1.125 mm side clearance remains contact-sensitive.',
        'historical_preinserted': {
            'release': {
                'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases/fastener_assembly',
                'manifest_sha256': '490a0ee47eb1132691515ff874d71cbd6ae71c02b8ccd36ebc4be4bb871456ca',
            },
            'scope': 'Historical pre-engaged bolt and preinserted tool. Its reset/contact evidence does not qualify the current loose-bolt task.',
        },
        'loose_bolt_candidate': {
            'status': 'prepared_unvalidated',
            'reset_variant': 'loose_bolt',
            'asset_root': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/fastener_assembly/loose_bolt_v5',
            'threaded_collision_candidate': {
                'asset_root': f'{DATA_ROOT}/assets/task_parts/threaded_sharpa/sdf1024_convex_partition_candidate02_20261018',
                'manifest_sha256': 'b81a9ea7e4003c72f8bb0f71ef607cb10235e68fb92df7732c42c98a42fa7850',
                'source_preserving': True,
                'native_geometry_evidence': 'mechanical_task_validation_20261018/fastener/native18',
                'scope': 'Candidate collision cooking only; loose pickup, start-thread and robot manipulation remain unqualified.',
            },
            'release': dict(LOOSE_BOLT_RELEASE),
            'task_owned_assets': ['mechanical_l4_thread_entry_guide', 'mechanical_l4_bolt_supply_rest',
                                  'mechanical_l4_tool_supply_rest'],
            'sequence': ['pickup_bolt', 'start_thread', 'insert_tool', 'settle', 'advance', 'release'],
            'entry_guide': {
                'kind': 'shallow_annular_thread_entry_funnel',
                'bottom_inner_radius_m': .013,
                'top_inner_radius_m': .019,
                'height_m': .006,
                'thread_joint': False,
                'pose_snap': False,
            },
            'native_validation_required': True,
        },
        'source_limitations': [
            'User selected the existing, historically contact-validated threaded_sharpa/v3 bundle. Preserve its exact files and scale; original threaded_tools/v1 remains unchanged.',
            'Bolt/fixture already contain 1.5x source transforms; do not scale them again. Driver is the existing 211 mm v3 geometry.',
            'Keep the v3 thread topology repairs, plate SDF and seven-piece bolt head. Do not substitute the unrepaired original colliders.',
            'Upstream asset revision 62fab5b70ccb0103c60c4ca39236abae2cc8e900; source commit 046f7bf48865f3d3b48e64614350777fb5d5d163.',
            'Tool is CC0; bolt/platform include separate upstream authored-part and NVIDIA thread terms. Preserve both provenance and licenses.',
            'Historical contact profile uses driver/bolt/fixture friction 0.6/0.01/0.75, solver 192/16, contact offset 0.3 mm and max depenetration speed 0.05 m/s; it is not the old uniform fallback profile.',
            'A separate authorized scripted contact diagnostic may apply bounded tool wrenches. Task execution remains robot joint actions only, with gravity enabled and no grasp weld.',
        ],
        'next_validation': ['Native loose-bolt reset and independent supply support; historical preinserted acceptance does not transfer.',
                            'Real bolt pickup, guided thread start, tool pickup/insertion, contact-backed short-arc advance and release.',
                            'Time-step/contact-offset convergence and sensitivity without a screw constraint.'],
    },
)

# Implementation and qualification are independent. This exact frozen release
# is loadable by the canonical worker; it has no native task acceptance yet.
TASK.update(
    status='implemented',
    entrypoint='scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.task',
    runtime_entrypoint='bench.runtime.service:EpisodeSession',
    release_schema='mechanical_fastener_task_release.v1',
    release_schema_version='isaacsim-mechanical-fastener-release-v1',
    source_release=dict(LOOSE_BOLT_RELEASE),
    rules={'intent': public_rules(30., reset_variant='loose_bolt')['instruction_zh'],
           'implementation': 'scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.rules',
           'contract': public_rules(30., reset_variant='loose_bolt')},
    evaluator={
        'implementation': 'scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.evaluator.SequenceEvaluator',
        'frozen_path': 'evaluator/sequence_evaluator.py',
        'thresholds': thresholds(30.),
        'scope': 'Post-reset loose-bolt pickup, thread start and tool insertion precede a settled engaged baseline; actual bolt rotation/lead and retained contact-backed net advance followed by release stability earn success.'},
    qualification={
        'release_status': 'prepared_unvalidated',
        'native_preflight': False, 'manipulation': False,
        'physics_accepted': False, 'task_policy_success_claimed': False,
        'scene_contact_evidence': [{
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261019/fastener/loose_scene_load_native07/report.json',
            'sha256': '78c14a05c6baa2583a390a49f935aa3068d34379d8e03918aab600cd4385ecd9',
            'scope': 'Native geometry witness only; the overall report is not a task pass.'}, {
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261019/fastener/l4_native_launcher_corrected01/result/report.json',
            'sha256': '20ff746566aa4906a7c1b31d16904f1016ba1e1b1777ec9c9bf8676693b1af5d',
            'scope': 'Historical failed loose-bolt robot phase probe; not manipulation acceptance.'}, {
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/contact_pgs_native01/result/report.json',
            'sha256': '9451127f26f30ed32ea33f6962963278e67c36e2bcd73a726c1327921e5e8f1b',
            'scope': 'PGS pre-engaged engineering contact diagnostic passes lead, passive stability and contact gates; no loose-bolt pickup, thread start or tool insertion.'}, {
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/contact_pgs_native01/complete_stream_motion_audit.json',
            'sha256': '248723ed8e38f7132cac9b55ee5bc95a685c1d6d4ce2f8b46c822e78a435989c',
            'scope': 'Read-only complete native and rolling 20 ms motion audit for the PGS engineering contact diagnostic; not robot manipulation acceptance.'}, {
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/pgs_http01/qualification.json',
            'sha256': '756258f9dfcb08c65e95e608e5d623d8bb33a3b3920a3a70e36ef37186f67132',
            'scope': 'PGS loose-bolt canonical reset and public HTTP interface smoke only; zero model calls and no assembly success.'}, {
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/pgs_replay01.json',
            'sha256': '5bc065107e8050f63b81c9089024014e5380888d9afe6e140b6c9680a0076535',
            'scope': 'Read-only frozen referee replay of the PGS HTTP smoke; no new native run or task credit.'}, {
            'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_scene_delivery_20261020/l4/pgs_support01.json',
            'sha256': '806ccc268e36f831f83afbf0d3d43e385769f333eb7cc881c01e4917571072f9',
            'scope': 'Read-only complete native supply-support audit of the PGS HTTP smoke; canonical loose reset only, not pickup or assembly.'}],
        'scope': 'Current catalog selects loose-bolt candidate v10 PGS with the v9 assets, reset, rules, thresholds and 0.5 ms timestep; only the scene solver selection changes. Its pre-engaged engineering contact, complete motion audit and canonical HTTP smoke pass within their separate scopes. Historical TGS failures and old preinserted evidence remain immutable. Pickup, thread start, tool engagement, advance and release remain unqualified.'},
)
TASK.pop('cannot_run_reason')
TASK['objects']['asset_bindings'] = {
    'frozen_scene': TASK['source_release']['path']+'/scene.json',
    'frozen_inventory': TASK['source_release']['path']+'/asset_file_hashes.json'}
TASK['layout']['initialization'] = 'Frozen scene.json plus complete initial_state.json; reset only.'
TASK['design']['status'] = 'frozen_candidate_with_separate_scene_evidence'
TASK['design']['dimensions_and_thresholds'] = 'Version-pinned engineering candidates, not native-accepted task tolerances.'
