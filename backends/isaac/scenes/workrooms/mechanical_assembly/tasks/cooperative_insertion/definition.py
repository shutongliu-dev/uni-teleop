"""Task10: one Agent with official left/right SHARPA; qualification stays explicit."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.cooperative_rules import public_rules,thresholds
TASK={
    'schema':'workroom_task_definition.v1','family':'mechanical_assembly','task_id':'cooperative_insertion',
    'task_number':10,'difficulty':10,'difficulty_calibrated':False,'title':'Task10 · 双臂协作插接',
    'status':'implemented','entrypoint':'scenes.workrooms.mechanical_assembly.tasks.cooperative_insertion.task',
    'release_schema':'mechanical_cooperative_task_release.v1',
    'release_schema_version':'isaacsim-mechanical-cooperative-release-v1',
    'objects':{'intended_roles':['movable_sleeve','independent_peg','collection_tray'],
        'asset_root':f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/cooperative_insertion/v4',
        'background':'31 fixed edge supply groups; not targets or scoring supports'},
    'layout':{'frame':'world','workspace':'mechanical_table_shared_center','parent_scene':'mechanical_assembly/scene.py',
        'same_environment_for_all_tasks':True,'initialization':'Independent supply parts; two original FR3s on the same table edge with official left/right SHARPA, controlled by one Agent'},
    'robot_requirements':{'arm':'fr3','hand_id':'dual_fr3_sharpa_w01','assembly_version':'isaacsim_6_1_v2',
        'articulations':2,'active_sides':['left','right'],
        'layout_id':'mechanical_same_side_left_right_sharpa_v1'},
    'rules':{'intent':public_rules()['instruction_zh'],
        'implementation':'scenes.workrooms.mechanical_assembly.tasks.cooperative_insertion.rules','contract':public_rules()},
    'evaluator':{'implementation':'scenes.workrooms.mechanical_assembly.cooperative_evaluation.SequenceEvaluator',
        'frozen_path':'evaluator/sequence_evaluator.py','thresholds':thresholds(),
        'scope':'Native distinct-hand cooperation, relative insertion, real support and released stability; retries allowed'},
    'source_release':{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/cooperative_insertion/v6',
 'manifest_sha256': 'd4df70390f901cfe1b61e3426f683d54efce06fa75315928707fabed249455b7'},
    'qualification':{'native_preflight': True,
 'object_interactions': True,
 'manipulation': False,
 'physics_accepted': False,
 'release_status': 'prepared_unvalidated',
 'agent_bench_interface': True,
 'outside_entry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_coop_entry_20261021/qualification.json',
                            'sha256': '8c33446448f008fd018af559a360339a9a127bf96ffa9e2e2d7e7bc6d6d2ea8d',
                            'passed': True,
                            'release_manifest_sha256': 'b653f4e8c5fd6667aded66546fa1372e0a0c684989b7ae1beb674c5e7659e60f',
                            'scope': 'Centered aligned gravity-only entry from10mm above the actual mouth '
                                     'to52mm depth;3s/6000 native frames;real support and300ms released '
                                     'rest;no robot insertion claim.',
                            'current_asset_applicability': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task10_finish_20261021/independent_audit/asset_v4.json',
                                                            'sha256': 'a38b06ece07f8ded98d2eff32d5460775ab5907f61589fd5ed5bd5a98e7248df',
                                                            'passed': True,
                                                            'scope': 'Actual physical task '
                                                                     'geometry/material/mass/reset identical '
                                                                     'to v3 (and previous identity-linked '
                                                                     'entry evidence); no new insertion '
                                                                     'execution.'}},
 'scope': 'Scene/object feasibility and ordinary Agent/Bench interface only; hand-object interaction belongs '
          'to data collection',
 'historical_opposed_layout_evidence': {'scene_contact_evidence': [{'kind': 'canonical_native_initialization',
                                                                    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_coop_20261021/canonical03/qualification.json',
                                                                    'sha256': 'c8d0cf44f6b535e2eb2c752c60dd26abd81f553dce3238368b96d521fd34b9e4',
                                                                    'passed': True,
                                                                    'scope': 'Previous v1 referee release, '
                                                                             'identical physical assets: two '
                                                                             'original fixed 29 DOF '
                                                                             'robots;1.2s/2400steps; five '
                                                                             'RGB-D views. V2 native '
                                                                             'scorer/interface check is '
                                                                             'separate.'},
                                                                   {'kind': 'passive_object_insertion',
                                                                    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_coop_20261021/drop_insert01/qualification.json',
                                                                    'sha256': '7b7daa31260b49d07958bd984592b033596e69a0c6ac7ff3561b064fd95a9cb1',
                                                                    'passed': True,
                                                                    'scope': 'Gravity-driven continuation '
                                                                             'from12mm already inserted '
                                                                             'to52mm, independent parts and '
                                                                             'released tray support; no '
                                                                             'outside-entry or robot grasp '
                                                                             'claim.'},
                                                                   {'kind': 'passive_insertion_full_trace_review',
                                                                    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_coop_20261021/passive_insertion_review02.json',
                                                                    'sha256': 'e212f65b60b94d41d848f1d4ddbb4e3c42f0cf6fc722ff2c44ddd9f17735d877',
                                                                    'passed': True,
                                                                    'scope': 'Saved2400native frames '
                                                                             'prove39.9999mm further '
                                                                             'insertion from12mm initial '
                                                                             'depth; no postreset pose '
                                                                             'writes or external loads, and '
                                                                             'no outside-entry claim.'}],
                                        'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/three_room_expansion_20261021/cooperative_http02/qualification.json',
                                                                      'sha256': '6fe662fe34bd6447e7fb749b889c71a8fd8bca1bc604ff4b25667d921f488239',
                                                                      'passed': True,
                                                                      'release_manifest_sha256': 'b653f4e8c5fd6667aded66546fa1372e0a0c684989b7ae1beb674c5e7659e60f',
                                                                      'scope': 'V2 live HTTP and actual '
                                                                               'Agent clients; same-action '
                                                                               'front/rear wrist motion, '
                                                                               'return, hold and '
                                                                               'replay;0.8s/1600 native '
                                                                               'steps;58 joints;five RGB-D '
                                                                               'views;no task credit or '
                                                                               'manipulation claim.'},
                                        'qualification_summary': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_coop_20261021/qualification_v2.json',
                                                                  'sha256': '65ab9bb743cad3d54f1638e14f9e017368ea0aed8bc2b07c29abaab30d2547f2'}},
 'historical_same_side_dual_right_evidence': {'scene_contact_evidence': [{'kind': 'canonical_native_support_and_runtime_layout',
                                                                          'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_cooperative_detail_20261021/http01_review.json',
                                                                          'sha256': '7883e9d9624554048dad4253eba1da66e681dc1099987533f9e684d1b6f698f6',
                                                                          'passed': True,
                                                                          'scope': 'Current v3: two '
                                                                                   'same-side original fixed '
                                                                                   '29 DOF roots;1600 '
                                                                                   'consecutive native '
                                                                                   'steps;300ms real '
                                                                                   'tabletop support;five '
                                                                                   'RGB-D views; no robot '
                                                                                   'manipulation claim.'},
                                                                         {'kind': 'unchanged_task_physics_identity',
                                                                          'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_cooperative_detail_20261021/final_asset_audit01/audit.json',
                                                                          'sha256': '74d18696b7336733accf38d0caf2b37b33641e8980daaa3a3fac7587160a519f',
                                                                          'passed': True,
                                                                          'scope': 'Current asset v2 task '
                                                                                   'colliders/mass/inertia/materials/reset '
                                                                                   'match v1 exactly; visual '
                                                                                   'details only and '
                                                                                   'relocated stands; '
                                                                                   'historical insertion '
                                                                                   'evidence is not a new '
                                                                                   'run.'}],
                                              'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_cooperative_detail_20261021/http01/qualification.json',
                                                                            'sha256': '219735d1e4787dc2d85ca696be6e4c8660395c863152ddbdc4ab194afe09b791',
                                                                            'passed': True,
                                                                            'scope': 'Current v3 one-Agent '
                                                                                     'live HTTP and actual '
                                                                                     'Agent clients; '
                                                                                     'same-action dual-wrist '
                                                                                     'motion/return/hold/replay;0.8s/1600steps;58 '
                                                                                     'joints;five RGB-D '
                                                                                     'views.',
                                                                            'release_manifest_sha256': 'b690d3c591ab657fa50a6a70746cc3a75fb84b9e6f465ab10534df7be5f14759'},
                                              'qualification_summary': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_cooperative_detail_20261021/qualification.json',
                                                                        'sha256': '1617d3f25b5158044b07cfe90d74ae061ec3cdda2c7921114576ba8f07882594',
                                                                        'passed': True,
                                                                        'scope': 'Current v3 same-side '
                                                                                 'layout, detailed '
                                                                                 'appearance, scene '
                                                                                 'initialization/support and '
                                                                                 'one-Agent dual-hand '
                                                                                 'interface; full '
                                                                                 'manipulation remains '
                                                                                 'false.'}},
 'historical_handedness_v4_evidence': {'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_cooperative_handedness_20261021/http01_review.json',
                                                                   'sha256': '9cf4df8f1f66219bd88e09f6d09ad5d18e0ef50d4e9c09e6e1c4d5b09938bd50',
                                                                   'kind': 'native_left_right_support_drives_and_layout',
                                                                   'passed': True,
                                                                   'scope': 'Current v4 real left/right '
                                                                            'hands;58 DOF;1600 consecutive '
                                                                            'native steps;300ms real '
                                                                            'tabletop support;five '
                                                                            'RGB-D;unchanged runtime '
                                                                            'drives;no hand-object '
                                                                            'operation.'},
                                                                  {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_cooperative_handedness_20261021/asset_v3_audit.json',
                                                                   'sha256': '676db1f6206a2549fcbdea2b9648f50ddaf66c3f719d6392c7c5a9c5a92ea67e',
                                                                   'kind': 'unchanged_task_physics_identity',
                                                                   'passed': True,
                                                                   'scope': 'All task physical '
                                                                            'geometry/mass/inertia/material/reset '
                                                                            'identical to v2; historical '
                                                                            'insertion applicability only.'}],
                                       'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_cooperative_handedness_20261021/http01/qualification.json',
                                                                     'sha256': '1cb14bc132007c258f85804bd298a2a144ecf0bdec4e0e97862aa2f7e6b2d555',
                                                                     'passed': True,
                                                                     'release_manifest_sha256': '75ba29c5324c65331bf53b65564d140b1530230f8e7c88a365d5d458bcaa79d2',
                                                                     'scope': 'Current v4 one-Agent '
                                                                              'left/right same-action '
                                                                              'motion/return/hold/replay;0.8s/1600steps;actual '
                                                                              'Agent clients;five RGB-D.'},
                                       'qualification_summary': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_cooperative_handedness_20261021/qualification.json',
                                                                 'sha256': '572e3ccbebd1039d21e6e3ebf7d3bcdaf14819c87cec48a461ee4f18a2679a7d',
                                                                 'passed': True,
                                                                 'scope': 'Real left/right installation, '
                                                                          'scene initialization/support and '
                                                                          'one-Agent dual-hand interface; '
                                                                          'full manipulation remains '
                                                                          'false.'}},
 'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task10_finish_20261021/http01_review03.json',
                             'sha256': 'cc72a7ed2b5be932f29b05aa03dff33f836ae17a73535b46e890480bdaccdf6c',
                             'kind': 'current_native_initialization_and_support',
                             'passed': True,
                             'scope': 'Current v5 one-Agent left/right motion, return and hold;1600 native '
                                      'steps;300ms real tabletop support;no grasp/assembly.'},
                            {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task10_finish_20261021/independent_audit/asset_v4.json',
                             'sha256': 'a38b06ece07f8ded98d2eff32d5460775ab5907f61589fd5ed5bd5a98e7248df',
                             'kind': 'unchanged_task_physics_identity',
                             'passed': True,
                             'scope': 'Actual physical task geometry/mass/inertia/material/reset identical '
                                      'to v3 asset; preserves historical object-interaction applicability.'}],
 'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task10_finish_20261021/http01/qualification.json',
                               'sha256': '1c3ff029a504932cd501faff27d82a51bbe013de608e3f1bfd69cc6841465b58',
                               'passed': True,
                               'release_manifest_sha256': 'd17b09a55f05110ec280e3e158bce1b205712fa219b718a0fdc3db97732a565f',
                               'scope': 'Current v5 one-Agent58DOF same-action left/right '
                                        'micro-motion,return,hold and idempotent replay;five RGB-D;catalog '
                                        'pin verified.'},
 'qualification_summary': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task10_finish_20261021/qualification.json',
                           'sha256': '11f86bc2548d9de21548cfa1fd9947d1c66fe4a840045a523940cb4c1da47ebf',
                           'passed': True,
                           'scope': 'Prop appearance, native initialization/support and one-Agent dual-hand '
                                    'interface; full manipulation remains false.'}},
    'validation':{'privileged_entrypoint':'scripts.validation.mechanical_cooperative_release:run',
        'hand_object_interaction_owner':'data_collection','task_policy_success_claimed':False},
}

TASK['qualification'] = {
    'status': 'native_interface_revalidated',
    'native_preflight': True, 'object_interactions': False,
    'agent_bench_interface': True, 'manipulation': False,
    'physics_accepted': False, 'release_status': 'prepared_unvalidated',
    'historical_v5_evidence': TASK['qualification'],
    'rule_repair': {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/all_task_rules_review_20261021/publication01.json',
        'sha256': '297e5f58bb2defcdbc183cb4348c037238161fabb6edab423dac8825175f23a3',
        'scope': 'V6 debits insertion withdrawals across contact gaps; positive progress still requires consecutive distinct-hand contact. Public rules, physical scene, reset, task and assets remain unchanged.'},
    'native_interface_evidence': {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/rule_repairs_native_20261021/qualification.json',
        'sha256': '708761e0f305bbd516ce4e5b25169308e495f3072df55cae96c139f69694b43f',
        'release_manifest_sha256': 'd4df70390f901cfe1b61e3426f683d54efce06fa75315928707fabed249455b7',
        'passed': True,
        'scope': 'V6 native HTTP/Agent simultaneous dual-wrist motion, five RGB-D streams, 1600 native steps and 41 exact scorer-summary replays. No hand-object contact or physical retreat branch coverage.'},
    'full_rule_physical_revalidation_complete': False,
    'scope': 'V6 native initialization and interface revalidated; CPU counterexamples cover the repaired history logic. Prior v5 evidence stays separate. No full native observer replay or robot assembly qualification.',
}
