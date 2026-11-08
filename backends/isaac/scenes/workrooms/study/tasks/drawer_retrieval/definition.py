"""Task5: a real task-owned sliding drawer, retrieval and relaxed reclosure."""
from bench.paths import DATA_ROOT
from scenes.workrooms.study.task_design import prepared_task

TASK = prepared_task(
    task_id='drawer_retrieval', number=5, title='Task5 · 开抽屉取物并关回',
    objects={'intended_roles': ['sliding_drawer', 'notebook', 'fixed_cabinet', 'green_target_tray'],
             'inventory_counts': {'drawer': 1, 'notebook': 1},
             'mechanism': 'Native 160 mm prismatic rail; passive damping only.'},
    design={'skills': ['handle_grasp', 'constrained_sliding', 'occluded_object_retrieval',
                       'sequential_open_retrieve_close'],
        'reset': 'Closed tabletop drawer contains notebook; receiving tray is empty.',
        'success': 'Open at least 100 mm; retrieve notebook into tray; close within 12 mm; release and settle 300 ms.',
        'recovery': 'Regrasp, reopen, table-drop recovery and arbitrary notebook orientation are allowed.',
        'fixed_room_unchanged': True},
    release_version='v4',
    manifest_sha256='08e4a953d8f31f49619d0b8a10086a287c8a3b7a1b9b04d147bbd7d0ad4913f0',
)
TASK['evaluator']['implementation'] = 'scenes.workrooms.study.drawer_evaluation.SequenceEvaluator'
TASK['recommended_budgets'] = {'sim_time_s': 180., 'max_actions': None}
TASK['qualification'].update(
    status='scene_and_object_checks_passed', native_preflight=True, object_interactions=True,
    manipulation=False,
    evidence_path=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/study_drawer_20261021/qualification.json',
    evidence_sha256='007f7660f77fc0b55d5a36b3295591e5b0e941126f89034172a8d9ee6f951874',
    scope='Selected v2 canonical 1 s/2000 native steps/four RGB-D; separate 6.8 s loaded drawer slide and 1.2 s tray support probes passed. No robot extraction or full task success is claimed.',
)

TASK['qualification']['agent_bench_interface'] = True
TASK['qualification']['native_interface_evidence'] = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/three_room_expansion_20261021/study_http01/qualification.json',
 'sha256': '2866429467c293906f085d0878fe0092f9e6664ddea7ce80546e18ea0af30ef2',
 'release_manifest_sha256': 'bf0cb1bb098f276bc2ce912e030affa28e4bfe4fb38aa2416fd15383ec807874',
 'passed': True,
 'scope': 'Live HTTP and actual Agent client contract checks, four RGB-D cameras, read-only IK, joint '
          'micro-motion, return, hold and idempotency. No hand-object policy qualification.'}

TASK['qualification'] = {
    'status': 'native_interface_revalidated',
    'native_preflight': True, 'object_interactions': False,
    'agent_bench_interface': True, 'manipulation': False,
    'physics_accepted': False, 'release_status': 'prepared_unvalidated',
    'task_policy_success_claimed': False,
    'historical_v2_evidence': TASK['qualification'],
    'rule_repair': {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/all_task_rules_review_20261021/publication01.json',
        'sha256': '297e5f58bb2defcdbc183cb4348c037238161fabb6edab423dac8825175f23a3',
        'scope': 'V4 requires the whole notebook in the opened drawer before outward extraction credit. Public rules, scene, reset, physics and assets are byte-identical to v2.'},
    'native_interface_evidence': {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/rule_repairs_native_20261021/qualification.json',
        'sha256': '708761e0f305bbd516ce4e5b25169308e495f3072df55cae96c139f69694b43f',
        'release_manifest_sha256': '08e4a953d8f31f49619d0b8a10086a287c8a3b7a1b9b04d147bbd7d0ad4913f0',
        'passed': True,
        'scope': 'V4 native HTTP/Agent micro-motion, four RGB-D streams, 1600 native steps and exact 1601-frame frozen replay. Opening/extraction branches were not physically exercised.'},
    'full_rule_physical_revalidation_complete': False,
    'scope': 'V4 native initialization and interface revalidated; CPU counterexamples cover the repaired history logic. Prior v2 evidence stays separate. No complete robot extraction or manipulation qualification.',
}
