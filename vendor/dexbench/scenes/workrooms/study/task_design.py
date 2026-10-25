"""Code-only study task metadata; exact reset and rules live in pinned releases."""
from copy import deepcopy

from .rules import THRESHOLDS, public_rules
from bench.paths import DATA_ROOT


COMMON_LAYOUT = {
    'frame': 'world', 'workspace': 'parent_scene_table_and_rear_shelves',
    'parent_scene': 'study/scene.py', 'same_environment_for_all_tasks': True,
    'robot_layout_id': 'mechanical_single_arm_v1',
    'robot_layout_module': 'scenes.workrooms.mechanical_assembly.robot_layout',
    'table_top_z_m': .78,
    'initialization': 'Pinned task USD subtree and complete initial_state.json; reset only.',
    'ownership': 'The fixed room owns empty surfaces; each task owns its movable props and robot stand.',
}
COMMON_ROBOT = {
    'arm': 'FR3', 'hand_id': 'sharpa_w01_right',
    'assembly_version': 'isaacsim_6_1_v2', 'articulations': 1,
    'dof_per_articulation': 29, 'base_constraint': 'native_fixed_base_v1',
    'layout_id': 'mechanical_single_arm_v1',
    'configuration_module': 'scenes.workrooms.mechanical_assembly.robot_layout',
    'base_position_m': [-.23, 0., .78],
    'constraint': 'All four study tasks use the same original robot, base and ready pose.',
}


def prepared_task(*, task_id, number, title, objects, design, release_version, manifest_sha256):
    """Bind one immutable engineering candidate without claiming task acceptance."""
    rules = public_rules(task_id)
    path = f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/study/' + task_id + '/' + release_version
    inventory = deepcopy(objects)
    inventory['asset_bindings'] = {
        'frozen_scene': path + '/scene.json',
        'frozen_inventory': path + '/asset_file_hashes.json',
    }
    return {
        'schema': 'workroom_task_definition.v1', 'family': 'study',
        'task_id': task_id, 'task_number': number, 'difficulty': number, 'title': title,
        'status': 'implemented', 'preparation_state': 'frozen_engineering_candidate',
        'entrypoint': 'scenes.workrooms.study.composition.build_task_inputs',
        'runtime_entrypoint': 'bench.runtime.service:EpisodeSession',
        'runtime_adapter': 'scenes.workrooms.study.runtime.build_runtime',
        'release_entrypoint': 'bench.isaacsim.release.task_inputs',
        'task_kind': 'single_agent',
        'release_schema': 'study_task_release.v1',
        'release_schema_version': 'isaacsim-study-release-v1',
        'objects': inventory, 'layout': deepcopy(COMMON_LAYOUT),
        'robot_requirements': deepcopy(COMMON_ROBOT),
        'rules': {
            'intent': rules['referee']['instruction_zh'],
            'implementation': f'scenes.workrooms.study.tasks.{task_id}.rules',
            'contract': rules, 'frozen_contract': path + '/rules.json',
            'contract_scope': 'Task overview only; exact identities, targets, budgets and rule hashes are in the pinned frozen_contract.',
        },
        'evaluator': {
            'implementation': 'scenes.workrooms.study.evaluation.SequenceEvaluator',
            'reference_implementation': f'scenes.workrooms.study.tasks.{task_id}.evaluator.SequenceEvaluator',
            'frozen_path': 'evaluator/sequence_evaluator.py',
            'thresholds': deepcopy(THRESHOLDS),
            'scope': 'Private complete native substeps feed the release-frozen evaluator; public claims never score.',
        },
        'source_release': {'path': path, 'manifest_sha256': manifest_sha256},
        'design': {
            'schema': 'study_task_design.v1', 'status': 'frozen_engineering_candidate',
            'difficulty_progression': 'Task1 sorting → Task2 stacking → Task3 retrieval → Task4 shelving; design order, not measured success rates.',
            'physical_contract': {
                'actions': 'Original robot joints only after reset.',
                'forbidden_shortcuts': ['grasp_weld', 'magnetic_snap', 'object_pose_playback', 'direct_object_force_or_torque'],
                'recovery': 'Correct wrong placement, grasp or orientation through robot actions within the frozen budget; table drops remain recoverable.',
                'floor_contact': 'A task body contacting the floor is terminal task failure.',
                'evidence_missing': 'Missing or inconsistent native evidence makes the referee invalid.',
                'budget_or_cancel': 'Incomplete episode; do not invent a physical task failure.',
            },
            **deepcopy(design),
        },
        'qualification': {
            'release_status': 'prepared_unvalidated', 'native_preflight': False,
            'manipulation': False, 'physics_accepted': False,
            'task_policy_success_claimed': False,
            'scope': 'Frozen runnable engineering candidate with CPU rule checks. Native startup, image review and robot manipulation require separate evidence for this exact release.',
        },
    }
