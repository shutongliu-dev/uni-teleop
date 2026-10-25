"""Shared task metadata with separate implementation and qualification states."""
from copy import deepcopy


COMMON_LAYOUT = {
    'frame': 'world', 'workspace': 'parent_scene_table',
    'parent_scene': 'mechanical_assembly/scene.py',
    'same_environment_for_all_tasks': True,
    'robot_layout_id': 'mechanical_single_arm_v1',
    'robot_layout_module': 'scenes.workrooms.mechanical_assembly.robot_layout',
    'initialization': None,
    'candidate_zones': {
        'status': 'design_reference_only_read_frozen_scene_for_positions',
        'assembly_center_xy_m': [.40, 0.],
        'assembly_station_row_axis': 'world_y', 'insertion_axis': 'world_z',
        'supplies': 'Read task-owned supply poses from the selected frozen scene.json; planned tasks have no runnable reset.',
    },
    'ownership': 'Room owns empty surfaces; each task owns all parts, trays, fixtures and tools.',
}

COMMON_ROBOT = {
    'proposed_arm': 'FR3', 'proposed_hand': 'SHARPA W01 right',
    'articulations': 1, 'base_constraint': 'native_fixed_base_v1',
    'layout_id': 'mechanical_single_arm_v1', 'articulation_selection': None,
    'constraint': 'One shared base and ready pose for every task; do not move the base toward the table to fit a task.',
}

PHYSICAL_CONTRACT = {
    'actions': 'Robot joint targets only after reset.',
    'movable_objects': 'Independent gravity-enabled rigid bodies with task-relevant collision geometry.',
    'fixtures': 'Fixed to a task-owned workplate before reset; the single hand need not hold a second part.',
    'forbidden_shortcuts': ['grasp_weld', 'magnetic_snap', 'object_pose_playback',
                           'direct_object_force_or_torque', 'screw_joint', 'gear_constraint'],
    'materials': 'Versioned simulation priors; do not increase friction to rescue a grasp.',
    'recovery': 'Slow regrasp, withdraw, realign and retry are allowed; reset is not an in-episode action.',
    'qualification': 'CPU geometry and IK do not qualify native grasp, insertion, thread or gear contact.',
    'engineering_diagnostics': 'Separately authorized bounded tool-wrench tests may audit asset contact. They are recorded as engineering diagnostics and never count as robot task success.',
}

PRIVILEGED_VALIDATION = {
    'mode': 'privileged_native',
    'validation_scope': 'privileged_native_only',
    'required': True,
    'privileged_entrypoint': None,
    'ordinary_agent_validation': False,
    'model_calls': 0,
    'object_state_writeback': False,
    'scope': 'Native privileged validation only; CPU contracts and public Agent episodes do not grant task qualification.',
}


def planned_task(*, task_id, title, difficulty, intent, objects, design):
    """Make independent metadata with explicit, non-executable candidate criteria."""
    return {
        'schema': 'workroom_task_definition.v1',
        'family': 'mechanical_assembly', 'task_id': task_id, 'title': title,
        'status': 'planned', 'difficulty': difficulty,
        'objects': deepcopy(objects), 'layout': deepcopy(COMMON_LAYOUT),
        'robot_requirements': deepcopy(COMMON_ROBOT),
        'rules': {'intent': intent, 'contract': None},
        'evaluator': {'implementation': None, 'thresholds': None},
        'source_release': None,
        'design': {
            'schema': 'mechanical_task_design.v1',
            'status': 'candidate_not_physics_validated',
            'dimensions_and_thresholds': 'Candidates only; freeze after native evidence, never treat as accepted tolerances.',
            'physical_contract': deepcopy(PHYSICAL_CONTRACT),
            **deepcopy(design),
        },
        'qualification': {
            'native_preflight': False, 'manipulation': False,
            'scope': 'Task design only. Asset existence and a static preview do not establish a runnable task.',
        },
        'validation': deepcopy(PRIVILEGED_VALIDATION),
        'cannot_run_reason': 'Frozen task reset, robot configuration, native contact qualification and evaluator are not released.',
    }


def bind_prepared_release(task, *, path, manifest_sha256, evidence, scope):
    """Expose a frozen L1–L3 scene candidate without granting task acceptance."""
    from .rules import public_rules, thresholds
    result = deepcopy(task)
    task_id = result['task_id']
    rules = public_rules(task_id)
    result.update(
        status='implemented',
        entrypoint=f'scenes.workrooms.mechanical_assembly.tasks.{task_id}.task',
        runtime_entrypoint='bench.runtime.service:EpisodeSession',
        release_schema='mechanical_assembly_task_release.v1',
        release_schema_version='isaacsim-mechanical-assembly-release-v1',
        source_release={'path': path, 'manifest_sha256': manifest_sha256},
        rules={'intent': rules['goal'],
               'implementation': 'scenes.workrooms.mechanical_assembly.rules', 'contract': rules},
        evaluator={
            'implementation': 'scenes.workrooms.mechanical_assembly.evaluation.SequenceEvaluator',
            'frozen_path': 'evaluator/sequence_evaluator.py', 'thresholds': thresholds(task_id)},
        qualification={
            'release_status': 'prepared_unvalidated', 'native_preflight': False,
            'manipulation': False, 'physics_accepted': False, 'task_policy_success_claimed': False,
            'scene_contact_evidence': deepcopy(evidence), 'scope': scope},
    )
    result.pop('cannot_run_reason', None)
    result['objects']['asset_bindings'] = {
        'frozen_scene': path+'/scene.json', 'frozen_inventory': path+'/asset_file_hashes.json'}
    result['layout']['initialization'] = 'Frozen scene.json plus complete initial_state.json; reset only.'
    result['design']['status'] = 'frozen_candidate_with_separate_scene_evidence'
    result['design']['dimensions_and_thresholds'] = 'Frozen scoring candidates; bounded scene contact evidence does not grant robot task acceptance.'
    return result
