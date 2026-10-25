"""Laboratory metadata, with implementation and bounded qualification separate."""
from copy import deepcopy
from .rules import TASK_IDS, TITLES, public_rules, thresholds
from bench.paths import DATA_ROOT

RELEASE_ROOT = f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory'
ASSET_ROOT = f'{DATA_ROOT}/assets/task_parts/laboratory'


def task_definition(task_id, *, manifest_sha256=None, asset_version='v1',
                    release_version='v1', evidence=(), native_preflight=False,
                    object_interactions=False):
    number = TASK_IDS.index(task_id)+1
    implemented = manifest_sha256 is not None
    rules = public_rules(task_id)
    result = {
        'schema': 'workroom_task_definition.v1', 'family': 'laboratory',
        'task_id': task_id, 'task_number': number, 'difficulty': number,
        'difficulty_calibrated': False, 'title': f'Task{number} · {TITLES[number-1]}',
        'status': 'implemented' if implemented else 'planned',
        'entrypoint': f'scenes.workrooms.laboratory.tasks.{task_id}.task',
        'release_schema': 'laboratory_manipulation_task_release.v1',
        'release_schema_version': 'isaacsim-laboratory-manipulation-release-v1',
        'objects': {'intended_roles': [
            ['sealed_samples', 'classification_trays'],
            ['sealed_vials', 'open_hole_rack', 'supply_cradle'],
            ['independent_screw_cap', 'sample_bottle', 'anti_rotation_holder', 'cap_tray'],
            ['physical_granules', 'scoop', 'weighing_cup', 'nominal_mass_platform'],
            ['source_vessel', 'receiving_vessels', 'physical_piston_tool', 'conserved_reduced_liquid'],
        ][number-1], 'dynamic_count': (4, 3, 2, 26, 5)[number-1],
            'fixed_fixture_count': (2, 2, 2, 3, 1)[number-1],
            'asset_root': f'{ASSET_ROOT}/{task_id}/{asset_version}',
            'asset_bindings': 'Versioned dynamic USD and explicit object metadata.' if implemented else None},
        'layout': {'frame': 'world', 'workspace': 'island_and_rear_counter',
            'parent_scene': 'laboratory/scene.py', 'same_environment_for_all_tasks': True,
            'initialization': 'Frozen task objects and original 29-joint ready state.' if implemented else None},
        'robot_requirements': {'arm': 'fr3', 'hand_id': 'sharpa_w01_right',
            'assembly_version': 'isaacsim_6_1_v2', 'articulations': 1,
            'layout_id': 'laboratory_two_bench_single_arm_v1', 'base_position_m': [.70, 1.2875, .78],
            'base_yaw_deg': 0., 'articulation_selection': 'single'},
        'rules': {'intent': rules['instruction_zh'], 'implementation': 'scenes.workrooms.laboratory.rules',
            'contract': rules if implemented else None},
        'evaluator': {'implementation': 'scenes.workrooms.laboratory.evaluation' if implemented else None,
            'reference_implementation': 'scenes.workrooms.laboratory.evaluation.SequenceEvaluator',
            'frozen_path': 'evaluator/sequence_evaluator.py', 'thresholds': thresholds(task_id),
            'scope': 'Native object poses, support, operation history and released 300 ms stability.'},
        'source_release': {'path': f'{RELEASE_ROOT}/{task_id}/{release_version}',
            'manifest_sha256': manifest_sha256} if implemented else None,
        'qualification': {'native_preflight': native_preflight,
            'object_interactions': object_interactions, 'manipulation': False, 'physics_accepted': False,
            'scene_contact_evidence': deepcopy(list(evidence)),
            'scope': 'Object-object feasibility only. Engineering tool boundaries are explicit; hand-object interaction is assigned to data collection. No robot task-success claim.'},
    }
    if not implemented:
        result['cannot_run_reason'] = 'Frozen executable release is not yet selected.'
    return result
