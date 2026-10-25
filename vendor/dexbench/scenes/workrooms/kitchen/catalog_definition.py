"""Catalog metadata for frozen kitchen candidates; no asset or simulator loads."""
from copy import deepcopy
import re

from .rules import public_rules, thresholds
from bench.paths import DATA_ROOT

RELEASE_ROOT = f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/kitchen'
EVIDENCE_ROOT = f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_tasks_20261019'
ASSET_ROOT = f'{DATA_ROOT}/assets/task_parts/kitchen'
TASK_IDS = ('breakfast_serving', 'produce_transfer', 'utensil_sorting', 'pot_lid_placement',
            'pantry_stocking', 'table_clearing')
TITLES = ('早餐配盘与整盘端送', '果蔬分类装盘', '餐具分类收纳', '冷灶锅具就位与锅盖盖合',
          '调味品与干货分类归位', '餐后餐桌分类清理')


def implemented_task_definition(task_id, *, intended_roles, inventory_counts,
                                dynamic_count, fixed_fixture_count,
                                manifest_sha256, asset_version='v1', release_version='v1',
                                contextual_prop_count=0, evidence_root=EVIDENCE_ROOT,
                                canonical_run='canonical', goal_support_run='goal_support',
                                canonical_sha256=None, goal_support_sha256=None):
    number = TASK_IDS.index(task_id)+1
    counts = (dynamic_count, fixed_fixture_count, contextual_prop_count, *inventory_counts.values())
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError('Kitchen inventory counts must be nonnegative integers')
    if sum(inventory_counts.values()) != dynamic_count+fixed_fixture_count:
        raise ValueError('Kitchen task inventory must match dynamic and scoring fixture counts')
    if manifest_sha256 is not None and not re.fullmatch(r'[0-9a-f]{64}', manifest_sha256):
        raise ValueError('Kitchen published release requires an actual SHA256 manifest digest')
    published = manifest_sha256 is not None
    if not published and (canonical_sha256 or goal_support_sha256):
        raise ValueError('An unpublished task cannot claim release-bound evidence')
    rules = public_rules(task_id)
    evidence = [{'path': f'{evidence_root}/{task_id}/{kind}/qualification.json',
                 'sha256': sha} for kind, sha in
                ((canonical_run, canonical_sha256), (goal_support_run, goal_support_sha256))
                if sha is not None]
    native_preflight = canonical_sha256 is not None
    scene_support = native_preflight and goal_support_sha256 is not None
    result = {
        'schema': 'workroom_task_definition.v1', 'family': 'kitchen',
        'task_id': task_id, 'task_number': number, 'difficulty': number,
        'difficulty_calibrated': False, 'title': f'Task{number} · {TITLES[number-1]}',
        'status': 'implemented' if published else 'planned',
        'preparation_state': 'frozen_executable_candidate' if published else 'awaiting_frozen_release',
        'entrypoint': f'scenes.workrooms.kitchen.tasks.{task_id}.task',
        'release_schema': 'kitchen_manipulation_task_release.v1',
        'release_schema_version': 'isaacsim-kitchen-manipulation-release-v1',
        'objects': {'intended_roles': list(intended_roles),
            'inventory_counts': deepcopy(inventory_counts),
            'dynamic_count': dynamic_count,
            'fixed_fixture_count': fixed_fixture_count,
            'contextual_prop_count': contextual_prop_count,
            'contextual_props_scored': False,
            'asset_root': f'{ASSET_ROOT}/{task_id}/{asset_version}',
            'asset_bindings': 'Exact dynamic USD, fixtures, physical metadata and dependencies are pinned in the frozen release.'},
        'layout': {'frame': 'world', 'workspace': 'cooking' if number == 4 else 'bar',
            'parent_scene': 'kitchen/scene.py', 'same_environment_for_all_tasks': True,
            'initialization': 'Frozen task-owned supply poses and original 29-joint ready state; seed is preserved without changing geometry.'},
        'robot_requirements': {'arm': 'fr3', 'hand_id': 'sharpa_w01_right',
            'assembly_version': 'isaacsim_6_1_v2', 'articulations': 1,
            'layout_id': 'kitchen_task_single_arm_v1', 'base_position_m': [.65, 0., .78],
            'base_yaw_deg': 90. if number == 4 else -90., 'articulation_selection': 'single'},
        'rules': {'intent': rules['instruction_zh'],
            'implementation': 'scenes.workrooms.kitchen.rules', 'contract': rules},
        'evaluator': {'implementation': 'scenes.workrooms.kitchen.evaluation' if published else None,
            'reference_implementation': 'scenes.workrooms.kitchen.evaluation.SequenceEvaluator',
            'frozen_path': 'evaluator/sequence_evaluator.py', 'thresholds': thresholds(task_id),
            'scope': 'Complete native poses/contact impulses, task order, actual support, release and 400 ms stability.'},
        'source_release': {'path': f'{RELEASE_ROOT}/{task_id}/{release_version}',
            'manifest_sha256': manifest_sha256} if published else None,
        'qualification': {'status': 'scene_support_checks_passed' if scene_support else 'scene_checks_pending',
            'native_preflight': native_preflight, 'scene_support': scene_support,
            'manipulation': False, 'physics_accepted': False,
            'scene_contact_evidence': evidence,
            'scope': 'Canonical reset and separate goal-support diagnostics qualify only their bounded scene checks. They do not establish robot pickup, transport, full task success, swept-path safety or calibrated contact dynamics. Prepared releases retain physics_accepted=false.'},
    }
    if not published:
        result['cannot_run_reason'] = 'Task authoring awaits an immutable release with a verified manifest digest.'
    return result
