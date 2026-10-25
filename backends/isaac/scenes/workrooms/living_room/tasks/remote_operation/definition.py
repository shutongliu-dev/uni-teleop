"""Executable frozen scene candidate; robot-policy success remains unqualified."""
from bench.paths import DATA_ROOT
from scenes.workrooms.living_room.task_design import robot_requirements
from scenes.workrooms.living_room.scene_candidates import scene_candidate
from .rules import public_rules

TASK = {
    'schema': 'workroom_task_definition.v1', 'family': 'living_room',
    'task_id': 'remote_operation', 'task_number': 1, 'difficulty': 1,
    'title': 'Task1 · 遥控电视开关机', 'status': 'implemented', 'user_confirmed_task': True,
    'entrypoint': 'scenes.workrooms.living_room.tasks.remote_operation.task',
    'preparation_state': 'frozen_executable_candidate',
    'release_schema': 'living_room_manipulation_task_release.v1',
    'release_schema_version': 'isaacsim-living-room-manipulation-release-v1',
    'scene_candidate': scene_candidate('remote_operation'),
    'objects': {'intended_roles': ['free_remote', 'two_passive_buttons', 'remote_holder', 'television'],
        'asset_root': f'{DATA_ROOT}/assets/task_parts/living_room/remote_operation/v5',
        'asset_bindings': 'Exact source USD, dependencies and metadata pinned in the frozen scene and release manifest.'},
    'layout': {'frame': 'world', 'workspace': 'parent_scene_table',
        'parent_scene': 'living_room/scene.py', 'same_environment_for_all_tasks': True,
        'initialization': 'Frozen object USD and original front-arm joint reset; episode seed binds the published goal.'},
    'robot_requirements': robot_requirements(1),
    'rules': {'intent': '用遥控器打开电视，连续保持开机10秒，再关机并将遥控器放回收纳架、松手稳定。',
        'implementation': 'scenes.workrooms.living_room.tasks.remote_operation.rules',
        'contract': public_rules()},
    'evaluator': {'implementation': 'scenes.workrooms.living_room.tasks.remote_operation.evaluator',
        'reference_implementation': 'scenes.workrooms.living_room.tasks.remote_operation.evaluator.SequenceEvaluator',
        'frozen_path': 'evaluator/sequence_evaluator.py', 'scope': 'Native handheld power-button press/release events toggle the television; require one continuous 10 s on interval followed by a remote power-off event and released, supported holder stability for 300 ms.'},
    'source_release': {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/v7',
        'manifest_sha256': 'e47e15ce25746f76fa83c3b5111715158d905767a7fb935f36408ffb38e6dcc1'},
    'qualification': {'status': 'candidate_native_smoke_passed', 'native_preflight': True,
        'manipulation': False, 'physics_accepted': False, 'public_http_smoke': False, 'public_functions_smoke': True,
        'evidence': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_task1_tv_20261019/native03/qualification.json',
        'evidence_sha256': '5c2bda7bc8d146f08b74d8ea361915b03d0f79893d1f5bf31c43f0a203ce130c',
        'scope': 'Frozen v7/source03 canonical reset, 300 ms public-function hold, 600 native substeps and four RGB-D cameras passed with zero task credit. Separate off/on/off RTX display fixtures passed with unchanged physical state. Full robot power-on, 10-second hold, power-off and holder return remain unqualified.'},
}
