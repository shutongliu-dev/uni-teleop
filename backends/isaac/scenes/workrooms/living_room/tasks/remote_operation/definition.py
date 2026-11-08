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
        'asset_root': f'{DATA_ROOT}/assets/task_parts/living_room/remote_operation/v6',
        'asset_bindings': 'Exact source USD, dependencies and metadata pinned in the frozen scene and release manifest.'},
    'layout': {'frame': 'world', 'workspace': 'parent_scene_table',
        'parent_scene': 'living_room/scene.py', 'same_environment_for_all_tasks': True,
        'initialization': 'Frozen object USD and original front-arm joint reset; episode seed binds the published goal.'},
    'robot_requirements': robot_requirements(1),
    'rules': {'intent': '拿起正面朝上的遥控器开启电视，显示10秒倒计时后自动关机，再归架松手稳定。',
        'implementation': 'scenes.workrooms.living_room.tasks.remote_operation.rules',
        'contract': public_rules()},
    'evaluator': {'implementation': 'scenes.workrooms.living_room.tasks.remote_operation.evaluator',
        'reference_implementation': 'scenes.workrooms.living_room.tasks.remote_operation.evaluator.SequenceEvaluator',
        'frozen_path': 'evaluator/sequence_evaluator.py', 'scope': 'Native handheld power-button press/release starts a visible 10 s countdown; automatic shutoff retains cycle credit, then require released supported holder stability for 300 ms.'},
    'source_release': {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/v8',
        'manifest_sha256': 'f523e34e292aa63ff0cd0539cdd8a1f0469a8a362278bd890bc9b5a82590481b'},
    'qualification': {'status': 'prepared_unvalidated', 'native_preflight': False,
        'manipulation': False, 'physics_accepted': False, 'public_http_smoke': False,
        'public_functions_smoke': False,
        'scope': 'Executable task and three installations; development validation is separate and is not a runtime dependency.'},
}


from scenes.workrooms.living_room.randomization.catalog import bind_definition

TASK = bind_definition(TASK)
