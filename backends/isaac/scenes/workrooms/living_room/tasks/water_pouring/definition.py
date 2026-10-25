"""Executable frozen scene candidate; robot-policy success remains unqualified."""
from bench.paths import DATA_ROOT
from scenes.workrooms.living_room.task_design import robot_requirements
from scenes.workrooms.living_room.scene_candidates import scene_candidate
from .rules import public_rules

TASK = {
    'schema': 'workroom_task_definition.v1', 'family': 'living_room',
    'task_id': 'water_pouring', 'task_number': 2, 'difficulty': 2,
    'title': 'Task2 · 单臂水壶倒水', 'status': 'implemented', 'user_confirmed_task': True,
    'entrypoint': 'scenes.workrooms.living_room.tasks.water_pouring.task',
    'preparation_state': 'frozen_executable_candidate',
    'release_schema': 'living_room_manipulation_task_release.v1',
    'release_schema_version': 'isaacsim-living-room-manipulation-release-v1',
    'scene_candidate': scene_candidate('water_pouring'),
    'objects': {'intended_roles': ['spouted_water_kettle', 'free_cup', 'water_particles', 'spill_tray'],
        'asset_root': f'{DATA_ROOT}/assets/task_parts/living_room/water_pouring/v2',
        'asset_bindings': 'Exact source USD, dependencies and metadata pinned in the frozen scene and release manifest.'},
    'layout': {'frame': 'world', 'workspace': 'parent_scene_table',
        'parent_scene': 'living_room/scene.py', 'same_environment_for_all_tasks': True,
        'initialization': 'Frozen object USD and original front-arm joint reset; episode seed binds the published goal.'},
    'robot_requirements': robot_requirements(1),
    'rules': {'intent': '抓起带把手水壶，向杯中倒入目标水量，控制洒漏，再将壶放回并松手。',
        'implementation': 'scenes.workrooms.living_room.tasks.water_pouring.rules',
        'contract': public_rules()},
    'evaluator': {'implementation': 'scenes.workrooms.living_room.tasks.water_pouring.evaluator',
        'reference_implementation': 'scenes.workrooms.living_room.tasks.water_pouring.evaluator.SequenceEvaluator',
        'frozen_path': 'evaluator/sequence_evaluator.py', 'scope': 'Original particle mass ledger, cumulative spill, actual spout transfer, supported vessel return and 300 ms release stability.'},
    'source_release': {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/v5',
        'manifest_sha256': 'aab64d8dacfab33aa873db69e23b5c2d0fe9fdba7c996a9bbf85751d1f1bdd67'},
    'qualification': {'status': 'candidate_native_smoke_passed', 'native_preflight': True,
        'manipulation': False, 'physics_accepted': False, 'public_http_smoke': False, 'public_functions_smoke': True,
        'evidence': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_water_coupling_20261018/publication/water_pouring02/qualification.json',
        'evidence_sha256': '2924c1ed5b789072fc1fede8f2b1d9caa205be6182e33000b326d9b2180df81d',
        'scope': 'Frozen v5 reset, public robot hold for 300 ms, 600 native substeps, four RGB-D cameras and observer continuity passed. Practical approximate liquid load; no full robot pouring, quantitative force calibration or HTTP transport claim.'},
}
