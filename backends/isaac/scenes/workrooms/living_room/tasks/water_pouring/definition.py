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
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/v6',
        'manifest_sha256': 'c31bd1582381ad850b3363570e701cd2623e720c20ed6b797d72461d412d28b5'},
    'qualification': {'status': 'candidate_native_smoke_passed',
                      'native_preflight': True,
                      'manipulation': False,
                      'physics_accepted': False,
                      'public_http_smoke': True,
                      'public_functions_smoke': True,
                      'agent_bench_interface': True,
                      'evidence': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/root/http_single_arm02/qualification.json',
                      'evidence_sha256': 'a920e751696d592f64e9b06a04b270cf3fd9474157a246bdfb32a9d462281feb',
                      'object_interaction_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/task2/qualification.json',
                                                      'sha256': '8ba8c0229257c59411af544b11c173ac65eb7c06df9aeac9852a478d0a9fb7b3',
                                                      'scope': 'Native object-only retention, aimed and missed pouring; not '
                                                               'robot manipulation.'},
                      'scope': 'V6/source_runtime05 single-arm actual HTTP/Agent selection, four RGB-D cameras, wrist '
                               'motion/return and hold; 1 s/2000 native steps, valid physics and normal process exits. '
                               'Separate object-only probes verify conservation, transfer and spilled-water classification. '
                               'No full robot pour or calibrated force claim.'},
}


from scenes.workrooms.living_room.randomization.catalog import bind_definition

TASK = bind_definition(TASK)
