"""Compose frozen study props with the original robot and selected fixed room."""
from copy import deepcopy
from pathlib import Path

from bench.isaacsim.release import checked, task_inputs
from bench.paths import load_json
from bench.protocol.task_referee import contract_sha256
from scenes.workrooms.registry import BASE_RELEASE, BASE_MANIFEST_SHA256, PHYSICS_OVERRIDES
from scenes.workrooms.observations import camera_specs
from scenes.workrooms.mechanical_assembly.robot_layout import robot_configuration, ready_configuration
from scenes.workrooms.mechanical_assembly.task_scene import DEFAULT_STAND_ROOT, STAND_MANIFEST_SHA256
from .scene import environment_reference


TASK_IDS = ('stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving')
SUPPORT_PATHS = {
    'table': '/World/Room/Geometry/worktop/Collider',
    'shelf_lower': '/World/Room/Geometry/rear_shelf_level_1/Collider',
    'shelf_upper': '/World/Room/Geometry/rear_shelf_level_2/Collider',
    'floor': '/World/Room/Geometry/room_floor/Collider',
}
TITLES = {'stationery_sorting': '文具分类', 'book_stacking': '叠放书本',
          'book_retrieval': '取出指定书本', 'book_shelving': '按类别归还书本'}


def candidate_inputs(task_id):
    from .task_scene import candidate_inputs as load_candidate
    return load_candidate(task_id)


def build_task_inputs(task_id, *, seed=0, stand_path=None):
    from .rules import validate_configuration
    if task_id not in TASK_IDS:
        raise ValueError('Unknown study task')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    if stand_path is not None:
        raise ValueError('Study tasks require the reviewed original robot stand; overrides are unsupported')
    candidate, manifest, metadata, dependencies = candidate_inputs(task_id)
    configuration = validate_configuration(metadata['configuration'])
    bodies = deepcopy(manifest['bodies'])
    if (configuration['task_id'] != task_id or set(bodies) != set(configuration['objects'])
            or any(path != '/World/StudyTask/Objects/' + name for name, path in bodies.items())
            or not set(configuration['supports']) <= set(SUPPORT_PATHS) - {'floor'}):
        raise ValueError('Study candidate body or support identities differ from its rules')
    checked(candidate['initial_usd'], candidate['initial_usd_sha256'])
    checked(candidate['manifest_path'], candidate['manifest_sha256'])
    for path, expected in dependencies.items():
        checked(path, expected)
    checked(BASE_RELEASE / 'release_manifest.json', BASE_MANIFEST_SHA256)
    _, scene, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal', 'soft_bodies'):
        scene.pop(key, None)
    robot = robot_configuration()
    initial, ready = ready_configuration(robot)
    reference = environment_reference()
    checked(reference['usd_path'], reference['usd_sha256'])
    checked(reference['dependency_manifest_path'], reference['dependency_manifest_sha256'])
    description = load_json(Path(reference['dependency_manifest_path']))
    scene.update(robot=robot, objects=[], fixtures=[], goal_markers=[], cloth={'enabled': False},
        workbench={'enabled': False}, workcell_details={'enabled': False}, workroom=reference,
        cameras=camera_specs(description, robot=robot),
        scenario={'family': 'study', 'task_id': task_id, 'seed': seed},
        description='Study task candidate; original robot and fixed parent environment.')
    scene['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    scene['isaacsim']['physics_dt_s'] = .0005
    scene['task_layers'] = [{'id': 'task', 'usd_path': candidate['initial_usd'],
        'sha256': candidate['initial_usd_sha256'], 'source_prim_path': '/World/StudyTask',
        'target_prim_path': '/World/StudyTask', 'dependencies_sha256': deepcopy(dependencies),
        'rigid_bodies': bodies, 'particle_sets': {}}]
    checked(DEFAULT_STAND_ROOT / 'manifest.json', STAND_MANIFEST_SHA256)
    scene['fixtures'].append({'id': 'robot_stand', 'asset_id': 'mechanical_robot_stand',
        'asset_root': str(DEFAULT_STAND_ROOT), 'position_m': [0., 0., 0.]})
    rules = deepcopy(configuration['rules'])
    task = {'schema_version': 'study-task-v1', 'id': task_id, 'task_id': task_id,
        'title_zh': TITLES[task_id], 'difficulty': 'engineering_candidate', 'cloth': False,
        'task_kind': 'single_agent', 'episode_seed': seed, 'initial_state': initial,
        'public_rules': rules, 'public_rules_sha256': contract_sha256(rules),
        'public_referee_contract_sha256': contract_sha256(rules['referee']),
        'public_rules_file': 'rules.json', 'public_goal': {'operation': task_id},
        'steps': [{'id': task_id, 'instruction_zh': rules['referee']['instruction_zh']}],
        'evaluation': {'configuration': configuration, 'body_paths': deepcopy(bodies),
            'support_paths': deepcopy(SUPPORT_PATHS), 'physics_dt_s': .0005},
        'runtime_adapter': 'study_native_v1', 'physics_accepted': False,
        'task_policy_success_claimed': False, 'recommended_budgets': {
            'sim_time_s': configuration['budget']['max_sim_time_s'],
            'max_actions': configuration['budget']['max_robot_actions'], 'wall_time_s': 1800.}}
    scene['study_task_provenance'] = {'candidate': deepcopy(candidate),
        'original_assets_modified': False, 'original_robot_dynamics_and_gains_preserved': True,
        'ready_design': ready, 'task_layer_scope': 'Task subtree only; static preview robots excluded',
        'robot_stand_manifest': str(DEFAULT_STAND_ROOT / 'manifest.json'),
        'robot_stand_manifest_sha256': STAND_MANIFEST_SHA256,
        'native_task_qualified': False}
    return {'scene': scene, 'task': task, 'initial_state': deepcopy(initial)}
