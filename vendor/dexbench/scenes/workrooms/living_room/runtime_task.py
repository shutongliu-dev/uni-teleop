"""Canonical single-arm living-room task composition; no manipulation policy."""
from copy import deepcopy
import importlib
import math
from pathlib import Path

from bench.isaacsim.release import checked, digest, task_inputs
from bench.paths import load_json
from bench.protocol.task_referee import contract_sha256
from scenes.workrooms.registry import BASE_RELEASE, BASE_MANIFEST_SHA256, PHYSICS_OVERRIDES
from scenes.workrooms.observations import camera_specs
from scenes.workrooms.mechanical_assembly.task_scene import DEFAULT_STAND_ROOT, STAND_MANIFEST_SHA256
from .scene import environment_reference
from .scene_candidates import scene_candidate
from .dual_robot_layout import robot_configurations, ready_configuration, placement_transform

TASK_IDS = ('remote_operation', 'water_pouring')
TASK_SCHEMA = 'living-room-manipulation-task-v1'
PHYSICS_DT = .0005


def evaluator_sources(task_id):
    root = Path(__file__).parent/'tasks'/task_id
    sources = {'sequence_evaluator.py': root/'evaluator.py', 'rules.py': root/'rules.py'}
    for name in ('native_observer.py', 'observer_geometry.py'):
        if (root/name).is_file(): sources[name] = root/name
    return sources


def bind_episode_task(task, seed, rules):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Episode seed must be an unsigned 32-bit integer')
    task = deepcopy(task)
    ident = task['id']
    task['episode_seed'] = seed
    task['evaluation'] = rules.evaluation_configuration(seed=seed,
        physics_dt_s=task.get('evaluation', {}).get('physics_dt_s', PHYSICS_DT),
        asset_metadata=task['asset_metadata'])
    if ident == 'remote_operation':
        task['public_rules'] = rules.public_rules(seed=seed)
        if task['evaluation']['schema'] == 'living_remote_evaluation.v3':
            television = task['public_rules']['television']
            task['public_goal'] = {'operation': 'cycle_television_power', 'frame_id': 'world_m_z_up',
                'object_id': 'remote', 'controlled_object_id': 'television',
                'power_button': television['toggle_button'],
                'initial_power_on': television['initial_power_on'],
                'required_on_s': television['required_on_s'],
                'description': task['public_rules']['instruction_zh']}
            if task['public_rules']['success'].get('holder_return_required'):
                task['public_goal'].update(target_id='holder', holder_return_required=True,
                    stable_duration_s=task['public_rules']['success']['stable_duration_s'])
        else:
            # Historical releases bind their own frozen rules, including the
            # seed-dependent two-key goal. Never reinterpret them as TV tasks.
            task['public_goal'] = {'operation': 'operate_remote', 'frame_id': 'world_m_z_up',
                'object_id': 'remote', 'button_sequence': task['evaluation']['button_sequence'],
                'description': task['public_rules']['instruction_zh']}
    else:
        task['public_rules'] = rules.public_rules()
        task['public_goal'] = {'operation': 'pour_water', 'frame_id': 'world_m_z_up',
            'object_id': 'kettle', 'initial_water_ml': 250., 'target_water_ml': 100.,
            'target_tolerance_ml': 10., 'target_interval_ml': [90., 110.], 'max_spill_ml': 5.,
            'description': task['public_rules']['instruction_zh']}
    if 'referee' in task['public_rules']:
        task['public_rules_file'] = 'rules.json'
        task['public_rules_sha256'] = contract_sha256(task['public_rules'])
        task['public_referee_contract_sha256'] = contract_sha256(task['public_rules']['referee'])
    return task


def build_task_inputs(task_id, *, seed=0):
    if task_id not in TASK_IDS: raise ValueError('Unknown living-room runtime task')
    checked(BASE_RELEASE/'release_manifest.json', BASE_MANIFEST_SHA256)
    _, base, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    scene = deepcopy(base)
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal', 'soft_bodies', 'scenario'):
        scene.pop(key, None)
    reference = environment_reference()
    candidate = scene_candidate(task_id)
    checked(candidate['manifest_path'], candidate['manifest_sha256'])
    checked(candidate['initial_usd'], candidate['initial_usd_sha256'])
    asset_root = Path(candidate['initial_usd']).parent
    manifest = load_json(Path(candidate['manifest_path']))
    checked(asset_root/'metadata.json', manifest['metadata_sha256'])
    metadata = load_json(asset_root/'metadata.json')
    robot = robot_configurations()['front']
    initial, ready = ready_configuration(robot)
    scene.update(robot=robot, objects=[], fixtures=[], goal_markers=[],
        cloth={'enabled': False}, workbench={'enabled': False}, workcell_details={'enabled': False},
        workroom=reference, description='Single-arm living-room '+task_id+' with frozen task objects and public robot joint actions.')
    scene['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    scene['isaacsim']['physics_dt_s'] = PHYSICS_DT
    scene['isaacsim']['workroom_lighting'] = {'schema': 'workroom-lighting-v1', 'family': 'living_room', 'profile': 'nominal'}
    scene['physics'] = {'dt_s': PHYSICS_DT, 'control_dt_s': .02,
                        'gravity_m_s2': [0, 0, -9.81], 'backend': 'native_isaacsim_physx_gpu'}
    scene['table'] = {'position_m': [.65, 0, .76], 'half_extents_m': [.5, .5, .02], 'friction': .4}
    checked(DEFAULT_STAND_ROOT/'manifest.json', STAND_MANIFEST_SHA256)
    stand_transform = placement_transform('front')
    scene['fixtures'] = [{'id': 'robot_stand', 'asset_id': 'mechanical_robot_stand',
        'asset_root': str(DEFAULT_STAND_ROOT), 'position_m': stand_transform[:3, 3].tolist(),
        'quaternion_wxyz': [math.sqrt(.5), 0., 0., math.sqrt(.5)]}]
    cameras = camera_specs(load_json(Path(reference['dependency_manifest_path'])))
    cameras['task'].update(eye=[1.35, -.52, 1.42], target=[.65, 0., .86], focal_length_mm=34.)
    scene['cameras'] = cameras
    if task_id == 'remote_operation':
        bodies = {'remote': metadata['remote']['body_path'], 'holder': metadata['holder']['body_path'],
                  **{name: info['body_path'] for name, info in metadata['buttons'].items()}}
        joints = {name: info['joint_path'] for name, info in metadata['buttons'].items()}
        particles = {}
    else:
        bodies = {name: info['path'] for name, info in metadata['containers'].items()}
        joints = {}
        w = metadata['water']
        particles = {'water': {'prim_path': w['particles_path'], 'system_path': w['system_path'],
            'particle_count': w['particle_count'], 'total_mass_kg': w['total_mass_kg'], 'position_attribute': 'points'}}
    task = {'schema_version': TASK_SCHEMA, 'id': task_id, 'task_id': task_id,
        'title_zh': 'Task1 · 遥控电视开关机' if task_id == 'remote_operation' else 'Task2 · 水壶倒水',
        'difficulty': 'single_arm_dexterous', 'cloth': False,
        'steps': [{'id': task_id, 'instruction_zh':
            '用遥控器打开电视，连续保持开机10秒，再关机并将遥控器放回收纳架、松手稳定。'
            if task_id == 'remote_operation' else '完成公开目标，松手并稳定归位。'}],
        'initial_state': deepcopy(initial), 'asset_metadata': metadata,
        'recommended_budgets': {'sim_time_s': 120., 'wall_time_s': 1800., 'max_actions': 6000},
        'training_policy_included': False, 'physics_accepted': False, 'task_policy_success_claimed': False,
        'known_limitations': list(candidate.get('known_limitations', [])),
        'qualification_scope': 'Published executable candidate; complete robot manipulation success is separate evidence.'}
    rules = importlib.import_module(f'{__package__}.tasks.{task_id}.rules')
    task = bind_episode_task(task, seed, rules)
    files = evaluator_sources(task_id)
    entry = 'native_observer.py' if 'native_observer.py' in files else 'sequence_evaluator.py'
    scene['native_task_bundle'] = {'schema': 'native-usd-task-bundle-v1',
        'usd_path': candidate['initial_usd'], 'usd_sha256': candidate['initial_usd_sha256'],
        'root_prim_path': '/World/LivingTask', 'rigid_bodies': bodies, 'joints': joints, 'particle_sets': particles,
        'observer': {'path': 'evaluator/'+entry, 'sha256': digest(files[entry]),
            'files': {name: digest(path) for name, path in files.items()},
            'class_name': 'NativeTaskObserver', 'configuration': deepcopy(task)}}
    if task_id == 'remote_operation' and task['evaluation']['schema'] == 'living_remote_evaluation.v3':
        scene['native_task_bundle']['visual_state_bindings'] = {
            'schema': 'boolean-material-bindings-v1', 'bindings': [{
                'id': 'television_power',
                'target_prim_path': '/World/Room/Geometry/television_screen/Visual',
                'state_path': ['television', 'power_on'],
                'false_material': {'diffuse_color_rgb': [.008, .012, .015],
                    'emissive_color_rgb': [0., 0., 0.], 'roughness': .11, 'metallic': .12},
                'true_material': {'diffuse_color_rgb': [.12, .35, .65],
                    'emissive_color_rgb': [.15, .65, 1.5], 'roughness': .3, 'metallic': 0.}}]}
    scene['living_room_provenance'] = {'asset_candidate': candidate, 'ready_design': ready,
        'stand_manifest_sha256': STAND_MANIFEST_SHA256, 'parent_environment_unchanged': True,
        'shared_controller_unchanged': True, 'task_object_pose_writes_after_reset': False}
    return {'scene': scene, 'task': task, 'initial_state': initial}
