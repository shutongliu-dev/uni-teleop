"""Compose laboratory task-owned USD with its pinned room and shared native bench."""
from copy import deepcopy
import json
from pathlib import Path

from bench.isaacsim.release import checked, digest, task_inputs
from bench.paths import load_json
from scenes.workrooms.registry import BASE_RELEASE, BASE_MANIFEST_SHA256, PHYSICS_OVERRIDES
from scenes.workrooms.observations import camera_specs
from scenes.workrooms.mechanical_assembly.task_scene import DEFAULT_STAND_ROOT, STAND_MANIFEST_SHA256
from .scene import environment_reference
from .task_robot_layout import TASK_IDS, robot_configuration, ready_configuration, placement_transform, base_quaternion

TASK_SCHEMA = 'laboratory-manipulation-task-v1'
ASSET_SCHEMA = 'laboratory-task-asset.v1'
PHYSICS_DT = .0005
ROOT_PRIM = '/World/LaboratoryTask'
from .rules import TITLES as TASK_TITLES
TITLES = dict(zip(TASK_IDS, TASK_TITLES))


def evaluator_sources(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laboratory task')
    root = Path(__file__).parent
    files = {'sequence_evaluator.py': root/'evaluation.py', 'rules.py': root/'rules.py',
             'native_evidence.py': root/'native_evidence.py'}
    if task_id == 'pipette_transfer':
        files['liquid_dynamics.py'] = root/'tasks/pipette_transfer/liquid.py'
    if task_id == 'solid_weighing':
        files['weight_readout.py'] = root/'weight_readout.py'
    return files



def load_asset_manifest(task_id, asset_manifest):
    """Accept a path or pinned descriptor; resolve every asset byte before use."""
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laboratory task')
    if isinstance(asset_manifest, dict):
        path = Path(asset_manifest['manifest_path']).resolve()
        checked(path, asset_manifest['manifest_sha256'])
    else:
        path = Path(asset_manifest).resolve()
    manifest = load_json(path)
    if (manifest.get('schema') != ASSET_SCHEMA or manifest.get('task_id') != task_id
            or manifest.get('root_prim_path') != ROOT_PRIM or manifest.get('presentation') is not False):
        raise ValueError('Expected the task-owned dynamic laboratory reset asset')
    asset_root = path.parent
    usd = asset_root/'task.usda'
    metadata_path = asset_root/'metadata.json'
    checked(usd, manifest['usd_sha256'])
    checked(metadata_path, manifest['metadata_sha256'])
    auxiliary = {str(asset_root/name): manifest[key] for name, key in (
        ('structural_audit.json', 'structural_audit_sha256'), ('ATTRIBUTION.txt', 'attribution_sha256'))}
    for filename, sha in auxiliary.items():
        checked(filename, sha)
    for dependency, row in manifest['dependencies'].items():
        checked(dependency, row['sha256'])
    metadata = load_json(metadata_path)
    if (metadata.get('task_id') != task_id or metadata.get('root_path', ROOT_PRIM) != ROOT_PRIM
            or metadata.get('presentation', False) is not False):
        raise ValueError('Task metadata identity differs from asset manifest')
    actors = metadata['actors']
    if not actors or len(set(actors.values())) != len(actors):
        raise ValueError('Task metadata requires unique dynamic actors')
    if any(not p.startswith(ROOT_PRIM+'/') for p in actors.values()):
        raise ValueError('Task actor escapes the declared subtree')
    descriptor = {'manifest_path': str(path), 'manifest_sha256': digest(path),
        'usd_path': str(usd), 'usd_sha256': manifest['usd_sha256'],
        'metadata_path': str(metadata_path), 'metadata_sha256': manifest['metadata_sha256'],
        'root_prim_path': ROOT_PRIM, 'dependencies': deepcopy(manifest['dependencies']),
        'auxiliary_files_sha256': auxiliary}
    return descriptor, metadata


def bind_episode_task(task, seed, rules):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Episode seed must be an unsigned 32-bit integer')
    task = deepcopy(task)
    task_id = task['id']
    task['episode_seed'] = seed
    task['public_rules'] = rules.public_rules(task_id, seed=seed)
    task['evaluation'] = rules.evaluation_configuration(task_id, task['asset_metadata'], seed=seed,
        physics_dt_s=task.get('evaluation', {}).get('physics_dt_s', PHYSICS_DT),
        room_support_paths=task.get('room_support_paths', {}))
    task['public_goal'] = {'operation': task_id, 'frame_id': 'world_m_z_up',
        'description': task['public_rules']['instruction_zh']}
    return rules.bind_task_rules(task)


def build_task_inputs(task_id, *, asset_manifest, seed=0):
    """Build a reviewable candidate; asset identity is supplied, never invented."""
    from . import rules
    candidate, metadata = load_asset_manifest(task_id, asset_manifest)
    checked(BASE_RELEASE/'release_manifest.json', BASE_MANIFEST_SHA256)
    _, source, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    scene = deepcopy(source)
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal', 'soft_bodies', 'scenario'):
        scene.pop(key, None)
    reference = environment_reference()
    checked(reference['usd_path'], reference['usd_sha256'])
    checked(reference['dependency_manifest_path'], reference['dependency_manifest_sha256'])
    robot = robot_configuration(task_id)
    initial, ready = ready_configuration(task_id, robot)
    scene.update(robot=robot, objects=[], fixtures=[], goal_markers=[],
        cloth={'enabled': False}, workbench={'enabled': False}, workcell_details={'enabled': False},
        workroom=reference, description='Laboratory '+task_id+' with independent dynamic task objects and public robot joint actions.')
    scene['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    scene['isaacsim']['physics_dt_s'] = PHYSICS_DT
    scene['isaacsim']['workroom_lighting'] = {'schema': 'workroom-lighting-v1', 'family': 'laboratory', 'profile': 'nominal'}
    scene['physics'] = {'dt_s': PHYSICS_DT, 'control_dt_s': .02,
        'gravity_m_s2': [0, 0, -9.81], 'backend': 'native_isaacsim_physx_gpu'}
    scene['table'] = {'position_m': [.65, 0., .875], 'half_extents_m': [1.2, .625, .025], 'friction': .4}
    checked(DEFAULT_STAND_ROOT/'manifest.json', STAND_MANIFEST_SHA256)
    transform = placement_transform(task_id)
    scene['fixtures'] = [{'id': 'robot_stand', 'asset_id': 'mechanical_robot_stand',
        'asset_root': str(DEFAULT_STAND_ROOT), 'position_m': transform[:3, 3].tolist(),
        'quaternion_wxyz': list(base_quaternion(task_id))}]
    cameras = camera_specs(load_json(Path(reference['dependency_manifest_path'])))
    cameras['overview'].update(eye=[2.85,.82,2.10], target=[.65,1.25,1.0], focal_length_mm=24.)
    cameras['task'].update(eye=[1.90,.65,2.65], target=[.70,1.25,.90], focal_length_mm=23.)
    cameras['room'].update(eye=[3.12,-2.20,2.38], target=[.30,.65,1.15], focal_length_mm=19.)
    scene['cameras'] = cameras
    task = {'schema_version': TASK_SCHEMA, 'id': task_id, 'task_id': task_id,
        'title_zh': TITLES[task_id], 'difficulty': 'single_arm_dexterous', 'cloth': False,
        'steps': [{'id': task_id, 'instruction_zh': '完成公开目标，松手后稳定放置；允许重新抓取和调整。'}],
        'initial_state': deepcopy(initial), 'asset_metadata': metadata,
        'room_support_paths': load_json(Path(reference['dependency_manifest_path']))['support_colliders'],
        'recommended_budgets': {'sim_time_s': 120., 'wall_time_s': 1800., 'max_actions': 6000},
        'training_policy_included': False, 'physics_accepted': False, 'task_policy_success_claimed': False,
        'known_limitations': ['Task-owned apparatus; fixed-room instruments remain decorative.',
            'Material and mass values are declared engineering priors, not measured physical calibration.',
            'Candidate task reset/contact evidence does not establish successful robot manipulation.'],
        'qualification_scope': 'Executable scene candidate; native scene checks and robot task success require separate bound evidence.'}
    task['known_limitations'].extend(metadata.get('known_limitations', metadata.get('limitations', [])))
    task = bind_episode_task(task, seed, rules)
    files = evaluator_sources(task_id)
    scene['native_task_bundle'] = {'schema': 'native-usd-task-bundle-v1',
        'usd_path': candidate['usd_path'], 'usd_sha256': candidate['usd_sha256'],
        'root_prim_path': ROOT_PRIM, 'rigid_bodies': deepcopy(metadata['actors']), 'joints': deepcopy(metadata.get('joints', {})),
        'particle_sets': deepcopy(metadata.get('particle_sets', {})),
        'observer': {'path': 'evaluator/sequence_evaluator.py', 'sha256': digest(files['sequence_evaluator.py']),
            'files': {name: digest(path) for name, path in files.items()},
            'class_name': 'NativeTaskObserver', 'configuration': deepcopy(task)}}
    if task_id == 'pipette_transfer':
        scene['native_task_bundle']['reduced_dynamics'] = {
            'path': 'evaluator/liquid_dynamics.py', 'sha256': digest(files['liquid_dynamics.py']),
            'files': {name: digest(path) for name, path in files.items()},
            'class_name': 'ReducedLiquidDynamics', 'configuration': deepcopy(task)}
        scene['native_task_bundle']['volume_visual_bindings'] = deepcopy(metadata['volume_visual_bindings'])
    if 'visual_state_bindings' in metadata:
        scene['native_task_bundle']['visual_state_bindings'] = deepcopy(metadata['visual_state_bindings'])
    scene['laboratory_task_provenance'] = {'asset_candidate': candidate, 'ready_design': ready,
        'stand_manifest_sha256': STAND_MANIFEST_SHA256, 'parent_environment_unchanged': True,
        'shared_controller_unchanged': True, 'task_object_pose_writes_after_reset': False,
        'physical_reset_seeded': False, 'seed_scope': 'Public rules only; physical reset is the frozen USD.'}
    return {'scene': scene, 'task': task, 'initial_state': initial}
