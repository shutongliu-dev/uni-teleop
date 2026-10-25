"""Compose kitchen task-owned USD with its pinned room and shared native bench."""
from copy import deepcopy
from pathlib import Path

from bench.isaacsim.release import checked, digest, task_inputs
from bench.paths import load_json
from scenes.workrooms.registry import BASE_RELEASE, BASE_MANIFEST_SHA256, PHYSICS_OVERRIDES
from scenes.workrooms.observations import camera_specs
from scenes.workrooms.mechanical_assembly.task_scene import DEFAULT_STAND_ROOT, STAND_MANIFEST_SHA256
from .scene import environment_reference
from .task_robot_layout import TASK_IDS, robot_configuration, ready_configuration, placement_transform, base_quaternion

TASK_SCHEMA = 'kitchen-manipulation-task-v1'
ASSET_SCHEMA = 'kitchen-task-asset.v1'
PHYSICS_DT = .0005
ROOT_PRIM = '/World/KitchenTask'
TITLES = {'breakfast_serving': '早餐配盘与整盘端送', 'produce_transfer': '果蔬分类装盘',
          'utensil_sorting': '餐具分类与对齐收纳', 'pot_lid_placement': '冷灶锅具就位与锅盖盖合',
          'pantry_stocking': '调味品与干货分类归位', 'table_clearing': '餐后餐桌分类清理'}


def evaluator_sources(task_id):
    if task_id not in TASK_IDS:
        raise ValueError('Unknown kitchen task')
    root = Path(__file__).parent
    return {'sequence_evaluator.py': root/'evaluation.py', 'rules.py': root/'rules.py',
            'native_evidence.py': root/'native_evidence.py'}


def load_asset_manifest(task_id, asset_manifest):
    """Accept a path or pinned descriptor; resolve every asset byte before use."""
    if task_id not in TASK_IDS:
        raise ValueError('Unknown kitchen task')
    if isinstance(asset_manifest, dict):
        path = Path(asset_manifest['manifest_path']).resolve()
        checked(path, asset_manifest['manifest_sha256'])
    else:
        path = Path(asset_manifest).resolve()
    manifest = load_json(path)
    if (manifest.get('schema') != ASSET_SCHEMA or manifest.get('task_id') != task_id
            or manifest.get('root_prim_path') != ROOT_PRIM or manifest.get('presentation') is not False):
        raise ValueError('Expected the task-owned dynamic kitchen reset asset')
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
        physics_dt_s=task.get('evaluation', {}).get('physics_dt_s', PHYSICS_DT))
    task['public_goal'] = {'operation': task_id, 'frame_id': 'world_m_z_up',
        'description': task['public_rules']['instruction_zh']}
    # New assets explicitly author public, static descriptions. Older releases
    # lack this field and keep their byte-for-byte historical goal contract.
    # Never project actors, reset poses, contact state or goal witness poses.
    public = task['asset_metadata'].get('public_goal', {})
    fields = {'objects': ('object_id', 'description', 'category', 'destination'),
              'destinations': ('destination_id', 'description', 'category'),
              'context_objects': ('object_id', 'description')}
    for group, allowed in fields.items():
        if group in public:
            rows = public[group]
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise ValueError('Kitchen public object descriptions must be a list of records')
            projected = [{key: row[key] for key in allowed if key in row} for row in rows]
            if any(not isinstance(value, str) or not value for row in projected for value in row.values()):
                raise ValueError('Kitchen public object descriptions must contain nonempty text')
            task['public_goal'][group] = projected
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
        workroom=reference, description='Kitchen '+task_id+' with independent dynamic task objects and public robot joint actions.')
    scene['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    scene['isaacsim']['physics_dt_s'] = PHYSICS_DT
    scene['isaacsim']['workroom_lighting'] = {'schema': 'workroom-lighting-v1', 'family': 'kitchen', 'profile': 'nominal'}
    scene['physics'] = {'dt_s': PHYSICS_DT, 'control_dt_s': .02,
        'gravity_m_s2': [0, 0, -9.81], 'backend': 'native_isaacsim_physx_gpu'}
    scene['table'] = {'position_m': [.45, -.95, .8775], 'half_extents_m': [1.4, .4, .0225], 'friction': .4}
    checked(DEFAULT_STAND_ROOT/'manifest.json', STAND_MANIFEST_SHA256)
    transform = placement_transform(task_id)
    scene['fixtures'] = [{'id': 'robot_stand', 'asset_id': 'mechanical_robot_stand',
        'asset_root': str(DEFAULT_STAND_ROOT), 'position_m': transform[:3, 3].tolist(),
        'quaternion_wxyz': list(base_quaternion(task_id))}]
    cameras = camera_specs(load_json(Path(reference['dependency_manifest_path'])))
    if task_id == 'pot_lid_placement':
        cameras['overview'].update(eye=[1.8, -.72, 1.85], target=[.30, .85, 1.01], focal_length_mm=25.)
        cameras['task'].update(eye=[.35, .05, 2.12], target=[.25, .91, .94], focal_length_mm=30.)
    else:
        cameras['overview'].update(eye=[1.8, -1.95, 1.9], target=[.55, -.88, 1.00], focal_length_mm=25.)
        cameras['task'].update(eye=[.50, -1.55, 2.05], target=[.50, -.90, .94], focal_length_mm=27.)
    scene['cameras'] = cameras
    limits = rules.thresholds(task_id)
    task = {'schema_version': TASK_SCHEMA, 'id': task_id, 'task_id': task_id,
        'title_zh': TITLES[task_id], 'difficulty': 'single_arm_dexterous', 'cloth': False,
        'steps': [{'id': task_id, 'instruction_zh': '完成公开目标，松手后稳定放置；允许重新抓取和调整。'}],
        'initial_state': deepcopy(initial), 'asset_metadata': metadata,
        'recommended_budgets': {'sim_time_s': limits['max_time_s'],
            'wall_time_s': 1800., 'max_actions': limits['max_actions']},
        'training_policy_included': False, 'physics_accepted': False, 'task_policy_success_claimed': False,
        'known_limitations': ['Rigid food and rigid utensils; no liquids, heating, cutting or deformable food.',
            'Material and mass values are declared engineering priors, not measured physical calibration.',
            'Candidate task reset/contact evidence does not establish successful robot manipulation.'],
        'qualification_scope': 'Executable scene candidate; native scene checks and robot task success require separate bound evidence.'}
    task = bind_episode_task(task, seed, rules)
    files = evaluator_sources(task_id)
    scene['native_task_bundle'] = {'schema': 'native-usd-task-bundle-v1',
        'usd_path': candidate['usd_path'], 'usd_sha256': candidate['usd_sha256'],
        'root_prim_path': ROOT_PRIM, 'rigid_bodies': deepcopy(metadata['actors']), 'joints': {}, 'particle_sets': {},
        'observer': {'path': 'evaluator/sequence_evaluator.py', 'sha256': digest(files['sequence_evaluator.py']),
            'files': {name: digest(path) for name, path in files.items()},
            'class_name': 'NativeTaskObserver', 'configuration': deepcopy(task)}}
    scene['kitchen_task_provenance'] = {'asset_candidate': candidate, 'ready_design': ready,
        'stand_manifest_sha256': STAND_MANIFEST_SHA256, 'parent_environment_unchanged': True,
        'shared_controller_unchanged': True, 'task_object_pose_writes_after_reset': False,
        'physical_reset_seeded': False, 'seed_scope': 'Public rules only; physical reset is the frozen USD.'}
    return {'scene': scene, 'task': task, 'initial_state': initial}
