"""Immutable study tasks, frozen scorers and ordinary native episode inputs."""
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

from bench.isaacsim.release import checked, digest, dump
from bench.paths import DATA_ROOT, load_json

SCHEMA = 'study_task_release.v1'
VERSION = 'isaacsim-study-release-v1'
TASK_SCHEMA = 'study-task-v1'
TASK_IDS = ('stationery_sorting', 'book_stacking', 'book_retrieval', 'book_shelving')
DATASETS = Path(f'{DATA_ROOT}/datasets')


def asset_inventory(scene):
    # The original mechanical robot and generic fixtures are also used here.
    from scenes.workrooms.mechanical_assembly.release import asset_inventory as shared_inventory
    inventory = shared_inventory(scene)
    for layer in scene['task_layers']:
        for path, expected in {layer['usd_path']: layer['sha256'],
                               **layer['dependencies_sha256']}.items():
            checked(path, expected)
            if path in inventory and inventory[path]['sha256'] != expected:
                raise ValueError('Conflicting study asset hash')
            inventory[path] = {'sha256': expected}
    return dict(sorted(inventory.items()))


def initial_state(scene, task):
    from bench.control.factory import public_robot_geometry, robot_configuration
    from bench.protocol.task_referee import project_task_rules
    ident = task.get('id')
    if (task.get('schema_version') != TASK_SCHEMA or ident not in TASK_IDS
            or task.get('task_id') != ident or task.get('task_kind', 'single_agent') != 'single_agent'
            or scene.get('workroom', {}).get('family') != 'study'
            or scene.get('scenario', {}).get('task_id') != ident
            or scene.get('cloth', {}).get('enabled') is not False
            or scene.get('objects') != []):
        raise ValueError('Study task, parent family or layer ownership mismatch')
    robot = scene['robot']
    if (robot.get('hand_id') != 'sharpa_w01_right'
            or robot.get('assembly_version') != 'isaacsim_6_1_v2' or robot.get('instances')):
        raise ValueError('Study tasks require the original single-arm right SHARPA assembly')
    from scenes.workrooms.mechanical_assembly.robot_layout import robot_configuration as reviewed_robot
    from scenes.workrooms.mechanical_assembly.task_scene import DEFAULT_STAND_ROOT
    expected_robot, actual_robot = reviewed_robot(), deepcopy(robot)
    expected_home, actual_home = expected_robot.pop('arm_home_rad'), actual_robot.pop('arm_home_rad', None)
    # The reviewed IK is recomputed by two Python/BLAS distributions. Its
    # floating-point roundoff is not a change to the frozen reset or controls.
    same_home = (isinstance(actual_home, list) and len(actual_home) == len(expected_home)
        and all(type(a) in (int, float) and math.isfinite(a) and abs(a-b) <= 1.e-12
                for a, b in zip(actual_home, expected_home)))
    if (actual_robot != expected_robot or not same_home or scene.get('fixtures') != [{'id': 'robot_stand',
            'asset_id': 'mechanical_robot_stand', 'asset_root': str(DEFAULT_STAND_ROOT),
            'position_m': [0., 0., 0.]}]):
        raise ValueError('Study tasks require the reviewed original robot configuration and stand')
    initial = deepcopy(task.get('initial_state'))
    model = public_robot_geometry(robot_configuration(scene))
    if not isinstance(initial, dict) or set(initial) != {'joint_positions_rad'}:
        raise ValueError('Study reset contains only complete robot joints')
    joints = initial['joint_positions_rad']
    if (not isinstance(joints, dict) or set(joints) != set(model.names)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in joints.values())):
        raise ValueError('Study reset requires every finite active robot joint')
    model.checked_joints(joints, enforce_limits=True)
    evaluation = task['evaluation']
    from .composition import SUPPORT_PATHS
    configuration = evaluation['configuration']
    if (evaluation.get('support_paths') != SUPPORT_PATHS
            or set(configuration['objects']) != set(evaluation['body_paths'])
            or not set(configuration['supports']) <= set(SUPPORT_PATHS) - {'floor'}):
        raise ValueError('Study body or support identity differs from its fixed native mapping')
    layers = scene.get('task_layers', [])
    if (len(layers) != 1 or layers[0].get('id') != 'task'
            or layers[0].get('source_prim_path') != '/World/StudyTask'
            or layers[0].get('target_prim_path') != '/World/StudyTask'):
        raise ValueError('Study tasks require one task-owned prop layer')
    bodies = {}
    for layer in layers:
        for name, path in layer['rigid_bodies'].items():
            if name in bodies or path in bodies.values():
                raise ValueError('Duplicate study native body')
            bodies[name] = path
        if layer.get('particle_sets'):
            raise ValueError('Study tasks require rigid task bodies only')
    if not bodies or bodies != evaluation['body_paths']:
        raise ValueError('Study evaluation differs from native body inventory')
    if (scene['isaacsim']['physics_dt_s'] != .0005 or evaluation['physics_dt_s'] != .0005
            or task['public_rules'] != evaluation['configuration']['rules']):
        raise ValueError('Study integration profile or frozen rules mismatch')
    if project_task_rules(task) is None:
        raise ValueError('Study requires its frozen public referee contract')
    return initial


def validate_authored_reset(scene, task):
    # CPU USD composition establishes that the frozen reset really belongs to
    # the authored bodies at preparation. Native launch verifies the frozen
    # dependency hashes before SimulationApp makes USD available; the mounted
    # runtime baseline is checked again by the frozen sequence evaluator.
    from pxr import Usd, UsdPhysics
    import numpy as np
    from bench.simulation.isaacsim_backend.task_layers import mount_task_layers
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    native = mount_task_layers(stage, scene['task_layers'], physics_scene_path='/World/PhysicsScene')
    if any(row['joints'] for row in native['layers']):
        raise ValueError('Study props cannot contain authored joints or world anchors')
    for name, row in native['objects'].items():
        expected = task['evaluation']['configuration']['objects'][name]['initial_pose']
        actual = row['initial_state']
        q, wanted = np.asarray(actual['quaternion_wxyz']), np.asarray(expected['quaternion_wxyz'])
        if (not np.allclose(actual['position_m'], expected['position_m'], atol=1.e-9, rtol=0)
                or min(np.linalg.norm(q-wanted), np.linalg.norm(q+wanted)) > 1.e-9):
            raise ValueError('Study evaluator reset differs from authored native body pose')


def _freeze_evaluator(destination):
    source = Path(__file__).parent
    destination.mkdir()
    for original, frozen in (('evaluation.py', 'sequence_evaluator.py'),
                             ('rules.py', 'rules.py'), ('native_observer.py', 'native_observer.py')):
        shutil.copy2(source / original, destination / frozen)


def validate_release(root, manifest, *, assets=True):
    from bench.isaacsim.release import _frozen_evaluator_module
    root = Path(root).resolve()
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('status') != 'prepared_unvalidated'
            or manifest.get('physics_accepted') is not False
            or manifest.get('task_policy_success_claimed') is not False
            or manifest.get('original_release_modified') is not False
            or manifest.get('source_release') is not None
            or manifest.get('task_kind') != 'single_agent'):
        raise ValueError('Study engineering release cannot claim native qualification')
    files = manifest.get('files', {})
    required = {'scene.json', 'task.json', 'initial_state.json', 'rules.json',
        'asset_file_hashes.json', 'physics_profile.json', 'source_provenance.json',
        'evaluator/sequence_evaluator.py', 'evaluator/rules.py', 'evaluator/native_observer.py'}
    if not required <= files.keys() or manifest.get('files_sha256') != {p: r['sha256'] for p, r in files.items()}:
        raise ValueError('Incomplete frozen study inventory')
    for relative, row in files.items():
        path = root / relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not path.resolve().is_relative_to(root)
                or any(p.is_symlink() for p in (path, *path.parents) if p != root)):
            raise ValueError('Study release path escapes its frozen directory')
        checked(path, row['sha256'])
    scene, task, initial, rules, profile = (load_json(root / name) for name in
        ('scene.json', 'task.json', 'initial_state.json', 'rules.json', 'physics_profile.json'))
    entries = manifest.get('tasks', [])
    if (len(entries) != 1 or entries[0].get('id') != task['id']
            or manifest.get('task_id') != task['id'] or entries[0].get('task_kind') != 'single_agent'
            or any(entries[0].get(k) != k + '.json'
                   or entries[0].get(k + '_sha256') != files[k + '.json']['sha256']
                   for k in ('scene', 'task', 'initial_state'))):
        raise ValueError('Study task entry differs from frozen inputs')
    if (initial_state(scene, task) != initial or task['public_rules'] != rules
            or profile.get('isaacsim_options') != scene['isaacsim']
            or manifest.get('physics_profile') != 'physics_profile.json'
            or manifest.get('physics_profile_sha256') != files['physics_profile.json']['sha256']):
        raise ValueError('Study frozen reset, rules or integration profile mismatch')
    inventory = load_json(root / 'asset_file_hashes.json')
    if manifest.get('asset_files_sha256') != {p: r['sha256'] for p, r in inventory.items()}:
        raise ValueError('Study asset inventory indexes disagree')
    if assets and inventory != asset_inventory(scene):
        raise ValueError('Study dependency inventory is incomplete or changed')
    _frozen_evaluator_module(root).SequenceEvaluator(task)
    return manifest


def prepare_task_release(task_id, output, *, seed=0, inputs=None):
    from .scene import environment_reference
    if task_id not in TASK_IDS:
        raise ValueError('Unknown study task')
    output = Path(output).resolve()
    if not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Choose a fresh study release directory under shared datasets')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Episode seed must be an unsigned 32-bit integer')
    if inputs is None:
        from .composition import build_task_inputs
        inputs = build_task_inputs(task_id, seed=seed)
    scene, task = deepcopy(inputs['scene']), deepcopy(inputs['task'])
    if scene['workroom'] != environment_reference() or task['id'] != task_id:
        raise ValueError('New study publication must use its selected fixed environment')
    initial = initial_state(scene, task)
    validate_authored_reset(scene, task)
    if 'initial_state' in inputs and inputs['initial_state'] != initial:
        raise ValueError('Supplied study reset differs from task reset')
    inventory = asset_inventory(scene)
    output.mkdir(parents=True)
    try:
        for name, value in (('scene.json', scene), ('task.json', task), ('initial_state.json', initial),
                            ('rules.json', task['public_rules']), ('asset_file_hashes.json', inventory)):
            dump(output / name, value)
        dump(output / 'physics_profile.json', {'backend': 'Isaac Sim native PhysX GPU',
            'isaacsim_options': deepcopy(scene['isaacsim']), 'material_calibrated': False,
            'native_task_success_claimed': False})
        _freeze_evaluator(output / 'evaluator')
        source = Path(__file__).parent
        dump(output / 'source_provenance.json', {'sources_sha256':
            {str(p): digest(p) for p in sorted(source.rglob('*.py'))},
            'parent_environment': scene['workroom'], 'original_assets_modified': False,
            'original_releases_modified': False, 'physics_accepted': False})
        files = {str(p.relative_to(output)): {'sha256': digest(p)}
            for p in sorted(output.rglob('*')) if p.is_file()}
        entry = {key: task[key] for key in ('id', 'title_zh', 'difficulty', 'cloth')}
        entry['task_kind'] = 'single_agent'
        for name in ('scene', 'task', 'initial_state'):
            entry.update({name: name + '.json', name + '_sha256': files[name + '.json']['sha256']})
        manifest = {'schema': SCHEMA, 'schema_version': VERSION, 'task_id': task_id,
            'release_id': task_id + '_' + output.name,
            'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_unvalidated', 'source_release': None,
            'original_release_modified': False, 'task_kind': 'single_agent', 'tasks': [entry],
            'files': files, 'files_sha256': {p: r['sha256'] for p, r in files.items()},
            'asset_files_sha256': {p: r['sha256'] for p, r in inventory.items()},
            'physics_profile': 'physics_profile.json',
            'physics_profile_sha256': files['physics_profile.json']['sha256'],
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'qualification_scope': 'Executable candidate; native initialization and manipulation remain unqualified.'}
        dump(output / 'release_manifest.json', manifest)
        validate_release(output, manifest)
        return manifest
    except BaseException:
        shutil.rmtree(output)
        raise


def task_inputs(root, task_id, manifest, *, seed=0, phase='nominal'):
    if phase != 'nominal' or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Study requires its nominal reset and an unsigned 32-bit episode seed')
    root = Path(root).resolve()
    validate_release(root, manifest)
    if task_id != manifest['task_id']:
        raise ValueError('Task is absent from study release')
    entry = deepcopy(manifest['tasks'][0])
    scene, task, initial = (load_json(root / entry[name])
                            for name in ('scene', 'task', 'initial_state'))
    # Seeds identify episodes; the published physical reset/goals stay frozen.
    task['episode_seed'] = seed
    scene['scenario']['seed'] = seed
    task['evaluation_phase'] = phase
    return entry, scene, task, initial
