"""Freeze mechanical NIST rules, referee, observer and canonical episode inputs."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump
from .nist_composition import TASK_IDS, TASK_SCHEMA, PHYSICS_DT, build_task_inputs, evaluator_sources, bind_episode_task, load_asset_manifest
from .scene import environment_reference
from bench.paths import DATA_ROOT, load_json

SCHEMA = 'mechanical_nist_task_release.v1'
VERSION = 'isaacsim-mechanical-nist-release-v1'
DATASETS = Path(f'{DATA_ROOT}/datasets')


def asset_inventory(scene):
    from bench.simulation.isaacsim_backend.workroom import verify_reference
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    reference = scene['workroom']
    verify_reference(reference)
    hashes = {reference['usd_path']: reference['usd_sha256'],
              reference['dependency_manifest_path']: reference['dependency_manifest_sha256']}
    description = load_json(Path(reference['dependency_manifest_path']))
    hashes.update({p: r['sha256'] for p, r in description['dependencies'].items()})
    robot = scene['robot']
    root = Path(robot['assets_root'])/'robots/assemblies/fr3_sharpa'/robot['assembly_version']
    source_file, import_file = root/'SOURCE.json', root/'USD_IMPORT.json'
    source, imported = load_json(source_file), load_json(import_file)
    hashes.update({str(source_file): digest(source_file), str(import_file): digest(import_file),
        imported['usd_path']: imported['usd_sha256'], **imported['derived_files'],
        source['urdf_path']: source['urdf_sha256'], **source['sources'], **source['mesh_inputs'], **source['license_sha256']})
    for spec in scene['fixtures']:
        asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        hashes[asset['manifest_path']] = asset['manifest_sha256']
        hashes.update(asset.get('source_dependencies_sha256', {}))
        for row in [asset, *asset['visual_parts'], *asset['collision_parts']]:
            if 'mesh_path' in row: hashes[row['mesh_path']] = row['sha256']
    candidate = scene['mechanical_nist_provenance']['asset_candidate']
    verified, metadata = load_asset_manifest(scene['native_task_bundle']['observer']['configuration']['id'], candidate)
    if verified != candidate:
        raise ValueError('Task asset descriptor differs from its manifest')
    if metadata != scene['native_task_bundle']['observer']['configuration']['asset_metadata']:
        raise ValueError('Task asset metadata differs from frozen referee input')
    manifest = load_json(Path(candidate['manifest_path']))
    hashes.update({candidate['manifest_path']: candidate['manifest_sha256'],
        candidate['usd_path']: candidate['usd_sha256'],
        str(Path(candidate['usd_path']).with_name('metadata.json')): manifest['metadata_sha256']})
    hashes.update({p: r['sha256'] for p, r in manifest['dependencies'].items()})
    hashes.update(candidate['auxiliary_files_sha256'])
    for path, value in hashes.items(): checked(path, value)
    return {p: {'sha256': h} for p, h in sorted(hashes.items())}


def initial_state(scene, task):
    from bench.control.factory import public_robot_geometry, robot_configuration as controller_configuration
    if (task.get('id') not in TASK_IDS or task.get('task_id') != task['id']
            or task.get('schema_version') != TASK_SCHEMA or scene.get('workroom') != environment_reference()):
        raise ValueError('NIST-derived mechanical task/parent identity differs')
    from .task_scene import DEFAULT_STAND_ROOT
    expected_fixtures = [{'id': 'robot_stand', 'asset_id': 'mechanical_robot_stand',
        'asset_root': str(DEFAULT_STAND_ROOT), 'position_m': [0., 0., 0.]}]
    if (scene.get('objects') != [] or scene.get('fixtures') != expected_fixtures
            or scene.get('cloth', {}).get('enabled') is not False
            or 'scenario' in scene
            or scene.get('mechanical_nist_provenance', {}).get('engineering_probe')
            or task.get('engineering_probe')):
        raise ValueError('NIST task must retain the original stand and task-owned bundle')
    if (scene.get('isaacsim', {}).get('physics_dt_s') != PHYSICS_DT
            or scene.get('physics', {}).get('dt_s') != PHYSICS_DT
            or scene.get('physics', {}).get('control_dt_s') != .02
            or task.get('evaluation', {}).get('physics_dt_s') != PHYSICS_DT):
        raise ValueError('NIST physical and referee clocks differ')
    from .robot_layout import robot_configuration, ready_configuration
    expected_robot = robot_configuration()
    # The original ready IK is recomputed by two pinned Python runtimes.
    # Their BLAS roundoff is ~4e-16 rad; all other robot fields stay exact.
    actual_robot = deepcopy(scene.get('robot', {}))
    actual_home = actual_robot.pop('arm_home_rad', [])
    expected_home = expected_robot.pop('arm_home_rad', [])
    if (actual_robot != expected_robot or len(actual_home) != len(expected_home)
            or any(not math.isclose(a, b, abs_tol=1.e-12, rel_tol=0.)
                   for a, b in zip(actual_home, expected_home))):
        raise ValueError('Frozen mechanical NIST robot differs from its task installation')
    expected_initial, _ = ready_configuration(scene['robot'])
    initial = deepcopy(task.get('initial_state'))
    if not isinstance(initial, dict) or set(initial) != {'joint_positions_rad'}:
        raise ValueError('NIST-derived mechanical task reset may contain only robot joint coordinates')
    expected_joints = expected_initial['joint_positions_rad']
    if (set(initial['joint_positions_rad']) != set(expected_joints)
            or any(not math.isclose(v, expected_joints[k], abs_tol=1.e-12, rel_tol=0.)
                   for k, v in initial['joint_positions_rad'].items())):
        raise ValueError('NIST-derived mechanical reset must preserve the original ready joints')
    model = public_robot_geometry(controller_configuration(scene))
    joints = initial['joint_positions_rad']
    if set(joints) != set(model.names) or any(type(v) not in (int, float) or not math.isfinite(v) for v in joints.values()):
        raise ValueError('Complete finite robot joint reset required')
    model.checked_joints(joints, enforce_limits=True)
    bundle = scene['native_task_bundle']
    candidate = scene['mechanical_nist_provenance']['asset_candidate']
    if (bundle['observer']['configuration'] != task or bundle['rigid_bodies'] != task['asset_metadata']['actors']
            or bundle['observer'].get('class_name') != 'NativeTaskObserver'
            or bundle['usd_path'] != candidate['usd_path'] or bundle['usd_sha256'] != candidate['usd_sha256']
            or bundle['root_prim_path'] != candidate['root_prim_path']
            or bundle['joints'] != task['asset_metadata'].get('joints', {})
            or bundle['particle_sets'] != task['asset_metadata'].get('particle_sets', {})):
        raise ValueError('Native observer, task physics and referee input differ')
    if bundle.get('reduced_dynamics') or bundle.get('volume_visual_bindings'):
        raise ValueError('NIST mechanisms require native passive bodies and constraints')
    return initial


def validate_release(root, manifest, *, assets=True):
    from bench.isaacsim.release import _frozen_evaluator_module
    root = Path(root).resolve()
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('status') != 'prepared_unvalidated'
            or manifest.get('physics_accepted') is not False or manifest.get('task_policy_success_claimed') is not False
            or manifest.get('source_release') is not None
            or manifest.get('original_release_modified') is not False):
        raise ValueError('NIST-derived mechanical release is an executable candidate, not manipulation qualification')
    files = manifest['files']
    required = {'scene.json', 'task.json', 'initial_state.json', 'rules.json', 'asset_file_hashes.json',
                'physics_profile.json', 'source_provenance.json', 'evaluator/sequence_evaluator.py', 'evaluator/rules.py'}
    required.add('evaluator/native_evidence.py')
    if not required <= files.keys() or manifest.get('files_sha256') != {p: v['sha256'] for p, v in files.items()}:
        raise ValueError('Incomplete frozen task inputs')
    if (manifest.get('physics_profile') != 'physics_profile.json'
            or manifest.get('physics_profile_sha256') != files['physics_profile.json']['sha256']):
        raise ValueError('Frozen physics profile reference differs')
    for name, row in files.items():
        path = root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents) if p != root):
            raise ValueError('Release file escapes its frozen root')
        checked(path, row['sha256'])
    scene, task, initial, rules, profile = [load_json(root/p) for p in
        ('scene.json', 'task.json', 'initial_state.json', 'rules.json', 'physics_profile.json')]
    if initial_state(scene, task) != initial or rules != task['public_rules'] or profile['isaacsim_options'] != scene['isaacsim']:
        raise ValueError('Frozen rule/reset/physics inputs disagree')
    entry = manifest['tasks']
    if len(entry) != 1 or manifest['task_id'] != task['id'] or entry[0]['id'] != task['id']:
        raise ValueError('Release requires exactly its declared task')
    for key in ('scene', 'task', 'initial_state'):
        if entry[0][key] != key+'.json' or entry[0][key+'_sha256'] != files[key+'.json']['sha256']:
            raise ValueError('Frozen task entry differs')
    observer = scene['native_task_bundle']['observer']
    if observer['path'] not in files or observer['sha256'] != files[observer['path']]['sha256']:
        raise ValueError('Observer entry is not frozen')
    expected = {p.removeprefix('evaluator/'): v['sha256'] for p, v in files.items() if p.startswith('evaluator/')}
    if observer['files'] != expected: raise ValueError('Observer helper inventory differs')
    if {str(p.relative_to(root/'evaluator')) for p in (root/'evaluator').rglob('*.py')} != set(expected):
        raise ValueError('Observer directory has undeclared or missing Python dependencies')
    inventory = load_json(root/'asset_file_hashes.json')
    candidate = scene['mechanical_nist_provenance']['asset_candidate']
    if manifest.get('asset_catalog') != {
            'path': candidate['manifest_path'], 'sha256': candidate['manifest_sha256']}:
        raise ValueError('Frozen asset catalog differs from the selected candidate')
    if manifest.get('asset_files_sha256') != {p: v['sha256'] for p, v in inventory.items()}:
        raise ValueError('Asset inventory indexes differ')
    if assets and inventory != asset_inventory(scene): raise ValueError('Frozen shared assets changed')
    evaluator = _frozen_evaluator_module(root)
    frozen_rules = importlib.import_module(evaluator.__package__+'.rules')
    if bind_episode_task(task, task['episode_seed'], frozen_rules) != task:
        raise ValueError('Frozen task differs from its rules or asset-bound evaluation')
    from bench.protocol.task_referee import project_task_rules
    project_task_rules(task)
    evaluator.SequenceEvaluator(task)
    return manifest


def episode_inputs(root, scene, task, seed):
    from bench.isaacsim.release import _frozen_evaluator_module
    evaluator = _frozen_evaluator_module(Path(root).resolve())
    rules = importlib.import_module(evaluator.__package__+'.rules')
    task = bind_episode_task(task, seed, rules)
    scene['native_task_bundle']['observer']['configuration'] = deepcopy(task)
    initial = initial_state(scene, task)
    relative = scene['native_task_bundle']['observer']['path']
    path = (Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()): raise ValueError('Observer escapes release')
    scene['native_task_bundle']['observer']['path'] = str(path)
    evaluator.SequenceEvaluator(task)
    return scene, task, initial


def prepare_task_release(task_id, output, *, asset_manifest):
    output = Path(output).resolve()
    if task_id not in TASK_IDS or not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Choose a fresh mechanical NIST task release under shared datasets')
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    project = Path(__file__).resolve().parents[3]
    modules = ('scenes.workrooms.mechanical_assembly.nist_composition',
        'scenes.workrooms.mechanical_assembly.robot_layout',
        'scenes.workrooms.mechanical_assembly.nist_rules',
        'scenes.workrooms.mechanical_assembly.nist_evaluation',
        'scenes.workrooms.mechanical_assembly.nist_native_evidence')
    for name in modules:
        importlib.import_module(name)
    snapshot = verify_if_snapshot(project, required_modules=modules, entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Prepare mechanical NIST releases from complete frozen project source')
    values = build_task_inputs(task_id, asset_manifest=asset_manifest)
    scene, task, initial = [values[k] for k in ('scene', 'task', 'initial_state')]
    inventory = asset_inventory(scene)
    snapshot = verify_if_snapshot(project, required_modules=modules, entrypoint_file=Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for name, value in [('scene.json', scene), ('task.json', task), ('initial_state.json', initial),
                            ('rules.json', task['public_rules']), ('asset_file_hashes.json', inventory)]:
            dump(staging/name, value)
        dump(staging/'physics_profile.json', {'backend': 'Isaac Sim 6.1 native PhysX GPU',
            'isaacsim_options': deepcopy(scene['isaacsim']), 'material_calibrated': False,
            'known_limitations': task['known_limitations'], 'native_task_success_claimed': False})
        (staging/'evaluator').mkdir()
        for name, path in evaluator_sources(task_id).items():
            shutil.copy2(path, staging/'evaluator'/name)
            checked(staging/'evaluator'/name, digest(path))
        dump(staging/'source_provenance.json', {'source_snapshot': snapshot, 'source_root': str(project), 'sources_sha256': {
            str(p): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
            'parent_environment': scene['workroom'], 'original_assets_modified': False,
            'original_releases_modified': False, 'shared_controller_unchanged': True})
        files = {str(p.relative_to(staging)): {'sha256': digest(p)} for p in sorted(staging.rglob('*')) if p.is_file()}
        entry = {k: task[k] for k in ('id', 'title_zh', 'difficulty', 'cloth')}
        for key in ('scene', 'task', 'initial_state'):
            entry.update({key: key+'.json', key+'_sha256': files[key+'.json']['sha256']})
        candidate = scene['mechanical_nist_provenance']['asset_candidate']
        manifest = {'schema': SCHEMA, 'schema_version': VERSION, 'task_id': task_id,
            'release_id': 'mechanical_nist_'+task_id+'_'+output.name, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_unvalidated', 'source_release': None, 'original_release_modified': False,
            'tasks': [entry], 'files': files, 'files_sha256': {p: v['sha256'] for p, v in files.items()},
            'asset_files_sha256': {p: v['sha256'] for p, v in inventory.items()},
            'physics_profile': 'physics_profile.json', 'physics_profile_sha256': files['physics_profile.json']['sha256'],
            'asset_catalog': {'path': candidate['manifest_path'], 'sha256': candidate['manifest_sha256']},
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'qualification_scope': 'Executable task with frozen rules/referee; full robot task success not claimed.'}
        dump(staging/'release_manifest.json', manifest)
        validate_release(staging, manifest)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
