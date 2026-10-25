"""Freeze independent mechanical tasks without changing their parent room."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import math
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump
from bench.paths import DATA_ROOT, load_json

SCHEMA = 'mechanical_assembly_task_release.v1'
VERSION = 'isaacsim-mechanical-assembly-release-v1'
TASK_SCHEMAS = {
    'parts_sorting': 'mechanical-assembly-task-v1',
    'peg_insertion': 'mechanical-peg-insertion-task-v1',
    'shaft_spacer_assembly': 'mechanical-shaft-spacer-task-v1',
    'gear_train_assembly': 'mechanical-gear-train-task-v1',
}
DATASETS = Path(f'{DATA_ROOT}/datasets')


def asset_inventory(scene):
    from bench.simulation.isaacsim_backend.presentation import presentation_dependencies
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from bench.simulation.isaacsim_backend.workroom import verify_reference
    from .robot_layout import robot_asset_reference
    reference = scene['workroom']
    verify_reference(reference)
    hashes = {reference['usd_path']: reference['usd_sha256'],
              reference['dependency_manifest_path']: reference['dependency_manifest_sha256']}
    description = load_json(Path(reference['dependency_manifest_path']))
    hashes.update({p: r['sha256'] for p, r in description['dependencies'].items()})
    hashes.update(presentation_dependencies({'presentation': scene.get('workroom_presentation')}))
    robot = robot_asset_reference(scene['robot'])
    hashes.update({robot['root_usd']: robot['root_usd_sha256'],
        robot['source_manifest']: robot['source_manifest_sha256'],
        robot['import_manifest']: robot['import_manifest_sha256'], **robot['derived_files']})
    for spec in scene['objects']+scene['fixtures']:
        hashes.update(presentation_dependencies(spec))
        asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        hashes[asset['manifest_path']] = asset['manifest_sha256']
        hashes.update(asset.get('source_dependencies_sha256', {}))
        for row in [asset, *asset['visual_parts'], *asset['collision_parts']]:
            if 'mesh_path' in row:
                hashes[row['mesh_path']] = row['sha256']
        for name in ('SOURCE.json', 'LICENSE.txt'):
            path = Path(spec['asset_root'])/name
            if path.is_file():
                hashes[str(path.resolve())] = digest(path)
    for path, value in hashes.items():
        checked(path, value)
    return {p: {'sha256': h} for p, h in sorted(hashes.items())}


def initial_state(scene, task):
    from bench.control.factory import public_robot_geometry, robot_configuration
    from .scene import environment_reference
    ident = task.get('id')
    if (ident not in TASK_SCHEMAS or task.get('task_id') != ident
            or task.get('schema_version') != TASK_SCHEMAS[ident]
            or scene.get('workroom') != environment_reference()
            or scene.get('scenario', {}).get('task_id') != ident
            or scene.get('cloth', {}).get('enabled') is not False):
        raise ValueError('Mechanical task identity, environment or schema mismatch')
    initial = deepcopy(task.get('initial_state'))
    if not isinstance(initial, dict) or set(initial) != {'joint_positions_rad'}:
        raise ValueError('Mechanical task reset contains only the complete ready robot joints')
    robot = scene['robot']
    if (robot.get('hand_id') != 'sharpa_w01_right'
            or robot.get('assembly_version') != 'isaacsim_6_1_v2'):
        raise ValueError('Mechanical assembly requires the selected single-arm SHARPA robot')
    model = public_robot_geometry(robot_configuration(scene))
    joints = initial['joint_positions_rad']
    if (not isinstance(joints, dict) or set(joints) != set(model.names)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in joints.values())):
        raise ValueError('Every active joint must have a finite reset coordinate')
    model.checked_joints(joints, enforce_limits=True)
    objects = scene.get('objects', [])
    if not objects or len({r['id'] for r in objects}) != len(objects):
        raise ValueError('Task dynamic inventory is empty or duplicated')
    expected = task.get('evaluation', {}).get('initial_objects', {})
    from .table_distractors import validate_distractors
    distractors = validate_distractors(scene, task)
    scored_objects = [row for row in objects if row['id'] not in distractors]
    if set(expected) != {row['id'] for row in scored_objects} or any(
            any(expected[row['id']].get(k) != row.get(k) for k in ('position_m', 'quaternion_wxyz'))
            for row in scored_objects):
        raise ValueError('Mechanical evaluator reset differs from the physical object inventory')
    if scene.get('mechanical_task_provenance', {}).get('engineering_probe') or task.get('engineering_probe'):
        raise ValueError('A reset-only diagnostic cannot replace the original task release')
    return initial


def validate_release(root, manifest, *, assets=True):
    from bench.isaacsim.release import _frozen_evaluator_module
    root = Path(root).resolve()
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('status') != 'prepared_unvalidated'
            or manifest.get('physics_accepted') is not False
            or manifest.get('task_policy_success_claimed') is not False
            or manifest.get('source_release') is not None
            or manifest.get('original_release_modified') is not False):
        raise ValueError('Independent mechanical preparation cannot claim native acceptance')
    required = {'scene.json', 'task.json', 'initial_state.json', 'rules.json', 'asset_file_hashes.json',
                'physics_profile.json', 'source_provenance.json', 'evaluator/sequence_evaluator.py', 'evaluator/rules.py', 'evaluator/native_evidence.py'}
    files = manifest.get('files', {})
    if not required <= files.keys() or manifest.get('files_sha256') != {p: r['sha256'] for p, r in files.items()}:
        raise ValueError('Mechanical release lacks its complete frozen inputs')
    for relative, row in files.items():
        path = root/relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not path.resolve().is_relative_to(root)
                or any(p.is_symlink() for p in (path, *path.parents) if p != root)):
            raise ValueError('Mechanical release file escapes its frozen directory')
        checked(path, row['sha256'])
    scene, task, initial, rules, profile = (load_json(root/n) for n in
        ('scene.json', 'task.json', 'initial_state.json', 'rules.json', 'physics_profile.json'))
    entries = manifest.get('tasks', [])
    if (len(entries) != 1 or manifest.get('task_id') != task['id'] or entries[0].get('id') != task['id']
            or any(entries[0].get(k) != k+'.json' or entries[0].get(k+'_sha256') != files[k+'.json']['sha256']
                   for k in ('scene', 'task', 'initial_state'))):
        raise ValueError('Mechanical task entry differs from its frozen inputs')
    if (initial_state(scene, task) != initial or task.get('public_rules') != rules
            or profile.get('isaacsim_options') != scene.get('isaacsim')
            or manifest.get('physics_profile') != 'physics_profile.json'
            or manifest.get('physics_profile_sha256') != files['physics_profile.json']['sha256']):
        raise ValueError('Mechanical reset, public rules or physics profile mismatch')
    if 'referee' in rules:
        from bench.protocol.task_referee import project_task_rules
        if 'evaluator/referee.py' not in files:
            raise ValueError('Mechanical referee source is not frozen')
        project_task_rules(task)
    inventory = load_json(root/'asset_file_hashes.json')
    if manifest.get('asset_files_sha256') != {p: r['sha256'] for p, r in inventory.items()}:
        raise ValueError('Mechanical asset inventory indexes disagree')
    if assets:
        if inventory != asset_inventory(scene):
            raise ValueError('Mechanical asset inventory is incomplete or changed')
        checked(manifest['asset_catalog']['path'], manifest['asset_catalog']['sha256'])
    _frozen_evaluator_module(root).SequenceEvaluator(task)
    return manifest


def prepare_task_release(task_id, output, **options):
    if task_id not in TASK_SCHEMAS:
        raise ValueError('Use the existing independent release for this task')
    output = Path(output).resolve()
    if not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Choose a fresh task release directory under shared datasets')
    module = importlib.import_module(f'scenes.workrooms.mechanical_assembly.tasks.{task_id}.task')
    values = module.build_task_inputs(**options)
    scene, task = values['scene'], values['task']
    initial = initial_state(scene, task)
    inventory = asset_inventory(scene)
    source = Path(__file__).parent
    evaluator_source = source/'tasks'/task_id if task_id == 'gear_train_assembly' else source
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for name, value in [('scene.json', scene), ('task.json', task), ('initial_state.json', initial),
                            ('rules.json', task['public_rules']), ('asset_file_hashes.json', inventory)]:
            dump(staging/name, value)
        dump(staging/'physics_profile.json', {'backend': 'Isaac Sim 6.1 native PhysX GPU',
            'isaacsim_options': deepcopy(scene['isaacsim']), 'material_calibrated': False,
            'native_task_success_claimed': False})
        (staging/'evaluator').mkdir()
        for name, dest in [('evaluation.py' if task_id != 'gear_train_assembly' else 'evaluator.py', 'sequence_evaluator.py'), ('rules.py', 'rules.py')]:
            shutil.copy2(evaluator_source/name, staging/'evaluator'/dest)
            checked(staging/'evaluator'/dest, digest(evaluator_source/name))
        shutil.copy2(source/'native_evidence.py', staging/'evaluator/native_evidence.py')
        checked(staging/'evaluator/native_evidence.py', digest(source/'native_evidence.py'))
        shutil.copy2(source/'referee.py', staging/'evaluator/referee.py')
        checked(staging/'evaluator/referee.py', digest(source/'referee.py'))
        dump(staging/'source_provenance.json', {
            'sources_sha256': {str(p): digest(p) for p in sorted(source.rglob('*.py'))},
            'parent_environment': scene['workroom'], 'original_assets_modified': False,
            'original_releases_modified': False, 'physics_accepted': False})
        files = {str(p.relative_to(staging)): {'sha256': digest(p)} for p in sorted(staging.rglob('*')) if p.is_file()}
        entry = {k: task[k] for k in ('id', 'title_zh', 'difficulty', 'cloth')}
        for name in ('scene', 'task', 'initial_state'):
            entry.update({name: name+'.json', name+'_sha256': files[name+'.json']['sha256']})
        catalog = Path(scene['objects'][0]['asset_root'])/'manifest.json'
        manifest = {'schema': SCHEMA, 'schema_version': VERSION, 'task_id': task_id,
            'release_id': task_id+'_'+output.name, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_unvalidated', 'source_release': None, 'original_release_modified': False,
            'tasks': [entry], 'files': files, 'files_sha256': {p: r['sha256'] for p, r in files.items()},
            'asset_files_sha256': {p: r['sha256'] for p, r in inventory.items()},
            'physics_profile': 'physics_profile.json', 'physics_profile_sha256': files['physics_profile.json']['sha256'],
            'asset_catalog': {'path': str(catalog), 'sha256': digest(catalog)},
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'qualification_scope': 'Frozen executable candidate; later scene dynamics evidence is attached separately.'}
        dump(staging/'release_manifest.json', manifest)
        validate_release(staging, manifest)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
