"""Frozen robot-installation derivatives of existing, independently valid tasks.

Original family validators remain strict. This format binds its original
release and permits only a reviewed robot/stand/reset/camera replacement and
explicit left/right operator adaptations of frozen rules and contact checks.
"""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump
from .robot_configurations import SCHEMA, VERSION, CONFIGURATION_IDS, eligible
from bench.paths import DATA_ROOT, load_json

DATASETS = Path(f'{DATA_ROOT}/datasets')
TERMINAL_SCORING_INHERITANCE = 'base_effective_contract.v1'
KINEMATIC_ROBOT_KEYS = frozenset({
    'hand_id', 'assembly_version', 'base_position_m', 'base_quaternion_wxyz',
    'mount_position_m', 'mount_quaternion_wxyz', 'arm_home_rad',
    'arm_home_rad_by_side', 'arm_base_poses', 'active_sides',
})


def _read(path):
    return load_json(Path(path))


def _sources(root):
    return {str(p.relative_to(root/'evaluator')): p.read_text()
            for p in sorted((root/'evaluator').rglob('*.py'))}


def _text_hash(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _check_installation(source_scene, configuration):
    from bench.control.factory import public_robot_geometry, robot_configuration
    robot = configuration['robot']
    if (robot.get('hand_id') != 'dual_fr3_sharpa_w01'
            or robot.get('assembly_version') != 'isaacsim_6_1_v2'
            or set(robot.get('arm_base_poses', {})) != {'left', 'right'}):
        raise ValueError('Installation requires the official left/right SHARPA assembly')
    original = source_scene['robot']
    for key in original.keys() | robot.keys():
        if key not in KINEMATIC_ROBOT_KEYS and original.get(key) != robot.get(key):
            raise ValueError('Installation must preserve robot parameters: '+key)
    model = public_robot_geometry(robot_configuration({'robot': robot}))
    initial = configuration['initial_state']
    if set(initial) != {'joint_positions_rad'} or len(model.names) != 58:
        raise ValueError('Installation requires exactly 58 reset joint coordinates')
    if set(initial['joint_positions_rad']) != set(model.names):
        raise ValueError('Installation reset must cover every active joint')
    for side in model.sides.values():
        side.checked_joints({name: initial['joint_positions_rad'][name]
                             for name in side.names}, enforce_limits=True)
    fixture = configuration['stand_fixture']
    if (fixture.get('id') != 'bimanual_robot_stand'
            or fixture.get('asset_id') != 'bimanual_option_stand'
            or fixture.get('position_m') != [0., 0., 0.]):
        raise ValueError('Installation stand must use its frozen world layout')


def _compose(source, configuration, *, inherit_terminal_scoring=True):
    from .robot_configuration_adaptation import adapt_payload
    from scenes.runtime import terminal_scoring_contract
    scene, task = _read(source/'scene.json'), _read(source/'task.json')
    family, task_id = scene['workroom']['family'], task['id']
    if (not eligible(family, task_id)
            or configuration['configuration_id'] not in CONFIGURATION_IDS[1:]
            or configuration['family'] != family or configuration['task_id'] != task_id):
        raise ValueError('Task is outside the explicit dual-arm configuration scope')
    _check_installation(scene, configuration)
    # Resolve exact frozen-task compatibility before adaptation changes task
    # identity. New derivatives carry the source's public contract explicitly.
    contract = terminal_scoring_contract(task) if inherit_terminal_scoring else None
    task, sources, adaptation = adapt_payload(family, task_id, task, _sources(source))
    if contract is not None:
        task['terminal_scoring_contract'] = deepcopy(contract)
    initial = deepcopy(configuration['initial_state'])
    task['initial_state'] = deepcopy(initial)
    task['active_sides'] = ['left', 'right']
    scene['robot'] = deepcopy(configuration['robot'])
    # Only robot stands are exchanged; task fixtures, supplies and room stay.
    removed = {'robot_stand', 'stand_left'}
    scene['fixtures'] = [row for row in scene.get('fixtures', []) if row['id'] not in removed]
    scene['fixtures'].append(deepcopy(configuration['stand_fixture']))
    if 'task_layers' in scene:
        scene['task_layers'] = [row for row in scene['task_layers'] if row['id'] != 'robot_stands']
    cameras = deepcopy(scene['cameras'])
    wrist = cameras.pop('wrist')
    for side in ('left', 'right'):
        cameras[side+'_wrist'] = {**deepcopy(wrist), 'mount_link': side+'_hand_flange',
            'title_zh': ('左' if side == 'left' else '右')+'腕部相机'}
    scene['cameras'] = cameras
    scene['robot_configuration'] = {
        'id': configuration['configuration_id'], 'family': family, 'task_id': task_id,
        'same_agent': True, 'active_sides': ['left', 'right'],
        'layout_id': configuration['layout_id'],
    }
    bundle = scene.get('native_task_bundle', {})
    for hook_name in ('observer', 'reduced_dynamics'):
        if hook_name not in bundle:
            continue
        hook = bundle[hook_name]
        hook['configuration'] = deepcopy(task)
        relative = str(Path(hook['path']).relative_to('evaluator'))
        hook['sha256'] = _text_hash(sources[relative])
        hook['files'] = {name: _text_hash(text) for name, text in sources.items()}
    return scene, task, initial, sources, adaptation


def _inventory(source, scene):
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    # Retain upstream dependencies for provenance, even the replaced stand.
    inventory = _read(source/'asset_file_hashes.json')
    hashes = {p: row['sha256'] for p, row in inventory.items()}
    robot = scene['robot']
    root = Path(robot['assets_root'])/'robots/assemblies/dual_fr3_sharpa_w01'/robot['assembly_version']
    for side in ('left', 'right'):
        source_path, import_path = root/f'SOURCE_{side}.json', root/f'USD_IMPORT_{side}.json'
        original, imported = _read(source_path), _read(import_path)
        hashes.update({str(source_path): digest(source_path), str(import_path): digest(import_path),
            imported['usd_path']: imported['usd_sha256'], **imported['derived_files'],
            original['urdf_path']: original['urdf_sha256'], **original['sources'],
            **original['mesh_inputs'], **original['license_sha256']})
    fixture = next(r for r in scene['fixtures'] if r['id'] == 'bimanual_robot_stand')
    asset = load_task_asset(fixture['asset_id'], fixture['asset_root'], verify_hashes=True)
    hashes[asset['manifest_path']] = asset['manifest_sha256']
    hashes.update(asset.get('source_dependencies_sha256', {}))
    for row in [asset, *asset['visual_parts'], *asset['collision_parts']]:
        if 'mesh_path' in row:
            hashes[row['mesh_path']] = row['sha256']
    for name in ('SOURCE.json', 'LICENSE.txt'):
        path = Path(fixture['asset_root'])/name
        if path.is_file():
            hashes[str(path.resolve())] = digest(path)
    for path, value in hashes.items():
        checked(path, value)
    return {p: {'sha256': h} for p, h in sorted(hashes.items())}


def validate_release(root, manifest, *, assets=True):
    from bench.isaacsim.release import validate, _frozen_evaluator_module
    root = Path(root).resolve()
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('task_kind') != 'single_agent'
            or manifest.get('status') != 'prepared_unvalidated'
            or manifest.get('physics_accepted') is not False
            or manifest.get('task_policy_success_claimed') is not False
            or manifest.get('original_release_modified') is not False):
        raise ValueError('Robot configuration preparation cannot claim task qualification')
    inheritance = manifest.get('terminal_scoring_inheritance')
    if ('terminal_scoring_inheritance' in manifest
            and inheritance != TERMINAL_SCORING_INHERITANCE):
        raise ValueError('Unknown robot configuration terminal scoring inheritance')
    base = manifest['base_release']
    source = Path(base['path']).resolve()
    if source == root or not source.is_relative_to(DATASETS):
        raise ValueError('Robot configuration must bind an independent original release')
    checked(source/'release_manifest.json', base['manifest_sha256'])
    original = validate(source, assets=assets)
    if original.get('schema') == SCHEMA or original.get('schema') != base['schema']:
        raise ValueError('Nested or changed base configuration release')
    files = manifest.get('files', {})
    expected_files = set(original['files']) | {'robot_configuration.json'}
    if set(files) != expected_files or manifest.get('files_sha256') != {p:r['sha256'] for p,r in files.items()}:
        raise ValueError('Configuration release file inventory is incomplete')
    for relative, row in files.items():
        path = root/relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not path.resolve().is_relative_to(root)
                or any(p.is_symlink() for p in (path, *path.parents) if p != root)):
            raise ValueError('Configuration file escapes its frozen root')
        checked(path, row['sha256'])
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                    and not ('__pycache__' in p.relative_to(root).parts and p.suffix == '.pyc')}
    if actual_files != set(files) | {'release_manifest.json'}:
        raise ValueError('Undeclared files in robot configuration release')
    configuration = _read(root/'robot_configuration.json')
    # Missing marker identifies the immutable original derivation algorithm;
    # validating a legacy release must not inject fields into its frozen bytes.
    scene, task, initial, sources, adaptation = _compose(source, configuration,
        inherit_terminal_scoring=inheritance == TERMINAL_SCORING_INHERITANCE)
    for name, expected in (('scene.json', scene), ('task.json', task),
                           ('initial_state.json', initial), ('rules.json', task['public_rules'])):
        if _read(root/name) != expected:
            raise ValueError('Unapproved task or scene change in robot configuration: '+name)
    for relative, text in sources.items():
        if (root/'evaluator'/relative).read_text() != text:
            raise ValueError('Unapproved frozen scorer change: '+relative)
    allowed = {'scene.json', 'task.json', 'initial_state.json', 'rules.json',
        'source_provenance.json', 'asset_file_hashes.json'} | {'evaluator/'+p for p in sources}
    if any(digest(root/p) != row['sha256'] for p,row in original['files'].items() if p not in allowed):
        raise ValueError('Configuration changed an immutable upstream dependency')
    if (manifest.get('family') != configuration['family']
            or manifest.get('task_id') != task['id']
            or manifest.get('configuration_id') != configuration['configuration_id']
            or manifest.get('adaptation') != adaptation):
        raise ValueError('Configuration manifest identity differs from frozen task')
    snapshot = manifest['source_snapshot']
    snapshot_path = Path(snapshot['path']).resolve()
    if not snapshot_path.is_relative_to(DATASETS):
        raise ValueError('Configuration provenance must bind its frozen source snapshot')
    checked(snapshot_path, snapshot['sha256'])
    expected_provenance = _read(source/'source_provenance.json')
    expected_provenance['robot_configuration_derivation'] = {
        'base_release': base, 'configuration_id': configuration['configuration_id'],
        'adaptation': adaptation, 'shared_controller_unchanged': True,
        'original_assets_modified': False, 'original_release_modified': False,
        'physics_profile_unchanged': True,
        'source_snapshot_manifest_sha256': snapshot['sha256'],
    }
    if inheritance is not None:
        expected_provenance['robot_configuration_derivation']['terminal_scoring_inheritance'] = inheritance
    if _read(root/'source_provenance.json') != expected_provenance:
        raise ValueError('Configuration provenance differs from its bound original and snapshot')
    entries = manifest.get('tasks', [])
    if len(entries) != 1 or entries[0].get('id') != task['id']:
        raise ValueError('Configuration must contain exactly one original task')
    for key in ('scene', 'task', 'initial_state'):
        if (entries[0].get(key) != key+'.json'
                or entries[0].get(key+'_sha256') != digest(root/(key+'.json'))):
            raise ValueError('Configuration task entry hashes differ')
    if (manifest.get('physics_profile') != 'physics_profile.json'
            or manifest.get('physics_profile_sha256') != digest(root/'physics_profile.json')
            or _read(root/'physics_profile.json')['isaacsim_options'] != scene['isaacsim']):
        raise ValueError('Configuration must preserve its complete physics profile')
    inventory = _read(root/'asset_file_hashes.json')
    if manifest.get('asset_files_sha256') != {p:r['sha256'] for p,r in inventory.items()}:
        raise ValueError('Configuration shared asset indexes differ')
    if assets and inventory != _inventory(source, scene):
        raise ValueError('Configuration shared asset inventory changed')
    _frozen_evaluator_module(root).SequenceEvaluator(task)
    return manifest


def prepare_release(source, output, configuration):
    """Publish to a fresh directory from a complete frozen project snapshot."""
    from bench.isaacsim.release import validate
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    project = Path(__file__).resolve().parents[2]
    if verify_if_snapshot(project, required_modules=(__name__,), entrypoint_file=Path(__file__)) is None:
        raise ValueError('Prepare robot configurations from complete frozen project source')
    source, output = Path(source).resolve(), Path(output).resolve()
    original = validate(source)
    if original.get('schema') == SCHEMA or output.exists() or not output.is_relative_to(DATASETS):
        raise ValueError('Choose a fresh, independent configuration release under datasets')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for relative, row in original['files'].items():
            checked(source/relative, row['sha256'])
            destination = staging/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/relative, destination)
            destination.chmod(0o644)
        scene, task, initial, sources, adaptation = _compose(source, configuration)
        for relative, text in sources.items():
            (staging/'evaluator'/relative).write_text(text)
        inventory = _inventory(source, scene)
        base = {'path': str(source), 'manifest_sha256': digest(source/'release_manifest.json'),
                'schema': original['schema']}
        provenance = _read(source/'source_provenance.json')
        provenance['robot_configuration_derivation'] = {
            'base_release': base, 'configuration_id': configuration['configuration_id'],
            'adaptation': adaptation, 'shared_controller_unchanged': True,
            'original_assets_modified': False, 'original_release_modified': False,
            'physics_profile_unchanged': True,
            'source_snapshot_manifest_sha256': digest(project/'source_snapshot_manifest.json'),
            'terminal_scoring_inheritance': TERMINAL_SCORING_INHERITANCE,
        }
        for name, value in (('scene.json', scene), ('task.json', task),
                ('initial_state.json', initial), ('rules.json', task['public_rules']),
                ('robot_configuration.json', configuration), ('asset_file_hashes.json', inventory),
                ('source_provenance.json', provenance)):
            dump(staging/name, value)
        files = {str(p.relative_to(staging)): {'sha256': digest(p)}
                 for p in sorted(staging.rglob('*')) if p.is_file()}
        entry = deepcopy(original['tasks'][0])
        for key in ('scene', 'task', 'initial_state'):
            entry[key] = key+'.json'
            entry[key+'_sha256'] = files[key+'.json']['sha256']
        manifest = {'schema': SCHEMA, 'schema_version': VERSION,
            'terminal_scoring_inheritance': TERMINAL_SCORING_INHERITANCE,
            'family': configuration['family'], 'task_id': task['id'], 'task_kind': 'single_agent',
            'configuration_id': configuration['configuration_id'], 'base_release': base,
            'source_snapshot': {'path': str(project/'source_snapshot_manifest.json'),
                                'sha256': digest(project/'source_snapshot_manifest.json')},
            'release_id': task['id']+'_'+configuration['configuration_id']+'_'+output.name,
            'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_unvalidated', 'original_release_modified': False,
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'tasks': [entry], 'files': files, 'files_sha256': {p:r['sha256'] for p,r in files.items()},
            'asset_files_sha256': {p:r['sha256'] for p,r in inventory.items()},
            'physics_profile': 'physics_profile.json',
            'physics_profile_sha256': files['physics_profile.json']['sha256'],
            'asset_catalog': deepcopy(original.get('asset_catalog')), 'adaptation': adaptation,
            'qualification_scope': 'Explicit single-Agent left/right robot installation; IK, native interface and manipulation evidence are separate.'}
        dump(staging/'release_manifest.json', manifest)
        validate_release(staging, manifest)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def task_inputs(root, task_id, manifest, *, seed=0, phase='nominal'):
    """Bind episode targets using this variant's frozen helpers and original semantics."""
    from bench.isaacsim.release import _frozen_evaluator_module
    root = Path(root).resolve()
    if (phase != 'nominal' or task_id != manifest['task_id']
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError('Configuration requires its exact task, nominal reset and uint32 seed')
    scene, task, initial = [_read(root/(key+'.json')) for key in ('scene', 'task', 'initial_state')]
    module = _frozen_evaluator_module(root)
    base_schema = manifest['base_release']['schema']
    binders = {
        'mechanical_nist_task_release.v1': 'scenes.workrooms.mechanical_assembly.nist_composition',
        'laboratory_manipulation_task_release.v1': 'scenes.workrooms.laboratory.composition',
        'living_room_manipulation_task_release.v1': 'scenes.workrooms.living_room.runtime_task',
    }
    if base_schema in binders:
        rules = importlib.import_module(module.__package__+'.rules')
        task = importlib.import_module(binders[base_schema]).bind_episode_task(task, seed, rules)
    elif base_schema == 'card_room_placement_task_release.v1':
        task = module.bind_episode_targets(task, seed)
    elif base_schema == 'study_task_release.v1':
        task['episode_seed'] = seed
        scene['scenario']['seed'] = seed
        task['evaluation_phase'] = phase
    elif base_schema == 'living_room_task_release.v1':
        task['seed'] = seed
        scene['scenario']['seed'] = seed
        task['evaluation_phase'] = phase
    bundle = scene.get('native_task_bundle', {})
    for name in ('observer', 'reduced_dynamics'):
        if name in bundle:
            hook = bundle[name]
            hook['configuration'] = deepcopy(task)
            path = (root/hook['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Configuration native helper escapes release')
            hook['path'] = str(path)
    module.SequenceEvaluator(task)
    return deepcopy(manifest['tasks'][0]), scene, task, initial
