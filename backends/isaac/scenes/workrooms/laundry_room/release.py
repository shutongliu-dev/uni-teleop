"""Immutable laundry task-only inputs, frozen scorer and shared-asset inventory."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import checked, digest, dump
from .assets import load_asset_manifest
from .composition import (build_task_inputs, evaluator_sources, CAPABILITIES,
                          ROBOT_CAPABILITIES, ROBOT_PHYSICS_OPTIONS, capabilities_for)
from .rules import (ALL_TASK_IDS as TASK_IDS, GARMENT_TASK_IDS, GARMENT_FOLDING_TASK_IDS, public_rules, bind_task_rules, digest as value_digest, rule_profile,
                    PRACTICAL_PROFILE, ENGINEERING_CONTROL, ROBOT_CONTROL, control_mode)
from .scene import environment_reference
from bench.paths import DATA_ROOT, load_json

SCHEMA = 'laundry_cloth_task_release.v1'
VERSION = 'isaacsim-laundry-cloth-release-v1'
DATASETS = Path(f'{DATA_ROOT}/datasets')


def asset_inventory(scene):
    from bench.simulation.isaacsim_backend.workroom import verify_reference
    reference = scene['workroom']
    verify_reference(reference)
    description = load_json(Path(reference['dependency_manifest_path']))
    hashes = {reference['usd_path']: reference['usd_sha256'],
        reference['dependency_manifest_path']: reference['dependency_manifest_sha256']}
    hashes.update({p: r['sha256'] for p, r in description['dependencies'].items()})
    candidate = scene['laundry_task_provenance']['asset_candidate']
    if not candidate:
        raise ValueError('A release requires a prepared towel asset manifest')
    path = Path(candidate['manifest_path'])
    task_id = load_json(path)['task_id']
    if task_id in GARMENT_FOLDING_TASK_IDS:
        from .garment_folding_assets import load_assets
        verified, _ = load_assets(task_id, path)
    else:
        verified, _ = load_asset_manifest(task_id, path)
    if candidate != verified:
        raise ValueError('Towel asset descriptor differs from its manifest')
    hashes.update(candidate['files_sha256'])
    if scene['robot'].get('enabled', True):
        from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
        from .robot_layout import STAND_ROOT, STAND_MANIFEST_SHA256
        robot = scene['robot']
        installation = scene.get('robot_configuration')
        single = installation is not None and installation['configuration_id'] == 'single_arm'
        if robot.get('hand_id') != ('sharpa_w01_right' if single else 'dual_fr3_sharpa_w01') or robot.get('assembly_version') != 'isaacsim_6_1_v2':
            raise ValueError('Laundry requires the pinned official left/right robot assembly')
        if single:
            from bench.simulation.isaacsim_backend.robot_asset import sharpa_assembly_dir
            root = sharpa_assembly_dir(robot['assets_root'], robot['assembly_version'])
            records = [(root/'SOURCE.json', root/'USD_IMPORT.json')]
        else:
            root = Path(robot['assets_root'])/'robots/assemblies/dual_fr3_sharpa_w01'/robot['assembly_version']
            records = [(root/f'SOURCE_{side}.json', root/f'USD_IMPORT_{side}.json') for side in ('left', 'right')]
        for source_path, import_path in records:
            source, imported = [load_json(path) for path in (source_path, import_path)]
            hashes.update({str(source_path): digest(source_path), str(import_path): digest(import_path),
                imported['usd_path']: imported['usd_sha256'], **imported['derived_files'],
                source['urdf_path']: source['urdf_sha256'], **source['sources'],
                **source['mesh_inputs'], **source['license_sha256']})
        if installation is None:
            checked(STAND_ROOT/'manifest.json', STAND_MANIFEST_SHA256)
            expected_fixtures = [{'id': 'robot_stand', 'asset_id': 'laundry_dual_robot_stand',
                                  'asset_root': str(STAND_ROOT), 'position_m': [0., 0., 0.]}]
        else:
            from .robot_configurations import installation as checked_installation
            if checked_installation(installation['configuration_id'], installation['stand_manifest']['path']) != installation:
                raise ValueError('Frozen robot installation descriptor differs')
            expected_fixtures = [installation['fixture']]
        if scene['fixtures'] != expected_fixtures:
            raise ValueError('Laundry robot fixture inventory differs from its pinned installation')
        for spec in scene['fixtures']:
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
    visuals = {cloth['id']: cloth['visual_profile'] for cloth in scene.get('cloths', []) if 'visual_profile' in cloth}
    if visuals:
        # Render-only fabric appearance: each profile and its baked maps are pinned.
        from .fabric_visual_release import profile_inventory
        hashes.update(profile_inventory(visuals))
    for p, sha in hashes.items():
        checked(p, sha)
    return {p: {'sha256': sha} for p, sha in sorted(hashes.items())}


def initial_state(scene, task):
    mode = control_mode(task)
    is_robot = mode == ROBOT_CONTROL
    if (task.get('id') not in TASK_IDS or task.get('id') != task.get('task_id')
            or task.get('capabilities') != capabilities_for(mode) or scene.get('workroom') != environment_reference()
            or (not is_robot and scene.get('robot') != {'enabled': False})
            or scene.get('cloth') != {'enabled': False}):
        raise ValueError('Laundry parent, task identity or engineering capabilities differ')
    candidate = scene['laundry_task_provenance']['asset_candidate']
    if not candidate:
        raise ValueError('Frozen task must reference its immutable towel asset')
    physics = deepcopy(scene['isaacsim'])
    layout_id = scene['laundry_task_provenance'].get('robot_layout_id')
    installation = scene.get('robot_configuration')
    if is_robot:
        from .robot_layout import LAYOUT_ID
        if installation is not None:
            from .robot_configurations import layout_id as configured_layout
            expected_layout = configured_layout(installation['configuration_id'],
                version=installation.get('layout_version', 1))
            if task.get('robot_configuration_id') != installation['configuration_id']:
                raise ValueError('Task and scene robot configurations disagree')
        else:
            expected_layout = LAYOUT_ID
            if 'robot_configuration_id' in task:
                raise ValueError('Configured task requires its explicit installation')
        if (layout_id != expected_layout or task.get('task_kind') != 'single_agent'
                or scene.get('named_cloth_initialization') != 'author-only'):
            raise ValueError('Ordinary laundry robot requires its explicit layout and author-only cloth initialization')
        for key, value in ROBOT_PHYSICS_OPTIONS.items():
            if physics.pop(key, None) != value:
                raise ValueError('Frozen laundry robot actuator/contact profile differs: '+key)
    elif layout_id is not None or installation is not None or 'robot_configuration_id' in task:
        raise ValueError('Engineering task cannot silently inherit a robot installation')
    selection = ({'robot_layout': layout_id} if installation is None else
                 {'configuration_id': installation['configuration_id'], 'stand_manifest': installation['stand_manifest']['path']})
    visuals = {cloth['id']: cloth['visual_profile'] for cloth in scene.get('cloths', [])
               if 'visual_profile' in cloth}
    expected = build_task_inputs(task['id'], asset_manifest=candidate['manifest_path'],
        seed=task['episode_seed'], physics_options=physics, rule_profile=rule_profile(task),
        public_material_contract=task.get('public_material_contract'),
        cloth_visual_profiles=visuals or None, **selection)
    if is_robot:
        # Source ready IK is recomputed in CPU and native Python/BLAS builds.
        # Only the pre-existing robot ready quantities allow 1e-12 roundoff;
        # cloth vertices, rules, physics and all other fields remain exact.
        import math
        def equivalent(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                return set(a) == set(b) and all(equivalent(a[key], b[key]) for key in a)
            if isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
            if type(a) in (int, float) and type(b) in (int, float):
                return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-12, rel_tol=0.)
            return a == b
        pairs = [
            (scene['robot'], expected['scene']['robot'], 'arm_home_rad' if installation is not None and installation['configuration_id'] == 'single_arm' else 'arm_home_rad_by_side'),
            (scene['laundry_task_provenance'], expected['scene']['laundry_task_provenance'], 'ready_design'),
            (task['initial_state'], expected['task']['initial_state'], 'joint_positions_rad')]
        for actual_parent, expected_parent, key in pairs:
            if not equivalent(actual_parent.get(key), expected_parent.get(key)):
                raise ValueError('Frozen dual robot ready state differs: '+key)
            expected_parent[key] = deepcopy(actual_parent[key])
        expected['initial_state']['joint_positions_rad'] = deepcopy(task['initial_state']['joint_positions_rad'])
    if scene != expected['scene'] or task != expected['task']:
        raise ValueError('Frozen laundry scene/task differs from its asset, rules or physics')
    return deepcopy(expected['initial_state'])


def validate_release(root, manifest, *, assets=True):
    root = Path(root).resolve()
    if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
            or manifest.get('status') != 'prepared_unvalidated' or manifest.get('physics_accepted') is not False
            or manifest.get('task_policy_success_claimed') is not False or manifest.get('source_release') is not None
            or manifest.get('capabilities') not in (CAPABILITIES, ROBOT_CAPABILITIES)):
        raise ValueError('Laundry preparation cannot claim native or robot qualification')
    files = manifest.get('files', {})
    required = {'scene.json', 'task.json', 'initial_state.json', 'rules.json', 'asset_file_hashes.json',
        'physics_profile.json', 'source_provenance.json', 'evaluator/sequence_evaluator.py', 'evaluator/rules.py'}
    is_robot = manifest.get('capabilities') == ROBOT_CAPABILITIES
    if is_robot:
        required.add('evaluator/native_observer.py')
    if set(files) != required or manifest.get('files_sha256') != {p: row['sha256'] for p, row in files.items()}:
        raise ValueError('Incomplete frozen laundry inputs')
    for name, row in files.items():
        path = root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Release file escapes its frozen root')
        checked(path, row['sha256'])
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                    and not ('__pycache__' in p.relative_to(root).parts and p.suffix == '.pyc')}
    if actual_files != required | {'release_manifest.json'}:
        raise ValueError('Undeclared files in frozen laundry release')
    scene, task, initial, rules, profile = [load_json(root/name) for name in
        ('scene.json', 'task.json', 'initial_state.json', 'rules.json', 'physics_profile.json')]
    if manifest.get('configuration_id') != task.get('robot_configuration_id'):
        raise ValueError('Manifest robot configuration differs from frozen task')
    if manifest.get('capabilities') != task.get('capabilities') or is_robot != (control_mode(task) == ROBOT_CONTROL):
        raise ValueError('Manifest and task robot capability/control modes disagree')
    if (initial_state(scene, task) != initial or rules != task['public_rules']
            or profile['isaacsim_options'] != scene['isaacsim'] or profile['material_calibrated'] is not False):
        raise ValueError('Frozen reset/rules/physics inputs disagree')
    if is_robot and (manifest.get('task_kind') != 'single_agent'
            or profile.get('robot_isaacsim_options') != ROBOT_PHYSICS_OPTIONS
            or profile.get('cloth_isaacsim_options') != scene['laundry_task_provenance']['cloth_physics_options']):
        raise ValueError('Frozen robot and cloth physical profiles must remain separately bound')
    if (manifest.get('physics_profile') != 'physics_profile.json'
            or manifest.get('physics_profile_sha256') != files['physics_profile.json']['sha256']):
        raise ValueError('Frozen physics profile reference differs')
    entries = manifest.get('tasks', [])
    if len(entries) != 1 or manifest.get('task_id') != task['id'] or entries[0].get('id') != task['id']:
        raise ValueError('Release task identity differs')
    for key in ('scene', 'task', 'initial_state'):
        if entries[0].get(key) != key+'.json' or entries[0].get(key+'_sha256') != files[key+'.json']['sha256']:
            raise ValueError('Frozen task file entry differs')
    if is_robot and entries[0].get('task_kind') != 'single_agent':
        raise ValueError('Robot laundry release must expose one single-Agent task')
    inventory = load_json(root/'asset_file_hashes.json')
    if manifest.get('asset_files_sha256') != {p: row['sha256'] for p, row in inventory.items()}:
        raise ValueError('Frozen shared inventory indexes differ')
    candidate = scene['laundry_task_provenance']['asset_candidate']
    if manifest.get('asset_catalog') != {'path': candidate['manifest_path'], 'sha256': candidate['manifest_sha256']}:
        raise ValueError('Towel manifest identity differs')
    if assets and inventory != asset_inventory(scene):
        raise ValueError('Laundry environment or towel assets changed')
    from bench.isaacsim.release import _frozen_evaluator_module
    _frozen_evaluator_module(root).SequenceEvaluator(task)
    if 'fabric_visual_derivation' in manifest:
        # Validates its whole base release, including any table-material derivation.
        from .fabric_visual_release import validate_derivation as validate_visual
        validate_visual(root, manifest, assets=assets)
    elif 'table_material_derivation' in manifest:
        from .table_material_release import validate_derivation
        validate_derivation(root, manifest, assets=assets)
    return manifest


def episode_inputs(root, scene, task, seed):
    """Canonical inputs preserve the explicitly frozen engineering/robot mode."""
    from bench.isaacsim.release import _frozen_evaluator_module
    module = _frozen_evaluator_module(Path(root).resolve())
    rules = importlib.import_module(module.__package__+'.rules')
    task = deepcopy(task)
    task['episode_seed'] = seed
    if control_mode(task) == ROBOT_CONTROL:
        if not hasattr(rules, 'ROBOT_CONTROL'):
            raise ValueError('Frozen engineering rules cannot be promoted to a robot episode')
        task['public_rules'] = (rules.task_public_rules(task) if 'robot_configuration_id' in task else
            rules.public_rules(task['id'], seed, rule_profile(task), control_mode=rules.ROBOT_CONTROL))
    else:
        task['public_rules'] = (rules.public_rules(task['id'], seed, rule_profile(task))
            if hasattr(rules, 'PRACTICAL_PROFILE') else rules.public_rules(task['id'], seed))
    task['evaluation']['public_rules_sha256'] = rules.digest(task['public_rules'])
    task = rules.bind_task_rules(task)
    initial = initial_state(scene, task)
    module.SequenceEvaluator(task)
    return scene, task, initial


def prepare_task_release(task_id, output, *, asset_manifest, physics_options=None,
                         rule_profile=None, robot_layout=None, configuration_id=None, stand_manifest=None,
                         public_material_contract=None, cloth_visual_profiles=None):
    output = Path(output).resolve()
    if task_id not in TASK_IDS or not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Choose a fresh laundry task release under shared datasets')
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    project = Path(__file__).resolve().parents[3]
    modules = tuple('scenes.workrooms.laundry_room.'+name for name in
        ('assets', 'composition', 'rules', 'evaluation', 'release', 'scene'))
    is_robot = robot_layout is not None or configuration_id is not None
    if is_robot:
        modules += ('scenes.workrooms.laundry_room.robot_layout', 'scenes.workrooms.laundry_room.native_observer',
                    'scenes.workrooms.mechanical_assembly.cooperative_robot_layout',
                    'scenes.workrooms.mechanical_assembly.robot_layout')
    if configuration_id is not None:
        modules += ('scenes.workrooms.laundry_room.robot_configurations',)
    if task_id in GARMENT_FOLDING_TASK_IDS:
        modules += ('scenes.workrooms.laundry_room.garment_folding_assets',)
    elif task_id in GARMENT_TASK_IDS:
        modules += ('scenes.workrooms.laundry_room.garments',)
    for name in modules:
        importlib.import_module(name)
    snapshot = verify_if_snapshot(project, required_modules=modules, entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Prepare laundry releases from a complete frozen project source snapshot')
    inputs = build_task_inputs(task_id, asset_manifest=asset_manifest, physics_options=physics_options,
                               rule_profile=rule_profile, robot_layout=robot_layout,
                               configuration_id=configuration_id, stand_manifest=stand_manifest,
                               public_material_contract=public_material_contract,
                               cloth_visual_profiles=cloth_visual_profiles)
    scene, task, initial = (inputs[key] for key in ('scene', 'task', 'initial_state'))
    from bench.protocol.task_referee import PublicContract
    PublicContract.model_validate(task['public_rules']['referee'])
    inventory = asset_inventory(scene)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for filename, value in [('scene.json', scene), ('task.json', task), ('initial_state.json', initial),
                ('rules.json', task['public_rules']), ('asset_file_hashes.json', inventory)]:
            dump(staging/filename, value)
        profile = {'backend': 'Isaac Sim 6.1 native PhysX GPU',
            'isaacsim_options': scene['isaacsim'], 'material_calibrated': False,
            'physical_thickness_m': scene['isaacsim']['cloth_thickness_m'], 'physical_thickness_response_accepted': False}
        if is_robot:
            profile.update(robot_isaacsim_options=deepcopy(ROBOT_PHYSICS_OPTIONS),
                           cloth_isaacsim_options=deepcopy(scene['laundry_task_provenance']['cloth_physics_options']))
        dump(staging/'physics_profile.json', profile)
        for name, path in evaluator_sources(robot=is_robot).items():
            (staging/'evaluator').mkdir(exist_ok=True)
            shutil.copy2(path, staging/'evaluator'/name)
            checked(staging/'evaluator'/name, digest(path))
        snapshot = verify_if_snapshot(project, required_modules=modules, entrypoint_file=Path(__file__))
        dump(staging/'source_provenance.json', {'source_snapshot': snapshot,
            'source_root': str(project), 'sources_sha256': {str(p): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
            'parent_environment': scene['workroom'], 'original_releases_modified': False,
            'shared_controller_unchanged': True, 'control_mode': control_mode(task)})
        files = {str(p.relative_to(staging)): {'sha256': digest(p)} for p in sorted(staging.rglob('*')) if p.is_file()}
        entry = {key: task[key] for key in ('id', 'title_zh', 'difficulty', 'cloth')}
        if is_robot:
            entry['task_kind'] = 'single_agent'
        for key in ('scene', 'task', 'initial_state'):
            entry.update({key: key+'.json', key+'_sha256': files[key+'.json']['sha256']})
        candidate = scene['laundry_task_provenance']['asset_candidate']
        manifest = {'schema': SCHEMA, 'schema_version': VERSION, 'task_id': task_id,
            'release_id': 'laundry_'+task_id+'_'+output.name, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_unvalidated', 'source_release': None, 'original_release_modified': False,
            'capabilities': deepcopy(task['capabilities']), 'tasks': [entry], 'files': files,
            'files_sha256': {p: row['sha256'] for p, row in files.items()},
            'asset_files_sha256': {p: row['sha256'] for p, row in inventory.items()},
            'physics_profile': 'physics_profile.json', 'physics_profile_sha256': files['physics_profile.json']['sha256'],
            'asset_catalog': {'path': candidate['manifest_path'], 'sha256': candidate['manifest_sha256']},
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'qualification_scope': task['qualification_scope']}
        if is_robot:
            manifest['task_kind'] = 'single_agent'
        if configuration_id is not None:
            manifest['configuration_id'] = configuration_id
        dump(staging/'release_manifest.json', manifest)
        validate_release(staging, manifest)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(task_id=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    if task_id is None:
        parser.add_argument('--task', choices=TASK_IDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--physics-options', type=Path)
    from .robot_layout import LAYOUT_ID
    parser.add_argument('--robot-layout', choices=(LAYOUT_ID,), help='Explicit single-Agent dual-arm candidate; omission preserves engineering inputs')
    from .robot_configurations import CONFIGURATION_IDS
    parser.add_argument('--configuration', choices=CONFIGURATION_IDS)
    parser.add_argument('--stand-manifest', type=Path)
    from .rules import ASSET_MATERIAL_CONTRACT
    parser.add_argument('--public-material-contract', choices=(ASSET_MATERIAL_CONTRACT,))
    args = parser.parse_args()
    options = load_json(args.physics_options) if args.physics_options else None
    print(json.dumps(prepare_task_release(task_id or args.task, args.output,
        asset_manifest=args.asset_manifest, physics_options=options, robot_layout=args.robot_layout,
        configuration_id=args.configuration, stand_manifest=args.stand_manifest,
        public_material_contract=args.public_material_contract), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
