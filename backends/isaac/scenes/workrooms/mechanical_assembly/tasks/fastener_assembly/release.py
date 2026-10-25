"""Freeze an independently versioned fastener task for the canonical worker."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import digest, dump, validate
from .task import (TASK_ID, STAND_ROOT, THREADED_ROOT, LOOSE_ASSET_ROOT,
                   build_task_inputs)
from bench.paths import DATA_ROOT, load_json

DATASET_ROOT = Path(f'{DATA_ROOT}/datasets')
RELEASE_SCHEMA = 'isaacsim-mechanical-fastener-release-v1'


def _asset_inventory(scene):
    from bench.simulation.isaacsim_backend.presentation import presentation_dependencies
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from bench.simulation.isaacsim_backend.workroom import verify_reference
    from scenes.workrooms.mechanical_assembly.robot_layout import robot_asset_reference
    from scenes.workrooms.mechanical_assembly.threaded_reset import TRACE, TRACE_SHA256

    verify_reference(scene['workroom'])
    reference = scene['workroom']
    expected = {reference['usd_path']: reference['usd_sha256'],
                reference['dependency_manifest_path']: reference['dependency_manifest_sha256'],
                str(TRACE): TRACE_SHA256}
    description = load_json(Path(reference['dependency_manifest_path']))
    expected.update({path: row['sha256'] for path, row in description['dependencies'].items()})
    expected.update(presentation_dependencies({'presentation': scene.get('workroom_presentation')}))
    robot = robot_asset_reference(scene['robot'])
    expected.update({robot['root_usd']: robot['root_usd_sha256'],
        robot['source_manifest']: robot['source_manifest_sha256'],
        robot['import_manifest']: robot['import_manifest_sha256'], **robot['derived_files']})
    for spec in scene['objects'] + scene['fixtures']:
        expected.update(presentation_dependencies(spec))
        asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        expected[asset['manifest_path']] = asset['manifest_sha256']
        expected.update(asset.get('source_dependencies_sha256', {}))
        for filename in ('SOURCE.json', 'LICENSE.txt'):
            path = Path(spec['asset_root'])/filename
            if path.is_file():
                expected[str(path.resolve())] = digest(path)
        for row in [asset, *asset['visual_parts'], *asset['collision_parts']]:
            if 'mesh_path' in row:
                expected[row['mesh_path']] = row['sha256']
    for path, expected_hash in expected.items():
        if digest(path) != expected_hash:
            raise ValueError('Original task dependency changed: '+path)
    return {str(Path(path).resolve()): {'sha256': value} for path, value in expected.items()}


def prepare_task_release(output, *, stand_root=STAND_ROOT, threaded_root=THREADED_ROOT,
                         target_turn_degrees=30., tool_clocking_degrees=0.,
                         reset_variant='preinserted', guide_root=LOOSE_ASSET_ROOT):
    """Write a new immutable preparation; never publish native acceptance."""
    output = Path(output).resolve()
    if not output.is_relative_to(DATASET_ROOT):
        raise ValueError('Task releases belong under shared datasets')
    if output.exists():
        raise FileExistsError('Choose a new release directory: '+str(output))
    values = build_task_inputs(stand_root=stand_root, threaded_root=threaded_root,
                               target_turn_degrees=target_turn_degrees,
                               tool_clocking_degrees=tool_clocking_degrees,
                               reset_variant=reset_variant, guide_root=guide_root)
    scene, task, initial = (values[key] for key in ('scene', 'task', 'initial_state'))
    assets = _asset_inventory(scene)
    catalog = Path(threaded_root).resolve()/'manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for name, value in [('scene.json', scene), ('task.json', task), ('initial_state.json', initial),
                            ('rules.json', task['public_rules']), ('asset_file_hashes.json', assets)]:
            dump(staging/name, value)
        profile = {'backend': 'Isaac Sim 6.1 native PhysX GPU',
            'isaacsim_options': deepcopy(scene['isaacsim']), 'material_calibrated': False,
            'source': 'Unchanged selected mechanical v3 diagnostic profile and original per-object options.',
            'current_task_native_accepted': False}
        dump(staging/'physics_profile.json', profile)
        evaluator = staging/'evaluator'
        evaluator.mkdir()
        for source_name, destination in [('evaluator.py', 'sequence_evaluator.py'), ('rules.py', 'rules.py')]:
            source = Path(__file__).with_name(source_name)
            shutil.copy2(source, evaluator/destination)
            if digest(source) != digest(evaluator/destination):
                raise ValueError('Evaluator changed during freezing')
        referee_source = Path(__file__).parents[2]/'referee.py'
        shutil.copy2(referee_source, evaluator/'referee.py')
        if digest(referee_source) != digest(evaluator/'referee.py'):
            raise ValueError('Referee changed during freezing')
        dump(staging/'source_provenance.json', {
            'sources_sha256': {str(p): digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'parent_environment': deepcopy(scene['workroom']),
            'original_assets_modified': False, 'original_releases_modified': False,
            'reset_simplifications': deepcopy(task['public_rules']['reset_simplifications']),
            'task_reset_conditions': deepcopy(task.get('reset_conditions')),
            'physics_accepted': False, 'task_policy_success_claimed': False})
        files = {str(path.relative_to(staging)): {'sha256': digest(path)}
                 for path in sorted(staging.rglob('*')) if path.is_file()}
        entry = {key: task[key] for key in ('id', 'title_zh', 'difficulty', 'cloth')}
        for kind in ('scene', 'task', 'initial_state'):
            entry[kind] = kind+'.json'
            entry[kind+'_sha256'] = files[kind+'.json']['sha256']
        manifest = {'schema_version': RELEASE_SCHEMA, 'schema': 'mechanical_fastener_task_release.v1',
            'task_id': TASK_ID, 'release_id': 'mechanical_fastener_'+output.name,
            'reset_variant': reset_variant,
            'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_unvalidated', 'source_release': None, 'original_release_modified': False,
            'tasks': [entry], 'files': files,
            'files_sha256': {name: row['sha256'] for name, row in files.items()},
            'asset_files_sha256': {name: row['sha256'] for name, row in assets.items()},
            'physics_profile': 'physics_profile.json', 'physics_profile_sha256': digest(staging/'physics_profile.json'),
            'asset_catalog': {'path': str(catalog), 'sha256': digest(catalog)},
            'task_asset_catalogs': ([{'path': str(Path(guide_root).resolve()/'manifest.json'),
                                     'sha256': digest(Path(guide_root).resolve()/'manifest.json')}]
                                    if reset_variant == 'loose_bolt' else []),
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'qualification_scope': 'Canonical-worker-loadable candidate task; source and CPU checks are not native manipulation acceptance.'}
        dump(staging/'release_manifest.json', manifest)
        validate(staging)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
