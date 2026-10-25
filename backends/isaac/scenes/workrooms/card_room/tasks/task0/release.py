"""Freeze a new single-arm placement release without changing existing games."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

from bench.isaacsim.release import digest, dump, validate
from .assets import pinned_usd_dependency
from .task import ASSET_ROOT, TASK_ID, build_task_inputs
from bench.paths import DATA_ROOT, load_json

DATASET_ROOT = Path(f'{DATA_ROOT}/datasets')
RELEASE_SCHEMA = 'isaacsim-card-room-placement-release-v1'
DEFAULT_RELEASE = DATASET_ROOT/'fr3_sharpa_bench/isaacsim/releases/card_room/task0/v3'


def _asset_inventory(scene):
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from bench.simulation.isaacsim_backend.workroom import verify_reference

    reference = scene['workroom']
    expected = dict(verify_reference(reference))
    expected.update({reference['usd_path']: reference['usd_sha256'],
                     reference['dependency_manifest_path']: reference['dependency_manifest_sha256']})
    description = load_json(Path(reference['dependency_manifest_path']))
    expected.update({path: row['sha256'] for path, row in description['dependencies'].items()})
    robot = scene['robot']
    assembly = Path(robot['assets_root'])/'robots/assemblies/fr3_sharpa'/robot['assembly_version']
    imported_path, source_path = assembly/'USD_IMPORT.json', assembly/'SOURCE.json'
    imported = load_json(imported_path)
    source = load_json(source_path)
    expected.update({str(imported_path): digest(imported_path), str(source_path): digest(source_path),
                     imported['usd_path']: imported['usd_sha256'], **imported['derived_files']})
    expected.update({source['urdf_path']: source['urdf_sha256'], **source['sources'],
                     **source['mesh_inputs'], **source['license_sha256']})
    for spec in scene['objects'] + scene['fixtures']:
        asset = load_task_asset(spec['asset_id'], spec['asset_root'], verify_hashes=True)
        expected[asset['manifest_path']] = asset['manifest_sha256']
        expected.update(asset.get('source_dependencies_sha256', {}))
        if asset.get('backend') == 'usd_reference':
            expected.update(pinned_usd_dependency(asset))
        for row in [asset, *asset['visual_parts'], *asset['collision_parts']]:
            if 'mesh_path' in row:
                expected[row['mesh_path']] = row['sha256']
    for path, expected_hash in expected.items():
        if digest(path) != expected_hash:
            raise ValueError('Pinned task dependency changed: '+str(path))
    return {str(Path(path).resolve()): {'sha256': value} for path, value in expected.items()}


def prepare_task_release(output=DEFAULT_RELEASE, *, asset_root=ASSET_ROOT, token_collision_asset=None,
                         board_collision_asset=None):
    """Write an immutable seed-zero template; episode seeds only choose targets."""
    output = Path(output).resolve()
    if not output.is_relative_to(DATASET_ROOT):
        raise ValueError('Task releases belong under shared datasets')
    if output.exists():
        raise FileExistsError('Choose a new release directory: '+str(output))
    values = build_task_inputs(asset_root=asset_root, seed=0, token_collision_asset=token_collision_asset,
                               board_collision_asset=board_collision_asset)
    scene, task, initial = (values[key] for key in ('scene', 'task', 'initial_state'))
    assets = _asset_inventory(scene)
    catalog = Path(asset_root).resolve()/'manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for name, value in [('scene.json', scene), ('task.json', task), ('initial_state.json', initial),
                            ('rules.json', task['public_rules']), ('asset_file_hashes.json', assets)]:
            dump(staging/name, value)
        dump(staging/'physics_profile.json', {
            'backend': 'Isaac Sim 6.1 native PhysX GPU', 'isaacsim_options': deepcopy(scene['isaacsim']),
            'material_calibrated': False, 'current_task_native_accepted': False,
            'source': 'Pinned single-arm v12 rigid-pad profile with explicit workroom v4 runtime options.'})
        evaluator = staging/'evaluator'
        evaluator.mkdir()
        for source_name, destination in [('evaluator.py', 'sequence_evaluator.py'), ('rules.py', 'rules.py')]:
            source = Path(__file__).with_name(source_name)
            shutil.copy2(source, evaluator/destination)
            if digest(source) != digest(evaluator/destination):
                raise ValueError('Evaluator changed during freezing')
        dump(staging/'source_provenance.json', {
            'sources_sha256': {str(p): digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'parent_environment': deepcopy(scene['workroom']),
            'composition': deepcopy(scene['card_room_placement_provenance']),
            'original_assets_modified': False, 'original_releases_modified': False,
            'physics_accepted': False, 'task_policy_success_claimed': False})
        files = {str(path.relative_to(staging)): {'sha256': digest(path)}
                 for path in sorted(staging.rglob('*')) if path.is_file()}
        entry = {key: task[key] for key in ('id', 'title_zh', 'difficulty', 'cloth')}
        for kind in ('scene', 'task', 'initial_state'):
            entry[kind] = kind+'.json'
            entry[kind+'_sha256'] = files[kind+'.json']['sha256']
        manifest = {'schema_version': RELEASE_SCHEMA, 'schema': 'card_room_placement_task_release.v1',
            'task_id': TASK_ID, 'release_id': 'card_room_task0_'+output.name,
            'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'prepared_unvalidated', 'source_release': None, 'original_release_modified': False,
            'tasks': [entry], 'files': files,
            'files_sha256': {name: row['sha256'] for name, row in files.items()},
            'asset_files_sha256': {name: row['sha256'] for name, row in assets.items()},
            'physics_profile': 'physics_profile.json', 'physics_profile_sha256': digest(staging/'physics_profile.json'),
            'asset_catalog': {'path': str(catalog), 'sha256': digest(catalog)},
            'physics_accepted': False, 'task_policy_success_claimed': False,
            'qualification_scope': 'Canonical single-arm task candidate; native initialization and manipulation are separate evidence.'}
        dump(staging/'release_manifest.json', manifest)
        validate(staging)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--asset-root', type=Path, default=ASSET_ROOT)
    parser.add_argument('--token-asset-root', type=Path)
    parser.add_argument('--token-asset-id')
    parser.add_argument('--token-manifest-sha256')
    parser.add_argument('--board-asset-root', type=Path)
    parser.add_argument('--board-asset-id')
    parser.add_argument('--board-manifest-sha256')
    args = parser.parse_args()
    selection = None
    if any((args.token_asset_root, args.token_asset_id, args.token_manifest_sha256)):
        if not all((args.token_asset_root, args.token_asset_id, args.token_manifest_sha256)):
            parser.error('Token collision candidate requires root, id and manifest SHA256 together')
        selection = {'asset_root': str(args.token_asset_root.resolve()), 'asset_id': args.token_asset_id,
                     'manifest_sha256': args.token_manifest_sha256}
    board_selection = None
    if any((args.board_asset_root, args.board_asset_id, args.board_manifest_sha256)):
        if not all((args.board_asset_root, args.board_asset_id, args.board_manifest_sha256)):
            parser.error('Board collision candidate requires root, id and manifest SHA256 together')
        board_selection = {'asset_root': str(args.board_asset_root.resolve()), 'asset_id': args.board_asset_id,
                           'manifest_sha256': args.board_manifest_sha256}
    manifest = prepare_task_release(args.output, asset_root=args.asset_root, token_collision_asset=selection,
                                    board_collision_asset=board_selection)
    print(json.dumps({'output': str(args.output), 'status': manifest['status'],
                      'manifest_sha256': digest(args.output/'release_manifest.json')}))


if __name__ == '__main__':
    main()
