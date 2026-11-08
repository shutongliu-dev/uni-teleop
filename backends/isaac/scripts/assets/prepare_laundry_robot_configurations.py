"""Prepare 15 independent laundry inputs from reviewed material/reset assets.

This writes an unqualified candidate index, never the production catalog.
Every option remains an ordinary joint-only release with its own installation.
"""
from pathlib import Path
import argparse
import json

from bench.isaacsim.release import checked, digest, dump
from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.laundry_room.rules import CURRENT_TASK_IDS, ASSET_MATERIAL_CONTRACT
from scenes.workrooms.laundry_room.robot_configurations import CONFIGURATION_IDS, installation
from scenes.workrooms.laundry_room.release import prepare_task_release
from scenes.workrooms.laundry_room.configuration_index import configuration_row


def prepare(asset_index, release_root, stand_root, version, output, *, public_material_contract=ASSET_MATERIAL_CONTRACT):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    project = Path(__file__).resolve().parents[2]
    snapshot = verify_if_snapshot(project,
        required_modules=('scripts.assets.prepare_laundry_robot_configurations',
                          'scenes.workrooms.laundry_room.configuration_index'), entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('Prepare candidates from a complete frozen source snapshot')
    asset_index, release_root, stand_root, output = map(lambda p: Path(p).resolve(),
        (asset_index, release_root, stand_root, output))
    if (not release_root.is_relative_to(OUTPUT_ROOT)
            or not output.is_relative_to(OUTPUT_ROOT) or output.exists()
            or not version or Path(version).name != version):
        raise ValueError('Use fresh versioned release/report paths under shared datasets')
    inputs = load_json(asset_index)
    if inputs.get('schema') != 'laundry_robot_configuration_assets.v1' or set(inputs.get('tasks', {})) != set(CURRENT_TASK_IDS):
        raise ValueError('Expected an explicit five-task material/reset asset index')
    for task, row in inputs['tasks'].items():
        if set(row) != set(CONFIGURATION_IDS):
            raise ValueError('Each task requires all three asset choices: '+task)
        for pin in row.values():
            checked(pin['path'], pin['sha256'])
    for identifier in CONFIGURATION_IDS:
        installation(identifier, stand_root/identifier/'manifest.json')
    targets = [release_root/task/identifier/version for task in CURRENT_TASK_IDS
               for identifier in CONFIGURATION_IDS]
    if any(path.exists() for path in targets):
        raise ValueError('All fifteen output releases must be fresh; preserve partial earlier attempts')
    report = {'schema': 'laundry_robot_configuration_candidates.v1', 'complete': False,
        'asset_index': {'path': str(asset_index), 'sha256': digest(asset_index)},
        'source_snapshot': snapshot, 'tasks': [], 'prepared_releases': [],
        'public_material_contract': public_material_contract,
        'native_qualified': False, 'robot_manipulation_qualified': False, 'physics_accepted': False}
    try:
        for task_id in CURRENT_TASK_IDS:
            paths = {}
            for identifier in CONFIGURATION_IDS:
                path = release_root/task_id/identifier/version
                prepare_task_release(task_id, path,
                    asset_manifest=inputs['tasks'][task_id][identifier]['path'],
                    configuration_id=identifier, stand_manifest=stand_root/identifier/'manifest.json',
                    public_material_contract=public_material_contract)
                paths[identifier] = path
                report['prepared_releases'].append({'task_id': task_id, 'configuration_id': identifier,
                    'path': str(path), 'manifest_sha256': digest(path/'release_manifest.json')})
                dump(output, report)
            report['tasks'].append(configuration_row(task_id, paths))
            for root in paths.values():
                for path in root.rglob('*'):
                    path.chmod(0o555 if path.is_dir() else 0o444)
                root.chmod(0o555)
            dump(output, report)
        report['complete'] = True
        dump(output, report)
        return report
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        dump(output, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-index', type=Path, required=True)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--stand-root', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.asset_index, args.release_root, args.stand_root, args.version, args.output)
    print(json.dumps({'complete': result['complete'], 'releases': len(result['prepared_releases'])}))


if __name__ == '__main__':
    main()
