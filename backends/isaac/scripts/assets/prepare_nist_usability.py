"""Freeze reviewed NIST usability assets and their original desktop dressing.

This produces candidates; catalog pins and evidence are published separately.
The four task IDs, fixed room, shared robot/control and numerical referee limits
remain unchanged. Historical assets and releases are never rewritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.isaacsim.release import checked, digest, dump, validate
from bench.paths import DATA_ROOT, load_json, OUTPUT_ROOT
from scenes.workrooms.mechanical_assembly.desktop_release import prepare_release
from scenes.workrooms.mechanical_assembly.nist_composition import TASK_IDS
from scenes.workrooms.mechanical_assembly.nist_release import prepare_task_release
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


def prepare(asset_root, output, report, tasks=None, asset_index=None):
    project = Path(__file__).resolve().parents[2]
    if verify_if_snapshot(project,
            required_modules=('scenes.workrooms.mechanical_assembly.nist_release',),
            entrypoint_file=Path(__file__)) is None:
        raise ValueError('Prepare from complete frozen source')
    asset_root, output, report = (Path(p).resolve() for p in (asset_root, output, report))
    if not asset_root.is_relative_to(f'{DATA_ROOT}/assets') or any(
            not p.is_relative_to(OUTPUT_ROOT) for p in (output, report)):
        raise ValueError('Use shared assets for inputs and shared datasets for releases/evidence')
    if output.exists() or report.exists():
        raise FileExistsError('Use fresh versioned paths')
    tasks = tuple(tasks) if tasks is not None else TASK_IDS
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(TASK_IDS):
        raise ValueError('Select distinct NIST tasks')
    assets = {}
    if asset_index is not None:
        candidates = load_json(Path(asset_index))
        if candidates.get('complete') is not True:
            raise ValueError('Asset candidate index must be complete')
        for row in candidates['tasks']:
            if row['task_id'] in assets:
                raise ValueError('Duplicate task in asset candidate index')
            assets[row['task_id']] = row['asset_manifest']
        if set(assets) != set(tasks):
            raise ValueError('Asset candidate index must match the selected tasks')
    report.mkdir(parents=True)
    index = {'schema': 'mechanical_nist_usability_candidates.v1', 'complete': False,
        'source_snapshot': {'path': str(project/'source_snapshot_manifest.json'),
                            'sha256': digest(project/'source_snapshot_manifest.json')},
        'tasks': [], 'native_qualified': False, 'manipulation_qualified': False}
    try:
        for task_id in tasks:
            manifest_path = asset_root/task_id/'usability_v1'/'manifest.json'
            if assets:
                reference = assets[task_id]
                manifest_path = Path(reference['path']).resolve()
                if not manifest_path.is_relative_to(f'{DATA_ROOT}/assets'):
                    raise ValueError('Candidate asset must be in shared assets')
                checked(manifest_path, reference['sha256'])
            metadata = load_json(manifest_path.parent/'metadata.json')
            if metadata.get('usability_profile') != 'nist_manipulation_scale_v1':
                raise ValueError('An explicit reviewed usability asset is required')
            base = output/task_id/'base_v1'
            desktop = output/task_id/'desktop_v1'
            prepare_task_release(task_id, base, asset_manifest=manifest_path)
            prepare_release(base, desktop)
            validate(desktop)
            index['tasks'].append({'family': 'mechanical_assembly', 'task_id': task_id,
                'asset_manifest': {'path': str(manifest_path), 'sha256': digest(manifest_path)},
                'base_release': {'path': str(base), 'manifest_sha256': digest(base/'release_manifest.json')},
                'source_release': {'path': str(desktop), 'manifest_sha256': digest(desktop/'release_manifest.json')}})
            dump(report/'candidate_index.json', index)
            print(json.dumps({'prepared': task_id, 'release': str(desktop)}), flush=True)
        index['complete'] = True
        dump(report/'candidate_index.json', index)
    except BaseException as exc:
        index['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        dump(report/'candidate_index.json', index)
        raise
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True,
                        help='Contains <task_id>/usability_v1/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--task', action='append', choices=TASK_IDS)
    parser.add_argument('--asset-index', type=Path, help='Optional exact versioned asset manifest identities')
    args = parser.parse_args()
    prepare(args.asset_root, args.output, args.report, args.task, args.asset_index)


if __name__ == '__main__':
    main()
