"""Author three living tasks' frozen initial cases from pinned original releases.

Run from a complete source snapshot. This creates inputs only; it never publishes
catalog defaults, runs a model, starts a simulator or claims physical validation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from bench.isaacsim.release import validate
from bench.paths import DATA_ROOT, OUTPUT_ROOT
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.living_room.randomization import TASK_IDS, CONFIGURATIONS, SEEDS
from scenes.workrooms.living_room.randomization.cases import plan_case
from scenes.workrooms.living_room.randomization.catalog import publication_row
from scenes.workrooms.living_room.randomization.common import (
    canonical_hash, checked_file, file_hash, read_json, write_json,
)
from scenes.workrooms.living_room.randomization.release import prepare_release
from scenes.workrooms.living_room.seed_variants import author_variant
from scripts.maintenance.freeze_project_sources import verify_runtime_imports


def checked_parents(request):
    if (set(request) != {'schema', 'parents'} or
            request['schema'] != 'living_randomization_preparation_request.v1' or
            set(request['parents']) != set(TASK_IDS)):
        raise ValueError('Explicit original releases for all three ordinary tasks required')
    result = {}
    for task_id in TASK_IDS:
        configs = request['parents'][task_id]
        if set(configs) != set(CONFIGURATIONS):
            raise ValueError('Each task requires three original installations')
        result[task_id] = {}
        for configuration_id in CONFIGURATIONS:
            reference = configs[configuration_id]
            if set(reference) != {'path', 'manifest_sha256'} or not Path(reference['path']).is_absolute():
                raise ValueError('Expected an absolute release and exact manifest digest')
            root = Path(reference['path']).resolve()
            checked_file(root/'release_manifest.json', reference['manifest_sha256'])
            manifest = validate(root)
            if (manifest['task_id'] != task_id or
                    manifest.get('configuration_id', 'single_arm') != configuration_id):
                raise ValueError('Original task or robot installation differs')
            if manifest['schema'] == 'living_room_randomized_release.v1':
                raise ValueError('Derive from original inputs, not a previous randomized release')
            scene, task = read_json(root/'scene.json'), read_json(root/'task.json')
            candidate = scene['living_room_provenance']['asset_candidate']
            parent_manifest = read_json(candidate['manifest_path'])
            asset = {key: candidate[key] for key in ('manifest_path', 'manifest_sha256',
                                                     'initial_usd', 'initial_usd_sha256')}
            asset.update(metadata=str(Path(asset['manifest_path']).parent/'metadata.json'),
                         metadata_sha256=parent_manifest['metadata_sha256'])
            if (scene['native_task_bundle']['usd_sha256'] != asset['initial_usd_sha256'] or
                    task['asset_metadata'] != read_json(asset['metadata'])):
                raise ValueError('Original scene, asset and evaluator metadata disagree')
            result[task_id][configuration_id] = {'root': root, 'asset': asset,
                                                'metadata': task['asset_metadata']}
        original = result[task_id]['single_arm']
        if any(value['asset'] != original['asset'] or value['metadata'] != original['metadata']
               for value in result[task_id].values()):
            raise ValueError('All installations must share the same task asset and initial geometry')
    return result


def prepare(request_path, output, asset_root, *, profile=None, attempts=None, task_ids=None):
    source_root = Path(__file__).resolve().parents[2]
    source = verify_runtime_imports(source_root, required_modules=('scripts.assets.prepare_living_randomization',
        'scenes.workrooms.living_room.randomization.release',
        'scenes.workrooms.living_room.seed_variants'), entrypoint_file=Path(__file__))
    output, asset_root = Path(output).resolve(), Path(asset_root).resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or not asset_root.is_relative_to(f'{DATA_ROOT}/assets'):
        raise ValueError('Use shared datasets for releases and shared assets for reusable geometry')
    request = read_json(request_path)
    if profile not in (None, 'parameterized_physics_v1'):
        raise ValueError('Unknown explicit parameterization profile')
    attempts = {} if attempts is None else deepcopy(attempts)
    if (not isinstance(attempts, dict) or (attempts and profile is None)
            or not set(attempts) <= {f'{task}:{seed}' for task in TASK_IDS for seed in SEEDS}
            or any(type(value) is not int or value < 0 for value in attempts.values())):
        raise ValueError('Layout attempts require an explicit parameterized profile')
    parents = checked_parents(request)
    task_ids = list(TASK_IDS) if task_ids is None else list(task_ids)
    if not task_ids or len(set(task_ids)) != len(task_ids) or not set(task_ids) <= set(TASK_IDS):
        raise ValueError('Select unique ordinary living tasks from the pinned request')
    output.mkdir(parents=True, exist_ok=False)
    asset_root.mkdir(parents=True, exist_ok=False)
    write_json(output/'request.json', request)
    record = {'schema': 'living_randomization_preparation.v1', 'complete': False,
        'request_sha256': canonical_hash(request), 'source_snapshot': source,
        'asset_root': str(asset_root), 'tasks': {}, 'model_calls': 0, 'native_steps': 0}
    if profile is not None:
        record.update(parameterization_profile=profile, layout_attempts=attempts)
    write_json(output/'preparation.json', record)
    validation = {'schema': 'living_randomized_validation_request.v1', 'duration_s': .8, 'cases': []}
    rows = {}
    for task_id in task_ids:
        original = parents[task_id]['single_arm']
        assets = {}
        for seed in SEEDS:
            directory = asset_root/task_id/f'seed_{seed:02d}'
            kwargs = ({} if profile is None else
                      {'generator_version': profile, 'configuration_id': 'single_arm',
                       'attempt': attempts.get(f'{task_id}:{seed}', 0)})
            plan = plan_case(task_id, original['metadata'], seed, **kwargs)
            author_variant(original['asset'], plan, directory)
            assets[seed] = str(directory/'manifest.json')
            print(f'Authored {task_id} seed {seed}', flush=True)
        releases = {}
        for configuration_id in CONFIGURATIONS:
            directory = output/'releases'/task_id/configuration_id
            prepare_release(parents[task_id][configuration_id]['root'], directory,
                            configuration_id=configuration_id, assets_by_seed=assets)
            releases[configuration_id] = str(directory)
            validation['cases'].extend({'task_id': task_id, 'configuration_id': configuration_id,
                'seed': seed, 'release': str(directory)} for seed in SEEDS)
        rows[task_id] = publication_row(task_definition('living_room', task_id, robot_profile='original'), releases)
        record['tasks'][task_id] = {'assets': assets, 'releases': deepcopy(releases)}
        write_json(output/'preparation.json', record)
        write_json(output/'catalog_rows.json', rows)
        write_json(output/'validation_request.json', validation)
    record.update(complete=True, cases=len(validation['cases']),
        catalog_rows_sha256=file_hash(output/'catalog_rows.json'),
        validation_request_sha256=file_hash(output/'validation_request.json'))
    write_json(output/'preparation.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--profile', choices=('parameterized_physics_v1',))
    parser.add_argument('--tasks', nargs='+', choices=TASK_IDS)
    parser.add_argument('--attempts', type=Path, help='Explicit task_id:seed to layout-attempt JSON map')
    args = parser.parse_args()
    result = prepare(args.request, args.output, args.asset_root, profile=args.profile,
                     attempts=read_json(args.attempts) if args.attempts else None, task_ids=args.tasks)
    print(f"Prepared {result['cases']} cases in {args.output}")


if __name__ == '__main__':
    main()
