"""Prepare explicitly selected ordinary tasks from their pinned original parents."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
from pathlib import Path

from bench.isaacsim.release import validate
from bench.paths import DATA_ROOT, OUTPUT_ROOT
from scenes.workrooms.catalog import task_definition
from scenes.workrooms.initial_randomization import TASK_FAMILIES, PLACEMENT_TASK_FAMILIES, SEEDS, configurations
from scenes.workrooms.initial_randomization.catalog import publication_row
from scenes.workrooms.initial_randomization.common import checked_file, file_hash, read_json, write_json
from scenes.workrooms.initial_randomization.release import adapter, prepare_release, _parent_configuration
from scripts.maintenance.freeze_project_sources import verify_runtime_imports


def original_task_definition(family, task_id):
    """Publish against original installations even after the catalog is seeded."""
    # Validate the public identity before constructing its module path. The
    # live catalog may already point at an earlier randomized publication.
    task_definition(family, task_id)
    from scenes.workrooms.robot_configurations import attach_robot_configurations
    module = importlib.import_module(f'scenes.workrooms.{family}.tasks.{task_id}.definition')
    return attach_robot_configurations(deepcopy(module.TASK))


def checked_parents(request):
    if (set(request) != {'schema', 'parents'} or request['schema'] != 'workroom_randomization_preparation_request.v1'
            or not isinstance(request['parents'], dict) or not request['parents']
            or not set(request['parents']) <= set(TASK_FAMILIES)):
        raise ValueError('Explicit pinned parents for selected ordinary tasks required')
    for task_id, row in request['parents'].items():
        if row['family'] != TASK_FAMILIES[task_id] or set(row['configurations']) != set(configurations(task_id)):
            raise ValueError('Parent task/family/installation set differs')
        for configuration_id, ref in row['configurations'].items():
            root = Path(ref['path']).resolve()
            checked_file(root/'release_manifest.json', ref['manifest_sha256'])
            manifest = validate(root)
            if (manifest['task_id'] != task_id
                    or _parent_configuration(read_json(root/'scene.json'), task_id) != configuration_id
                    or manifest.get('schema') == 'workroom_randomized_release.v1'):
                raise ValueError('Original parent task/installation differs')
    return request['parents']


def prepare(request_path, output, asset_root, task_ids=None, *, profile=None, attempts=None):
    source_root = Path(__file__).resolve().parents[2]
    source = verify_runtime_imports(source_root, required_modules=(
        'scripts.assets.prepare_workroom_randomization', 'scenes.workrooms.initial_randomization.release'),
        entrypoint_file=Path(__file__))
    output, asset_root = Path(output).resolve(), Path(asset_root).resolve()
    if not output.is_relative_to(OUTPUT_ROOT) or not asset_root.is_relative_to(f'{DATA_ROOT}/assets'):
        raise ValueError('Use shared datasets for inputs/evidence and assets for reusable objects')
    request = read_json(request_path); parents = checked_parents(request)
    if profile not in (None, 'parameterized_physics_v1'):
        raise ValueError('Unknown explicit parameterization profile')
    attempts = {} if attempts is None else deepcopy(attempts)
    if (not isinstance(attempts, dict) or (attempts and profile is None)
            or not set(attempts) <= {f'{task}:{seed}' for task in parents for seed in SEEDS}
            or any(type(value) is not int or value < 0 for value in attempts.values())):
        raise ValueError('Layout attempts require an explicit parameterized profile')
    task_ids = list(parents) if task_ids is None else task_ids
    if not task_ids or len(set(task_ids)) != len(task_ids) or not set(task_ids) <= parents.keys():
        raise ValueError('Select explicit unique tasks from the pinned preparation request')
    output.mkdir(parents=True, exist_ok=False)
    asset_root.mkdir(parents=True, exist_ok=True)
    record = {'schema': 'workroom_randomization_preparation.v1', 'complete': False,
              'request_sha256': file_hash(request_path), 'source_snapshot': source,
              'tasks': {}, 'model_calls': 0, 'native_steps': 0}
    if profile is not None:
        record.update(parameterization_profile=profile, layout_attempts=attempts)
    write_json(output/'request.json', request)
    write_json(output/'preparation.json', record)
    validation = {'schema': 'workroom_randomized_validation_request.v1', 'duration_s': .8, 'cases': []}
    rows = {}
    for task_id in task_ids:
        family = TASK_FAMILIES[task_id]; row = parents[task_id]
        assets = {}
        for seed in SEEDS:
            directory = (asset_root/family/profile/output.name/task_id
                         if profile is not None else asset_root/family/task_id)
            if profile is None and task_id in PLACEMENT_TASK_FAMILIES:
                # New publications get a fresh immutable asset namespace;
                # historical ten-task directory conventions remain unchanged.
                directory /= output.name
            directory /= f'seed_{seed:02d}'
            default = configurations(task_id)[0]
            kwargs = ({} if profile is None else
                      {'profile': profile, 'attempt': attempts.get(f'{task_id}:{seed}', 0)})
            path = adapter(task_id).prepare_asset(Path(row['configurations'][default]['path']), seed, directory, **kwargs)
            assets[seed] = str(Path(path).resolve())
            print(f'Authored {task_id} seed {seed}', flush=True)
        releases = {}
        for configuration_id, parent in row['configurations'].items():
            directory = output/'releases'/task_id/configuration_id
            prepare_release(parent['path'], directory, configuration_id=configuration_id, assets_by_seed=assets)
            releases[configuration_id] = str(directory)
            validation['cases'].extend({'task_id': task_id, 'configuration_id': configuration_id,
                'seed': seed, 'release': str(directory)} for seed in SEEDS)
        rows[task_id] = publication_row(original_task_definition(family, task_id), releases)
        record['tasks'][task_id] = {'assets': assets, 'releases': releases}
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
    parser.add_argument('--tasks', nargs='+', choices=tuple(TASK_FAMILIES))
    parser.add_argument('--profile', choices=('parameterized_physics_v1',))
    parser.add_argument('--attempts', type=Path, help='Explicit task_id:seed to layout-attempt JSON map')
    args = parser.parse_args()
    result = prepare(args.request, args.output, args.asset_root, args.tasks,
                     profile=args.profile, attempts=read_json(args.attempts) if args.attempts else None)
    print(f"Prepared {result['cases']} physical cases")


if __name__ == '__main__':
    main()
