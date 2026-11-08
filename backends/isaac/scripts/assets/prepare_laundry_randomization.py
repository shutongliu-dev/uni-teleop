"""Prepare finite laundry assets/releases from explicit immutable parents, no native work."""
import argparse
from pathlib import Path

from scenes.workrooms.laundry_room.randomization import (CONFIGURATION_IDS, GENERATOR_VERSION,
                                                        SEEDS, TASK_IDS, checked_generator_version)
from scenes.workrooms.laundry_room.randomization import common
from scenes.workrooms.laundry_room.randomization.common import checked_file, file_hash, read_json, write_json
from scenes.workrooms.laundry_room.randomization.appearance import prepare_profile
from scenes.workrooms.laundry_room.randomization.author import prepare_asset, parent_visuals
from scenes.workrooms.laundry_room.randomization.release import prepare_release


def prepare(request, *, assets_output, releases_output, output):
    assets_output, releases_output, output = [Path(p).resolve() for p in (assets_output, releases_output, output)]
    if (request.get('schema') != 'laundry_seed_preparation_request.v1'
            or set(request) != {'schema', 'parents'} or not request['parents']
            or assets_output.exists() or releases_output.exists() or output.exists()
            or not assets_output.is_relative_to(common.ASSET_ROOT)
            or not releases_output.is_relative_to(common.RELEASE_ROOT)):
        raise ValueError('Explicit parent request and fresh shared preparation destinations required')
    source_provenance = common.source_provenance()
    seen, parents = set(), []
    for row in request['parents']:
        required = {'task_id', 'configuration_id', 'release', 'release_manifest_sha256'}
        if (not required <= set(row)
                or set(row)-required-{'generator_version', 'visual_profiles_by_seed', 'attempts_by_seed'}):
            raise ValueError('Preparation requires exact parent task/configuration/release pins')
        identity = (row['task_id'], row['configuration_id'])
        if identity in seen or identity[0] not in TASK_IDS or identity[1] not in CONFIGURATION_IDS:
            raise ValueError('Unknown or repeated parent installation')
        seen.add(identity)
        checked_file(Path(row['release'])/'release_manifest.json', row['release_manifest_sha256'])
        values = common.parent_inputs(row['release'])
        if (values[0]['task_id'], values[0]['configuration_id']) != identity:
            raise ValueError('Request differs from the actual original task/installation')
        checked_generator_version(*identity, row.get('generator_version', GENERATOR_VERSION))
        attempts = row.get('attempts_by_seed', {})
        if (not isinstance(attempts, dict) or not set(attempts) <= {str(s) for s in SEEDS}
                or any(type(value) is not int or value < 0 for value in attempts.values())
                or (attempts and row.get('generator_version') != 'parameterized_physics_v1')):
            raise ValueError('Layout attempts require the explicit parameterized profile')
        if ('visual_profiles_by_seed' in row
                and (not isinstance(row['visual_profiles_by_seed'], dict)
                     or set(row['visual_profiles_by_seed']) != {str(s) for s in SEEDS if s})):
            raise ValueError('Reused appearance must explicitly cover nonzero seeds 1 through 9')
        parents.append((row, values))
    output.mkdir(parents=True)
    write_json(output/'request.json', request)
    report = {'schema': 'laundry_seed_preparation.v1', 'complete': False, 'cases': [],
              'releases': [], 'native_started': False, 'model_calls': 0,
              'source_provenance': source_provenance,
              'catalog_selection_changed': False, 'scope': 'Immutable input preparation only.'}
    cache = {}
    for row, values in parents:
        task_id, configuration = row['task_id'], row['configuration_id']
        old_profiles = parent_visuals(values[1]); seed_assets = {}
        for seed in SEEDS:
            parameterized = row.get('generator_version') == 'parameterized_physics_v1'
            attempt = row.get('attempts_by_seed', {}).get(str(seed), 0)
            parameters = None
            if parameterized:
                from scenes.workrooms.laundry_room.randomization.cases import case_plan
                parameters = case_plan(task_id, configuration, seed, values[2]['asset_metadata'],
                    generator_version='parameterized_physics_v1', attempt=attempt)['parameters']
            profiles = None
            if seed and 'visual_profiles_by_seed' in row:
                profiles = row['visual_profiles_by_seed'][str(seed)]
            elif seed:
                profiles = {}
                for cloth_id, parent_pin in old_profiles.items():
                    key = ((parent_pin['sha256'], task_id, seed, 'parameterized_physics_v1')
                           if parameterized else (parent_pin['sha256'], seed))
                    if key not in cache:
                        destination = assets_output/'appearance'/parent_pin['sha256']
                        if parameterized:
                            destination /= task_id
                        destination /= f'seed_{seed:02d}'
                        kwargs = {'parameters': parameters} if parameterized else {}
                        cache[key] = prepare_profile(parent_pin, seed, destination, **kwargs)
                    profiles[cloth_id] = cache[key]
            destination = assets_output/task_id/configuration/f'seed_{seed:02d}'
            kwargs = {'attempt': attempt} if parameterized else {}
            path = prepare_asset(row['release'], seed, destination, visual_profiles=profiles,
                                 generator_version=row.get('generator_version', GENERATOR_VERSION), **kwargs)
            seed_assets[seed] = path
            report['cases'].append({'task_id': task_id, 'configuration_id': configuration, 'seed': seed,
                                    'asset_manifest': str(path), 'sha256': file_hash(path)})
            write_json(output/'preparation.json', report)
        release = releases_output/task_id/configuration
        prepare_release(row['release'], release, assets_by_seed=seed_assets, configuration_id=configuration)
        report['releases'].append({'task_id': task_id, 'configuration_id': configuration,
                                  'release': str(release), 'manifest_sha256': file_hash(release/'release_manifest.json')})
        write_json(output/'preparation.json', report)
    final_source = common.source_provenance()
    if any(final_source[key] != source_provenance[key]
           for key in ('source_manifest_sha256', 'author_sources_sha256')):
        raise ValueError('Author source identity changed during preparation')
    report['source_provenance_final'] = final_source
    report['complete'] = True
    report['appearance_profiles_count'] = len(cache)
    write_json(output/'preparation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'assets-output', 'releases-output', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    report = prepare(read_json(args.request), assets_output=args.assets_output,
                     releases_output=args.releases_output, output=args.output)
    print({'complete': report['complete'], 'cases': len(report['cases']), 'releases': len(report['releases'])})


if __name__ == '__main__':
    main()
