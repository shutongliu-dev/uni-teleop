"""Bind newly authored ordinary instances to both preserved physical robots.

The request pins newly prepared original-robot releases and existing L-mount
robot descriptors. No robot-dependent world resampling is allowed here. Existing
ready coordinates are only inputs; new geometry/IK/native evidence is required.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

from bench.paths import OUTPUT_ROOT
from scenes.workrooms.parameterized.pairs import (
    PROFILES, pair_record, read, reference, file_hash, checked_file)
from scenes.workrooms.robot_revision.release import prepare_release

SCHEMA = 'parameterized_pair_preparation_request.v1'
ROOT = Path(__file__).resolve().parents[2]


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def _pin(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'manifest_sha256'}
            or not Path(value['path']).is_absolute()):
        raise ValueError('An absolute immutable release reference is required')
    checked_file(Path(value['path'])/'release_manifest.json', value['manifest_sha256'])
    return Path(value['path']).resolve()


def checked_request(path):
    from scenes.workrooms.parameterized.catalog import TASKS
    request = read(path)
    if (not isinstance(request, dict) or set(request) != {'schema', 'installations', 'output'}
            or request.get('schema') != SCHEMA or not isinstance(request['installations'], list)
            or not request['installations']):
        raise ValueError('Explicit paired installation request required')
    output = Path(request['output'])
    if (not output.is_absolute() or not output.resolve().is_relative_to(OUTPUT_ROOT)
            or output.exists()):
        raise ValueError('Use a new shared dataset publication directory')
    seen = set()
    for row in request['installations']:
        if set(row) != {'family', 'task_id', 'configuration_id', 'original_release', 'previous_l_mount_release'}:
            raise ValueError('Incomplete paired installation references')
        identity = '/'.join((row['family'], row['task_id']))
        key = (identity, row['configuration_id'])
        if (identity not in TASKS or key in seen
                or row['configuration_id'] not in ('single_arm', 'dual_same_side', 'dual_opposed')):
            raise ValueError('Unknown or repeated physical installation')
        seen.add(key)
        for name in ('original_release', 'previous_l_mount_release'):
            _pin(row[name])
    return request


def run(request_path):
    from bench.isaacsim.release import validate
    from scenes.workrooms.catalog import task_definition
    from scenes.workrooms.parameterized.catalog import publication_row
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    modules = ('scripts.assets.prepare_parameterized_pairs', 'scenes.workrooms.parameterized.pairs',
               'scenes.workrooms.robot_revision.release', 'scenes.workrooms.robot_revision.inputs')
    source = verify_if_snapshot(ROOT, required_modules=modules, entrypoint_file=Path(__file__))
    if not source or source.get('runtime_imports_verified') is not True:
        raise ValueError('Pair preparation must run from a complete frozen source snapshot')
    request = checked_request(request_path)
    output = Path(request['output']).resolve()
    output.mkdir(parents=True)
    report = {'schema': 'parameterized_pair_preparation.v1', 'complete': False,
              'request': {'path': str(Path(request_path).resolve()), 'sha256': file_hash(request_path)},
              'source_start': source, 'source_final': None, 'installations': [],
              'robot_profiles': deepcopy(PROFILES), 'catalog_modified': False,
              'physics_accepted': False, 'task_policy_success_claimed': False,
              'model_calls': 0, 'native_steps': 0,
              'scope': 'Frozen shared tasks and robot-only derivatives; ready poses require new validation.'}
    _write(output/'preparation.json', report)
    validation = {'schema': 'parameterized_workspace_request.v1', 'cases': []}
    grouped = {}
    try:
        for row in request['installations']:
            original, previous = [_pin(row[key]) for key in ('original_release', 'previous_l_mount_release')]
            old_manifest, previous_manifest = [validate(path) for path in (original, previous)]
            if (old_manifest.get('seed_values') != list(range(10))
                    or old_manifest.get('task_id') != row['task_id']
                    or old_manifest.get('configuration_id') != row['configuration_id']
                    or previous_manifest.get('task_id') != row['task_id']
                    or previous_manifest.get('configuration_id') != row['configuration_id']
                    or previous_manifest.get('schema') != 'workroom_robot_revision_release.v1'):
                raise ValueError('Requested installation differs from the actual frozen releases')
            revision_name = previous_manifest['robot_revision']
            revision = read(checked_file(previous/revision_name,
                                         previous_manifest['files_sha256'][revision_name]))
            asset_pin = previous_manifest['robot_asset_manifest']
            checked_file(asset_pin['path'], asset_pin['sha256'])
            relative = Path(row['family'])/row['task_id']/row['configuration_id']
            revised = output/'releases'/relative
            prepare_release(original, revised, revision=revision,
                            robot_asset_manifest=asset_pin['path'], configuration_id=row['configuration_id'],
                            observer_compatibility=previous_manifest.get('observer_compatibility', {}).get('version'))
            # The publisher above validated asset dependencies. Re-read every
            # frozen case and enforce exact pair identity independently here.
            pair = pair_record(original, revised, validate_assets=False)
            pair_path = output/'pairs'/relative.with_suffix('.json')
            _write(pair_path, pair)
            pair_ref = {'path': str(pair_path), 'sha256': file_hash(pair_path)}
            record = {key: deepcopy(row[key]) for key in ('family', 'task_id', 'configuration_id')}
            record.update(robot_profiles=deepcopy(pair['robot_profiles']), pair_reference=pair_ref)
            report['installations'].append(record)
            for profile in PROFILES:
                for seed in range(10):
                    validation['cases'].append({
                        'family': row['family'], 'task_id': row['task_id'],
                        'configuration_id': row['configuration_id'], 'seed': seed,
                        'release': pair['robot_profiles'][profile]['path'],
                        'release_manifest_sha256': pair['robot_profiles'][profile]['manifest_sha256'],
                        'robot_profile': profile, 'pair_reference': deepcopy(pair_ref),
                        'physical_case_sha256': pair['physical_case_sha256'][str(seed)]})
            identity = row['family']+'/'+row['task_id']
            definition = task_definition(row['family'], row['task_id'], robot_profile='original')
            public_key = row['configuration_id'] if definition.get('robot_configurations') else 'fixed'
            grouped.setdefault(identity, {'definition': definition, 'pairs': {}})['pairs'][public_key] = pair_ref
            _write(output/'preparation.json', report)
            _write(output/'validation_request.json', validation)
            print(json.dumps({'event': 'paired', **record}, ensure_ascii=False), flush=True)
        catalog_rows = {}
        for identity, group in grouped.items():
            options = group['definition'].get('robot_configurations') or {'fixed': group['definition']}
            if set(options) == set(group['pairs']):
                catalog_rows[identity] = publication_row(group['definition'], group['pairs'])
        _write(output/'catalog_rows.json', catalog_rows)
        report['source_final'] = verify_if_snapshot(ROOT, required_modules=modules, entrypoint_file=Path(__file__))
        if not report['source_final'] or report['source_final'].get('runtime_imports_verified') is not True:
            raise ValueError('Pair preparation source identity failed to close')
        report.update(complete=True, case_count=len(validation['cases']),
                      catalog_rows_sha256=file_hash(output/'catalog_rows.json'),
                      validation_request_sha256=file_hash(output/'validation_request.json'))
        _write(output/'preparation.json', report)
    except BaseException as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        _write(output/'preparation.json', report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    result = run(parser.parse_args().request)
    print(json.dumps({'complete': result['complete'], 'case_count': result['case_count'],
                      'catalog_modified': False}))


if __name__ == '__main__':
    main()
