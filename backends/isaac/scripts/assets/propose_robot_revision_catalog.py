"""Build all current robot catalog pins in a dataset draft, never source in place."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path

from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.robot_revision import catalog
from scenes.workrooms.robot_revision.assets import checked_file, file_hash, SUPPORTED_VERSIONS
from scenes.workrooms.robot_revision.inputs import canonical_hash
from scripts.assets.prepare_robot_revision_releases import read_request, descriptors

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return load_json(Path(path))


def proposal(preparation_path, preparation_sha256, assembly_version):
    """Reuse original descriptor and publication validators; no report-based waiver."""
    if assembly_version not in SUPPORTED_VERSIONS:
        raise ValueError('Catalog proposal requires an explicit registered robot version')
    if read(catalog.PINS) != {'schema': 'robot_revision_catalog.v1', 'tasks': {}}:
        raise ValueError('Propose from the original catalog with an empty robot overlay')
    preparation_path = checked_file(preparation_path, preparation_sha256)
    value = read(preparation_path)
    if (value.get('schema') != 'robot_revision_preparation.v1' or value.get('complete') is not True
            or value.get('requested_installations') != 113 or len(value.get('rows', [])) != 113
            or any(value.get(k) is not False for k in ('physics_accepted', 'task_policy_success_claimed',
                    'inherited_qualification', 'catalog_modified', 'old_inputs_modified'))
            or value.get('native_steps') != 0 or value.get('model_calls') != 0):
        raise ValueError('Catalog proposal needs the complete unqualified 113-installation preparation')
    reference = value['request']
    request_path = checked_file(reference['path'], reference['sha256'])
    request = read_request(request_path)
    if Path(request['output']).resolve() != preparation_path.resolve().parent:
        raise ValueError('Preparation report is not at its requested publication output')
    expected = descriptors(request)  # Exact parent matrix/ready/asset/source checks.
    key = lambda r: (r['family'], r['task_id'], r['configuration_id'])
    original = {key(r): r for r in expected}
    actual = {key(r): r for r in value['rows']}
    if len(original) != 113 or len(actual) != 113 or set(actual) != set(original):
        raise ValueError('Prepared installation set differs from the exact original matrix')
    grouped = {}
    manifest_refs = []
    for identity, old in original.items():
        row = actual[identity]
        if (any(row.get(k) != old[k] for k in ('installation_id', 'internal_configuration_id',
                                             'parent_release', 'ready_reference'))
                or row.get('revision_sha256') != canonical_hash(old['revision'])
                or row.get('status') != 'prepared_unvalidated'):
            raise ValueError('Prepared row changed original parent, installation or ready identity')
        robot = old['revision']['robot']
        versions = ({x['assembly_version'] for x in robot['instances'].values()}
                    if robot['hand_id'] == 'fr3_sharpa_instances' else {robot['assembly_version']})
        if versions != {assembly_version}:
            raise ValueError('Prepared descriptor does not use the exact requested robot version')
        release = Path(row['release']['path']).resolve()
        if release != Path(request['output']).resolve()/'releases'/row['installation_id']:
            raise ValueError('Prepared release is outside its exact planned installation path')
        manifest_path = checked_file(release/'release_manifest.json', row['release']['manifest_sha256'])
        manifest = read(manifest_path)
        if (manifest.get('robot_revision_sha256') != row['revision_sha256']
                or manifest.get('configuration_id') != row['internal_configuration_id']
                or manifest.get('selection_kind') != row['selection_kind']):
            raise ValueError('Prepared manifest differs from its derivation receipt')
        name = row['family']+'/'+row['task_id']
        group = grouped.setdefault(name, {})
        public_key = row['configuration_id'] or 'fixed'
        if public_key in group:
            raise ValueError('Duplicate public installation menu key')
        group[public_key] = release
        manifest_refs.append({'installation_id': row['installation_id'], **row['release']})
    if set(grouped) != set(catalog.CURRENT_TASKS):
        raise ValueError('Proposal omitted or added a current task')
    from scenes.workrooms.catalog import task_definition
    tasks = {}
    for name, releases in sorted(grouped.items()):
        family, task = name.split('/')
        # This is the existing full release/parent/menu validator. Do not
        # substitute a locally copied binder or development qualification.
        tasks[name] = catalog.publication_row(task_definition(family, task), releases)
    return {'schema': 'robot_revision_catalog.v1', 'tasks': tasks}, manifest_refs


def source_receipt():
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    import_module('scripts.assets.propose_robot_revision_catalog')
    value = verify_if_snapshot(ROOT, required_modules=('scripts.assets.propose_robot_revision_catalog',
        'scripts.assets.prepare_robot_revision_releases', 'scenes.workrooms.robot_revision.catalog'),
        entrypoint_file=Path(__file__))
    if not value or not value['runtime_imports_verified']:
        raise ValueError('Catalog proposal requires a complete frozen source')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--preparation-sha256', required=True)
    parser.add_argument('--assembly-version', choices=sorted(SUPPORTED_VERSIONS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Catalog drafts need a fresh shared dataset directory')
    source_start = source_receipt()
    value, references = proposal(args.preparation, args.preparation_sha256, args.assembly_version)
    source_final = source_receipt()
    # Recheck all publication identities before writing the proposal receipt.
    checked_file(args.preparation, args.preparation_sha256)
    for ref in references:
        checked_file(Path(ref['path'])/'release_manifest.json', ref['manifest_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    path = output/'catalog_pins.json'
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    summary = {'schema': 'robot_revision_catalog_proposal.v1', 'complete': True,
        'preparation': {'path': str(args.preparation.resolve()), 'sha256': args.preparation_sha256},
        'assembly_version': args.assembly_version, 'tasks': 51, 'installations': 113,
        'proposal': {'path': str(path), 'sha256': file_hash(path)}, 'releases': references,
        'source_start': source_start, 'source_final': source_final,
        'catalog_modified': False, 'inputs_modified': False, 'qualification_inherited': False,
        'scope': 'Input-valid draft for review; no source catalog activation or native/manipulation acceptance.'}
    summary_path = output/'proposal.json'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'proposal': summary['proposal'], 'summary_sha256': file_hash(summary_path)}))


if __name__ == '__main__':
    main()
