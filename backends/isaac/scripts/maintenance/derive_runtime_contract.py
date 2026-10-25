#!/usr/bin/env python3
"""Build a new source contract for an explicitly pinned service-only replacement.

This is deliberately not a general controller/native/evaluator migration tool.
The old/new complete snapshots must have identical inventory membership, with
only the service source declared by the parent snapshot layout changed. Native absolute source keys
relocate and relative frontend keys retain their spelling. Old and new bytes,
the v2 derivation, and an independent explicit change request must all agree.
Controller descriptors and execution-environment hashes require separate real
construction in the new root; this utility rejects them as stale input fields.
No native process, model call or benchmark action is started.
"""
import argparse
import importlib.util
import json
from pathlib import Path

_DERIVER = Path(__file__).resolve().with_name('derive_project_sources.py')
_SPEC = importlib.util.spec_from_file_location('runtime_source_deriver', _DERIVER)
sources = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sources)
SCHEMA = 'explicit_service_runtime_source_contract_v1'
LEGACY_SCHEMA = 'contact_history_runtime_sensor_transition_v1'
LEGACY_FIELDS = frozenset({'reference_public_metadata', 'reference_public_metadata_sha256',
    'reference_source_inventory', 'reference_source_inventory_sha256', 'source_transition',
    'added_runtime_sources', 'integration_ready', 'integration_ready_sha256',
    'integration_expected_sources', 'sensor_config', 'sensor_provenance', 'scope', 'path_rebase'})
PIN_KEYS = frozenset({'robot_asset_sha256', 'physics_profile_sha256', 'sensor_config_sha256',
                      'task_release_sha256', 'evaluator_sha256'})
SUPPORTED_CHANGE = 'bench/runtime/service.py'
LEGACY_SUPPORTED_CHANGE = 'interface/agent/service.py'


def _service_source(parent_manifest):
    """Choose the service member from verified layout metadata, never imports.

    Versions 1--4 retain the old three source roots and interface/agent service.
    Versions 5--6 declare the separate agent root and bench/runtime service.
    This selects a member within one layout; it does not rename historical files.
    """
    roots = sources.freeze.snapshot_source_trees(parent_manifest)
    return SUPPORTED_CHANGE if 'agent' in roots else LEGACY_SUPPORTED_CHANGE


def _read(path):
    path = sources._absolute(path); sources._regular(path)
    raw = path.read_bytes()
    return path, raw, sources._json(raw)


def derive_runtime_contract(contract_file, derived_project, request_file):
    old_file, old_raw, contract = _read(contract_file)
    request_file, request_raw, request = _read(request_file)
    new = sources._absolute(derived_project)
    required = {'schema', 'old_contract_sha256', 'derived_manifest_sha256', 'allowed_source_changes'}
    if (not isinstance(request, dict) or set(request) != required
            or request['schema'] != 'explicit_service_runtime_change_request_v1'):
        raise ValueError('Expected exact explicit service change request schema')
    if sources._sha_bytes(old_raw) != sources._hash(request['old_contract_sha256'], 'old contract'):
        raise ValueError('Old contract file SHA mismatch')
    manifest_file = new / sources.MANIFEST_NAME
    new_manifest_sha = sources._sha_file(manifest_file)
    if new_manifest_sha != sources._hash(request['derived_manifest_sha256'], 'derived manifest'):
        raise ValueError('Derived manifest SHA mismatch')
    verified = sources.verify_derivation(new)
    if verified['derivation_verified'] is not True or verified['offline_parent_transition_verified'] is not True:
        raise ValueError('A fully verified v2 source derivation is required')
    manifest = sources._json(manifest_file.read_bytes())
    provenance = sources._json((new / sources.PROVENANCE_NAME).read_bytes())
    old = sources._absolute(manifest['source_root'])
    parent = sources._parent(old, provenance['parent_manifest_sha256'])
    supported_change = _service_source(parent)
    old_files, new_files = parent['files_sha256'], manifest['files_sha256']
    if set(old_files) != set(new_files):
        raise ValueError('Runtime inventory additions/deletions require separate explicit enumeration')
    changes = [{'path': name, 'old_sha256': old_files[name], 'new_sha256': new_files[name]}
               for name in sorted(old_files) if old_files[name] != new_files[name]]
    allowed = request['allowed_source_changes']
    if (not isinstance(allowed, list) or not allowed
            or any(not isinstance(row, dict) or set(row) != {'path', 'old_sha256', 'new_sha256'} for row in allowed)):
        raise ValueError('Expected nonempty explicit old/new source change list')
    names = set()
    for row in allowed:
        sources._relative(row['path'])
        sources._hash(row['old_sha256'], 'allowed old SHA'); sources._hash(row['new_sha256'], 'allowed new SHA')
        if row['path'] in names:
            raise ValueError('Duplicate allowed source change')
        names.add(row['path'])
    if sorted(allowed, key=lambda row: row['path']) != changes:
        raise ValueError('Explicit allowed changes do not equal the complete snapshot content changes')
    if names != {supported_change}:
        raise ValueError('Only service-only replacement is supported; controller/native/other changes need additional contract derivation')
    keys = {'schema', 'runtime_source_inventory', 'runtime_code_sha256'}
    if (not isinstance(contract, dict) or contract.get('schema') not in {SCHEMA, LEGACY_SCHEMA}
            or not keys <= set(contract)
            or set(contract) - keys - PIN_KEYS - (LEGACY_FIELDS if contract['schema'] == LEGACY_SCHEMA else frozenset())):
        raise ValueError('Unsupported contract fields/schema: opaque implementation/controller/environment descriptors may not be carried forward')
    for key in PIN_KEYS & set(contract):
        sources._hash(contract[key], key)
    inventory = contract['runtime_source_inventory']
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('Expected nonempty old runtime source inventory')
    if sources._canonical_hash(inventory) != sources._hash(contract['runtime_code_sha256'], 'old runtime hash'):
        raise ValueError('Old runtime inventory hash mismatch')
    before_pins = {str(old_file): sources._sha_bytes(old_raw), str(request_file): sources._sha_bytes(request_raw),
                   str(manifest_file): new_manifest_sha,
                   str(Path(__file__).resolve()): sources._sha_file(Path(__file__).resolve()),
                   str(_DERIVER): sources._sha_file(_DERIVER),
                   str(sources._VERIFIER_PATH): sources._sha_file(sources._VERIFIER_PATH)}
    output, records, covered_changes = {}, [], set()
    for key, old_digest in inventory.items():
        sources._hash(old_digest, 'runtime source')
        if not isinstance(key, str):
            raise ValueError('Runtime source keys must be strings')
        path = Path(key)
        if path.is_absolute():
            path = sources._absolute(key)
            if not path.is_relative_to(old):
                raise ValueError('External absolute runtime keys are outside this service-only derivation')
            relative = str(path.relative_to(old))
            new_key, kind = str(new / relative), 'absolute_project_relocated'
        else:
            relative = sources._relative(key)
            new_key, kind = relative, 'relative_project'
        if relative not in old_files or old_digest != old_files[relative]:
            raise ValueError('Old runtime source is not bound to the verified parent inventory')
        target_digest = new_files[relative]
        if old_digest != target_digest:
            covered_changes.add(relative)
        if new_key in output:
            raise ValueError('Runtime key collision')
        for root, digest in ((old, old_digest), (new, target_digest)):
            source = root / relative; sources._regular(source)
            if sources._sha_file(source) != digest:
                raise ValueError('Runtime source bytes changed during derivation')
        output[new_key] = target_digest
        records.append({'old_key': key, 'new_key': new_key, 'relative_source': relative,
                        'old_sha256': old_digest, 'new_sha256': target_digest, 'kind': kind})
    if covered_changes != names:
        raise ValueError('Changed service source is absent from the old runtime contract')
    sources._parent(old, provenance['parent_manifest_sha256'])
    sources.verify_derivation(new)
    for path, digest in before_pins.items():
        sources._regular(Path(path))
        if sources._sha_file(Path(path)) != digest:
            raise ValueError('Contract/request/snapshot/tool changed during derivation')
    # Legacy integration/source_transition/path_rebase fields describe the old
    # experiment. Preserve them verbatim as historical proof, never as current
    # claims beside a changed service hash.
    stable_pins = {key: contract[key] for key in sorted(PIN_KEYS & set(contract))}
    result = {'schema': SCHEMA, 'runtime_source_inventory': output,
              'runtime_code_sha256': sources._canonical_hash(output), **stable_pins}
    proof = {'schema': 'verified_service_runtime_content_derivation_v1',
             'old_project': str(old), 'new_project': str(new),
             'old_contract_path': str(old_file), 'old_contract_file_sha256': sources._sha_bytes(old_raw),
             'old_contract_canonical_sha256': sources._canonical_hash(contract),
             'historical_parent_contract_utf8': old_raw.decode('utf-8'),
             'historical_only_parent_fields': sorted(set(contract) - keys - PIN_KEYS),
             'new_contract_canonical_sha256': sources._canonical_hash(result),
             'request_path': str(request_file), 'request_sha256': sources._sha_bytes(request_raw),
             'request_utf8': request_raw.decode('utf-8'),
             'parent_manifest_sha256': provenance['parent_manifest_sha256'],
             'parent_source_utility_version': parent.get('source_utility_version'),
             'supported_service_source': supported_change,
             'derived_manifest_sha256': new_manifest_sha,
             'derived_runtime_source_identity_sha256': verified['runtime_source_identity_sha256'],
             'old_runtime_code_sha256': contract['runtime_code_sha256'],
             'new_runtime_code_sha256': result['runtime_code_sha256'],
             'allowed_source_changes': allowed, 'sources': records,
             'unchanged_nonruntime_fields': stable_pins,
             'tool_and_input_sha256': before_pins, 'native_launches': 0, 'model_calls': 0,
             'protected_source_roots': sorted({str(old), str(new), *manifest['source_root_ancestors']}),
             'native_metadata_validation_required': True,
             'runtime_membership_scope': 'Preserves the supplied old runtime members; complete membership must be checked against actual service/native metadata before use.',
             'not_derived': ['execution_environment_sha256', 'controller/system0 descriptors and implementation hashes'],
             'scope': 'Service source replacement only; old/new source bytes and complete v2 inventory verified. Actual service/native metadata and other root-sensitive pins must be checked before a new run.'}
    return result, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True)
    parser.add_argument('--derived-project', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--proof', required=True)
    args = parser.parse_args()
    output = sources._absolute(args.output, must_exist=False)
    proof_path = sources._absolute(args.proof, must_exist=False)
    if output == proof_path or output.exists() or proof_path.exists():
        raise ValueError('Contract and proof need distinct fresh output files')
    result, proof = derive_runtime_contract(args.contract, args.derived_project, args.request)
    protected = [Path(root) for root in proof['protected_source_roots']]
    if any(path.is_relative_to(root) for path in (output, proof_path) for root in protected):
        raise ValueError('Contract/proof output must be outside parent, derived and original source roots')
    if not output.parent.is_dir() or not proof_path.parent.is_dir():
        raise ValueError('Prepare separate output directories before deriving contract/proof')
    for path, value in ((output, result), (proof_path, proof)):
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'runtime_code_sha256': result['runtime_code_sha256'],
                      'keys': len(result['runtime_source_inventory']), 'native_launches': 0, 'model_calls': 0}))


if __name__ == '__main__':
    main()
