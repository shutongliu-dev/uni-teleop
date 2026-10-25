"""Derive a source-only runtime contract for a byte-identical source copy.

Relative project keys stay relative. Absolute keys inside the original project
move to the new project. Other absolute keys require an exact shared-file
allowlist and remain unchanged. Both locations must match the original digest;
this never accepts an observed runtime hash in place of source verification.
Controller/implementation/environment identities must be rebuilt for the new
root and are refused here, including unknown opaque fields. A source relocation
is not a complete native launch contract.

Member spellings come from the supplied frozen contract. Relocation preserves
old simulation/ paths and new bench/simulation/ paths independently; it never
translates one source layout into another, even when the bytes happen to match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


SCHEMA = 'verified_runtime_source_relocation_v1'
SOURCE_SCHEMAS = frozenset({SCHEMA, 'explicit_service_runtime_source_contract_v1',
                            'contact_history_runtime_sensor_transition_v1'})
# These describe pinned content, not a controller or launch-root identity. They
# are carried as supplied pins, not revalidated asset/physics/sensor evidence.
CONTENT_PIN_KEYS = frozenset({'robot_asset_sha256', 'physics_profile_sha256',
                             'sensor_config_sha256', 'task_release_sha256',
                             'evaluator_sha256'})


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _file(path: Path, expected: str):
    if not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError(f'Source must be an existing regular file without aliases: {path}')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f'Source digest mismatch: {path}')


def rebase_runtime_contract(contract, old_project, new_project, *, shared_files=()):
    required = {'runtime_source_inventory', 'runtime_code_sha256'}
    if (not isinstance(contract, dict) or not required <= set(contract)
            or set(contract) - required - CONTENT_PIN_KEYS - {'schema', 'path_rebase'}
            or ('schema' in contract and contract['schema'] not in SOURCE_SCHEMAS)):
        raise ValueError('Unsupported contract fields/schema: controller, implementation and '
                         'environment identities must be rebuilt for the new root')
    for key in CONTENT_PIN_KEYS & set(contract):
        if not isinstance(contract[key], str) or re.fullmatch('[0-9a-f]{64}', contract[key]) is None:
            raise ValueError(f'Invalid content pin SHA-256: {key}')
    old = Path(old_project).absolute()
    new = Path(new_project).absolute()
    if (old.is_relative_to(new) or new.is_relative_to(old) or not old.is_dir() or not new.is_dir()
            or old.resolve() != old or new.resolve() != new):
        raise ValueError('Existing disjoint project roots without aliases are required')
    inventory = contract.get('runtime_source_inventory')
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError('A nonempty original source inventory is required')
    if canonical_hash(inventory) != contract.get('runtime_code_sha256'):
        raise ValueError('Original runtime inventory hash mismatch')
    shared = {str(Path(p).absolute()) for p in shared_files}
    rebased, provenance, used = {}, [], set()
    for key, digest in inventory.items():
        if (not isinstance(key, str) or not key or '\\' in key
                or any(p in ('', '.', '..') for p in key.lstrip('/').split('/'))
                or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None):
            raise ValueError('Invalid source key or SHA-256')
        path = Path(key)
        if not path.is_absolute():
            source, target, target_key, kind = old/path, new/path, key, 'relative_project'
        elif path.is_relative_to(old):
            source, target = path, new/path.relative_to(old)
            target_key, kind = str(target), 'absolute_project_relocated'
        elif key in shared:
            source = target = path
            target_key, kind = key, 'shared_absolute_unchanged'
            used.add(key)
        else:
            raise ValueError(f'Unregistered external source: {key}')
        if target_key in rebased:
            raise ValueError(f'Rebased key collision: {target_key}')
        _file(source, digest)
        _file(target, digest)
        rebased[target_key] = digest
        provenance.append({'old_key': key, 'new_key': target_key, 'kind': kind,
                           'old_file': str(source), 'new_file': str(target), 'sha256': digest})
    if shared != used:
        raise ValueError('Shared-file allowlist contains unused entries')
    # Reconstruct only the documented source contract. In particular, historical
    # path_rebase is not a current promise and must never be copied forward.
    result = {'schema': SCHEMA, 'runtime_source_inventory': rebased,
              'runtime_code_sha256': canonical_hash(rebased),
              **{key: contract[key] for key in sorted(CONTENT_PIN_KEYS & set(contract))}}
    result['path_rebase'] = {'schema': 'verified_runtime_contract_path_rebase_v1',
        'verifier_version': 'runtime_contract_path_rebase_verifier_v3',
        'old_project': str(old), 'new_project': str(new),
        'original_contract_sha256': canonical_hash(contract),
        'old_runtime_code_sha256': contract['runtime_code_sha256'],
        'new_runtime_code_sha256': result['runtime_code_sha256'],
        'all_source_digests_preserved': True, 'sources': provenance,
        'full_runtime_contract_generated': False,
        'controller_environment_rebuild_required': True,
        'scope': 'Source inventory relocation only; supplied content pins are retained, '
                 'not revalidated. Build controller and environment identities at the new root.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--old-project', type=Path, required=True)
    parser.add_argument('--new-project', type=Path, required=True)
    parser.add_argument('--shared-file', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = rebase_runtime_contract(json.loads(args.contract.read_text()),
        args.old_project, args.new_project, shared_files=args.shared_file)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
