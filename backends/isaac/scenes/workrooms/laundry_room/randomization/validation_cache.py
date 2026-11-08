"""Reuse semantic checks only while every validated input byte remains identical.

This process-local cache contains no task state or qualification. Every lookup
rehashes its complete file closure and rechecks the existing strict directory
inventories. No timestamp, external report or persistent cache is trusted.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from threading import Lock

from bench.paths import materialize
from . import SEEDS
from .common import SOURCE_FILES, PARAMETERIZED_SOURCE_FILES, checked_file, local_path, read_json


@dataclass(frozen=True)
class Proof:
    files: tuple[tuple[str, str], ...]
    directories: tuple[tuple[str, tuple[str, ...], bool], ...]
    local_paths: tuple[tuple[str, str], ...]
    resolved_files: tuple[tuple[str, str], ...]
    parent_identity: tuple[str, str, str, str]

    def verify(self):
        from ..robot_configuration_catalog import RELEASES
        task, configuration, parent_path, parent_sha = self.parent_identity
        if RELEASES[task]['configurations'][configuration]['source_release'] != {
                'path': parent_path, 'manifest_sha256': parent_sha}:
            raise ValueError('Cached laundry parent differs from its original catalog pin')
        for path, resolved in self.resolved_files:
            if str(Path(path).resolve(strict=True)) != resolved:
                raise ValueError('Laundry input logical path changed its resolved file')
        for path, expected in self.files:
            checked_file(path, expected)
        for root, relative in self.local_paths:
            local_path(root, relative)
        for root, expected, allow_bytecode in self.directories:
            if directory_files(Path(root), allow_bytecode) != expected:
                raise ValueError('Laundry input directory inventory changed')


def directory_files(root, allow_bytecode):
    return tuple(sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                        and not (allow_bytecode and '__pycache__' in p.relative_to(root).parts
                                 and p.suffix == '.pyc')))


_CACHE = OrderedDict()
_LOCK = Lock()
_LIMIT = 32


def manifest_input(root, supplied):
    """Bind strict JSON values and SHA to one read, including boolean types."""
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('Duplicate JSON key: '+key)
            value[key] = item
        return value
    def same(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return (set(a) == set(b) and all(type(k) is str for k in a)
                    and all(same(a[k], b[k]) for k in a))
        if isinstance(a, list):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite JSON')
        return result
    blob = (Path(root)/'release_manifest.json').read_bytes()
    # Frozen JSON stores data:// references; the supplied manifest was read with load_json.
    parsed = materialize(json.loads(blob, object_pairs_hook=pairs, parse_float=finite_float,
                                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON'))))
    if not isinstance(parsed, dict) or not same(parsed, supplied):
        raise ValueError('Laundry manifest argument differs from its frozen file')
    return hashlib.sha256(blob).hexdigest()


def get(key):
    from bench.isaacsim.validation_epoch import semantic_reuse_enabled
    if not semantic_reuse_enabled():
        return None
    with _LOCK:
        proof = _CACHE.get(key)
        if proof is not None:
            _CACHE.move_to_end(key)
    if proof is not None:
        try:
            proof.verify()
        except (ValueError, OSError, KeyError):
            discard(key)
            raise
    return proof


def discard(key):
    with _LOCK:
        _CACHE.pop(key, None)


def store(key, proof):
    from bench.isaacsim.validation_epoch import semantic_reuse_enabled
    if not semantic_reuse_enabled():
        return
    with _LOCK:
        _CACHE[key] = proof
        _CACHE.move_to_end(key)
        while len(_CACHE) > _LIMIT:
            _CACHE.popitem(last=False)


def clear():
    """Clear only process-local acceleration; never changes frozen inputs."""
    with _LOCK:
        _CACHE.clear()


def collect(root, manifest, manifest_sha):
    """Build an expected closure before semantic evaluation, then verify it.

    All digests below originate in pinned declarations, never from a post-check
    read of a potentially changed dependency. A final verify before insertion
    therefore cannot certify different bytes than the successful cold check.
    """
    files, directories, local_paths, resolved_files = {}, {}, set(), {}

    def add(path, expected):
        path = str(Path(path).absolute())
        if path in files and files[path] != expected:
            raise ValueError('Conflicting laundry input hash declarations')
        checked_file(path, expected)
        files[path] = expected
        resolved = str(Path(path).resolve(strict=True))
        if path in resolved_files and resolved_files[path] != resolved:
            raise ValueError('Laundry input logical path changed during closure collection')
        resolved_files[path] = resolved

    def add_directory(path, names, allow_bytecode=False):
        path = Path(path).resolve()
        expected = tuple(sorted(names))
        if directory_files(path, allow_bytecode) != expected:
            raise ValueError('Laundry input directory inventory changed')
        directories[str(path)] = (expected, allow_bytecode)

    def add_local(root, name, expected):
        add(local_path(root, name), expected)
        local_paths.add((str(Path(root).resolve()), str(name)))

    def provenance(value):
        source = value['source_snapshot']
        snapshot = Path(source['snapshot_root'])
        if not snapshot.is_absolute() or source.get('runtime_imports_verified') is not True:
            raise ValueError('Missing frozen laundry source identity')
        add(snapshot/'source_snapshot_manifest.json', value.get('source_manifest_sha256'))
        hashes = value['author_sources_sha256']
        if set(hashes) not in (set(SOURCE_FILES), set(PARAMETERIZED_SOURCE_FILES)):
            raise ValueError('Incomplete frozen laundry author inventory')
        snapshot_manifest = read_json(snapshot/'source_snapshot_manifest.json')
        for name, digest in hashes.items():
            if snapshot_manifest['files_sha256'].get(name) != digest:
                raise ValueError('Frozen author source differs from snapshot declaration')
            add_local(snapshot, name, digest)

    root = Path(root).resolve()
    add(root/'release_manifest.json', manifest_sha)
    for name, expected in manifest['files_sha256'].items():
        add_local(root, name, expected)
    add_directory(root, {*manifest['files_sha256'], 'release_manifest.json'}, True)
    provenance(read_json(root/'source_provenance.json'))
    parent = manifest['base_release']
    parent_root = Path(parent['path']).resolve()
    add(parent_root/'release_manifest.json', parent['manifest_sha256'])
    parent_manifest = read_json(parent_root/'release_manifest.json')
    for name, expected in parent_manifest['files_sha256'].items():
        add_local(parent_root, name, expected)
    add_directory(parent_root, {*parent_manifest['files_sha256'], 'release_manifest.json'}, True)
    # Includes both all parent physical assets and all new appearance/seed files.
    # This actual-content check is mandatory for both False and True cache keys.
    for path, expected in manifest['asset_files_sha256'].items():
        add(path, expected)
    for seed in SEEDS:
        case = manifest['cases'][str(seed)]
        path = Path(case['asset_manifest']).resolve()
        add(path, case['asset_manifest_sha256'])
        asset = read_json(path)
        provenance(asset['source_provenance'])
        for name, expected in asset['files_sha256'].items():
            add_local(path.parent, name, expected)
        add_directory(path.parent, {*asset['files_sha256'], path.name})
        if seed:
            for pin in asset['visual_profiles'].values():
                add(pin['path'], pin['sha256'])
                source_path = Path(pin['path']).parent/'SOURCE.json'
                # Its expected digest must already be an explicit release dependency.
                add(source_path, manifest['asset_files_sha256'][str(source_path)])
                provenance(read_json(source_path))
    proof = Proof(tuple(sorted(files.items())), tuple((p, *row) for p, row in sorted(directories.items())),
                  tuple(sorted(local_paths)),
                  tuple(sorted(resolved_files.items())),
                  (manifest['task_id'], manifest['configuration_id'], str(parent_root), parent['manifest_sha256']))
    proof.verify()
    return proof
