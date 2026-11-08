"""Strict finite JSON, immutable hashes and paths for frozen physical inputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from bench.paths import load_json, portable, undistributed_bytecode


def canonical_hash(value):
    # Identities hash data:// references, never this machine's data root.
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    from bench.isaacsim.validation_epoch import file_sha256
    return file_sha256(path)


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return load_json(Path(path), object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Legacy task contracts generate ordered goal lists from dictionary order.
    # Canonical hashes sort keys; serialized task inputs preserve their order.
    path.write_text(json.dumps(portable(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def checked_file(path, expected):
    if isinstance(expected, str) and len(expected) == 64 and undistributed_bytecode(path):
        return Path(path)  # pinned bytecode cache, not shipped; see bench.paths.undistributed_bytecode
    if not isinstance(expected, str) or len(expected) != 64 or file_hash(path) != expected:
        raise ValueError('Frozen input differs: ' + str(path))
    return Path(path)


def local_path(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    if (relative.is_absolute() or '..' in relative.parts
            or not (root/relative).resolve().is_relative_to(root)
            or any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts)+1))):
        raise ValueError('Input path escapes frozen release')
    return root/relative


def vector(value, count):
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('Finite vector required')
    return value


def checked_poses(poses):
    if not isinstance(poses, dict) or not poses:
        raise ValueError('Actual authored object pose inventory is required')
    paths = set()
    for row in poses.values():
        path = row.get('prim_path')
        if not isinstance(path, str) or not path.startswith('/World/') or path in paths:
            raise ValueError('Unique world object path required')
        paths.add(path)
        vector(row['position_m'], 3)
        q = vector(row['quaternion_wxyz'], 4)
        if abs(sum(v*v for v in q)-1.) > 1e-6:
            raise ValueError('Unit authored quaternion required')
