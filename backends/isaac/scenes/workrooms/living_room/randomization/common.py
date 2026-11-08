"""Small canonical-JSON and path helpers shared by reset preparation/loading."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import TASK_IDS, SEEDS, CONFIGURATIONS
from bench.paths import load_json, portable, undistributed_bytecode


def canonical_hash(value):
    # Identities hash data:// references, never this machine's data root.
    return hashlib.sha256(json.dumps(portable(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return load_json(Path(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable(value), indent=2, sort_keys=True, allow_nan=False)+'\n')


def checked_file(path, expected):
    if isinstance(expected, str) and len(expected) == 64 and undistributed_bytecode(path):
        return Path(path)  # pinned bytecode cache, not shipped; see bench.paths.undistributed_bytecode
    if not isinstance(expected, str) or len(expected) != 64 or file_hash(path) != expected:
        raise ValueError('Frozen file identity differs: '+str(path))
    return Path(path)


def local_path(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    path = root/relative
    if (relative.is_absolute() or '..' in relative.parts or
            not path.resolve().is_relative_to(root) or
            any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts)+1))):
        raise ValueError('Frozen case path escapes release')
    return path


def selection(task_id, seed, configuration_id=None):
    if task_id not in TASK_IDS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Ordinary living-room randomization requires a known task and seed 0..9')
    if configuration_id is not None and configuration_id not in CONFIGURATIONS:
        raise ValueError('Unknown ordinary robot installation')


def vector(value, count, label):
    if (not isinstance(value, (list, tuple)) or len(value) != count or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('Finite '+label+' required')
    return [float(x) for x in value]


def checked_poses(poses):
    if not isinstance(poses, dict) or not poses:
        raise ValueError('Actual USD body pose inventory required')
    for name, row in poses.items():
        if not isinstance(name, str) or not row['prim_path'].startswith('/World/LivingTask/'):
            raise ValueError('Unexpected body identity or prim path')
        vector(row['position_m'], 3, 'body position')
        q = vector(row['quaternion_wxyz'], 4, 'body quaternion')
        if abs(sum(x*x for x in q)-1.) > 1.e-8:
            raise ValueError('Unit body quaternion required')
    return poses
