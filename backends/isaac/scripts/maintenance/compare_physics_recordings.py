"""Compare immutable native recordings, excluding contact report arithmetic.

Records source/input hashes and compares every overlapping command, joint,
robot-link/object pose and velocity, plus every captured native state array.
This is an analysis artifact: it neither changes nor re-scores either episode.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def compare_tree(a, b, stat, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f'Dictionary keys differ at {path}')
        for key in sorted(a):
            compare_tree(a[key], b[key], stat, path+'/'+str(key))
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError(f'List shape differs at {path}')
        for i, (left, right) in enumerate(zip(a, b)):
            compare_tree(left, right, stat, path+'/'+str(i))
        return
    if (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f'Nonfinite value at {path}')
        error = abs(a-b)
        stat['numeric_values'] += 1
        stat['maximum_absolute_difference'] = max(stat['maximum_absolute_difference'], error)
        different = a != b
    else:
        if type(a) is not type(b):
            raise ValueError(f'Value type differs at {path}: {type(a)} / {type(b)}')
        different = a != b
    if different:
        stat['differing_values'] += 1
        if stat['first_difference'] is None:
            stat['first_difference'] = {'path': path, 'old': a, 'new': b}


def compare_records(old, new):
    fields = ['command', 'joint_positions_rad', 'joint_velocities_rad_s',
              'robot', 'objects', 'time_s', 'sim_time_s', 'last_duration_s']
    stats = {field: {'numeric_values': 0, 'differing_values': 0,
                    'maximum_absolute_difference': 0., 'first_difference': None} for field in fields}
    counts = [0, 0]
    overlap = 0
    with old.open() as left, new.open() as right:
        for index, pair in enumerate(zip_longest(left, right)):
            rows = [json.loads(line) if line is not None else None for line in pair]
            for i, row in enumerate(rows):
                counts[i] += row is not None
            if None in rows:
                continue
            a, b = rows
            if a['kind'] != b['kind']:
                raise ValueError(f'Record kinds differ at frame {index}')
            overlap += 1
            for field, stat in stats.items():
                old_value = a.get('command') if field == 'command' else a['observation'][field]
                new_value = b.get('command') if field == 'command' else b['observation'][field]
                compare_tree(old_value, new_value, stat, f'frame/{index}/{field}')
    return {'old_frames': counts[0], 'new_frames': counts[1], 'overlap_frames': overlap,
            'full_lengths_equal': counts[0] == counts[1], 'fields': stats,
            'overlap_exact_equal': all(s['differing_values'] == 0 for s in stats.values())}


def compare_states(old, new):
    stats = {}
    with np.load(old, allow_pickle=False) as left, np.load(new, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise ValueError('Native state array inventories differ')
        for key in sorted(left.files):
            a, b = left[key], right[key]
            if a.shape[1:] != b.shape[1:]:
                raise ValueError(f'State array dimensions differ: {key}')
            n = min(a.shape[0], b.shape[0])
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f'Nonfinite state array: {key}')
            delta = np.abs(a[:n].astype(np.float64)-b[:n].astype(np.float64))
            stats[key] = {'old_shape': list(a.shape), 'new_shape': list(b.shape),
                'old_dtype': str(a.dtype), 'new_dtype': str(b.dtype),
                'overlap_frames': n, 'full_lengths_equal': a.shape[0] == b.shape[0],
                'maximum_absolute_difference': float(delta.max(initial=0.)),
                'differing_values': int(np.count_nonzero(a[:n] != b[:n])),
                'overlap_exact_equal': bool(np.array_equal(a[:n], b[:n]))}
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-private', type=Path, required=True)
    parser.add_argument('--new-private', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh comparison output')
    paths = {version: {name: root/name for name in ('physics.jsonl', 'states.npz', 'metadata.json')}
             for version, root in (('old', args.old_private), ('new', args.new_private))}
    inputs = {version: {name: {'path': str(path.resolve()), 'sha256': digest(path)}
              for name, path in values.items()} for version, values in paths.items()}
    records = compare_records(paths['old']['physics.jsonl'], paths['new']['physics.jsonl'])
    arrays = compare_states(paths['old']['states.npz'], paths['new']['states.npz'])
    for version, values in paths.items():
        for name, path in values.items():
            if digest(path) != inputs[version][name]['sha256']:
                raise RuntimeError('Input changed during comparison: '+str(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.output.with_name(args.output.stem+'_source.py')
    if source.exists():
        raise FileExistsError(source)
    shutil.copy2(__file__, source)
    report = {'schema': 'native_physics_trace_comparison.v1',
        'source': {'path': str(source.resolve()), 'sha256': digest(source)}, 'inputs': inputs,
        'scope': 'Commands, complete robot/object pose and velocity observations, simulation clocks, and every captured native state array.',
        'excluded': ['Contact arithmetic intentionally changed by the recording fix.',
                     'RGB/wall-clock timing and task/audit verdicts are not physical state comparisons.'],
        'records': records, 'native_state_arrays': arrays,
        'full_physical_trace_exact_equal': records['full_lengths_equal'] and records['overlap_exact_equal']
            and all(s['full_lengths_equal'] and s['overlap_exact_equal'] for s in arrays.values()),
        'episodes_modified': False, 'episodes_rescored': False}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'frames': records['overlap_frames'],
        'full_physical_trace_exact_equal': report['full_physical_trace_exact_equal']}))


if __name__ == '__main__':
    main()
