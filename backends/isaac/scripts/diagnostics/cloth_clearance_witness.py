"""Locate one sampled self-thickness violation using the unchanged CUDA oracle.

This is a single-frame diagnostic, never a task score or a full-stream audit.
Subsets retain original material vertex IDs, so structural exclusions are exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.audit_laundry_cloth import NativeArchive, artifact, require

PROJECT = Path(__file__).resolve().parents[2]


def isolate_pair(items, score, target, tolerance=1.e-10):
    """Find a pair attaining a pairwise maximum with logarithmic subset queries."""
    items = list(items)
    reaches = lambda group: len(group) >= 2 and score(group) >= target-tolerance
    require(reaches(items), 'Full face set no longer reproduces the target')
    while len(items) > 2:
        middle = len(items)//2
        left, right = items[:middle], items[middle:]
        if reaches(left):
            items = left
        elif reaches(right):
            items = right
        else:
            # Neither half contains a maximum internally: its two faces must
            # lie on opposite sides. The other half stays fixed while bisecting.
            while len(left) > 1:
                middle = len(left)//2
                first, second = left[:middle], left[middle:]
                left = first if reaches(first+right) else second
            while len(right) > 1:
                middle = len(right)//2
                first, second = right[:middle], right[middle:]
                right = first if reaches(left+first) else second
            items = left+right
    require(len(items) == 2 and reaches(items), 'Could not reproduce the maximum with two faces')
    return items


def run(args):
    from bench.evaluation.cloth.gpu_intersections import ClothIntersectionAudit

    modules = ('scripts.diagnostics.cloth_clearance_witness',
               'scripts.validation.audit_laundry_cloth', 'bench.evaluation.cloth.gpu_intersections')
    source = verify_if_snapshot(PROJECT, required_modules=modules, entrypoint_file=Path(__file__))
    require(source is not None, 'A complete frozen source is required')
    output = args.output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT) and not output.exists(),
            'Choose fresh shared-dataset output')
    output.mkdir(parents=True)
    identities = {key: artifact(path) for key, path in
                  (('states', args.states), ('task_input', args.task_input))}
    task = load_json(args.task_input)
    spec = next(row for row in task['asset_metadata']['cloths'] if row['id'] == args.cloth_id)
    triangles = np.asarray(spec['triangles'], dtype=np.int32)
    with NativeArchive(args.states, output) as arrays:
        require(np.array_equal(arrays[args.cloth_id+'_triangles'], triangles), 'Frozen topology differs')
        q = np.array(arrays[args.cloth_id+'_q'][args.frame], dtype=np.float32)
        qd = np.array(arrays[args.cloth_id+'_qd'][args.frame])
        mask = np.array(arrays[args.cloth_id+'_control_mask'][args.frame])
        time_s = float(arrays['time_s'][args.frame])
        native_step = int(arrays['native_step'][args.frame])
    cache, calls = {}, []

    def query(indices):
        key = tuple(indices)
        if key not in cache:
            oracle = ClothIntersectionAudit({args.cloth_id: triangles[list(indices)]},
                {args.cloth_id: len(q)}, thickness_by_id={args.cloth_id: spec['physical_thickness_m']},
                device=args.device)
            result = oracle.check({args.cloth_id: q})
            cache[key] = result
            calls.append({'faces': len(indices), 'penetration_m': result['max_cloth_envelope_penetration_m']})
        return cache[key]['max_cloth_envelope_penetration_m']

    full = list(range(len(triangles)))
    target = query(full)
    require(target > 0., 'This frame has no positive nonstructural thickness penetration')
    pair = isolate_pair(full, query, target)
    rest = np.asarray(spec['rest_vertices_m'])
    witness = [{'face_index': i, 'vertex_ids': triangles[i].tolist(),
                'positions_m': q[triangles[i]].tolist(), 'velocities_m_s': qd[triangles[i]].tolist(),
                'rest_positions_m': rest[triangles[i]].tolist(),
                'controlled_vertices': mask[triangles[i]].tolist()} for i in pair]
    final_source = verify_if_snapshot(PROJECT, required_modules=modules, entrypoint_file=Path(__file__))
    for row in identities.values():
        artifact(row['path'], row['sha256'])
    result = {'schema': 'cloth_single_frame_clearance_witness.v1', 'task_id': task['id'],
        'cloth_id': args.cloth_id, 'native_step': native_step, 'time_s': time_s,
        'source': source, 'source_final': final_source,
        'source_snapshot_manifest': artifact(PROJECT/'source_snapshot_manifest.json'),
        'inputs': identities, 'device': args.device,
        'max_envelope_penetration_m': target,
        'implied_mid_surface_separation_m': spec['physical_thickness_m']-target,
        'oracle_result_for_pair': cache[tuple(pair)], 'faces': witness, 'subset_queries': calls,
        'scope': 'Sampled single-cloth pair witness only; not a task score, full-stream audit or force measurement.'}
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'faces': pair, 'penetration_m': target,
                      'subset_queries': len(calls)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--task-input', type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--cloth-id', default='towel')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
