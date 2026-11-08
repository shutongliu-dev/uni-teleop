"""Check the changed NIST props against all three actual robot installations.

Discrete middle-fingertip approaches and original USD collider bounds only;
no held-object sweep, grasp or native manipulation qualification is inferred.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import checked, task_inputs
from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.mechanical_assembly.robot_layout import stand_geometry
from scripts.diagnostics.check_bimanual_options import (
    CollisionBounds, content_digest, digest, run, solve_target, task_points,
)


def single(release, task_id):
    _, scene, _, initial = task_inputs(release, task_id, seed=17)
    samples, _, _ = task_points('mechanical_assembly', task_id, release)
    robot = scene['robot']
    model = public_robot_geometry(robot_configuration(scene))
    joints = model.checked_joints(initial['joint_positions_rad'], enforce_limits=True)
    checker = CollisionBounds(scene, robot, {'right': model}, joints, {'right': stand_geometry()})
    checker.approach_surfaces(samples)
    ready = {'right': checker.boxes('right', joints)}
    ready_check, stands = checker.check(ready), checker.stand_check()
    rows = []
    for sample in samples:
        answer = solve_target(model, joints, sample['fingertip_target_m'], 'right', 0.,
                              sample['approach'], checker, ready)
        rows.append({**sample, 'solution': answer,
                     'covered': answer['converged'] and answer['clearance_passed']})
    return {'schema': 'nist_single_arm_workspace.v1', 'task_id': task_id,
        'family': 'mechanical_assembly', 'configuration_id': 'single_arm', 'release': str(release),
        'source_files': {name: digest(Path(release)/name) for name in
                         ('scene.json', 'task.json', 'initial_state.json', 'release_manifest.json')},
        'configuration_binding': {'robot_sha256': content_digest(robot),
                                  'initial_state_sha256': content_digest(initial)},
        'asset_dependencies': checker.dependencies, 'actual_collision_shape_count': len(checker.robot_shapes['right']),
        'ready_clearance': ready_check, 'stand_clearance': stands, 'samples': rows,
        'sample_count': len(rows), 'covered_count': sum(row['covered'] for row in rows),
        'passed': all(row['covered'] for row in rows) and ready_check['clear'] and stands['clear'],
        'native_steps': 0, 'manipulation_qualified': False, 'continuous_path_qualified': False,
        'scope': 'Discrete source/fixture approaches with original single-arm USD collider bounds. No grasp or continuous path.'}


def job(args):
    task_id, reference, cid, output = args
    release = Path(reference['path'])
    checked(release/'release_manifest.json', reference['manifest_sha256'])
    report = single(release, task_id) if cid == 'single_arm' else run('mechanical_assembly', task_id, cid, release)
    path = Path(output)/(task_id+'_'+cid+'.json')
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return {'family': 'mechanical_assembly', 'task_id': task_id, 'configuration_id': cid,
            'report_path': str(path), 'report_sha256': digest(path), 'passed': report['passed'],
            'sample_count': report['sample_count'], 'covered_count': report['covered_count'],
            'configuration_binding': report['configuration_binding'], 'source_files': report['source_files']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-index', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence belongs under shared datasets')
    if args.workers < 1:
        raise ValueError('Positive worker count required')
    index = load_json(args.base_index)
    if index.get('complete') is not True:
        raise ValueError('Complete candidate base index required')
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [(row['task_id'], row['source_release'], cid, str(args.output)) for row in index['tasks']
            for cid in ('single_arm', 'dual_same_side', 'dual_opposed')]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(job, args) for args in jobs]):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: (row['task_id'], row['configuration_id']))
    report = {'schema': 'nist_usability_workspace_matrix.v1', 'rows': rows,
        'passed': all(row['passed'] for row in rows), 'variant_count': len(rows),
        'passed_count': sum(row['passed'] for row in rows),
        'sample_count': sum(row['sample_count'] for row in rows),
        'covered_count': sum(row['covered_count'] for row in rows),
        'base_index': {'path': str(args.base_index.resolve()), 'sha256': digest(args.base_index)},
        'native_steps': 0, 'manipulation_qualified': False, 'continuous_path_qualified': False}
    with (args.output/'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
