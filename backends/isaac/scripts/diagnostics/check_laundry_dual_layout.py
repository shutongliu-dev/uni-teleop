"""Discrete dual-hand IK and imported collision-bound checks for laundry.

Targets come from the existing successful cloth inputs and slow engineering
paths, but the robot follows no trajectory and grasps nothing in this probe.
The actual soft-curl middle fingertip, rather than the wrist, is placed near
each cloth corner. Original robot USD collider bounds are transformed by FK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from bench.paths import DATA_ROOT, load_json, OUTPUT_ROOT
from scenes.workrooms.laundry_room import robot_layout as layout

PROJECT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/laundry_tasks_20261021/releases')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nearest(first, second, first_names, second_names):
    first, second = np.asarray(first), np.asarray(second)
    delta = np.maximum(np.maximum(first[:, None, 0]-second[None, :, 1],
                                  second[None, :, 0]-first[:, None, 1]), 0.)
    gaps = np.linalg.norm(delta, axis=2)
    i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
    return {'minimum_aabb_gap_m': float(gaps[i, j]),
            'first_shape': first_names[i], 'second_shape': second_names[j],
            'positive_separation': bool(gaps[i, j] > 0.),
            'clearance_over_5mm': bool(gaps[i, j] > .005),
            'aabb_overlap_pair_count': int((gaps == 0).sum())}


def task_samples(release, task_id):
    """Read immutable geometry and enumerate both corner approach/path/retreat."""
    from scripts.validation.laundry_engineering_motion import build_motion

    release = Path(release).resolve()
    task, scene = (load_json(release/name) for name in ('task.json', 'scene.json'))
    if task['id'] != task_id:
        raise ValueError('Wrong task release: '+str(release))
    if task_id not in ('towel_folding', 'towel_stacking'):
        return surface_samples(release, task, scene)
    if task_id == 'towel_folding':
        options = {'fold_trajectory': 'bezier', 'fold_peak_height_m': .16,
                   'fold_grip_span_m': [.04, .04]}
        times = [0., *np.linspace(.8, 7.2, 17).tolist(), 9.5, 10., 10.5, 10.75]
    else:
        options = {'stack_trajectory': 'low-clearance', 'stack_grip_span_m': [.04, .04],
                   'stack_grip_layout': 'crease-edge'}
        times = [0., *np.linspace(.6, 7.8, 19).tolist(), 8.5, 9., 9.5, 9.75]
    motion = build_motion(task, scene, **options)
    samples = []
    for clearance in (.12, .06):
        samples.append({'phase': 'approach', 'native_reference_time_s': 0.,
                        'fingertip_clearance_m': clearance,
                        'corners_m': {'right': motion.pose(0, 0.)[0], 'left': motion.pose(1, 0.)[0]}})
    for t in times:
        samples.append({'phase': 'transport_or_fold' if 0. < t < motion.release_time_s else
                        'withdrawal' if t > motion.release_time_s else 'near_cloth_or_release',
                        'native_reference_time_s': t,
                        'fingertip_clearance_m': layout.workspace_contract()['probe_fingertip_clearance_m'],
                        'corners_m': {'right': motion.pose(0, t)[0], 'left': motion.pose(1, t)[0]}})
    return samples, {'release': str(release), 'task_id': task_id,
                     'file_sha256': {name: digest(release/name) for name in
                                     ('release_manifest.json', 'task.json', 'scene.json')},
                     'motion_options': options,
                     'observation_correction_applied': False,
                     'scope': 'Nominal corner path from frozen initial cloth; live sliding and grasp-induced deformation are not predicted.'}


def surface_samples(release, task, scene):
    """Near-surface workspace probes for flattening and folded garments."""
    row = next(value for value in task['asset_metadata']['cloths'] if value['id'] == 'towel')
    points = np.asarray(row['initial_vertices_m'])
    lo, hi = points.min(0), points.max(0)
    target_xy = np.asarray(task['asset_metadata']['target_center_xy_m'])
    delta = np.r_[target_xy-(lo[:2]+hi[:2])/2, 0.]
    if len(task['asset_metadata']['cloths']) == 2:
        base = np.asarray(task['asset_metadata']['cloths'][1]['initial_vertices_m'])
        delta[2] = base[:, 2].max()-lo[2]+row['physical_thickness_m']
    samples = []
    for fraction in (.10, .90):
        corners = {}
        for side, edge in (('left', .90), ('right', .10)):
            desired = lo[:2]+[fraction*(hi[0]-lo[0]), edge*(hi[1]-lo[1])]
            distance = np.linalg.norm(points[:, :2]-desired, axis=1)
            closest = np.flatnonzero(distance <= distance.min()+.005)
            index = closest[np.argmax(points[closest, 2])]
            corners[side] = points[index]
        for phase, alpha, lift, clearance in (
                ('approach', 0., 0., .12), ('near_cloth', 0., 0., .015),
                ('lift', 0., .08, .015),
                *[('transport', float(t), .08, .015) for t in np.linspace(.1, .9, 9)],
                ('placement', 1., 0., .015), ('withdrawal', 1., 0., .12)):
            samples.append({'phase': phase, 'native_reference_time_s': None,
                'fingertip_clearance_m': clearance,
                'corners_m': {side: (point+alpha*delta+[0.,0.,lift]).tolist()
                              for side, point in corners.items()}})
    return samples, {'release': str(release), 'task_id': task['id'],
        'file_sha256': {name: digest(release/name) for name in
                       ('release_manifest.json', 'task.json', 'scene.json')},
        'motion_options': {'probe': 'two_longitudinal_surface_sections', 'lift_m': .08},
        'observation_correction_applied': False,
        'scope': 'Discrete native-mesh corner workspace and placement probes, not a grasp policy or swept-path check.'}


def run(fold_release, stack_release, extra_releases=()):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from bench.control.factory import public_robot_geometry, robot_configuration
    from scenes.workrooms.laundry_room.scene import environment_reference

    robot, source_initial, ready = layout.robot_inputs()
    models = public_robot_geometry(robot_configuration({'robot': robot})).sides
    assets = {}
    source_root = Path(robot['assets_root'])/'robots/assemblies/dual_fr3_sharpa_w01/isaacsim_6_1_v2'
    for side in layout.SIDES:
        manifest_path = source_root/f'USD_IMPORT_{side}.json'
        imported = load_json(manifest_path)
        for path, expected in {imported['usd_path']: imported['usd_sha256'], **imported['derived_files']}.items():
            if digest(path) != expected:
                raise ValueError('Changed original robot asset: '+path)
        assets[side] = {'import_manifest': str(manifest_path), 'import_manifest_sha256': digest(manifest_path),
                        'root_usd': imported['usd_path'], 'root_usd_sha256': imported['usd_sha256'],
                        'derived_files': imported['derived_files']}
    environment = environment_reference()
    if digest(environment['usd_path']) != environment['usd_sha256']:
        raise ValueError('Changed fixed laundry room')
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    UsdGeom.Xform.Define(stage, '/World/Environment').GetPrim().GetReferences().AddReference(
        environment['usd_path'], environment['source_prim'])
    roots = {}
    for side in layout.SIDES:
        root = UsdGeom.Xform.Define(stage, '/World/Robots/'+side).GetPrim()
        root.GetReferences().AddReference(assets[side]['root_usd'], '/fr3_sharpa_'+side)
        roots[side] = root
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)

    def bounds(prim):
        value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        result = np.asarray([value.GetMin(), value.GetMax()])
        if not np.isfinite(result).all() or (result[1] < result[0]).any():
            raise ValueError('Invalid collision bound: '+str(prim.GetPath()))
        return result

    def shapes(root):
        return [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Boundable)]

    obstacles = [(str(prim.GetPath()), bounds(prim))
                 for prim in shapes(stage.GetPrimAtPath('/World/Environment'))]
    robot_shapes = {side: shapes(root) for side, root in roots.items()}
    shape_names = {side: [str(prim.GetPath()) for prim in rows] for side, rows in robot_shapes.items()}
    obstacle_names = [name for name, _ in obstacles]
    obstacle_boxes = np.asarray([box for _, box in obstacles])
    stands = layout.stand_geometry()
    stand_boxes, stand_names = {}, {}
    for side, stand in stands.items():
        stand_boxes[side] = np.asarray([[np.asarray(item['position_m'])-np.asarray(item['size_m'])/2,
                                         np.asarray(item['position_m'])+np.asarray(item['size_m'])/2]
                                        for item in stand['boxes']])
        stand_names[side] = [side+'/'+item['name'] for item in stand['boxes']]

    def place(side, joints):
        model = models[side]
        own = model.checked_joints({name: joints[name] for name in model.names}, enforce_limits=True)
        poses = model.arm.forward(own, model.arm_base)
        poses.update(model.hand.forward(own, poses[model.arm_tip] @ model.wrist_offset))
        targets = [(prim, poses[prim.GetName()]) for prim in Usd.PrimRange(roots[side])
                   if prim.IsA(UsdGeom.Xform) and prim.GetName() in poses]
        for prim, matrix in sorted(targets, key=lambda row: str(row[0].GetPath()).count('/')):
            parent = transforms.GetLocalToWorldTransform(prim.GetParent())
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'endpointProbe').Set(
                Gf.Matrix4d(matrix.T.tolist())*parent.GetInverse())
            transforms.Clear()
        for prim, matrix in targets:
            np.testing.assert_allclose(np.asarray(transforms.GetLocalToWorldTransform(prim)).T,
                                       matrix, atol=1e-7, rtol=0.)
        cache.Clear()
        return np.asarray([bounds(prim) for prim in robot_shapes[side]])

    def checks(boxes):
        return {'environment': {side: nearest(boxes[side], obstacle_boxes,
                    shape_names[side], obstacle_names) for side in layout.SIDES},
                'other_stand': {side: nearest(boxes[side], stand_boxes['right' if side == 'left' else 'left'],
                    shape_names[side], stand_names['right' if side == 'left' else 'left']) for side in layout.SIDES},
                'between_robots': nearest(boxes['right'], boxes['left'],
                    shape_names['right'], shape_names['left'])}

    ready_boxes = {side: place(side, source_initial['joint_positions_rad']) for side in layout.SIDES}
    ready_checks = checks(ready_boxes)
    reports, releases = {}, []
    tasks = [('towel_folding', fold_release), ('towel_stacking', stack_release)]
    tasks.extend((load_json(Path(path)/'task.json')['id'], Path(path))
                 for path in extra_releases)
    for task_id, release in tasks:
        samples, binding = task_samples(release, task_id)
        releases.append(binding)
        rows = []
        for sample in samples:
            pair, boxes = {}, {}
            for side in layout.SIDES:
                model = models[side]
                wrist = layout.hand_probe_transform(side, sample['corners_m'][side],
                    fingertip_clearance_m=sample['fingertip_clearance_m'])
                target = np.linalg.inv(model.arm_base) @ wrist @ np.linalg.inv(model.wrist_offset)
                solved = model.arm.inverse(target, model.arm_tip,
                    {name: source_initial['joint_positions_rad'][name] for name in model.arm.names},
                    iterations=180, multi_starts=0, position_tolerance=.0001, rotation_tolerance=.002)
                joints = {**source_initial['joint_positions_rad'], **solved['joint_positions_rad']}
                boxes[side] = place(side, joints)
                own = {name: joints[name] for name in model.names}
                public = model.public_geometry(own)
                actual_wrist = model.arm.forward(own, model.arm_base)[model.arm_tip] @ model.wrist_offset
                fingertip = (actual_wrist @ [*public['fingertip_positions_m']['middle'], 1.])[:3]
                expected_tip = np.asarray(sample['corners_m'][side]) + [0., 0., sample['fingertip_clearance_m']]
                pair[side] = {'cloth_corner_m': sample['corners_m'][side],
                    'wrist_target_matrix': wrist.tolist(), 'actual_middle_fingertip_m': fingertip.tolist(),
                    'middle_fingertip_target_error_m': float(np.linalg.norm(fingertip-expected_tip)),
                    'minimum_arm_joint_margin_rad': float(min(min(joints[j.name]-j.lower, j.upper-joints[j.name])
                                                              for j in model.arm.active)),
                    'solver': solved,
                    'individual_checks': checks({**ready_boxes, side: boxes[side]})}
            rows.append({**sample, 'hands': pair, 'paired_checks': checks(boxes)})
        reports[task_id] = rows
    nonfloor = [(name, box) for name, box in obstacles if box[1][2] > .051]
    stand_checks = {side: nearest(stand_boxes[side], np.asarray([box for _, box in nonfloor]),
        stand_names[side], [name for name, _ in nonfloor]) for side in layout.SIDES}
    stand_checks['between_stands'] = nearest(stand_boxes['left'], stand_boxes['right'],
                                            stand_names['left'], stand_names['right'])
    all_rows = [row for rows in reports.values() for row in rows]
    all_checks = [ready_checks, *[row['paired_checks'] for row in all_rows],
                  *[hand['individual_checks'] for row in all_rows for hand in row['hands'].values()]]
    checks_clear = lambda checks: (checks['between_robots']['clearance_over_5mm'] and
        all(value['clearance_over_5mm'] for group in ('environment', 'other_stand') for value in checks[group].values()))
    failures = [{'task_id': task_id, 'sample_index': i, 'phase': row['phase'],
                 'time_s': row['native_reference_time_s'], 'hands_ik': {side: hand['solver']['converged']
                   for side, hand in row['hands'].items()}, 'paired_bounds_clear': checks_clear(row['paired_checks']),
                 'individual_bounds_clear': {side: checks_clear(hand['individual_checks']) for side, hand in row['hands'].items()}}
                for task_id, rows in reports.items() for i, row in enumerate(rows)
                if not (checks_clear(row['paired_checks']) and all(hand['solver']['converged'] and
                    checks_clear(hand['individual_checks']) for hand in row['hands'].values()))]
    all_ik = all(hand['solver']['converged'] for row in all_rows for hand in row['hands'].values())
    all_clear = all(checks_clear(row) for row in all_checks)
    passed = all_ik and all_clear and all(row['clearance_over_5mm'] for row in stand_checks.values())
    return {'schema': 'laundry_dual_robot_cpu.v1', 'layout_id': layout.LAYOUT_ID,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
        'source_sha256': {str(path.relative_to(PROJECT)): digest(path) for path in
                          (Path(__file__), Path(layout.__file__), PROJECT/'scripts/validation/laundry_engineering_motion.py')},
        'robot_configuration': robot, 'ready_design': ready, 'robot_placements': layout.placements(),
        'robot_assets': assets, 'environment': environment, 'task_sources': releases,
        'actual_robot_collision_shape_counts': {side: len(rows) for side, rows in robot_shapes.items()},
        'obstacle_shape_count': len(obstacles), 'stand_geometry': stands, 'stand_checks': stand_checks,
        'ready_checks': ready_checks, 'task_samples': reports,
        'paired_sample_count': len(all_rows), 'individual_hand_sample_count': 2*len(all_rows),
        'all_endpoint_ik_converged': bool(all_ik), 'all_robot_endpoint_bounds_clear_over_5mm': bool(all_clear),
        'failed_samples': failures, 'passed': bool(passed),
        'native_runtime_started': False, 'model_calls': 0, 'manipulation_qualified': False,
        'scope': 'Discrete endpoint IK and conservative AABBs of actual imported robot collision shapes, unchanged room colliders, and task stands. No cooked native collision, self-collision, swept-path, grasp, cloth coupling, or full operation qualification.',
        'orientation_limitation': 'Fixed downward hand posture at sampled corner positions does not reproduce the engineering attachment rotation or guarantee a viable cloth-fold grasp.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-release', type=Path, default=RELEASE_ROOT/'towel_folding/practical_v2')
    parser.add_argument('--stack-release', type=Path, default=RELEASE_ROOT/'towel_stacking/practical_v2')
    parser.add_argument('--extra-release', type=Path, action='append', default=[],
                        help='Additional flattening/garment robot releases to probe')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        parser.error('Experiment evidence belongs under shared datasets')
    if output.exists():
        raise FileExistsError('Preserve historical evidence: '+str(output))
    result = run(args.fold_release, args.stack_release, args.extra_release)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'paired_sample_count', 'individual_hand_sample_count',
        'all_endpoint_ik_converged', 'all_robot_endpoint_bounds_clear_over_5mm', 'failed_samples')}))
    print(json.dumps({'report': str(output), 'sha256': digest(output)}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
