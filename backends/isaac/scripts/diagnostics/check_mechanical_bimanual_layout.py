"""CPU discrete IK and actual-USD bounds for the official left/right cooperative layout.

No native worker, controller, trajectory or grasp policy runs here.  Each arm's
own supply and the shared tray are sampled independently, with the other arm
at its ready pose.  Additional paired wrist samples test separated endpoints;
they are not physical grasps or a manipulation acceptance criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.mechanical_assembly import cooperative_robot_layout as layout
from scenes.workrooms.mechanical_assembly import robot_layout as source_layout

PROJECT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(sample_set='approach', asset_manifest=None):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from scenes.workrooms.mechanical_assembly.scene import environment_reference
    from scenes.workrooms.mechanical_assembly.desktop_dressing import task_layer
    from scenes.workrooms.mechanical_assembly.cooperative_assets import ASSET_ROOT, ROOT
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot

    source_snapshot = verify_if_snapshot(PROJECT,
        required_modules=('scenes.workrooms.mechanical_assembly.cooperative_robot_layout',
                          'scenes.workrooms.mechanical_assembly.robot_layout'),
        entrypoint_file=Path(__file__))

    from bench.control.factory import public_robot_geometry, robot_configuration
    from bench.geometry import rpy_matrix, transform
    robot, source_initial, ready = layout.robot_inputs(layout.MIRRORED_LAYOUT_ID)
    geometry = public_robot_geometry(robot_configuration({'robot': robot}))
    models = geometry.sides
    assets = {}
    source_root = Path(robot['assets_root'])/'robots/assemblies/dual_fr3_sharpa_w01/isaacsim_6_1_v2'
    for side in layout.MIRRORED_SIDES:
        manifest_path = source_root/f'USD_IMPORT_{side}.json'
        imported = load_json(manifest_path)
        for path, expected in imported['derived_files'].items():
            if digest(path) != expected:
                raise ValueError('Changed source asset: '+path)
        assets[side] = {'import_manifest': str(manifest_path), 'import_manifest_sha256': digest(manifest_path),
                        'root_usd': imported['usd_path'], 'root_usd_sha256': imported['usd_sha256'],
                        'derived_files': imported['derived_files']}

    def solve(side, point):
        model = models[side]
        target = (np.linalg.inv(model.arm_base)
                  @ transform(point, rpy_matrix(layout.placements(layout.MIRRORED_LAYOUT_ID)[side]['wrist_probe_rpy_rad']))
                  @ np.linalg.inv(model.wrist_offset))
        return model.arm.inverse(target, model.arm_tip,
            {name: source_initial['joint_positions_rad'][name] for name in model.arm.names},
            iterations=180, multi_starts=0, position_tolerance=.0001, rotation_tolerance=.002)
    environment = environment_reference()
    dressing = task_layer()
    task_manifest_path = Path(asset_manifest or ASSET_ROOT/'manifest.json').resolve()
    task_manifest = load_json(task_manifest_path)
    for path, expected in ((environment['usd_path'], environment['usd_sha256']),
                           (dressing['usd_path'], dressing['sha256']),
                           (task_manifest['usd_path'], task_manifest['usd_sha256'])):
        if digest(path) != expected:
            raise ValueError('Changed source asset: '+path)
    # The anonymous stage is never saved and is the sole owner of pose probes.
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    for name, path, source in (
        ('Environment', environment['usd_path'], '/World'),
        ('Desktop', dressing['usd_path'], dressing['source_prim_path']),
        ('Task', task_manifest['usd_path'], ROOT),
    ):
        UsdGeom.Xform.Define(stage, '/World/'+name).GetPrim().GetReferences().AddReference(path, source)
    roots = {}
    for side in layout.MIRRORED_SIDES:
        root = UsdGeom.Xform.Define(stage, '/World/Robots/'+side).GetPrim()
        root.GetReferences().AddReference(assets[side]['root_usd'], '/fr3_sharpa_'+side)
        roots[side] = root
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)

    def bounds(prim):
        value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        result = np.asarray([value.GetMin(), value.GetMax()])
        if not np.isfinite(result).all() or (result[1] < result[0]).any():
            raise ValueError('Invalid bound: '+str(prim.GetPath()))
        return result

    def collision_shapes(root):
        return [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Boundable)]

    obstacles = []
    for name in ('Environment', 'Desktop', 'Task'):
        for prim in collision_shapes(stage.GetPrimAtPath('/World/'+name)):
            # Old frozen task stands belong to the historical opposed layout;
            # new task-owned stand boxes are checked explicitly below.
            if 'robot_stand_' in str(prim.GetPath()):
                continue
            obstacles.append((str(prim.GetPath()), bounds(prim)))
    shapes = {side: collision_shapes(root) for side, root in roots.items()}
    obstacle_boxes = np.asarray([box for _, box in obstacles])

    def place(side, initial):
        model = models[side]
        joints = model.checked_joints({name: initial['joint_positions_rad'][name] for name in model.names}, enforce_limits=True)
        poses = model.arm.forward(joints, model.arm_base)
        poses.update(model.hand.forward(joints, poses[model.arm_tip] @ model.wrist_offset))
        targets = [(prim, poses[prim.GetName()])
                   for prim in Usd.PrimRange(roots[side])
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
        return np.asarray([bounds(prim) for prim in shapes[side]])

    def nearest(first, second, first_names, second_names):
        delta = np.maximum(np.maximum(first[:, None, 0]-second[None, :, 1],
                                      second[None, :, 0]-first[:, None, 1]), 0.)
        gaps = np.linalg.norm(delta, axis=2)
        i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
        return {'minimum_aabb_gap_m': float(gaps[i, j]),
                'first_shape': first_names[i], 'second_shape': second_names[j],
                'positive_separation': bool(gaps[i, j] > 0.),
                'clearance_over_5mm': bool(gaps[i, j] > .005)}

    shape_names = {side: [str(prim.GetPath()) for prim in rows] for side, rows in shapes.items()}
    obstacle_names = [name for name, _ in obstacles]
    ready_bounds = {side: place(side, source_initial) for side in layout.MIRRORED_SIDES}

    def checks(boxes):
        return {'environment': {side: nearest(boxes[side], obstacle_boxes,
                    shape_names[side], obstacle_names) for side in layout.MIRRORED_SIDES},
                'between_robots': nearest(boxes['right'], boxes['left'],
                    shape_names['right'], shape_names['left'])}

    if sample_set not in ('approach', 'exploratory_low_wrist'):
        raise ValueError('Unknown endpoint sample set')
    # Public wrists are not grasp centres.  The deliberately low exploratory
    # set retains its observed task-shape overlaps; it is not an approach plan.
    heights = (1.16, 1.20) if sample_set == 'approach' else (1.00, 1.08, 1.16)
    samples = []
    for side, sign in (('right', -1), ('left', 1)):
        for region, x, y in (('own_supply', .65, sign*.205),
                             ('shared_tray', .49, 0.),
                             ('near_shared_assembly', .40, sign*.10)):
            for z in heights:
                point = [x, y, z]
                solved = solve(side, point)
                initial = {'joint_positions_rad': {
                    **source_initial['joint_positions_rad'], **solved['joint_positions_rad']}}
                boxes = dict(ready_bounds)
                boxes[side] = place(side, initial)
                samples.append({'side': side, 'region': region, 'wrist_position_m': point,
                    'wrist_rpy_rad': layout.placements(layout.MIRRORED_LAYOUT_ID)[side]['wrist_probe_rpy_rad'],
                    'solver': solved, 'checks': checks(boxes)})
                place(side, source_initial)
    paired = []
    for right_z, left_z in ((1.08, 1.16), (1.16, 1.08)):
        boxes, pair = {}, {}
        for side, y, z in (('right', -.12, right_z), ('left', .12, left_z)):
            point = [.40, y, z]
            solved = solve(side, point)
            initial = {'joint_positions_rad': {
                **source_initial['joint_positions_rad'], **solved['joint_positions_rad']}}
            boxes[side] = place(side, initial)
            pair[side] = {'wrist_position_m': point, 'solver': solved}
        paired.append({'robots': pair, 'checks': checks(boxes)})
    ready_checks = checks(ready_bounds)
    stands = layout.stand_geometry(layout.MIRRORED_LAYOUT_ID)
    # Stand-table feet have geometric XY clearance; floor contact is intentional.
    stand_checks = []
    for side, stand in stands.items():
        stand_boxes = np.asarray([[np.asarray(item['position_m'])-np.asarray(item['size_m'])/2,
                                  np.asarray(item['position_m'])+np.asarray(item['size_m'])/2]
                                 for item in stand['boxes']])
        names = [side+'/'+item['name'] for item in stand['boxes']]
        nonfloor = [(name, box) for name, box in obstacles if box[1][2] > .051]
        stand_checks.append({'side': side, 'fixed_obstacles_except_floor': nearest(stand_boxes,
            np.asarray([box for _, box in nonfloor]), names, [name for name, _ in nonfloor])})
    all_checks = [ready_checks, *[row['checks'] for row in samples], *[row['checks'] for row in paired]]
    all_separated = all(
        row['between_robots']['clearance_over_5mm']
        and all(value['clearance_over_5mm'] for value in row['environment'].values())
        for row in all_checks)
    all_ik = all(row['solver']['converged'] for row in samples) and all(
        row['solver']['converged'] for pair in paired for row in pair['robots'].values())
    git_head = None if source_snapshot is not None else subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    source_snapshot_after = verify_if_snapshot(PROJECT,
        required_modules=('scenes.workrooms.mechanical_assembly.cooperative_robot_layout',
                          'scenes.workrooms.mechanical_assembly.robot_layout'),
        entrypoint_file=Path(__file__))
    return {'schema': 'mechanical_cooperative_left_right_cpu.v1',
        'layout_id': layout.MIRRORED_LAYOUT_ID, 'project': str(PROJECT),
        'sample_set': sample_set, 'independent_wrist_heights_m': list(heights),
        'git_head': git_head, 'source_snapshot': source_snapshot_after,
        'evidence_scope': 'frozen_source_cpu_probe' if source_snapshot is not None else 'exploratory_editable_source_cpu_probe',
        'source_sha256': {str(path): digest(path) for path in (
            Path(__file__), Path(layout.__file__), Path(source_layout.__file__))},
        'robot_assets': assets, 'robot_configuration': robot, 'ready_design': ready, 'environment': environment, 'desktop': dressing,
        'task_manifest': task_manifest, 'task_manifest_path': str(task_manifest_path),
        'task_manifest_sha256': digest(task_manifest_path), 'robot_placements': layout.placements(layout.MIRRORED_LAYOUT_ID),
        'stand_geometry': stands, 'stand_checks': stand_checks,
        'actual_robot_collision_shape_counts': {side: len(rows) for side, rows in shapes.items()},
        'obstacle_shape_count': len(obstacles), 'ready_checks': ready_checks,
        'individual_samples': samples, 'paired_samples': paired,
        'all_endpoint_ik_converged': bool(all_ik),
        'all_robot_endpoint_bounds_clear_over_5mm': bool(all_separated),
        'passed': bool(all_ik and all_separated and all(
            row['fixed_obstacles_except_floor']['clearance_over_5mm'] for row in stand_checks)),
        'model_calls': 0, 'native_runtime_started': False, 'manipulation_qualified': False,
        'scope': 'Actual imported USD collision-shape conservative world AABBs at discrete original FK/IK poses; no native cooked-shape, self-collision, swept path, finger grasp, object operation or full manipulation qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-set', choices=('approach', 'exploratory_low_wrist'), default='approach')
    parser.add_argument('--asset-manifest', type=Path, help='Task asset manifest; no asset is changed')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        parser.error('Write experiment evidence under shared datasets')
    if output.exists():
        raise FileExistsError('Preserve existing evidence: '+str(output))
    result = run(args.sample_set, args.asset_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in (
        'passed', 'all_endpoint_ik_converged', 'all_robot_endpoint_bounds_clear_over_5mm',
        'actual_robot_collision_shape_counts', 'obstacle_shape_count')}))
    print(json.dumps({'report': str(output), 'sha256': digest(output)}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
