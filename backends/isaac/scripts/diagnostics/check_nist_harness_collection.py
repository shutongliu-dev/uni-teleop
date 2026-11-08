"""Static full-geometry packing witness for two intact NIST harness leads.

Author-only transforms preserve every physical joint anchor and fixed frame.
This never steps a simulator or establishes a removal/grasp trajectory.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from bench.paths import load_json, OUTPUT_ROOT
from scenes.workrooms.mechanical_assembly.nist_assets import _q, _r


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def segment_distance(a, b, c, d):
    u, v, w = b-a, d-c, a-c
    aa, bb, cc, dd, ee = u@u, u@v, v@v, u@w, v@w
    den = aa*cc-bb*bb
    s = np.clip((bb*ee-cc*dd)/den, 0., 1.) if den > 1e-20 else 0.
    t = (bb*s+ee)/cc
    if t < 0:
        t, s = 0., np.clip(-dd/aa, 0., 1.)
    elif t > 1:
        t, s = 1., np.clip((bb-dd)/aa, 0., 1.)
    return float(np.linalg.norm(w+s*u-t*v))


def witness(asset, output):
    asset, output = Path(asset).resolve(), Path(output).resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Evidence belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    metadata = load_json(asset/'metadata.json')
    if metadata['task_id'] != 'nist_wire_harness':
        raise ValueError('Expected the two-lead harness task')
    source = Usd.Stage.Open(str(asset/'task.usda'))
    objects = {o['id']: o for o in metadata['objects']}
    length = objects['red_link_000']['analytic_geometry']['cylinder_length_m']
    wire_radius = objects['red_link_000']['analytic_geometry']['radius_m']
    turn_radius = length/(2*math.sin(math.pi/8))
    tray = metadata['fixture_metadata']['collection_tray']
    tray_center = np.array(tray['initial_position_m'])
    # Actual authored Cube floor/wall dimensions, rather than a copied old bin.
    boxes = []
    for path in tray['collider_paths']:
        prim = source.GetPrimAtPath(path)
        if not prim.IsA(UsdGeom.Cube):
            raise ValueError('Expected exact box tray solids')
        transform = UsdGeom.Xformable(prim).GetLocalTransformation()
        corners = np.array([transform.Transform(Gf.Vec3d(*p)) for p in itertools.product([-.5,.5], repeat=3)])+tray_center
        boxes.append(np.array([corners.min(0), corners.max(0)]))
    floor_z = boxes[0][1, 2]
    low = np.array([boxes[2][1, 0], boxes[4][1, 1], floor_z])
    high = np.array([boxes[1][0, 0], boxes[3][0, 1], floor_z+1.])
    node_z = floor_z+wire_radius+.0003
    poses, lines = {}, {}
    for color, shift_x, shift_y in [('red', -.115, -.030), ('blue', -.083, .030)]:
        start, cy = tray_center[0]+shift_x, tray_center[1]+shift_y
        cx = start+13*length
        nodes = [[start+i*length, cy-turn_radius, node_z] for i in range(14)]
        nodes += [[cx+turn_radius*math.cos(t), cy+turn_radius*math.sin(t), node_z]
                  for t in np.linspace(-math.pi/2, math.pi/2, 5)[1:]]
        nodes += [[cx-i*length, cy+turn_radius, node_z] for i in range(1, 14)]
        nodes = np.array(nodes)
        if len(nodes) != 31 or not np.allclose(np.linalg.norm(np.diff(nodes, axis=0), axis=1), length, atol=1e-12):
            raise ValueError('Packing may not shorten or stretch any wire link')
        lines[color] = nodes.tolist()
        for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
            angle = math.atan2(b[1]-a[1], b[0]-a[0])
            poses[f'{color}_link_{i:03}'] = [*((a+b)/2), 0., 0., math.sin(angle/2), math.cos(angle/2)]
        for side, index in [('a', 0), ('b', -1)]:
            name = f'{color}_end_{side}'
            link = f'{color}_link_{0 if side == "a" else 29:03}'
            rotation = _r(poses[link][3:])@_r(objects[link]['initial_pose_xyzw'][3:]).T@_r(objects[name]['initial_pose_xyzw'][3:])
            joint = UsdPhysics.Joint(source.GetPrimAtPath(metadata['joints'][name+'_wire']))
            anchor = np.array(joint.GetLocalPos0Attr().Get())
            poses[name] = [*(nodes[index]-rotation@anchor), *_q(rotation)]
    bounds, segments = {}, {}
    for name, obj in objects.items():
        p, rotation = np.array(poses[name][:3]), _r(poses[name][3:])
        world_parts = []
        for path in obj['collider_paths']:
            prim = source.GetPrimAtPath(path)
            if prim.IsA(UsdGeom.Capsule):
                capsule = UsdGeom.Capsule(prim)
                axis = rotation[:, 0]*capsule.GetHeightAttr().Get()/2
                endpoints = np.array([p-axis, p+axis])
                radius = capsule.GetRadiusAttr().Get()
                segments[name] = (endpoints, radius)
                world_parts.append(np.array([endpoints.min(0)-radius, endpoints.max(0)+radius]))
            elif prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                local = UsdGeom.Xformable(prim).GetLocalTransformation()
                points = np.array([local.Transform(Gf.Vec3d(*v)) for v in mesh.GetPointsAttr().Get()])
                world_parts.append(points@rotation.T+p)
            else:
                raise ValueError('Unexpected actor collision type: '+prim.GetTypeName())
        points = np.concatenate(world_parts)
        bounds[name] = [points.min(0).tolist(), points.max(0).tolist()]
    filtered, joints = set(), []
    for name, path in metadata['joints'].items():
        joint = UsdPhysics.Joint(source.GetPrimAtPath(path))
        anchors, frames, names = [], [], []
        for i in [0, 1]:
            body = str(getattr(joint, f'GetBody{i}Rel')().GetTargets()[0]).split('/')[-1]
            names.append(body)
            rotation = _r(poses[body][3:])
            anchors.append(np.array(poses[body][:3])+rotation@np.array(getattr(joint, f'GetLocalPos{i}Attr')().Get()))
            q = getattr(joint, f'GetLocalRot{i}Attr')().Get()
            frames.append(rotation@_r([*q.GetImaginary(), q.GetReal()]))
        filtered.add(frozenset(names))
        angle = math.acos(float(np.clip((np.trace(frames[0].T@frames[1])-1)/2, -1, 1))) if joint.GetPrim().IsA(UsdPhysics.FixedJoint) else 0.
        joints.append({'id': name, 'anchor_gap_m': float(np.linalg.norm(anchors[0]-anchors[1])), 'fixed_frame_angle_rad': angle})
    pairs = []
    for a, b in itertools.combinations(objects, 2):
        if frozenset([a, b]) in filtered:
            continue
        if a in segments and b in segments:
            sa, ra = segments[a]; sb, rb = segments[b]
            gap = segment_distance(*sa, *sb)-ra-rb
        else:
            al, ah = np.array(bounds[a]); bl, bh = np.array(bounds[b])
            gap = float(np.linalg.norm(np.maximum(np.maximum(al-bh, bl-ah), 0.)))
        pairs.append({'pair': [a,b], 'clearance_m': gap})
    all_bounds = np.array(list(bounds.values()))
    actor_low, actor_high = all_bounds[:, 0].min(0), all_bounds[:, 1].max(0)
    removal = metadata['task_parameters']['removal']
    removal_low = np.array(removal['center_m'])-removal['half_extents_m']
    removal_high = np.array(removal['center_m'])+removal['half_extents_m']
    checks = {'all_64_complete_actors': len(objects) == len(poses) == 64,
        'full_geometry_inside_real_tray': bool(np.all(actor_low >= low-1e-8) and np.all(actor_high[:2] <= high[:2]+1e-8)),
        'full_geometry_inside_scored_removal_region': bool(np.all(actor_low >= removal_low-1e-8) and np.all(actor_high <= removal_high+1e-8)),
        'all_unfiltered_actor_pairs_separate': min(x['clearance_m'] for x in pairs) > 1e-6,
        'original_joint_anchors_agree': max(x['anchor_gap_m'] for x in joints) < 1e-6,
        'original_fixed_joint_frames_agree': max(x['fixed_frame_angle_rad'] for x in joints) < 1e-6}
    report = {'schema': 'nist_harness_collection_geometry.v2', 'passed': all(checks.values()), 'checks': checks,
        'asset_manifest': {'path': str(asset/'manifest.json'), 'sha256': digest(asset/'manifest.json')},
        'usd_sha256': digest(asset/'task.usda'), 'metadata_sha256': digest(asset/'metadata.json'),
        'wire_length_each_m': length*30, 'wire_radius_m': wire_radius, 'all_actor_count': len(objects),
        'actor_bounds_m': bounds, 'joints': joints,
        'minimum_unfiltered_pair_clearance': min(pairs, key=lambda x:x['clearance_m']),
        'minimum_tray_plan_margin_m': float(min(*(actor_low[:2]-low[:2]), *(high[:2]-actor_high[:2]))),
        'minimum_removal_margin_m': float(min(*(actor_low-removal_low), *(removal_high-actor_high))),
        'method': 'Two complete equal-chord U leads with staggered connector ends; original physical shell meshes, exact capsule geometry, all passive anchors and fixed relative frames retained.',
        'scope': 'Static geometric existence only; no simulation, stability, hand-object operation, removal trajectory or task success.'}
    for filename, value in [('qualification.json', report), ('poses_xyzw.json', poses), ('centerlines.json', lines)]:
        (output/filename).write_text(json.dumps(value, indent=2)+'\n')
    stage = Usd.Stage.CreateNew(str(output/'author_only_collection.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    root = UsdGeom.Xform.Define(stage, metadata['root_prim_path'])
    root.GetPrim().GetReferences().AddReference(str(asset/'task.usda'), metadata['root_prim_path'])
    for name, pose in poses.items():
        obj = UsdGeom.Xformable(stage.GetPrimAtPath(metadata['actors'][name]))
        obj.ClearXformOpOrder()
        obj.AddTranslateOp(opSuffix='collectionWitness').Set(Gf.Vec3d(*pose[:3]))
        obj.AddOrientOp(opSuffix='collectionWitness').Set(Gf.Quatf(pose[6], Gf.Vec3f(*pose[3:6])))
    stage.GetRootLayer().Save()
    (output/'files_sha256.json').write_text(json.dumps({p.name:digest(p) for p in output.iterdir() if p.is_file()}, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = witness(args.asset, args.output)
    print(json.dumps({k:result[k] for k in ['passed','minimum_tray_plan_margin_m','minimum_unfiltered_pair_clearance']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
