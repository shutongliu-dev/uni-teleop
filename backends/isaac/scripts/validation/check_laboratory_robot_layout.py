"""Check discrete laboratory wrist poses against actual original USD shapes.

This CPU check never steps physics, interpolates a trajectory or moves objects.
The presentation specimens are excluded because they illustrate future targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.paths import load_json

PROJECT = Path(__file__).resolve().parents[2]


def check_endpoints(preview):
    from pxr import Gf, Usd, UsdGeom
    from scenes.workrooms.laboratory.robot_layout import (
        link_world_transforms, robot_configuration, workspace_audit)
    from scripts.assets.build_card_room import digest

    manifest = load_json(preview / 'manifest.json')
    if digest(manifest['scene']) != manifest['scene_sha256']:
        raise ValueError('Laboratory preview USD hash changed')
    if digest(preview / 'scene_description.json') != manifest['description_sha256']:
        raise ValueError('Laboratory preview description changed')
    for path, record in manifest['dependencies'].items():
        if digest(path) != record['sha256']:
            raise ValueError('Laboratory preview dependency changed: ' + path)
    # A new anonymous layer owns every probe override; the published scene is
    # never edited or saved.
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/World')
    root.GetPrim().GetReferences().AddReference(manifest['scene'], '/World')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             ['default', 'render', 'proxy', 'guide'], False, True)

    def bounds(prim):
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        value = np.asarray([box.GetMin(), box.GetMax()])
        if not np.isfinite(value).all() or (value[1] < value[0]).any():
            raise ValueError('Invalid USD bound: ' + str(prim.GetPath()))
        return value

    obstacles = []
    excluded = ('robot_stand_', 'preview_anchor_', 'preview_samples_')
    for group in stage.GetPrimAtPath('/World/Geometry').GetChildren():
        if group.GetName().startswith(excluded):
            continue
        for prim in Usd.PrimRange(group, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Boundable):
                obstacles.append((str(prim.GetPath()), bounds(prim)))
    obstacle_boxes = np.asarray([box for _, box in obstacles])
    robot = robot_configuration()
    audit = workspace_audit(robot)
    robot_root = stage.GetPrimAtPath('/World/TaskRobot')
    shapes = [prim for prim in Usd.PrimRange(robot_root, Usd.TraverseInstanceProxies())
              if prim.IsA(UsdGeom.Boundable)]
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    for sample in audit['samples']:
        poses = link_world_transforms(robot, sample)
        targets = [(prim, poses[prim.GetName()])
                   for prim in Usd.PrimRange(robot_root)
                   if prim.GetName() in poses and prim.IsA(UsdGeom.Xform)]
        for prim, matrix in sorted(targets, key=lambda item: str(item[0].GetPath()).count('/')):
            parent = transforms.GetLocalToWorldTransform(prim.GetParent())
            local = Gf.Matrix4d(matrix.T.tolist()) * parent.GetInverse()
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            # Reuse the same override operation on subsequent samples.
            xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, 'endpointProbe').Set(local)
            transforms.Clear()
        for prim, matrix in targets:
            np.testing.assert_allclose(
                np.asarray(transforms.GetLocalToWorldTransform(prim)).T,
                matrix, atol=1.e-7, rtol=0.)
        cache.Clear()
        actual = np.asarray([bounds(prim) for prim in shapes])
        delta = np.maximum(np.maximum(
            actual[:, None, 0] - obstacle_boxes[None, :, 1],
            obstacle_boxes[None, :, 0] - actual[:, None, 1]), 0.)
        gaps = np.linalg.norm(delta, axis=2)
        i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
        sample['actual_usd_environment_clearance'] = {
            'minimum_aabb_gap_m': float(gaps[i, j]),
            'robot_shape': str(shapes[i].GetPath()), 'environment_shape': obstacles[j][0],
            'clearance_over_5mm': bool(gaps[i, j] > .005),
        }
    results = [item['actual_usd_environment_clearance'] for item in audit['samples']]
    audit.update(
        preview_manifest=str(preview / 'manifest.json'),
        preview_manifest_sha256=digest(preview / 'manifest.json'),
        actual_usd_shape_count=len(shapes), environment_shape_count=len(obstacles),
        all_endpoints_clear_environment=all(item['clearance_over_5mm'] for item in results),
        minimum_environment_gap_m=min(item['minimum_aabb_gap_m'] for item in results),
        exclusions=['Stand/base mounting contact', 'Presentation sample objects',
                    'Self-collision and trajectories are not checked'],
        scope='Discrete source FK/IK and conservative actual-USD shape bounds against fixed room only; no native motion, grasp or task-success qualification.')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    import scenes.workrooms.laboratory.robot_layout
    snapshot = verify_if_snapshot(PROJECT,
        required_modules=('scenes.workrooms.laboratory.robot_layout',),
        entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Check from complete frozen source')
    if args.output.exists():
        raise FileExistsError('Preserve previous evidence: ' + str(args.output))
    result = check_endpoints(args.preview.resolve())
    snapshot = verify_if_snapshot(PROJECT,
        required_modules=('scenes.workrooms.laboratory.robot_layout',),
        entrypoint_file=Path(__file__))
    result['source_snapshot'] = snapshot
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'sample_count', 'all_samples_converged', 'all_endpoints_clear_environment',
        'minimum_environment_gap_m', 'maximum_position_error_m')}))
    return 0 if result['all_endpoints_clear_environment'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
