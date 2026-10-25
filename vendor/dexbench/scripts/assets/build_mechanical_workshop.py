"""Build a new immutable empty-table mechanical workshop from frozen sources."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bench.paths import DATA_ROOT
from scripts.assets.build_card_room import dependency_inventory, digest

PROJECT = Path(__file__).resolve().parents[2]


def audit_environment(stage):
    """Check actual USD geometry, empty work volume and static ownership."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.assets.workrooms.mechanical_workshop import RESERVED_TASK_VOLUME, ROBOT_FLOOR_RESERVE
    if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Workshop must use metres and Z up')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    def box(prim):
        extent = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        return np.array([extent.GetMin(), extent.GetMax()])
    intrusions = []
    for prim in stage.GetPrimAtPath('/World/Geometry').GetChildren():
        actual = box(prim)
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite environment bounds: ' + str(prim.GetPath()))
        # The floor belongs beneath both empty volumes, not inside their space.
        if actual[1, 2] <= .001:
            continue
        for label, volume in [('table_access', RESERVED_TASK_VOLUME), ('robot_access', ROBOT_FLOOR_RESERVE)]:
            low, high = np.array(volume['low']), np.array(volume['high'])
            if (np.minimum(actual[1], high) - np.maximum(actual[0], low) > 1.e-5).all():
                intrusions.append({'path': str(prim.GetPath()), 'volume': label})
    if intrusions:
        raise ValueError('Static workshop intrudes into task space: ' + repr(intrusions))
    table = box(stage.GetPrimAtPath('/World/Geometry/worktop/Collider'))
    np.testing.assert_allclose(table, [[.15, -.5, .74], [1.15, .5, .78]], atol=1.e-7)
    colliders = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
        joint = prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
        if rigid or joint or prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError('Environment contains active dynamics: ' + str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0:
                raise ValueError('Invalid collider transform: ' + str(prim.GetPath()))
            colliders.append(str(prim.GetPath()))
    placeholder = stage.GetPrimAtPath('/World/Robot')
    if not placeholder or placeholder.GetChildren() or placeholder.HasAuthoredReferences():
        raise ValueError('Robot placeholder must remain empty')
    return {'status': 'passed', 'units': 'metres', 'up_axis': 'Z',
            'table_bounds_m': table.tolist(), 'collider_count': len(colliders),
            'reserved_task_volume': RESERVED_TASK_VOLUME, 'robot_floor_reserve': ROBOT_FLOOR_RESERVE,
            'intrusions': [], 'active_robot_articulations': 0, 'physics_scenes': 0,
            'scope': 'CPU USD structural audit only; no native contact, reachability or task acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=Path(f'{DATA_ROOT}/assets'))
    args = parser.parse_args()
    from pxr import Usd
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from bench.simulation.assets.workrooms.mechanical_workshop import build_mechanical_workshop
    snapshot = verify_if_snapshot(PROJECT, required_modules=('bench.simulation.assets.workrooms.mechanical_workshop',),
                                  entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Build from a complete frozen project source')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve old assets; choose a new version: ' + str(output))
    output.mkdir(parents=True)
    path = output / 'scene.usda'
    stage = Usd.Stage.CreateNew(str(path))
    scene = build_mechanical_workshop(stage, args.assets_root.resolve(), output)
    audit = audit_environment(stage)
    stage.GetRootLayer().Save()
    from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL
    builtins = {name: name for name in BUILTIN_MDL}
    scene['dependencies'] = dependency_inventory(path, scene['dependencies'], runtime_builtins=builtins)
    scene['runtime_builtin_dependencies'] = builtins
    def write(name, data):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    write('scene_description.json', scene)
    write('structural_audit.json', audit)
    write('manifest.json', {'schema': 'scence.mechanical_workshop_asset.v1', 'variant': scene['variant'],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_snapshot': snapshot,
        'source_root': str(PROJECT), 'scene': str(path), 'scene_sha256': digest(path),
        'description_sha256': digest(output / 'scene_description.json'),
        'native_physics_accepted': False, 'native_rendered': False,
        'scope': scene['scope'], 'dependencies': scene['dependencies']})
    (output / 'ATTRIBUTION.txt').write_text(
        'Mechanical workshop. Original procedural geometry: interface project.\n'
        'NVIDIA SimReady furniture/tools retain their original CC BY 4.0 source licenses.\n'
        'Poly Haven HDRI and ambientCG textures retain their CC0 source records.\n'
        'Generated labels use the existing project font pipeline. All original sources remain\n'
        'at versioned shared asset paths in the dependency inventory.\n'
        'No task objects, robot or manipulation qualification are part of this environment.\n')
    print(json.dumps({'asset': str(output), 'scene_sha256': digest(path),
                      'structural_audit': audit['status'], 'dependency_count': len(scene['dependencies'])}))


if __name__ == '__main__':
    main()
