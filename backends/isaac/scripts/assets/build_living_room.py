#!/usr/bin/env python3
"""Build an immutable static living room from a complete source snapshot.

CPU USD authoring only: this tool starts no simulator, GPU or robot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from bench.paths import DATA_ROOT

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.assets.build_card_room import dependency_inventory, digest


def audit_environment(stage, *, reserved_task_volume):
    """Inspect actual support geometry and keep the room free of runtime actors."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Living room must use metres and Z up')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    low, high = np.array(reserved_task_volume['low']), np.array(reserved_task_volume['high'])
    geometry = stage.GetPrimAtPath('/World/Geometry')
    if not geometry:
        raise ValueError('Living room is missing /World/Geometry')
    intrusions = []
    for prim in geometry.GetChildren():
        extent = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        actual = np.array([extent.GetMin(), extent.GetMax()])
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite environment bounds: ' + str(prim.GetPath()))
        if (np.minimum(actual[1], high) - np.maximum(actual[0], low) > 1.e-5).all():
            intrusions.append(str(prim.GetPath()))
    if intrusions:
        raise ValueError('Static room intrudes into task volume: ' + repr(intrusions))
    support = stage.GetPrimAtPath('/World/Geometry/worktop/Collider')
    if not support or not support.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('Living room is missing the static worktop collider')
    table = bounds.ComputeWorldBound(support).ComputeAlignedRange()
    table_bounds = [list(table.GetMin()), list(table.GetMax())]
    np.testing.assert_allclose(table_bounds, [[.15, -.5, .74], [1.15, .5, .78]], atol=1.e-7)
    colliders = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.IsA(UsdPhysics.Joint)
                or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Scene)):
            raise ValueError('Environment owns a runtime physics actor: ' + str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0:
                raise ValueError('Invalid static collision transform: ' + str(prim.GetPath()))
            colliders.append(str(prim.GetPath()))
    placeholder = stage.GetPrimAtPath('/World/Robot')
    if not placeholder or placeholder.GetChildren() or placeholder.HasAuthoredReferences():
        raise ValueError('Robot compatibility placeholder must remain empty')
    return {'status': 'passed', 'units': 'metres', 'up_axis': 'Z',
            'table_bounds_m': table_bounds, 'collider_count': len(colliders),
            'reserved_task_volume': reserved_task_volume, 'intrusions': [],
            'robot_articulations': 0, 'physics_scenes': 0,
            'scope': 'CPU USD structural check only; no native contact or manipulation acceptance'}


def light_inventory(stage):
    """Record composed USD light inputs and transforms, including fixture direction."""
    from pxr import Sdf, Usd, UsdGeom, UsdLux

    def value_json(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Sdf.AssetPath):
            return {'path': value.path, 'resolved_path': value.resolvedPath}
        return [value_json(item) for item in value]

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    lights = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        lights.append({'path': str(prim.GetPath()), 'type': prim.GetTypeName(),
            'role': prim.GetCustomDataByKey('scence:lightRole'),
            'world_transform': value_json(cache.GetLocalToWorldTransform(prim)),
            'inputs': {attribute.GetName(): value_json(attribute.Get())
                       for attribute in prim.GetAttributes()
                       if attribute.GetName().startswith('inputs:')}})
    if not lights:
        raise ValueError('Living room must contain authored lights')
    return lights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=Path(f'{DATA_ROOT}/assets'))
    args = parser.parse_args()
    from pxr import Usd
    from bench.simulation.assets.workrooms.living_room import (
        RESERVED_TASK_VOLUME, VARIANT, build_living_room)
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot = verify_if_snapshot(PROJECT,
        required_modules=('bench.simulation.assets.workrooms.living_room',),
        entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Build from a complete frozen project source, not an editable checkout')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve old assets; choose a new version: ' + str(output))
    output.mkdir(parents=True)
    stage_path = output / 'scene.usda'
    stage = Usd.Stage.CreateNew(str(stage_path))
    scene = build_living_room(stage, args.assets_root.resolve(), output)
    if scene.get('family') != 'living_room' or scene.get('variant') != VARIANT:
        raise ValueError('Living-room authoring returned a different scene identity')
    audit = audit_environment(stage, reserved_task_volume=RESERVED_TASK_VOLUME)
    scene['authored_lights'] = light_inventory(stage)
    stage.GetRootLayer().Save()
    from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL
    builtins = {name: name for name in BUILTIN_MDL}
    scene['dependencies'] = dependency_inventory(stage_path, scene['dependencies'], runtime_builtins=builtins)
    scene['runtime_builtin_dependencies'] = builtins

    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    write('scene_description.json', scene)
    write('structural_audit.json', audit)
    write('manifest.json', {'schema': 'scence.living_room_asset.v1', 'variant': scene['variant'],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_snapshot': snapshot,
        'source_root': str(PROJECT), 'scene': str(stage_path), 'scene_sha256': digest(stage_path),
        'description_sha256': digest(output / 'scene_description.json'),
        'structural_audit_sha256': digest(output / 'structural_audit.json'),
        'native_physics_accepted': False, 'native_rendered': False,
        'scope': scene['scope'], 'dependencies': scene['dependencies']})
    (output / 'ATTRIBUTION.txt').write_text(
        'Residential living room. Original procedural geometry: interface project.\n'
        'Imported furniture, textures and lighting resources retain their original\n'
        'licenses and source records at the versioned shared dependency locations.\n'
        'See scene_description.json and manifest.json for exact dependency hashes.\n'
        'No robot, task objects or native manipulation qualification is part of this asset.\n')
    print(json.dumps({'asset': str(output), 'scene_sha256': digest(stage_path),
        'structural_audit': audit['status'], 'light_count': len(scene['authored_lights']),
        'dependency_count': len(scene['dependencies'])}))


if __name__ == '__main__':
    main()
