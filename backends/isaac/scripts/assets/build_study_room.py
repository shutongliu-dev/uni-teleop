"""Build a versioned empty-desk study from a complete source snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bench.paths import DATA_ROOT
from scripts.assets.build_card_room import dependency_inventory, digest

PROJECT = Path(__file__).resolve().parents[2]


def audit_environment(stage):
    """Check actual desk/shelf supports, clear task spaces and static ownership."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.assets.workrooms.study_room import (
        TABLE, ROOM_SPEC, SHELF_BAYS, RESERVED_TASK_VOLUME, ROBOT_FLOOR_RESERVE)

    if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Study must use metres and Z up')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    collision_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        ['default', 'render', 'proxy', 'guide'], False, True)

    def bounds(prim, *, collision=False):
        if not prim:
            raise ValueError('Missing study geometry')
        extent = (collision_cache if collision else cache).ComputeWorldBound(prim).ComputeAlignedRange()
        result = np.array([extent.GetMin(), extent.GetMax()])
        if not np.isfinite(result).all() or (result[1] < result[0]).any():
            raise ValueError('Invalid study geometry bounds: ' + str(prim.GetPath()))
        return result

    geometry = stage.GetPrimAtPath('/World/Geometry')
    if not geometry:
        raise ValueError('Study is missing /World/Geometry')
    volumes = [('desk_access', RESERVED_TASK_VOLUME), ('robot_access', ROBOT_FLOOR_RESERVE)]
    volumes.extend(('shelf_' + bay['id'], bay) for bay in SHELF_BAYS)
    intrusions = []
    for prim in geometry.GetChildren():
        actual = bounds(prim)
        if actual[1, 2] <= .00501:
            continue
        for label, volume in volumes:
            if (np.minimum(actual[1], volume['high'])
                    - np.maximum(actual[0], volume['low']) > 1.e-5).all():
                intrusions.append({'path': str(prim.GetPath()), 'volume': label})
    if intrusions:
        raise ValueError('Study geometry intrudes into reserved space: ' + repr(intrusions))

    def support(path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Missing static support collider: ' + path)
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            raise ValueError('Disabled static support collider: ' + path)
        return bounds(prim, collision=True)

    table = support(TABLE['collider_path'])
    center, size = np.array(TABLE['center_xy']), np.array(TABLE['size_m'])
    np.testing.assert_allclose(table[:, :2], [center - size / 2, center + size / 2], atol=1.e-7)
    np.testing.assert_allclose(table[1, 2], TABLE['top_z'], atol=1.e-7)
    if table[1, 2] - table[0, 2] < .025:
        raise ValueError('Study desk has no substantial supporting top')
    floor = support('/World/Geometry/room_floor/Collider')
    xmin, xmax, ymin, ymax = ROOM_SPEC['bounds']
    np.testing.assert_allclose(floor[:, :2], [[xmin, ymin], [xmax, ymax]], atol=1.e-7)
    np.testing.assert_allclose(floor[1, 2], 0., atol=1.e-7)
    if floor[0, 2] >= floor[1, 2]:
        raise ValueError('Study floor has no supporting thickness')
    shelf_reports = []
    for bay in SHELF_BAYS:
        slab = support(bay['collider_path'])
        np.testing.assert_allclose(slab[1, 2], bay['support_z_m'], atol=1.e-7)
        if (slab[0, :2] > np.array(bay['low'])[:2] + 1.e-6).any() or (
                slab[1, :2] < np.array(bay['high'])[:2] - 1.e-6).any():
            raise ValueError('Shelf bay extends beyond its support: ' + bay['id'])
        if bay['low'][2] < bay['support_z_m'] - 1.e-6:
            raise ValueError('Shelf empty volume enters its support: ' + bay['id'])
        shelf_reports.append({**bay, 'support_bounds_m': slab.tolist(), 'empty': True})

    colliders = []
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.IsA(UsdPhysics.Joint)
                or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Scene)):
            raise ValueError('Fixed study contains a runtime actor: ' + str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))
            if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0.:
                raise ValueError('Invalid static collider transform: ' + str(prim.GetPath()))
            colliders.append(str(prim.GetPath()))
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
                actual = bounds(prim, collision=True)
                for label, volume in volumes:
                    if (np.minimum(actual[1], volume['high'])
                            - np.maximum(actual[0], volume['low']) > 1.e-5).all():
                        raise ValueError('Study collider intrudes into reserved space: '
                                         + str(prim.GetPath()) + ' / ' + label)
    placeholder = stage.GetPrimAtPath('/World/Robot')
    if not placeholder or placeholder.GetChildren() or placeholder.HasAuthoredReferences():
        raise ValueError('Fixed study robot placeholder must remain empty')
    return {'status': 'passed', 'units': 'metres', 'up_axis': 'Z',
            'table_bounds_m': table.tolist(), 'floor_bounds_m': floor.tolist(),
            'shelf_bays': shelf_reports,
            'reserved_task_volume': RESERVED_TASK_VOLUME,
            'robot_floor_reserve': ROBOT_FLOOR_RESERVE, 'intrusions': [],
            'collider_count': len(colliders), 'robot_articulations': 0, 'physics_scenes': 0,
            'scope': 'CPU USD structure and clear volumes only; no native contact or book manipulation qualification'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=Path(f'{DATA_ROOT}/assets'))
    args = parser.parse_args()
    from pxr import Usd
    from bench.simulation.assets.workrooms.study_room import build_study_room
    from scripts.assets.build_living_room import light_inventory
    from scripts.media.render_workroom import validate_camera_geometry
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot = verify_if_snapshot(PROJECT, required_modules=('bench.simulation.assets.workrooms.study_room',),
                                  entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Build from a complete frozen source snapshot')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous assets; choose a new version: ' + str(output))
    output.mkdir(parents=True)
    path = output / 'scene.usda'
    stage = Usd.Stage.CreateNew(str(path))
    scene = build_study_room(stage, args.assets_root.resolve(), output)
    audit = audit_environment(stage)
    audit['camera_geometry'] = validate_camera_geometry(stage, scene['cameras'])
    scene['authored_lights'] = light_inventory(stage)
    stage.GetRootLayer().Save()
    from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL
    builtins = {name: name for name in BUILTIN_MDL}
    scene['dependencies'] = dependency_inventory(path, scene['dependencies'], runtime_builtins=builtins)
    scene['runtime_builtin_dependencies'] = builtins

    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    write('scene_description.json', scene)
    write('structural_audit.json', audit)
    write('manifest.json', {'schema': 'scence.study_room_asset.v1', 'variant': scene['variant'],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_snapshot': snapshot,
        'source_root': str(PROJECT), 'scene': str(path), 'scene_sha256': digest(path),
        'description_sha256': digest(output / 'scene_description.json'),
        'structural_audit_sha256': digest(output / 'structural_audit.json'),
        'native_physics_accepted': False, 'native_rendered': False,
        'scope': scene['scope'], 'dependencies': scene['dependencies']})
    (output / 'ATTRIBUTION.txt').write_text(
        'Study library. Original procedural architecture and woodwork: interface project.\n'
        'Shared imported furniture, PBR textures, fonts and HDR lighting retain their original\n'
        'source, version and license records, included in the dependency inventory.\n'
        'See scene_description.json and manifest.json for exact resources and hashes.\n'
        'Movable books, task robots and task qualification are not included in this fixed environment.\n')
    print(json.dumps({'asset': str(output), 'scene_sha256': digest(path),
                      'structural_audit': audit['status'], 'empty_shelf_bays': len(audit['shelf_bays']),
                      'dependency_count': len(scene['dependencies'])}))


if __name__ == '__main__':
    main()
