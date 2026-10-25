#!/usr/bin/env python3
"""Author a new immutable card-room environment from a complete source snapshot.

CPU USD tool: no Isaac application, GPU or robot simulation is started.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_environment(stage):
    """Check the actual USD support, ownership and reserved task volume."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.assets.workrooms.card_room import RESERVED_TASK_VOLUME
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    low = np.array(RESERVED_TASK_VOLUME['low'])
    high = np.array(RESERVED_TASK_VOLUME['high'])
    intrusions = []
    for prim in stage.GetPrimAtPath('/World/Geometry').GetChildren():
        box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        lo = np.array(box.GetMin()); hi = np.array(box.GetMax())
        if not np.isfinite([lo, hi]).all():
            raise ValueError('Nonfinite environment bounds: ' + str(prim.GetPath()))
        if (np.minimum(hi, high) - np.maximum(lo, low) > 1.e-5).all():
            intrusions.append(str(prim.GetPath()))
    if intrusions:
        raise ValueError('Room intrudes into task volume: ' + repr(intrusions))
    table = bounds.ComputeWorldBound(stage.GetPrimAtPath('/World/Geometry/worktop/Collider')).ComputeAlignedRange()
    np.testing.assert_allclose(table.GetMin(), [.15, -.5, .74], atol=1.e-7)
    np.testing.assert_allclose(table.GetMax(), [1.15, .5, .78], atol=1.e-7)
    colliders = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
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
    return {'status': 'passed', 'collider_count': len(colliders),
            'table_bounds_m': [list(table.GetMin()), list(table.GetMax())],
            'reserved_task_volume': RESERVED_TASK_VOLUME, 'intrusions': intrusions,
            'robot_articulations': 0, 'physics_scenes': 0,
            'scope': 'CPU structural check only; no native contact or manipulation acceptance'}


def dependency_inventory(stage_path, explicit, *, runtime_builtins=None):
    from pxr import UsdUtils
    from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(stage_path))
    # Isaac Sim core MDL modules are resolved by the local installation, not packaged.
    provided = BUILTIN_MDL | set(runtime_builtins or ())
    unknown = [name for name in unresolved if name not in provided]
    if unknown:
        raise ValueError('Unresolved USD dependencies: ' + repr(unknown))
    paths = set(explicit)
    paths.update(str(layer.realPath) for layer in layers if layer.realPath)
    paths.update(map(str, assets))
    paths.discard(str(stage_path))
    # Furniture derivations preserve the upstream source and license locations.
    for path in list(paths):
        if Path(path).name != 'DERIVATION.json':
            continue
        source = load_json(Path(path)).get('source_provenance')
        if source:
            paths.add(source)
            paths.add(str(Path(source).with_name('LICENSE.txt')))
    inventory = {}
    for path in sorted(paths):
        source = Path(path).resolve(strict=True)
        if not source.is_file():
            raise ValueError('Dependency is not a regular file: ' + str(source))
        inventory[str(source)] = {'sha256': digest(source), 'bytes': source.stat().st_size}
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=Path(f'{DATA_ROOT}/assets'))
    args = parser.parse_args()
    from pxr import Usd
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from bench.simulation.assets.workrooms.card_room import build_card_room
    verified = verify_if_snapshot(PROJECT, required_modules=('bench.simulation.assets.workrooms.card_room',),
                                  entrypoint_file=Path(__file__))
    if verified is None:
        parser.error('Build from a complete frozen project source, not an editable checkout')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new version; existing build directories are immutable: ' + str(output))
    output.mkdir(parents=True)
    stage_path = output / 'scene.usda'
    stage = Usd.Stage.CreateNew(str(stage_path))
    scene = build_card_room(stage, args.assets_root.resolve(), output)
    audit = audit_environment(stage)
    stage.GetRootLayer().Save()
    scene['dependencies'] = dependency_inventory(stage_path, scene['dependencies'])
    (output / 'scene_description.json').write_text(json.dumps(scene, ensure_ascii=False, indent=2) + '\n')
    (output / 'structural_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    provenance = {'schema': 'scence.card_room_asset.v1', 'variant': scene['variant'],
                  'created_utc': datetime.now(timezone.utc).isoformat(),
                  'source_snapshot': verified, 'source_root': str(PROJECT),
                  'scene': str(stage_path), 'scene_sha256': digest(stage_path),
                  'description_sha256': digest(output / 'scene_description.json'),
                  'native_physics_accepted': False, 'native_rendered': False,
                  'scope': scene['scope'], 'dependencies': scene['dependencies']}
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'ATTRIBUTION.txt').write_text(
        'Heritage card-room environment. Original procedural geometry: interface project.\n'
        'Poly Haven furniture/HDRI and ambientCG textures: CC0. Original licenses and\n'
        'source metadata are retained at the pinned shared dependency locations.\n'
        'Noto Sans CJK: SIL Open Font License, pinned font LICENSE.txt.\n'
        'No robot, game pieces, controller, or native physics qualification is part of this asset.\n')
    print(json.dumps({'asset': str(output), 'scene_sha256': provenance['scene_sha256'],
                      'structural_audit': audit['status'], 'dependency_count': len(scene['dependencies'])}))


if __name__ == '__main__':
    main()
