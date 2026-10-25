"""Build an immutable, task-independent laboratory from complete frozen source."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bench.paths import DATA_ROOT
from scripts.assets.build_card_room import dependency_inventory, digest
from scripts.assets.build_living_room import light_inventory

PROJECT = Path(__file__).resolve().parents[2]


def audit_environment(stage, description):
    """Check actual support geometry, unobstructed task space and static ownership."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from scripts.media.render_workroom import validate_camera_geometry

    if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Laboratory must be metre-scale and Z up')
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    colliders = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if (prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint)):
            raise ValueError('Static room contains runtime physics: '+str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))
            if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0:
                raise ValueError('Invalid collision transform: '+str(prim.GetPath()))
            colliders.append(str(prim.GetPath()))
    support = {}
    for role, paths in description['support_colliders'].items():
        support[role] = []
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if (not prim or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                raise ValueError('Missing or disabled support collider: '+path)
            bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            support[role].append({'path':path, 'bounds_m':[list(bounds.GetMin()),list(bounds.GetMax())]})
    np.testing.assert_allclose(support['table'][0]['bounds_m'],
                               [[-.55,-.625,.855],[1.85,.625,.90]], atol=1.e-6)
    np.testing.assert_allclose(support['rear_counter'][0]['bounds_m'],
                               [[-2.4,1.95,.855],[2.15,2.65,.90]], atol=1.e-6)
    rectangles = {
        'island': description['layout']['clear_task_rectangle_xy_m'],
        'rear_counter': description['layout']['work_surfaces']['rear_counter']['clear_task_rectangle_xy_m'],
    }
    intrusions = []
    geometry = stage.GetPrimAtPath('/World/Geometry')
    for role, rectangle in rectangles.items():
        low = np.array([rectangle[0], rectangle[2], .9001])
        high = np.array([rectangle[1], rectangle[3], 1.65])
        for prim in geometry.GetChildren():
            bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            actual = np.array([bounds.GetMin(),bounds.GetMax()])
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite bounds: '+str(prim.GetPath()))
            if (np.minimum(actual[1],high)-np.maximum(actual[0],low)>1.e-5).all():
                intrusions.append({'surface':role,'path':str(prim.GetPath())})
    if intrusions:
        raise ValueError('Reserved laboratory task volume is obstructed: '+repr(intrusions))
    placeholder = stage.GetPrimAtPath('/World/Robot')
    if not placeholder or placeholder.GetChildren() or placeholder.HasAuthoredReferences():
        raise ValueError('Robot placeholder must remain empty')
    return {'status':'passed', 'units':'metres', 'up_axis':'Z',
            'support_surfaces':support, 'collider_count':len(colliders),
            'geometry_groups':len(geometry.GetChildren()), 'reserved_task_intrusions':intrusions,
            'reserved_task_rectangles_xy_m':rectangles,
            'camera_geometry':validate_camera_geometry(stage, description['cameras']),
            'scope':'USD structure, support dimensions and empty task volume; no native physics acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=Path(f'{DATA_ROOT}/assets'))
    args = parser.parse_args()
    from pxr import Usd
    from bench.simulation.assets.workrooms.laboratory import build_laboratory, VARIANT
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot = verify_if_snapshot(PROJECT,
        required_modules=('bench.simulation.assets.workrooms.laboratory',), entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Build from complete frozen source, not an editable checkout')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve previous versions: '+str(output))
    output.mkdir(parents=True)
    stage_path = output/'scene.usda'
    stage = Usd.Stage.CreateNew(str(stage_path))
    description = build_laboratory(stage,args.assets_root.resolve(),output)
    audit = audit_environment(stage,description)
    description['authored_lights'] = light_inventory(stage)
    stage.GetRootLayer().Save()
    from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL
    builtins = {name: name for name in BUILTIN_MDL}
    description['dependencies'] = dependency_inventory(stage_path,description['dependencies'],runtime_builtins=builtins)
    description['runtime_builtin_dependencies'] = builtins

    def write(name, value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

    write('scene_description.json',description)
    write('structural_audit.json',audit)
    write('manifest.json',{'schema':'scence.laboratory_asset.v1','variant':VARIANT,
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_snapshot':snapshot,
        'source_root':str(PROJECT),'scene':str(stage_path),'scene_sha256':digest(stage_path),
        'description_sha256':digest(output/'scene_description.json'),
        'structural_audit_sha256':digest(output/'structural_audit.json'),
        'native_physics_accepted':False,'native_rendered':False,
        'scope':description['scope'],'dependencies':description['dependencies']})
    (output/'ATTRIBUTION.txt').write_text(
        'Laboratory: original procedural architecture, casework, instruments and dressing.\n'
        'Poly Haven microscope/HDRI and ambientCG materials: CC0; source licenses retained.\n'
        'Noto Sans CJK font: SIL Open Font License; pinned dependency metadata retained.\n'
        'See scene_description.json for source paths and complete dependency hashes.\n'
        'Static environment only; no manipulation or native contact qualification.\n')
    print(json.dumps({'asset':str(output),'scene_sha256':digest(stage_path),
                      'structural_audit':audit['status'],'geometry_groups':audit['geometry_groups'],
                      'dependencies':len(description['dependencies'])}))


if __name__ == '__main__':
    main()
