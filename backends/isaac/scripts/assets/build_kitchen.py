"""Build a versioned galley kitchen from complete frozen project source."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bench.paths import DATA_ROOT
from scripts.assets.build_card_room import dependency_inventory, digest
from scripts.assets.build_living_room import light_inventory

PROJECT=Path(__file__).resolve().parents[2]


def audit_environment(stage, description):
    """Verify real support bounds and the installation aisle in the USD stage."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    if UsdGeom.GetStageMetersPerUnit(stage)!=1. or UsdGeom.GetStageUpAxis(stage)!='Z':
        raise ValueError('Kitchen must be metres, Z up')
    geometry=stage.GetPrimAtPath('/World/Geometry')
    if not geometry:
        raise ValueError('Kitchen has no geometry')
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    world=UsdGeom.XformCache(Usd.TimeCode.Default())
    collisions=[]; disabled_bodies=[]; disabled_joints=[]; support={}
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.Scene) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError('Fixed environment contains runtime physics: '+str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
                raise ValueError('Active rigid body in static environment: '+str(prim.GetPath()))
            disabled_bodies.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            if UsdPhysics.Joint(prim).GetJointEnabledAttr().Get():
                raise ValueError('Active joint in static environment: '+str(prim.GetPath()))
            disabled_joints.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mat=np.asarray(world.GetLocalToWorldTransform(prim))
            if not np.isfinite(mat).all() or np.linalg.det(mat[:3,:3])<=0:
                raise ValueError('Invalid collision transform: '+str(prim.GetPath()))
            collisions.append(str(prim.GetPath()))
    for role,paths in description['support_colliders'].items():
        support[role]=[]
        for path in ([paths] if isinstance(paths,str) else paths):
            prim=stage.GetPrimAtPath(path)
            if (not prim or not prim.HasAPI(UsdPhysics.CollisionAPI)
                    or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False):
                raise ValueError('Missing or disabled declared support collider: '+path)
            bounds=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            support[role].append({'path':path,'bounds_m':[list(bounds.GetMin()),list(bounds.GetMax())]})
    table=support['table'][0]['bounds_m']
    np.testing.assert_allclose(table,[[-.95,-1.35,.855],[1.85,-.55,.90]],atol=1e-6)
    # Every shape crossing the actual central floor-to-wrist column is checked.
    low=np.array([.43,-.23,.001]); high=np.array([.87,.23,1.80]); intrusions=[]
    for prim in geometry.GetChildren():
        if prim.GetName() in ('room_floor',):
            continue
        bounds=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        actual=np.array([bounds.GetMin(),bounds.GetMax()])
        if not np.isfinite(actual).all():
            raise ValueError('Non-finite geometry bounds: '+str(prim.GetPath()))
        if (np.minimum(actual[1],high)-np.maximum(actual[0],low)>1e-5).all():
            intrusions.append(str(prim.GetPath()))
    if intrusions:
        raise ValueError('Central robot installation column is obstructed: '+repr(intrusions))
    # Check the authored objects against the advertised bar task surface.
    # Lights above it are allowed; tabletop clutter in the usable prism is not.
    rectangle=description['layout']['work_surfaces']['bar']['clear_task_rectangle_xy_m']
    clear_low=np.array([rectangle[0],rectangle[2],.9001])
    clear_high=np.array([rectangle[1],rectangle[3],1.65])
    task_intrusions=[]
    for prim in geometry.GetChildren():
        bounds=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        actual=np.array([bounds.GetMin(),bounds.GetMax()])
        if (np.minimum(actual[1],clear_high)-np.maximum(actual[0],clear_low)>1e-5).all():
            task_intrusions.append(str(prim.GetPath()))
    if task_intrusions:
        raise ValueError('Reserved bar task rectangle is obstructed: '+repr(task_intrusions))
    placeholder=stage.GetPrimAtPath('/World/Robot')
    if not placeholder or placeholder.GetChildren() or placeholder.HasAuthoredReferences():
        raise ValueError('Robot placeholder must remain empty')
    from scripts.media.render_workroom import validate_camera_geometry
    cameras=validate_camera_geometry(stage,description['cameras'])
    return {'status':'passed','units':'metres','up_axis':'Z','support_surfaces':support,
            'collider_count':len(collisions),'disabled_imported_bodies':disabled_bodies,
            'disabled_imported_joints':disabled_joints,'robot_column_intrusions':intrusions,
            'reserved_task_rectangle_xy_m':rectangle,'reserved_task_intrusions':task_intrusions,
            'robot_aisle_width_m':1.12,'camera_geometry':cameras,
            'scope':'CPU USD structure, support bounds and static installation column only; no dynamics'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets-root',type=Path,default=Path(f'{DATA_ROOT}/assets'))
    args=parser.parse_args()
    from pxr import Usd
    from bench.simulation.assets.workrooms.kitchen import build_kitchen,VARIANT
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot=verify_if_snapshot(PROJECT,required_modules=('bench.simulation.assets.workrooms.kitchen',),entrypoint_file=Path(__file__))
    if snapshot is None:
        parser.error('Build from complete frozen source, not an editable checkout')
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve existing assets; select a new version: '+str(output))
    output.mkdir(parents=True)
    stage_path=output/'scene.usda'; stage=Usd.Stage.CreateNew(str(stage_path))
    scene=build_kitchen(stage,args.assets_root.resolve(),output)
    audit=audit_environment(stage,scene); scene['authored_lights']=light_inventory(stage)
    stage.GetRootLayer().Save()
    from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL
    builtins={name:name for name in BUILTIN_MDL}
    scene['dependencies']=dependency_inventory(stage_path,scene['dependencies'],runtime_builtins=builtins)
    scene['runtime_builtin_dependencies']=builtins

    def write(name,value):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

    write('scene_description.json',scene); write('structural_audit.json',audit)
    write('manifest.json',{'schema':'scence.kitchen_asset.v1','variant':VARIANT,
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_snapshot':snapshot,
        'source_root':str(PROJECT),'scene':str(stage_path),'scene_sha256':digest(stage_path),
        'description_sha256':digest(output/'scene_description.json'),
        'structural_audit_sha256':digest(output/'structural_audit.json'),
        'native_physics_accepted':False,'native_rendered':False,'scope':scene['scope'],
        'dependencies':scene['dependencies']})
    (output/'ATTRIBUTION.txt').write_text(
        'Galley kitchen: original procedural architecture, cabinetry and luminaires.\n'
        'Poly Haven kitchen props / textures / sky: CC0, individual authors in SOURCE.json.\n'
        'NVIDIA SimReady appliances: source licenses and provenance retained with original assets.\n'
        'See manifest.json and scene_description.json for complete source and dependency hashes.\n'
        'Static environment only; robot preview is separate and manipulation is unqualified.\n')
    print(json.dumps({'asset':str(output),'scene_sha256':digest(stage_path),
                      'structural_audit':audit['status'],'dependencies':len(scene['dependencies'])}))


if __name__=='__main__':
    main()
