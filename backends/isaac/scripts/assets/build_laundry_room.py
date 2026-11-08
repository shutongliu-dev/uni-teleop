"""Build a versioned furnished laundry room from a complete source snapshot."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bench.paths import DATA_ROOT
from scripts.assets.build_card_room import dependency_inventory, digest

PROJECT = Path(__file__).resolve().parents[2]


def audit_environment(stage):
    """Check metric supports, actual clear table volume and static scene ownership."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux
    from bench.simulation.assets.workrooms.laundry_room import TABLE, ROOM_SPEC, RESERVED_TASK_VOLUME
    if UsdGeom.GetStageMetersPerUnit(stage)!=1. or UsdGeom.GetStageUpAxis(stage)!='Z':
        raise ValueError('Laundry room must use metres and Z up')
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
    def bounds(path):
        prim=stage.GetPrimAtPath(path)
        if not prim:raise ValueError('Missing geometry: '+str(path))
        bound=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        result=np.array([bound.GetMin(),bound.GetMax()])
        if not np.isfinite(result).all() or (result[1]<result[0]).any():
            raise ValueError('Invalid bounds: '+str(path))
        return result
    volumes={'folding_surface':RESERVED_TASK_VOLUME}
    intrusions=[]
    for prim in stage.GetPrimAtPath('/World/Geometry').GetChildren():
        actual=bounds(prim.GetPath())
        for label,volume in volumes.items():
            if (np.minimum(actual[1],volume['high'])-np.maximum(actual[0],volume['low'])>1e-5).all():
                intrusions.append({'path':str(prim.GetPath()),'volume':label})
    if intrusions:raise ValueError('Scenery enters reserved folding space: '+repr(intrusions))
    table=bounds(TABLE['collider_path']);center=np.array(TABLE['center_xy']);size=np.array(TABLE['size_m'])
    np.testing.assert_allclose(table[:,:2],[center-size/2,center+size/2],atol=1e-7)
    np.testing.assert_allclose(table[1,2],TABLE['top_z'],atol=1e-7)
    floor=bounds('/World/Geometry/room_floor/Collider');xmin,xmax,ymin,ymax=ROOM_SPEC['bounds']
    np.testing.assert_allclose(floor[:,:2],[[xmin,ymin],[xmax,ymax]],atol=1e-7)
    np.testing.assert_allclose(floor[1,2],0.,atol=1e-7)
    colliders=[]; meshes=0;triangles=0;lights=[]
    transforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if (prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.IsA(UsdPhysics.Joint)
                or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Scene)):
            raise ValueError('Static room contains a runtime physics actor: '+str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            meshes+=1; mesh=UsdGeom.Mesh(prim)
            triangles+=sum(max(0,int(n)-2) for n in mesh.GetFaceVertexCountsAttr().Get())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            mat=np.asarray(transforms.GetLocalToWorldTransform(prim))
            if not np.isfinite(mat).all() or np.linalg.det(mat[:3,:3])<=0:
                raise ValueError('Invalid collider transform: '+str(prim.GetPath()))
            colliders.append(str(prim.GetPath()))
            actual=bounds(prim.GetPath())
            v=RESERVED_TASK_VOLUME
            if (np.minimum(actual[1],v['high'])-np.maximum(actual[0],v['low'])>1e-5).all():
                raise ValueError('Collider enters clear table volume: '+str(prim.GetPath()))
        if prim.HasAPI(UsdLux.LightAPI):
            if prim.GetCustomDataByKey('scence:nominalIntensity') is None:
                raise ValueError('Missing light baseline: '+str(prim.GetPath()))
            lights.append(str(prim.GetPath()))
    for path in (TABLE['collider_path'],'/World/Geometry/room_floor/Collider'):
        if not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI):raise ValueError('Missing support collider')
    placeholder=stage.GetPrimAtPath('/World/Robot')
    if not placeholder or placeholder.GetChildren() or placeholder.HasAuthoredReferences():
        raise ValueError('Fixed room must not contain a robot')
    required=('washing_machine','heat_pump_dryer','sink_bottom','gooseneck_faucet','entrance_door','window_mullion')
    for name in required:
        if not stage.GetPrimAtPath('/World/Geometry/'+name):raise ValueError('Missing room fixture: '+name)
    return {'status':'passed','units':'metres','up_axis':'Z','room_area_m2':(xmax-xmin)*(ymax-ymin),
            'floor_bounds_m':floor.tolist(),'table_bounds_m':table.tolist(),
            'reserved_task_volume':RESERVED_TASK_VOLUME,'intrusions':[],
            'geometry_groups':len(stage.GetPrimAtPath('/World/Geometry').GetChildren()),
            'mesh_count':meshes,'triangle_count':triangles,'collider_count':len(colliders),'lights':lights,
            'robot_articulations':0,'physics_scenes':0,
            'scope':'USD geometry, dependencies and static ownership; no native contact or task acceptance'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets-root',type=Path,default=Path(f'{DATA_ROOT}/assets'))
    args=parser.parse_args()
    from pxr import Usd
    from bench.simulation.assets.workrooms.laundry_room import build_laundry_room
    from scripts.assets.build_living_room import light_inventory
    from scripts.media.render_workroom import validate_camera_geometry
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    snapshot=verify_if_snapshot(PROJECT,required_modules=('bench.simulation.assets.workrooms.laundry_room',),entrypoint_file=Path(__file__))
    if snapshot is None:parser.error('Build from a complete frozen source snapshot')
    output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a new immutable asset version: '+str(output))
    output.mkdir(parents=True);path=output/'scene.usda';stage=Usd.Stage.CreateNew(str(path))
    scene=build_laundry_room(stage,args.assets_root.resolve(),output)
    audit=audit_environment(stage)
    audit['camera_geometry']=validate_camera_geometry(stage,scene['cameras'])
    scene['authored_lights']=light_inventory(stage)
    stage.GetRootLayer().Save()
    from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL,MDL_ROOT
    builtins={name:str(MDL_ROOT/name) for name in BUILTIN_MDL}
    scene['dependencies']=dependency_inventory(path,scene['dependencies'],runtime_builtins=builtins)
    scene['runtime_builtin_dependencies']=builtins
    def write(name,value):(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    write('scene_description.json',scene);write('structural_audit.json',audit)
    write('manifest.json',{'schema':'scence.laundry_room_asset.v1','variant':scene['variant'],
            'created_utc':datetime.now(timezone.utc).isoformat(),'source_snapshot':snapshot,'source_root':str(PROJECT),
            'scene':str(path),'scene_sha256':digest(path),'description_sha256':digest(output/'scene_description.json'),
            'structural_audit_sha256':digest(output/'structural_audit.json'),
            'native_physics_accepted':False,'native_rendered':False,'scope':scene['scope'],
            'dependencies':scene['dependencies']})
    (output/'ATTRIBUTION.txt').write_text(
        'Laundry room: original architecture, casework and procedural dressing by interface project.\n'
        'Innerscene appliance/furniture models and Poly Haven PBR/scans/HDR: CC0.\n'
        'ambientCG surface textures: CC0; Noto font: SIL Open Font License.\n'
        'Original source, licenses, immutable derivations and dependency hashes are preserved in shared assets.\n'
        'All clothes and appliances in this room are static scenery; no manipulation task is implemented.\n')
    print(json.dumps({'asset':str(output),'scene_sha256':digest(path),'audit':audit['status'],
                      'geometry_groups':audit['geometry_groups'],'dependencies':len(scene['dependencies'])}))


if __name__=='__main__':main()
