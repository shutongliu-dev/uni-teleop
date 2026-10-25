"""CPU USD appearance checks against fixed source meshes and physics opinions."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from bench.paths import DATA_ROOT
pytest.importorskip('pxr.Usd')
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from bench.simulation.assets.task_parts import load_obj_mesh, load_task_asset
from scenes.workrooms.mechanical_assembly.visual_tools import MATERIAL_KEYS, author_tool_details

GEAR_ROOT=Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/gear_train_assembly/v2')
DRIVER=Path(f'{DATA_ROOT}/assets/task_parts/threaded_sharpa/sdf1024_convex_partition_candidate02_20261018/mechanical_screwdriver_sharpa_v3.usda')


def _stage(role, nested=False):
    stage=Usd.Stage.CreateInMemory()
    root=UsdGeom.Xform.Define(stage,'/World/Tool').GetPrim()
    anchor=UsdGeom.Xform.Define(stage,'/World/Tool/ReferenceAnchor').GetPrim() if nested else root
    deps=[]
    if role in ('screwdriver','straight_hex_screwdriver'):
        anchor.GetReferences().AddReference(str(DRIVER))
        deps=[DRIVER]
    else:
        asset=load_task_asset(role,GEAR_ROOT,verify_hashes=True)
        body=UsdPhysics.RigidBodyAPI.Apply(root)
        body.CreateRigidBodyEnabledAttr(True)
        mass=UsdPhysics.MassAPI.Apply(root);mass.CreateMassAttr(asset['mass_kg'])
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*asset['inertia_diag_kg_m2']))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*asset['center_of_mass_m']))
        for kind in ('visual','collision'):
            for i,part in enumerate(asset[kind+'_parts']):
                path=str(anchor.GetPath())+f'/{kind}_{i:03d}'
                points,faces=load_obj_mesh(part['mesh_path']);mesh=UsdGeom.Mesh.Define(stage,path)
                mesh.CreatePointsAttr(points.tolist());mesh.CreateFaceVertexCountsAttr([3]*len(faces))
                mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
                if kind=='collision':
                    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
                    mesh.CreateVisibilityAttr('invisible')
                deps.append(Path(part['mesh_path']))
    materials={k:UsdShade.Material.Define(stage,'/Looks/'+k) for k in MATERIAL_KEYS}
    return stage,root,materials,deps


def _physics(stage):
    result={}
    for p in stage.Traverse():
        row={}
        for a in p.GetAttributes():
            if a.GetName().startswith(('physics:','physx')) or (p.HasAPI(UsdPhysics.CollisionAPI)
                    and a.GetName() in ('points','faceVertexCounts','faceVertexIndices','xformOpOrder')):
                row[a.GetName()]=str(a.Get())
        if row: result[str(p.GetPath())]=row
    return result


@pytest.mark.parametrize('role,nested',[('screwdriver',False),('straight_hex_screwdriver',True),
                                        ('socket_crank',True),('open_end_wrench',False)])
def test_visual_detail_stays_within_source_and_preserves_physics(role,nested):
    stage,root,materials,deps=_stage(role,nested)
    physics=deepcopy(_physics(stage))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}
    old_meshes={str(p.GetPath()): (str(UsdGeom.Mesh(p).GetPointsAttr().Get()),
                  str(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get())) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)}
    report=author_tool_details(stage,root.GetPath(),role,materials=materials)
    assert _physics(stage)==physics
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in deps}
    for path,(points,faces) in old_meshes.items():
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath(path))
        assert (str(mesh.GetPointsAttr().Get()),str(mesh.GetFaceVertexIndicesAttr().Get()))==(points,faces)
    for path in report['hidden_source_visual_paths']:
        assert UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr().Get()=='invisible'
    for path in report['retained_working_visual_paths']:
        assert UsdGeom.Imageable(stage.GetPrimAtPath(path)).ComputeVisibility()!='invisible'
    points=[]
    for path in report['created_mesh_paths']:
        prim=stage.GetPrimAtPath(path);mesh=UsdGeom.Mesh(prim)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        xyz=np.asarray(mesh.GetPointsAttr().Get());points.extend(xyz)
        count=sum(mesh.GetFaceVertexCountsAttr().Get())
        normal=np.asarray(mesh.GetNormalsAttr().Get())
        uv=np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get())
        assert normal.shape==(count,3) and uv.shape==(count,2)
        assert np.isfinite(xyz).all() and np.isfinite(uv).all()
        assert np.allclose(np.linalg.norm(normal,axis=1),1.,atol=2.e-7)
        assert UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    lo,hi=report['source_envelope_m'];points=np.asarray(points)
    assert np.all(points>=np.asarray(lo)-2.e-7) and np.all(points<=np.asarray(hi)+2.e-7)
    if role!='open_end_wrench':
        xy=points[:, :2]-([-0.07,0.] if role=='socket_crank' else [0.,0.])
        # Only inspect the unchanged cylindrical mid-grip, excluding the lever.
        z0,z1=(.058,.140) if role=='socket_crank' else (.113,.195)
        radius=np.linalg.norm(xy[(points[:,2]>=z0-1e-7)&(points[:,2]<=z1+1e-7)],axis=1)
        assert np.max(radius)==pytest.approx(report['dimensions']['handle_diameter_m']/2,abs=1e-8)
    assert not report['physics_authored'] and not report['source_assets_modified']
    with pytest.raises(ValueError,match='already exist'):
        author_tool_details(stage,root.GetPath(),role,materials=materials)


def test_wrong_source_scale_and_collision_visual_reject_before_authoring():
    stage,root,materials,_=_stage('screwdriver')
    visual=UsdGeom.Mesh(stage.GetPrimAtPath('/World/Tool/Visuals/handle'))
    visual.GetPointsAttr().Set((np.asarray(visual.GetPointsAttr().Get())*1.1).tolist())
    before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match='dimensions differ'):
        author_tool_details(stage,root.GetPath(),'screwdriver',materials=materials)
    assert stage.GetRootLayer().ExportToString()==before
    stage,root,materials,_=_stage('open_end_wrench')
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath('/World/Tool/visual_003'))
    before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match='noncolliding source visual'):
        author_tool_details(stage,root.GetPath(),'open_end_wrench',materials=materials)
    assert stage.GetRootLayer().ExportToString()==before
