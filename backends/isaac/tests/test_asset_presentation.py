import hashlib

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from bench.simulation.isaacsim_backend.presentation import apply_presentation, physical_fingerprint


def base():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/World/Objects/part')
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.1)
    mesh = UsdGeom.Mesh.Define(stage, str(root.GetPath())+'/mesh')
    mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3]); mesh.CreateFaceVertexIndicesAttr([0,1,2])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mat = UsdShade.Material.Define(stage, '/World/Friction')
    UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateStaticFrictionAttr(.4)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat, materialPurpose='physics')
    return stage


def descriptor(tmp_path, edit):
    stage = Usd.Stage.CreateInMemory()
    stage.OverridePrim('/Asset')
    edit(stage)
    path = tmp_path/'appearance.usda'
    stage.GetRootLayer().Export(str(path))
    return {'presentation': {'schema':'asset_presentation.v1', 'usd_path':str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'prim_path':'/Asset',
        'dependencies_sha256':{}}}


def test_overlay_remaps_shaders_preserves_shared_collision_mesh(tmp_path):
    def edit(stage):
        mesh = stage.OverridePrim('/Asset/mesh')
        mesh.CreateAttribute('normals',Sdf.ValueTypeNames.Normal3fArray).Set([(0,0,1)]*3)
        material=UsdShade.Material.Define(stage,'/Asset/Looks/metal')
        shader=UsdShade.Shader.Define(stage,'/Asset/Looks/metal/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material)
    scene=base(); before=physical_fingerprint(scene,'/World/Objects/part')
    record=apply_presentation(scene,'/World/Objects/part',descriptor(tmp_path,edit))
    assert record['physical_inputs_unchanged']
    assert before == physical_fingerprint(scene,'/World/Objects/part')
    mat,_=UsdShade.MaterialBindingAPI(scene.GetPrimAtPath('/World/Objects/part/mesh')).ComputeBoundMaterial()
    assert str(mat.GetPath())=='/World/Objects/part/Looks/metal'
    assert str(mat.ComputeSurfaceSource()[0].GetPath())=='/World/Objects/part/Looks/metal/Shader'


@pytest.mark.parametrize('kind',['points','mass','new_collider','external_target','hash'])
def test_rejects_nonvisual_changes_before_mutation(tmp_path,kind):
    def edit(stage):
        mesh=stage.OverridePrim('/Asset/mesh')
        if kind=='points':
            mesh.CreateAttribute('points',Sdf.ValueTypeNames.Point3fArray).Set([(0,0,4)])
        elif kind=='mass':
            UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath('/Asset')).CreateMassAttr(10.)
        elif kind=='new_collider':
            UsdPhysics.CollisionAPI.Apply(UsdGeom.Mesh.Define(stage,'/Asset/detail').GetPrim())
        elif kind=='external_target':
            mesh.CreateRelationship('material:binding').SetTargets(['/Outside'])
    spec=descriptor(tmp_path,edit)
    if kind=='hash': spec['presentation']['sha256']='0'*64
    scene=base(); before=scene.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): apply_presentation(scene,'/World/Objects/part',spec)
    assert before==scene.GetRootLayer().ExportToString()
