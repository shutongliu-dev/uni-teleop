from copy import deepcopy
import math
import pytest
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from bench.simulation.isaacsim_backend.volume_visual_state import CylinderVolumeBindings,SCHEMA


def scene():
    stage=Usd.Stage.CreateInMemory()
    body=UsdGeom.Xform.Define(stage,'/World/Cup');body.AddTranslateOp().Set(Gf.Vec3d(1.,2.,3.))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.1)
    collider=UsdGeom.Cube.Define(stage,'/World/Cup/Collider');UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    visual=UsdGeom.Cylinder.Define(stage,'/World/Cup/Liquid')
    visual.CreateAxisAttr('Z');visual.CreateRadiusAttr(.02);visual.CreateHeightAttr(.001)
    visual.AddTranslateOp().Set(Gf.Vec3d(0.,0.,.0045))
    spec={'schema':SCHEMA,'bindings':[{'id':'water','target_prim_path':'/World/Cup/Liquid',
        'state_path':['detail','water_ml'],'cross_section_area_m2':math.pi*.02**2,'floor_z_m':.004,'maximum_height_m':.06}]}
    return stage,spec


def test_volume_height_visibility_and_idempotence_keep_physics_unchanged():
    stage,spec=scene();binding=CylinderVolumeBindings(stage,spec)
    body=stage.GetPrimAtPath('/World/Cup');collider=stage.GetPrimAtPath('/World/Cup/Collider')
    originals=[{a.GetName():a.Get() for a in prim.GetAttributes()} for prim in (body,collider)]
    result=binding.sync({'detail':{'water_ml':10.}},native_step=12)
    assert result['bindings'][0]['height_m']==pytest.approx(10e-6/(math.pi*.02**2))
    assert result['bindings'][0]['changed']
    before=stage.GetRootLayer().ExportToString()
    again=binding.sync({'detail':{'water_ml':10.}},native_step=12)
    assert not again['bindings'][0]['changed']
    assert stage.GetRootLayer().ExportToString()==before
    empty=binding.sync({'detail':{'water_ml':0.}},native_step=13)
    assert empty['bindings'][0]['visibility']=='invisible'
    for prim,old in zip((body,collider),originals):
        assert {a.GetName():a.Get() for a in prim.GetAttributes()}==old


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,10000.,True])
def test_invalid_volume_is_rejected_before_writes(value):
    stage,spec=scene();binding=CylinderVolumeBindings(stage,spec)
    before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match='finite'):
        binding.sync({'detail':{'water_ml':value}},native_step=0)
    assert stage.GetRootLayer().ExportToString()==before


def test_collision_or_rigid_geometry_is_not_a_visual_target():
    for api in (UsdPhysics.CollisionAPI,UsdPhysics.RigidBodyAPI):
        stage,spec=scene();api.Apply(stage.GetPrimAtPath('/World/Cup/Liquid'))
        with pytest.raises(ValueError,match='noncollision'):
            CylinderVolumeBindings(stage,spec)


def test_invalid_dimension_or_physics_descendant_is_rejected():
    stage,spec=scene();spec['bindings'][0]['maximum_height_m']=float('nan')
    with pytest.raises(ValueError,match='finite'):CylinderVolumeBindings(stage,spec)
    stage,spec=scene()
    child=UsdGeom.Cube.Define(stage,'/World/Cup/Liquid/UnexpectedCollider');UsdPhysics.CollisionAPI.Apply(child.GetPrim())
    with pytest.raises(ValueError,match='leaf'):CylinderVolumeBindings(stage,spec)
