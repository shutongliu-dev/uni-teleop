from copy import deepcopy
import math

import numpy as np
import pytest

from scenes.workrooms.parameterized.spec import sample_parameters
from scenes.workrooms.parameterized.usd_authoring import apply_layout, apply_physics, group_layout, world_pose


def stage_with_bodies():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/World')
    for name, x in [('a', 0.), ('b', .25)]:
        body = UsdGeom.Xform.Define(stage, '/World/'+name)
        body.AddTranslateOp().Set(Gf.Vec3d(x, .1, .8))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim()); mass.CreateMassAttr(.2)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001, .002, .003))
        shape = UsdGeom.Cube.Define(stage, '/World/'+name+'/collision')
        shape.CreateSizeAttr(.02); UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return stage


def test_real_physics_binding_and_exactly_one_mass_scale():
    from pxr import UsdPhysics, UsdShade
    stage = stage_with_bodies(); params = sample_parameters('study', 'book_stacking', 'single_arm', 1)
    original = deepcopy(params)
    receipt = apply_physics(stage, {'a':'/World/a', 'b':'/World/b'}, '/World', params)
    for name in ('a', 'b'):
        mass = UsdPhysics.MassAPI(stage.GetPrimAtPath('/World/'+name))
        assert mass.GetMassAttr().Get() == pytest.approx(.2*params['physics']['mass_factor'])
        assert list(mass.GetDiagonalInertiaAttr().Get()) == pytest.approx(np.array([.001,.002,.003])*params['physics']['mass_factor'])
        material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/World/'+name+'/collision')).ComputeBoundMaterial(materialPurpose='physics')
        assert material
        assert UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() == pytest.approx(.5*params['physics']['friction_factor'])
    material = receipt['material_physics']['materials']['chosen_simulator_prior']
    assert material['baseline_origin'] == 'chosen_simulator_prior_not_measured'
    assert params == original


def test_canonical_preserves_stage_bytes_and_unbound_contact_unknown():
    stage = stage_with_bodies(); before = stage.GetRootLayer().ExportToString()
    params = sample_parameters('study', 'book_stacking', 'single_arm', 0)
    apply_layout(stage, [{'id':'a','paths':['/World/a']}], params)
    receipt = apply_physics(stage, {'a':'/World/a'}, '/World', params)
    assert stage.GetRootLayer().ExportToString() == before
    assert receipt['material_physics']['materials']['chosen_simulator_prior']['applied'] is None


def test_independent_groups_full_radius_and_real_world_transform():
    stage = stage_with_bodies(); params = sample_parameters('study', 'book_stacking', 'single_arm', 2)
    groups = [{'id':n,'paths':['/World/'+n]} for n in ('a','b')]
    expected_radius = math.hypot(*params['layout']['translation_xy_m'])
    receipts = apply_layout(stage, groups, params)
    assert receipts[0]['translation_xy_m'] != receipts[1]['translation_xy_m']
    for row in receipts:
        assert math.hypot(*row['translation_xy_m']) == pytest.approx(expected_radius)
        path = row['paths'][0]
        delta = np.array(world_pose(stage,path)['position_m'])-row['before'][path]['position_m']
        assert delta[:2] == pytest.approx(row['translation_xy_m'])
        assert delta[2] == pytest.approx(0)


def test_joint_chain_cannot_be_split():
    from pxr import UsdPhysics
    stage=stage_with_bodies(); joint=UsdPhysics.FixedJoint.Define(stage,'/World/joint')
    joint.CreateBody0Rel().SetTargets(['/World/a']); joint.CreateBody1Rel().SetTargets(['/World/b'])
    params=sample_parameters('mechanical_assembly','nist_cable_routing','single_arm',1)
    with pytest.raises(ValueError, match='splits a connected joint'):
        apply_layout(stage,[{'id':'a','paths':['/World/a']}],params)


def test_disabled_collision_stays_disabled_with_explicit_material():
    from pxr import UsdPhysics, UsdShade
    stage=stage_with_bodies();prim=stage.GetPrimAtPath('/World/a/collision')
    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    params=sample_parameters('laboratory','pipette_transfer','single_arm',1)
    result=apply_physics(stage,{'a':'/World/a'},'/World',params)
    assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
    material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
    assert material
    assert result['material_physics']['collision_disabled_paths_preserved']==['/World/a/collision']
