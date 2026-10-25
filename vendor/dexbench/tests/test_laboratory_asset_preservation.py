"""A cosmetic revision cannot hide physical changes beneath its allowlist."""
from copy import deepcopy

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from scripts.validation.laboratory_asset_preservation import fixed_dressing_issues, physics_inventory

ROOT = '/World/LaboratoryTask'
DRESSING = ROOT+'/DesktopDressing'


def _stage(tmp_path, name='task.usda'):
    stage = Usd.Stage.CreateNew(str(tmp_path/name))
    UsdGeom.Xform.Define(stage, ROOT)
    cube = UsdGeom.Cube.Define(stage, ROOT+'/Object/Shape')
    cube.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return stage, cube


def test_default_equal_mass_with_new_time_samples_is_detected(tmp_path):
    stage, cube = _stage(tmp_path)
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(.1)
    stage.GetRootLayer().Save()
    before = physics_inventory(stage.GetRootLayer().realPath)
    mass.Set(.1, 0.)
    mass.Set(10., 1.)
    stage.GetRootLayer().Save()
    after = physics_inventory(stage.GetRootLayer().realPath)
    assert before != after
    assert after[str(cube.GetPath())]['fields']['physics:mass']['time_samples']['1'] == '10.0'


def test_inherited_collision_material_rebinding_is_detected(tmp_path):
    stage, cube = _stage(tmp_path)
    holder = stage.GetPrimAtPath(ROOT+'/Object')
    materials = []
    for name in ('Original', 'Changed'):
        mat = UsdShade.Material.Define(stage, ROOT+'/Materials/'+name)
        UsdPhysics.MaterialAPI.Apply(mat.GetPrim()).CreateStaticFrictionAttr(.3)
        materials.append(mat)
    binding = UsdShade.MaterialBindingAPI.Apply(holder)
    binding.Bind(materials[0], materialPurpose='physics')
    stage.GetRootLayer().Save()
    before = physics_inventory(stage.GetRootLayer().realPath)
    binding.Bind(materials[1], materialPurpose='physics')
    stage.GetRootLayer().Save()
    after = physics_inventory(stage.GetRootLayer().realPath)
    path = str(cube.GetPath())
    assert before[path] != after[path]
    assert after[path]['fields']['resolved_physics_material'].endswith('/Changed')


def test_animated_parent_transform_cannot_hide_behind_same_default_pose(tmp_path):
    stage, cube = _stage(tmp_path)
    parent = UsdGeom.Xform.Define(stage, ROOT+'/Object')
    translate = parent.AddTranslateOp()
    translate.Set(Gf.Vec3d(0.))
    stage.GetRootLayer().Save()
    before = physics_inventory(stage.GetRootLayer().realPath)
    translate.Set(Gf.Vec3d(0.), 0.)
    translate.Set(Gf.Vec3d(1., 0., 0.), 1.)
    stage.GetRootLayer().Save()
    after = physics_inventory(stage.GetRootLayer().realPath)
    path = str(cube.GetPath())
    assert before[path]['fields']['world_transform'] == after[path]['fields']['world_transform']
    assert before[path] != after[path]


def _background(tmp_path):
    stage = Usd.Stage.CreateNew(str(tmp_path/'background.usda'))
    UsdGeom.Xform.Define(stage, DRESSING)
    cube = UsdGeom.Cube.Define(stage, DRESSING+'/FixedSupply')
    cube.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    metadata = {'root_path': DRESSING, 'collider_count': 1, 'dynamic_actor_count': 0,
        'items': [{'id': 'supply', 'collider_paths': [str(cube.GetPath())],
                   'dynamic': False, 'task_object': False, 'scoring_support': False}]}
    stage.GetRootLayer().Save()
    return stage, cube, metadata


def test_fixed_background_is_accepted_only_with_complete_inventory(tmp_path):
    stage, _, metadata = _background(tmp_path)
    path = stage.GetRootLayer().realPath
    assert fixed_dressing_issues(path, metadata) == []
    missing = deepcopy(metadata)
    missing['items'][0]['collider_paths'] = []
    assert any('inventory' in issue for issue in fixed_dressing_issues(path, missing))
    scoring = deepcopy(metadata)
    scoring['items'][0]['scoring_support'] = True
    assert any('support' in issue for issue in fixed_dressing_issues(path, scoring))


def test_dressing_namespace_does_not_allow_dynamic_mass_or_physics_world(tmp_path):
    stage, cube, metadata = _background(tmp_path)
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.)
    UsdPhysics.Scene.Define(stage, DRESSING+'/PhysicsScene')
    stage.GetRootLayer().Save()
    issues = fixed_dressing_issues(stage.GetRootLayer().realPath, metadata)
    assert any('Non-static background' in issue and 'FixedSupply' in issue for issue in issues)
    assert any('Non-static background' in issue and 'PhysicsScene' in issue for issue in issues)


def test_background_animation_is_not_static(tmp_path):
    stage, cube, metadata = _background(tmp_path)
    cube.CreateSizeAttr().Set(.2, 1.)
    stage.GetRootLayer().Save()
    assert any('Animated fixed background' in issue
               for issue in fixed_dressing_issues(stage.GetRootLayer().realPath, metadata))
