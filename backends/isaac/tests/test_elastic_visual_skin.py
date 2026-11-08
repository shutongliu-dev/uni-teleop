"""Skinning uses native samples while physics and the rest mesh stay unchanged."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from bench.simulation.isaacsim_backend.elastic_visual_skin import ElasticVisualSkin, ROOT


def test_fresh_reset_recreates_only_owned_nonphysical_skin_output():
    from bench.simulation.isaacsim_backend.elastic_visual_skin import clear_owned_skins_before_reset
    stage, config, objects, _ = fixture()
    before = stage.GetRootLayer().ExportToString()
    skin = ElasticVisualSkin(stage, config)
    skin.sync(objects, 55)
    clear_owned_skins_before_reset(stage)
    assert stage.GetRootLayer().ExportToString() == before
    recreated = ElasticVisualSkin(stage, config)
    assert recreated.sync(objects, 0)['source_native_step'] == 0


def test_visual_reset_refuses_unowned_or_physical_output():
    from bench.simulation.isaacsim_backend.elastic_visual_skin import clear_owned_skins_before_reset
    stage, config, _, _ = fixture()
    UsdGeom.Xform.Define(stage, ROOT)
    with pytest.raises(ValueError, match='Unowned'):
        clear_owned_skins_before_reset(stage)
    stage.RemovePrim(ROOT)
    ElasticVisualSkin(stage, config)
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(ROOT))
    with pytest.raises(ValueError, match='Physical prim'):
        clear_owned_skins_before_reset(stage)


def fixture():
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, '/World/Body')
    body.AddTranslateOp().Set(Gf.Vec3d(4., 5., 6.))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.0792)
    mesh = UsdGeom.Mesh.Define(stage, '/World/Body/RestSkin')
    points = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateVisibilityAttr('invisible')
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying')
    uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    material = UsdShade.Material.Define(stage, '/World/Material')
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material)
    cells = [{'id': 'a', 'composite_rest_center_m': [-.5, 0., 0.]},
             {'id': 'b', 'composite_rest_center_m': [.5, 0., 0.]}]
    binding = {'prim_path': str(mesh.GetPath()), 'rest_points_m': points.tolist(),
        'member_ids': ['a', 'b'], 'vertex_member_indices': [[0, 1]]*4,
        'vertex_weights': [[.75, .25], [.25, .75], [.25, .75], [.75, .25]]}
    composites = {'sponge': {'cells': cells, 'visual_skin_bindings': [binding]}}
    poses = {row['id']: {'position_m': row['composite_rest_center_m'], 'quaternion_wxyz': [1., 0., 0., 0.]}
             for row in cells}
    return stage, composites, poses, points


def test_whole_rigid_transform_preserves_mesh_uv_material_and_physics():
    stage, config, objects, points = fixture()
    source = stage.GetPrimAtPath('/World/Body/RestSkin')
    old_attributes = {a.GetName(): a.Get() for a in source.GetAttributes()}
    old_body = {a.GetName(): a.Get() for a in stage.GetPrimAtPath('/World/Body').GetAttributes()}
    skin = ElasticVisualSkin(stage, config)
    rotation = Rotation.from_euler('xyz', [.2, -.3, .8])
    xyz = np.array([1., 2., 3.])
    q = rotation.as_quat()
    for row in objects.values():
        row['position_m'] = (rotation.apply(row['position_m'])+xyz).tolist()
        row['quaternion_wxyz'] = [q[3], *q[:3]]
    receipt = skin.sync(objects, 12)
    output = UsdGeom.Mesh(stage.GetPrimAtPath(receipt['output_paths'][0]))
    np.testing.assert_allclose(output.GetPointsAttr().Get(), rotation.apply(points)+xyz, atol=3e-7)
    assert list(output.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 3]
    np.testing.assert_array_equal(UsdGeom.PrimvarsAPI(output).GetPrimvar('st').Get(),
                                  UsdGeom.PrimvarsAPI(source).GetPrimvar('st').Get())
    material, _ = UsdShade.MaterialBindingAPI(output).ComputeBoundMaterial()
    assert str(material.GetPath()) == '/World/Material'
    np.testing.assert_array_equal(UsdGeom.XformCache().GetLocalToWorldTransform(output.GetPrim()), np.eye(4))
    assert {a.GetName(): a.Get() for a in source.GetAttributes()} == old_attributes
    assert {a.GetName(): a.Get() for a in stage.GetPrimAtPath('/World/Body').GetAttributes()} == old_body
    assert not output.GetPrim().HasAPI(UsdPhysics.CollisionAPI)


def test_local_compression_follows_weights_and_duplicate_ticks_are_idempotent():
    stage, config, objects, points = fixture()
    skin = ElasticVisualSkin(stage, config)
    objects['b']['position_m'] = [.5, 0., -.2]
    receipt = skin.sync(objects, 5)
    output = UsdGeom.Mesh(stage.GetPrimAtPath(receipt['output_paths'][0]))
    expected = points.copy()
    expected[:, 2] = [-.05, -.15, -.15, -.05]
    np.testing.assert_allclose(output.GetPointsAttr().Get(), expected, atol=1e-8)
    before = stage.GetRootLayer().ExportToString()
    assert skin.sync(objects, 5)['changed'] is False
    assert stage.GetRootLayer().ExportToString() == before
    objects['a']['position_m'] = [-.5, 0., 1.]
    with pytest.raises(ValueError, match='same native tick'):
        skin.sync(objects, 5)
    assert stage.GetRootLayer().ExportToString() == before
    with pytest.raises(ValueError, match='backwards'):
        skin.sync(objects, 4)


@pytest.mark.parametrize('weights', [[[.7, .2]]*4, [[1.1, -.1]]*4, [[float('nan'), 0.]]*4])
def test_invalid_weights_are_rejected_before_any_output_authoring(weights):
    stage, config, _, _ = fixture()
    config['sponge']['visual_skin_bindings'][0]['vertex_weights'] = weights
    with pytest.raises(ValueError, match='normalized member weights'):
        ElasticVisualSkin(stage, config)
    assert not stage.GetPrimAtPath(ROOT)


def test_reject_collision_target_or_physics_added_after_creation():
    stage, config, objects, _ = fixture()
    collision = UsdGeom.Mesh.Define(stage, '/World/Collider')
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    bad = deepcopy(config)
    bad['sponge']['visual_skin_bindings'][0]['output_prim_path'] = '/World/Collider'
    with pytest.raises(ValueError, match='noncollision'):
        ElasticVisualSkin(stage, bad)
    skin = ElasticVisualSkin(stage, config)
    output = skin.bindings[0]['mesh']
    UsdPhysics.CollisionAPI.Apply(output.GetPrim())
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='noncollision'):
        skin.sync(objects, 1)
    assert stage.GetRootLayer().ExportToString() == before


def test_missing_or_invalid_native_sample_never_partially_updates_output():
    stage, config, objects, _ = fixture()
    skin = ElasticVisualSkin(stage, config)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='missing a native member'):
        skin.sync({'a': objects['a']}, 0)
    objects['b']['quaternion_wxyz'] = [0., 0., 0., 0.]
    with pytest.raises(ValueError, match='unit length'):
        skin.sync(objects, 0)
    assert stage.GetRootLayer().ExportToString() == before


def test_left_handed_template_retains_its_orientation_and_normal_direction():
    stage, config, objects, _ = fixture()
    UsdGeom.Mesh(stage.GetPrimAtPath('/World/Body/RestSkin')).CreateOrientationAttr('leftHanded')
    skin = ElasticVisualSkin(stage, config)
    receipt = skin.sync(objects, 0)
    output = UsdGeom.Mesh(stage.GetPrimAtPath(receipt['output_paths'][0]))
    assert output.GetOrientationAttr().Get() == 'leftHanded'
    np.testing.assert_allclose(output.GetNormalsAttr().Get(), [[0., 0., -1.]]*4)


def binding_fixture(*, compact):
    stage, config, _, points = fixture()
    cells = [{'id': 'cell_'+str(i), 'composite_rest_center_m': [x, y, z]}
             for i, (x, y, z) in enumerate([
                 (-1., -1., 0.), (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.),
                 (0., 0., -.1), (0., 0., .1), (.3, .1, .2), (-.3, .1, .2),
                 (.3, -.1, .2), (-.3, -.1, .2)])]
    config['sponge']['cells'] = cells
    binding = config['sponge']['visual_skin_bindings'][0]
    binding['member_ids'] = [c['id'] for c in cells]
    # This is the original asset author's operation sequence, including ties
    # and exact zero distances; the compact reader must reproduce it exactly.
    centers = np.array([c['composite_rest_center_m'] for c in cells])
    d2 = np.sum((points[:, None]-centers[None])**2, axis=2)
    nearest = np.argsort(d2, axis=1)[:, :8]
    weights = 1/np.maximum(np.take_along_axis(d2, nearest, axis=1), 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)
    binding['vertex_member_indices'] = nearest.tolist()
    binding['vertex_weights'] = weights.tolist()
    if compact:
        for key in ('rest_points_m', 'vertex_member_indices', 'vertex_weights'):
            del binding[key]
        binding.update(binding_schema='eight_nearest_inverse_squared_v1', neighbor_count=8,
                       minimum_squared_distance_m2=1e-12, point_count=len(points),
                       rest_points_sha256=hashlib.sha256(np.asarray(points, np.float64).tobytes()).hexdigest())
    # Subset material and UV indices must survive as well as parent bindings.
    source = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Body/RestSkin'))
    UsdGeom.PrimvarsAPI(source).GetPrimvar('st').SetIndices([3, 2, 1, 0])
    subset = UsdGeom.Subset.Define(stage, '/World/Body/RestSkin/Subset')
    subset.CreateElementTypeAttr('face')
    subset.CreateFamilyNameAttr('materialBind')
    subset.CreateIndicesAttr([0])
    material = UsdShade.Material.Define(stage, '/World/SubsetMaterial')
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    poses = {c['id']: {'position_m': c['composite_rest_center_m'], 'quaternion_wxyz': [1., 0., 0., 0.]}
             for c in cells}
    return stage, config, poses


@pytest.mark.parametrize('deform', [False, True])
def test_compact_binding_matches_expanded_points_normals_uv_materials_without_physics_writes(deform):
    outputs, plans = [], []
    for compact in (False, True):
        stage, config, objects = binding_fixture(compact=compact)
        physics_before = {str(p.GetPath()): {a.GetName(): a.Get() for a in p.GetAttributes()}
                          for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Body'))}
        skin = ElasticVisualSkin(stage, config)
        plans.append(skin.bindings[0])
        rotation = Rotation.from_euler('xyz', [.2, -.3, .8])
        quat = rotation.as_quat()
        for i, pose in enumerate(objects.values()):
            point = np.array(pose['position_m'], float)
            if deform:
                point[2] -= .01*(i+1)
            pose['position_m'] = (rotation.apply(point)+[.1, .2, .3]).tolist()
            pose['quaternion_wxyz'] = [quat[3], *quat[:3]]
        skin.sync(objects, 0)
        assert {str(p.GetPath()): {a.GetName(): a.Get() for a in p.GetAttributes()}
                for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Body'))} == physics_before
        out = stage.GetPrimAtPath(ROOT+'/sponge/skin_0')
        outputs.append({str(p.GetPath()): {
            'attributes': {a.GetName(): a.Get() for a in p.GetAttributes()},
            'relationships': {r.GetName(): r.GetTargets() for r in p.GetRelationships()}}
            for p in Usd.PrimRange(out)})
    np.testing.assert_array_equal(plans[0]['indices'], plans[1]['indices'])
    np.testing.assert_array_equal(plans[0]['weights'], plans[1]['weights'])
    np.testing.assert_array_equal(plans[0]['local'], plans[1]['local'])
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('key,value', [
    ('binding_schema', 'unknown'), ('neighbor_count', True), ('neighbor_count', 8.),
    ('neighbor_count', 7), ('point_count', True), ('point_count', 3),
    ('minimum_squared_distance_m2', 0.), ('minimum_squared_distance_m2', float('nan')),
    ('rest_points_sha256', '0'*64), ('rest_points_m', [[0., 0., 0.]])])
def test_compact_binding_rejects_mismatch_before_writing_any_output(key, value):
    stage, config, _ = binding_fixture(compact=True)
    config['sponge']['visual_skin_bindings'][0][key] = value
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='compact binding'):
        ElasticVisualSkin(stage, config)
    assert stage.GetRootLayer().ExportToString() == before


def test_compact_binding_rejects_template_changed_since_hash():
    stage, config, _ = binding_fixture(compact=True)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Body/RestSkin'))
    points = np.asarray(mesh.GetPointsAttr().Get()).copy()
    points[0, 0] += .0001
    mesh.GetPointsAttr().Set(points.tolist())
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='immutable template points'):
        ElasticVisualSkin(stage, config)
    assert stage.GetRootLayer().ExportToString() == before
