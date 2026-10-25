"""Generic fixture-only triangle representation with real CPU OpenUSD composition."""
from copy import deepcopy
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.asset_catalog import validate_triangle_mesh_role, prepare_triangle_mesh


@pytest.mark.parametrize('dynamic,declared', [(True, False), (True, True), (False, True), (False, None)])
def test_raw_triangle_mesh_cannot_be_a_dynamic_object(dynamic, declared):
    asset = {'dynamic': declared, 'collision_parts': [{'geometry': 'triangle_mesh'}]}
    with pytest.raises(ValueError, match='non-dynamic fixtures'):
        validate_triangle_mesh_role(asset, dynamic=dynamic)


INVALID_MESH_CASES = ('empty_vertices', 'wrong_vertex_shape', 'nan_vertex', 'inf_vertex',
    'empty_faces', 'wrong_face_shape', 'float_indices', 'bool_indices', 'negative_index',
    'out_of_range_index', 'repeated_index', 'collinear_points', 'wrong_scale_shape',
    'zero_scale', 'negative_scale', 'nan_scale', 'inf_scale', 'bool_scale',
    'scale_overflow', 'vertex_overflow', 'float32_collapse', 'scale_underflow')


def triangle_inputs(case=None):
    points = np.array([[0., 0., 0.], [.03, 0., 0.], [0., .03, 0.]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    scale = [1., 1., 1.]
    if case == 'empty_vertices': points = np.empty((0, 3))
    elif case == 'wrong_vertex_shape': points = points[:, :2]
    elif case == 'nan_vertex': points[0, 0] = np.nan
    elif case == 'inf_vertex': points[0, 0] = np.inf
    elif case == 'empty_faces': faces = np.empty((0, 3), dtype=np.int64)
    elif case == 'wrong_face_shape': faces = np.array([[0, 1, 2, 0]])
    elif case == 'float_indices': faces = faces.astype(float)
    elif case == 'bool_indices': faces = faces.astype(bool)
    elif case == 'negative_index': faces[0, 0] = -1
    elif case == 'out_of_range_index': faces[0, 2] = len(points)
    elif case == 'repeated_index': faces[0, 1] = faces[0, 0]
    elif case == 'collinear_points': points[2] = points[1]*2
    elif case == 'wrong_scale_shape': scale = [1., 1.]
    elif case == 'zero_scale': scale[0] = 0.
    elif case == 'negative_scale': scale[0] = -1.
    elif case == 'nan_scale': scale[0] = np.nan
    elif case == 'inf_scale': scale[0] = np.inf
    elif case == 'bool_scale': scale[0] = True
    elif case == 'scale_overflow': scale[0] = 1e39
    elif case == 'vertex_overflow': points[1, 0] = 1e39
    elif case == 'float32_collapse': points = np.array([[1e9, 0., 0.], [1e9+1., 0., 0.], [1e9, 1., 0.]])
    elif case == 'scale_underflow': scale[0] = 1e-60
    return points, faces, scale


@pytest.mark.parametrize('case', INVALID_MESH_CASES)
def test_triangle_mesh_input_validation_rejects_malformed_or_collapsed_geometry(case):
    with pytest.raises(ValueError, match='triangle_mesh'):
        prepare_triangle_mesh(*triangle_inputs(case))


def test_triangle_mesh_keeps_valid_open_surface_and_existing_float32_scaling():
    points, faces, _ = triangle_inputs()
    original_points, original_faces = points.copy(), faces.copy()
    scale = [2., 3., .5]
    vertices, indices = prepare_triangle_mesh(points, faces, scale)
    expected = points.astype(np.float32)
    expected *= np.asarray(scale, dtype=np.float32)
    np.testing.assert_array_equal(vertices, expected)
    np.testing.assert_array_equal(indices, faces)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(faces, original_faces)
    assert vertices.dtype == np.float32 and indices.dtype == np.int32
    assert not np.shares_memory(vertices, points) and not np.shares_memory(indices, faces)


@pytest.fixture
def scene(monkeypatch):
    pxr = pytest.importorskip('pxr')
    from pxr import Sdf
    class CollisionSchema:
        def __init__(self, prim): self.prim = prim
        @classmethod
        def Apply(cls, prim): return cls(prim)
        def CreateContactOffsetAttr(self, value):
            return self.prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(value)
        def CreateRestOffsetAttr(self, value):
            return self.prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(value)
    monkeypatch.setattr(pxr, 'PhysxSchema', SimpleNamespace(PhysxCollisionAPI=CollisionSchema), raising=False)
    monkeypatch.delitem(sys.modules, 'bench.simulation.isaacsim_backend.scene', raising=False)
    module = importlib.import_module('bench.simulation.isaacsim_backend.scene')
    from pxr import UsdShade, UsdPhysics
    def material(stage, name, friction):
        value = UsdShade.Material.Define(stage, '/World/PhysicsMaterials/'+name)
        api = UsdPhysics.MaterialAPI.Apply(value.GetPrim())
        api.CreateStaticFrictionAttr(friction)
        api.CreateDynamicFrictionAttr(friction)
        api.CreateRestitutionAttr(0.)
        return value
    monkeypatch.setattr(module, 'rigid_material', material)
    return module


@pytest.mark.parametrize('case', INVALID_MESH_CASES)
@pytest.mark.parametrize('kind', ['collision', 'visual'])
def test_all_triangle_parts_are_preflighted_before_authoring(scene, monkeypatch, case, kind):
    from pxr import Usd, UsdGeom
    good_points, good_faces, _ = triangle_inputs()
    bad_points, bad_faces, scale = triangle_inputs(case)
    part = {'geometry': 'triangle_mesh', 'mesh_path': '/valid.obj'}
    asset = {'dynamic': False, 'visual_parts': [deepcopy(part)],
             'collision_parts': [deepcopy(part)], 'manifest_sha256': 'a'*64, 'asset_root': '/test'}
    asset[kind+'_parts'].append({'geometry': 'triangle_mesh', 'mesh_path': '/invalid.obj', 'scale': scale})
    reads = []
    monkeypatch.setattr(scene, 'load_task_asset', lambda *a, **k: deepcopy(asset))
    def read(path):
        reads.append(path)
        return ((bad_points.copy(), bad_faces.copy()) if path == '/invalid.obj'
                else (good_points.copy(), good_faces.copy()))
    monkeypatch.setattr(scene, 'load_obj_mesh', read)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, '/Existing')
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='triangle_mesh'):
        scene.add_task_asset(stage, {'id': 'rejected', 'asset_id': 'fixture', 'position_m': [0., 0., 0.]}, dynamic=False)
    assert stage.GetRootLayer().ExportToString() == before
    assert '/valid.obj' in reads and '/invalid.obj' in reads


@pytest.mark.parametrize('kind,approximation', [('mesh', 'convexHull'), ('convex_mesh', 'convexHull'), ('triangle_mesh', 'none')])
def test_explicit_triangle_preserves_geometry_and_existing_convex_defaults(scene, monkeypatch, kind, approximation):
    from pxr import Usd, UsdGeom, UsdPhysics
    from bench.simulation.isaacsim_backend.static_mesh_contacts import adapt_static_meshes, geometry_signature
    points = np.array([[0., 0., 0.], [.03, 0., 0.], [0., .03, 0.]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    part = {'geometry': kind, 'mesh_path': '/test/mesh.obj'}
    asset = {'dynamic': False, 'visual_parts': [], 'collision_parts': [part],
             'manifest_sha256': 'a'*64, 'asset_root': '/test'}
    monkeypatch.setattr(scene, 'load_task_asset', lambda *a, **k: deepcopy(asset))
    monkeypatch.setattr(scene, 'load_obj_mesh', lambda *a: (points.copy(), faces.copy()))
    stage = Usd.Stage.CreateInMemory()
    info = scene.add_task_asset(stage, {'id': 'arbitrary_support', 'asset_id': 'support',
        'position_m': [1., 2., 3.], 'friction': .4}, dynamic=False)
    path = '/World/Fixtures/arbitrary_support/collision_000'
    prim = stage.GetPrimAtPath(path)
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == approximation
    np.testing.assert_array_equal(UsdGeom.Mesh(prim).GetPointsAttr().Get(), points)
    np.testing.assert_array_equal(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), faces.reshape(-1))
    before = geometry_signature(prim)
    result = adapt_static_meshes(stage, 'kinematic_mesh_v1')
    assert result['collider_actor_paths'][path] == path
    assert info['collision_paths'] == [path]
    assert geometry_signature(prim) == before
    assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True
    if kind == 'triangle_mesh':
        with pytest.raises(ValueError, match='non-dynamic fixtures'):
            scene.add_task_asset(stage, {'id': 'rejected', 'asset_id': 'support'}, dynamic=True)
        assert not stage.GetPrimAtPath('/World/Objects/rejected')
