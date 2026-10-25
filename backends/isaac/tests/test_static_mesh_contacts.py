"""Original geometry, ownership boundaries and continuous fixed-actor guards."""
import numpy as np
import pytest

from bench.simulation.isaacsim_backend.static_mesh_contacts import MODE, FixedActorGuard, adapt_static_meshes, geometry_signature


@pytest.fixture
def usd():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


def mesh(stage, path, usd):
    _, geom, physics = usd
    value = geom.Mesh.Define(stage, path)
    value.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    value.CreateFaceVertexCountsAttr([3])
    value.CreateFaceVertexIndicesAttr([0, 1, 2])
    physics.CollisionAPI.Apply(value.GetPrim())
    physics.MeshCollisionAPI.Apply(value.GetPrim()).CreateApproximationAttr('none')
    return value.GetPrim()


def test_absent_opt_in_is_byte_identical_and_invalid_mode_fails(usd):
    stage = usd[0].Stage.CreateInMemory()
    mesh(stage, '/World/Mesh', usd)
    before = stage.GetRootLayer().ExportToString()
    assert adapt_static_meshes(stage) is None
    assert before == stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='Unknown'):
        adapt_static_meshes(stage, 'implicit')


def test_original_transformed_mesh_and_material_survive(usd):
    from pxr import UsdShade
    stage = usd[0].Stage.CreateInMemory()
    parent = usd[1].Xform.Define(stage, '/World/Fixture')
    parent.AddTranslateOp().Set((1, 2, 3))
    parent.AddRotateZOp().Set(35)
    parent.AddScaleOp().Set((.2, .3, .4))
    prim = mesh(stage, '/World/Fixture/Mesh', usd)
    material = UsdShade.Material.Define(stage, '/World/Material')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose='physics')
    before = geometry_signature(prim)
    result = adapt_static_meshes(stage, MODE)
    assert result['kinematic_body_paths'] == ['/World/Fixture/Mesh']
    assert result['collider_actor_paths'] == {'/World/Fixture/Mesh': '/World/Fixture/Mesh'}
    assert geometry_signature(prim) == before
    assert usd[2].RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True


def test_disabled_instance_owner_preserves_compound_and_source(usd, tmp_path):
    path = tmp_path / 'source.usda'
    source = usd[0].Stage.CreateNew(str(path))
    root = usd[1].Xform.Define(source, '/Asset').GetPrim()
    usd[2].RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(False)
    mesh(source, '/Asset/First', usd)
    mesh(source, '/Asset/Second', usd)
    source.GetRootLayer().Save()
    original = path.read_bytes()
    stage = usd[0].Stage.CreateInMemory()
    owner = usd[1].Xform.Define(stage, '/World/Compound').GetPrim()
    owner.GetReferences().AddReference(str(path), '/Asset')
    owner.SetInstanceable(True)
    result = adapt_static_meshes(stage, MODE)
    assert owner.IsInstance()
    assert result['kinematic_body_paths'] == ['/World/Compound']
    assert set(result['collider_actor_paths'].values()) == {'/World/Compound'}
    assert len(result['collider_actor_paths']) == 2
    assert path.read_bytes() == original


def test_nested_disabled_bodies_fail_before_any_mutation(usd):
    stage = usd[0].Stage.CreateInMemory()
    for path in ('/World/A', '/World/A/B'):
        usd[2].RigidBodyAPI.Apply(usd[1].Xform.Define(stage, path).GetPrim()).CreateRigidBodyEnabledAttr(False)
    mesh(stage, '/World/A/B/Mesh', usd)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='Nested disabled'):
        adapt_static_meshes(stage, MODE)
    assert stage.GetRootLayer().ExportToString() == before


def test_enabled_objects_and_excluded_deformables_unchanged(usd):
    stage = usd[0].Stage.CreateInMemory()
    active = usd[1].Xform.Define(stage, '/World/Object').GetPrim()
    usd[2].RigidBodyAPI.Apply(active)
    mesh(stage, '/World/Object/Mesh', usd)
    mesh(stage, '/World/Soft/Mesh', usd)
    before = stage.GetRootLayer().ExportToString()
    result = adapt_static_meshes(stage, MODE, exclude_paths=['/World/Soft'])
    assert result['kinematic_body_paths'] == []
    assert stage.GetRootLayer().ExportToString() == before


# rooms_20261017_v3 includes additional perimeter props; every original and new
# mesh must still have exactly one unambiguous fixed native actor.
@pytest.mark.parametrize('family,expected', [('mechanical_assembly', 28), ('laboratory', 13),
    ('kitchen', 44), ('study', 4), ('card_room', 55), ('living_room', 15)])
def test_all_published_meshes_have_unambiguous_actors(usd, family, expected):
    from scenes.workrooms.registry import reference
    from bench.simulation.isaacsim_backend.workroom import add_workroom, digest
    spec = reference(family)
    stage = usd[0].Stage.CreateInMemory()
    room = add_workroom(stage, spec)
    result = adapt_static_meshes(stage, MODE)
    assert len(result['mesh_approximations']) == expected
    assert not set(room['contact_unobserved_paths']) - set(result['collider_actor_paths'])
    assert digest(spec['usd_path']) == spec['usd_sha256']
    for collider, actor in result['collider_actor_paths'].items():
        body = usd[2].RigidBodyAPI(stage.GetPrimAtPath(actor))
        assert body.GetRigidBodyEnabledAttr().Get() and body.GetKinematicEnabledAttr().Get()
        current = stage.GetPrimAtPath(collider)
        while str(current.GetPath()) != actor:
            assert not current.HasAPI(usd[2].RigidBodyAPI)
            current = current.GetParent()


def test_guard_checks_entire_batch_and_catches_transient_motion():
    initial = np.array([[0., 0, 0, 0, 0, 0, 1], [2, 3, 4, 0, 0, 0, 1]])
    guard = FixedActorGuard(['/A', '/B'], initial)
    same = initial.copy()
    same[:, 3:] *= -1
    guard.check(['/A', '/B'], same)
    moved = initial.copy()
    moved[1, 0] += .0001
    with pytest.raises(RuntimeError, match='moved'):
        guard.check(['/A', '/B'], moved)
    guard.check(['/A', '/B'], initial)
    assert guard.report()['maximum_translation_m'] > 1.e-6
    assert guard.report()['checked_batches'] == 3
    with pytest.raises(RuntimeError, match='order'):
        guard.check(['/B', '/A'], initial)


def test_probe_preserves_holes_and_original_compound_owner(usd, tmp_path):
    from scripts.validation.validate_static_mesh_contacts import make_probe_scene
    from bench.simulation.isaacsim_backend.workroom import add_workroom
    spec = make_probe_scene(tmp_path)
    stage = usd[0].Stage.CreateInMemory()
    add_workroom(stage, spec['workroom'])
    contract = adapt_static_meshes(stage, spec['isaacsim']['static_mesh_contact_mode'])
    assert len(contract['kinematic_body_paths']) == 3
    assert len(contract['mesh_approximations']) == 6
    assert sorted(contract['mesh_approximations'].values()) == ['convexDecomposition', 'none', 'none', 'none', 'none', 'sdf']
    convex = stage.GetPrimAtPath('/World/Room/Geometry/convex')
    assert convex.GetAttribute('physxConvexDecompositionCollision:shrinkWrap').Get() is True
    assert len(spec['objects']) == 6
    assert all(spec['workroom_diagnostics'][x['id']]['support_top_z_m'] == 0
               for x in spec['objects'] if x['id'].endswith('_hole'))


def test_probe_meshes_use_local_centres_without_mass_overrides(usd, tmp_path):
    from scripts.validation.validate_static_mesh_contacts import make_probe_scene
    spec = make_probe_scene(tmp_path)
    stage = usd[0].Stage.Open(spec['workroom']['usd_path'])
    cache = usd[1].XformCache()
    for name, x in [('triangle_compound', .45), ('convex', 1.2), ('sdf', 1.95)]:
        root = stage.GetPrimAtPath('/World/Geometry/' + name)
        all_points = []
        for prim in usd[0].PrimRange(root):
            assert not prim.HasAPI(usd[2].MassAPI)
            if not prim.IsA(usd[1].Mesh):
                continue
            points = np.asarray(usd[1].Mesh(prim).GetPointsAttr().Get())
            assert np.max(np.abs(points[:, :2])) <= .25
            assert np.max(np.abs(points[:, 2])) <= .020001
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
            all_points.extend((np.column_stack([points, np.ones(len(points))]) @ matrix)[:, :3])
        world = np.asarray(all_points)
        assert np.allclose(world.min(axis=0), [x-.25, -.25, .76], atol=1.e-7, rtol=0)
        assert np.allclose(world.max(axis=0), [x+.25, .25, .8], atol=1.e-7, rtol=0)


def test_sdf_fixture_is_watertight_oriented_ring_with_analytic_area_and_volume(usd, tmp_path):
    from collections import Counter
    from scripts.validation.validate_static_mesh_contacts import make_probe_scene
    spec = make_probe_scene(tmp_path)
    stage = usd[0].Stage.Open(spec['workroom']['usd_path'])
    mesh = usd[1].Mesh(stage.GetPrimAtPath('/World/Geometry/sdf'))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    assert points.shape == (16, 3) and triangles.shape == (32, 3)
    assert len({tuple(p) for p in points}) == 16
    assert len({tuple(sorted(t)) for t in triangles}) == 32
    directed = Counter((int(t[i]), int(t[(i+1) % 3])) for t in triangles for i in range(3))
    undirected = Counter(tuple(sorted(edge)) for edge in directed.elements())
    assert len(undirected) == 48 and set(undirected.values()) == {2}
    assert all(count == 1 and directed[b, a] == 1 for (a, b), count in directed.items())
    assert len(points) - len(undirected) + len(triangles) == 0  # genus-one boundary
    a, b, c = (points[triangles[:, i]] for i in range(3))
    normal = np.cross(b-a, c-a)
    assert np.all(np.linalg.norm(normal, axis=1) > 0)
    area = float(np.linalg.norm(normal, axis=1).sum()/2)
    volume = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()/6)
    assert area == pytest.approx(.532, abs=1.e-8)
    assert volume == pytest.approx(.0084, abs=1.e-9)
    # Offsets into the outward and inward halfspaces distinguish every face,
    # including the hole walls; an inverted local face cannot pass by volume.
    centres = (a+b+c)/3
    unit = normal/np.linalg.norm(normal, axis=1)[:, None]
    def inside(p):
        return (np.max(np.abs(p[:2])) < .25 and np.max(np.abs(p[:2])) > .10 and abs(p[2]) < .02)
    assert all(inside(p - n*1.e-5) and not inside(p + n*1.e-5) for p, n in zip(centres, unit))
