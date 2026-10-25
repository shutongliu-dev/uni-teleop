"""Opt-in support materials preserve scenery and report native evidence honestly."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from bench.simulation.isaacsim_backend import workroom_materials as materials
from bench.simulation.isaacsim_backend.workroom import SUPPORT_PATHS, add_workroom, digest


def request():
    return {'schema': materials.SCHEMA, 'surfaces': {
        name: {'static_friction': .4, 'dynamic_friction': .3, 'restitution': .0,
               'friction_combine': 'average', 'restitution_combine': 'max',
               'parameter_source': f'{name}: explicit simulation prior for this test'}
        for name in ('table', 'floor')}}


@pytest.fixture
def stage():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    value = Usd.Stage.CreateInMemory()
    old = UsdShade.Material.Define(value, '/World/OriginalMaterial')
    UsdPhysics.MaterialAPI.Apply(old.GetPrim()).CreateStaticFrictionAttr(.7)
    for path in [*SUPPORT_PATHS.values(), '/World/Room/Decoration']:
        prim = UsdGeom.Cube.Define(value, path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(old, materialPurpose='physics')
    return value


@pytest.mark.parametrize('field,value', [
    ('static_friction', float('inf')), ('dynamic_friction', float('nan')),
    ('static_friction', True), ('dynamic_friction', '0.3'), ('restitution', -1.),
    ('restitution', 1.1), ('static_friction', .1), ('friction_combine', 'maximum'),
    ('restitution_combine', None), ('parameter_source', ''), ('parameter_source', None),
])
def test_invalid_physics_or_missing_provenance_rejected(field, value):
    spec = request()
    spec['surfaces']['table'][field] = value
    with pytest.raises(ValueError):
        materials.material_options(spec)


@pytest.mark.parametrize('edit', [
    lambda spec: spec.update(schema='other'),
    lambda spec: spec.update(extra=True),
    lambda spec: spec.update(surfaces={}),
    lambda spec: spec['surfaces'].update(decoration=spec['surfaces']['table']),
    lambda spec: spec['surfaces']['table'].update(rolling_friction=.1),
    lambda spec: spec['surfaces']['floor'].pop('dynamic_friction'),
])
def test_unknown_fields_and_implicit_coefficients_rejected(edit):
    spec = request()
    edit(spec)
    with pytest.raises(ValueError):
        materials.material_options(spec)


def test_absent_opt_in_leaves_stage_and_legacy_result_unchanged(stage):
    before = stage.GetRootLayer().ExportToString()
    assert materials.apply_materials(stage, SUPPORT_PATHS) is None
    assert stage.GetRootLayer().ExportToString() == before
    assert materials.runtime_material_check(None, None, {}) is None


def test_only_explicit_supports_bound_and_numeric_fields_resolve(stage):
    from pxr import UsdPhysics, UsdShade
    before = deepcopy(request())
    record = materials.apply_materials(stage, SUPPORT_PATHS, before)
    assert before == request()
    assert record['physics_accepted'] is False
    assert record['combine_priority'] == ['average', 'min', 'multiply', 'max']
    for name, row in record['surfaces'].items():
        actual = row['usd_resolved']
        assert actual['material_path'] == f'{materials.MATERIAL_ROOT}/{name}'
        assert actual['static_friction'] == pytest.approx(.4)
        assert actual['dynamic_friction'] == pytest.approx(.3)
        assert actual['friction_combine'] == 'average'
        assert actual['restitution_combine'] == 'max'
        assert not stage.GetPrimAtPath(SUPPORT_PATHS[name]).HasAPI(UsdPhysics.RigidBodyAPI)
    material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath('/World/Room/Decoration')).ComputeBoundMaterial('physics')
    assert str(material.GetPath()) == '/World/OriginalMaterial'


def test_partial_opt_in_does_not_touch_other_support(stage):
    spec = request()
    del spec['surfaces']['floor']
    materials.apply_materials(stage, SUPPORT_PATHS, spec)
    assert materials.resolved_material(stage, SUPPORT_PATHS['floor'])['material_path'] == '/World/OriginalMaterial'


def test_preflight_rejects_invalid_target_before_material_writes(stage):
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='enabled collider'):
        materials.apply_materials(stage, {**SUPPORT_PATHS, 'table': '/World/Missing'}, request())
    assert stage.GetRootLayer().ExportToString() == before


def test_reference_material_override_preserves_source_bytes_and_legacy_reference(stage, tmp_path):
    from pxr import Usd, UsdGeom
    from bench.simulation.isaacsim_backend.workroom import SCHEMA
    source = Usd.Stage.CreateNew(str(tmp_path / 'source.usda'))
    UsdGeom.SetStageMetersPerUnit(source, 1.)
    UsdGeom.SetStageUpAxis(source, 'Z')
    # Export the synthetic room subtree as /World, with a parked visual robot.
    UsdGeom.Xform.Define(source, '/World')
    from pxr import Sdf
    Sdf.CopySpec(stage.GetRootLayer(), '/World/Room', source.GetRootLayer(), '/World')
    UsdGeom.Xform.Define(source, '/World/Robot')
    source.GetRootLayer().Save()
    path = tmp_path / 'source.usda'
    original = path.read_bytes()
    manifest = tmp_path / 'inventory.json'
    manifest.write_text(json.dumps({'family': 'test', 'dependencies': {}}))
    spec = {'schema': SCHEMA, 'family': 'test', 'source_prim': '/World', 'usd_path': str(path),
            'usd_sha256': digest(path), 'dependency_manifest_path': str(manifest),
            'dependency_manifest_sha256': digest(manifest)}
    legacy = add_workroom(Usd.Stage.CreateInMemory(), spec)
    configured = add_workroom(Usd.Stage.CreateInMemory(), spec, surface_materials=request())
    assert 'surface_materials' not in legacy
    assert configured['solid_paths'] == legacy['solid_paths']
    assert path.read_bytes() == original
    assert configured['source_file_hashes'] == legacy['source_file_hashes']


def native_fixture(stage, *, kinematic=False, compound=False):
    from pxr import UsdPhysics
    record = materials.apply_materials(stage, SUPPORT_PATHS, request())
    info = {'workbench': {'surface_materials': record}}
    views = {}
    mapping = {}
    if kinematic:
        for name, path in SUPPORT_PATHS.items():
            owner = path.rsplit('/', 1)[0] if compound else path
            body = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(owner))
            body.CreateRigidBodyEnabledAttr(True)
            body.CreateKinematicEnabledAttr(True)
            mapping[path] = owner
            views[owner] = SimpleNamespace(count=1, max_shapes=2, prim_paths=[owner],
                                           get_material_properties=lambda: np.asarray([[[.4, .3, 0.], [.4, .3, 0.]]]))
        info['workbench']['collider_actor_paths'] = mapping
    sim = SimpleNamespace(create_rigid_body_view=lambda path: views[path])
    paths = {row['collider_path']: [row['material_path']] for row in record['surfaces'].values()}
    return info, sim, paths, views


def test_static_native_binding_is_not_misreported_as_native_numeric_readback(stage):
    info, sim, paths, _ = native_fixture(stage)
    result = materials.runtime_material_check(stage, sim, info, native_material_paths=paths.__getitem__)
    assert result['passed']
    assert not result['all_native_coefficients_verified']
    assert not result['native_combine_verified']
    for row in result['surfaces'].values():
        assert row['native_binding_verified'] and not row['native_coefficients_verified']
        assert row['coefficient_evidence'] == 'usd_only_static_actor'
        assert row['native_shapes'] is None


@pytest.mark.parametrize('compound', [False, True])
def test_kinematic_actor_readback_checks_every_shape_without_traversal_mapping(stage, compound):
    info, sim, paths, views = native_fixture(stage, kinematic=True, compound=compound)
    result = materials.runtime_material_check(stage, sim, info, native_material_paths=paths.__getitem__)
    assert result['all_native_coefficients_verified']
    assert not result['native_combine_verified']
    assert all(len(row['native_shapes']) == 2 for row in result['surfaces'].values())
    view = views[info['workbench']['collider_actor_paths'][SUPPORT_PATHS['table']]]
    view.get_material_properties = lambda: np.asarray([[[.4, .3, 0.], [.5, .3, 0.]]])
    with pytest.raises(RuntimeError, match='shape 1'):
        materials.runtime_material_check(stage, sim, info, native_material_paths=paths.__getitem__)


@pytest.mark.parametrize('failure', ['native_binding', 'empty_binding', 'usd_binding', 'combine',
                                    'native_identity', 'native_nan', 'native_padding', 'actor_mapping'])
def test_native_and_usd_mismatches_fail_closed(stage, failure):
    from pxr import UsdShade
    info, sim, paths, views = native_fixture(stage, kinematic=True)
    collider = SUPPORT_PATHS['table']
    if failure == 'native_binding':
        paths[collider] = ['/World/OtherMaterial']
    elif failure == 'empty_binding':
        paths[collider] = []
    elif failure == 'usd_binding':
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(collider)).Bind(
            UsdShade.Material.Get(stage, '/World/OriginalMaterial'), materialPurpose='physics')
    elif failure == 'combine':
        stage.GetPrimAtPath(materials.MATERIAL_ROOT+'/table').GetAttribute('physxMaterial:frictionCombineMode').Set('min')
    elif failure == 'native_identity':
        views[collider].prim_paths = ['/World/WrongActor']
    elif failure == 'native_nan':
        views[collider].get_material_properties = lambda: np.asarray([[[float('nan'), .3, 0.], [.4, .3, 0.]]])
    elif failure == 'native_padding':
        views[collider].max_shapes = 3
    elif failure == 'actor_mapping':
        info['workbench']['collider_actor_paths'][collider] = '/World/WrongActor'
    with pytest.raises(RuntimeError):
        materials.runtime_material_check(stage, sim, info, native_material_paths=paths.__getitem__)


def test_readback_rejects_compound_with_other_material_instead_of_guessing_shape_order(stage):
    from pxr import UsdGeom, UsdPhysics, UsdShade
    info, sim, paths, _ = native_fixture(stage, kinematic=True, compound=True)
    other = UsdGeom.Cube.Define(stage, '/World/Room/Geometry/worktop/OtherCollider').GetPrim()
    UsdPhysics.CollisionAPI.Apply(other)
    UsdShade.MaterialBindingAPI.Apply(other).Bind(UsdShade.Material.Get(stage, '/World/OriginalMaterial'),
                                                materialPurpose='physics')
    with pytest.raises(RuntimeError, match='ambiguous'):
        materials.runtime_material_check(stage, sim, info, native_material_paths=paths.__getitem__)


def test_original_mesh_material_survives_fixed_actor_adaptation(stage):
    from pxr import UsdGeom, UsdPhysics
    from bench.simulation.isaacsim_backend.static_mesh_contacts import adapt_static_meshes, geometry_signature
    path = SUPPORT_PATHS['floor']
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    record = materials.apply_materials(stage, SUPPORT_PATHS, request())
    before = geometry_signature(mesh.GetPrim())
    actors = adapt_static_meshes(stage, 'kinematic_mesh_v1')
    assert geometry_signature(mesh.GetPrim()) == before
    assert actors['collider_actor_paths'] == {path: path}
    info = {'workbench': {'surface_materials': record, **actors}}
    view = SimpleNamespace(count=1, max_shapes=1, prim_paths=[path],
                           get_material_properties=lambda: np.asarray([[[.4, .3, 0.]]]))
    sim = SimpleNamespace(create_rigid_body_view=lambda query: view if query == path else None)
    native_paths = {row['collider_path']: [row['material_path']] for row in record['surfaces'].values()}
    result = materials.runtime_material_check(stage, sim, info, native_material_paths=native_paths.__getitem__)
    assert result['surfaces']['floor']['native_coefficients_verified']
    assert not result['surfaces']['table']['native_coefficients_verified']
