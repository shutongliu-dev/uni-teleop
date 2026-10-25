"""Offline USD composition tests; these do not claim native PhysX acceptance.

Run with the CPU project interpreter and the usd-core site-packages on sys.path.
PhysxSchema setters are small real-attribute writers because usd-core does not
ship NVIDIA's schema plug-in. No SimulationApp or GPU is started.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import pytest

from bench.paths import DATA_ROOT


@pytest.fixture
def loader(monkeypatch):
    pxr = pytest.importorskip('pxr')
    from pxr import Sdf, UsdPhysics, UsdShade

    class Schema:
        prefix = ''

        def __init__(self, prim):
            self.prim = prim

        @classmethod
        def Apply(cls, prim):
            return cls(prim)

        def __getattr__(self, method):
            if not method.startswith('Create') or not method.endswith('Attr'):
                raise AttributeError(method)
            suffix = method[6:-4]
            name = self.prefix + ':' + suffix[0].lower() + suffix[1:]

            def create(value):
                kind = Sdf.ValueTypeNames.Token if isinstance(value, str) else Sdf.ValueTypeNames.Bool if isinstance(value, bool) else (
                    Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float)
                attr = self.prim.CreateAttribute(name, kind, custom=False)
                attr.Set(value)
                return attr
            return create

    stub = types.ModuleType('pxr.PhysxSchema')
    for name, prefix in [('PhysxRigidBodyAPI', 'physxRigidBody'),
                         ('PhysxContactReportAPI', 'physxContactReport'),
                         ('PhysxCollisionAPI', 'physxCollision'),
                         ('PhysxMaterialAPI', 'physxMaterial')]:
        setattr(stub, name, type(name, (Schema,), {'prefix': prefix}))
    monkeypatch.setitem(sys.modules, 'pxr.PhysxSchema', stub)
    monkeypatch.setattr(pxr, 'PhysxSchema', stub, raising=False)

    def rigid_material(stage, name, friction):
        material = UsdShade.Material.Define(stage, '/World/PhysicsMaterials/' + name)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(float(friction))
        api.CreateDynamicFrictionAttr(float(friction))
        api.CreateRestitutionAttr(0.)
        return material

    scene = types.ModuleType('bench.simulation.isaacsim_backend.scene')
    scene.rigid_material = rigid_material
    monkeypatch.setitem(sys.modules, 'bench.simulation.isaacsim_backend.scene', scene)
    monkeypatch.delitem(sys.modules, 'bench.simulation.isaacsim_backend.usd_asset', raising=False)
    return importlib.import_module('bench.simulation.isaacsim_backend.usd_asset')


def source(tmp_path, *, authored=True, root_material=False, gravity_disabled=False, anchored=False):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    path = tmp_path / 'source.usda'
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, '/Part').GetPrim()
    stage.SetDefaultPrim(root)
    if anchored:
        UsdGeom.Xformable(root).AddScaleOp().Set((1.5, 1.5, 1.5))
        root = UsdGeom.Xform.Define(stage, '/Part/Body').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.12)
    collider = UsdGeom.Mesh.Define(stage, str(root.GetPath()) + '/Thread').GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr('sdf')
    collider.CreateAttribute('physxSDFMeshCollision:sdfResolution', Sdf.ValueTypeNames.Int).Set(512)
    if authored:
        for prim, name, kind, value in [
            (root, 'physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int, 192),
            (root, 'physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int, 1),
            (root, 'physxRigidBody:enableSpeculativeCCD', Sdf.ValueTypeNames.Bool, False),
            (root, 'physxRigidBody:disableGravity', Sdf.ValueTypeNames.Bool, gravity_disabled),
            (root, 'physxContactReport:threshold', Sdf.ValueTypeNames.Float, .25),
            (collider, 'physxCollision:contactOffset', Sdf.ValueTypeNames.Float, .0002),
            (collider, 'physxCollision:restOffset', Sdf.ValueTypeNames.Float, -.00005),
        ]:
            prim.CreateAttribute(name, kind, custom=False).Set(value)
        material = UsdShade.Material.Define(stage, '/Part/Contact')
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(.6)
        api.CreateDynamicFrictionAttr(.5)
        api.CreateRestitutionAttr(.1)
        UsdShade.MaterialBindingAPI.Apply(root if root_material else collider).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    stage.GetRootLayer().Save()
    return {'usd_path': str(path), 'usd_default_prim': '/Part', 'asset_root': str(tmp_path),
            'manifest_sha256': '0' * 64, 'source_dependencies_sha256': {}, 'runtime_builtin_dependencies': [],
            'collision_parts': [{'usd_prim_path': str(collider.GetPath()), 'approximation': 'sdf'}],
            'visual_parts': [], 'material_prim_paths': [], 'mass_kg': .12, 'physics_assumptions': {},
            'bounds_min_m': [-.01, -.01, 0.], 'bounds_max_m': [.01, .01, .04]}


def place(loader, asset, *, dynamic=True, policy='preserve_authored_v1', **kwargs):
    from pxr import Usd
    stage = Usd.Stage.CreateInMemory()
    spec = {'id': 'part', 'asset_id': 'part', 'position_m': [.3, .2, .78], **kwargs}
    if policy is not None:
        spec['usd_physics_policy'] = policy
    info = loader.add_usd_asset(stage, spec, asset, dynamic=dynamic)
    return stage, info


@pytest.mark.parametrize('root_material', [False, True])
def test_preserves_authored_physics_materials_and_sdf_without_source_writes(loader, tmp_path, root_material):
    asset = source(tmp_path, root_material=root_material)
    original = hashlib.sha256(Path(asset['usd_path']).read_bytes()).hexdigest()
    stage, info = place(loader, asset, usd_physics_fallbacks={
        'contact_offset_m': .0004, 'solver_position_iterations': 64})
    report = info['usd_physics_parameters']
    values = report['body_parameters']
    assert values['physxRigidBody:solverPositionIterationCount'] == {'value': 192, 'origin': 'source_authored'}
    assert values['physxRigidBody:solverVelocityIterationCount']['value'] == 1
    assert values['physxRigidBody:enableSpeculativeCCD']['value'] is False
    assert values['physxRigidBody:disableGravity']['value'] is False
    assert values['physxContactReport:threshold']['value'] == pytest.approx(.25)
    collider_path = info['collision_paths'][0]
    offsets = report['collider_parameters'][collider_path]
    assert offsets['physxCollision:contactOffset']['value'] == pytest.approx(.0002)
    assert offsets['physxCollision:contactOffset']['origin'] == 'source_authored'
    assert offsets['physxCollision:restOffset']['value'] == pytest.approx(-.00005)
    material = report['collider_materials'][collider_path]
    assert material['origin'] == 'source_authored'
    assert material['parameters']['physics:staticFriction'] == pytest.approx(.6)
    assert material['parameters']['physics:dynamicFriction'] == pytest.approx(.5)
    assert stage.GetPrimAtPath(info['path']).GetAttribute('physics:mass').Get() == pytest.approx(.12)
    collider = stage.GetPrimAtPath(collider_path)
    assert collider.GetAttribute('physics:approximation').Get() == 'sdf'
    assert collider.GetAttribute('physxSDFMeshCollision:sdfResolution').Get() == 512
    layer_text = stage.GetRootLayer().ExportToString()
    assert 'solverPositionIterationCount' not in layer_text
    assert 'physxCollision:contactOffset' not in layer_text
    assert hashlib.sha256(Path(asset['usd_path']).read_bytes()).hexdigest() == original
    json.dumps(info)


def test_policy_uses_the_child_body_and_preserves_its_source_scale(loader, tmp_path):
    from pxr import UsdGeom
    stage, info = place(loader, source(tmp_path, anchored=True, root_material=True))
    assert info['path'] == '/World/Objects/part/Part/Body'
    assert info['usd_physics_parameters']['body_path'] == info['path']
    assert info['collision_paths'] == [info['path'] + '/Thread']
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(info['path']))
    assert [matrix[i][i] for i in range(3)] == pytest.approx([1.5, 1.5, 1.5])
    assert tuple(matrix.ExtractTranslation()) == pytest.approx([.3, .2, .78])
    assert info['usd_physics_parameters']['collider_materials'][info['path'] + '/Thread']['origin'] == 'source_authored'


def test_fallbacks_are_explicit_and_material_is_labelled(loader, tmp_path):
    stage, info = place(loader, source(tmp_path, authored=False), usd_physics_fallbacks={
        'contact_offset_m': .0002, 'rest_offset_m': 0.,
        'solver_position_iterations': 192, 'solver_velocity_iterations': 1},
        usd_physics_fallbacks_provenance={'source': 'pinned upstream scene settings'})
    report = info['usd_physics_parameters']
    assert report['body_parameters']['physxRigidBody:solverPositionIterationCount'] == {
        'value': 192, 'origin': 'task_fallback'}
    assert report['body_parameters']['physxRigidBody:disableGravity'] == {
        'value': False, 'origin': 'runtime_fallback'}
    assert report['body_parameters']['physxRigidBody:enableSpeculativeCCD'] == {
        'value': True, 'origin': 'runtime_fallback'}
    path = info['collision_paths'][0]
    assert report['collider_parameters'][path]['physxCollision:contactOffset'] == {
        'value': .0002, 'origin': 'task_fallback'}
    assert report['collider_materials'][path]['origin'] == 'runtime_fallback'
    assert report['collider_materials'][path]['parameters']['physics:staticFriction'] == pytest.approx(.65)
    assert report['fallback_provenance'] == {'source': 'pinned upstream scene settings'}
    assert stage.GetPrimAtPath(info['path']).GetAttribute('physxRigidBody:disableGravity').Get() is False


def test_fixture_is_kinematic_without_rewriting_source_solver_settings(loader, tmp_path):
    from pxr import UsdPhysics
    stage, info = place(loader, source(tmp_path), dynamic=False)
    body = stage.GetPrimAtPath(info['path'])
    assert UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() is True
    assert body.GetAttribute('physxRigidBody:solverPositionIterationCount').Get() == 192
    assert info['contact_filter_paths'] == [info['path']]
    assert info['usd_physics_parameters']['body_parameters'] == {}


def test_explicit_disabled_gravity_is_rejected(loader, tmp_path):
    with pytest.raises(ValueError, match='must enable gravity'):
        place(loader, source(tmp_path, gravity_disabled=True))


def test_legacy_path_keeps_its_overrides_and_metadata(loader, tmp_path):
    stage, info = place(loader, source(tmp_path), policy=None)
    assert 'usd_physics_parameters' not in info
    body = stage.GetPrimAtPath(info['path'])
    assert body.GetAttribute('physxRigidBody:solverPositionIterationCount').Get() == 16
    assert body.GetAttribute('physxRigidBody:solverVelocityIterationCount').Get() == 4
    assert body.GetAttribute('physxRigidBody:enableSpeculativeCCD').Get() is True
    assert body.GetAttribute('physxContactReport:threshold').Get() == 0.
    collider = stage.GetPrimAtPath(info['collision_paths'][0])
    assert collider.GetAttribute('physxCollision:contactOffset').Get() == pytest.approx(.001)
    assert collider.GetAttribute('physxCollision:restOffset').Get() == 0.


@pytest.mark.parametrize('authored', [False, True])
def test_explicit_contact_profile_fills_only_missing_parameters(loader, tmp_path, authored):
    asset = source(tmp_path, authored=authored)
    original = hashlib.sha256(Path(asset['usd_path']).read_bytes()).hexdigest()
    stage, info = place(loader, asset,
        usd_physics_fallbacks={'max_depenetration_velocity_m_s': .05, 'linear_damping': 0.,
            'angular_damping': 0., 'sleep_threshold': 0., 'enable_speculative_ccd': False,
            'solver_velocity_iterations': 16},
        usd_physics_material_fallback={'static_friction': .01, 'dynamic_friction': .02,
                                       'restitution': 0., 'friction_combine': 'average'})
    report = info['usd_physics_parameters']
    assert report['body_parameters']['physxRigidBody:maxDepenetrationVelocity'] == {'value': .05, 'origin': 'task_fallback'}
    assert report['body_parameters']['physxRigidBody:linearDamping']['value'] == 0.
    assert report['body_parameters']['physxRigidBody:angularDamping']['value'] == 0.
    assert report['body_parameters']['physxRigidBody:sleepThreshold']['value'] == 0.
    assert report['body_parameters']['physxRigidBody:enableSpeculativeCCD']['value'] is False
    assert report['body_parameters']['physxRigidBody:solverVelocityIterationCount']['value'] == (1 if authored else 16)
    material = report['collider_materials'][info['collision_paths'][0]]
    assert material['origin'] == ('source_authored' if authored else 'task_fallback')
    assert material['parameters']['physics:staticFriction'] == pytest.approx(.6 if authored else .01)
    assert material['parameters']['physics:dynamicFriction'] == pytest.approx(.5 if authored else .02)
    assert material['friction_combine'] == {'value': 'average', 'origin': 'task_fallback'}
    assert hashlib.sha256(Path(asset['usd_path']).read_bytes()).hexdigest() == original


def test_original_v3_profile_composes_with_exact_role_materials(loader, tmp_path):
    from scenes.workrooms.mechanical_assembly.threaded_sharpa_catalog import (
        ASSET_IDS, SHARPA_ROOT, prepare_threaded_sharpa_catalog)
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    if not SHARPA_ROOT.is_dir():
        pytest.skip('Pinned shared v3 assets unavailable')
    root = tmp_path / 'catalog'
    prepared = prepare_threaded_sharpa_catalog(root)
    for role, friction in [('screwdriver', .6), ('bolt', .01), ('fixture', .75)]:
        asset = load_task_asset(ASSET_IDS[role], root, verify_hashes=True)
        options = dict(prepared['per_role_spec_options'][role])
        options.pop('usd_physics_policy')
        stage, info = place(loader, asset, dynamic=role != 'fixture', **options)
        report = info['usd_physics_parameters']
        assert info['collision_parts'] == {'screwdriver': 3, 'bolt': 8, 'fixture': 4}[role]
        for material in report['collider_materials'].values():
            assert material['parameters']['physics:staticFriction'] == pytest.approx(friction)
            assert material['origin'] == ('source_authored' if role == 'screwdriver' else 'task_fallback')
            assert material['friction_combine'] == {'value': 'average', 'origin': 'task_fallback'}
        if role != 'fixture':
            assert report['body_parameters']['physxRigidBody:solverVelocityIterationCount']['value'] == 16
            assert report['body_parameters']['physxRigidBody:enableSpeculativeCCD']['value'] is False


@pytest.mark.parametrize('preserve', [True, False])
def test_real_v3_disabled_head_is_audited_without_changing_collision_or_source(loader, preserve):
    from pxr import Usd, UsdPhysics, UsdShade
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    root = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_workshop/threaded_v2')
    if not (root / 'manifest.json').is_file():
        pytest.skip('Pinned shared v3 catalog unavailable')
    asset = load_task_asset('mechanical_bolt_sharpa_v3', root, verify_hashes=True)
    original = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in asset['source_dependencies_sha256']}
    options = dict(asset['recommended_spec_options']) if preserve else {}
    options.pop('usd_physics_policy', None)
    stage, info = place(loader, asset, policy='preserve_authored_v1' if preserve else None, **options)
    disabled_path = info['path'] + '/collisions_head'
    disabled = stage.GetPrimAtPath(disabled_path)
    assert disabled and UsdPhysics.CollisionAPI(disabled).GetCollisionEnabledAttr().Get() is False
    assert info['collision_parts'] == len(info['collision_paths']) == 8
    assert disabled_path not in info['collision_paths']
    assert len([p for p in Usd.PrimRange(stage.GetPrimAtPath(info['path']))
                if p.HasAPI(UsdPhysics.CollisionAPI)]) == 9
    source_stage = Usd.Stage.Open(asset['usd_path'])
    source_head = source_stage.GetPrimAtPath(asset['usd_default_prim'] + '/allen_bolt/collisions_head')
    for name in ('points', 'faceVertexCounts', 'faceVertexIndices', 'physics:approximation',
                 'physxCollision:contactOffset', 'physxCollision:restOffset'):
        assert disabled.GetAttribute(name).Get() == source_head.GetAttribute(name).Get()
        assert not stage.GetRootLayer().GetAttributeAtPath(disabled_path + '.' + name)
    assert not stage.GetRootLayer().GetAttributeAtPath(disabled_path + '.physics:collisionEnabled')
    if preserve:
        report = info['usd_physics_parameters']
        assert report['active_collision_paths'] == info['collision_paths']
        assert report['disabled_collision_paths'] == [disabled_path]
        assert set(report['collider_materials']) == set(info['collision_paths']) | {disabled_path}
        audit = report['disabled_colliders'][disabled_path]
        assert audit['collision_enabled'] is False and audit['collision_enabled_source_authored']
        assert audit['approximation'] == 'sdf'
        assert audit['preserved_offsets']['physxCollision:contactOffset']['value'] == pytest.approx(.0003)
        resolved, _ = UsdShade.MaterialBindingAPI(disabled).ComputeBoundMaterial('physics')
        assert UsdPhysics.MaterialAPI(resolved.GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(.01)
        assert report['collider_materials'][disabled_path]['origin'] == 'task_fallback'
        assert len(report['collider_parameters']) == 8, 'No offset authoring for disabled source geometry'
    else:
        assert 'usd_physics_parameters' not in info
        assert not stage.GetRootLayer().GetPrimAtPath(disabled_path), 'Legacy does not directly edit a disabled collider'
    assert {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in original} == original


@pytest.mark.parametrize('kwargs', [
    {'policy': 'unknown'}, {'friction': .9},
    {'policy': None, 'usd_physics_fallbacks': {}},
    {'usd_physics_fallbacks': {'unknown': 1}},
    {'usd_physics_fallbacks': []},
    {'usd_physics_fallbacks': {'contact_offset_m': 0.}},
    {'usd_physics_fallbacks': {'rest_offset_m': float('nan')}},
    {'usd_physics_fallbacks': {'solver_position_iterations': 0}},
    {'usd_physics_fallbacks': {'solver_velocity_iterations': 1.5}},
    {'usd_physics_fallbacks': {'solver_position_iterations': True}},
    {'usd_physics_fallbacks': {'contact_offset_m': .0002, 'rest_offset_m': .001}},
    {'usd_physics_fallbacks': {'enable_speculative_ccd': 1}},
    {'usd_physics_fallbacks': {'max_depenetration_velocity_m_s': -.01}},
    {'policy': None, 'usd_physics_material_fallback': {}},
    {'usd_physics_material_fallback': {'static_friction': .1}},
    {'usd_physics_material_fallback': {'static_friction': -.1, 'dynamic_friction': .1,
                                      'restitution': 0., 'friction_combine': 'average'}},
])
def test_invalid_policy_and_fallbacks_fail_closed(loader, tmp_path, kwargs):
    with pytest.raises(ValueError):
        place(loader, source(tmp_path, authored=False), **kwargs)
