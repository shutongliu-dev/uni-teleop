"""Workroom composition must never import a second robot/physics world."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench.paths import load_json
from scenes import workrooms
from scenes.workrooms import registry
from bench.simulation.isaacsim_backend import workroom
from bench.simulation.isaacsim_backend.cameras import resolve_cameras


@pytest.fixture
def published_assets():
    if not registry.CURRENT_RELEASE.is_file() or not (registry.BASE_RELEASE / 'release_manifest.json').is_file():
        pytest.skip('Published workroom pointer and v12 release are not installed')


@pytest.mark.parametrize('key', ['version', 'scene_manifest', 'scene_manifest_sha256'])
def test_release_pointer_cannot_silently_diverge(tmp_path, monkeypatch, key):
    pointer = tmp_path / 'current_release.json'
    value = {'schema': 'scence.release_pointer.v1', 'version': registry.VERSION,
             'scene_manifest': str(registry.ASSET_ROOT / 'manifest.json'),
             'scene_manifest_sha256': registry.COLLECTION_SHA256}
    value[key] = 'different'
    pointer.write_text(json.dumps(value))
    monkeypatch.setattr(registry, 'CURRENT_RELEASE', pointer)
    with pytest.raises(ValueError, match='release pointer'):
        registry.reference('laboratory')


def test_scene_derivation_preserves_frozen_robot_and_physics(published_assets):
    from bench.isaacsim.release import task_inputs
    _, original, _, _ = task_inputs(registry.BASE_RELEASE, '01_cube_push')
    value = workrooms.scene('laboratory')
    robot = deepcopy(value['robot'])
    robot['base_position_m'] = original['robot']['base_position_m']
    assert robot['arm_home_rad'] == [0., -.55, 0., -2.15, 0., 1.65, .75]
    robot['arm_home_rad'] = original['robot']['arm_home_rad']
    assert robot == original['robot']
    assert value['isaacsim'] == {**original['isaacsim'], **registry.PHYSICS_OVERRIDES,
                                'workroom_lighting': {'schema': 'workroom-lighting-v1',
                                                     'family': 'laboratory', 'profile': 'nominal'}}
    assert value['isaacsim']['robot_material_binding'] == 'collider_instance_v1'
    assert 'robot_pad_compliance' not in value['isaacsim']
    assert value['isaacsim']['robot_contact_materials'] == original['isaacsim']['robot_contact_materials']
    assert value['workroom_provenance']['physics_overrides'] == registry.PHYSICS_OVERRIDES
    assert value['workroom_provenance']['runtime_profile'] == registry.RUNTIME_PROFILE
    assert value['objects'][0]['asset_id'] == original['objects'][0]['asset_id']
    assert value['workroom_provenance']['physics_accepted'] is False
    assert set(resolve_cameras(value)) == {'overview', 'task', 'room', 'wrist'}
    assert value['objects'][0]['position_m'][2] > .78 + .03
    assert value['objects'][1]['position_m'][2] > .03
    assert workroom.digest(registry.BASE_RELEASE / 'release_manifest.json') == registry.BASE_MANIFEST_SHA256
    value['robot']['arm_home_rad'][0] = 1.5
    assert workrooms.scene('laboratory', diagnostic=False)['robot']['arm_home_rad'][0] == 0.
    assert workrooms.scene('laboratory', diagnostic=False)['objects'] == []


def test_export_is_env_ready_and_cannot_overwrite(published_assets, tmp_path):
    target = tmp_path / 'scene'
    record = workrooms.export_scene('study', target, diagnostic=False)
    value = load_json(target / 'scene.json')
    assert value['workroom']['family'] == 'study'
    assert value['objects'] == []
    assert record['status'] == 'diagnostic_unvalidated'
    assert record['task_qualification'] is False
    assert record['scene_sha256'] == workroom.digest(target / 'scene.json')
    with pytest.raises(FileExistsError):
        workrooms.export_scene('study', target)


@pytest.fixture
def synthetic_room(tmp_path):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom, UsdPhysics
    source = tmp_path / 'source.usda'
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    robot = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(robot)
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    stage.DefinePrim('/Render', 'Scope')
    for name in ('worktop', 'room_floor'):
        prim = UsdGeom.Cube.Define(stage, f'/World/Geometry/{name}/Collider').GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
    stage.GetRootLayer().Save()
    inventory = tmp_path / 'scene_description.json'
    inventory.write_text(json.dumps({'family': 'synthetic', 'dependencies': {}}))
    spec = {'schema': workroom.SCHEMA, 'family': 'synthetic', 'source_prim': '/World',
            'usd_path': str(source), 'usd_sha256': workroom.digest(source),
            'dependency_manifest_path': str(inventory), 'dependency_manifest_sha256': workroom.digest(inventory)}
    return spec


def test_reference_excludes_native_settings_and_robot_without_editing_asset(synthetic_room):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    own = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(own)
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    info = workroom.add_workroom(stage, synthetic_room)
    assert stage.GetPrimAtPath('/World/Robot').IsActive()
    assert not stage.GetPrimAtPath('/World/Room/Robot').IsActive()
    assert not stage.GetPrimAtPath('/Render')
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]) == 1
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]) == 1
    assert set(info['solid_paths']) == set(workroom.SUPPORT_PATHS.values())
    assert workroom.digest(synthetic_room['usd_path']) == synthetic_room['usd_sha256']
    with pytest.raises(ValueError, match='already exists'):
        workroom.add_workroom(stage, synthetic_room)


@pytest.mark.parametrize('unexpected', ['body', 'joint', 'scene'])
def test_environment_dynamics_are_rejected(synthetic_room, unexpected):
    from pxr import Usd, UsdGeom, UsdPhysics
    source = Usd.Stage.Open(synthetic_room['usd_path'])
    if unexpected == 'body':
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(source, '/World/Unexpected').GetPrim())
    elif unexpected == 'joint':
        UsdPhysics.FixedJoint.Define(source, '/World/Unexpected')
    else:
        UsdPhysics.Scene.Define(source, '/World/Unexpected')
    source.GetRootLayer().Save()
    synthetic_room['usd_sha256'] = workroom.digest(synthetic_room['usd_path'])
    with pytest.raises(ValueError, match='active dynamics'):
        workroom.add_workroom(Usd.Stage.CreateInMemory(), synthetic_room)


def test_modified_source_is_rejected_before_composition(synthetic_room):
    from pxr import Usd
    Path(synthetic_room['usd_path']).write_text('#usda 1.0\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        workroom.add_workroom(Usd.Stage.CreateInMemory(), synthetic_room)


def test_explicitly_disabled_decoration_schemas_are_static(synthetic_room):
    from pxr import Usd, UsdGeom, UsdPhysics
    source = Usd.Stage.Open(synthetic_room['usd_path'])
    body = UsdGeom.Xform.Define(source, '/World/StaticDecoration').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(False)
    UsdPhysics.FixedJoint.Define(source, '/World/StaticJoint').CreateJointEnabledAttr(False)
    source.GetRootLayer().Save()
    synthetic_room['usd_sha256'] = workroom.digest(synthetic_room['usd_path'])
    result = workroom.add_workroom(Usd.Stage.CreateInMemory(), synthetic_room)
    assert result['kind'] == 'referenced_workroom'


def test_static_mesh_collision_is_preserved_but_not_promised_as_gpu_contact_filter(synthetic_room):
    from pxr import Usd, UsdGeom, UsdPhysics
    source = Usd.Stage.Open(synthetic_room['usd_path'])
    prim = UsdGeom.Mesh.Define(source, '/World/StaticMesh').GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    source.GetRootLayer().Save()
    synthetic_room['usd_sha256'] = workroom.digest(synthetic_room['usd_path'])
    stage = Usd.Stage.CreateInMemory()
    result = workroom.add_workroom(stage, synthetic_room)
    path = '/World/Room/StaticMesh'
    assert stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
    assert path in result['solid_paths'] and path in result['contact_unobserved_paths']
    assert path not in result['contact_filter_paths']


@pytest.mark.parametrize('family', workrooms.FAMILIES)
def test_published_room_has_static_supports_and_clear_diagnostic_fall_paths(published_assets, family):
    pytest.importorskip('pxr.Usd')
    report = workrooms.validate(family)
    assert report['status'] == 'cpu_preflight_passed'
    assert report['physics_accepted'] is False
    assert set(report['supports'].values()) <= set(report['solid_paths'])
    assert set(report['supports'].values()) <= set(report['contact_filter_paths'])
    assert report['diagnostic_clearance_checked'] == ['cube_floor', 'cube_table']
    assert report['dependency_file_hashes']
