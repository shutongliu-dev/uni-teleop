"""CPU boundary fixtures for diagnostics; these are not native qualification."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from scenes.workrooms.initial_randomization import TASK_FAMILIES, configurations
from scripts.validation import workroom_randomized_workspace as workspace


def request(tmp_path, cases=None):
    cases = cases or [{'task_id': task, 'configuration_id': config, 'seed': seed, 'release': '/release'}
        for task in TASK_FAMILIES for config in configurations(task) for seed in range(10)]
    path = tmp_path/'request.json'
    path.write_text(json.dumps({'schema': 'workroom_randomized_validation_request.v1', 'cases': cases}))
    return path


def test_all_declared_cases_and_family_normalization(tmp_path):
    result = workspace.read_request(request(tmp_path))
    assert len(result['cases']) == sum(10*len(configurations(task)) for task in TASK_FAMILIES)
    assert result['duration_s'] == .8
    assert all(c['family'] == TASK_FAMILIES[c['task_id']] for c in result['cases'])
    from scripts.validation.living_randomized_workspace import read_request
    with pytest.raises(ValueError, match='schema'):
        read_request(request(tmp_path))


@pytest.mark.parametrize('update', [{'family': 'study'}, {'configuration_id': 'dual_opposed'},
    {'task_id': 'tower_building'}, {'task_id': 'polyomino_territory_practice'},
    {'seed': True}, {'seed': 10}, {'release': 'relative'}, {'extra': 1}])
def test_invalid_new_case_rejected(tmp_path, update):
    case = {'task_id': 'produce_transfer', 'configuration_id': 'single_arm', 'seed': 0, 'release': '/release'}
    case.update(update)
    with pytest.raises(ValueError):
        workspace.read_request(request(tmp_path, [case]))


def test_duplicate_fields_and_cases_rejected(tmp_path):
    p = request(tmp_path)
    p.write_text('{"schema":"x","schema":"x"}')
    with pytest.raises(ValueError):
        workspace.read_request(p)
    c = {'task_id': 'parts_sorting', 'configuration_id': 'single_arm', 'seed': 0, 'release': '/release'}
    with pytest.raises(ValueError, match='Duplicate case'):
        workspace.read_request(request(tmp_path, [c, c]))


def scene_fixture(family='study', representation='task_layers'):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    body_path = '/World/Objects/thing' if representation == 'objects' else '/World/Task/Thing'
    body = UsdGeom.Xform.Define(stage, body_path)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim()); mass.CreateMassAttr(.125)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0)); mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1, Gf.Vec3f(0)))
    collider = UsdGeom.Cube.Define(stage, body_path+'/Collider'); collider.CreateSizeAttr(.1)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    # A large visual leaf is deliberately excluded from collision readback.
    UsdGeom.Cube.Define(stage, body_path+'/Picture').CreateSizeAttr(10.)
    root = {'root_prim_path': '/World/Task', 'rigid_bodies': {'thing': body_path}}
    scene = ({'objects': [{'id': 'thing'}]} if representation == 'objects' else
             {'native_task_bundle': root} if representation == 'native_task_bundle' else
             {'task_layers': [{'source_prim_path': '/World/Task', 'rigid_bodies': root['rigid_bodies']}]})
    task = {'id': 'fixture', 'reset_case': {'family': family,
        'actual_body_poses_m': {'thing': {'prim_path': body_path, 'position_m': [0,0,0], 'quaternion_wxyz': [1,0,0,0]}},
        'static_geometry': {'bodies': [{'object_id': 'thing', 'local_bounds_m': [[-.05]*3,[.05]*3], 'nominal_mass_kg': .125}]},
        'fixed_support_paths': ['/World/Task/Tray']}}
    return stage, scene, task


@pytest.mark.parametrize('representation', ['objects', 'native_task_bundle', 'task_layers'])
def test_actual_dynamic_inventory_and_collider_readback(representation):
    stage, scene, task = scene_fixture(representation=representation)
    result = workspace.verify_case_geometry(stage, scene, task)
    assert result['passed'] and len(result['bodies']['thing']['enabled_colliders']) == 1
    assert result['bodies']['thing']['maximum_bounds_error_m'] < 1e-8


@pytest.mark.parametrize('kind', ['extra_actor', 'mass', 'bounds', 'missing_actor', 'duplicate_geometry',
                                  'duplicate_declaration', 'kinematic', 'disabled_collider', 'root_scale'])
def test_actual_geometry_mismatch_fails_closed(kind):
    from pxr import Gf, UsdGeom, UsdPhysics
    stage, scene, task = scene_fixture()
    body = stage.GetPrimAtPath('/World/Task/Thing')
    if kind == 'extra_actor':
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, '/World/Task/Extra').GetPrim())
    elif kind == 'mass':
        UsdPhysics.MassAPI(body).GetMassAttr().Set(.126)
    elif kind == 'bounds':
        UsdGeom.Cube(stage.GetPrimAtPath('/World/Task/Thing/Collider')).GetSizeAttr().Set(.102)
    elif kind == 'missing_actor':
        stage.RemovePrim('/World/Task/Thing')
    elif kind == 'duplicate_geometry':
        task['reset_case']['static_geometry']['bodies'] *= 2
    elif kind == 'duplicate_declaration':
        scene['task_layers'] *= 2
    elif kind == 'kinematic':
        UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(True)
    elif kind == 'disabled_collider':
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/Task/Thing/Collider')).GetCollisionEnabledAttr().Set(False)
    else:
        UsdGeom.Xformable(body).AddScaleOp().Set(Gf.Vec3f(1.01))
    with pytest.raises(ValueError):
        workspace.verify_case_geometry(stage, scene, task)


def test_rotated_analytic_cylinder_uses_surface_not_its_rotated_box():
    from pxr import UsdGeom, UsdPhysics
    stage, scene, task = scene_fixture()
    stage.RemovePrim('/World/Task/Thing/Collider')
    cyl = UsdGeom.Cylinder.Define(stage, '/World/Task/Thing/Collider')
    cyl.CreateRadiusAttr(.05); cyl.CreateHeightAttr(.1); cyl.CreateAxisAttr('Z')
    cyl.AddRotateZOp().Set(45.)
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    assert workspace.verify_case_geometry(stage, scene, task)['passed']


def test_real_canonical_mechanical_catalog_mount_preserves_source_bounds_and_mass():
    from bench.simulation.isaacsim_backend.asset_catalog import load_task_asset
    from scenes.workrooms.mechanical_assembly.tasks.parts_sorting.definition import TASK
    path = Path(TASK['source_release']['path'])/'scene.json'
    if not path.is_file():
        pytest.skip('Shared original mechanical assets are not installed')
    scene = load_json(path); poses = {}; geometry = []
    for obj in scene['objects']:
        asset = load_task_asset(obj['asset_id'], asset_root=obj['asset_root'], verify_hashes=True)
        bounds = asset['aabb_m'] if 'aabb_m' in asset else [asset['bounds_min_m'], asset['bounds_max_m']]
        poses[obj['id']] = {'prim_path': '/World/Objects/'+obj['id'],
            'position_m': obj['position_m'], 'quaternion_wxyz': obj['quaternion_wxyz']}
        geometry.append({'object_id': obj['id'], 'local_bounds_m': bounds, 'nominal_mass_kg': asset['mass_kg']})
    task = {'reset_case': {'actual_body_poses_m': poses, 'static_geometry': {'bodies': geometry}}}
    stage, sources = workspace.composed_support_stage(scene)
    result = workspace.verify_case_geometry(stage, scene, task)
    assert set(result['bodies']) == {'block','cylinder','ring'}
    assert any('ring.usda' in p for p in sources)
    workspace.living.verify_authored_poses(stage, workspace.body_paths(scene, task), poses)


def test_replacing_task_geometry_retains_original_robot_meshes_for_self_collision():
    from scenes.workrooms.study.tasks.book_stacking.definition import TASK
    root = Path(TASK['source_release']['path'])
    if not (root/'scene.json').is_file():
        pytest.skip('Shared original study assets are not installed')
    scene = load_json(root/'scene.json')
    initial = load_json(root/'initial_state.json')
    models, stands, _ = workspace.living.installation(scene, initial, 'study', 'book_stacking', 'single_arm')
    checker = workspace.CollisionBounds(scene, scene['robot'], models, initial['joint_positions_rad'], stands)
    arm = workspace.living.ArmSelfCollision(checker)
    assert len(arm.shapes['right']) == 8
    assert not any(p.startswith('/Robots/') for p in checker.obstacle_names)


