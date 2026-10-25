"""CPU USD contract tests; these do not grant native dynamics qualification."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('pxr.Usd')
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

from bench.simulation.isaacsim_backend.task_layers import mount_task_layers
from bench.simulation.isaacsim_backend.task_readback import (
    TaskLayerReadback, prismatic_state, validate_particle_arrays)
from bench.simulation.isaacsim_backend.workroom import BUILTIN_MDL


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bench_stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdPhysics.Scene.Define(stage, '/World/BenchPhysics')
    UsdGeom.Xform.Define(stage, '/World/Robot')
    return stage


def mount(stage, spec):
    return mount_task_layers(stage, [spec], physics_scene_path='/World/BenchPhysics')


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'task.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.Xform.Define(stage, '/World/Task')
    # These must never be pulled in by a subtree reference.
    UsdPhysics.Scene.Define(stage, '/World/PreviewPhysics')
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, '/World/PreviewRobot').GetPrim())
    for name, position in [('base', (0, 0, 1)), ('slider', (0, 0, 1.1))]:
        body = UsdGeom.Xform.Define(stage, '/World/Task/' + name)
        body.AddTranslateOp().Set(position)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.125)
        collision = UsdGeom.Cube.Define(stage, str(body.GetPath()) + '/Collision')
        collision.CreateSizeAttr(.04)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/Task/SliderJoint')
    joint.CreateBody0Rel().SetTargets(['/World/Task/base'])
    joint.CreateBody1Rel().SetTargets(['/World/Task/slider'])
    joint.CreateAxisAttr('Z')
    joint.CreateLowerLimitAttr(0.)
    joint.CreateUpperLimitAttr(.2)
    fixture = UsdGeom.Cube.Define(stage, '/World/Task/Support')
    UsdPhysics.CollisionAPI.Apply(fixture.GetPrim())
    stage.GetRootLayer().Save()
    return {'id': 'props', 'usd_path': str(path), 'sha256': digest(path),
        'source_prim_path': '/World/Task', 'target_prim_path': '/World/Imported',
        'rigid_bodies': {'base': '/World/Imported/base', 'slider': '/World/Imported/slider'},
        'dependencies_sha256': {}, 'particle_sets': {}}


def test_mount_only_subtree_and_preserve_authored_physics(source):
    stage = bench_stage()
    info = mount(stage, source)
    assert not stage.GetPrimAtPath('/World/PreviewRobot')
    assert not stage.GetPrimAtPath('/World/PreviewPhysics')
    assert stage.GetPrimAtPath('/World/Robot')
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]) == 1
    assert info['static_collision_paths'] == ['/World/Imported/Support']
    assert info['objects']['slider']['initial_state']['position_m'] == [0, 0, 1.1]
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath('/World/Imported/SliderJoint'))
    assert joint.GetBody0Rel().GetTargets() == [Sdf.Path('/World/Imported/base')]
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(.2)
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath('/World/Imported/base')).GetMassAttr().Get() == .125
    assert digest(source['usd_path']) == source['sha256']


@pytest.mark.parametrize('fault', ['hash', 'missing_body', 'alias', 'occupied', 'replacement'])
def test_reject_ambiguous_or_changed_layer(source, fault):
    stage = bench_stage()
    if fault == 'hash':
        source['sha256'] = '0' * 64
    elif fault == 'missing_body':
        del source['rigid_bodies']['slider']
    elif fault == 'alias':
        source['rigid_bodies']['alias'] = source['rigid_bodies']['base']
    elif fault == 'replacement':
        UsdGeom.Xform.Define(stage, '/World/Imported')
    with pytest.raises(ValueError):
        mount_task_layers(stage, [source], physics_scene_path='/World/BenchPhysics',
                          occupied_objects=['base'] if fault == 'occupied' else [])


def test_static_layer_and_unpinned_dependency(source, tmp_path):
    stage = Usd.Stage.Open(source['usd_path'])
    for name in ('base', 'slider', 'SliderJoint'):
        stage.RemovePrim('/World/Task/' + name)
    dependency = tmp_path / 'texture.bin'
    dependency.write_bytes(b'pinned resource')
    stage.GetPrimAtPath('/World/Task').CreateAttribute('texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(dependency)))
    stage.GetRootLayer().Save()
    source['sha256'], source['rigid_bodies'] = digest(source['usd_path']), {}
    with pytest.raises(ValueError, match='Unpinned'):
        mount(bench_stage(), source)
    source['dependencies_sha256'] = {str(dependency): digest(dependency)}
    info = mount(bench_stage(), source)
    assert info['objects'] == {}
    assert len(info['static_collision_paths']) == 1
    dependency.write_bytes(b'changed resource')
    with pytest.raises(ValueError, match='hash mismatch'):
        mount(bench_stage(), source)


def test_external_joint_body_cannot_silently_become_world_anchor(source):
    stage = Usd.Stage.Open(source['usd_path'])
    UsdPhysics.PrismaticJoint(stage.GetPrimAtPath('/World/Task/SliderJoint')).GetBody0Rel().SetTargets(
        ['/World/PreviewRobot'])
    stage.GetRootLayer().Save()
    source['sha256'] = digest(source['usd_path'])
    target = bench_stage()
    with pytest.raises(ValueError, match='external relationship target'):
        mount(target, source)
    assert not target.GetPrimAtPath('/World/Imported')


def test_prismatic_velocity_has_com_and_rotating_axis_terms():
    joint = {'body0': 'base', 'body1': 'slider', 'axis': 'X',
        'local_pos0': [.2, -.1, .05], 'local_pos1': [-.1, .3, .15],
        'local_rot0_xyzw': [0, 0, np.sin(.2), np.cos(.2)], 'local_rot1_xyzw': [0, 0, 0, 1]}
    bodies = {
        'base': {'pose_xyzw': np.array([1., 2., 3., 0, 0, 0, 1]),
                 'velocity': np.array([.3, -.2, .1, 0, 0, .7]), 'com_local_m': [.05, -.03, .02]},
        'slider': {'pose_xyzw': np.array([1.2, 2.1, 3.1, 0, 0, 0, 1]),
                   'velocity': np.array([-.1, .3, -.2, 0, 0, -.4]), 'com_local_m': [-.02, .01, .05]}}
    position, velocity = prismatic_state(joint, bodies)
    moved = deepcopy(bodies)
    dt = 1.e-7
    for value in moved.values():
        omega = value['velocity'][5]
        angle = omega * dt
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        com = np.asarray(value['com_local_m'])
        value['pose_xyzw'][:3] += value['velocity'][:3] * dt + com - rotation @ com
        value['pose_xyzw'][3:] = [0, 0, np.sin(angle/2), np.cos(angle/2)]
    assert velocity == pytest.approx((prismatic_state(joint, moved)[0]-position)/dt, abs=1.e-7)


def test_particle_inventory_rejects_reindexing_loss_and_nonfinite():
    points, velocities, ids = np.zeros((2, 3)), np.ones((2, 3)), np.array([101, 303])
    assert np.array_equal(validate_particle_arrays(points, velocities, ids, 2)[2], ids)
    for bad_ids in ([101, 101], [303, 101]):
        with pytest.raises(RuntimeError, match='IDs'):
            validate_particle_arrays(points, velocities, bad_ids, 2, ids)
    with pytest.raises(RuntimeError, match='inventory'):
        validate_particle_arrays(points[:1], velocities, ids, 2, ids)
    points[0, 0] = np.nan
    with pytest.raises(RuntimeError, match='Nonfinite'):
        validate_particle_arrays(points, velocities, ids, 2, ids)


@pytest.mark.parametrize('task,count,joints,particles', [
    ('remote_operation', 4, 2, 0), ('water_pouring', 2, 0, 488),
    ('tower_building', 36, 0, 0), ('tower_extraction', 36, 0, 0)])
def test_frozen_scene_subtrees_cpu_inventory_and_reset(task, count, joints, particles):
    from scenes.workrooms.living_room.scene_candidates import scene_candidate
    candidate = scene_candidate(task)
    if not Path(candidate['initial_usd']).is_file():
        pytest.skip('Pinned shared task source not available')
    original = Usd.Stage.Open(candidate['initial_usd'])
    root = original.GetPrimAtPath(candidate['reference_prim_path'])
    bodies = {p.GetName().lower(): str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    deps = {}
    for prim in Usd.PrimRange(root):
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath) and value.path and value.path not in BUILTIN_MDL:
                path = value.resolvedPath or Sdf.ComputeAssetPathRelativeToLayer(original.GetRootLayer(), value.path)
                deps[path] = digest(path)
    particle_sets = {'water': {'path': '/World/LivingTask/WaterParticles',
        'system_path': '/World/LivingTask/WaterSystem', 'count': particles,
        'total_mass_kg': .25, 'position_attribute': 'points'}} if particles else {}
    source = {'id': 'task', 'usd_path': candidate['initial_usd'], 'sha256': candidate['initial_usd_sha256'],
        'source_prim_path': str(root.GetPath()), 'target_prim_path': str(root.GetPath()),
        'rigid_bodies': bodies, 'dependencies_sha256': deps, 'particle_sets': particle_sets}
    stage = bench_stage()
    info = mount(stage, source)
    readback = TaskLayerReadback(stage, info['layers'])
    assert len(info['objects']) == count
    assert len(readback.joints) == joints
    assert len([p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]) == 1
    # Every authored physics attribute survives the subtree composition exactly.
    for prim in Usd.PrimRange(root):
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(('physics:', 'physx')) and attr.HasAuthoredValueOpinion():
                assert stage.GetPrimAtPath(prim.GetPath()).GetAttribute(attr.GetName()).Get() == attr.Get()
    first = next(iter(bodies.values()))
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(first))
    original_transform = xform.GetLocalTransformation()
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(opSuffix='test_reset').Set((40, 50, 60))
    if particles:
        points = stage.GetPrimAtPath(particle_sets['water']['path'])
        initial = readback.initial_particles['water']
        points.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(np.zeros((particles, 3), dtype=np.float32)))
        assert stage.GetPrimAtPath(particle_sets['water']['system_path']).GetRelationship('simulationOwner').GetTargets() == [Sdf.Path('/World/BenchPhysics')]
    readback.restore_before_world_reset()
    assert xform.GetLocalTransformation() == original_transform
    if particles:
        restored = readback._usd_particle_arrays(particle_sets['water'])
        assert all(np.array_equal(a, b) for a, b in zip(initial, restored))
        assert len(restored[2]) == particles
    assert digest(candidate['initial_usd']) == candidate['initial_usd_sha256']


def test_payload_uses_supplied_same_tick_state_and_numpy_arrays(source):
    stage = bench_stage()
    info = mount(stage, source)
    readback = TaskLayerReadback(stage, info['layers'])
    readback.coms = {'base': np.zeros(3), 'slider': np.zeros(3)}
    object_q = np.array([[0, 0, 1, 0, 0, 0, 1], [0, 0, 1.125, 0, 0, 0, 1]])
    object_qd = np.zeros((2, 6))
    object_qd[1, 2] = .4
    arrays = readback.capture_arrays(SimpleNamespace(current_time_step_index=17, current_time=.0085),
                                     ['base', 'slider'], object_q, object_qd)
    assert all(isinstance(a, np.ndarray) for a in arrays.values())
    state = {**arrays, 'object_q': object_q, 'object_qd': object_qd,
             'native_physics_step_index': np.asarray(17), 'native_world_time_s': np.asarray(.0085)}
    payload = readback.payload(state, ['base', 'slider'])
    assert payload['native_physics_step_index'] == 17
    assert payload['joints']['/World/Imported/SliderJoint']['position_m'] == .125
    assert payload['joints']['/World/Imported/SliderJoint']['velocity_m_s'] == .4
    assert payload['sampling']['physics_steps_added'] == 0
    assert payload['sampling']['particle_force_or_weight_feedback_qualified'] is False
