"""Hinge composition/readback regressions; CPU geometry is not native dynamics."""
from copy import deepcopy
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle, joint_positions
from bench.simulation.isaacsim_backend.revolute_readback import revolute_coordinate
from bench.simulation.isaacsim_backend.task_layers import mount_task_layers
from bench.simulation.isaacsim_backend.task_readback import TaskLayerReadback, revolute_state


def author(tmp_path, *, world_anchor=False, mutation=None):
    path = tmp_path/'hinge.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    root = '/World/Hinge'
    UsdGeom.Xform.Define(stage, root)
    bodies = {}
    for name in ('base', 'panel'):
        body = UsdGeom.Xform.Define(stage, root+'/'+name)
        body.AddTranslateOp().Set((0., 0., .8))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.2)
        shape = UsdGeom.Cube.Define(stage, root+'/'+name+'/shape')
        shape.CreateSizeAttr(.04)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        bodies[name] = str(body.GetPath())
    joint = UsdPhysics.RevoluteJoint.Define(stage, root+'/joint')
    joint.CreateBody0Rel().SetTargets([] if world_anchor else [bodies['base']])
    joint.CreateBody1Rel().SetTargets([bodies['panel']])
    joint.CreateAxisAttr('Y')
    joint.CreateLowerLimitAttr(-100.)
    joint.CreateUpperLimitAttr(0.)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0., 0., .8 if world_anchor else 0.))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
    if mutation in ('damping', 'moving_drive', 'spring_drive'):
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(1. if mutation == 'spring_drive' else 0.)
        drive.CreateDampingAttr(.02)
        drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(3. if mutation == 'moving_drive' else 0.)
    elif mutation == 'bad_axis':
        joint.GetAxisAttr().Set('W')
    elif mutation == 'bad_limits':
        joint.GetUpperLimitAttr().Set(-110.)
    elif mutation == 'disabled':
        joint.CreateJointEnabledAttr(False)
    elif mutation == 'same_body':
        joint.GetBody0Rel().SetTargets([bodies['panel']])
    elif mutation == 'bad_rotation':
        joint.CreateLocalRot0Attr(Gf.Quatf(0., 0., 0., 0.))
    stage.GetRootLayer().Save()
    return path, bodies, str(joint.GetPath())


def mount(path, bodies, joint, *, bundle):
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, '/physicsScene')
    if bundle:
        spec = {'schema': 'native-usd-task-bundle-v1', 'usd_path': str(path), 'usd_sha256': sha,
                'root_prim_path': '/World/Hinge', 'rigid_bodies': bodies,
                'joints': {'hinge': joint}, 'particle_sets': {}}
        info = add_bundle(stage, spec, physics_scene_path='/physicsScene')
    else:
        spec = {'id': 'hinge', 'usd_path': str(path), 'sha256': sha,
                'source_prim_path': '/World/Hinge', 'target_prim_path': '/World/Hinge',
                'rigid_bodies': bodies, 'dependencies_sha256': {}, 'particle_sets': {}}
        info = mount_task_layers(stage, [spec], physics_scene_path='/physicsScene')
    return stage, info


@pytest.mark.parametrize('bundle', [False, True])
@pytest.mark.parametrize('damping', [False, True])
def test_passive_hinge_import_preserves_source_constraints(tmp_path, bundle, damping):
    path, bodies, joint = author(tmp_path, mutation='damping' if damping else None)
    source = path.read_bytes()
    original = Usd.Stage.Open(str(path))
    stage, info = mount(path, bodies, joint, bundle=bundle)
    before, after = original.GetPrimAtPath(joint), stage.GetPrimAtPath(joint)
    assert before.GetTypeName() == after.GetTypeName() == 'PhysicsRevoluteJoint'
    assert before.GetAppliedSchemas() == after.GetAppliedSchemas()
    for attr in before.GetAttributes():
        assert attr.Get() == after.GetAttribute(attr.GetName()).Get()
    for rel in before.GetRelationships():
        assert rel.GetTargets() == after.GetRelationship(rel.GetName()).GetTargets()
    row = info['joints']['hinge'] if bundle else info['layers'][0]['joints'][0]
    assert row['axis'] == 'Y' and row['limits_deg'] == [-100., 0.]
    assert ('angular_damping' in row) == damping
    assert path.read_bytes() == source
    if bundle:
        positions = joint_positions(info['objects'], info['joints'])
        assert positions['hinge']['position_rad'] == 0.
        assert positions['hinge']['anchor_separation_m'] == 0.
        json.dumps(positions, allow_nan=False)


@pytest.mark.parametrize('bundle', [False, True])
@pytest.mark.parametrize('mutation', ['moving_drive', 'spring_drive', 'bad_axis', 'bad_limits',
                                     'disabled', 'same_body', 'bad_rotation'])
def test_passive_hinge_rejects_active_or_invalid_constraints(tmp_path, bundle, mutation):
    path, bodies, joint = author(tmp_path, mutation=mutation)
    reason = {'moving_drive': 'forbids', 'spring_drive': 'forbids',
              'bad_axis': 'XYZ axis', 'bad_limits': 'ordered limits',
              'disabled': 'enabled joint', 'same_body': 'distinct bodies',
              'bad_rotation': 'unit'}[mutation]
    with pytest.raises(ValueError, match=reason):
        mount(path, bodies, joint, bundle=bundle)


@pytest.mark.parametrize('axis', ['X', 'Y', 'Z'])
def test_rotated_moving_joint_frames_report_signed_angle_and_true_derivative(axis):
    i = 'XYZ'.index(axis)
    local0 = Rotation.from_euler('xyz', [.3, -.2, .1])
    body0 = Rotation.from_euler('xyz', [-.1, .4, .2])
    local1 = Rotation.from_euler('xyz', [.2, .1, -.3])
    joint0 = body0*local0
    hinge = Rotation.from_rotvec(np.eye(3)[i]*-.9)
    body1 = joint0*hinge*local1.inv()
    omega0 = np.array([.4, -.3, .2])
    omega1 = omega0+joint0.apply(np.eye(3)[i])*.7
    row = {'axis': axis, 'body0': 'a', 'body1': 'b',
           'local_rot0_xyzw': local0.as_quat(), 'local_rot1_xyzw': local1.as_quat()}
    bodies = {'a': {'pose_xyzw': np.r_[1., 2., 3., body0.as_quat()], 'velocity': np.r_[0., 0., 0., omega0]},
              'b': {'pose_xyzw': np.r_[1., 2., 3., body1.as_quat()], 'velocity': np.r_[0., 0., 0., omega1]}}
    angle, velocity = revolute_state(row, bodies)
    assert angle == pytest.approx(-.9) and velocity == pytest.approx(.7)
    dt = 1.e-7
    moved = deepcopy(bodies)
    for name, rotation, omega in [('a', body0, omega0), ('b', body1, omega1)]:
        moved[name]['pose_xyzw'][3:] = (Rotation.from_rotvec(omega*dt)*rotation).as_quat()
    assert (revolute_state(row, moved)[0]-angle)/dt == pytest.approx(velocity, abs=1.e-7)


def test_layer_world_hinge_captures_radians_without_changing_prismatic_payload(tmp_path):
    path, bodies, joint = author(tmp_path, world_anchor=True)
    stage, info = mount(path, bodies, joint, bundle=False)
    reader = TaskLayerReadback(stage, info['layers'])
    reader.coms = {name: np.zeros(3) for name in bodies}
    q = np.array([[0., 0., .8, 0., 0., 0., 1.],
                  [0., 0., .8, *Rotation.from_euler('y', -.8).as_quat()]])
    qd = np.zeros((2, 6)); qd[1, 4] = -.2
    original = q.copy(), qd.copy()
    arrays = reader.capture_arrays(SimpleNamespace(), list(bodies), q, qd)
    state = {'object_q': q, 'object_qd': qd, 'native_physics_step_index': 1,
             'native_world_time_s': .0005, **arrays}
    payload = reader.payload(state, list(bodies))['joints'][joint]
    assert payload['kind'] == 'revolute'
    assert payload['position_rad'] == pytest.approx(-.8)
    assert payload['velocity_rad_s'] == pytest.approx(-.2)
    assert 'position_m' not in payload
    assert np.array_equal(q, original[0]) and np.array_equal(qd, original[1])


def test_degenerate_joint_geometry_is_not_a_lock_or_force_reading():
    with pytest.raises(ValueError, match='singular'):
        revolute_coordinate(Rotation.from_euler('z', math.pi/2).as_matrix(), 'X')
