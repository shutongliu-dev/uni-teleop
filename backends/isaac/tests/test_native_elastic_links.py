"""Reject active/locked substitutes and preserve actual relative link geometry."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.native_task_bundle import (
    SCHEMA, add_bundle, joint_positions, passive_elastic_drives,
)


def assembly(tmp_path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    path = tmp_path/'elastic.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, '/World/ElasticTask')
    bodies = {name: '/World/ElasticTask/'+name for name in ('a', 'b')}
    for index, body_path in enumerate(bodies.values()):
        x = UsdGeom.Xform.Define(stage, body_path)
        x.AddTranslateOp().Set(Gf.Vec3d(0., 0., index*.02))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()).CreateKinematicEnabledAttr(False)
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, body_path+'/Collider').GetPrim())
    j = UsdPhysics.Joint.Define(stage, '/World/ElasticTask/Link')
    j.CreateBody0Rel().SetTargets([Sdf.Path(bodies['a'])])
    j.CreateBody1Rel().SetTargets([Sdf.Path(bodies['b'])])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(0., 0., .01))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0., 0., -.01))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1.))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
    j.CreateExcludeFromArticulationAttr(True)
    for axis in ('transX', 'transY', 'transZ', 'rotX', 'rotY', 'rotZ'):
        api = UsdPhysics.DriveAPI.Apply(j.GetPrim(), axis)
        api.CreateTypeAttr('force')
        api.CreateStiffnessAttr(1000.)
        api.CreateDampingAttr(1.)
        api.CreateTargetPositionAttr(0.)
        api.CreateTargetVelocityAttr(0.)
        api.CreateMaxForceAttr(float('inf'))
    stage.GetRootLayer().Save()
    spec = {'schema': SCHEMA, 'usd_path': str(path),
            'usd_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'root_prim_path': '/World/ElasticTask', 'rigid_bodies': bodies,
            'joints': {'link': str(j.GetPath())}, 'particle_sets': {}}
    return stage, j.GetPrim(), spec


def test_authored_material_link_loads_without_changing_its_drives(tmp_path):
    from pxr import Usd
    source, joint, spec = assembly(tmp_path)
    original = source.GetRootLayer().ExportToString()
    destination = Usd.Stage.CreateInMemory()
    result = add_bundle(destination, spec, physics_scene_path='/World/PhysicsScene')
    assert result['joints']['link']['kind'] == 'elastic_d6'
    assert result['joints']['link']['elastic_drives']['transZ']['stiffness'] == 1000.
    assert passive_elastic_drives(destination.GetPrimAtPath(joint.GetPath())) == passive_elastic_drives(joint)
    assert source.GetRootLayer().ExportToString() == original


def test_declared_composite_requires_connected_internal_material_graph(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    source, _, spec = assembly(tmp_path)
    spec['elastic_composites'] = {'a': {'parent_rigid_body': 'a', 'member_ids': ['a', 'b']}}
    assert add_bundle(Usd.Stage.CreateInMemory(), spec, physics_scene_path='/World/P')['elastic_composites']
    spec['elastic_composites']['a']['member_ids'] = ['a']
    with pytest.raises(ValueError, match='leaves its composite'):
        add_bundle(Usd.Stage.CreateInMemory(), spec, physics_scene_path='/World/P')
    path = '/World/ElasticTask/c'
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(source, path).GetPrim())
    source.GetRootLayer().Save()
    spec['usd_sha256'] = hashlib.sha256(Path(spec['usd_path']).read_bytes()).hexdigest()
    spec['rigid_bodies']['c'] = path
    spec['elastic_composites']['a']['member_ids'] = ['a', 'b', 'c']
    with pytest.raises(ValueError, match='connected passive material links'):
        add_bundle(Usd.Stage.CreateInMemory(), spec, physics_scene_path='/World/P')


@pytest.mark.parametrize('field,value', [('type', 'acceleration'), ('targetPosition', .1),
    ('targetVelocity', .1), ('stiffness', float('inf')), ('stiffness', 0.),
    ('damping', -1.), ('maxForce', 1.)])
def test_active_or_nonfinite_links_are_rejected(tmp_path, field, value):
    stage, joint, _ = assembly(tmp_path)
    joint.GetAttribute('drive:transZ:physics:'+field).Set(value)
    with pytest.raises(ValueError, match='passive finite force springs'):
        passive_elastic_drives(joint)


def test_hard_limit_is_not_an_elastic_material_link(tmp_path):
    from pxr import UsdPhysics
    stage, joint, _ = assembly(tmp_path)
    limit = UsdPhysics.LimitAPI.Apply(joint, 'rotX')
    limit.CreateLowAttr(1.)
    limit.CreateHighAttr(-1.)
    with pytest.raises(ValueError, match='no hard limits'):
        passive_elastic_drives(joint)


def test_relative_geometry_rotates_with_both_bodies_without_inferred_force():
    # Joint0 x lies on world -x after a half-turn around z.
    poses = {'a': {'position_m': [1., 2., 3.], 'quaternion_wxyz': [0., 0., 0., 1.]},
             'b': {'position_m': [.999, 2., 3.], 'quaternion_wxyz': [0., 0., 0., 1.]}}
    links = {'l': {'kind': 'elastic_d6', 'body0': 'a', 'body1': 'b',
                  'anchor0_m': [0., 0., 0.], 'anchor1_m': [0., 0., 0.],
                  'local_rotation0_wxyz': [1., 0., 0., 0.],
                  'local_rotation1_wxyz': [1., 0., 0., 0.]}}
    row = joint_positions(poses, links)['l']
    np.testing.assert_allclose(row['anchor_delta_joint0_m'], [.001, 0., 0.], atol=1e-15)
    np.testing.assert_allclose(row['relative_rotation_matrix'], np.eye(3))
    assert row['force_measured'] is False
