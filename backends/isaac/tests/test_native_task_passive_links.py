"""Passive assembly links preserve USD physics and expose geometry-only evidence."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from bench.simulation.isaacsim_backend.native_task_bundle import (
    SCHEMA, add_bundle, joint_positions,
)


def chain_asset(tmp_path, *, mutation=None, closed=False):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    path = tmp_path/'passive_chain.usda'
    stage = Usd.Stage.CreateNew(str(path))
    root = '/World/Chain'
    UsdGeom.Xform.Define(stage, root)
    actors = {}
    for i in range(3):
        name = 'link_'+str(i)
        body = UsdGeom.Xform.Define(stage, root+'/'+name)
        body.AddTranslateOp().Set((.04*i, 0., .8))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.02)
        shape = UsdGeom.Capsule.Define(stage, root+'/'+name+'/shape')
        shape.CreateAxisAttr('X')
        shape.CreateHeightAttr(.032)
        shape.CreateRadiusAttr(.004)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        actors[name] = str(body.GetPath())
    joints = {}
    for i, cls in enumerate((UsdPhysics.SphericalJoint, UsdPhysics.FixedJoint)):
        joint = cls.Define(stage, root+'/joint_'+str(i))
        joint.CreateBody0Rel().SetTargets([actors['link_'+str(i)]])
        joint.CreateBody1Rel().SetTargets([actors['link_'+str(i+1)]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(.02, 0., 0.))
        joint.CreateLocalPos1Attr(Gf.Vec3f(-.02, 0., 0.))
        joint.CreateCollisionEnabledAttr(False)
        if i == 0:
            joint.CreateAxisAttr('X')
            joint.CreateConeAngle0LimitAttr(60.)
            joint.CreateConeAngle1LimitAttr(60.)
        joints['joint_'+str(i)] = str(joint.GetPath())
    if closed:
        joint = UsdPhysics.SphericalJoint.Define(stage, root+'/closing_link')
        joint.CreateBody0Rel().SetTargets([actors['link_2']])
        joint.CreateBody1Rel().SetTargets([actors['link_0']])
        # Geometry author owns these reset anchors; the importer does not solve
        # the chain or declare that a supplied reset is dynamically stable.
        joint.CreateLocalPos0Attr(Gf.Vec3f(-.04, 0., 0.))
        joint.CreateLocalPos1Attr(Gf.Vec3f(.04, 0., 0.))
        joints['closing_link'] = str(joint.GetPath())
    first = stage.GetPrimAtPath(joints['joint_0'])
    if mutation == 'drive':
        UsdPhysics.DriveAPI.Apply(first, 'angular').CreateTargetVelocityAttr(1.)
    elif mutation == 'disabled':
        first.GetAttribute('physics:jointEnabled').Set(False)
    elif mutation == 'self_link':
        first.GetRelationship('physics:body1').SetTargets([actors['link_0']])
    elif mutation == 'static_endpoint':
        fixed = UsdGeom.Cube.Define(stage, root+'/fixture')
        UsdPhysics.CollisionAPI.Apply(fixed.GetPrim())
        first.GetRelationship('physics:body1').SetTargets([fixed.GetPath()])
    elif mutation == 'world_endpoint':
        first.GetRelationship('physics:body1').SetTargets([])
    elif mutation == 'external_endpoint':
        first.GetRelationship('physics:body1').SetTargets([Sdf.Path('/World/Robot/link')])
    elif mutation == 'invalid_rotation':
        first.GetAttribute('physics:localRot0').Set(Gf.Quatf(0., 0., 0., 0.))
    elif mutation == 'invalid_anchor':
        first.GetAttribute('physics:localPos0').Set(Gf.Vec3f(float('nan'), 0., 0.))
    elif mutation == 'revolute':
        first.SetTypeName('PhysicsRevoluteJoint')
    stage.GetRootLayer().Save()
    return {'schema': SCHEMA, 'usd_path': str(path),
        'usd_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'root_prim_path': root, 'rigid_bodies': actors, 'joints': joints, 'particle_sets': {}}


@pytest.mark.parametrize('closed', [False, True])
def test_chain_composition_preserves_source_and_joint_properties(tmp_path, closed):
    from pxr import Usd

    row = chain_asset(tmp_path, closed=closed)
    source = open(row['usd_path'], 'rb').read()
    original = Usd.Stage.Open(row['usd_path'])
    stage = Usd.Stage.CreateInMemory()
    info = add_bundle(stage, row, physics_scene_path='/physicsScene')
    assert set(info['objects']) == set(row['rigid_bodies'])
    assert set(info['joints']) == set(row['joints'])
    assert info['joints']['joint_0']['kind'] == 'spherical'
    assert info['joints']['joint_1']['kind'] == 'fixed'
    for path in row['joints'].values():
        before, after = original.GetPrimAtPath(path), stage.GetPrimAtPath(path)
        assert before.GetTypeName() == after.GetTypeName()
        for attr in before.GetAttributes():
            assert attr.Get() == after.GetAttribute(attr.GetName()).Get()
        for rel in before.GetRelationships():
            assert rel.GetTargets() == after.GetRelationship(rel.GetName()).GetTargets()
    assert open(row['usd_path'], 'rb').read() == source
    states = joint_positions(info['objects'], info['joints'])
    assert all(r['anchor_separation_m'] < 1.e-8 for r in states.values())
    json.dumps(states, allow_nan=False)


@pytest.mark.parametrize('mutation,reason', [
    ('drive', 'no drives'), ('disabled', 'enabled joint'), ('self_link', 'distinct bodies'),
    ('static_endpoint', 'declared dynamic bodies'), ('world_endpoint', 'declared dynamic bodies'),
    ('external_endpoint', 'declared dynamic bodies'), ('invalid_rotation', 'unit wxyz'),
    ('invalid_anchor', 'finite vector'), ('revolute', 'supports prismatic'),
])
def test_passive_links_reject_active_invalid_or_external_constraints(tmp_path, mutation, reason):
    from pxr import Usd

    row = chain_asset(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match=reason):
        add_bundle(Usd.Stage.CreateInMemory(), row, physics_scene_path='/physicsScene')


@pytest.mark.parametrize('mutation', ['missing', 'duplicate'])
def test_passive_joint_inventory_remains_exact(tmp_path, mutation):
    from pxr import Usd

    row = chain_asset(tmp_path)
    if mutation == 'missing':
        del row['joints']['joint_1']
    else:
        row['joints']['alias'] = row['joints']['joint_0']
    with pytest.raises(ValueError, match='joint inventory'):
        add_bundle(Usd.Stage.CreateInMemory(), row, physics_scene_path='/physicsScene')


@pytest.mark.parametrize('kind', ['spherical', 'fixed'])
def test_passive_readback_uses_both_local_frames_and_native_body_poses(kind):
    # Body0 rotates 90 degrees in world Z, so its local X anchor reaches world Y.
    q90 = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    objects = {
        'a': {'position_m': [1., 2., 3.], 'quaternion_wxyz': q90},
        'b': {'position_m': [1., 3., 3.], 'quaternion_wxyz': [1., 0., 0., 0.]}}
    row = {'kind': kind, 'body0': 'a', 'body1': 'b',
           'anchor0_m': [1., 0., 0.], 'anchor1_m': [0., 0., 0.],
           'local_rotation0_wxyz': [1., 0., 0., 0.], 'local_rotation1_wxyz': q90}
    original = deepcopy(objects)
    result = joint_positions(objects, {'link': row})['link']
    assert result['anchor_separation_m'] < 1.e-12
    assert np.allclose(result['relative_rotation_matrix'], np.eye(3))
    assert objects == original
    # Separate and rotate the body: report the actual constraint error without
    # projecting the body back to its joint or fabricating successful assembly.
    objects['b']['position_m'][2] += .004
    objects['b']['quaternion_wxyz'] = q90
    result = joint_positions(objects, {'link': row})['link']
    assert result['anchor_separation_m'] == pytest.approx(.004)
    assert result['anchor_delta_world_m'] == pytest.approx([0., 0., .004])
    assert np.allclose(result['relative_rotation_matrix'], [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    assert 'position_m' not in result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('field,value', [
    ('position_m', [0., float('inf'), 0.]),
    ('quaternion_wxyz', [0., 0., 0., 0.]),
    ('quaternion_wxyz', [1., 0., float('nan'), 0.]),
])
def test_passive_readback_rejects_nonfinite_or_invalid_native_states(tmp_path, field, value):
    from pxr import Usd

    row = chain_asset(tmp_path)
    info = add_bundle(Usd.Stage.CreateInMemory(), row, physics_scene_path='/physicsScene')
    info['objects']['link_0'][field] = value
    with pytest.raises(ValueError, match='finite|unit wxyz'):
        joint_positions(info['objects'], info['joints'])


def test_prismatic_readback_retains_exact_original_shape():
    objects = {
        'a': {'position_m': [0., 0., 0.], 'quaternion_wxyz': [1., 0., 0., 0.]},
        'b': {'position_m': [0., 0., .002], 'quaternion_wxyz': [1., 0., 0., 0.]}}
    rows = {'button': {'body0': 'a', 'body1': 'b', 'anchor0_m': [0., 0., 0.],
                       'anchor1_m': [0., 0., 0.], 'axis0': [0., 0., 1.]}}
    assert joint_positions(objects, rows) == {
        'button': {'position_m': .002, 'body0': 'a', 'body1': 'b'}}
