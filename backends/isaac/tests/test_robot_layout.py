"""Opposed original-geometry dual-FR3 composition and FK contracts, CPU only."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.robot_layout import arm_base_poses, robot_base_pose, world_base_poses, reference_robot
from bench.simulation.isaacsim_backend.robot_base import MODE, apply_base_constraint, runtime_base_check

ASSETS = Path(f'{DATA_ROOT}/assets')
ASSEMBLY = ASSETS / 'robots/assemblies/dual_fr3_sharpa_w01/isaacsim_6_1_v2'
ROBOT = {'hand_id': 'dual_fr3_sharpa_w01', 'assembly_version': 'isaacsim_6_1_v2',
         'base_position_m': [.65, 0., .95],
         'arm_base_poses': {
             'left': {'position_m': [0., -.52, 0.], 'quaternion_wxyz': [2**-.5, 0., 0., 2**-.5]},
             'right': {'position_m': [0., .52, 0.], 'quaternion_wxyz': [2**-.5, 0., 0., -2**-.5]}}}


def test_layout_omission_keeps_registry_offsets_and_identity_orientation():
    robot = {'hand_id': 'dual_fr3_sharpa_w01', 'base_position_m': [0., 0., .77]}
    assert world_base_poses(robot) == {
        'left': {'position_m': [0., .3, .77], 'quaternion_wxyz': [1., 0., 0., 0.]},
        'right': {'position_m': [0., -.3, .77], 'quaternion_wxyz': [1., 0., 0., 0.]}}
    assert arm_base_poses({}) is None
    assert world_base_poses({}) == {'single': robot_base_pose({})}


@pytest.mark.parametrize('problem', ['single', 'missing_side', 'extra_side', 'missing_quaternion',
                                    'extra_key', 'nonunit', 'nan', 'position_size'])
def test_layout_rejects_ambiguous_or_nonrigid_poses(problem):
    robot = deepcopy(ROBOT)
    if problem == 'single': robot['hand_id'] = 'sharpa_w01_right'
    elif problem == 'missing_side': del robot['arm_base_poses']['left']
    elif problem == 'extra_side': robot['arm_base_poses']['third'] = robot['arm_base_poses']['left']
    elif problem == 'missing_quaternion': del robot['arm_base_poses']['left']['quaternion_wxyz']
    elif problem == 'extra_key': robot['arm_base_poses']['left']['scale'] = [1, 1, 2]
    elif problem == 'nonunit': robot['arm_base_poses']['left']['quaternion_wxyz'] = [2, 0, 0, 0]
    elif problem == 'nan': robot['arm_base_poses']['left']['position_m'][0] = float('nan')
    elif problem == 'position_size': robot['arm_base_poses']['left']['position_m'] = [0, 0]
    with pytest.raises(ValueError): arm_base_poses(robot)


def test_common_rotation_and_independent_side_rotations_match_public_fk():
    from bench.geometry import quaternion_matrix, transform
    from bench.runtime.controllers import robot_configuration, public_robot_geometry
    robot = deepcopy(ROBOT)
    robot['base_quaternion_wxyz'] = [2**-.5, 0., 0., 2**-.5]
    geometry = public_robot_geometry(robot_configuration({'robot': robot}))
    for side, pose in world_base_poses(robot).items():
        assert np.allclose(geometry.sides[side].arm_base,
                           transform(pose['position_m'], quaternion_matrix(pose['quaternion_wxyz'])), atol=1.e-12)
        own = geometry.sides[side]
        assert len(own.names) == 29
        assert all(n.startswith((f'fr3_{side}_', f'{side}_')) for n in own.names)
    assert set(geometry.sides['left'].names).isdisjoint(geometry.sides['right'].names)


@pytest.fixture
def dual_stage():
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom
    usd = {side: ASSEMBLY / f'usd_{side}/fr3_sharpa_{side}/fr3_sharpa_{side}.usda'
           for side in ('left', 'right')}
    if not all(path.is_file() for path in usd.values()):
        pytest.skip('Pinned dual FR3 + SHARPA USDs not mounted')
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    # A reversed input mapping must not reverse native control/reset ownership.
    reference_robot(stage, root, dict(reversed(list(usd.items()))), ROBOT)
    return stage, root, usd


def test_original_dual_usd_rotations_anchors_topology_and_source_bytes(dual_stage):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage, root, usd = dual_stage
    original_files = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                      for source in usd.values() for p in source.parent.rglob('*') if p.is_file()}
    def physical_inventory():
        return {str(p.GetPath()): {a.GetName(): str(a.Get()) for a in p.GetAttributes()
                    if a.HasAuthoredValueOpinion() or not a.GetName().startswith('newton:')}
                for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if p.IsA(UsdPhysics.RevoluteJoint) or p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)}
    before = physical_inventory()
    record = apply_base_constraint(stage, root, {'robot': ROBOT, 'isaacsim': {'robot_base_constraint': MODE}})
    assert before == physical_inventory()
    assert len(record['articulations']) == 2
    for side, expected in zip(('left', 'right'), record['articulations']):
        assert len(expected['expected_dof_names']) == 29
        assert len(expected['expected_link_names']) == 43
        base = stage.GetPrimAtPath(expected['root_body_path'])
        frame = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pose = world_base_poses(ROBOT)[side]
        assert list(frame.ExtractTranslation()) == pytest.approx(pose['position_m'])
        actual_q = frame.ExtractRotationQuat()
        assert abs(float(actual_q.GetReal()) * pose['quaternion_wxyz'][0]
                   + np.dot(actual_q.GetImaginary(), pose['quaternion_wxyz'][1:])) == pytest.approx(1., abs=1.e-6)
        joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(expected['articulation_path']))
        assert not joint.GetBody0Rel().GetTargets()
        assert joint.GetBody1Rel().GetTargets() == [base.GetPath()]
        assert list(joint.GetLocalPos0Attr().Get()) == pytest.approx(pose['position_m'])
        q = joint.GetLocalRot0Attr().Get()
        assert Gf.IsClose(Gf.Matrix3d(q), Gf.Matrix3d(actual_q), 1.e-6)
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ['/World'])
    descs = parsed[UsdPhysics.ObjectType.Articulation][1]
    assert len(descs) == 2
    assert {str(d.rootPrims[0]) for d in descs} == {r['articulation_path'] for r in record['articulations']}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in original_files.items())


def test_dual_preflight_failure_on_second_arm_does_not_edit_first(dual_stage):
    from pxr import Usd, UsdPhysics
    stage, root, _ = dual_stage
    right_joints = [UsdPhysics.FixedJoint(p) for p in Usd.PrimRange(root)
                    if p.IsA(UsdPhysics.FixedJoint) and '/Right/' in str(p.GetPath())
                    and any(t.name == 'fr3_right_link0' for t in UsdPhysics.Joint(p).GetBody1Rel().GetTargets())]
    assert len(right_joints) == 1
    right_joints[0].CreateJointEnabledAttr(False)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='enabled'):
        apply_base_constraint(stage, root, {'robot': ROBOT, 'isaacsim': {'robot_base_constraint': MODE}})
    assert stage.GetRootLayer().ExportToString() == before


def test_dual_cross_arm_joint_is_not_an_independent_installation(dual_stage):
    from pxr import Usd, UsdPhysics
    stage, root, _ = dual_stage
    bases = [p.GetPath() for p in Usd.PrimRange(root) if p.GetName() in ('fr3_left_link0', 'fr3_right_link0')]
    bridge = UsdPhysics.FixedJoint.Define(stage, '/World/Robot/Left/Physics/Bridge')
    bridge.CreateBody0Rel().SetTargets([bases[0]])
    bridge.CreateBody1Rel().SetTargets([bases[1]])
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='ownership'):
        apply_base_constraint(stage, root, {'robot': ROBOT, 'isaacsim': {'robot_base_constraint': MODE}})
    assert stage.GetRootLayer().ExportToString() == before


def test_native_dual_check_matches_by_path_and_rejects_wrong_pose_or_owned_joints(dual_stage):
    stage, root, _ = dual_stage
    request = apply_base_constraint(stage, root, {'robot': ROBOT, 'isaacsim': {'robot_base_constraint': MODE}})
    records = list(reversed(request['articulations']))
    views = []
    for record in records:
        q = record['initial_root_quaternion_wxyz']
        pose = np.array([[*record['initial_root_position_m'], *q[1:], q[0]]])
        names = [record['root_body_name']] + [n for n in record['expected_link_names'] if n != record['root_body_name']]
        meta = SimpleNamespace(fixed_base=True, dof_names=record['expected_dof_names'], link_names=names)
        views.append(SimpleNamespace(count=1, get_metatype=lambda index, m=meta: m,
                                     get_root_transforms=lambda p=pose: p))
    composite = SimpleNamespace(views=views)
    info = {'articulation_path': [r['articulation_path'] for r in records], 'robot_base_constraint': request}
    result = runtime_base_check(composite, info)
    assert result['all_fixed_base'] and all(r['root_pose_verified'] for r in result['articulations'])
    views[0].get_root_transforms()[0, 0] += .01
    with pytest.raises(RuntimeError, match='installation pose'):
        runtime_base_check(composite, info)
