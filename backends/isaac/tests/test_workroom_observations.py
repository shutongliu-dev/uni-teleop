"""Fixed wrist extrinsics and independent geometric checks; no native GPU."""
from copy import deepcopy
from pathlib import Path
import math

import numpy as np
import pytest

from bench.geometry import rpy_matrix, transform
from bench.paths import DATA_ROOT
from scenes.workrooms.observations import camera_specs, validation_wrist_poses
from bench.simulation.isaacsim_backend.cameras import resolve_cameras
from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv


def description():
    return {'cameras': {name: {'eye': [1.2, -1.3, 1.7], 'target': [.5, 0., .8],
                               'focal_length_mm': focal}
                        for name, focal in [('workcell', 30.), ('tabletop', 41.), ('room', 23.)]}}


def test_four_views_have_independent_intrinsics_and_no_world_target_for_wrist():
    original = description()
    before = deepcopy(original)
    first = camera_specs(original)
    resolved = resolve_cameras({'cameras': first, 'table': {'position_m': [8., 9., 10.],
                                                           'half_extents_m': [1., 1., .5]}})
    assert list(resolved) == ['overview', 'task', 'room', 'wrist']
    wrist = resolved['wrist']
    assert wrist['mount_link'] == 'right_hand_flange'
    assert wrist['aim_target_m'] is None and wrist['aim_source'] is None
    assert wrist['standoff_m'] == 0.
    assert wrist['offset_xyz_m'][2] < .05
    assert np.linalg.norm(wrist['offset_xyz_m']) < .09
    assert wrist['horizontal_aperture_mm'] / wrist['focal_length_mm'] == 2.
    assert wrist['vertical_aperture_mm'] / wrist['horizontal_aperture_mm'] == 720 / 960
    first['overview']['eye'][0] = 99.
    first['wrist']['offset_xyz_m'][0] = 99.
    assert original == before
    assert camera_specs(original)['overview']['eye'][0] == 1.2
    assert camera_specs(original)['wrist']['offset_xyz_m'][0] < 0.


@pytest.mark.parametrize('names',[('front','rear'),('first','second')])
def test_explicit_instances_get_source_wrist_extrinsic_under_each_namespace(names):
    from test_robot_instances import robot
    declaration=robot(names)
    before=deepcopy(declaration)
    original=camera_specs(description())
    views=camera_specs(description(),robot=declaration)
    assert set(views)=={'overview','task','room'}|{n+'_wrist' for n in names}
    for name in ('overview','task','room'):assert views[name]==original[name]
    for namespace in names:
        expected={**original['wrist'],'mount_link':namespace+'_hand_flange'}
        assert views[namespace+'_wrist']==expected
    assert declaration==before
    assert camera_specs(description(),robot={'hand_id':'sharpa_w01_right'})==original
    env=FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.cameras=resolve_cameras({'cameras':views});env._link_camera_paths={}
    env.scene_info={'robot_link_paths':{n+'_hand_flange':'/World/Robot/'+n+'/'+n+'_hand_flange' for n in names}}
    for namespace in names:
        assert env._camera_prim_path(namespace+'_wrist') == (
            '/World/Robot/'+namespace+'/'+namespace+'_hand_flange/bench_camera_'+namespace+'_wrist')


@pytest.mark.parametrize('resolution', [(0, 480), (640, -1), (640.5, 480), (True, 480)])
def test_resolution_requires_actual_positive_integers(resolution):
    with pytest.raises(ValueError, match='positive integers'):
        camera_specs(description(), resolution=resolution)


def test_wrist_is_parented_once_and_moves_as_a_rigid_link_child():
    pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.stage = Usd.Stage.CreateInMemory()
    link_path = '/World/Robot/right_hand_flange'
    parent = UsdGeom.Xform.Define(env.stage, link_path)
    parent_op = parent.AddTransformOp()
    initial = transform([.3, -.3, 1.2], rpy_matrix([.3, -.6, .2]))
    parent_op.Set(Gf.Matrix4d(*initial.T.tolist()))
    env._link_camera_paths = {}
    env.scene_info = {'robot_link_paths': {'right_hand_flange': link_path}}
    env.cameras = resolve_cameras({'cameras': camera_specs(description())})
    paths = env._author_cameras()
    prim = env.stage.GetPrimAtPath(paths['wrist'])
    assert str(prim.GetParent().GetPath()) == link_path
    spec = env.cameras['wrist']
    # Independent column-vector Rz*Ry*Rx reference, not link_camera_matrix.
    expected_local = transform(spec['offset_xyz_m'], rpy_matrix(spec['offset_rpy_rad']))
    local = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T
    np.testing.assert_allclose(local, expected_local, atol=1e-14)
    # Optical -z points down H+z with a small palmar tilt, irrespective of world.
    np.testing.assert_allclose(local[:3, :3] @ [0., 0., -1.],
                               [math.sin(.15), 0., math.cos(.15)], atol=1e-14)
    for rpy, xyz in [([0., 0., 0.], [.2, 0., 1.]),
                     ([math.pi, 0., 0.], [.34, -.3, 1.12]),
                     ([.2, -.4, .8], [.4, -.2, 1.3])]:
        moved = transform(xyz, rpy_matrix(rpy))
        parent_op.Set(Gf.Matrix4d(*moved.T.tolist()))
        actual = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        np.testing.assert_allclose(actual, moved @ expected_local, atol=1e-14)
        np.testing.assert_allclose(np.linalg.inv(moved) @ actual, expected_local, atol=1e-14)
        env._author_cameras()  # A second capture cannot re-aim/re-author it.
        np.testing.assert_array_equal(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()).T, local)
    assert list(UsdGeom.Camera(prim).GetClippingRangeAttr().Get()) == pytest.approx([.015, 8.])


def test_validation_goals_are_reachable_in_the_pinned_v2_assembly():
    from bench.control.fingertip import Fr3SharpaGeometry
    from bench.geometry import quaternion_matrix
    assets = Path(f'{DATA_ROOT}/assets')
    if not (assets / 'robots/franka/fr3/fr3_no_hand.absolute.urdf').is_file():
        pytest.skip('Pinned FR3/SHARPA shared assets unavailable')
    geometry = Fr3SharpaGeometry(assets, [-.13, 0., .78], assembly_version='isaacsim_6_1_v2')
    initial = dict(zip(geometry.arm.names, [0., -.55, 0., -2.15, 0., 1.65, .75]))
    target_position = [.30, -.35, .803]
    for pose in validation_wrist_poses(target_position).values():
        world = transform(pose['position_m'], quaternion_matrix(pose['quaternion_wxyz']))
        goal = np.linalg.inv(geometry.base) @ world @ np.linalg.inv(geometry.wrist_offset)
        result = geometry.arm.inverse(goal, 'fr3_link8', initial, iterations=180,
                                      position_tolerance=.0005, rotation_tolerance=.005)
        assert result['converged'], result
        actual = geometry.base @ geometry.arm.forward(result['joint_positions_rad'])['fr3_link8'] @ geometry.wrist_offset
        np.testing.assert_allclose(actual[:3, 3], pose['position_m'], atol=.0005)
