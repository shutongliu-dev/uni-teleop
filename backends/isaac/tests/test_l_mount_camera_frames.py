"""Actual immutable USD composition and camera authoring, no Kit/GPU/render."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from bench.paths import DATA_ROOT
from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
from bench.simulation.isaacsim_backend.cameras import resolve_cameras
from bench.simulation.isaacsim_backend.l_mount import VERSION, prepared_assembly, prepared_import
from bench.simulation.isaacsim_backend.robot_frames import nonphysical_frame_paths
from bench.simulation.isaacsim_backend.robot_layout import reference_robot

ASSETS = Path(f'{DATA_ROOT}/assets')


@pytest.fixture(params=['single', 'dual', 'instances'])
def actual_stage(request):
    kind = request.param
    if kind == 'single':
        urdf, source = prepared_assembly(ASSETS)
        usd = prepared_import(urdf)
        robot = {'hand_id': 'sharpa_w01_right', 'assembly_version': VERSION}
    elif kind == 'dual':
        members = {s: prepared_assembly(ASSETS, side=s) for s in ('left', 'right')}
        source = {'sides': {s: value[1] for s, value in members.items()}}
        usd = {s: prepared_import(value[0], side=s) for s, value in members.items()}
        robot = {'hand_id': 'dual_fr3_sharpa_w01', 'assembly_version': VERSION}
    else:
        from bench.simulation.isaacsim_backend.instance_asset import assemble_instance
        members = {n: assemble_instance(n, ASSETS, source_version=VERSION) for n in ('front', 'rear')}
        source = {'sides': {n: value[1] for n, value in members.items()}}
        usd = {n: prepared_import(value[0]) for n, value in members.items()}
        robot = {'hand_id': 'fr3_sharpa_instances', 'assembly_version': 'namespaced_v1',
            'instances': {n: {'hand_id': 'sharpa_w01_right', 'assembly_version': VERSION,
                'base_position_m': [0., y, 0.], 'base_quaternion_wxyz': [1., 0., 0., 0.]}
                for n, y in (('front', -.4), ('rear', .4))}}
    robot.update(base_position_m=[.2, -.3, .8], base_quaternion_wxyz=[2**-.5, 0., 0., 2**-.5])
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    reference_robot(stage, root, usd, robot)
    rigid = {p.GetName(): str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    return stage, robot, source, rigid


def physical_inventory(stage):
    return {str(p.GetPath()): {a.GetName(): str(a.Get()) for a in p.GetAttributes()}
            for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdPhysics.Joint)}


def test_five_actual_assets_keep_virtual_frames_out_of_rigid_inventory(actual_stage):
    stage, robot, source, rigid = actual_stage
    before = physical_inventory(stage); original = deepcopy(rigid)
    frames = nonphysical_frame_paths(stage, robot, rigid, source)
    assert len(frames) == (1 if robot['hand_id'] == 'sharpa_w01_right' else 2)
    assert not set(frames) & set(rigid)
    assert frames == nonphysical_frame_paths(stage, robot, rigid)  # direct build_scene fallback
    assert rigid == original and physical_inventory(stage) == before
    assert all(stage.GetPrimAtPath(path).GetParent().HasAPI(UsdPhysics.RigidBodyAPI) for path in frames.values())


@pytest.mark.parametrize('aim', [False, True])
def test_camera_paths_and_aim_use_real_virtual_frame_world_pose(actual_stage, aim, monkeypatch):
    stage, robot, source, rigid = actual_stage
    before = physical_inventory(stage)
    frames = nonphysical_frame_paths(stage, robot, rigid, source)
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.stage = stage; env.scene_info = {'robot_link_paths': rigid, 'robot_frame_paths': frames}
    env._link_camera_paths = {}
    specs = {f'wrist_{i}': {'mount_link': name, 'aim': False, 'offset_xyz_m': [.01, .02, .03],
             **({'aim_target_m': [0., 0., .7], 'standoff_m': 0.} if aim else {})}
             for i, name in enumerate(frames)}
    env.cameras = resolve_cameras({'cameras': specs})
    import bench.simulation.isaacsim_backend.env as env_module
    original = env_module.aimed_link_matrix; received = []
    def checked_matrix(matrix, *args):
        received.append(np.asarray(matrix).copy())
        return original(matrix, *args)
    monkeypatch.setattr(env_module, 'aimed_link_matrix', checked_matrix)
    paths = env._author_cameras()
    cache = UsdGeom.XformCache()
    for index, (name, frame_path) in enumerate(frames.items()):
        key = f'wrist_{index}'
        assert paths[key] == frame_path+'/bench_camera_'+key
        camera = stage.GetPrimAtPath(paths[key]); assert camera.IsA(UsdGeom.Camera)
        assert camera.GetParent().GetPath() == stage.GetPrimAtPath(frame_path).GetPath()
        if aim:
            np.testing.assert_allclose(received[index], cache.GetLocalToWorldTransform(stage.GetPrimAtPath(frame_path)), atol=1e-14)
        else:
            local = np.asarray(UsdGeom.Xformable(camera).GetLocalTransformation())
            np.testing.assert_allclose(local[3, :3], [.01, .02, .03], atol=1e-14)
    assert physical_inventory(stage) == before
    assert set(rigid).isdisjoint(frames)


@pytest.mark.parametrize('problem', ['mass', 'rigid', 'collision', 'missing', 'transform', 'reset_stack', 'duplicate'])
def test_virtual_frame_source_mismatch_is_rejected(actual_stage, problem):
    stage, robot, source, rigid = actual_stage
    source = deepcopy(source)
    row = next(iter(source['sides'].values())) if 'sides' in source else source
    name, spec = next(iter(row['virtual_frames'].items()))
    prim = stage.GetPrimAtPath(rigid[spec['parent_body']]+'/'+name)
    if problem in ('mass', 'rigid', 'collision'):
        {'mass': UsdPhysics.MassAPI, 'rigid': UsdPhysics.RigidBodyAPI, 'collision': UsdPhysics.CollisionAPI}[problem].Apply(prim)
    elif problem == 'missing':
        prim.SetActive(False)
    elif problem == 'transform':
        spec['matrix_column_vector'][0][3] += .001
    elif problem == 'reset_stack':
        UsdGeom.Xformable(prim).SetResetXformStack(True)
    else:
        source = {'sides': {'a': row, 'b': row}}
    with pytest.raises(ValueError):
        nonphysical_frame_paths(stage, robot, rigid, source)


def test_legacy_frame_lookup_is_unchanged_and_alias_is_rejected():
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.scene_info = {'robot_link_paths': {'right_hand_flange': '/World/PhysicalFlange'}}
    env._link_camera_paths = {}
    env.cameras = resolve_cameras({'cameras': {'wrist': {'mount_link': 'right_hand_flange'}}})
    assert env._camera_prim_path('wrist') == '/World/PhysicalFlange/bench_camera_wrist'
    assert nonphysical_frame_paths(None, {}, env.scene_info['robot_link_paths'], {}) == {}
    env.scene_info['robot_frame_paths'] = {'right_hand_flange': '/World/DifferentFrame'}
    with pytest.raises(ValueError, match='aliases'):
        env._camera_mount_paths()
