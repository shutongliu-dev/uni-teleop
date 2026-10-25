"""Strict public tools and analytically known metric projection; no simulator."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math

import numpy as np
import pytest

from agent.model_client.tools import ToolError, parse_tool_call, project_depth_point


@pytest.fixture
def rgbd_inputs():
    depth = np.full((4, 6), 2., dtype=np.float32)
    depth[0, 0] = 0.
    stream = BytesIO()
    np.save(stream, depth, allow_pickle=False)
    raw = stream.getvalue()
    camera = {'camera_id':'task','sim_time_s':.4, 'rgb_uri':'/episodes/ep/images/3/task',
        'rgb_sha256':'a'*64,'width':6,'height':4,'depth_uri':'/episodes/ep/depth/3/task',
        'depth_sha256':hashlib.sha256(raw).hexdigest(), 'depth_unit':'m',
        'depth_encoding':'float32_npy','depth_semantics':'camera_z','depth_invalid_value':0.,
        'depth_sim_time_s':.4,'depth_rgb_aligned':True,
        'intrinsics_3x3':[2.,0.,3.,0.,4.,2.,0.,0.,1.],
        'world_from_camera':{'frame_id':'world','position_m':[1.,2.,3.],
                             'quaternion_wxyz':[1.,0.,0.,0.]}}
    obs = {'episode_id':'ep','sequence':20,'sim_time_s':.4,'instruction':'A public task.',
        'profile':'vision_proprio','last_action_status':'completed',
        'proprioception':{'joint_positions_rad':{'joint':0.},'joint_velocities_rad_s':{'joint':0.}},
        'cameras':[camera]}
    rows = [{'metadata':deepcopy(camera),'depth_npy_bytes':raw,'depth_m':depth}]
    return rows, obs


def test_tool_parser_accepts_only_one_explicit_call_and_preserves_sparse_move():
    assert parse_tool_call({'tool':'observe','arguments':{}}) == {'tool':'observe','arguments':{}}
    assert parse_tool_call('```json\n{"tool":"point_at","arguments":{"camera_id":"task","u":10,"v":20}}\n```') == {
        'tool':'point_at','arguments':{'camera_id':'task','u':10,'v':20}}
    call = {'tool':'move_geometry','arguments':{'fingertip_delta_m':{'thumb':[.001,0.,0.]},
                                             'duration_s':3.,'reason':'A small public adjustment'}}
    assert parse_tool_call(call) == call
    assert parse_tool_call({'tool':'stop','arguments':{'reason':'Done observing'}})['arguments']['reason'] == 'Done observing'


@pytest.mark.parametrize('value', [
    {'tool':'grasp','arguments':{}}, {'tool':'locate_object','arguments':{'object':'cube'}},
    {'tool':'observe','arguments':{'object':'cube'}}, {'tool':'observe','arguments':[]},
    {'tool':['observe'],'arguments':{}}, {'tool':'observe'},
    {'tool':'observe','arguments':{},'python':'import os'},
    {'tool':'point_at','arguments':{'camera_id':'task','u':True,'v':0}},
    {'tool':'point_at','arguments':{'camera_id':'task','u':1.,'v':0}},
    {'tool':'point_at','arguments':{'camera_id':'task','u':-1,'v':0}},
    {'tool':'point_at','arguments':{'camera_id':'../private','u':0,'v':0}},
    {'tool':'point_at','arguments':{'camera_id':'task','u':0,'v':0,'object':'cube'}},
    {'tool':'move_geometry','arguments':{'action':'stop'}},
    {'tool':'move_geometry','arguments':{'duration_s':3.}},
    {'tool':'stop','arguments':{'object_pose':[0,0,0]}},
    '{"tool":"observe","tool":"stop","arguments":{}}',
    '[{"tool":"observe","arguments":{}}]',
])
def test_unknown_ambiguous_hidden_or_malformed_tool_requests_fail(value):
    with pytest.raises(ToolError):
        parse_tool_call(value)


def test_point_at_uses_half_pixel_centers_and_camera_z_not_ray_range(rgbd_inputs):
    rows, obs = rgbd_inputs
    result = project_depth_point(rows, obs, 'task', 5, 3)
    # K^-1 [5.5,3.5,1] * z = [2.5,.75,2]; then add world translation.
    np.testing.assert_allclose(result['world_position_m'], [3.5,2.75,5.], atol=1e-12)
    assert result['pixel_center_uv'] == [5.5,3.5]
    assert result['depth_m'] == 2. and result['sequence'] == 20
    assert result['camera_revision'] == 3 and result['episode_id'] == 'ep'
    assert result['depth_sha256'] == obs['cameras'][0]['depth_sha256']
    assert result['depth_semantics'] == 'camera_z' and result['frame_id'] == 'world'
    assert np.linalg.norm(np.array(result['world_position_m']) - [1.,2.,3.]) > result['depth_m']
    json.dumps(result, allow_nan=False)


def test_point_at_applies_optical_extrinsic_rotation_and_general_pinhole_skew(rgbd_inputs):
    rows, obs = rgbd_inputs
    camera = obs['cameras'][0]
    camera['world_from_camera']['quaternion_wxyz'] = [math.sqrt(.5),0.,0.,math.sqrt(.5)]
    camera['intrinsics_3x3'][1] = .5
    rows[0]['metadata'] = deepcopy(camera)
    result = project_depth_point(rows, obs, 'task', 5, 3)
    # With skew, optical point is [2.3125,.75,2]; +90deg world z gives [-.75,2.3125,2].
    np.testing.assert_allclose(result['world_position_m'], [.25,4.3125,5.], atol=1e-12)


@pytest.mark.parametrize('camera,u,v', [('task',0,0),('task',6,0),('task',0,4),
                                      ('task',True,1),('task',1,1.),('missing',1,1)])
def test_missing_invalid_or_out_of_bounds_pixel_never_fabricates_a_point(rgbd_inputs, camera, u, v):
    with pytest.raises(ToolError):
        project_depth_point(*rgbd_inputs, camera, u, v)


@pytest.mark.parametrize('change', [
    lambda rows,obs: rows[0].update(depth_npy_bytes=b'not NPY'),
    lambda rows,obs: rows[0]['depth_m'].__setitem__((1,1),3.),
    lambda rows,obs: rows[0]['metadata'].update(depth_sha256='f'*64),
    lambda rows,obs: rows[0].update(png_bytes=b'unverified RGB'),
    lambda rows,obs: rows.append(deepcopy(rows[0])),
    lambda rows,obs: rows.clear(),
    lambda rows,obs: obs['cameras'].append(deepcopy(obs['cameras'][0])),
    lambda rows,obs: obs.update(sim_time_s=.5),
    lambda rows,obs: obs.update(last_action_status='running'),
    lambda rows,obs: obs.update(privileged_object_poses={}),
    lambda rows,obs: obs['cameras'][0].update(depth_uri='/episodes/other/depth/3/task'),
    lambda rows,obs: obs['cameras'][0].update(depth_uri='/episodes/ep/depth/4/task'),
])
def test_unbound_stale_private_or_tampered_sensor_inputs_fail(rgbd_inputs, change):
    change(*rgbd_inputs)
    with pytest.raises(ToolError):
        project_depth_point(*rgbd_inputs, 'task', 1, 1)


@pytest.mark.parametrize('change', [
    lambda camera: camera['intrinsics_3x3'].__setitem__(0,0.),
    lambda camera: camera['intrinsics_3x3'].__setitem__(4,-1.),
    lambda camera: camera['intrinsics_3x3'].__setitem__(8,2.),
    lambda camera: camera['world_from_camera'].update(frame_id='robot_base'),
    lambda camera: camera.update(depth_semantics='ray_range'),
])
def test_invalid_calibration_or_semantics_is_rejected_even_if_rows_match(rgbd_inputs, change):
    rows, obs = rgbd_inputs
    change(obs['cameras'][0])
    rows[0]['metadata'] = deepcopy(obs['cameras'][0])
    with pytest.raises(ToolError):
        project_depth_point(rows, obs, 'task', 1, 1)


def test_no_depth_is_explicitly_unavailable(rgbd_inputs):
    rows, obs = rgbd_inputs
    camera = obs['cameras'][0]
    for name in list(camera):
        if name.startswith('depth_'):
            camera.pop(name)
    rows[0]['metadata'] = deepcopy(camera)
    with pytest.raises(ToolError, match='no aligned calibrated metric depth'):
        project_depth_point(rows, obs, 'task', 1, 1)
