"""Sparse public FK goals and pin/frame/budget rejection; no physics claim."""
from copy import deepcopy
import math

import numpy as np
import pytest

from bench.geometry import quaternion_matrix, rotvec_matrix
from agent.model_client.actions import ActionValidationError
from agent.model_client.geometry_actions import (
    GeometryActionLimits, build_geometry_action, parse_geometry_action,
)


@pytest.fixture
def public_geometry_inputs():
    sha = 'a' * 64
    policy = {'policy_id':'classical_fingertip', 'version':'1.0.0', 'implementation_sha256':sha,
        'trained':False, 'observation_profile':'vision_proprio', 'skills':['move_geometry','hold'],
        'geometry_inputs':['wrist_pose','fingertip_positions','hand_posture'],
        'camera_observation_mode':'decision_only', 'supported_robot_ids':['sharpa_w01_right']}
    caps = {'robot_id':'sharpa_w01_right', 'robot_asset_sha256':sha,
        'arm_joints':['arm'], 'hand_joints':['finger'],
        'finger_names':['thumb','index','middle','ring','little'],
        'joint_limits': {name:{'lower_rad':-2.,'upper_rad':2.} for name in ('arm','finger')},
        'tracks':['direct_joints','system0'], 'system0':policy,
        'frame_ids':['world','robot_base','right_hand_flange']}
    spec = {'episode_id':'ep', 'agent_id':'model', 'agent_version':'v1', 'agent_config_sha256':sha,
        'task_id':'arbitrary_task', 'task_release_sha256':sha, 'evaluator_sha256':sha,
        'robot_id':'sharpa_w01_right', 'robot_asset_sha256':sha, 'physics_profile_sha256':sha,
        'seed':0, 'track':'system0', 'system0':deepcopy(policy), 'instruction':'A public task.',
        'budgets':{'sim_time_s':10.,'wall_time_s':60.,'max_actions':4,'max_model_calls':4,'max_action_sim_s':5.}}
    obs = {'episode_id':'ep', 'sequence':2, 'sim_time_s':.4, 'instruction':'A public task.',
        'profile':'vision_proprio', 'proprioception':{
            'joint_positions_rad':{'arm':0.,'finger':0.}, 'joint_velocities_rad_s':{'arm':0.,'finger':0.},
            'geometry':{'wrist':{'frame_id':'world','position_m':[.4,.1,.9],
                'quaternion_wxyz':[math.sqrt(.5),0.,math.sqrt(.5),0.]},
                'fingertip_frame_id':'right_hand_flange',
                'fingertip_positions_m':{'thumb':[.05,.04,.17],'index':[.05,-.04,.17],
                                       'middle':[.04,-.05,.16]}}}, 'cameras':[],
        'last_action_status':'completed'}
    return obs, caps, spec


def build(inputs, decision=None, **kwargs):
    value = decision or {'action':'move_geometry', 'wrist_delta_m':[.01,0.,-.01],
        'fingertip_delta_m':{'thumb':[0.,-.002,0.],'index':[0.,.002,0.]}, 'duration_s':3.}
    return build_geometry_action(value, *inputs, action_id='model_2',
        expected_sequence=kwargs.pop('expected_sequence',2), **kwargs)


def test_sparse_fingers_and_world_wrist_use_public_fk_without_mutation(public_geometry_inputs):
    before = deepcopy(public_geometry_inputs)
    result = build(public_geometry_inputs)
    assert result['kind'] == 'system0' and result['skill'] == 'move_geometry'
    assert result['timeout_sim_s'] == 3. and result['observation_sequence'] == 2
    assert result['geometry']['fingertip_frame_id'] == 'right_hand_flange'
    assert set(result['geometry']['fingertip_positions_m']) == {'thumb','index'}
    np.testing.assert_allclose(result['geometry']['wrist']['position_m'], [.41,.1,.89])
    np.testing.assert_allclose(result['geometry']['fingertip_positions_m']['thumb'], [.05,.038,.17])
    np.testing.assert_allclose(result['geometry']['fingertip_positions_m']['index'], [.05,-.038,.17])
    assert public_geometry_inputs == before
    assert result['geometry']['hand_posture_rad'] is None
    assert result['object_effect'] == 'none' and result['target'] is None


def test_world_axis_rotation_is_left_multiplied(public_geometry_inputs):
    current = public_geometry_inputs[0]['proprioception']['geometry']['wrist']
    result = build(public_geometry_inputs, {'action':'move_geometry',
        'wrist_rotation_delta_rad':[.1,0.,0.], 'duration_s':3.})
    expected = rotvec_matrix([.1,0.,0.]) @ quaternion_matrix(current['quaternion_wxyz'])
    np.testing.assert_allclose(quaternion_matrix(result['geometry']['wrist']['quaternion_wxyz']), expected, atol=1e-12)
    assert result['geometry']['wrist']['position_m'] == current['position_m']
    assert result['geometry']['fingertip_positions_m'] == {}


@pytest.mark.parametrize('decision', [
    {'action':'joint_delta','delta_rad':{'arm':.01},'duration_s':1.},
    {'action':'move_geometry','duration_s':1.},
    {'action':'move_geometry','wrist_delta_m':[0.,0.],'duration_s':1.},
    {'action':'move_geometry','wrist_delta_m':[True,0.,0.],'duration_s':1.},
    {'action':'move_geometry','wrist_delta_m':[float('nan'),0.,0.],'duration_s':1.},
    {'action':'move_geometry','fingertip_delta_m':{},'duration_s':1.},
    {'action':'move_geometry','fingertip_delta_m':{'pinky':[0.,0.,0.]},'duration_s':1.},
    {'action':'move_geometry','fingertip_delta_m':{'index':[0.,0.,0.]},'duration_s':0.},
    {'action':'move_geometry','wrist_delta_m':[0.,0.,0.],'duration_s':1.,'object_pose':[0,0,0]},
    {'action':'stop','wrist_delta_m':[0.,0.,0.]},
    '{"action":"stop","action":"move_geometry"}',
    '{"action":"move_geometry","wrist_delta_m":[NaN,0,0],"duration_s":1}',
])
def test_invalid_ambiguous_or_privileged_decisions_are_rejected(decision):
    with pytest.raises(ActionValidationError):
        parse_geometry_action(decision)


@pytest.mark.parametrize('fields', [
    {'wrist_delta_m':[.04,.04,0.]}, {'wrist_rotation_delta_rad':[.11,.11,0.]},
    {'fingertip_delta_m':{'index':[.008,.008,0.]}}, {'wrist_delta_m':[1e300,0.,0.]},
    {'wrist_delta_m':[0.,0.,0.], 'duration_s':.3},
    {'wrist_delta_m':[0.,0.,0.], 'duration_s':5.02},
    {'wrist_delta_m':[0.,0.,0.], 'duration_s':.41},
])
def test_norm_and_timing_limits_are_rejected_without_clamping(public_geometry_inputs, fields):
    with pytest.raises(ActionValidationError):
        build(public_geometry_inputs, {'action':'move_geometry','duration_s':3.,**fields})


@pytest.mark.parametrize('change', [
    lambda obs,caps,spec: obs.update(sequence=3),
    lambda obs,caps,spec: obs.update(episode_id='other'),
    lambda obs,caps,spec: obs.update(last_action_status='running'),
    lambda obs,caps,spec: obs.update(privileged_object_poses={}),
    lambda obs,caps,spec: obs['proprioception'].update(geometry=None),
    lambda obs,caps,spec: obs['proprioception']['geometry']['wrist'].update(frame_id='robot_base'),
    lambda obs,caps,spec: obs['proprioception']['joint_positions_rad'].update(extra=0.),
    lambda obs,caps,spec: caps.update(robot_asset_sha256='f'*64),
    lambda obs,caps,spec: caps['system0'].update(implementation_sha256='f'*64),
    lambda obs,caps,spec: caps.update(finger_names=['thumb']),
    lambda obs,caps,spec: spec['budgets'].update(max_action_sim_s=1.),
    lambda obs,caps,spec: obs.update(sim_time_s=9.),
])
def test_stale_private_mismatched_or_overbudget_requests_fail(public_geometry_inputs, change):
    change(*public_geometry_inputs)
    with pytest.raises(ActionValidationError):
        build(public_geometry_inputs)


def test_stop_submits_nothing_and_budget_attempts_are_counted(public_geometry_inputs):
    assert build(public_geometry_inputs, {'action':'stop'}) is None
    with pytest.raises(ActionValidationError):
        build(public_geometry_inputs, actions_used=4)
    with pytest.raises(ActionValidationError):
        build(public_geometry_inputs, expected_sequence=True)


@pytest.mark.parametrize('kwargs', [
    {'max_wrist_translation_m':True}, {'max_wrist_rotation_rad':float('nan')},
    {'max_fingertip_translation_m':0.}, {'min_duration_s':6.,'max_duration_s':5.},
])
def test_invalid_geometry_limit_configuration(kwargs):
    with pytest.raises(ActionValidationError):
        GeometryActionLimits(**kwargs)


@pytest.mark.parametrize('name', ['position_tolerance_m', 'orientation_tolerance_rad'])
@pytest.mark.parametrize('value', [True, False, 0, -1., float('nan'), float('inf'), None, '0.001'])
def test_invalid_requested_wrist_precision_is_rejected_in_actions_and_tools(name, value):
    from agent.model_client.tools import parse_tool_call
    action = {'action': 'move_geometry', 'wrist_delta_m': [0., 0., -.0001], 'duration_s': 1., name: value}
    with pytest.raises(ActionValidationError):
        parse_geometry_action(action)
    with pytest.raises(ActionValidationError):
        parse_tool_call({'tool': 'move_geometry', 'arguments': {k: v for k, v in action.items() if k != 'action'}})


def test_precision_passthrough_preserves_original_defaults_and_wider_requests(public_geometry_inputs):
    old = build(public_geometry_inputs)
    assert old['geometry']['position_tolerance_m'] == .01
    assert old['geometry']['orientation_tolerance_rad'] == .1
    for position, rotation in ((.000025, .005), (1., 2.)):
        action = {'action': 'move_geometry', 'wrist_delta_m': [0., 0., -.0001], 'duration_s': 1.,
                  'position_tolerance_m': position, 'orientation_tolerance_rad': rotation}
        checked = build(public_geometry_inputs, action)
        assert checked['geometry']['position_tolerance_m'] == position
        assert checked['geometry']['orientation_tolerance_rad'] == rotation


@pytest.mark.parametrize('field', ['fingertip_tolerance_m', 'rotation_tolerance_rad'])
def test_nonexistent_precision_fields_stay_rejected(field):
    with pytest.raises(ActionValidationError):
        parse_geometry_action({'action': 'move_geometry', 'wrist_delta_m': [0., 0., .001],
                               'duration_s': 1., field: .001})


@pytest.mark.parametrize('kind', ['hold', 'stop'])
def test_wrist_precision_is_only_for_geometry_moves(kind):
    with pytest.raises(ActionValidationError):
        parse_geometry_action({'action': kind, 'duration_s': .4, 'position_tolerance_m': .0001})


@pytest.mark.parametrize('free', [False, True], ids=['fixed', 'free'])
@pytest.mark.parametrize('interface', ['fingertip_action', 'fingertip_tools', 'fingertip_code'])
def test_precision_survives_all_actual_dispatch_routes(monkeypatch, tmp_path, free, interface):
    from agent.model_client import runner
    from test_model_attempt_accounting import setup_route
    arguments = {'wrist_delta_m': [0., 0., -.0001], 'duration_s': 1.,
                 'position_tolerance_m': .000025, 'orientation_tolerance_rad': .005}
    if interface.endswith('action'):
        response = {'action': 'move_geometry', **arguments}
    elif interface.endswith('tools'):
        response = {'tool': 'move_geometry', 'arguments': arguments}
    else:
        response = 'move_geometry(wrist_delta_m=[0,0,-0.0001],duration_s=1,position_tolerance_m=0.000025,orientation_tolerance_rad=0.005)'
    args, bench, _ = setup_route(monkeypatch, tmp_path, interface, [response], free=free)
    args.max_actions = 1
    report = runner.run(args)
    assert report['error_type'] is None
    assert report['submitted_actions'] == 1
    assert len(bench.commands) == 1
    assert bench.commands[0]['geometry']['position_tolerance_m'] == .000025
    assert bench.commands[0]['geometry']['orientation_tolerance_rad'] == .005
