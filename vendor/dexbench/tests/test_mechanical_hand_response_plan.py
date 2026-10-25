"""Real-asset CPU geometry and reference checks, without native acceptance."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bench.geometry import matrix_quaternion, quaternion_matrix, rpy_matrix
from bench.paths import load_json
from scenes.workrooms.mechanical_assembly.robot_layout import geometry_model
from scenes.workrooms.mechanical_assembly.tasks.fastener_assembly.definition import TASK
from scripts.validation.mechanical_hand_response_plan import build_hand_response_plan


@pytest.fixture(scope='module')
def configured():
    # This diagnostic starts around the historical preinserted handle.
    release = Path(TASK['design']['historical_preinserted']['release']['path'])
    scene = load_json(release/'scene.json')
    initial = load_json(release/'initial_state.json')
    before = deepcopy((scene, initial))
    plan = build_hand_response_plan(scene, initial)
    assert (scene, initial) == before
    return scene, initial, geometry_model(scene['robot']), plan


def test_preclose_reset_locates_real_handle_and_preserves_original_inputs(configured):
    scene, initial, model, plan = configured
    tool = next(row for row in scene['objects'] if row['id']=='screwdriver')
    tool_rotation = quaternion_matrix(tool['quaternion_wxyz'])
    wrist = model.public_geometry(plan['initial_state']['joint_positions_rad'])['wrist']
    wrist_rotation = quaternion_matrix(wrist['quaternion_wxyz'])
    x, z = plan['grasp_candidate']['axis_in_hand_xz_m']
    predicted_center = np.asarray(wrist['position_m'])+wrist_rotation @ np.array([x, 0., z])
    actual_center = np.asarray(tool['position_m'])+tool_rotation @ np.array([0., 0., .156])
    assert np.linalg.norm(predicted_center-actual_center)<.00006
    assert np.linalg.norm(wrist_rotation[:, 1]-tool_rotation[:, 2])<.001
    assert plan['initial_state'] != initial
    assert plan['original_initial_state']==initial
    assert not plan['grasp_gate_blocks_motion']
    assert not plan['object_external_wrenches'] and not plan['object_state_writes_after_reset']
    assert not plan['object_attachments'] and not plan['public_functions_used']
    assert not plan['native_qualified']
    json.dumps(plan, allow_nan=False)


def test_actual_fk_performs_small_axis_orbit_without_axial_command_and_reverses(configured):
    scene, _, model, plan = configured
    phases = {row['name']: row for row in plan['phases']}
    fixed = model.public_geometry(phases['hold_grasp']['joint_positions_rad'])['wrist']
    origin = np.array(plan['tool_pose_xyzw'][:3]); axis = np.array(plan['tool_axis_world'])
    start = np.array(fixed['position_m'])-origin
    projected = start-np.dot(start, axis)*axis
    original_hand = {n: phases['hold_grasp']['joint_positions_rad'][n] for n in model.hand.names}
    for name, degrees in [('turn_03', -3.), ('turn_06', -6.), ('return_03', -3.), ('return_06', 0.)]:
        q = phases[name]['joint_positions_rad']
        wrist = model.public_geometry(q)['wrist']
        delta = np.array(wrist['position_m'])-origin
        radial = delta-np.dot(delta, axis)*axis
        signed = math.atan2(np.dot(axis, np.cross(projected, radial)), np.dot(projected, radial))
        assert math.degrees(signed)==pytest.approx(degrees, abs=.03)
        assert np.dot(delta-start, axis)==pytest.approx(0., abs=.00006)
        assert np.linalg.norm(radial)==pytest.approx(np.linalg.norm(projected), abs=.00006)
        assert {n:q[n] for n in model.hand.names}==original_hand
        assert len(model.checked_joints(q, enforce_limits=True))==29
    assert phases['return_06']['joint_positions_rad']==phases['close']['joint_positions_rad']
    assert phases['final_hold']['joint_positions_rad']==phases['open']['joint_positions_rad']
    assert plan['commanded_axial_advance_m']==0.


def test_finite_difference_of_command_path_respects_original_joint_speeds(configured):
    _, _, model, plan = configured
    previous = plan['initial_state']['joint_positions_rad']
    for phase in plan['phases']:
        duration = phase['duration_s']; target = phase['joint_positions_rad']
        assert duration/.02==pytest.approx(round(duration/.02), abs=1.e-10)
        assert set(target)==set(model.names)
        times = np.linspace(0., duration, 1001)
        u = times/duration
        weights = 10*u**3-15*u**4+6*u**5
        for name in model.names:
            path = previous[name]+weights*(target[name]-previous[name])
            assert max(abs(np.diff(path)/np.diff(times)))<=plan['joint_limits'][name]['reference_speed_limit_rad_s']+1.e-8
        previous=target
    phases={row['name']:row for row in plan['phases']}
    assert .5<=phases['close']['duration_s']<=1.
    assert all(phases[name]['duration_s']==pytest.approx(.3) for name in
               ['settle','hold_grasp','hold_after_turn','final_hold'])
    assert plan['total_duration_s']<10.


def test_tilted_tool_reference_keeps_its_own_axis_height_and_source_pose(configured):
    scene, initial, model, original_plan = configured
    before = deepcopy(scene)
    pose = original_plan['tool_pose_xyzw']
    rotation = rpy_matrix([math.radians(8.), 0., 0.]) @ quaternion_matrix(np.asarray(pose)[[6, 3, 4, 5]])
    q = matrix_quaternion(rotation)
    plan = build_hand_response_plan(scene, initial, tool_pose_xyzw=[*pose[:3], *q[[1, 2, 3, 0]]])
    axis = rotation[:, 2]
    phases = {row['name']: row for row in plan['phases']}
    start = np.asarray(model.public_geometry(phases['close']['joint_positions_rad'])['wrist']['position_m'])
    end = np.asarray(model.public_geometry(phases['turn_06']['joint_positions_rad'])['wrist']['position_m'])
    origin = np.asarray(pose[:3])
    start_radial = start-origin-np.dot(start-origin, axis)*axis
    end_radial = end-origin-np.dot(end-origin, axis)*axis
    angle = math.atan2(np.dot(axis, np.cross(start_radial, end_radial)), np.dot(start_radial, end_radial))
    assert math.degrees(angle)==pytest.approx(-6., abs=.03)
    assert np.dot(end-start, axis)==pytest.approx(0., abs=.00006)
    assert abs(end[2]-start[2])>.0001  # World Z need not stay fixed for a tilted tool.
    assert scene==before
    assert plan['axis_source']=='caller_supplied_tool_pose_requires_external_readback_provenance'


@pytest.mark.parametrize('degrees',[0., -6., 6.1, float('nan'), True])
def test_invalid_or_excessive_turn_is_rejected(configured, degrees):
    scene, initial, _, _ = configured
    with pytest.raises(ValueError, match='six degrees'):
        build_hand_response_plan(scene, initial, turn_degrees=degrees)


def test_missing_reset_joint_and_bad_tool_reference_fail_before_native(configured):
    scene, initial, _, _ = configured
    damaged=deepcopy(initial);damaged['joint_positions_rad'].pop('fr3_joint1')
    with pytest.raises(ValueError, match='exactly'):
        build_hand_response_plan(scene, damaged)
    with pytest.raises(ValueError, match='Tool planning reference'):
        build_hand_response_plan(scene, initial, tool_pose_xyzw=[0., 0., .9, 0., 0., 0., 2.])
