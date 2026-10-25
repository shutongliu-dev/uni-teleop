"""CPU geometry checks for the two real L5 tool insertion boundaries."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation.validate_gear_driven_tool import submit_socket_target, target_pose
from scripts.validation.validate_gear_tool_function import build_tool_inputs, target_for_phase


@pytest.mark.parametrize('case_id,body,initial,engaged,insert_s,withdraw_m', [
    ('socket_insert', 'socket_crank', [.4, -.13, .9002], [.4, -.13, .8202], 1., .08),
    ('wrench_insert', 'open_end_wrench', [.352, -.13, .824], [.392, -.13, .824], .5, .05),
])
def test_real_tool_trajectory_inserts_rotates_about_input_and_withdraws_with_clamped_speed(
        case_id, body, initial, engaged, insert_s, withdraw_m):
    inputs = build_tool_inputs(case_id)
    contract = inputs['tool_function_contract']
    phases = {phase['id']: phase['duration_s'] for phase in contract['phases']}
    assert len(inputs['cases']) == 1 and inputs['cases'][0]['id'] == case_id
    assert contract['body'] == body
    np.testing.assert_allclose(contract['initial_position_m'], initial, atol=1.e-8)
    np.testing.assert_allclose(contract['engaged_position_m'], engaged, atol=1.e-8)
    np.testing.assert_allclose(contract['rotation_center_m'][:2], [.4, -.13], atol=1.e-8)
    np.testing.assert_allclose(inputs['cases'][0]['reset_objects'][body]['position_m'], initial,
                               atol=1.e-8)
    assert phases['insert'] == pytest.approx(insert_s)
    assert contract['drive_turn_rad'] == pytest.approx(np.deg2rad(20.))
    assert contract['angular_speed_rad_s'] == pytest.approx(np.deg2rad(15.))

    def pose(phase, time_s):
        result = np.asarray(target_for_phase(contract, phase, time_s))
        assert result.shape == (1, 7) and np.isfinite(result).all()
        assert np.linalg.norm(result[0, 3:]) == pytest.approx(1., abs=1.e-7)
        return result[0]

    start = pose('insert', 0.)
    inserted = pose('insert', phases['insert'])
    turned = pose('drive', phases['drive'])
    withdrawn = pose('withdraw', phases['withdraw'])
    np.testing.assert_allclose(start[:3], initial, atol=1.e-7)
    np.testing.assert_allclose(inserted[:3], engaged, atol=1.e-7)
    theta = np.deg2rad(20.)
    np.testing.assert_allclose(turned[3:], [0., 0., np.sin(theta/2), np.cos(theta/2)],
                               atol=1.e-7)
    if body == 'open_end_wrench':
        expected_turned = [.4-.008*np.cos(theta), -.13-.008*np.sin(theta), .824]
        expected_withdraw_direction = [-np.cos(theta), -np.sin(theta), 0.]
    else:
        expected_turned = engaged
        expected_withdraw_direction = [0., 0., 1.]
    np.testing.assert_allclose(turned[:3], expected_turned, atol=1.e-7)
    np.testing.assert_allclose(contract['withdraw_direction_world'], expected_withdraw_direction,
                               atol=1.e-7)
    np.testing.assert_allclose(withdrawn[:3], np.asarray(expected_turned)
        + withdraw_m*np.asarray(expected_withdraw_direction), atol=1.e-7)
    assert withdraw_m >= contract['minimum_withdraw_distance_m']
    for phase in ('insert', 'drive', 'withdraw'):
        np.testing.assert_array_equal(pose(phase, phases[phase]+1.), pose(phase, phases[phase]))
        times = np.linspace(0., phases[phase], 41)
        poses = np.asarray([pose(phase, time_s) for time_s in times])
        speeds = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)/np.diff(times)
        assert max(speeds) <= .08001
        yaws = 2*np.arctan2(poses[:, 5], poses[:, 6])
        assert max(abs(np.diff(yaws)/np.diff(times))) <= np.deg2rad(15.)+1.e-5


def test_generalized_boundary_only_writes_the_selected_wrench():
    calls = []

    class Wrench:
        count = 1

        def set_transforms(self, data, indices):
            calls.append((data.copy(), indices))

        def set_velocities(self, *_):
            pytest.fail('The boundary must not prescribe dynamic velocity')

        def set_kinematic_targets(self, *_):
            pytest.fail('The GPU backend does not implement kinematic targets')

    class PassiveBody:
        def __getattr__(self, _):
            pytest.fail('Only the selected tool may receive a boundary command')

    env = SimpleNamespace(_object_views={
        'open_end_wrench': Wrench(), 'socket_crank': PassiveBody(),
        'input_gear': PassiveBody(), 'idler_gear': PassiveBody(), 'output_gear': PassiveBody(),
    }, _tensor=lambda value: value, _one=[0], scene_info={'objects': {
        'open_end_wrench': {'path': '/World/Objects/open_end_wrench'},
    }})
    pose = target_pose([.392, -.13, .824], np.deg2rad(20.))
    submit_socket_target(env, pose, tool='open_end_wrench')
    assert len(calls) == 1 and calls[0][1] == [0]
    np.testing.assert_array_equal(calls[0][0], pose)
