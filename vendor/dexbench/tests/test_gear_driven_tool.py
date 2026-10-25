"""The engineering motion boundary can address only the actual socket."""
from types import SimpleNamespace
import numpy as np
import pytest

from scripts.validation.validate_gear_driven_tool import assess, submit_socket_target, target_pose


def test_target_submission_never_writes_passive_gears_or_dynamic_states(monkeypatch):
    calls = []
    class Socket:
        count = 1
        def set_kinematic_targets(self, *_):
            pytest.fail('GPU tensor kinematic targets are not implemented')
        def set_transforms(self, data, indices):
            calls.append((data.copy(), indices))
        def set_velocities(self, *_):
            pytest.fail('No dynamic velocity writeback is permitted')
    class Gear:
        def __getattr__(self, _):
            pytest.fail('Passive gear views cannot receive a boundary command')
    env = SimpleNamespace(_object_views={'socket_crank': Socket(), 'input_gear': Gear(),
        'idler_gear': Gear(), 'output_gear': Gear()}, _tensor=lambda x: x, _one=[0],
        scene_info={'objects': {'socket_crank': {'path': '/World/Objects/socket_crank'}}})
    pose = target_pose([.4, -.13, .8202], np.deg2rad(15.)*.0005)
    submit_socket_target(env, pose)
    assert len(calls) == 1 and calls[0][1] == [0]
    assert np.array_equal(calls[0][0], pose)
    yaw = 2*np.arctan2(calls[0][0][0, 5], calls[0][0][0, 6])
    assert abs(yaw/.0005-np.deg2rad(15.)) < 1.e-6


def _completed_transmission(*, actual_lift=.08, final_tool_impulse=0.):
    """Actual goal/rest poses with the three interfaces touching at different steps."""
    gears = ('input_gear', 'idler_gear', 'output_gear')
    positions = {n: [.4, y, .8021] for n, y in zip(gears, (-.13, 0., .13))}
    positions['socket_crank'] = [.4, -.13, .8202]
    initial_angles = dict(input_gear=0., idler_gear=5.625, output_gear=0., socket_crank=0.)
    final_angles = dict(input_gear=30., idler_gear=5.625-18.75, output_gear=30., socket_crank=30.)
    def contact(a, b, impulse):
        return {'actor0': '/World/Objects/'+a, 'actor1': b,
            'impulse_magnitude_sum_ns': impulse, 'impulse_vector_ns': [0., 0., impulse],
            'min_separation_m': 0.}
    def row(phase, angles, lift=0., contacts=()):
        poses = {n: target_pose(p, np.deg2rad(angles[n]))[0].tolist() for n, p in positions.items()}
        poses['socket_crank'][2] += lift
        return {'phase': phase, 'poses_xyzw': poses, 'native_contacts': list(contacts),
            'simultaneous_drive_chain_samples': 0}
    rows = [row('settle', initial_angles)]
    for a, b in (('socket_crank', 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear')):
        rows.append(row('drive', final_angles, contacts=[contact(a, '/World/Objects/'+b, .001)]))
    support = [contact(n, '/World/Fixtures/gear_workplate/collision_001', .001) for n in gears]
    tool_report = contact('socket_crank', '/World/Objects/input_gear', final_tool_impulse)
    rows.extend(row('free_gear_hold', final_angles, actual_lift, support+[tool_report]) for _ in range(16))
    inputs = {'operator_boundary_contract': {'minimum_output_turn_rad': np.deg2rad(25.),
        'maximum_output_turn_rad': np.deg2rad(35.), 'minimum_actual_withdraw_height_m': .06},
        'task': {'physical_geometry': {'module_m': .005}},
        'cases': [{'reset_objects': {n: {'position_m': positions[n]} for n in positions}}]}
    flags = {n: {'rigid_body_enabled': True, 'kinematic_enabled': n == 'socket_crank'} for n in positions}
    return assess(rows, inputs, flags)


def test_real_goal_with_intermittent_meshes_and_zero_load_contact_reports_passes():
    result = _completed_transmission()
    assert result['passed'], result
    assert result['contact_diagnostics']['actual_simultaneous_contact_chain'] is False
    assert result['actual_tool_withdrawal']['tail_zero_load_contact_reports'] == 16
    assert result['actual_tool_withdrawal']['minimum_tail_height_increase_m'] == pytest.approx(.08)


@pytest.mark.parametrize('lift,impulse', [(.04, 0.), (.08, .001)])
def test_goal_still_fails_when_tool_has_not_withdrawn_or_still_loads_gears(lift, impulse):
    result = _completed_transmission(actual_lift=lift, final_tool_impulse=impulse)
    assert result['checks']['output_reaches_original_task_sector']
    assert result['checks']['tool_withdrawn_without_final_touch'] is False
    assert result['passed'] is False
