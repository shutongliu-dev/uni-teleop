"""Small positive/negative examples for practical goal evidence."""
from copy import deepcopy
import math

from scripts.validation.gear_goal_contract import build_goal_inputs, assess_goal
from scripts.validation.gear_scene_contract import OBJECTS
from scripts.validation.gear_transmission_checks import GEARS, compact_native_row


def evidence():
    inputs = build_goal_inputs()
    inputs['task']['evaluation']['physics_dt_s'] = dt = .01
    case = inputs['cases'][0]
    paths = {n: '/World/Objects/'+n for n in OBJECTS}
    fixture = '/World/Fixtures/gear_workplate'
    info = {'objects': {n: {'path': p} for n, p in paths.items()}, 'fixtures': {'gear_workplate': {'path': fixture}}}
    rows = []
    for phase, duration in (('settle', 1.), ('goal_drive', 3.), ('release', 1.)):
        for i in range(round(duration/dt)):
            theta = 0. if phase == 'settle' else math.radians(30.) if phase == 'release' else math.radians(10.)*(i+1)*dt
            poses = {}
            for name, spec in case['reset_objects'].items():
                q, p = spec['quaternion_wxyz'], spec['position_m']
                if name in GEARS or name == 'socket_crank':
                    yaw = math.pi/32-.625*theta if name == 'idler_gear' else theta
                    q = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
                poses[name] = [*p, *q[1:], q[0]]
            pairs = [('socket_crank', 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear')]
            contacts = [{'actor0': paths[a], 'actor1': paths[b], 'normal_impulses_ns': [.001], 'normals': [[1., 0., 0.]]} for a, b in pairs]
            contacts += [{'actor0': paths[n], 'actor1': fixture+'/thrust', 'normal_impulses_ns': [.001], 'normals': [[0., 0., 1.]]} for n in GEARS]
            step = len(rows)+1
            rows.append({'native_step': step, 'time_s': step*dt, 'poses_xyzw': poses,
                'velocities': {n: [0.]*6 for n in OBJECTS}, 'maximum_penetration_m': 0.,
                'raw_contacts': contacts, 'robot_interference': [],
                'wrench': {'native_step': step, 'time_s': step*dt, 'phase': phase,
                    'body': 'socket_crank', 'world_frame': True, 'force_n': [0., 0., 0.],
                    'torque_nm': [0., 0., .5 if phase == 'goal_drive' else 0.],
                    'wrench_api_called': phase == 'goal_drive'}})
    return rows, info, dt, case, inputs


def test_early_goal_reach_and_release_pass_with_physical_slack_and_small_jitter():
    rows, info, dt, case, inputs = evidence()
    assert inputs['probe_contract']['maximum_torque_nm'] == 1.6
    assert inputs['probe_contract']['speed_load_coefficient_nm_s_per_rad'] == 5.
    assert inputs['probe_contract']['phases'][1]['target_speed_rad_s'] == math.radians(15.)
    # 0.3 mm terminal motion exceeds the historical 0.1 mm drift gate, yet is
    # clearly within the user-approved 1 mm practical released-rest envelope.
    rows[-5]['poses_xyzw']['socket_crank'][0] += .0003
    rows[-5]['velocities']['socket_crank'][0] = .0021
    # Real native float32 saturation from native05 is 1.600000023841858 Nm.
    next(r for r in rows if r['wrench']['phase'] == 'goal_drive')['wrench']['torque_nm'][2] = 1.600000023841858
    result = assess_goal(rows, info, dt, case, inputs)
    assert result['passed'], result
    assert result['goal']['actual_drive_duration_s'] == 3.
    assert abs(result['goal']['final_output_turn_degrees']-30.) < 1.e-9
    assert not result['task_success_claimed']
    assert assess_goal([compact_native_row(r) for r in rows], info, dt, case, inputs) == result


def test_turning_input_without_output_or_without_real_contact_does_not_pass():
    rows, info, dt, case, inputs = evidence()
    stalled = deepcopy(rows)
    for row in stalled:
        row['poses_xyzw']['output_gear'][3:] = [0., 0., 0., 1.]
    result = assess_goal(stalled, info, dt, case, inputs)
    assert not result['passed'] and not result['checks']['final_output_inside_task_sector']
    for row in rows:
        row['raw_contacts'] = [c for c in row['raw_contacts'] if c['actor0'] != info['objects']['idler_gear']['path']]
    result = assess_goal(rows, info, dt, case, inputs)
    assert not result['passed'] and not result['checks']['every_real_drive_interface_contacted']


def test_control_observations_require_actual_simultaneous_native_chain():
    rows, info, dt, case, inputs = evidence()
    inputs['observation_contract'] = {'state_period_s': .02}
    rows = [compact_native_row(r) for r in rows[1::2]]
    for r in rows:
        r['contact_totals_source'] = 'native_backend_pair_sums_window.v1'
        r['simultaneous_drive_chain_samples'] = 2
    result = assess_goal(rows, info, dt, case, inputs)
    assert result['passed'] and result['native_samples'] == 500
    assert result['goal']['actual_drive_duration_s'] == 3.
    for r in rows:
        r['simultaneous_drive_chain_samples'] = 0
    result = assess_goal(rows, info, dt, case, inputs)
    assert not result['passed'] and not result['checks']['complete_chain_during_output_motion']
