"""Practical new-layout gear goal probe; actual contact motion, zero task score."""
from copy import deepcopy
import math

import numpy as np

from scenes.workrooms.mechanical_assembly.native_evidence import angle, rotation
from scripts.validation.gear_scene_contract import OBJECTS, build_inputs
from scripts.validation.gear_transmission_checks import GEARS, axial_rotation, positive_contact, probe_contract


def build_goal_inputs(release=None, *, practical=False, tool_load='high_torque'):
    inputs = build_inputs(release)
    evaluation = inputs['task']['evaluation']
    high = probe_contract('high_torque')
    speed = float(evaluation['maximum_drive_angular_speed_rad_s'])
    phases = [
        {'id': 'settle', 'duration_s': 1., 'target_speed_rad_s': 0., 'loaded': False},
        {'id': 'goal_drive', 'duration_s': 8., 'target_speed_rad_s': speed, 'loaded': True},
        {'id': 'release', 'duration_s': 1., 'target_speed_rad_s': 0., 'loaded': False}]
    contract = deepcopy(inputs['probe_contract'])
    contract.update(phases=deepcopy(phases), maximum_torque_nm=high['maximum_torque_nm'],
        speed_load_coefficient_nm_s_per_rad=high['speed_load_coefficient_nm_s_per_rad'],
        engineering_variant='new_layout_goal_high_torque_v1', passive_wrench_api_calls=False,
        scope='Declared bounded high-torque preset on the independent socket only; passive gears and reset-only placement. Practical scene goal evidence, no robot task score.')
    if tool_load == 'lever_40n':
        if not practical:
            raise ValueError('The declared lever load requires practical observation')
        contract.update(maximum_torque_nm=.07*40., integral_load_coefficient_nm_per_rad=8.,
            wake_loaded_tool=True, engineering_variant='socket_70mm_lever_40n_v1',
            maximum_equivalent_handle_force_n=40., lever_radius_m=.07,
            scope='Bounded real-tool engineering load: 40 N at the existing 70 mm lever, at most 2.8 Nm. Explicit wake selects the loaded singleton socket only. Gears remain passive; no material or shared-controller changes.')
    elif tool_load != 'high_torque':
        raise ValueError('Unknown declared real-tool load')
    case = deepcopy(inputs['cases'][-1])
    case.update(id='target_goal', phases=deepcopy(phases), duration_s=10., maximum_duration_s=10.,
        early_drive_stop_allowed=True,
        scope='Preassembled engineering goal probe; actual three-gear transmission and release, no assembly/grasp/tool-selection credit.')
    inputs.update(schema='mechanical_gear_goal_probe.v1', probe_contract=contract, cases=[case],
        goal_contract={'target_rad': evaluation['nominal_output_turn_rad'],
            'minimum_output_turn_rad': evaluation['minimum_output_turn_rad'],
            'maximum_output_turn_rad': evaluation['maximum_output_turn_rad'],
            'cutoff_tolerance_rad': math.radians(3.),
            'baseline': 'Last actual native output pose in settle, immediately before goal_drive.',
            'maximum_drive_duration_s': 8., 'release_duration_s': 1.,
            'rest_window_s': .3, 'maximum_rest_angle_range_rad': math.radians(.5),
            'maximum_rest_position_range_m': .001,
            'gross_penetration_limit_m': inputs['task']['physical_geometry']['module_m'],
            'gross_penetration_basis': 'One full 5 mm gear module; the original 0.5 mm safety reference is reported separately.',
            'fine_geometry_drift_and_exact_backlash_are_acceptance_gates': False})
    if practical:
        inputs['observation_contract'] = {'state_period_s': .02,
            'contact_source': 'native_backend_pair_sums_window.v1',
            'native_contact_sampling_s': inputs['scene']['isaacsim']['physics_dt_s'],
            'tool_load_receipts_every_native_step': True,
            'scope': 'Actual control-end states and all native backend pair summaries; no raw contact point archive or full-substep object motion claim.'}
    return inputs


def assess_goal(rows, scene_info, dt, case, inputs):
    """Judge goal reach and practical physical behavior, including early stop."""
    result = {'passed': False, 'task_success_claimed': False, 'physics_accepted': False,
        'robot_manipulation_qualified': False, 'checks': {},
        'scope': 'Practical preassembled scene contact evidence only; strict historical geometry, fine drift and exact backlash are references.'}
    try:
        c, goal, evaluation = inputs['probe_contract'], inputs['goal_contract'], inputs['task']['evaluation']
        if case['id'] != 'target_goal' or case['active_gears'] != list(GEARS) or dt != evaluation['physics_dt_s']:
            raise ValueError('The declared current full-train goal case and native timestep are required')
        native_dt = dt
        dt = inputs.get('observation_contract', {}).get('state_period_s', dt)
        native_stride = round(dt/native_dt)
        if not rows:
            raise ValueError('Actual native evidence is required')
        phases = [r['wrench']['phase'] for r in rows]
        counts = {p: phases.count(p) for p in ('settle', 'goal_drive', 'release')}
        n_settle, n_drive, n_release = (counts[p] for p in ('settle', 'goal_drive', 'release'))
        complete = (phases == ['settle']*n_settle+['goal_drive']*n_drive+['release']*n_release
            and n_settle == round(1./dt) and 0 < n_drive <= round(goal['maximum_drive_duration_s']/dt)
            and n_release == round(goal['release_duration_s']/dt)
            and all(type(r['native_step']) is int and abs(r['time_s']-(i+1)*dt) < 1.e-8
                and (not i or r['native_step'] == rows[i-1]['native_step']+native_stride) for i, r in enumerate(rows)))
        if not complete:
            raise ValueError('Contiguous native settle, actual-length goal_drive and full release evidence is required')
        poses, velocities = {}, {}
        for name in OBJECTS:
            poses[name] = np.asarray([r['poses_xyzw'][name] for r in rows], float)
            velocities[name] = np.asarray([r['velocities'][name] for r in rows], float)
            if (poses[name].shape != (len(rows), 7) or velocities[name].shape != (len(rows), 6)
                    or not np.isfinite(poses[name]).all() or not np.isfinite(velocities[name]).all()
                    or np.max(np.abs(np.linalg.norm(poses[name][:, 3:], axis=1)-1.)) > 1.e-3):
                raise ValueError('Every original body needs a finite native pose and velocity stream')
        checks = result['checks']
        checks['complete_finite_native_stream'] = True
        checks['tool_only_bounded_load'] = all(
            r['wrench']['native_step'] == r['native_step'] and abs(r['wrench']['time_s']-r['time_s']) < 1.e-9
            and r['wrench']['body'] == 'socket_crank' and r['wrench']['world_frame'] is True
            and r['wrench']['force_n'] == [0., 0., 0.] and len(r['wrench']['torque_nm']) == 3
            and np.isfinite(r['wrench']['torque_nm']).all() and r['wrench']['torque_nm'][:2] == [0., 0.]
            and abs(r['wrench']['torque_nm'][2]) <= float(np.float32(c['maximum_torque_nm']))*(1.+2*np.finfo(np.float32).eps)
            and (phase == 'goal_drive' or r['wrench']['torque_nm'] == [0., 0., 0.])
            and (phase == 'goal_drive' or r['wrench'].get('wrench_api_called') is False)
            for r, phase in zip(rows, phases))
        checks['actual_tool_load_present'] = any(abs(r['wrench']['torque_nm'][2]) > 1.e-8 for r in rows[n_settle:n_settle+n_drive])
        checks['no_robot_interference'] = not any(r['robot_interference'] for r in rows)
        baseline = n_settle-1
        turns = {name: np.unwrap([axial_rotation(p) for p in poses[name]]) for name in (*GEARS, 'socket_crank')}
        relative = {name: values-values[baseline] for name, values in turns.items()}
        result['baseline'] = {'native_step': rows[baseline]['native_step'], 'time_s': rows[baseline]['time_s'],
            'gear_yaw_rad': {n: float(turns[n][baseline]) for n in GEARS}}
        final = {name: float(values[-1]) for name, values in relative.items()}
        checks['correct_gear_directions'] = final['input_gear'] > 0. and final['idler_gear'] < 0. and final['output_gear'] > 0.
        # Normal startup slack is excluded; there is no exact backlash gate.
        drive = np.arange(n_settle, n_settle+n_drive)
        selected = drive[relative['output_gear'][drive] >= c['startup_input_rotation_excluded_rad']]
        slopes = {}
        for name in GEARS[1:]:
            if len(selected) >= 2:
                x, y = relative['input_gear'][selected], relative[name][selected]
                x0 = x-x.mean()
                denominator = float(x0 @ x0)
                slopes[name] = float(x0 @ (y-y.mean())/denominator) if denominator > 1.e-14 else None
            else:
                slopes[name] = None
        expected = inputs['transmission_geometry_contract']['signed_angle_ratios']
        checks['reasonable_contact_transmission_ratios'] = all(slopes[n] is not None
            and abs(slopes[n]-expected[key]) <= c['ratio_slope_tolerance']
            for n, key in (('idler_gear', 'idler_per_input'), ('output_gear', 'output_per_input')))
        result['transmission'] = {'final_relative_turn_degrees': {n: math.degrees(v) for n, v in final.items()},
            'ratio_slopes_after_startup_slack': slopes, 'expected_ratios': expected,
            'ratio_reference_tolerance': c['ratio_slope_tolerance']}
        paths = {n: scene_info['objects'][n]['path'] for n in (*GEARS, 'socket_crank')}
        pairs = (('socket_crank', 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear'))
        moving = np.diff(turns['output_gear'], prepend=turns['output_gear'][0]) > 1.e-10
        contacts = {a+'__'+b: [positive_contact(r, paths[a], paths[b]) for r in rows[n_settle:n_settle+n_drive]] for a, b in pairs}
        checks['every_real_drive_interface_contacted'] = all(any(values) for values in contacts.values())
        chain = [all(contacts[key][i] for key in contacts) for i in range(n_drive)]
        if 'observation_contract' in inputs:
            chain = [r['simultaneous_drive_chain_samples'] > 0 for r in rows[n_settle:n_settle+n_drive]]
        checks['complete_chain_during_output_motion'] = any(hit and moving[n_settle+i] for i, hit in enumerate(chain))
        fixture = scene_info['fixtures']['gear_workplate']['path']
        supported = {n: [positive_contact(r, paths[n], fixture, upward=True, second_is_prefix=True)
            for r in rows[-round(goal['rest_window_s']/dt):]] for n in GEARS}
        checks['all_gears_supported_after_release'] = all(any(values) for values in supported.values())
        result['contact_observations'] = {'interface_positive_fraction': {key: float(np.mean(v)) for key, v in contacts.items()},
            'complete_chain_fraction_during_drive': float(np.mean(chain)),
            'final_support_positive_fraction': {key: float(np.mean(v)) for key, v in supported.items()},
            'scope': 'Real native normal contact evidence; summed compound impulses are not calibrated force or torque.'}
        checks['final_output_inside_task_sector'] = goal['minimum_output_turn_rad'] <= final['output_gear'] <= goal['maximum_output_turn_rad']
        result['goal'] = {'target_degrees': math.degrees(goal['target_rad']),
            'sector_degrees': [math.degrees(goal['minimum_output_turn_rad']), math.degrees(goal['maximum_output_turn_rad'])],
            'final_output_turn_degrees': math.degrees(final['output_gear']), 'actual_drive_duration_s': n_drive*dt,
            'maximum_output_turn_degrees': math.degrees(float(relative['output_gear'].max()))}
        rest = {}
        for name in (*GEARS, 'socket_crank'):
            tail = poses[name][-round(goal['rest_window_s']/dt)-1:]
            rotations = [rotation([float(p[6]), *p[3:6].tolist()]) for p in tail]
            position_range = float(np.linalg.norm(tail[:, :3]-tail[0, :3], axis=1).max())
            angle_range = max(angle(rotations[0], r) for r in rotations)
            rest[name] = {'position_range_m': position_range, 'angle_range_degrees': math.degrees(angle_range),
                'passed': position_range <= goal['maximum_rest_position_range_m'] and angle_range <= goal['maximum_rest_angle_range_rad']}
        result['released_rest_window'] = rest
        checks['can_stop_after_real_load_release'] = all(v['passed'] for v in rest.values())
        penetration_samples = np.asarray([r['maximum_penetration_m'] for r in rows], float)
        if not np.isfinite(penetration_samples).all() or np.any(penetration_samples < 0.):
            raise ValueError('Complete finite nonnegative native penetration evidence is required')
        penetration = float(penetration_samples.max())
        maxima = {n: {'linear_m_s': float(np.linalg.norm(velocities[n][:, :3], axis=1).max()),
            'angular_rad_s': float(np.linalg.norm(velocities[n][:, 3:], axis=1).max())} for n in OBJECTS}
        checks['no_gross_penetration_or_explosive_motion'] = (math.isfinite(penetration) and penetration >= 0.
            and penetration <= goal['gross_penetration_limit_m']
            and all(v['linear_m_s'] <= c['abort_body_linear_speed_m_s'] and v['angular_rad_s'] <= c['abort_body_angular_speed_rad_s'] for v in maxima.values()))
        result['safety_reference'] = {'maximum_penetration_m': penetration,
            'original_task_penetration_bound_met': penetration <= evaluation['maximum_reported_penetration_m'],
            'gross_penetration_limit_m': goal['gross_penetration_limit_m'], 'native_speed_maxima': maxima,
            'original_global_speed_bounds_met': all(v['linear_m_s'] <= evaluation['maximum_object_linear_speed_m_s']
                and v['angular_rad_s'] <= evaluation['maximum_object_angular_speed_rad_s'] for v in maxima.values()),
            'scope': 'Original safety references stay visible; fine motion, 0.1 mm drift, exact backlash and micrometre geometry are not practical goal gates.'}
        result['native_samples'] = len(rows)*native_stride
        result['state_observations'] = len(rows)
        result['observation_period_s'] = dt
        result['passed'] = all(checks.values())
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        result['error'] = str(exc)
    return result
