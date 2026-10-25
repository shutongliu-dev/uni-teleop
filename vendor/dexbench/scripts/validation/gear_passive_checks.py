"""Pure rest-window evidence using the frozen L5 canonical thresholds."""
import math

import numpy as np

from scenes.workrooms.mechanical_assembly.native_evidence import EPS, angle, rotation, within
from scenes.workrooms.mechanical_assembly.tasks.gear_train_assembly.rules import GEARS, TOOLS, thresholds
from scripts.validation.gear_transmission_checks import (contact_totals, positive_contact,
    _block_fractions, transmission_geometry_contract)


def canonical_passive_window(rows, scene_info, *, end_time_s, phase, dt, geometry=None):
    """Check 300 ms plus its baseline; no target-angle or robot-task credit.

    Native speed/seat/anchor bounds cover every recorded substep. Observed
    velocity uses the canonical 20 ms grid. Support and release use every
    contact substep, including zero-pressure touching manifolds.
    """
    c = thresholds()
    result = {'passed': False, 'task_success_claimed': False, 'phase': phase,
        'thresholds': c, 'objects': {}, 'checks': {},
        'scope': 'Canonical passive-window prerequisites only; external loaded motion and preassembled reset earn no task credit.'}
    try:
        task_geometry = transmission_geometry_contract(scene_info, geometry)
        centers = np.asarray(task_geometry['shaft_centers_world_xy_m'], dtype=float)
        if phase not in ('settle', 'final_hold') or dt != c['physics_dt_s'] or not math.isfinite(end_time_s):
            raise ValueError('Frozen L5 phase and physics timestep required')
        duration = c['assembly_settle_s'] if phase == 'settle' else c['release_hold_s']
        start = end_time_s-duration
        sample_count = round(duration/dt)
        stride = round(c['control_period_s']/dt)
        selected = [r for r in rows if start-EPS <= r['time_s'] <= end_time_s+EPS]
        if (len(selected) != sample_count+1 or abs(selected[0]['time_s']-start) > EPS
                or abs(selected[-1]['time_s']-end_time_s) > EPS
                or any(b['native_step']-a['native_step'] != 1 or abs(b['time_s']-a['time_s']-dt) > EPS
                       for a, b in zip(selected, selected[1:]))):
            raise ValueError('Complete passive window and preceding pose baseline required')
        if sample_count % stride:
            raise ValueError('Passive duration does not contain complete control windows')
        paths = {n: scene_info['objects'][n]['path'] for n in GEARS+TOOLS}
        fixture = scene_info['fixtures']['gear_workplate']['path']
        checks = result['checks']
        checks['complete_native_and_observed_sampling'] = True
        checks['zero_external_wrench'] = all(
            r['wrench']['native_step'] == r['native_step'] and r['wrench']['body'] == 'socket_crank'
            and r['wrench']['phase'] == phase and r['wrench']['world_frame'] is True
            and r['wrench']['force_n'] == [0., 0., 0.] and r['wrench']['torque_nm'] == [0., 0., 0.]
            for r in selected)
        checks['penetration_bound'] = all(math.isfinite(r['maximum_penetration_m'])
            and 0. <= r['maximum_penetration_m'] <= c['maximum_reported_penetration_m'] for r in selected)
        released_bodies = {paths[n] for n in (GEARS if phase == 'settle' else GEARS+TOOLS)}
        checks['released_without_robot_manifolds'] = not any(
            (a in released_bodies and within(b, '/World/Robot'))
            or (b in released_bodies and within(a, '/World/Robot'))
            for r in selected[1:] for a, b, *_ in contact_totals(r))
        for name in GEARS+TOOLS:
            poses = np.asarray([r['poses_xyzw'][name] for r in selected], dtype=float)
            velocities = np.asarray([r['velocities'][name] for r in selected], dtype=float)
            if (poses.shape != (sample_count+1, 7) or velocities.shape != (sample_count+1, 6)
                    or not np.isfinite(poses).all() or not np.isfinite(velocities).all()):
                raise ValueError('Complete finite six-body poses and velocities required')
            rotations = [rotation([float(p[6]), *p[3:6].tolist()]) for p in poses]
            observed = np.arange(0, sample_count+1, stride)
            linear = float(np.linalg.norm(velocities[:, :3], axis=1).max())
            angular = float(np.linalg.norm(velocities[:, 3:], axis=1).max())
            observed_linear = float(np.linalg.norm(np.diff(poses[observed, :3], axis=0), axis=1).max()/c['control_period_s'])
            observed_angular = max(angle(rotations[a], rotations[b])/c['control_period_s']
                for a, b in zip(observed, observed[1:]))
            body_checks = {'global_motion_bound': max(linear, observed_linear) <= c['maximum_object_linear_speed_m_s']+EPS
                and max(angular, observed_angular) <= c['maximum_object_angular_speed_rad_s']+EPS}
            metrics = {'maximum_native_linear_speed_m_s': linear, 'maximum_native_angular_speed_rad_s': angular,
                'maximum_20ms_linear_speed_m_s': observed_linear, 'maximum_20ms_angular_speed_rad_s': observed_angular}
            if name in GEARS:
                position_range = float(np.linalg.norm(poses[:, :3]-poses[0, :3], axis=1).max())
                angle_range = max(angle(rotations[0], r) for r in rotations)
                target = [*centers[GEARS.index(name)], .802]
                axis_error = float(np.linalg.norm(poses[:, :2]-target[:2], axis=1).max())
                height_error = float(np.abs(poses[:, 2]-target[2]).max())
                tilt = max(math.acos(float(np.clip(r[2, 2], -1., 1.))) for r in rotations)
                support = [positive_contact(r, paths[name], fixture, upward=True, second_is_prefix=True) for r in selected[1:]]
                body_checks.update(native_linear_rest=linear <= c['maximum_stable_linear_speed_m_s']+EPS,
                    native_angular_rest=angular <= c['maximum_stable_angular_speed_rad_s']+EPS,
                    observed_linear_rest=observed_linear <= c['maximum_stable_linear_speed_m_s']+EPS,
                    observed_angular_rest=observed_angular <= c['maximum_stable_observed_angular_speed_rad_s']+EPS,
                    anchor_position_range=position_range <= c['maximum_stable_position_range_m'],
                    anchor_angle_range=angle_range <= c['maximum_stable_angle_range_rad'],
                    seated=(axis_error <= c['maximum_axis_offset_m'] and height_error <= c['maximum_bottom_height_error_m']
                        and tilt <= c['maximum_axis_tilt_rad']),
                    upward_support_each_20ms=all(f > 0. for f in _block_fractions(support, stride)))
                metrics.update(anchor_position_range_m=position_range, anchor_angle_range_rad=angle_range,
                    maximum_axis_offset_m=axis_error, maximum_bottom_error_m=height_error, maximum_tilt_rad=tilt)
            result['objects'][name] = {'passed': bool(all(body_checks.values())), 'checks': body_checks, **metrics}
        checks['all_canonical_body_conditions'] = all(v['passed'] for v in result['objects'].values())
        result.update(start_time_s=start, end_time_s=end_time_s, native_intervals=sample_count,
            observed_intervals=sample_count//stride, passed=bool(all(checks.values())))
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        result['error'] = str(exc)
    return result
