"""Independent end-of-record gate using a release's unchanged frozen scorer.

Legacy scoring may latch an earlier success. This check requires the actual
last complete stability interval to remain released, quiet and at the goal.
Collision validity and operation history remain the full-stream audit's job.
"""
import math

import numpy as np

from .rules import GARMENT_PROFILES, GARMENT_TASK_IDS, PRACTICAL_PROFILE, GARMENT_FOLDING_TASK_IDS, PRACTICAL_PROFILES, STACKING_PRACTICAL_PROFILE


def terminal_window(arrays, evaluator):
    limits = evaluator.limits
    dt = evaluator.c['physics_dt_s']
    count = int(math.ceil(limits['stable_s']/dt))+1
    total = len(arrays['time_s'])
    practical_stacking = evaluator.c.get('rule_profile') == STACKING_PRACTICAL_PROFILE
    lag = evaluator.stability_lag_steps if practical_stacking else 0
    history_count = count+lag
    result = {'schema': 'laundry_terminal_window.v1', 'required': True,
        'required_duration_s': limits['stable_s'], 'required_samples': count,
        'samples_checked': min(count, total), 'complete': False, 'passed': False,
        'complete_release': False, 'stable': False, 'geometry_passed': False,
        'scope': 'Actual terminal window with frozen geometry/quietness thresholds; full-stream physics and operation history are separate.'}
    if practical_stacking:
        result.update(schema='laundry_terminal_window.v2', required_history_samples=history_count,
            stability_position_interval_s=limits['stability_position_interval_s'])
    if total < history_count:
        return {**result, 'reason_codes': ['terminal_window_incomplete']}
    history_steps = np.asarray(arrays['native_step'][-history_count:])
    history_times = np.asarray(arrays['time_s'][-history_count:])
    if practical_stacking and (history_steps.shape != (history_count,) or history_steps.dtype.kind not in 'iu'
            or history_times.shape != (history_count,) or not np.isfinite(history_times).all()
            or not np.array_equal(np.diff(history_steps), np.ones(history_count-1, dtype=int))
            or not np.allclose(history_times, history_steps*dt, atol=1.e-8, rtol=0)):
        raise ValueError('Incomplete native history for 20 ms stability positions')
    steps = np.asarray(arrays['native_step'][-count:])
    times = np.asarray(arrays['time_s'][-count:])
    if (steps.shape != (count,) or steps.dtype.kind not in 'iu'
            or times.shape != (count,) or not np.isfinite(times).all()
            or not np.array_equal(np.diff(steps), np.ones(count-1, dtype=int))
            or not np.allclose(times, steps*dt, atol=1.e-8, rtol=0)):
        raise ValueError('Incomplete or inconsistent terminal native time window')
    cloths = {}
    max_speed = pose_speed = drift = raw_p95 = pose_p95 = 0.
    practical = evaluator.c.get('rule_profile') in PRACTICAL_PROFILES
    controlled_samples = 0
    interval_p95 = 0.
    for name, spec in evaluator.specs.items():
        q = np.asarray(arrays[name+'_q'][-count:], dtype=float)
        qd = np.asarray(arrays[name+'_qd'][-count:], dtype=float)
        mask = np.asarray(arrays[name+'_control_mask'][-count:])
        if (q.shape != qd.shape or q.shape != (count, len(spec['rest_vertices_m']), 3)
                or not np.isfinite(q).all() or not np.isfinite(qd).all()
                or mask.shape != q.shape[:2] or mask.dtype.kind != 'b'):
            raise ValueError('Incomplete or nonfinite terminal cloth arrays')
        if not np.array_equal(np.asarray(arrays[name+'_triangles']), evaluator.meshes[name]['triangles']):
            raise ValueError('Terminal native cloth topology changed')
        if practical_stacking:
            history_q = np.asarray(arrays[name+'_q'][-history_count:], dtype=float)
            if history_q.shape != (history_count, len(spec['rest_vertices_m']), 3) or not np.isfinite(history_q).all():
                raise ValueError('Incomplete positions for 20 ms stability history')
            interval_p95 = max(interval_p95, float(np.percentile(np.linalg.norm(
                history_q[lag:]-history_q[:-lag], axis=-1)/limits['stability_position_interval_s'], 95, axis=1).max()))
        cloths[name] = q
        max_speed = max(max_speed, float(np.linalg.norm(qd, axis=-1).max()))
        pose_speed = max(pose_speed, float(np.linalg.norm(np.diff(q, axis=0), axis=-1).max()/dt))
        if practical:
            raw_p95 = max(raw_p95, float(np.percentile(np.linalg.norm(qd, axis=-1), 95, axis=1).max()))
            pose_p95 = max(pose_p95, float(np.percentile(np.linalg.norm(np.diff(q, axis=0), axis=-1)/dt, 95, axis=1).max()))
        drift = max(drift, float(np.linalg.norm(q-q[0], axis=-1).max()))
        controlled_samples += int(mask.any(axis=1).sum())
    first_bad = None
    target = np.asarray(evaluator.meta['target_center_xy_m'])
    for index in range(count):
        points = {name: values[index] for name, values in cloths.items()}
        if evaluator.task_id == 'towel_flattening':
            good, geometry = evaluator._flat('towel', points['towel'], evaluator._mesh_metrics('towel', points['towel']))
        elif evaluator.task_id == 'towel_folding':
            good, geometry = evaluator._fold('towel', points['towel'], target, table_supported=True)
        elif evaluator.task_id == 'towel_stacking':
            good, geometry = evaluator._stack(points)
        elif evaluator.task_id in GARMENT_FOLDING_TASK_IDS:
            good, geometry = evaluator._garment_fold(points)
        elif evaluator.task_id in GARMENT_TASK_IDS:
            good, geometry = evaluator._garment_stack(points)
        else:
            raise ValueError('Unknown frozen laundry task for terminal geometry')
        if not good and first_bad is None:
            first_bad = {'native_step': int(steps[index]), 'time_s': float(times[index]), 'geometry': geometry}
    reasons = []
    if controlled_samples:
        reasons.append('terminal_control_retained')
    raw_gate, pose_gate = (raw_p95, pose_p95) if practical else (max_speed, pose_speed)
    if practical_stacking:
        raw_gate = pose_gate = interval_p95
    if not practical_stacking and raw_gate > limits['max_node_speed_m_s']:
        reasons.append('terminal_raw_speed_exceeded')
    if pose_gate > limits['max_node_speed_m_s']:
        reasons.append('terminal_control_period_pose_speed_exceeded' if practical_stacking else 'terminal_pose_speed_exceeded')
    if drift > limits['stable_node_drift_m']:
        reasons.append('terminal_drift_exceeded')
    if first_bad is not None:
        reasons.append('terminal_goal_geometry_failed')
    if times[-1] >= limits['max_time_s']:
        reasons.append('terminal_budget_exhausted')
    if practical:
        result.update(rule_profile=evaluator.c['rule_profile'], speed_node_percentile=95,
            raw_speed_p95_m_s=raw_p95, pose_speed_p95_m_s=pose_p95,
            scope='Practical frozen geometry, full release, per-cloth/per-frame P95 speeds and all-node drift; full-stream evidence is separate.')
    if practical_stacking:
        result.update(control_period_pose_speed_p95_m_s=interval_p95, stability_history_complete=True,
            scope='Task3 practical v3: per-cloth 20 ms actual-position P95 speed, all-node 8 mm drift, rest-area P99/absolute height guards; raw per-dt speeds retained. Full-stream collision, robot separation and operation evidence remain separate.')
    return {**result, 'complete': True, 'passed': not reasons,
        'start_native_step': int(steps[0]), 'end_native_step': int(steps[-1]),
        'start_time_s': float(times[0]), 'end_time_s': float(times[-1]),
        'duration_s': float(times[-1]-times[0]), 'complete_release': controlled_samples == 0,
        'within_budget': bool(times[-1] < limits['max_time_s']),
        'controlled_cloth_samples': controlled_samples,
        'max_node_speed_m_s': max_speed, 'pose_speed_m_s': pose_speed, 'max_node_drift_m': drift,
        'stable': max(raw_gate, pose_gate) <= limits['max_node_speed_m_s'] and drift <= limits['stable_node_drift_m'],
        'geometry_passed': first_bad is None, 'first_geometry_failure': first_bad,
        'terminal_geometry': geometry, 'reason_codes': reasons}
