"""Declared-mass momentum accounting for a contact-free sponge diagnostic.

No solver impulse is converted into an inferred mass and no damping correction
is subtracted from observations. All reductions use float64, including the pad.
"""
import numpy as np
from scipy.spatial.transform import Rotation

CONFIG = {'dt_s': .0005, 'fall_steps': 100, 'pulse_steps': 100,
          'release_steps': 200, 'lift_m': .3, 'force_world_n': [.4, 0., 0.],
          'momentum_relative_tolerance': .15, 'release_relative_tolerance': .05,
          'minimum_obstacle_gap_m': .01, 'maximum_bond_error_m': .001,
          'minimum_j': .1, 'minimum_edge_ratio': .5, 'maximum_edge_ratio': 1.5}


def force_for_step(step):
    """Force integrated over the interval ending at this native step."""
    if type(step) is not int or not 0 <= step <= 400:
        raise ValueError('Native step must be an integer in [0,400]')
    return np.array(CONFIG['force_world_n'] if 100 < step <= 200 else [0., 0., 0.])


def lumped_masses(foam):
    rest = np.asarray(foam['rest_points_m'], dtype=np.float64)
    tets = np.asarray(foam['tetrahedra'], dtype=np.int64)
    mass = float(foam['total_mass_kg'])
    if (rest.ndim != 2 or rest.shape[1] != 3 or tets.ndim != 2 or tets.shape[1] != 4
            or not len(tets) or tets.min() < 0 or tets.max() >= len(rest)
            or not np.isfinite(rest).all() or not np.isfinite(mass) or mass <= 0):
        raise ValueError('Complete finite rest topology and positive declared foam mass required')
    volumes = np.linalg.det(rest[tets[:, 1:]] - rest[tets[:, :1]]) / 6.
    if (volumes <= 0).any():
        raise ValueError('Positive rest tetrahedra required')
    weights = np.zeros(len(rest), dtype=np.float64)
    np.add.at(weights, tets.ravel(), np.repeat(volumes / 4., 4))
    if (weights <= 0).any():
        raise ValueError('Every archived foam node needs positive incident volume')
    return weights * (mass / weights.sum())


def evaluate(raw, foam, native_pad_mass, pad_com_local, gravity_world):
    """All indices are exact native-step boundaries, independent of float time."""
    keys = ('time_s', 'native_step', 'pad_q', 'pad_qd', 'foam_q', 'foam_qd', 'force_world_n')
    data = {k: np.asarray(raw[k], dtype=np.float64) for k in keys}
    expected_shapes = {'time_s': (401,), 'native_step': (401,), 'pad_q': (401, 7),
        'pad_qd': (401, 6), 'foam_q': (401, len(foam['rest_points_m']), 3),
        'foam_qd': (401, len(foam['rest_points_m']), 3), 'force_world_n': (401, 3)}
    if any(data[k].shape != expected_shapes[k] for k in keys):
        raise ValueError('Complete401-frame/400-step archive required')
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError('Nonfinite native archive')
    time = data['time_s']; dt = np.diff(time)
    if (not np.array_equal(data['native_step']-data['native_step'][0], np.arange(401))
            or abs(time[0]) > 1e-12
            or not np.allclose(dt, float(np.float32(CONFIG['dt_s'])), atol=1e-10, rtol=0)):
        raise ValueError('Missing or irregular native integration step')
    expected_force = np.array([force_for_step(i) for i in range(401)])
    if not np.array_equal(data['force_world_n'], expected_force.astype(np.float32).astype(float)):
        raise ValueError('Applied force differs from the frozen pulse schedule')
    if not np.isfinite(native_pad_mass) or native_pad_mass <= 0:
        raise ValueError('Positive actual pad mass required')
    gravity = np.asarray(gravity_world, dtype=float)
    if gravity.shape != (3,) or not np.allclose(gravity, [0., 0., -9.81], atol=1e-5, rtol=0):
        raise ValueError('Unchanged downward gravity required')
    weights = lumped_masses(foam); total = weights.sum() + native_pad_mass
    momentum = np.einsum('n,fnk->fk', weights, data['foam_qd']) + native_pad_mass*data['pad_qd'][:, :3]
    rotation = Rotation.from_quat(data['pad_q'][:, 3:])
    # Isaac's pinned SciPy Cython backend requires a writable buffer here;
    # broadcast_to alone returns a read-only view even for ordinary CPU data.
    pad_center = rotation.apply(np.broadcast_to(np.asarray(pad_com_local), (401, 3)).copy()) + data['pad_q'][:, :3]
    com = (np.einsum('n,fnk->fk', weights, data['foam_q']) + native_pad_mass*pad_center) / total
    impulse = np.cumsum(data['force_world_n'][1:] * dt[:, None], axis=0)
    pulse_impulse = impulse[199] - impulse[99]
    pulse_dp = momentum[200]-momentum[100]
    relative_x = abs(pulse_dp[0]-pulse_impulse[0])/abs(pulse_impulse[0])
    release_delta = momentum[400, 0]-momentum[200, 0]
    release_ratio = abs(release_delta)/max(abs(momentum[200, 0]), 1e-15)
    motion = []
    for name, first, last in [('pulse', 100, 200), ('release', 200, 400)]:
        integrated = np.sum((momentum[first:last, 0]+momentum[first+1:last+1, 0])/(2*total)*dt[first:last])
        displacement = com[last, 0]-com[first, 0]
        motion.append({'phase': name, 'com_displacement_x_m': float(displacement),
            'integrated_com_velocity_x_m': float(integrated),
            'relative_q_vs_qd_error': float(abs(displacement-integrated)/max(abs(integrated), 1e-12))})
    gravity_windows = []
    for name, first, last in [('fall', 0, 100), ('pulse', 100, 200), ('release', 200, 400), ('all', 0, 400)]:
        expected = total*gravity[2]*(time[last]-time[first])
        actual = momentum[last, 2]-momentum[first, 2]
        integrated_z = np.sum((momentum[first:last, 2]+momentum[first+1:last+1, 2])/(2*total)*dt[first:last])
        displacement_z = com[last, 2]-com[first, 2]
        gravity_windows.append({'phase': name, 'first_step': first, 'last_step': last,
            'expected_impulse_ns': float(expected), 'observed_delta_p_ns': float(actual),
            'relative_error': float(abs(actual-expected)/abs(expected)),
            'com_displacement_z_m': float(displacement_z), 'integrated_com_velocity_z_m': float(integrated_z),
            'relative_q_vs_qd_error': float(abs(displacement_z-integrated_z)/max(abs(integrated_z), 1e-12))})
    threshold = float(foam['damping_receipt']['settling_threshold_m_s'])
    node_pose_speed = np.linalg.norm(np.diff(data['foam_q'], axis=0)/dt[:, None, None], axis=2)
    node_raw_speed = np.linalg.norm(data['foam_qd'][1:], axis=2)
    # Both observable speeds must exceed the threshold throughout pulse/release.
    # This is not a readback of the private solver's intermediate posDelta buffer.
    settling_possible = (node_pose_speed[100:] <= threshold) | (node_raw_speed[100:] <= threshold)
    checks = {'complete_native_stream': True, 'finite_states': True,
        'horizontal_pulse_momentum': bool(relative_x <= CONFIG['momentum_relative_tolerance']),
        'release_horizontal_momentum': bool(release_ratio <= CONFIG['release_relative_tolerance']),
        'actual_horizontal_com_motion': all(x['com_displacement_x_m'] > 0
            and x['relative_q_vs_qd_error'] <= CONFIG['momentum_relative_tolerance'] for x in motion),
        'gravity_momentum_all_windows': all(x['relative_error'] <= CONFIG['momentum_relative_tolerance'] for x in gravity_windows),
        'actual_gravity_com_motion': all(x['com_displacement_z_m'] < 0
            and x['relative_q_vs_qd_error'] <= CONFIG['momentum_relative_tolerance'] for x in gravity_windows),
        'no_horizontal_force_before_pulse': bool(abs(momentum[100, 0]-momentum[0, 0]) <= .15*abs(pulse_impulse[0])),
        'no_unforced_transverse_momentum': bool(np.max(np.abs(momentum[:, 1]-momentum[0, 1])) <= .15*abs(pulse_impulse[0])),
        'settling_not_indicated_during_pulse_release': not bool(settling_possible.any())}
    measurements = {'declared_foam_mass_kg': float(weights.sum()), 'native_pad_mass_kg': native_pad_mass,
        'total_accounted_mass_kg': float(total), 'node_mass_ratio': float(weights.max()/weights.min()),
        'node_mass_source': 'Declared foam mass times incident rest-tet volume/4, normalized; not a native inverse-mass readback.',
        'actual_force_pulse_impulse_ns': pulse_impulse.tolist(), 'pulse_delta_p_ns': pulse_dp.tolist(),
        'pulse_relative_x_error': float(relative_x), 'release_delta_px_ns': float(release_delta),
        'release_relative_x_change': float(release_ratio), 'gravity_windows': gravity_windows,
        'actual_horizontal_motion': motion,
        'settling_candidate_node_samples_in_pulse_release': int(settling_possible.sum()),
        'minimum_observed_node_pose_speed_pulse_release_m_s': float(node_pose_speed[100:].min()),
        'minimum_observed_node_raw_speed_pulse_release_m_s': float(node_raw_speed[100:].min()),
        'damping_correction_applied': False, 'damping_receipt': foam['damping_receipt'],
        'scope': 'Declared-model total momentum consistency. Does not separately identify actual native foam mass, attachment impulses or damping implementation.'}
    derived = {'time_s': time, 'node_masses_kg': weights, 'momentum_ns': momentum,
               'com_world_m': com, 'com_velocity_m_s': momentum/total,
               'external_impulse_cumulative_ns': impulse}
    return checks, measurements, derived
