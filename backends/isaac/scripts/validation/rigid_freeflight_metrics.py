"""Independent rigid-body calibration of the sponge free-flight wrench API."""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.validation.sponge_freeflight_metrics import CONFIG, force_for_step

MASS_KG = .0792
SIZE_M = np.array([.120, .078, .045])
CENTER_M = np.array([.25, -.67, 1.2225])
INERTIA_KG_M2 = MASS_KG / 12. * (np.sum(SIZE_M**2)-SIZE_M**2)
CORNERS_M = np.array([[x, y, z] for x in [-1., 1.] for y in [-1., 1.] for z in [-1., 1.]])*SIZE_M/2.


def evaluate(raw, mass, com_local, gravity_z):
    """Use actual native rigid mass, COM and velocities; no synthetic FEM data."""
    shapes = {'time_s': (401,), 'native_step': (401,), 'body_q': (401, 7), 'body_qd': (401, 6),
              'force_world_n': (401, 3), 'force_position_world_m': (401, 3), 'torque_world_nm': (401, 3)}
    data = {k: np.asarray(raw[k], dtype=float) for k in shapes}
    if any(data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in shapes.items()):
        raise ValueError('Complete finite 401-frame rigid archive required')
    if not np.isclose(mass, MASS_KG, rtol=2e-6, atol=1e-10) or not np.isclose(gravity_z, -9.81, atol=1e-5, rtol=0):
        raise ValueError('Original nominal rigid calibration mass and gravity required')
    dt = np.diff(data['time_s'])
    if (data['time_s'][0] != 0 or not np.allclose(dt, float(np.float32(CONFIG['dt_s'])), atol=1e-10, rtol=0)
            or not np.array_equal(data['native_step']-data['native_step'][0], np.arange(401))):
        raise ValueError('Missing or irregular native step')
    force = np.asarray([force_for_step(i) for i in range(401)], dtype=np.float32).astype(float)
    if not np.array_equal(force, data['force_world_n']) or np.any(data['torque_world_nm']):
        raise ValueError('Exact world-X force schedule and zero applied torque required')
    rotation = Rotation.from_quat(data['body_q'][:, 3:])
    centers = rotation.apply(np.tile(np.asarray(com_local, dtype=float), (401, 1)))+data['body_q'][:, :3]
    if not np.array_equal(data['force_position_world_m'][1:], centers[:-1].astype(np.float32).astype(float)):
        raise ValueError('Applied native wrench positions must match preceding world COM')
    p = mass*data['body_qd'][:, :3]
    impulse = np.sum(force[101:201]*dt[100:200, None], axis=0)
    dp = p[200]-p[100]
    pulse_error = abs(dp[0]-impulse[0])/impulse[0]
    release_error = abs(p[400, 0]-p[200, 0])/max(abs(p[200, 0]), 1e-15)
    gravity_windows, motion = [], []
    for phase, first, last in [('fall', 0, 100), ('pulse', 100, 200), ('release', 200, 400), ('all', 0, 400)]:
        expected = mass*gravity_z*(data['time_s'][last]-data['time_s'][first]); observed = p[last, 2]-p[first, 2]
        integrated = np.sum(.5*(p[first:last]+p[first+1:last+1])*dt[first:last, None], axis=0)/mass
        displacement = centers[last]-centers[first]
        gravity_windows.append({'phase': phase, 'expected_impulse_ns': float(expected), 'observed_delta_p_ns': float(observed),
            'relative_error': float(abs(observed-expected)/abs(expected)), 'actual_z_displacement_m': float(displacement[2]),
            'q_vs_qd_relative_error': float(abs(displacement[2]-integrated[2])/max(abs(integrated[2]), 1e-15))})
        if phase in ('pulse', 'release'):
            motion.append({'phase': phase, 'actual_x_displacement_m': float(displacement[0]),
                'integrated_x_velocity_m': float(integrated[0]),
                'q_vs_qd_relative_error': float(abs(displacement[0]-integrated[0])/max(abs(integrated[0]), 1e-15))})
    checks = {'finite_complete_native_stream': True, 'same_world_COM_wrench_schedule': True,
        'horizontal_pulse_momentum': bool(pulse_error <= .15), 'release_horizontal_momentum': bool(release_error <= .05),
        'gravity_momentum_all_windows': all(x['relative_error'] <= .15 for x in gravity_windows),
        'actual_gravity_com_motion': all(x['actual_z_displacement_m'] < 0 and x['q_vs_qd_relative_error'] <= .15 for x in gravity_windows),
        'actual_horizontal_com_motion': all(x['actual_x_displacement_m'] > 0 and x['q_vs_qd_relative_error'] <= .15 for x in motion),
        'no_horizontal_momentum_before_pulse': bool(abs(p[100, 0]-p[0, 0]) <= .15*impulse[0]),
        'no_unforced_transverse_momentum': bool(np.max(np.abs(p[:, 1]-p[0, 1])) <= .15*impulse[0]),
        'no_unforced_rotation': bool(np.max(np.linalg.norm(data['body_qd'][:, 3:], axis=1)) <= .001)}
    measurements = {'native_rigid_mass_kg': float(mass), 'actual_force_pulse_impulse_ns': impulse.tolist(),
        'pulse_delta_p_ns': dp.tolist(), 'pulse_relative_x_error': float(pulse_error), 'release_relative_x_change': float(release_error),
        'gravity_windows': gravity_windows, 'horizontal_motion': motion,
        'damping_correction_applied': False, 'rigid_only_no_fem_mass_approximation': True}
    return checks, measurements, {'time_s': data['time_s'], 'momentum_ns': p, 'com_world_m': centers}
