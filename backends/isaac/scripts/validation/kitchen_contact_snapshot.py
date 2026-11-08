"""Read-only snapshot of three native contact APIs at one unchanged physics tick."""
from pathlib import Path
import numpy as np
from bench.evaluation.cloth.cloth_robot_collision import state_digest
from bench.simulation.isaacsim_backend.contact_sample import copy_array


def capture_contact_snapshot(env, output):
    """Preserve unmodified native buffers, including reported surplus impulses."""
    output = Path(output)
    before = env.capture_state()
    clock = (int(env.world.current_time_step_index), float(env.world.current_time), float(env.time_s))
    raw = tuple(copy_array(value) for value in env._rigid_contacts.get_contact_data(dt=1.))
    if len(raw) != 6:
        raise RuntimeError('Expected all six native contact buffers')
    counts, starts = raw[-2:]
    if counts.shape != starts.shape or np.any(counts < 0):
        raise RuntimeError('Malformed native contact indices')
    active = np.argwhere(counts > 0)
    used = max((int(starts[i, j])+int(counts[i, j]) for i, j in active), default=0)
    if used >= env._rigid_contacts.max_contact_data_count:
        raise RuntimeError('Contact data reached capacity')
    net = copy_array(env._rigid_contacts.get_net_contact_forces(dt=1.))
    matrix = copy_array(env._rigid_contacts.get_contact_force_matrix(dt=1.))
    after = env.capture_state()
    if state_digest(before) != state_digest(after) or clock != (
            int(env.world.current_time_step_index), float(env.world.current_time), float(env.time_s)):
        raise RuntimeError('Native contact getters changed physical state or time')
    arrays = dict(zip(('scalar_impulses', 'points', 'normals', 'separations'), [x[:used] for x in raw[:4]]))
    arrays.update(counts=counts, starts=starts, net_impulses=net, pair_impulses=matrix)
    if not all(np.isfinite(x).all() for x in arrays.values()):
        raise RuntimeError('Nonfinite native contact snapshot')
    np.savez_compressed(output, **arrays)
    return {'schema': 'native_contact_api_snapshot.v1', 'read_only': True,
            'physics_dt_s': float(env.dt), 'API_dt_argument_s': 1.,
            'native_step': clock[0], 'native_time_s': clock[1], 'public_time_s': clock[2],
            'sensor_paths': env._contact_sensor_paths, 'filter_paths': env._contact_filter_paths,
            'used_contact_entries': used, 'buffer_capacity': env._rigid_contacts.max_contact_data_count,
            'state_sha256_before': state_digest(before), 'state_sha256_after': state_digest(after),
            'array_shapes': {k: list(v.shape) for k, v in arrays.items()},
            'scope': 'Unmodified getters at the same terminal tick. API dt=1 returns impulse-valued data. No division by shape count, correction, threshold change or claim that the three APIs are independent sensors.'}
