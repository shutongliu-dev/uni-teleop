"""Reset-only pose seed from the accepted v3 contact test's released state.

No recorded velocities or intermediate trajectory are replayed. The new robot
scene must independently qualify this placement with zero reset velocities.
"""
from pathlib import Path
import hashlib
import numpy as np

from bench.paths import DATA_ROOT

TRACE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/'
             'threaded_asset_native_20261016/runs/contact_validation_06/result/drive_pi_trace.npz')
TRACE_SHA256 = 'b6e4e7d7d80b2f507047b7756e46ae4196a55ae56678f34a0cb3a52e9c1cf051'


def released_pose_seed(fixture_position_m=(.38, 0., .78)):
    if hashlib.sha256(TRACE.read_bytes()).hexdigest() != TRACE_SHA256:
        raise ValueError('Accepted threaded pose source changed')
    with np.load(TRACE, allow_pickle=False) as data:
        if str(data['state_layout']) != 'driver_xyz_xyzw,bolt_xyz_xyzw,driver_COM_velocity_6,bolt_COM_velocity_6':
            raise ValueError('Unexpected native trace layout')
        state = data['state'][-1].astype(float)
        record = data['trace'][-1].astype(float)
    if state.shape != (26,) or not np.isfinite(state).all() or record[0] != 20. or record[1] != 3.:
        raise ValueError('Expected the final t=20s state, two seconds after wrench release')
    offset = np.asarray(fixture_position_m, dtype=float)-[-.75, 0., 0.]
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('Expected finite fixture origin')
    poses = {}
    for name, start in [('screwdriver', 0), ('bolt', 7)]:
        pose = state[start:start+7]
        q = pose[[6, 3, 4, 5]]
        q /= np.linalg.norm(q)
        poses[name] = {'position_m': (pose[:3]+offset).tolist(), 'quaternion_wxyz': q.tolist()}
    return poses, {'schema': 'mechanical_threaded_reset_seed.v1',
        'source_trace': str(TRACE), 'source_trace_sha256': TRACE_SHA256,
        'source_time_s': 20., 'source_release_duration_s': 2.,
        'source_fixture_position_m': [-.75, 0., 0.],
        'fixture_position_m': list(fixture_position_m), 'translation_m': offset.tolist(),
        'source_measured_engagement_depth_m': float(record[2]),
        'reset_velocities': 'zero', 'trajectory_replay': False,
        'scope': 'New reset condition seeded from exact accepted v3 asset poses; its gravity stability and robot manipulation require fresh native evidence.'}
