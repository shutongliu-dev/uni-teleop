"""Truth transport checks for native diagnostics; not physical qualification."""
import pytest

from scripts.diagnostics.review_living_remote_interactions import body_state, frame


def test_native_corrupt_quaternion_is_not_normalized_into_valid_evidence():
    with pytest.raises(ValueError, match='Invalid native'):
        body_state({'pose_xyzw': [0., 0., 1., 0., 0., 0., 2.], 'velocity': [0.]*6})


def test_native_nonfinite_velocity_is_rejected():
    with pytest.raises(ValueError, match='Invalid native'):
        body_state({'pose_xyzw': [0., 0., 1., 0., 0., 0., 1.], 'velocity': [float('nan')]+[0.]*5})


def test_two_sensor_directions_never_double_count_one_physical_pair():
    row = {'tick': 1, 'time_s': .0005,
        'bodies': {n: {'pose_xyzw': [0., 0., 1., 0., 0., 0., 1.], 'velocity': [0.]*6}
                   for n in ('body', 'power', 'confirm', 'holder')},
        'joints': {n: {'q_m': 0.} for n in ('power', 'confirm')},
        'contacts': [{'sensor': a, 'filter': b, 'count': 2,
                      'normal_impulse_l1_ns': .001, 'min_separation_m': -.00005}
                     for a,b in (('body','holder'),('holder','body'))]}
    native = frame(row, {'time_s': 0.}, .0005)
    assert len(native['contacts']) == 1
    assert native['contacts'][0]['impulse_magnitude_sum_ns'] == .001
    assert native['objects']['remote']['quaternion_wxyz'] == [1., 0., 0., 0.]
