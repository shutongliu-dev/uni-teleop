"""CPU contract tests; native RGB-D acceptance also needs real RTX captures."""
import numpy as np
import pytest

from bench.simulation.isaacsim_backend.depth import validate_camera_z_depth
from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv


def test_native_depth_preserves_meters_and_explicitly_accounts_for_no_hit(tmp_path):
    raw = np.array([[0, .25, np.inf], [2.75, 4.5, -0.0]], dtype=np.float32)
    original = raw.copy()
    depth, stats = validate_camera_z_depth(raw, width=3, height=2)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(depth, [[0, .25, 0], [2.75, 4.5, 0]])
    assert not np.shares_memory(depth, raw)
    assert depth.dtype == np.float32 and depth.flags.c_contiguous
    assert stats['valid_pixel_count'] == stats['invalid_pixel_count'] == 3
    assert stats['raw_zero_pixel_count'] == 2
    assert stats['raw_positive_infinity_pixel_count'] == 1
    assert stats['minimum_valid_depth_m'] == .25
    assert stats['maximum_valid_depth_m'] == 4.5
    path = tmp_path / 'depth.npy'
    np.save(path, depth, allow_pickle=False)
    restored = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(restored, depth)
    assert restored.dtype == np.float32


@pytest.mark.parametrize('raw, message', [
    (np.array([[1, np.nan]], dtype=np.float32), 'NaN'),
    (np.array([[1, -np.inf]], dtype=np.float32), 'negative'),
    (np.array([[1, -.1]], dtype=np.float32), 'negative'),
    (np.array([[0, np.inf]], dtype=np.float32), 'no valid'),
    (np.array([[1, 2]], dtype=np.float64), 'float32'),
    (np.array([[1, 2]], dtype=object), 'float32'),
    (np.array([1, 2], dtype=np.float32), 'shape'),
    (np.ones((1, 2, 1), dtype=np.float32), 'shape'),
])
def test_bad_native_depth_is_rejected(raw, message):
    with pytest.raises(RuntimeError, match=message):
        validate_camera_z_depth(raw, width=2, height=1)


@pytest.mark.parametrize('key', ['joint_q', 'object_q', 'particle_q', 'time_s',
                                  'native_world_time_s', 'native_physics_step_index'])
def test_rgbd_capture_rejects_physical_state_changes(key):
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    states = iter([{key: np.array([0.])}, {key: np.array([1.])}])
    env.capture_state = lambda: next(states)
    env._render_frames = lambda *args, **kwargs: {'rgb': np.zeros((1, 1, 3))}
    with pytest.raises(RuntimeError, match='changed physical state'):
        env.render_rgbd('overview', 1, 1)


def test_rgbd_capture_requires_depth_and_preserves_frame():
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    frame = {'rgb': np.zeros((1, 1, 3), dtype=np.uint8),
             'depth_m': np.ones((1, 1), dtype=np.float32)}
    env.capture_state = lambda: {'joint_q': np.zeros(29), 'time_s': np.array(0.)}
    calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {'task': frame}
    env._render_frames = capture
    assert env.render_rgbd('task', 1, 1) is frame
    assert calls == [((('task',), 1, 1), {'include_depth': True})]
    assert env.render('task', 1, 1) is frame['rgb']
    assert calls[-1] == ((('task',), 1, 1), {'include_depth': False})
