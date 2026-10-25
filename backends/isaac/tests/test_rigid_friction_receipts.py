"""Friction recording preserves independent native indexing and invalid evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from bench.simulation.isaacsim_backend.contact_sample import copy_array
from scripts.validation.rigid_friction_receipts import FrictionReceipts


def fixture(tmp_path):
    counts = np.array([[1, 0], [0, 1]], dtype=np.uint32)
    starts = np.array([[3, 99], [99, 6]], dtype=np.uint32)
    vectors = np.full((12, 3), np.nan)
    points = np.full((12, 3), np.nan)
    vectors[3:5] = [[1, 2, 3], [4, 5, 6]]
    vectors[6] = [-5, -7, -9]
    points[3:5] = [[.1, .2, .3], [.4, .5, .6]]
    points[6] = [.7, .8, .9]
    # Snapshot the original normal mapping before the shared native buffers
    # are overwritten by a friction query with different anchor counts.
    normal = tuple(copy_array(a) for a in (counts, starts))
    def query(dt):
        assert dt == 1.
        counts[0, 0] = 2
        return vectors, points, counts, starts
    view = SimpleNamespace(sensor_count=2, filter_count=2,
        max_contact_data_count=12, get_friction_data=query)
    env = SimpleNamespace(_rigid_contacts=view, _contact_sensor_paths=['/A', '/B'],
        _contact_filter_paths=[['/B', '/table'], ['/table', '/A']],
        world=SimpleNamespace(current_time_step_index=42))
    sample = SimpleNamespace(for_env=lambda e: normal)
    return env, sample, (vectors, points, counts, starts), FrictionReceipts(env, tmp_path)


def test_independent_anchor_indices_directed_pairs_and_buffer_reuse(tmp_path):
    env, sample, arrays, recorder = fixture(tmp_path)
    row = recorder.capture(env, sample)
    assert sample.for_env(env)[0][0, 0] == 1
    assert row['directed_anchor_count'] == 3
    a, b = row['pairs']
    assert (a['sensor_index'], a['filter_index'], a['start'], a['count']) == (0, 0, 3, 2)
    assert (b['sensor_index'], b['filter_index'], b['start'], b['count']) == (1, 1, 6, 1)
    assert a['raw_friction_vectors_dt1'] == [[1, 2, 3], [4, 5, 6]]
    assert b['raw_anchor_points'] == [[.7, .8, .9]]
    arrays[0][:] = 0
    arrays[1][:] = 0
    arrays[2][:] = 0
    assert a['raw_friction_vectors_dt1'] == [[1, 2, 3], [4, 5, 6]]
    mapping = load_json(tmp_path/'friction_mapping.json')
    assert mapping['sensor_paths'] == ['/A', '/B']
    assert mapping['filter_paths_per_sensor'][1][1] == '/A'
    assert mapping['native_frame_sign_units_and_time_semantics_calibrated'] is False
    recorder.accept_readonly()
    assert recorder.pending_arrays is None


def test_empty_frame_ignores_unused_values_and_starts(tmp_path):
    env, sample, arrays, recorder = fixture(tmp_path)
    env._rigid_contacts.get_friction_data = lambda dt: arrays
    arrays[2][:] = 0
    arrays[0][:] = np.nan
    arrays[1][:] = np.nan
    assert recorder.capture(env, sample)['pairs'] == []


@pytest.mark.parametrize('defect', ['capacity', 'out_of_bounds', 'negative_count',
    'negative_start', 'float_count', 'nan_vector', 'inf_anchor', 'wrong_shape', 'missing_array'])
def test_invalid_capture_preserves_raw_arrays_before_failure(tmp_path, defect):
    env, sample, original, recorder = fixture(tmp_path)
    arrays = list(original)
    if defect == 'capacity': arrays[3][0, 0] = 11
    if defect == 'out_of_bounds': arrays[3][0, 0] = 12
    if defect in ('negative_count', 'negative_start'):
        index = 2 if defect == 'negative_count' else 3
        arrays[index] = arrays[index].astype(np.int64)
        arrays[index][0, 0] = -1
    if defect == 'float_count': arrays[2] = arrays[2].astype(float)
    if defect == 'nan_vector': arrays[0][3, 0] = np.nan
    if defect == 'inf_anchor': arrays[1][3, 0] = np.inf
    if defect == 'wrong_shape': arrays[3] = arrays[3][:1]
    if defect == 'missing_array': arrays.pop()
    env._rigid_contacts.get_friction_data = lambda dt: arrays
    with pytest.raises(ValueError): recorder.capture(env, sample)
    saved = np.load(tmp_path/'friction_capture_failure.npz')
    for i, array in enumerate(arrays):
        np.testing.assert_equal(saved[f'raw_{i}'], array)
    failure = load_json(tmp_path/'friction_capture_failure.json')
    assert failure['native_step'] == 42
    assert failure['raw_array_count'] == len(arrays)


def test_mapping_change_is_rejected(tmp_path):
    env, sample, arrays, recorder = fixture(tmp_path)
    env._contact_filter_paths[0][0] = '/wrong'
    with pytest.raises(ValueError, match='mapping changed'):
        recorder.capture(env, sample)
    assert not (tmp_path/'friction_capture_failure.npz').exists()


def test_partial_copy_failure_retains_arrays_before_and_after_bad_buffer(tmp_path):
    env, sample, original, recorder = fixture(tmp_path)
    class BadBuffer:
        def numpy(self):
            raise RuntimeError('Broken native buffer')
    arrays = list(original)
    arrays[1] = BadBuffer()
    env._rigid_contacts.get_friction_data = lambda dt: arrays
    with pytest.raises(ValueError, match='Failed to copy'):
        recorder.capture(env, sample)
    saved = np.load(tmp_path/'friction_capture_failure.npz')
    assert set(saved.files) == {'raw_0', 'raw_2', 'raw_3'}
    np.testing.assert_equal(saved['raw_0'], original[0])
    failure = load_json(tmp_path/'friction_capture_failure.json')
    assert failure['copied_array_indices'] == [0, 2, 3]
    assert failure['copy_failures'][0]['index'] == 1


def test_readonly_rejection_preserves_raw_and_both_states(tmp_path):
    env, sample, original, recorder = fixture(tmp_path)
    recorder.capture(env, sample)
    before = {'object_q': np.array([[0., 0., 0., 0., 0., 0., 1.]])}
    after = {'object_q': before['object_q']+.001}
    recorder.reject_readonly(env, before, after, RuntimeError('Read-only observation modified state'))
    saved = np.load(tmp_path/'friction_capture_failure.npz')
    assert len(saved.files) == 6
    np.testing.assert_equal(saved['before_object_q'], before['object_q'])
    np.testing.assert_equal(saved['after_object_q'], after['object_q'])
    for i, array in enumerate(original):
        np.testing.assert_equal(saved[f'raw_{i}'], array)
    failure = load_json(tmp_path/'friction_capture_failure.json')
    assert failure['phase'] == 'readonly'
    assert failure['valid_frame'] is False
    assert recorder.pending_arrays is None


@pytest.mark.parametrize('mode', ['after_clock_mutation', 'after_state_exception',
    'two_valid_steps', 'disabled'])
def test_observer_records_only_completed_readonly_frames(tmp_path, monkeypatch, mode):
    import scripts.validation.validate_assembly_contacts as mod
    counts = np.array([[1]], dtype=np.uint32)
    starts = np.array([[2]], dtype=np.uint32)
    vectors, points = np.ones((8, 3)), np.ones((8, 3))*.1
    def friction(dt):
        assert mode != 'disabled', 'Default observer must not call friction API'
        return vectors, points, counts, starts
    view = SimpleNamespace(sensor_count=1, filter_count=1, max_contact_data_count=8,
        get_contact_data=lambda dt: (np.ones((8, 1)), points, vectors, np.ones((8, 1)), counts, starts),
        get_friction_data=friction)
    env = SimpleNamespace(_rigid_contacts=view, _contact_sensor_paths=['/A'],
        _contact_filter_paths=[['/B']], _reset_count=0, dt=.001, time_s=.041,
        _current_step_time=.042, world=SimpleNamespace(current_time_step_index=41, current_time=.041),
        scene_spec={} if mode == 'disabled' else {'engineering_contact_friction_capture': True},
        scene_info={'robot_link_paths': {}}, object_names=[], capture_calls=0)
    def capture():
        env.capture_calls += 1
        if env.capture_calls == 2:
            if mode == 'after_state_exception':
                raise RuntimeError('Injected after-capture read failure')
            if mode == 'after_clock_mutation':
                env.world.current_time += .001
        return {'object_q': np.zeros((0, 7)), 'object_qd': np.zeros((0, 6))}
    def step(render):
        env.world.current_time_step_index += 1
        env.world.current_time += env.dt
    env.capture_state, env.world.step = capture, step
    env._project_rigid_contacts = lambda sample: []
    monkeypatch.setattr(mod, 'contact_rows', lambda env, sample: [])
    monkeypatch.setattr(mod, 'declared_mount_pairs', lambda env: [])
    observer = mod.Observer(env, tmp_path)
    invalid = mode.startswith('after_')
    try:
        if invalid:
            with pytest.raises(RuntimeError):
                observer.step(render=False)
        else:
            for _ in range(2):
                observer.step(render=False)
                env.time_s = env._current_step_time
                observer.exchange.consume(env)
                env._current_step_time = env.time_s+env.dt
        observer.stream.flush()
        rows = (tmp_path/'native_frames.jsonl').read_text().splitlines()
        assert len(rows) == (0 if invalid else 2)
        if observer.friction is not None:
            assert observer.friction.pending_arrays is None
        if invalid:
            failure = load_json(tmp_path/'friction_capture_failure.json')
            assert failure['valid_frame'] is False and failure['phase'] == 'readonly'
            saved = np.load(tmp_path/'friction_capture_failure.npz')
            assert {f'raw_{i}' for i in range(4)} <= set(saved.files)
            assert {'before_object_q', 'before_object_qd'} <= set(saved.files)
            if mode == 'after_clock_mutation':
                assert {'after_object_q', 'after_object_qd'} <= set(saved.files)
        else:
            assert not (tmp_path/'friction_capture_failure.json').exists()
            assert (tmp_path/'friction_mapping.json').exists() == (mode != 'disabled')
    finally:
        assert observer.close()['all_samples_consumed_once'] == (not invalid)
