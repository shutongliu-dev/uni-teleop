"""Original-episode recording cadence and failure handling without native startup."""
import hashlib
import json

import numpy as np
import pytest

from bench.paths import load_json
from bench.runtime.demo_recording import DemoRecordingError, NativeDemoRecorder


class NativeDouble:
    camera_ids = ['task', 'overview']
    cameras = {name: {'resolution': [8, 6]} for name in camera_ids}

    def __init__(self):
        self.time = 0.
        self.state = np.zeros(3)
        self.render_calls = []
        self.mode = None

    def capture_state(self):
        return {'positions': self.state, 'time': np.array([self.time])}

    def render_rgbd(self, *, camera, width, height):
        self.render_calls.append((camera, self.time))
        if self.mode == 'second_camera_error' and camera == 'overview':
            raise ValueError('native camera failed')
        if self.mode == 'mutate':
            self.state[0] += 1
        rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
        if self.mode == 'blank':
            rgb[:] = 0
        return {'rgb': rgb, 'depth_m': np.ones((height, width), np.float32),
                'sim_time_s': self.time, 'depth_sim_time_s': self.time + (.02 if self.mode == 'timestamp' else 0.)}


SCORE = {'success': False, 'placed_count': 0, 'required_count': 3}


def recorder(tmp_path, env=None):
    env = env or NativeDouble()
    return env, NativeDemoRecorder(env, tmp_path, sampling_rate_hz=10,
                                   camera_ids=['task', 'overview'], control_period_s=.02)


def tick(env, rec, index, score=SCORE):
    env.time = index * .02
    rec.record({'sim_time_s': env.time}, score)


def manifest(tmp_path):
    return load_json(tmp_path / 'demo_recording/recording.json')


def rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / 'demo_recording/frames.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('value', [None, '', '0'])
def test_disabled_has_no_capture_or_artifacts(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv('BENCH_DEMO_FPS', raising=False)
    else:
        monkeypatch.setenv('BENCH_DEMO_FPS', value)
    monkeypatch.setenv('BENCH_DEMO_CAMERAS', '../invalid')
    env = NativeDouble()
    assert NativeDemoRecorder.from_environment(env, tmp_path, control_period_s=.02) is None
    assert not list(tmp_path.iterdir()) and not env.render_calls


def test_cadence_reset_and_nonuniform_terminal_without_physics_mutation(tmp_path):
    env, rec = recorder(tmp_path)
    for index in range(13):
        tick(env, rec, index)
    final = {'success': True, 'placed_count': 3, 'required_count': 3}
    rec.close(score=final, reason='finish_request', completed=True)
    saved = rows(tmp_path)
    assert [r['state_index'] for r in saved] == [0, 5, 10, 12]
    assert [r['sim_time_s'] for r in saved] == pytest.approx([0, .1, .2, .24])
    assert saved[-1]['capture_kind'] == 'terminal' and saved[-1]['score'] == final
    assert np.array_equal(env.state, np.zeros(3)) and env.time == .24
    report = manifest(tmp_path)
    assert report['completed'] and report['last_control_step'] == 12
    assert report['sim_duration_s'] == .24
    assert report['interval_summary']['terminal_nonuniform_interval_s'] == pytest.approx(.04)
    assert report['frames_sha256'] == hashlib.sha256((tmp_path/'demo_recording/frames.jsonl').read_bytes()).hexdigest()
    for row in saved:
        for camera in row['cameras'].values():
            assert hashlib.sha256((tmp_path/'demo_recording'/camera['path']).read_bytes()).hexdigest() == camera['sha256']
            assert [camera['width'], camera['height']] == [8, 6]
    assert not list(tmp_path.rglob('*.npy'))
    prior = (tmp_path/'demo_recording/recording.json').read_bytes()
    rec.close(score=SCORE, reason='worker_exit_without_finish')
    assert (tmp_path/'demo_recording/recording.json').read_bytes() == prior


@pytest.mark.parametrize('end', [0, 5, 10])
def test_on_grid_finish_does_not_duplicate_frame(tmp_path, end):
    env, rec = recorder(tmp_path)
    for index in range(end + 1):
        tick(env, rec, index)
    rec.close(score=SCORE, reason='finish_request', completed=True)
    assert [r['state_index'] for r in rows(tmp_path)] == list(range(0, end + 1, 5))
    assert manifest(tmp_path)['interval_summary']['terminal_nonuniform_interval_s'] is None


@pytest.mark.parametrize('mode', ['mutate', 'timestamp', 'blank', 'second_camera_error'])
def test_capture_failure_is_latched_and_cleanup_does_not_render(tmp_path, mode):
    env, rec = recorder(tmp_path)
    tick(env, rec, 0)
    env.mode = mode
    for index in range(1, 5):
        tick(env, rec, index)
    with pytest.raises(DemoRecordingError):
        tick(env, rec, 5)
    first = manifest(tmp_path)['error']
    calls = list(env.render_calls)
    rec.close(score=SCORE, reason='worker_exit_without_finish')
    report = manifest(tmp_path)
    assert report['error'] == first and report['status'] == 'failed' and not report['completed']
    assert report['frame_count'] == 1 and len(rows(tmp_path)) == 1
    assert env.render_calls == calls
    with pytest.raises(DemoRecordingError):
        tick(env, rec, 6)


def test_terminal_failure_cannot_be_relabelled_completed(tmp_path):
    env, rec = recorder(tmp_path)
    tick(env, rec, 0)
    tick(env, rec, 1)
    env.mode = 'timestamp'
    with pytest.raises(DemoRecordingError):
        rec.close(score=SCORE, reason='finish_request', completed=True)
    rec.close(score=SCORE, reason='worker_exit_without_finish')
    assert not manifest(tmp_path)['completed'] and manifest(tmp_path)['error']['stage'] == 'close'


def test_abnormal_close_does_not_make_terminal_frame(tmp_path):
    env, rec = recorder(tmp_path)
    tick(env, rec, 0)
    tick(env, rec, 1)
    rec.close(score=SCORE, reason='worker_exit_without_finish')
    assert len(rows(tmp_path)) == 1
    assert manifest(tmp_path)['status'] == 'incomplete'


@pytest.mark.parametrize('fps,cameras', [('nan', ['task']), ('inf', ['task']), (-1, ['task']),
                                       (3, ['task']), (10, []), (10, ['task', 'task']),
                                       (10, ['unknown']), (10, ['../task'])])
def test_invalid_configuration_is_explicit(tmp_path, fps, cameras):
    env = NativeDouble()
    with pytest.raises(DemoRecordingError):
        NativeDemoRecorder(env, tmp_path, sampling_rate_hz=fps,
                           camera_ids=cameras, control_period_s=.02)
    assert manifest(tmp_path)['error']['stage'] == 'configuration'
    assert not env.render_calls


def test_missing_control_step_fails_even_without_due_capture(tmp_path):
    env, rec = recorder(tmp_path)
    tick(env, rec, 0)
    with pytest.raises(DemoRecordingError):
        tick(env, rec, 2)
    rec.close(score=SCORE, reason='worker_exit_without_finish')
    assert manifest(tmp_path)['error']['type'] == 'ValueError'
