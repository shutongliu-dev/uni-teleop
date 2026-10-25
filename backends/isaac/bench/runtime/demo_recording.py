"""Opt-in private RGB capture of the original native episode, never a replay."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image


class DemoRecordingError(RuntimeError):
    """A requested original-episode recording could not be completed."""


class NativeDemoRecorder:
    @classmethod
    def from_environment(cls, env, output, *, control_period_s):
        raw = os.environ.get('BENCH_DEMO_FPS', '')
        if raw in ('', '0'):
            return None
        return cls(env, output, sampling_rate_hz=raw,
                   camera_ids=os.environ.get('BENCH_DEMO_CAMERAS', 'task,overview').split(','),
                   control_period_s=control_period_s)

    def __init__(self, env, output, *, sampling_rate_hz, camera_ids, control_period_s):
        self.env = env
        self.output = Path(output) / 'demo_recording'
        self.output.mkdir(parents=True, exist_ok=False)
        self.frame_count = 0
        self.last_step = -1
        self.last_time = None
        self.last_frame_time = None
        self.last_frame_step = None
        self.minimum_interval = self.maximum_interval = None
        self.terminal_nonuniform_interval = None
        self.closed = False
        self.error = None
        self.hasher = hashlib.sha256()
        self.stream = None
        self.report = {
            'schema': 'native_episode_demo_recording.v1', 'enabled': True,
            'completed': False, 'reason': 'recording', 'status': 'recording',
            'capture_source': 'original_native_episode_rtx',
            'requested_sampling_rate_hz': sampling_rate_hz,
            'environment': {'BENCH_DEMO_FPS': str(sampling_rate_hz),
                            'BENCH_DEMO_CAMERAS': ','.join(camera_ids)},
            'sampling_rate_hz': None, 'control_period_s': control_period_s,
            'stride': None, 'camera_ids': list(camera_ids),
            'frames_file': 'frames.jsonl', 'object_state_replay': False,
            'physics_advanced_by_capture': False,
            'scope': 'Actual RGB at recorded simulation timestamps; no interpolation or wall-clock idle frames.'}
        try:
            if isinstance(sampling_rate_hz, bool) or not re.fullmatch(r'[0-9]+', str(sampling_rate_hz)):
                raise ValueError('BENCH_DEMO_FPS must be a positive integer divisor of 50')
            fps = int(sampling_rate_hz)
            if fps <= 0 or fps > 50 or 50 % fps or not math.isclose(control_period_s, .02, abs_tol=1e-12):
                raise ValueError('Demo capture requires 20ms control steps and an FPS divisor of 50')
            if (not camera_ids or len(set(camera_ids)) != len(camera_ids)
                    or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name)
                           or name not in env.camera_ids for name in camera_ids)):
                raise ValueError('Demo cameras must be distinct existing camera IDs')
            self.stride = 50 // fps
            self.camera_configuration = {name: {'width': env.cameras[name]['resolution'][0],
                                                'height': env.cameras[name]['resolution'][1]}
                                         for name in camera_ids}
            self.report.update(sampling_rate_hz=fps, stride=self.stride,
                               sampling_period_s=1. / fps, cameras=self.camera_configuration)
            self.stream = (self.output / 'frames.jsonl').open('xb')
            self._save()
        except BaseException as exc:
            self._fail('configuration', exc)
            if self.stream is not None:
                try:
                    self.stream.close()
                except BaseException as close_error:
                    exc.add_note('Recording stream cleanup failed: ' + type(close_error).__name__)
            raise DemoRecordingError('Native demo configuration failed; inspect demo_recording/recording.json') from exc

    def configuration(self):
        return {key: self.report[key] for key in (
            'schema', 'enabled', 'capture_source', 'environment', 'sampling_rate_hz',
            'control_period_s', 'stride', 'camera_ids', 'cameras', 'sampling_period_s')}

    def _save(self):
        self.report.update(frame_count=self.frame_count, last_control_step=self.last_step,
                           sim_duration_s=self.last_time, frames_sha256=self.hasher.hexdigest(),
                           error=self.error, interval_summary={
                               'minimum_s': self.minimum_interval, 'maximum_s': self.maximum_interval,
                               'terminal_nonuniform_interval_s': self.terminal_nonuniform_interval})
        target = self.output / 'recording.json'
        temporary = self.output / '.recording.json.tmp'
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False) + '\n')
        temporary.replace(target)

    def _fail(self, stage, exc):
        if self.error is None:
            self.error = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc),
                          'state_index': self.last_step, 'sim_time_s': self.last_time}
        self.report.update(completed=False, reason='recording_error', status='failed')
        try:
            self._save()
        except BaseException as sealing_error:
            exc.add_note('Recording error metadata could not be sealed: ' + type(sealing_error).__name__)

    def _state(self):
        # Some adapters return array views; retain an independent pre-render copy.
        return {key: np.array(value, copy=True) for key, value in self.env.capture_state().items()}

    def _unchanged(self, before):
        after = self.env.capture_state()
        if before.keys() != after.keys() or any(not np.array_equal(before[k], after[k]) for k in before):
            raise RuntimeError('Demo rendering changed physical state')

    def record(self, observation, score):
        """Visit reset and every executed control step, including uncaptured steps."""
        if self.closed or self.error is not None:
            raise DemoRecordingError('Native demo recorder is closed or failed')
        try:
            step = self.last_step + 1
            stamp = float(observation['sim_time_s'])
            if not math.isfinite(stamp) or not math.isclose(stamp, step * .02, abs_tol=1e-8):
                raise ValueError('Demo observations must cover reset and consecutive 20ms steps')
            self.last_step, self.last_time = step, stamp
            if step % self.stride == 0:
                self._capture(score, 'reset' if step == 0 else 'periodic')
        except BaseException as exc:
            self._fail('capture', exc)
            raise DemoRecordingError('Native demo capture failed; inspect demo_recording/recording.json') from exc

    def _capture(self, score, kind):
        before = self._state()
        cameras = {}
        for name in self.report['camera_ids']:
            width, height = (self.camera_configuration[name][key] for key in ('width', 'height'))
            camera_before = self._state()
            capture = self.env.render_rgbd(camera=name, width=width, height=height)
            self._unchanged(camera_before)
            if any(not math.isclose(float(capture[key]), self.last_time, abs_tol=1e-9)
                   for key in ('sim_time_s', 'depth_sim_time_s')):
                raise RuntimeError('Demo RGB-D timestamp differs from the physical observation')
            rgb = np.asarray(capture['rgb'])
            if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3) or float(rgb.std()) < 1:
                raise RuntimeError('Invalid or blank native demo RGB frame')
            relative = Path('rgb') / name / f'{self.frame_count:08d}.png'
            path = self.output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(path)
            cameras[name] = {'path': relative.as_posix(),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'width': width, 'height': height}
        self._unchanged(before)
        interval = None if self.last_frame_time is None else self.last_time - self.last_frame_time
        row = {'schema': 'native_episode_demo_frame.v1', 'frame_index': self.frame_count,
               'state_index': self.last_step, 'sim_time_s': self.last_time, 'capture_kind': kind,
               'interval_from_previous_s': interval,
               'score': {key: score.get(key) for key in ('placed_count', 'required_count', 'success')},
               'cameras': cameras, 'capture_state_unchanged': True,
               'rgbd_timestamp_verified': True}
        raw = (json.dumps(row, allow_nan=False) + '\n').encode()
        self.stream.write(raw)
        self.stream.flush()
        self.hasher.update(raw)
        self.frame_count += 1
        self.last_frame_time, self.last_frame_step = self.last_time, self.last_step
        if interval is not None:
            self.minimum_interval = interval if self.minimum_interval is None else min(interval, self.minimum_interval)
            self.maximum_interval = interval if self.maximum_interval is None else max(interval, self.maximum_interval)
            if kind == 'terminal' and self.last_step % self.stride:
                self.terminal_nonuniform_interval = interval
        self._save()

    def close(self, *, score, reason, completed=False):
        """Only normal finish may capture a tail; cleanup never retries a render."""
        if self.closed:
            return
        primary_error = None
        try:
            if self.error is None:
                if completed:
                    if self.last_step < 0:
                        raise RuntimeError('Cannot finish a recording without its reset frame')
                    if self.last_frame_step != self.last_step:
                        self._capture(score, 'terminal')
                self.report.update(completed=completed, reason=reason,
                                   status='completed' if completed else 'incomplete')
            self._save()
        except BaseException as exc:
            primary_error = exc
            self._fail('close', exc)
            raise DemoRecordingError('Native demo finalization failed; inspect demo_recording/recording.json') from exc
        finally:
            self.closed = True
            if self.stream is not None:
                try:
                    self.stream.close()
                except BaseException as exc:
                    if primary_error is not None:
                        primary_error.add_note('Recording stream cleanup failed: ' + type(exc).__name__)
                    else:
                        self._fail('stream_close', exc)
                        raise DemoRecordingError('Native demo stream close failed') from exc
