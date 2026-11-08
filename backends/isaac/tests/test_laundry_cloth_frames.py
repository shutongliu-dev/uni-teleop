"""CPU fault injection for presentation guards; this is not RTX qualification."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from scripts.media import laundry_cloth_frames as media


def test_task_camera_center_moves_only_the_thickness_view():
    original = media.task_cameras([.65, -.55])
    assert original == media.CAMERAS
    stacked = media.task_cameras([.95, -.55])
    assert stacked['task'] == original['task']
    assert stacked['room'] == original['room']
    np.testing.assert_allclose(stacked['thickness']['eye'], [1.8, -1.42, 1.12])
    np.testing.assert_allclose(stacked['thickness']['target'], [.95, -.55, .91])
    # Callers may customize a returned definition without changing later tasks.
    stacked['thickness']['eye'][0] = 99.
    assert media.task_cameras([.65, -.55]) == original


@pytest.mark.parametrize('center', [[.65], [.65, -.55, .9], [float('nan'), 0.]])
def test_task_camera_center_rejects_invalid_coordinates(center):
    with pytest.raises(ValueError, match='finite XY pair'):
        media.task_cameras(center)


class ClothView:
    def __init__(self):
        self.q = np.zeros((1, 4, 3), dtype=np.float32)
        self.qd = self.q.copy()

    def get_simulation_nodal_positions(self):
        return self.q

    def get_simulation_nodal_velocities(self):
        return self.qd


@pytest.fixture
def native():
    return SimpleNamespace(current_time=.125, current_time_step_index=250), {
        'towel': ClothView(), 'towel_base': ClothView()}


@pytest.mark.parametrize('change', ['q', 'qd', 'other_cloth', 'time', 'step', 'signed_zero'])
def test_render_guard_rejects_every_native_change(native, change):
    world, views = native
    with pytest.raises(RuntimeError, match='changed native'):
        with media.unchanged_native_state(world, views):
            if change == 'q':
                views['towel'].q[0, 0, 0] = 1.e-8
            elif change == 'qd':
                views['towel'].qd[0, 0, 0] = 1.e-8
            elif change == 'other_cloth':
                views['towel_base'].q[0, 0, 0] = 1.e-8
            elif change == 'time':
                world.current_time += 1.e-8
            elif change == 'step':
                world.current_time_step_index += 1
            else:
                views['towel'].q[0, 0, 0] = -0.


def test_native_snapshot_is_copied_and_hash_binds_velocities(native):
    world, views = native
    before = media.native_state(world, views)
    views['towel'].qd[0, 0, 2] = .1
    assert before['towel:qd'][0, 0, 2] == 0
    assert media.state_digest(before) != media.state_digest(media.native_state(world, views))


def test_guard_checks_physics_even_when_renderer_raises(native):
    world, views = native
    with pytest.raises(RuntimeError, match='changed native'):
        with media.unchanged_native_state(world, views):
            world.current_time_step_index += 1
            raise ValueError('Renderer failed after an unintended physics step')


def test_media_directory_rejects_external_and_symlink_destinations(tmp_path, monkeypatch):
    root = tmp_path / 'media'
    root.mkdir()
    outside = tmp_path / 'private'
    outside.mkdir()
    (root / 'escape').symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(media, 'MEDIA_ROOT', root)
    assert media.media_directory(root / 'run01') == root / 'run01'
    for path in (root, outside, root / 'escape' / 'run01', root / '..' / 'private'):
        with pytest.raises(ValueError, match='subdirectory'):
            media.media_directory(path)


def test_capture_does_not_publish_an_image_after_state_mutation(tmp_path, native, monkeypatch):
    from bench.simulation.isaacsim_backend import render_sync

    world, views = native
    capture = media.LaundryClothFrameCapture.__new__(media.LaundryClothFrameCapture)
    capture.output, capture.world, capture.views = tmp_path, world, views
    capture.closed, capture.cameras, capture.settings = False, {'task': {}}, None
    capture.visuals = {name: SimpleNamespace(sync=lambda q: None) for name in views}
    capture.frames = []
    monkeypatch.setattr(render_sync, 'verify_settings', lambda settings: {})

    def bad_render(name):
        views['towel'].q[0, 0, 0] = .001
        return np.zeros((960, 1280, 3), dtype=np.uint8), {}

    capture._render = bad_render
    with pytest.raises(RuntimeError, match='changed native'):
        capture.capture('final')
    assert capture.frames == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('tag', ['../final', 'a/b', '', 'bad name', 'a' * 97])
def test_frame_names_cannot_escape_media_directory(tag):
    with pytest.raises(ValueError):
        media._name(tag)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='Video tools unavailable')
def test_video_preserves_irregular_native_timestamps_and_rejects_changed_png(tmp_path):
    from PIL import Image

    # Tiny synthetic CPU fixtures verify encoding only; they are never published
    # in the project media tree or described as native RTX evidence.
    capture = media.LaundryClothFrameCapture.__new__(media.LaundryClothFrameCapture)
    capture.output, capture.frames = tmp_path, []
    for index, timestamp in enumerate([0., .125, .375]):
        path = tmp_path / f'{index}.png'
        Image.new('RGB', (16, 16), (index * 80, 40, 50)).save(path)
        capture.frames.append({'native_time_s': timestamp, 'images': {'task': {
            'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}})
    result = capture.encode_video(fps=8.)
    probe = subprocess.run([shutil.which('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', result['path']],
        check=True, capture_output=True, text=True)
    times = [float(row['best_effort_timestamp_time']) for row in json.loads(probe.stdout)['frames']]
    assert times == pytest.approx([0., .125, .25, .375], abs=1.e-6)
    assert result['video_source_indices'] == [0, 1, 1, 2]
    Path(result['path']).unlink()
    Path(capture.frames[0]['images']['task']['path']).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='changed before video encoding'):
        capture.encode_video(fps=8.)
