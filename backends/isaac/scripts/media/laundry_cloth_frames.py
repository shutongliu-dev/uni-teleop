"""Actual RTX frames of the current engineering cloth state, without stepping.

Call after a completed native physics step. Supply render_sync.startup_arguments()
to SimulationApp before creating the renderer. This helper only creates cameras,
render products and nonphysical cloth skins; it never sets native cloth state.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from bench.paths import OUTPUT_ROOT


MEDIA_ROOT = Path(f'{OUTPUT_ROOT}/media/laundry_tasks_20261021')
CAMERAS = {
    'room': {'eye': [-.75, -2.10, 1.72], 'target': [.15, .65, 1.05], 'focal_length_mm': 17.5},
    # Keep the eye inside the furnished room; the front wall is at y=-2.3 m.
    'task': {'eye': [2.20, -1.95, 2.30], 'target': [.65, -.55, .97], 'focal_length_mm': 32.},
    'thickness': {'eye': [1.50, -1.42, 1.12], 'target': [.65, -.55, .91], 'focal_length_mm': 50.},
}


def task_cameras(target_center_xy_m):
    """Center only the thickness close-up on the frozen task's target."""
    center = np.asarray(target_center_xy_m, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Task camera target center must be a finite XY pair')
    offset = center - np.asarray([.65, -.55])
    cameras = deepcopy(CAMERAS)
    for key in ('eye', 'target'):
        cameras['thickness'][key][:2] = (
            np.asarray(cameras['thickness'][key][:2]) + offset).tolist()
    return cameras


def _name(value):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]{0,95}', value) is None:
        raise ValueError('Frame and camera names must be short filesystem-safe identifiers')
    return value


def media_directory(path):
    """Constrain presentation artifacts, including symlinks, to the media tree."""
    root, path = MEDIA_ROOT.resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'Laundry media must use a subdirectory of {MEDIA_ROOT}')
    return path


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def native_state(world, cloth_views):
    """Snapshot exact native arrays and clocks, not a presentation transform."""
    state = {'native_time_s': np.asarray([world.current_time], dtype=np.float64),
             'native_step': np.asarray([world.current_time_step_index], dtype=np.int64)}
    if not np.isfinite(state['native_time_s']).all() or state['native_step'][0] < 0:
        raise RuntimeError('Native simulation clock must be finite and nonnegative')
    for name, view in cloth_views.items():
        for label, getter in (('q', view.get_simulation_nodal_positions),
                              ('qd', view.get_simulation_nodal_velocities)):
            value = _array(getter())
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 3 or not np.isfinite(value).all():
                raise RuntimeError('Each named cloth must have one complete finite native surface')
            state[name + ':' + label] = value
    return state


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = np.asarray(state[name])
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def assert_unchanged(before, after):
    if before.keys() != after.keys() or any(
        before[key].dtype != after[key].dtype or before[key].shape != after[key].shape
        or before[key].tobytes() != after[key].tobytes() for key in before
    ):
        raise RuntimeError('RTX capture changed native cloth state, physics step or simulation time')


@contextmanager
def unchanged_native_state(world, cloth_views):
    before = native_state(world, cloth_views)
    try:
        yield before
    finally:
        assert_unchanged(before, native_state(world, cloth_views))


class LaundryClothFrameCapture:
    """Reusable render products; capture(tag) publishes only verified native frames."""

    def __init__(self, *, stage, world, cloth_views, cloth_specs, triangles, output,
                 cloth_paths=None, cameras=None, width=1280, height=960,
                 render_updates=8, warmup_updates=24, max_updates=120, app=None):
        from bench.simulation.isaacsim_backend.cloth_visuals import ClosedClothVisual
        from bench.simulation.isaacsim_backend.geometry import cloth_grid
        from bench.simulation.isaacsim_backend.render_sync import verify_settings
        import carb

        del app  # World.render owns the nonphysics Kit update.
        self.output = media_directory(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError('Use a fresh laundry media directory')
        self.stage, self.world, self.views = stage, world, dict(cloth_views)
        self.width, self.height = int(width), int(height)
        if self.width < 1280 or self.height < 960 or self.width % 2 or self.height % 2:
            raise ValueError('Laundry frames require even dimensions of at least 1280 by 960')
        self.render_updates, self.warmup_updates, self.max_updates = map(
            int, (render_updates, warmup_updates, max_updates))
        if not 1 <= self.render_updates <= self.warmup_updates <= self.max_updates:
            raise ValueError('Invalid bounded RTX warmup/update counts')
        specs = (dict(cloth_specs) if isinstance(cloth_specs, dict)
                 else {row['id']: row for row in cloth_specs})
        if not self.views or set(specs) != set(self.views) or set(triangles) != set(self.views):
            raise ValueError('Exact named cloth specs, topologies and native views required')
        if cloth_paths is not None and set(cloth_paths) != set(self.views):
            raise ValueError('Exact named native cloth prim paths required')
        self.cameras = deepcopy(CAMERAS if cameras is None else cameras)
        for name in self.cameras:
            _name(name)
        self.settings = carb.settings.get_settings()
        self.render_sync = verify_settings(self.settings)
        self.products, self.visuals, self.frames = {}, {}, []
        self.closed = False
        self.output.mkdir(parents=True, exist_ok=True)
        with unchanged_native_state(self.world, self.views) as before:
            for name, spec in specs.items():
                path = (cloth_paths or {}).get(name, '/World/Cloths/' + name)
                if not stage.GetPrimAtPath(path):
                    raise ValueError('Missing native cloth prim: ' + path)
                rest, _ = cloth_grid(spec)
                options = spec.get('isaacsim', {})
                thickness = spec.get('visual_thickness_m', options.get('cloth_thickness_m'))
                if thickness is None:
                    raise ValueError('Declare physical/visual cloth thickness before rendering')
                color = options.get('cloth_visual_color', [.72, .77, .66])
                if 'visual_profile' in spec:
                    from bench.simulation.isaacsim_backend.fabric_visuals import (
                        FabricClothVisual, load_visual_profile)
                    self.visuals[name] = FabricClothVisual(stage, path, name, triangles[name], rest,
                        thickness, load_visual_profile(spec['visual_profile']),
                        before[name + ':q'][0], dye_linear_rgb=color)
                    continue
                self.visuals[name] = ClosedClothVisual(stage, path, name, triangles[name], rest,
                    thickness, color,
                    texture_paths=spec.get('visual_texture_paths'),
                    initial_world_vertices=before[name + ':q'][0],
                    texture_tile_size_m=spec.get('visual_texture_tile_size_m'),
                    **({'visual_face_groups': spec['visual_face_groups']}
                       if 'visual_face_groups' in spec else {}))

    def _product(self, name):
        import omni.hydratexture
        import omni.replicator.core as rep
        from carb.eventdispatcher import get_eventdispatcher
        from pxr import Gf, UsdGeom
        from scripts.media.render_workroom import create_camera

        if name in self.products:
            return self.products[name], False
        definition = self.cameras[name]
        path = create_camera(self.stage, 'laundry_media_' + name, definition)
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
        camera.CreateVerticalApertureAttr(36. * self.height / self.width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(*definition.get('clipping_m', [.01, 8.])))
        product = rep.create.render_product(path, (self.width, self.height))
        annotators = {key: rep.AnnotatorRegistry.get_annotator(key) for key in ('rgb', 'camera_params')}
        for annotator in annotators.values():
            annotator.attach([product])
        row = {'product': product, 'annotators': annotators, 'events': 0, 'info': {}}

        def on_frame(event):
            info = product.hydra_texture.get_frame_info(event['result_handle'])
            row['events'] += 1
            row['info'] = {key: info[key] for key in ('progression', 'frame_number', 'status') if key in info}

        row['subscription'] = get_eventdispatcher().observe_event(
            observer_name='laundry_media_' + name,
            event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
            on_event=on_frame, filter=product.hydra_texture.get_event_key())
        self.products[name] = row
        return row, True

    def _render(self, name):
        import omni.usd
        from bench.simulation.isaacsim_backend.render_sync import camera_calibration

        row, fresh = self._product(name)
        start_events = row['events']
        minimum = self.warmup_updates if fresh else self.render_updates
        for update in range(self.max_updates):
            # Check every render, so an intermediate step cannot be hidden by a
            # later reset or by clocks returning to their original values.
            with unchanged_native_state(self.world, self.views):
                self.world.render()
            pending = int(omni.usd.get_context().get_stage_loading_status()[2])
            rgb = np.asarray(row['annotators']['rgb'].get_data())
            if (update + 1 >= minimum and pending == 0 and row['events'] > start_events
                    and rgb.shape in ((self.height, self.width, 3), (self.height, self.width, 4))):
                break
        else:
            raise RuntimeError('RTX did not produce a fresh, complete laundry frame within the update bound')
        rgb = rgb[:, :, :3].astype(np.uint8, copy=True)
        if float(rgb.std()) < 2.:
            raise RuntimeError('RTX returned a nearly blank laundry frame')
        calibration = camera_calibration(row['annotators']['camera_params'].get_data(),
                                         width=self.width, height=self.height)
        return rgb, {'camera': deepcopy(self.cameras[name]), 'camera_calibration': calibration,
                     'render_updates': update + 1, 'pending_stage_loads': pending,
                     'new_drawable_events': row['events'] - start_events,
                     'renderer_mode': self.settings.get('/rtx/rendermode')}

    def capture(self, tag, *, camera_ids=('task',)):
        """Save current pixels. Use all three camera IDs for initial/final stills."""
        from PIL import Image
        from bench.simulation.isaacsim_backend.render_sync import verify_settings

        if self.closed:
            raise RuntimeError('Laundry capture is closed')
        tag = _name(tag)
        camera_ids = tuple(camera_ids)
        if not camera_ids or len(set(camera_ids)) != len(camera_ids) or not set(camera_ids) <= set(self.cameras):
            raise ValueError('Select unique configured laundry cameras')
        if any((self.output / name / (tag + '.png')).exists() for name in camera_ids):
            raise FileExistsError('A native frame with this tag already exists')
        self.render_sync = verify_settings(self.settings)
        rendered = {}
        with unchanged_native_state(self.world, self.views) as before:
            for name, visual in self.visuals.items():
                visual.sync(before[name + ':q'][0])
            for name in camera_ids:
                rendered[name] = self._render(name)
        # No image is published if any camera changed physics or failed.
        record = {'tag': tag, 'native_time_s': float(before['native_time_s'][0]),
                  'native_step': int(before['native_step'][0]),
                  'native_state_sha256': state_digest(before), 'native_rtx': True,
                  'capture_state_unchanged': True, 'images': {}}
        for name, (rgb, metadata) in rendered.items():
            directory = self.output / name
            directory.mkdir(exist_ok=True)
            path = directory / (tag + '.png')
            Image.fromarray(rgb).save(path)
            record['images'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'size_px': [self.width, self.height], **metadata}
        self.frames.append(record)
        self._manifest()
        return deepcopy(record)

    def _manifest(self):
        record = {'schema': 'laundry_native_rtx_frames.v1', 'frames': self.frames,
                  'render_synchronization': self.render_sync,
                  'cloth_visuals': {name: visual.metadata for name, visual in self.visuals.items()},
                  'cloth_state_writes': False, 'physics_steps_during_capture': 0,
                  'task_success_claimed_by_media': False}
        temporary = self.output / 'frames.json.tmp'
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(self.output / 'frames.json')

    def encode_video(self, *, camera_id='task', fps=8.):
        """Sample-and-hold recorded PNGs on the video clock; never interpolate state."""
        from PIL import Image

        camera_id = _name(camera_id)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Positive finite video frame rate required')
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('ffmpeg is unavailable; original RTX PNGs remain usable')
        selected = []
        for frame in self.frames:
            if camera_id not in frame['images']:
                continue
            if selected and frame['native_time_s'] < selected[-1]['native_time_s']:
                raise ValueError('Cannot encode a sequence with reversed native time')
            if selected and frame['native_time_s'] == selected[-1]['native_time_s']:
                selected[-1] = frame
            else:
                selected.append(frame)
        if len(selected) < 2:
            raise ValueError('Video requires at least two distinct native timestamps')
        destination = self.output / (camera_id + '.mp4')
        if destination.exists():
            raise FileExistsError(destination)
        paths = []
        for frame in selected:
            row = frame['images'][camera_id]
            path = Path(row['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError('Recorded RTX image changed before video encoding')
            paths.append(path)
        times = np.asarray([row['native_time_s'] for row in selected])
        relative = times - times[0]
        count = int(np.ceil(relative[-1] * fps - 1.e-9)) + 1
        source_indices = np.searchsorted(relative, np.arange(count) / fps + 1.e-9, side='right') - 1
        with Image.open(paths[0]) as image:
            width, height = image.size
        with (self.output / (camera_id + '_ffmpeg.log')).open('wb') as log:
            command = [executable, '-hide_banner', '-nostdin', '-n', '-f', 'rawvideo',
                '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', str(fps),
                '-i', 'pipe:0', '-c:v', 'libx264', '-threads', '2', '-filter_threads', '1',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            try:
                previous, pixels = None, None
                for index in source_indices:
                    if index != previous:
                        with Image.open(paths[index]) as image:
                            if image.size != (width, height):
                                raise ValueError('Recorded frame dimensions changed during the sequence')
                            pixels = image.convert('RGB').tobytes()
                        previous = index
                    process.stdin.write(pixels)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError('ffmpeg failed; inspect the retained encoding log')
            except BaseException:
                process.kill()
                process.wait()
                raise
        metadata = {'source': 'recorded_native_RTX_PNGs', 'camera_id': camera_id,
                    'source_frames': len(selected), 'video_frames': count, 'fps': fps,
                    'native_time_range_s': [float(times[0]), float(times[-1])],
                    'timing': 'fixed fps; hold most recent captured native state, never show a future state',
                    'maximum_frame_grid_delay_s': 1. / fps,
                    'source_native_times_s': times.tolist(), 'video_source_indices': source_indices.tolist(),
                    'final_hold_s': count / fps - float(relative[-1]), 'interpolation': False,
                    'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
        (self.output / (camera_id + '_video.json')).write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata

    def close(self):
        if self.closed:
            return
        for row in self.products.values():
            for annotator in row['annotators'].values():
                annotator.detach([row['product']])
            row['subscription'] = None
            row['product'].destroy()
        self.products.clear()
        self.closed = True
