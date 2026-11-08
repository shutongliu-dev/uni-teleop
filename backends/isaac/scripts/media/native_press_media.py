"""Native RTX pictures at existing compression-probe phase boundaries.

This is called by the simulation owner; it never steps/resets physics or sets
object state. It adds only diagnostic cameras and lights, reuses the canonical
RTX capture/calibration path, and checks every World.render against full native
state. Pictures describe the real test platen/proxy table, not a robot task.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from bench.paths import OUTPUT_ROOT
from scripts.validation.deformable_visual_probe import _np

PHASES = ('settle', 'loaded_hold', 'recover')


def _state(world, volume, pad, extra_rigid_views=None):
    result = {'step': int(world.current_time_step_index), 'time': float(world.current_time),
        'foam_q': _np(volume.get_simulation_nodal_positions()).copy(),
        'foam_qd': _np(volume.get_simulation_nodal_velocities()).copy(),
        'pad_q': _np(pad.get_transforms()).copy(), 'pad_qd': _np(pad.get_velocities()).copy()}
    for name, view in sorted((extra_rigid_views or {}).items()):
        if not isinstance(name, str) or not name or not name.replace('_', '').isalnum():
            raise ValueError('Extra rigid views require simple named identities')
        result['extra_'+name+'_q'] = _np(view.get_transforms()).copy()
        result['extra_'+name+'_qd'] = _np(view.get_velocities()).copy()
    return result


def _assert_same(before, after):
    if set(before) != set(after):
        raise RuntimeError('Presentation changed its physical state inventory')
    if before['step'] != after['step'] or before['time'] != after['time']:
        raise RuntimeError('Presentation advanced the native physics clock')
    for name in set(before)-{'step', 'time'}:
        if not np.array_equal(before[name], after[name]):
            raise RuntimeError('Presentation changed native physical state: '+name)


def _state_sha256(state):
    result = {}
    for name in sorted(set(state)-{'step', 'time'}):
        a = np.ascontiguousarray(state[name])
        header = json.dumps({'shape': a.shape, 'dtype': a.dtype.str}, sort_keys=True).encode()
        result[name] = hashlib.sha256(header+b'\n'+a.tobytes()).hexdigest()
    return result


class _RenderOnlyAdapter:
    """The existing capture_image loop calls update(), mapped to World.render."""
    def __init__(self, world, volume, pad, reference, extra_rigid_views=None):
        self.world, self.volume, self.pad, self.reference = world, volume, pad, reference
        self.extra_rigid_views = dict(extra_rigid_views or {})
        self.render_calls = 0

    def update(self):
        self.world.render()
        self.render_calls += 1
        _assert_same(self.reference, _state(self.world, self.volume, self.pad, self.extra_rigid_views))


def press_cameras(pad_pose_xyzw):
    """Freeze two world cameras from the first phase; later phases never refit."""
    pose = np.asarray(pad_pose_xyzw, dtype=float)
    rotation = Rotation.from_quat(pose[3:])
    def point(local):
        return (rotation.apply(local)+pose[:3]).tolist()
    shared = {'focal_length_mm': 55., 'horizontal_aperture_mm': 36.,
        'vertical_aperture_mm': 27., 'resolution': [1920, 1440], 'clipping_m': [.01, 4.]}
    return {
        'side': {**shared, 'eye': point([0., -.32, .04]), 'target': point([0., 0., .025])},
        'macro': {**shared, 'eye': point([.18, -.23, .092]), 'target': point([0., -.014, .022])},
    }


class NativePressMedia:
    """Usage: create after reset, sample at settle/loaded_hold/recover ends.

    SimulationApp must be started with render_sync.startup_arguments(). Every
    phase retains two raw native PNGs, actual sample count, RTX camera_params,
    state fingerprints and a receipt even when a capture fails. No asset
    material, visual geometry, collider, force or solver parameter is changed.
    """
    def __init__(self, stage, foam, output):
        import carb
        from pxr import UsdLux
        from bench.isaacsim.release import digest
        from bench.simulation.isaacsim_backend.render_sync import verify_settings

        self.stage, self.foam, self.output = stage, deepcopy(foam), Path(output).resolve()
        if not self.output.is_relative_to(OUTPUT_ROOT):
            raise ValueError('Native presentation images belong under demo/interface')
        self.output.mkdir(parents=True, exist_ok=False)
        settings = carb.settings.get_settings()
        self.sync = verify_settings(settings)
        self.mode_before = settings.get('/rtx/rendermode')
        settings.set_string('/rtx/rendermode', 'PathTracing')
        self.mode = settings.get('/rtx/rendermode')
        if self.mode != 'PathTracing':
            raise RuntimeError('Native press media requires RTX PathTracing')
        self.args = SimpleNamespace(width=1920, height=1440, samples=64, render_updates=240)
        self.cameras = None
        self.seen = set()
        # A bare physical platen scene has no task-room illumination. These
        # diagnostic lights are new prims and do not replace asset materials.
        light_root = '/World/NativePressMediaLights'
        if stage.GetPrimAtPath(light_root):
            raise ValueError('Native press media lights already exist')
        stage.DefinePrim(light_root, 'Scope')
        dome = UsdLux.DomeLight.Define(stage, light_root+'/Dome')
        dome.CreateIntensityAttr(700.)
        from pxr import Gf, UsdGeom
        dome.CreateColorAttr(Gf.Vec3f(.93, .96, 1.))
        key = UsdLux.DistantLight.Define(stage, light_root+'/Key')
        key.CreateIntensityAttr(1600.)
        key.CreateAngleAttr(16.)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35., -20., -25.))
        modules = [Path(__file__), Path(__file__).parents[2]/'bench/simulation/isaacsim_backend/render_sync.py',
                   Path(__file__).with_name('render_task_gallery.py')]
        self.provenance = {'module_sources_sha256': {str(p): digest(p) for p in modules},
            'source_root': str(Path(__file__).resolve().parents[2]),
            'cwd': str(Path.cwd()), 'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'volume_metadata_sha256': hashlib.sha256(json.dumps(foam, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'referenced_stage_layers_sha256': {layer.realPath: digest(layer.realPath)
                for layer in stage.GetUsedLayers() if not layer.anonymous and layer.realPath},
            'render_sync': self.sync, 'renderer_mode_before': self.mode_before,
            'renderer_mode': self.mode, 'diagnostic_light_root': light_root,
            'asset_geometry_or_material_modified': False, 'physics_state_setters': 0}

    def sample(self, label, world, volume, pad, *, extra_rigid_views=None):
        from bench.isaacsim.release import dump
        from scripts.media.render_task_gallery import capture_image

        if label not in PHASES:
            raise ValueError('Native press media is limited to the three declared phases')
        if label in self.seen:
            raise ValueError('Never overwrite a native press phase')
        self.seen.add(label)
        output = self.output/label
        output.mkdir(exist_ok=False)
        extra = dict(extra_rigid_views or {})
        reference = _state(world, volume, pad, extra)
        if not all(np.isfinite(reference[k]).all() for k in set(reference)-{'step', 'time'}):
            raise ValueError('Cannot present nonfinite native state')
        if self.cameras is None:
            if label != 'settle':
                raise ValueError('Capture settle first to freeze the camera poses')
            self.cameras = press_cameras(reference['pad_q'][0])
        adapter = _RenderOnlyAdapter(world, volume, pad, reference, extra)
        record = {'schema': 'native_bonded_sponge_press_media.v1', 'status': 'capturing',
            'task_id': 'surface_wiping', 'phase': label, 'native_step': reference['step'],
            'native_time_s': reference['time'], 'camera_lock_phase': 'settle',
            'physics_steps_during_capture': 0, 'model_calls': 0,
            'state_before_sha256': _state_sha256(reference), 'images': {},
            'extra_rigid_views': {name: list(view.prim_paths) for name, view in extra.items()},
            'provenance': self.provenance,
            'scope': 'Actual native platen/proxy-table diagnostic state; no robot manipulation or calibrated foam claim.'}
        try:
            for name, definition in self.cameras.items():
                frame = {'task_id': record['task_id']}
                capture_image(self.stage, adapter, 'sponge_press_'+name, deepcopy(definition),
                              output, self.args, frame)
                record['images'].update(frame['images'])
            _assert_same(reference, _state(world, volume, pad, extra))
            record.update(status='captured', state_after_sha256=_state_sha256(_state(world, volume, pad, extra)),
                world_render_calls=adapter.render_calls, physical_state_bitwise_unchanged=True)
            return record
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                          world_render_calls=adapter.render_calls)
            raise
        finally:
            dump(output/'provenance.json', record)
