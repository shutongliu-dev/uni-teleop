"""Candidate native worker integration; all evidence stays in private storage.

This module is not installed in the running Agent services. Build the observer
after reset and before accepting actions; close it only on terminal cleanup.
"""
from pathlib import Path
import json

import numpy as np

from bench.evaluation.cloth.cloth_robot_collision import collect_colliders, state_digest
from bench.evaluation.physics import _task_pieces
from bench.simulation.isaacsim_backend.camera_calibration import calibrate_camera
from .geometry_guard import ClothPhysicsObserver, GeometryModel


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def build_cloth_observer(env, directory):
    """Prepare both fixed public cameras, validate geometry, then attach once.

    Camera prims/render products must exist before the whole-stage inventory.
    Rendering is checked for physical-state invariance, as in the public worker.
    No reset, preload, gravity offset or reference trajectory is introduced.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if env.scene_spec.get('cloth', {}).get('enabled') is not True:
        raise ValueError('Cloth observer requires an explicitly enabled cloth scene')
    dt = float(env.world.get_physics_dt())
    if not np.isfinite(dt) or abs(dt - .0005) > 1e-10:
        raise ValueError('This version requires the frozen 0.5 ms cloth physics profile')
    before = state_digest(env.capture_state())
    for camera in ('overview', 'task'):
        env.render(camera=camera, width=640, height=480)
        calibrate_camera(env.stage, camera, 640, 480)
        env.render_rgbd(camera=camera, width=640, height=480)
    after = state_digest(env.capture_state())
    if before != after:
        raise RuntimeError('Preparing Agent cameras changed physical state')
    meta = env.metadata()
    collision = meta['scene_info']['robot_self_collision']
    if collision.get('enabled') is not True or collision.get('cross_finger_exclusions') != []:
        raise ValueError('Agent cloth qualification requires enabled self collision without cross-finger exclusions')
    export = collect_colliders(env)
    pieces, assets, omitted = _task_pieces({'scene': env.scene_spec, 'metadata': meta})
    if omitted or any('planes' not in piece for piece in pieces):
        raise ValueError('All task solids require verified supported collision geometry')
    state = env.capture_state()
    if state_digest(state) != before:
        raise RuntimeError('Reading native cloth geometry changed physical state')
    model = GeometryModel(export, meta, env.triangles, state['particle_q'], pieces)
    dump(directory / 'colliders.json', export)
    dump(directory / 'task_geometry.json', {'pieces': jsonable(pieces), 'asset_input_sha256': assets})
    dump(directory / 'setup.json', {'schema': 'cloth_agent_physics_observer_setup_v1',
         'camera_setup_preserved_physics': True, 'initial_state_sha256': before,
         'robot_self_collision': collision, 'controller_reference_consumed': False,
         'state_preload_or_reset_applied': False, 'physical_steps_added': 0,
         'agent_feedback_contains_guard_evidence': False,
         'limitations': ['Cooked cloth/robot envelopes and reported rigid contacts only; no full robot self-mesh certificate.',
                        'The first bad integration state can only be retained and stopped after it occurs.']})
    observer = ClothPhysicsObserver(env, model, directory / 'substeps')
    try:
        observer.attach()
    except BaseException as exc:
        try:
            observer.close()
        except BaseException as sealing:
            dump(directory / 'setup_seal_error.json', {'error_type': type(sealing).__name__})
        dump(directory / 'setup_failure.json', {'error_type': type(exc).__name__})
        raise
    return observer


def seal_observer(observer):
    """Called from worker finally; never resume this episode after cleanup."""
    if observer is None:
        return None
    result = observer.close()
    if result['status'] != 'passed':
        raise RuntimeError('Cloth substep physics evidence failed')
    return result
