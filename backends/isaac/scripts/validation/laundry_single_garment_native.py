"""Native single-garment diagnostics with the actual dual robot installation.

This diagnostic owns its explicit engineering boundary. It does not promote
ordinary robot episodes or rewrite free cloth nodes. Input/source identities,
all native states and real RTX acquisition are recorded separately from scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from bench.paths import DATA_ROOT, load_json

PROJECT = Path(__file__).resolve().parents[2]
DATASETS = Path(f'{DATA_ROOT}/datasets')


def pin(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            sha.update(block)
    return {'path': str(path), 'sha256': sha.hexdigest()}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


class NativeArrays:
    """Bounded-memory, lossless native readback; partial runs stay partial."""
    def __init__(self, output, samples):
        self.output, self.samples, self.count = output, samples, 0
        self.arrays = {}
        output.mkdir()

    def append(self, state):
        if self.count >= self.samples:
            raise ValueError('Native stream overflow')
        if self.count and set(state) != set(self.arrays):
            raise ValueError('Native stream fields changed')
        for name, value in state.items():
            data = np.asarray(value)
            if not np.isfinite(data).all():
                raise ValueError('Nonfinite native state: '+name)
            if name not in self.arrays:
                if not name.replace('_', '').isalnum():
                    raise ValueError('Unsafe native field name')
                self.arrays[name] = np.lib.format.open_memmap(
                    self.output/(name+'.npy'), mode='w+', dtype=data.dtype,
                    shape=(self.samples, *data.shape))
            target = self.arrays[name]
            if target.shape[1:] != data.shape or target.dtype != data.dtype:
                raise ValueError('Native state topology/dtype changed: '+name)
            target[self.count] = data
        self.count += 1

    def close(self):
        for value in self.arrays.values():
            value.flush()
        return {'schema': 'laundry_native_npy_stream.v1', 'valid_samples': self.count,
            'allocated_samples': self.samples, 'complete': self.count == self.samples,
            'files': {name: {**pin(self.output/(name+'.npy')), 'dtype': str(a.dtype),
                            'shape': list(a.shape)} for name, a in self.arrays.items()},
            'scope': 'Raw native arrays; only valid_samples are observations, never padded completion.'}


def advance(env):
    """Use the same native gravity feedforward and position drives as env.step."""
    if getattr(env, '_arm_gravity_mask', None) is not None:
        gravity = env._robot_view.get_gravity_compensation_forces()
        gravity = gravity[:, gravity.shape[1]-env._arm_gravity_mask.shape[1]:]
        env._robot_view.set_dof_actuation_forces((gravity*env._arm_gravity_mask).contiguous(), env._one)
    env.world.step(render=False)
    env.time_s += env.dt


def robot_readback(env):
    from bench.simulation.isaacsim_backend.env import _np
    return {'joint_q': _np(env._robot_view.get_dof_positions())[0],
            'joint_qd': _np(env._robot_view.get_dof_velocities())[0],
            'robot_body_q': _np(env._robot_view.get_link_transforms())[0],
            'robot_body_qd': _np(env._robot_view.get_link_velocities())[0]}


def run(args):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    source = verify_if_snapshot(PROJECT, required_modules=(
        'scripts.validation.laundry_single_garment_native',), entrypoint_file=Path(__file__))
    if source is None:
        raise ValueError('Run diagnostics from a complete frozen source snapshot')
    scene = load_json(args.scene)
    initial = load_json(args.initial)
    if scene.get('robot', {}).get('hand_id') != 'dual_fr3_sharpa_w01' or len(scene.get('cloths', [])) != 1:
        raise ValueError('Exactly one garment and the actual dual FR3/SHARPA installation required')
    if scene.get('named_cloth_initialization') != 'author-only':
        raise ValueError('Single-garment diagnostics preserve native authored initialization')
    dt = float(scene['physics']['dt_s'])
    steps = round(args.duration/dt)
    if not math.isfinite(args.duration) or args.duration <= 0 or abs(steps*dt-args.duration) > 1.e-9:
        raise ValueError('Duration must contain a positive integer count of native steps')
    output = args.output.resolve()
    if not output.is_relative_to(DATASETS) or output.exists():
        raise ValueError('Use a fresh shared dataset output')
    output.mkdir(parents=True)
    report = {'schema': 'laundry_single_garment_native.v1', 'complete': False, 'mode': args.mode,
        'source': source, 'source_manifest': pin(PROJECT/'source_snapshot_manifest.json'),
        'scene_input': pin(args.scene), 'initial_input': pin(args.initial),
        'robot_present': True, 'robot_control': 'native joint position drives with original gravity feedforward',
        'cloth_position_velocity_writes': 0, 'model_calls': 0,
        'task_success_claimed': False, 'robot_grasp_qualified': False, 'physics_accepted': False,
        'scope': 'Explicit native engineering diagnostic, not an ordinary episode or autonomous policy result.'}
    env = capture = writer = None
    exit_code = 1
    try:
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv, _np
        env = FR3DexterousIsaacEnv(scene)
        env.reset(initial_state=initial)
        if not math.isclose(env.dt, dt, rel_tol=0, abs_tol=1.e-12):
            raise ValueError('Native and declared time steps disagree')
        report['native_initialization'] = env.named_cloth_initialization_receipt
        report['joint_names'], report['body_names'] = env.joint_names, env.body_names
        env.stage.Export(str(output/'initial.usda'))
        writer = NativeArrays(output/'states', steps+1)
        start_step, start_clock = int(env.world.current_time_step_index), float(env.world.current_time)
        cloth_id = scene['cloths'][0]['id']
        view = env._cloth_views[cloth_id]
        mask = np.zeros(len(_np(view.get_simulation_nodal_positions())[0]), dtype=np.uint8)
        def state(step):
            return {**robot_readback(env), 'cloth_q': _np(view.get_simulation_nodal_positions())[0],
                'cloth_qd': _np(view.get_simulation_nodal_velocities())[0], 'control_mask': mask.copy(),
                'time_s': np.asarray(step*dt), 'native_step': np.asarray(step, dtype=np.int64),
                'world_step': np.asarray(env.world.current_time_step_index, dtype=np.int64),
                'world_time_s': np.asarray(env.world.current_time)}
        writer.append(state(0))
        if args.media_output:
            from scripts.media.laundry_cloth_frames import LaundryClothFrameCapture
            capture = LaundryClothFrameCapture(stage=env.stage, world=env.world, app=env.app,
                cloth_views=env._cloth_views, cloth_specs=scene['cloths'], triangles=env.cloth_triangles,
                cloth_paths={k: v['path'] for k,v in env.scene_info['cloths'].items()},
                output=args.media_output, cameras={'task': scene['cameras']['overview']})
        def photograph(tag):
            before = robot_readback(env)
            capture.capture(tag, camera_ids=('task',))
            after = robot_readback(env)
            if any(not np.array_equal(before[k], after[k]) for k in before):
                raise RuntimeError('Rendering changed the native robot state')
        if capture:
            photograph('step_000000')
        stride = round(1/(args.media_fps*dt))
        last_print = time.monotonic()
        for step in range(1, steps+1):
            advance(env)
            writer.append(state(step))
            if capture and step % stride == 0:
                photograph(f'step_{step:06d}')
            if time.monotonic()-last_print >= 20 or step == steps:
                print(json.dumps({'event': 'native_progress', 'step': step, 'time_s': step*dt,
                    'target_time_s': args.duration}), flush=True)
                last_print = time.monotonic()
        if int(env.world.current_time_step_index)-start_step != steps:
            raise RuntimeError('Native step clock disagrees with recording')
        if abs(float(env.world.current_time)-start_clock-args.duration) > 2.e-6:
            raise RuntimeError('Native time disagrees with recording')
        if capture:
            photograph('final')
            report['video'] = capture.encode_video(camera_id='task', fps=args.media_fps)
        env.stage.Export(str(output/'final.usda'))
        report.update(complete=True, native_steps=steps, duration_s=steps*dt,
            cloth_id=cloth_id, initialization_is_operation_credit=False)
        exit_code = 0
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        if writer is not None:
            report['states'] = writer.close()
        if capture is not None:
            capture.close()
        dump(output/'native_report.json', report)
        print(json.dumps({'event': 'native_finished', 'complete': report['complete'], 'output': str(output)}), flush=True)
        if env is not None:
            if exit_code:
                try:
                    raise RuntimeError('Native diagnostic failed; inspect '+str(output/'native_report.json'))
                finally:
                    env.close()
            env.close()
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--mode', choices=('settle',), default='settle')
    parser.add_argument('--duration', type=float, default=3.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media-output', type=Path)
    parser.add_argument('--media-fps', type=int, choices=(2,4,8), default=4)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
