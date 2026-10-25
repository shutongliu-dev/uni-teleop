"""Run an actual native Isaac Sim episode; controls contain 29 joint targets only."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from .release import DEFAULT, PROJECT, digest, dump, load_evaluator, task_inputs, validate
from bench.paths import load_json


def generated_action(home, limits, step, count, baseline):
    q = dict(home)
    if baseline == 'joint-response':
        # Smooth small excursion and return, including fingers whose home is a limit.
        fraction = math.sin(math.pi * (step + 1) / count) ** 2
        for name, value in q.items():
            amplitude = .035 if name.startswith('fr3_') else .12
            low, high = limits[name]
            direction = 1 if high - value >= value - low else -1
            amplitude = min(amplitude, max(0., (high - value if direction == 1 else value - low) * .8))
            q[name] = float(value + direction * amplitude * fraction)
    elif baseline != 'hold':
        raise ValueError(baseline)
    return {'joint_positions_rad': q, 'duration_s': .02}


def load_actions(path):
    value = load_json(Path(path))
    if not isinstance(value, list) or not value:
        raise ValueError('Action JSON must contain a nonempty list of full joint-target actions')
    for action in value:
        if not isinstance(action, dict) or set(action) != {'joint_positions_rad', 'duration_s'}:
            raise ValueError('Only joint_positions_rad and duration_s may be supplied in actions')
        if action['duration_s'] != .02 or isinstance(action['duration_s'], bool):
            raise ValueError('Every action must have duration_s = 0.02')
    return value


def _errors(actual, expected):
    result = {}
    for name in expected:
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        result[name] = float(np.max(np.abs(a - b))) if a.size and a.shape == b.shape else 0. if a.shape == b.shape else None
    return result


def _response(states, names):
    q = np.stack([s['joint_q'] for s in states])
    targets = np.stack([s['joint_target_q'] for s in states])
    actual, command = np.max(np.abs(q - q[0]), axis=0), np.max(np.abs(targets - targets[0]), axis=0)
    return {'joint_count': len(names), 'actual_state_frames': len(states),
            'commanded_joints': int(np.sum(command > 1.e-5)),
            'responding_joints': int(np.sum((actual > 1.e-4) & (command > 1.e-5))),
            'all_29_joint_positions_finite': len(names) == 29 and bool(np.isfinite(q).all()),
            'per_joint': {name: {'command_excursion_rad': float(command[i]), 'actual_excursion_rad': float(actual[i]),
                                'final_target_error_rad': float(q[-1, i] - targets[-1, i])} for i, name in enumerate(names)},
            'interpretation': 'Measured response to small joint-target excursions; not a grasping or manipulation policy.'}


def rollout(args):
    output, demo = args.output.resolve(), args.demo.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a fresh episode output: {output}')
    diagnostic = getattr(args, 'diagnostic', None)
    entry, scene, task, initial = task_inputs(args.release, args.task, 'nominal' if diagnostic else args.phase, seed=args.seed)
    if diagnostic:
        if args.phase != 'contact_probe' or set(diagnostic['reset_override']) != {'joint_positions_rad'}:
            raise ValueError('Contact probe permits only a declared robot joint reset override')
        initial = deepcopy(diagnostic['reset_override'])
    if args.phase == 'qualification_probe' and args.actions:
        raise ValueError('Qualification uses its separate hold baseline; do not pool imported controller actions')
    if args.fps <= 0 or 50 % args.fps:
        raise ValueError('fps must be a positive divisor of 50')
    imported = load_actions(args.actions) if args.actions else None
    count = len(imported) if imported else round(args.duration / .02)
    if count < 1 or (not imported and abs(count * .02 - args.duration) > 1.e-8):
        raise ValueError('duration must be a positive multiple of 0.02 s')
    stride = 50 // args.fps
    if not args.no_video and count % stride:
        raise ValueError('Video duration must contain an integer number of frames at selected fps')
    baseline = ('contact-probe-' + diagnostic['mode']) if diagnostic else 'json-joint-actions' if imported else ('hold' if args.phase == 'qualification_probe' else args.baseline)
    output.mkdir(parents=True, exist_ok=True)
    demo.mkdir(parents=True, exist_ok=True)
    dump(output / 'scene.json', scene)
    shutil.copy2(args.release / entry['task'], output / 'task.json')
    dump(output / 'reset_override.json', initial)
    if diagnostic:
        dump(output / 'diagnostic.json', diagnostic)
    release = validate(args.release)
    dump(output / 'release_reference.json', {'path': str(args.release.resolve()), 'manifest_sha256': digest(args.release / 'release_manifest.json'), 'task_entry': entry})
    sources = sorted(set([*(p for p in Path(__file__).parent.glob('*.py') if p.name != 'audit.py'), *(PROJECT / 'bench/simulation/isaacsim_backend').glob('*.py'),
                          *(PROJECT / 'bench/simulation/assets').glob('*.py'),
                          PROJECT / 'scripts/isaac_python.sh', PROJECT / 'bench/apps/fr3_sharpa.headless.kit',
                          PROJECT / 'bench/control/__init__.py', PROJECT / 'bench/control/kinematics.py', PROJECT / 'bench/geometry.py']))
    dump(output / 'source_hashes.json', {str(p): digest(p) for p in sources})
    for source in sources:
        destination = output / 'source_snapshot' / source.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copytree(args.release / 'evaluator', output / 'evaluator')
    runtime_root = (Path(os.environ['ISAAC_SIM_RUNTIME']).parent if 'ISAAC_SIM_RUNTIME' in os.environ
                    else (PROJECT / 'runtime/isaacsim').resolve())
    runtime_sources = [Path(os.environ.get('ISAAC_PYTHON_LAUNCHER', str(runtime_root / 'python_compat.sh'))),
                       runtime_root / 'compat/ubuntu22.04-glibc2.35-v1/provenance.json']
    dump(output / 'environment_hashes.json', {
        'files': {str(p): digest(p) for p in runtime_sources if p.is_file()},
        'missing_files': [str(p) for p in runtime_sources if not p.is_file()],
        'ISAAC_ALLOW_UNVALIDATED_DRIVER': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
        'scope': 'Invocation-local compatibility wrapper and library provenance; bundled runtime is unchanged.'})
    if imported:
        shutil.copy2(args.actions, output / 'submitted_actions.json')
    evaluator = load_evaluator(args.release)(task)
    states, actions, scores, frame_map, contact_counts, contact_impulses = [], [], [], [], [], []
    env = None
    started = time.perf_counter()
    completed = False
    diagnostic_samples = []
    try:
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=args.width, render_height=args.height)
        obs = env.reset(seed=args.seed, initial_state=initial)
        initial_state = env.capture_state()
        if diagnostic:
            from .controllers import ContactTargets, contact_sample, contact_result
            feedback = ContactTargets(initial['joint_positions_rad'], env.joint_limits, diagnostic['integral_gain_s_inv'])
        dump(output / 'metadata.json', env.metadata())
        with (output / 'observations.jsonl').open('w') as stream:
            for index in range(count + 1):
                if index:
                    action = deepcopy(imported[index - 1]) if imported else generated_action(env.home, env.joint_limits, index - 1, count, baseline)
                    if diagnostic:
                        action, feedback_record = feedback.action(action, obs)
                    obs = env.step(action)
                    actions.append(action)
                state = env.capture_state()
                if diagnostic:
                    diagnostic_samples.append({**contact_sample(obs),
                        **(diagnostic['planned_trace'][index - 1] if index else {'phase': 'reset'}),
                        **(feedback_record if index else {})})
                if not obs['metrics']['state_finite'] or any(not np.isfinite(v).all() for v in state.values()):
                    raise RuntimeError(f'Nonfinite state at step {index}')
                score = evaluator.update(obs)
                stream.write(json.dumps(obs, allow_nan=False) + '\n')
                stream.flush()
                states.append(state); scores.append(score)
                contacts = obs.get('contacts', {})
                contact_counts.append(int(contacts.get('rigid_contact_point_count', 0)))
                contact_impulses.append(float(contacts.get('rigid_impulse_magnitude_sum_ns', 0.)))
                if contacts.get('errors'):
                    raise RuntimeError(f"Contact callback errors: {contacts['errors']}")
                if not args.no_video and index < count and index % stride == 0:
                    from PIL import Image
                    from .video import frame_card
                    rgb = env.render(camera='task', width=args.width, height=args.height)
                    overview = env.render(camera='overview', width=args.width, height=args.height)
                    if float(np.std(rgb)) < 1. or float(np.std(overview)) < 1.:
                        raise RuntimeError('RTX returned an effectively constant/blank frame')
                    after_render = env.capture_state()
                    render_delta = _errors(after_render, state)
                    if any(v is None or v > 1.e-7 for v in render_delta.values()):
                        raise RuntimeError(f'Rendering changed physical state: {render_delta}')
                    frame_id = len(frame_map)
                    rawdir, carddir = output / 'frames_raw', output / 'frames'
                    rawdir.mkdir(exist_ok=True); carddir.mkdir(exist_ok=True)
                    rawpath, overviewpath, cardpath = rawdir / f'{frame_id:06d}.png', rawdir / f'{frame_id:06d}_overview.png', carddir / f'{frame_id:06d}.png'
                    Image.fromarray(rgb).save(rawpath); Image.fromarray(overview).save(overviewpath)
                    frame_card(rgb, overview, task, score, obs['time_s'], 'contact_probe_' + diagnostic['mode'] if diagnostic else args.phase).save(cardpath)
                    frame_map.append({'frame': frame_id, 'state_index': index, 'physical_time_s': obs['time_s'],
                                      'main_camera': 'overview' if args.phase == 'nominal' else 'task',
                                      'inset_camera': 'task' if args.phase == 'nominal' else 'overview',
                                      'raw_rgb': str(rawpath), 'raw_sha256': digest(rawpath), 'overview_rgb': str(overviewpath),
                                      'overview_sha256': digest(overviewpath), 'video_frame': str(cardpath), 'video_frame_sha256': digest(cardpath),
                                      'render_state_max_delta': render_delta, 'rgb_std': float(np.std(rgb)),
                                      'overview_rgb_std': float(np.std(overview))})
                if index % 25 == 0:
                    print(json.dumps({'task': task['id'], 'phase': args.phase, 'step': index, 'time_s': obs['time_s'],
                                      'progress': score['progress'], 'rigid_contact_points': contact_counts[-1]}, ensure_ascii=False), flush=True)
        np.savez_compressed(output / 'states.npz', **{key: np.stack([s[key] for s in states]) for key in states[0]})
        np.save(output / 'triangles.npy', env.triangles)
        dump(output / 'actions.json', actions)
        dump(output / 'scores.json', scores)
        dump(output / 'condition_inspection.json', evaluator.inspect_conditions(obs))
        if diagnostic:
            dump(output / 'contact_trace.json', diagnostic_samples)
            diagnostic_result = contact_result(diagnostic_samples, diagnostic['mode'])
            dump(output / 'contact_result.json', diagnostic_result)
        reset_obs = env.reset(seed=args.seed, initial_state=initial)
        replay_states = [env.capture_state()]
        reset_errors = _errors(replay_states[0], initial_state)
        replay_errors = []
        for index, action in enumerate(actions[:5]):
            env.step(action)
            replay_states.append(env.capture_state())
            replay_errors.append(_errors(replay_states[-1], states[index + 1]))
        np.savez_compressed(output / 'reset_replay_states.npz', **{key: np.stack([s[key] for s in replay_states]) for key in replay_states[0]})
        reset_evidence = {'same_seed': args.seed, 'same_reset_override': initial, 'reset_time_s': reset_obs['time_s'],
                          'reset_max_absolute_error': reset_errors, 'replayed_action_count': len(replay_errors),
                          'replay_max_absolute_errors': replay_errors,
                          'initial_restored_within_1e_6': all(v is not None and v <= 1.e-6 for v in reset_errors.values()),
                          'replay_finite': all(np.isfinite(v).all() for s in replay_states for v in s.values()),
                          'note': 'Numerical replay differences reported, not assumed deterministic.'}
        dump(output / 'reset_evidence.json', reset_evidence)
        response = _response(states, env.joint_names)
        dump(output / 'joint_response.json', response)
        metadata = env.metadata()
        result = {'task_id': task['id'], 'title_zh': task['title_zh'], 'phase': args.phase, 'baseline': baseline,
                  'status': 'physical_rollout_completed', 'simulation_duration_s': float(states[-1]['time_s']),
                  'state_count': len(states), 'action_count': len(actions), 'all_states_finite': True,
                  'actual_joint_count': len(env.joint_names), 'responding_joints': response['responding_joints'],
                  'reset_initial_restored': reset_evidence['initial_restored_within_1e_6'],
                  'max_rigid_contact_points': max(contact_counts), 'total_reported_rigid_impulse_ns': sum(contact_impulses),
                  'solver_contact_source': obs['contacts']['source'], 'cloth_contact_reporting': obs['contacts'].get('cloth_contact_reporting'),
                  'evaluator_final': scores[-1], 'max_evaluator_progress': max(s['progress'] for s in scores),
                  'controller_task_success_claimed': False,
                  'qualification_excluded_from_nominal_success': args.phase == 'qualification_probe',
                  'diagnostic_excluded_from_nominal_success': bool(diagnostic),
                  'contact_response': diagnostic_result if diagnostic else None,
                  'joint_targets_only': True, 'object_or_cloth_state_writes_after_reset': False,
                  'material_calibrated': False, 'wall_time_s': time.perf_counter() - started,
                  'native_isaacsim_app': metadata['native_isaacsim_app'], 'physics_engine': metadata['physics_engine']}
        initial_hashes = load_json(output / 'source_hashes.json')
        changed = [path for path, expected in initial_hashes.items() if digest(path) != expected]
        result['source_files_changed_during_run'] = changed
        if changed:
            result['status'] = 'physical_rollout_completed_sources_changed'
        dump(output / 'result.json', result)
        if frame_map:
            from .video import encode_frames
            video_path = demo / f'{args.phase}.mp4'
            video_info = encode_frames(output / 'frames', video_path, args.fps)
            shutil.copy2(output / 'frames/000000.png', demo / f'{args.phase}.png')
            dump(output / 'recording.json', {'frames': frame_map, 'fps': args.fps, 'playback_speed': 1.,
                 'still_holds': [], 'slow_motion_segments': [], 'physics_interpolation': False,
                 'simulation_duration_s': count * .02, 'video': str(video_path), 'video_sha256': digest(video_path), 'ffprobe': video_info})
        dump(output / 'artifact_hashes.json', {str(p.relative_to(output)): digest(p)
             for p in sorted(output.rglob('*'))
             if p.is_file() and p.name != 'artifact_hashes.json' and '__pycache__' not in p.parts})
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        completed = True
    except BaseException as exc:
        dump(output / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                                      'state_frames_captured': len(states), 'simulation_completed': completed})
        if states and not (output / 'states.npz').exists():
            np.savez_compressed(output / 'partial_states.npz', **{key: np.stack([s[key] for s in states]) for key in states[0]})
        dump(output / 'actions.json', actions)
        raise
    finally:
        if env is not None:
            env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=DEFAULT)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--demo', required=True, type=Path)
    parser.add_argument('--phase', choices=['nominal', 'qualification_probe'], default='nominal')
    parser.add_argument('--baseline', choices=['hold', 'joint-response'], default='joint-response')
    parser.add_argument('--actions', type=Path, help='JSON list of complete 29-DOF joint actions; nominal episodes only')
    parser.add_argument('--duration', type=float, default=3.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--no-video', action='store_true')
    rollout(parser.parse_args())


if __name__ == '__main__':
    main()
