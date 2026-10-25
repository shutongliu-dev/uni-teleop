"""Check a frozen scene in GPU Isaac, with stationary robot drives and no model.

This is an asset/initialization check, not manipulation-task qualification.
It preserves every 20 ms state and the existing independent physics audit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.isaacsim.release import dump
from bench.evaluation.physics import audit_native_episode
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_stability(initial, final, impulses, *, drift_limit=.015):
    """Require actual support evidence as well as finite, stationary bodies."""
    if set(initial['objects']) != set(final['objects']):
        raise ValueError('Object inventory changed during settling')
    objects = {}
    for name, before in initial['objects'].items():
        after = final['objects'][name]
        position = np.asarray(after['position_m'], dtype=float)
        velocity = np.asarray(after['linear_velocity_m_s'], dtype=float)
        omega = np.asarray(after['angular_velocity_rad_s'], dtype=float)
        start = np.asarray(before['position_m'], dtype=float)
        if any(x.shape != (3,) or not np.isfinite(x).all() for x in (position, velocity, omega, start)):
            raise ValueError('Missing or nonfinite object state')
        drift = float(np.linalg.norm(position-start))
        speed, angular_speed = float(np.linalg.norm(velocity)), float(np.linalg.norm(omega))
        impulse = float(impulses.get(name, 0.))
        objects[name] = {'drift_m': drift, 'final_speed_m_s': speed,
            'final_angular_speed_rad_s': angular_speed, 'support_impulse_ns': impulse,
            'passed': bool(drift <= drift_limit and speed <= .04 and angular_speed <= .3
                           and math.isfinite(impulse) and impulse > 0.)}
    return {'objects': objects, 'all_dynamic_objects_supported_and_stable':
            all(row['passed'] for row in objects.values()), 'maximum_drift_m': drift_limit,
            'scope': 'Passive settling and observed support contacts; no task or grasp success.'}


def seal_report_before_shutdown(output, report, env):
    """Kit fast shutdown may terminate this process without returning to Python.

    Persist evidence first. A caller must additionally observe exit status zero;
    this pre-shutdown report cannot certify completion of native teardown.
    """
    report['native_shutdown_verified'] = False
    report['native_shutdown_status'] = 'not_observed_by_report'
    report['external_acceptance_required'] = 'report.passed and observed_process_exit_code == 0'
    dump(output/'report.json', report)
    print(json.dumps({'event':'scene_report_sealed', 'passed':report['passed'],
                      'output':str(output), 'native_shutdown_verified':False}), flush=True)
    if env is not None:
        if not report['passed']:
            # env.close preserves an active exception as native exit status 1.
            # Without this, Kit would terminate a failed validation with zero.
            try:
                raise RuntimeError('Native scene validation failed; inspect the sealed report')
            finally:
                env.close()
        try:
            env.close()
        except Exception as exc:
            report['passed'] = False
            report['close_error'] = {'type':type(exc).__name__, 'detail':str(exc)}
            dump(output/'report.json', report)
            raise
    return report


def run(args):
    root = Path(__file__).absolute().parents[2]
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    verification = verify_if_snapshot(root,
        required_modules=('scripts.validation.validate_scenario',
                          'bench.simulation.isaacsim_backend.env', 'bench.evaluation.physics'),
        entrypoint_file=Path(__file__))
    if verification is None:
        raise ValueError('Run scene validation from a complete frozen source snapshot')
    count = round(args.duration / .02)
    if not 1 <= count <= 500 or abs(count*.02-args.duration) > 1e-9:
        raise ValueError('Duration must be a positive 20 ms multiple at most 10 seconds')
    scene = load_json(args.scene)
    frozen_seed = scene.get('scenario', {}).get('seed')
    if args.seed is None:
        args.seed = frozen_seed if frozen_seed is not None else 0
    if type(args.seed) is not int or not 0 <= args.seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    if 'scenario' in scene and (type(frozen_seed) is not int or args.seed != frozen_seed):
        raise ValueError('Seed must match the frozen scenario initialization')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'scene.json', scene)
    dump(output/'runtime_imports.json', verification)
    dump(output/'input.json', {'scene_path':str(args.scene.resolve()),
        'scene_sha256':digest(args.scene), 'seed':args.seed, 'device':args.device,
        'duration_s':args.duration, 'model_calls':0})
    env = observer = initial = final = None
    states, error, close_error = [], None, None
    impulses = {row['id']: 0. for row in scene.get('objects', [])}
    try:
        env = FR3DexterousIsaacEnv(scene, device=args.device, render_width=640, render_height=480)
        initial = env.reset(seed=args.seed)
        states.append(env.capture_state())
        dump(output/'metadata.json', env.metadata())
        np.save(output/'triangles.npy', env.triangles)
        if scene.get('cloth', {}).get('enabled') is True:
            from scenes.cloth.worker_observer import build_cloth_observer
            observer = build_cloth_observer(env, output/'cloth_physics')
        from PIL import Image
        Image.fromarray(env.render('overview', width=640, height=480)).save(output/'initial.png')
        action = {'joint_positions_rad':deepcopy(env.action_spec()['initial_joint_positions_rad']),
                  'duration_s':.02}
        supports = {'/World/Table', '/World/Ground'} | {
            row['path'] for row in env.scene_info.get('fixtures', {}).values()}
        with (output/'physics.jsonl').open('x') as stream:
            stream.write(json.dumps({'kind':'reset', 'observation':initial}, allow_nan=False)+'\n')
            for _ in range(count):
                final = env.step(action)
                states.append(env.capture_state())
                stream.write(json.dumps({'kind':'step', 'command':action, 'observation':final}, allow_nan=False)+'\n')
                stream.flush()
                for contact in final['contacts']['rigid']:
                    pair = (contact['actor0'], contact['actor1'])
                    for name, info in env.scene_info['objects'].items():
                        if info['path'] not in pair:
                            continue
                        other = pair[1] if pair[0] == info['path'] else pair[0]
                        if any(other == p or other.startswith(p+'/') for p in supports):
                            impulses[name] += float(contact['impulse_magnitude_sum_ns'])
        if observer is not None:
            from scenes.cloth.worker_observer import seal_observer
            seal_observer(observer)
        Image.fromarray(env.render('overview', width=640, height=480)).save(output/'final.png')
    except Exception as exc:
        error = {'type':type(exc).__name__, 'detail':str(exc)}
    finally:
        if observer is not None and not observer.closed:
            try:
                from scenes.cloth.worker_observer import seal_observer
                seal_observer(observer)
            except Exception as exc:
                close_error = {'type':type(exc).__name__, 'detail':str(exc)}
        if states and env is not None:
            try:
                from bench.runtime.worker import seal_native_recording
                seal_native_recording(output, states, env.triangles,
                    {'evaluated':False, 'scope':'Scene initialization only'},
                    reason='finish_request' if error is None and close_error is None else 'scene_validation_error')
            except Exception as exc:
                close_error = {'type':type(exc).__name__, 'detail':str(exc)}
    audit, stability, source = {'physics_valid':False}, None, None
    try:
        audit = audit_native_episode(output)
        dump(output/'physics_audit.json', audit)
        stability = scene_stability(initial, final, impulses) if initial and final and error is None else None
        source = verify_if_snapshot(root, required_modules=('scripts.validation.validate_scenario',),
                                    entrypoint_file=Path(__file__))
    except Exception as exc:
        error = error or {'stage':'final_evidence', 'type':type(exc).__name__, 'detail':str(exc)}
    passed = bool(error is None and close_error is None and audit['physics_valid']
                  and stability and stability['all_dynamic_objects_supported_and_stable'] and source)
    report = {'schema':'native_scene_initialization_check_v1', 'passed':passed,
        'model_calls':0, 'task_success_claimed':False, 'object_state_writeback':False,
        'error':error, 'close_error':close_error, 'physics_valid':audit['physics_valid'],
        'stability':stability, 'recorded_states':len(states),
        'scene_sha256':digest(output/'scene.json'), 'runtime_imports_verified':bool(source),
        'scope':'Native scene loading, contact evidence and passive stability; no task qualification.'}
    report['artifacts_sha256'] = {str(p.relative_to(output)):digest(p)
                                for p in output.rglob('*') if p.is_file()}
    return seal_report_before_shutdown(output, report, env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--seed', type=int, help='Defaults to the frozen scenario initialization seed')
    parser.add_argument('--duration', type=float, default=1.2)
    report = run(parser.parse_args())
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
