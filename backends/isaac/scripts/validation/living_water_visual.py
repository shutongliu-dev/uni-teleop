"""Frozen-source native water-surface/RGB-D check with a short robot hold.

Uses the frozen pouring task, its physical particle ledger and its evaluator.
Only ordinary robot joint targets are submitted after reset. No object/particle
state setters, model calls or pouring-success claims are made. Run through
scripts/isaac_python.sh from a complete frozen source snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from bench.paths import DATA_ROOT, load_json, OUTPUT_ROOT
from scripts.validation.living_television_visual import actual_cuda_identity, same_physics


KIT = 'bench/apps/workroom.headless.kit'
RUNTIME = Path(os.environ.get('ISAAC_PATH', '/nonexistent-isaac-runtime'))
DEFAULT_RELEASE = Path(f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/v5')
CAMERAS = ('overview', 'task', 'room', 'wrist')
MODULES = ('scripts.validation.living_water_visual', 'scripts.validation.living_television_visual',
    'bench.isaacsim.release', 'bench.simulation.isaacsim_backend.env',
    'scripts.validation.board_game_runtime_smoke')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def copied_state(env):
    return {key: np.array(value, copy=True) for key, value in env.capture_state().items()}


def particle_digest(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


def validate_inventory(state, observation, task, score, *, baseline=False):
    """Check raw identities and the independently frozen observer's mass ledger."""
    config = task['evaluation']
    count = config['particle_count']
    require(count == 488, 'This diagnostic expects the published 488-particle practical profile')
    name = config['particle_set_id']
    points, velocities, ids = (np.asarray(state[f'task_particle_{name}_{key}'])
                               for key in ('q', 'qd', 'ids'))
    require(points.shape == velocities.shape == (count, 3)
            and np.isfinite(points).all() and np.isfinite(velocities).all(),
            'Missing/nonfinite raw particle positions or velocities')
    require(ids.shape == (count,) and ids.dtype.kind in 'iu'
            and np.array_equal(ids, np.arange(count)), 'Physical particle IDs changed')
    ids_sha = hashlib.sha256(ids.astype('<i8', copy=False).tobytes()).hexdigest()
    require(ids_sha == config['particle_ids_sha256'], 'Raw particle IDs differ from frozen identity')
    mass = count * config['particle_mass_kg']
    require(math.isclose(mass, .25, rel_tol=0., abs_tol=1e-12), 'Frozen water mass changed')
    native = observation['task_state']
    require(native.get('errors') == [] and isinstance(native.get('reset_id'), str)
            and native['reset_id'], 'Water observer is missing or invalid')
    water, window = native['water'], native['sampling']
    require(water['particle_count'] == count and water['ids_sha256'] == ids_sha,
            'Observer particle inventory differs from raw physical particles')
    counts = water['counts']
    require(set(counts) == {'kettle', 'cup', 'spilled_outside', 'in_flight_or_unresolved'}
            and all(type(value) is int and value >= 0 for value in counts.values())
            and sum(counts.values()) == count, 'Water compartments do not conserve particle inventory')
    require(int(state['native_physics_step_index']) == window['end_step']
            and math.isclose(float(state['native_world_time_s']), window['end_time_s'], abs_tol=1e-8, rel_tol=0.)
            and math.isclose(float(state['time_s']), observation['sim_time_s'], abs_tol=1e-8, rel_tol=0.),
            'Raw physical state and observer do not share the native tick/time')
    require(score.get('success') is False and score.get('progress') == 0.
            and score.get('evidence_status') == 'available' and not score.get('irrecoverable_reason'),
            'Reset/robot hold earned task credit or has invalid scoring evidence')
    fields = ('initial_water_ml', 'kettle_water_ml', 'cup_water_ml', 'current_spill_ml', 'unresolved_ml')
    values = {key: score[key] for key in fields}
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= -1e-8
                for value in values.values()), 'Invalid scored water-volume ledger')
    require(math.isclose(values['initial_water_ml'], 250., abs_tol=1e-8, rel_tol=0.)
            and math.isclose(sum(values[key] for key in fields[1:]), values['initial_water_ml'],
                             abs_tol=1e-8, rel_tol=0.), 'Scored water volumes do not conserve initial mass')
    if baseline:
        require(window['sampled_substeps'] == 0 and counts['kettle'] == count,
                'Reset must establish the unchanged all-in-kettle baseline')
    require(observation['metrics']['state_finite'] is True
            and not observation['terminated'] and not observation['truncated'],
            'Native observation is nonfinite or terminated')
    return {'particle_count': count, 'ids_sha256': ids_sha, 'input_sha256': particle_digest(points),
        'declared_total_mass_kg': mass, 'counts': counts, 'volume_ledger_ml': values,
        'sampling': window, 'reset_id': native['reset_id'], 'task_success': False,
        'scope': 'Raw identity/count and frozen nominal-mass ledger; not measured force calibration.'}


def validate_surfaces(receipts, points, *, expect_cached=False):
    """Bind the render-only mesh to the same unsmoothed source points."""
    require(isinstance(receipts, list) and len(receipts) == 1,
            'Expected exactly one real particle-surface receipt')
    expected = particle_digest(points)
    for row in receipts:
        require(isinstance(row, dict) and row.get('input_sha256') == expected
                and row.get('particle_count') == len(points),
                'Surface mesh is missing or uses stale/different physical particles')
        require(all(type(row.get(key)) is int and row[key] > 0 for key in ('vertices', 'triangles')),
                'Particle surface is empty or has invalid mesh counts')
        require(isinstance(row.get('mesh_path'), str) and row['mesh_path'].startswith('/World/')
                and row.get('physics_state_writeback') is False,
                'Surface is not an explicitly read-only world mesh')
        require(type(row.get('reused_same_particle_positions')) is bool,
                'Surface must report whether the identical input mesh was reused')
        if expect_cached:
            require(row['reused_same_particle_positions'] is True,
                    'Repeated same-tick rendering did not reuse the identical particle surface')
    return receipts


def validate_same_tick_surfaces(first, repeated):
    fields = ('input_sha256', 'particle_count', 'vertices', 'triangles', 'mesh_path', 'physics_state_writeback')
    require([{key: row[key] for key in fields} for row in first]
            == [{key: row[key] for key in fields} for row in repeated],
            'Repeated same-tick rendering changed the surface identity or geometry counts')


def validate_native_advance(before, after, *, dt, duration):
    expected = round(duration / dt)
    # PhysX exposes its step duration through a float32 binding. Bound that
    # conversion error without relaxing the exact integer substep requirement.
    clock_tolerance = max(1e-8, 2 * expected * abs(float(np.float32(dt)) - dt))
    require(int(after['native_physics_step_index']) - int(before['native_physics_step_index']) == expected
            and math.isclose(float(after['native_world_time_s']) - float(before['native_world_time_s']),
                             duration, rel_tol=0., abs_tol=clock_tolerance)
            and math.isclose(float(after['time_s']) - float(before['time_s']), duration, rel_tol=0., abs_tol=1e-8),
            'Robot hold did not advance exactly the requested native steps and clocks')
    return expected


def render_group(env, label, observation, state, args, output, media, report):
    from PIL import Image
    from bench.isaacsim.release import digest, dump
    name = 'task_particle_water_q'
    reference_observer = env._task_reader.observe()
    require(reference_observer == observation['task_state'], 'Observer changed before image capture')
    np.savez_compressed(output / (label + '_physics_state.npz'), **state)
    records = []
    for camera in CAMERAS:
        captures = []
        for attempt in range(2):
            before = copied_state(env)
            same_physics(state, before)
            started = time.monotonic()
            frame = env.render_rgbd(camera=camera, width=args.width, height=args.height)
            elapsed = time.monotonic() - started
            same_physics(before, copied_state(env))
            require(env._task_reader.observe() == reference_observer, 'Rendering altered the frozen task observer')
            require(frame['sim_time_s'] == float(state['time_s'])
                    and frame['depth_sim_time_s'] == float(state['time_s']), 'RGB-D timestamps are stale')
            receipts = validate_surfaces(frame.get('particle_surfaces'), before[name], expect_cached=attempt == 1)
            require(frame['rgb'].dtype == np.uint8 and frame['rgb'].shape == (args.height, args.width, 3),
                    'Native RGB image has an unexpected format')
            depth = np.asarray(frame['depth_m'])
            require(depth.shape == (args.height, args.width) and np.isfinite(depth).all()
                    and np.any(depth > 0), 'Native camera depth is missing or invalid')
            captures.append({'attempt': attempt + 1, 'wall_time_including_render_warmup_s': elapsed,
                'native_step': int(before['native_physics_step_index']),
                'native_time_s': float(before['native_world_time_s']), 'physics_steps_added': 0,
                'physics_arrays_identical_before_after': True, 'particle_surfaces': receipts,
                'camera_calibration': frame['camera_calibration'],
                'render_synchronization': frame['render_synchronization']})
        validate_same_tick_surfaces(captures[0]['particle_surfaces'], captures[1]['particle_surfaces'])
        require(captures[0]['camera_calibration'] == captures[1]['camera_calibration'],
                'Same-tick camera calibration changed between captures')
        image_path, depth_path = media / f'{label}_{camera}.png', output / f'{label}_{camera}_depth_m.npy'
        Image.fromarray(frame['rgb']).save(image_path)
        np.save(depth_path, frame['depth_m'])
        record = {'phase': label, 'camera': camera, 'captures': captures,
            'rgb': {'path': str(image_path), 'sha256': digest(image_path)},
            'depth': {'path': str(depth_path), 'sha256': digest(depth_path), 'semantics': frame['depth_semantics']},
            'physics_state': {'path': str(output / (label + '_physics_state.npz')),
                              'sha256': digest(output / (label + '_physics_state.npz'))},
            'same_tick_surface_reuse_verified': True, 'observer_unchanged': True}
        records.append(record)
        report['captures'].append(record)
        dump(output / f'{label}_{camera}_capture.json', record)
        print(json.dumps({'event': 'water_surface_camera_complete', 'phase': label, 'camera': camera,
                          'native_step': int(state['native_physics_step_index'])}), flush=True)
    return records


def run(args):
    from bench.isaacsim.release import digest, dump, load_evaluator, task_inputs
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from scripts.validation.board_game_runtime_smoke import gpu_selection

    project = Path(__file__).resolve().parents[2]
    output, media = args.output.resolve(), args.media_output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT), 'Store evidence under shared datasets')
    require(media.is_relative_to(OUTPUT_ROOT), 'Store PNG media under demo/interface/scenes')
    require(not output.exists() and not media.exists(), 'Use new output and media directories')
    output.mkdir(parents=True, exist_ok=False)
    media.mkdir(parents=True, exist_ok=False)
    env = None
    report = {'schema': 'living_water_surface_native_diagnostic.v1', 'passed': False,
        'started_at': datetime.now(timezone.utc).isoformat(), 'source_snapshot': str(project),
        'task_id': 'water_pouring', 'seed': args.seed, 'requested_hold_s': args.duration,
        'scope': 'Canonical reset, robot target hold, four same-tick RGB-D cameras and a second reset; visual/source consistency only.',
        'model_calls': 0, 'post_reset_object_particle_state_writes': 0,
        'task_success_claimed': False, 'robot_pouring_qualified': False,
        'physics_acceptance_claimed': False, 'http_transport_tested': False,
        'captures': [], 'phases': [], 'hold_observations': []}
    try:
        source = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
        require(source is not None and Path.cwd().resolve() == project,
                'Run from the root of a complete frozen source snapshot')
        manifest = load_json(project / 'source_snapshot_manifest.json')
        require(manifest['files_sha256'].get(KIT) == digest(project / KIT)
                and Path(os.environ.get('ISAAC_BENCH_EXPERIENCE', '')).resolve() == project / KIT,
                'Select the headless Kit experience included in this frozen snapshot')
        require(Path(os.environ.get('ISAAC_PATH', '')).resolve() == RUNTIME.resolve()
                and Path(sys.executable).resolve() == (RUNTIME / 'kit/python/bin/python3.12').resolve(),
                'Use the pinned Isaac Sim 6.1.0 interpreter and runtime')
        report['source_manifest_sha256'] = digest(project / 'source_snapshot_manifest.json')
        report['runtime'] = {'root': str(RUNTIME), 'version': (RUNTIME / 'VERSION').read_text().strip(),
            'version_sha256': digest(RUNTIME / 'VERSION'), 'python': sys.executable,
            'kit_experience': str(project / KIT), 'kit_experience_sha256': digest(project / KIT)}
        dump(output / 'runtime_imports.json', source)
        report['gpu'] = gpu_selection(args.gpu_uuid)
        report['environment'] = {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'ISAAC_BENCH_SINGLE_CUDA_DEVICE', 'ISAAC_ALLOW_UNVALIDATED_DRIVER',
            'ISAAC_BENCH_EXPERIENCE', 'ISAAC_PATH', 'ISAAC_PYTHON_LAUNCHER')}
        release = args.release.resolve()
        entry, scene, task, initial = task_inputs(release, 'water_pouring', seed=args.seed)
        require(task['evaluation']['schema'] == 'living_water_evaluation_v1', 'Expected the frozen PBD water task')
        report['release'] = {'path': str(release), 'manifest_sha256': digest(release / 'release_manifest.json'),
            'scene_sha256': digest(release / entry['scene']), 'task_sha256': digest(release / entry['task'])}
        for filename, value in [('resolved_scene.json', scene), ('resolved_task.json', task), ('canonical_initial_state.json', initial)]:
            dump(output / filename, value)
        report['input_files_sha256'] = {name: digest(output / name) for name in
            ('resolved_scene.json', 'resolved_task.json', 'canonical_initial_state.json')}
        started = time.monotonic()
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', render_width=args.width, render_height=args.height)
        report['native_startup_wall_s'] = time.monotonic() - started
        require(set(env.camera_ids) == set(CAMERAS), 'Expected the four original single-arm RGB-D cameras')
        started = time.monotonic()
        observation = env.reset(seed=args.seed, initial_state=initial)
        report['first_reset_wall_s'] = time.monotonic() - started
        report['actual_cuda_device'] = actual_cuda_identity(args.gpu_uuid)
        config = env.app.config
        require(config.get('active_cuda_gpus') == [0] and config.get('physics_gpu') == 0
                and config.get('max_gpu_count') == 1 and config.get('multi_gpu') is False,
                'Native app did not select the verified single RTX/physics GPU')
        report['actual_app_gpu_selection'] = {key: config.get(key) for key in
            ('active_cuda_gpus', 'physics_gpu', 'max_gpu_count', 'multi_gpu', 'renderer')}
        metadata = env.metadata()
        require(metadata['experience']['path'] == str(project / KIT)
                and metadata['experience']['sha256'] == digest(project / KIT)
                and metadata['native_isaacsim_app'] is True, 'Native experience differs from the frozen source')
        dump(output / 'native_metadata.json', metadata)
        report['native_metadata_sha256'] = digest(output / 'native_metadata.json')
        evaluator = load_evaluator(release)(task)

        def record_phase(label, obs, *, baseline, score=None):
            state = copied_state(env)
            if score is None:
                score = evaluator.update(obs)
            ledger = validate_inventory(state, obs, task, score, baseline=baseline)
            from pxr import UsdPhysics
            water = task['asset_metadata']['water']
            mass = UsdPhysics.MassAPI(env.stage.GetPrimAtPath(water['particles_path'])).GetMassAttr().Get()
            require(math.isclose(mass, water['total_mass_kg'], abs_tol=1e-8, rel_tol=0.), 'Composed authored water mass changed')
            phase = {'label': label, 'ledger': ledger, 'score': score, 'observer': obs['task_state'],
                'native_step': int(state['native_physics_step_index']), 'native_time_s': float(state['native_world_time_s']),
                'sim_time_s': float(state['time_s']), 'composed_authored_mass_kg': mass}
            report['phases'].append(phase)
            dump(output / (label + '_observation.json'), obs)
            render_group(env, label, obs, state, args, output, media, report)
            return state, ledger

        before, initial_ledger = record_phase('reset01', observation, baseline=True)
        report['reset_initialization_native_step'] = int(before['native_physics_step_index'])
        report['reset_initialization_native_time_s'] = float(before['native_world_time_s'])
        targets = {name: float(observation['robot']['joint_targets_rad'][name]) for name in env.actuated_joint_names}
        action = {'joint_positions_rad': targets, 'duration_s': .02}
        dump(output / 'robot_hold_action.json', action)
        started = time.monotonic()
        for index in range(round(args.duration / .02)):
            observation = env.step(action)
            state = copied_state(env)
            score = evaluator.update(observation)
            ledger = validate_inventory(state, observation, task, score)
            report['hold_observations'].append({'control_index': index + 1, 'ledger': ledger, 'score': score})
        report['robot_hold_wall_s'] = time.monotonic() - started
        # Do not feed the final control observation to the streaming evaluator twice.
        after, held_ledger = record_phase('hold01', observation, baseline=False, score=score)
        report['post_reset_physics_steps'] = validate_native_advance(before, after, dt=env.dt, duration=args.duration)
        report['robot_hold_control_actions'] = round(args.duration / .02)
        points_changed = not np.array_equal(before['task_particle_water_q'], after['task_particle_water_q'])
        report['hold_changed_raw_particles'] = points_changed
        if points_changed:
            require(initial_ledger['input_sha256'] != held_ledger['input_sha256'], 'Hold reused stale particle inputs')
        previous_reset_id = observation['task_state']['reset_id']
        started = time.monotonic()
        observation = env.reset(seed=args.seed, initial_state=initial)
        report['second_reset_wall_s'] = time.monotonic() - started
        evaluator = load_evaluator(release)(task)
        require(observation['task_state']['reset_id'] != previous_reset_id, 'Second reset reused the prior observer episode')
        returned, reset_ledger = record_phase('reset02', observation, baseline=True)
        report['second_reset'] = {'new_observer_reset_id': True, 'particle_ids_restored': True,
            'all_particles_in_kettle': reset_ledger['counts']['kettle'] == 488,
            'raw_positions_equal_first_reset': bool(np.array_equal(before['task_particle_water_q'], returned['task_particle_water_q'])),
            'maximum_position_difference_from_first_reset_m': float(np.abs(before['task_particle_water_q'] - returned['task_particle_water_q']).max()),
            'native_step': int(returned['native_physics_step_index']),
            'native_time_s': float(returned['native_world_time_s'])}
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        try:
            final = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
            require(final is not None, 'Missing final frozen-source verification')
            dump(output / 'runtime_imports_final.json', final)
        except BaseException as exc:
            report['passed'] = False
            report['source_verification_failure'] = str(exc)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['shutdown_note'] = 'Report flushed before Kit fast shutdown; process exit code must be checked separately.'
        dump(output / 'visual_diagnostic.json', report)
        print(json.dumps({'output': str(output), 'media': str(media), 'passed': report['passed']}), flush=True)
        if env is not None:
            if report['passed']:
                env.close()
            else:
                try:
                    raise RuntimeError('Water surface native diagnostic failed; see ' + str(output / 'visual_diagnostic.json'))
                except RuntimeError:
                    env.close()
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media-output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--duration', type=float, default=.2, help='Robot target hold in seconds, 20 ms multiples')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=960)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or not (320 <= args.width <= 3840 and 240 <= args.height <= 2160):
        parser.error('Use an unsigned 32-bit seed and resolution 320–3840 by 240–2160')
    if not .1 <= args.duration <= .3 or not math.isclose(round(args.duration / .02) * .02, args.duration, abs_tol=1e-9):
        parser.error('Use a 0.10–0.30 second hold in exact 20 ms multiples')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
