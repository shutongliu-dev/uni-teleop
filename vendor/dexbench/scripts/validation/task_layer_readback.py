"""Private, bounded native readback/reset diagnostic for any task-layer release.

The two trials restore the same release-owned reset state, then submit only
ordinary 20 ms joint-target holds. Readback never advances physics. This is
engineering evidence, not task success or particle force-transfer qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import traceback

import numpy as np

from bench.paths import OUTPUT_ROOT


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def clock(env):
    return (int(env.world.current_time_step_index), float(env.world.current_time), float(env.time_s))


def read_without_step(env):
    """Check repeat reads of the same native tick without a physics advance."""
    before = clock(env)
    first, second = env.capture_state(), env.capture_state()
    if before != clock(env):
        raise RuntimeError('State fetch/read advanced a native or public physics clock')
    if set(first) != set(second) or any(not np.isfinite(a).all() for a in first.values()):
        raise RuntimeError('Nonfinite or inconsistent readback inventory')
    if any(not np.array_equal(first[key], second[key]) for key in first):
        raise RuntimeError('Repeated reads changed state within one native tick')
    return first


def stack_states(states):
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise RuntimeError('Native state keys changed during one trial')
    return {key: np.stack([state[key] for state in states]) for key in sorted(keys)}


def max_distance(a, b):
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.linalg.norm(difference, axis=-1).max()) if difference.size else 0.


def review_trials(first, second, particle_sets, *, dt, reset_tolerance_m,
                  trajectory_tolerance_m, velocity_tolerance_m_s):
    """Compare original particle identities and measured, indexed trajectories."""
    result = {'particle_sets': {}, 'native_clocks_consecutive': True,
        'reset_distribution_reproduced': True, 'trajectory_within_declared_tolerances': True}
    for trial in (first, second):
        result['native_clocks_consecutive'] &= bool(
            np.all(np.diff(trial['native_physics_step_index']) == 1)
            and np.allclose(np.diff(trial['native_world_time_s']), dt, atol=1.e-9, rtol=0))
    result['matching_sample_inventory'] = bool(set(first) == set(second)
        and all(first[key].shape == second[key].shape for key in first))
    if not result['matching_sample_inventory']:
        raise RuntimeError('Reset trials have different frame or body inventories')
    for name, item in particle_sets.items():
        q, qd, ids = (f'task_particle_{name}_{suffix}' for suffix in ('q', 'qd', 'ids'))
        a, b = first[q], second[q]
        av, bv = first[qd], second[qd]
        ids_stable = bool(np.all(first[ids] == first[ids][0])
            and np.all(second[ids] == first[ids][0])
            and first[ids].shape[1] == item['count'])
        initial_distance, trajectory_distance = max_distance(a[0], b[0]), max_distance(a, b)
        velocity_distance = max_distance(av, bv)
        changed = [{'position_change_max_m': max_distance(x, x[0]),
                    'velocity_change_max_m_s': max_distance(v, v[0])}
                   for x, v in ((a, av), (b, bv))]
        updated = all(row['position_change_max_m'] > 1.e-8 and row['velocity_change_max_m_s'] > 1.e-7
                      for row in changed)
        within = trajectory_distance <= trajectory_tolerance_m and velocity_distance <= velocity_tolerance_m_s
        result['particle_sets'][name] = {
            'count': item['count'], 'original_ids_stable': ids_stable,
            'both_trials_show_position_and_velocity_changes': updated,
            'reset_initial_max_distance_m': initial_distance,
            'trajectory_max_distance_m': trajectory_distance,
            'trajectory_velocity_max_distance_m_s': velocity_distance,
            'observed_changes': changed, 'trajectory_within_declared_tolerances': within,
            'mass_source': 'Pinned authored inventory; no native particle force or measured mass claim'}
        result['reset_distribution_reproduced'] &= ids_stable and initial_distance <= reset_tolerance_m
        result['trajectory_within_declared_tolerances'] &= within
    result['object_reset_max_position_difference_m'] = max_distance(first['object_q'][0, :, :3], second['object_q'][0, :, :3])
    result['reset_distribution_reproduced'] &= result['object_reset_max_position_difference_m'] <= reset_tolerance_m
    result['object_trajectory_max_position_difference_m'] = max_distance(first['object_q'][..., :3], second['object_q'][..., :3])
    result['initial_native_clocks'] = [
        {'physics_step_index': int(trial['native_physics_step_index'][0]),
         'world_time_s': float(trial['native_world_time_s'][0])} for trial in (first, second)]
    result['sample_count_per_trial'] = len(first['native_physics_step_index'])
    return result


def hold_trial(env, steps, *, archive_path=None):
    """Tap the existing post-step contact acquisition; never add a solver step."""
    samples = [read_without_step(env)]
    initial_clock = clock(env)
    targets = {name: float(env._targets[0, env.joint_names.index(name)]) for name in env.actuated_joint_names}
    action = {'joint_positions_rad': targets, 'duration_s': .02}
    # Bundle observers sample after contacts. Capture after that sample so its
    # arrays belong to this tick, rather than the preceding observer frame.
    hook = '_sample_task_state' if getattr(env, '_task_reader', None) is not None else '_read_rigid_contacts'
    original = getattr(env, hook)

    def sample_after_native_contact_read(*args, **kwargs):
        original(*args, **kwargs)
        samples.append(read_without_step(env))

    setattr(env, hook, sample_after_native_contact_read)
    try:
        for _ in range(steps):
            env.step(deepcopy(action))
    finally:
        setattr(env, hook, original)
    expected = steps * round(.02/env.dt)
    final_clock = clock(env)
    state = stack_states(samples)
    # Match env.step's 1e-8 s allowance for each 20 ms interval. PhysX stores
    # dt as a float32 value, so an exact 0.5 ms request accumulates roundoff.
    elapsed = final_clock[1]-initial_clock[1]
    tolerance = steps*1.e-8
    if archive_path is not None:
        np.savez_compressed(archive_path, **state)
        dump(Path(archive_path).with_suffix('.clock.json'), {
            'initial_clock': initial_clock, 'final_clock': final_clock,
            'expected_native_steps': expected, 'observed_samples': len(samples),
            'expected_elapsed_s': steps*.02, 'observed_elapsed_s': elapsed,
            'elapsed_error_s': elapsed-steps*.02, 'elapsed_tolerance_s': tolerance})
    if len(samples) != expected+1 or final_clock[0]-initial_clock[0] != expected:
        raise RuntimeError('Native observation count differs from ordinary 20 ms holds')
    if not np.isclose(elapsed, steps*.02, atol=tolerance, rtol=0):
        raise RuntimeError(f'Native elapsed time differs from commanded holds: '
                           f'observed={elapsed}, expected={steps*.02}, tolerance={tolerance}')
    return state, action



def native_scene_statistics(env):
    """Supplemental native GPU allocation evidence, not authored particle counts."""
    try:
        import omni.physx
        from omni.physx.bindings._physx import PhysicsSceneStats
        from pxr import PhysicsSchemaTools, Sdf, UsdUtils
        stats = PhysicsSceneStats()
        stage_id = UsdUtils.StageCache.Get().GetId(env.stage).ToLongInt()
        path = env.world.get_physics_context().prim_path
        ok = omni.physx.get_physx_statistics_interface().get_physx_scene_statistics(
            stage_id, PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path)), stats)
        fields = ('gpu_mem_heap_simulation', 'gpu_mem_heap_simulation_particles',
                  'gpu_mem_heap_particles', 'gpu_mem_particles', 'gpu_mem_particle_contacts')
        return {'query_succeeded': bool(ok), 'physics_scene_path': str(path),
                'native_statistics': {key: int(getattr(stats, key)) for key in fields}}
    except Exception as exc:
        return {'query_succeeded': False, 'error': f'{type(exc).__name__}: {exc}'}

def run(args):
    from bench.isaacsim.release import digest, task_inputs, validate
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    from scripts.validation.board_game_runtime_smoke import gpu_selection

    project = Path(__file__).resolve().parents[2]
    source = verify_if_snapshot(project, required_modules=(
        'scripts.validation.task_layer_readback', 'bench.isaacsim.release',
        'bench.simulation.isaacsim_backend.env'), entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != project:
        raise ValueError('Run this diagnostic from its complete frozen source snapshot')
    if not 1 <= args.control_steps <= 10:
        raise ValueError('Bounded readback check accepts 1–10 ordinary 20 ms control steps')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Diagnostic output belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'native_task_layer_readback.v1', 'passed': False,
        'release': str(args.release.resolve()), 'task_id': args.task, 'seed': args.seed,
        'source_snapshot': str(project), 'model_calls': 0,
        'public_interface_test': False, 'task_success_qualified': False,
        'particle_force_or_weight_feedback_qualified': False,
        'post_reset_state_write_actions': False, 'actions': 'Unchanged reset joint targets; 20 ms holds only',
        'reset_contract': 'Constructor initialization plus two explicit identical release-owned resets; native initialization steps retained.',
        'numerical_tolerances': {'reset_position_m': args.reset_tolerance_m,
            'trajectory_position_m': args.trajectory_tolerance_m,
            'trajectory_velocity_m_s': args.velocity_tolerance_m_s},
        'scope': 'Private readback/reset engineering test; not a manipulation or fluid-dynamics qualification.'}
    env = None
    try:
        report['gpu'] = gpu_selection(args.gpu_uuid)
        dump(output/'runtime_imports.json', source)
        validate(args.release)
        _, scene, task, initial = task_inputs(args.release, args.task, seed=args.seed)
        scene = scene_with_hand(scene, scene['robot']['hand_id'])
        if not scene.get('task_layers') and not scene.get('native_task_bundle'):
            raise ValueError('Selected release has no declared native task asset observations')
        dump(output/'scene.json', scene)
        dump(output/'task.json', task)
        dump(output/'initial_state.json', initial)
        report['release_manifest_sha256'] = digest(args.release/'release_manifest.json')
        report['source_manifest_sha256'] = digest(project/'source_snapshot_manifest.json')
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', render_width=640, render_height=480)
        trials, actions = [], []
        for index in range(2):
            reset_before = clock(env)
            env.reset(seed=args.seed, initial_state=initial)
            reset_after = clock(env)
            dump(output/f'metadata_{index}.json', env.metadata())
            trial, action = hold_trial(env, args.control_steps, archive_path=output/f'trial_{index}.npz')
            trials.append(trial)
            actions.append(action)
            dump(output/f'trial_{index}.json', {'reset_clock_before': reset_before,
                'reset_clock_after': reset_after, 'samples': len(trial['native_physics_step_index']),
                'readback_added_physics_steps': 0, 'same_tick_reads_identical': True,
                'action': action, 'control_steps': args.control_steps,
                'archive_sha256': digest(output/f'trial_{index}.npz')})
            print(json.dumps({'event': 'readback_trial_complete', 'trial': index,
                              'native_samples': len(trial['native_physics_step_index'])}), flush=True)
        report['physics_dt_s'] = env.dt
        report['native_scene_statistics'] = native_scene_statistics(env)
        report['same_reset_targets'] = actions[0] == actions[1]
        reader = env._task_reader if getattr(env, '_task_reader', None) is not None else env._task_readback
        particles = ({name: {'count': row['particle_count']} for name, row in reader.info['particle_sets'].items()}
                     if getattr(env, '_task_reader', None) is not None else reader.particles)
        report['review'] = review_trials(*trials, particles, dt=env.dt,
            reset_tolerance_m=args.reset_tolerance_m, trajectory_tolerance_m=args.trajectory_tolerance_m,
            velocity_tolerance_m_s=args.velocity_tolerance_m_s)
        for name, item in particles.items():
            authored = reader.initial_particles[name]
            report['review']['particle_sets'][name]['initial_distance_from_authored_m'] = [
                max_distance(trial[f'task_particle_{name}_q'][0], authored[0]) for trial in trials]
            report['review']['particle_sets'][name]['ids_match_pinned_source'] = bool(
                np.array_equal(trials[0][f'task_particle_{name}_ids'][0], authored[2]))
        checks = report['review']
        report['passed'] = bool(report['same_reset_targets'] and checks['native_clocks_consecutive']
            and checks['matching_sample_inventory'] and checks['reset_distribution_reproduced']
            and checks['trajectory_within_declared_tolerances']
            and all(item['ids_match_pinned_source'] and item['both_trials_show_position_and_velocity_changes']
                    for item in checks['particle_sets'].values()))
        report['source_snapshot_verified_after'] = verify_if_snapshot(project,
            required_modules=('scripts.validation.task_layer_readback', 'bench.simulation.isaacsim_backend.task_readback'),
            entrypoint_file=Path(__file__)) is not None
        validate(args.release)
        if not report['passed']:
            raise RuntimeError('Readback/reset check did not pass; retain measured differences and scope')
    except BaseException as exc:
        report['passed'] = False
        report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        # SimulationApp.fast_shutdown may terminate the process inside close;
        # persist success/failure before releasing the native application.
        dump(output/'report.json', report)
        try:
            if env is not None:
                env.close()
        except BaseException as exc:
            report['passed'] = False
            report['cleanup_error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            dump(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--control-steps', type=int, default=3)
    parser.add_argument('--reset-tolerance-m', type=float, default=1.e-4)
    parser.add_argument('--trajectory-tolerance-m', type=float, default=1.e-3)
    parser.add_argument('--velocity-tolerance-m-s', type=float, default=.02)
    args = parser.parse_args()
    for value in (args.reset_tolerance_m, args.trajectory_tolerance_m, args.velocity_tolerance_m_s):
        if not np.isfinite(value) or value < 0:
            parser.error('Numerical tolerances must be finite and nonnegative')
    os.environ.setdefault('ISAAC_BENCH_EXPERIENCE', str(Path(__file__).resolve().parents[2]/'bench/apps/workroom.headless.kit'))
    run(args)


if __name__ == '__main__':
    main()
