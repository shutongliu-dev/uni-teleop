"""Read-only native kitchen support probe from a canonical or separate goal reset.

The default starts with assembled objects; --canonical-reset checks supplies.
It tests geometry and gravity support, never awards manipulation qualification,
and never writes object poses after reset. Public-function validation is separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import traceback

from bench.isaacsim.release import digest, dump, task_inputs
from bench.paths import OUTPUT_ROOT, load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

MODULES = ('scripts.validation.kitchen_support_probe',
    'scripts.validation.kitchen_asset_identity', 'scripts.validation.kitchen_scene_evidence',
    'scripts.validation.kitchen_motion_diagnostics',
    'bench.simulation.isaacsim_backend.env')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(args):
    project = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT), 'Evidence belongs in datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'kitchen_goal_support_probe.v1', 'passed': False,
        'task_id': args.task_id, 'model_calls': 0, 'task_success_claimed': False,
        'canonical_reset': args.canonical_reset, 'object_writes_after_reset': False,
        'scope': ('Canonical source reset; gravity, support and native motion only.' if args.canonical_reset else
                  'Separately authored assembled reset; gravity, support and fit only.'),
        'checks': {}}
    env, volume_archive = None, None
    states = []
    try:
        source = verify_if_snapshot(project,
            required_modules=('scripts.validation.kitchen_support_probe',), entrypoint_file=Path(__file__))
        require(source is not None and project == Path.cwd().resolve(), 'Use a complete frozen source root')
        report['source'] = source
        report['source_manifest_sha256'] = digest(project/'source_snapshot_manifest.json')
        report['gpu_uuid'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        report['isaac_allow_unvalidated_driver'] = os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER')
        report['release_manifest_sha256'] = digest(args.release/'release_manifest.json')
        _, scene, task, initial = task_inputs(args.release, args.task_id)
        manifest = load_json(args.asset_manifest)
        require(manifest['presentation'] is (not args.canonical_reset) and manifest['task_id'] == args.task_id,
                'Asset reset kind must match the requested canonical or separate goal probe')
        usd = args.asset_manifest.parent/'task.usda'
        metadata_path = args.asset_manifest.parent/'metadata.json'
        require(digest(usd) == manifest['usd_sha256'] and digest(metadata_path) == manifest['metadata_sha256'],
                'Diagnostic asset hashes differ')
        metadata = load_json(metadata_path)
        scene = deepcopy(scene)
        bundle = scene['native_task_bundle']
        report['canonical_task_usd'] = {'path': bundle['usd_path'], 'sha256': bundle['usd_sha256']}
        canonical_metadata = deepcopy(task['asset_metadata'])
        bundle.update(usd_path=str(usd.resolve()), usd_sha256=digest(usd))
        if metadata.get('volume_bodies'):
            bundle['volume_bodies'] = deepcopy(metadata['volume_bodies'])
        require(bundle['rigid_bodies'] == metadata['actors'], 'Diagnostic object inventory differs')
        from bench.isaacsim.release import _frozen_evaluator_module
        frozen_evaluator = _frozen_evaluator_module(args.release.resolve())
        frozen_rules = importlib.import_module(frozen_evaluator.__package__+'.rules')
        task['asset_metadata'] = deepcopy(metadata)
        task['evaluation'] = frozen_rules.evaluation_configuration(args.task_id, metadata, seed=0,
            physics_dt_s=task['evaluation']['physics_dt_s'])
        bundle['observer']['configuration'] = deepcopy(task)
        dump(output/'task.json', task)
        report['diagnostic_asset_manifest'] = str(args.asset_manifest.resolve())
        report['diagnostic_asset_manifest_sha256'] = digest(args.asset_manifest)
        dump(output/'scene.json', scene)
        dump(output/'metadata.json', metadata)
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', headless=True)
        # Isaac exposes pxr after SimulationApp starts. Validate the complete
        # graph before reset or any measured physical step.
        from scripts.validation.kitchen_asset_identity import canonical_physics_identity, diagnostic_physics_identity
        identity_check = canonical_physics_identity if args.canonical_reset else diagnostic_physics_identity
        report['checks']['asset_identity'] = identity_check(
            Path(report['canonical_task_usd']['path']), usd, canonical_metadata, metadata)
        dump(output/'asset_identity.json', report['checks']['asset_identity'])
        require(report['checks']['asset_identity']['passed'],
                'Diagnostic asset differs from canonical physics beyond declared reset poses')
        observation = env.reset(seed=0, initial_state=initial)
        diagnostic_spoons = [row for row in metadata['objects']
                             if row.get('spoon_convex_collision_partition') or row.get('spoon_split_bowl_collision_partition')]
        convex_spoons = [row for row in metadata['objects']
                         if row.get('spoon_convex_collision_partition')]
        if convex_spoons:
            from bench.evaluation.cloth.cloth_robot_collision import collect_colliders
            from pxr import UsdPhysics
            disabled = {row['body_path']+'/Collision/utensil':
                        UsdPhysics.CollisionAPI(env.stage.GetPrimAtPath(row['body_path']+'/Collision/utensil')).GetCollisionEnabledAttr().Get() is False
                        for row in convex_spoons}
            cooked = collect_colliders(env, body_paths={row['id']: row['body_path'] for row in convex_spoons})
            dump(output/'native_spoon_colliders.json', cooked)
            expected = {path for row in convex_spoons for path in row['collider_paths']}
            actual = {row['path'] for row in cooked['colliders']}
            report['checks']['native_spoon_colliders'] = {
                'passed': cooked['status'] == 'complete' and actual == expected and all(disabled.values())
                    and all(row['collision_count'] == len(row['collider_paths']) == 220 for row in convex_spoons)
                    and all(row['status'] == 'verified_cooking_representation' and len(row['pieces']) == 1
                            and row['approximation'] == 'convexHull' for row in cooked['colliders']),
                'path': str(output/'native_spoon_colliders.json'),
                'sha256': digest(output/'native_spoon_colliders.json'),
                'active_collision_prims': len(actual), 'expected_collision_prims': len(expected),
                'old_SDF_collision_disabled': disabled,
                'scope': 'Official live native property-query inventory and cooking representation for every spoon piece; old disabled SDF absent. No state or USD layer writes.'}
        if metadata.get('volume_bodies'):
            from bench.runtime.native_volume_archive import NativeVolumeArchive
            volume_archive = NativeVolumeArchive(output/'native_volumes', env._task_reader)
            states.append(env.capture_state())
        from scripts.validation.kitchen_motion_diagnostics import NativeMotionDiagnostics
        motion = NativeMotionDiagnostics(task['evaluation']['thresholds'],
                                         task['evaluation']['physics_dt_s'])
        native = observation['task_state']
        motion.observe({'baseline': True, 'time_s': observation['time_s'],
            'native_step': native['native_step'], 'native_time_s': native['native_time_s'],
            'physics_dt_s': task['evaluation']['physics_dt_s'],
            'objects': observation['objects']})
        original_frame = env._task_reader.observer.on_frame

        def observed_frame(frame):
            motion.observe(frame)
            original_frame(frame)

        env._task_reader.observer.on_frame = observed_frame
        dump(output/'runtime_metadata.json', env.metadata())
        evaluator = frozen_evaluator.SequenceEvaluator(task)
        score = evaluator.update(observation)
        records = [{'kind': 'reset', 'observation': observation, 'score': score}]
        action = env.hold_current_targets()
        for index in range(round(args.duration/.02)):
            observation = env.step(action)
            if volume_archive is not None:
                states.append(env.capture_state())
            score = evaluator.update(observation)
            records.append({'kind': 'step', 'observation': observation, 'score': score})
            if index % 10 == 0:
                print(json.dumps({'event': 'support_probe', 'time_s': observation['time_s']}), flush=True)
        with (output/'physics.jsonl').open('w') as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        if diagnostic_spoons:
            from scripts.validation.kitchen_contact_snapshot import capture_contact_snapshot
            snapshot = capture_contact_snapshot(env, output/'native_contact_snapshot.npz')
            dump(output/'native_contact_snapshot.json', snapshot)
            report['native_contact_snapshot'] = {
                'metadata_path': str(output/'native_contact_snapshot.json'),
                'metadata_sha256': digest(output/'native_contact_snapshot.json'),
                'arrays_path': str(output/'native_contact_snapshot.npz'),
                'arrays_sha256': digest(output/'native_contact_snapshot.npz')}
        report['native_motion_diagnostics'] = motion.summary(window_s=.4)
        dump(output/'native_motion_summary.json', report['native_motion_diagnostics'])
        motion.save_raw(output/'native_motion.jsonl')
        if volume_archive is not None:
            import numpy as np
            np.savez_compressed(output/'states.npz', **{key: np.stack([state[key] for state in states])
                                                      for key in states[0]})
            volume_archive.close(actual_native_step=env.world.current_time_step_index, reason='finish_request')
            from scripts.validation.native_volume_archive_review import review_native_volumes
            evidence = {}
            report['checks']['full_native_volumes'] = review_native_volumes(output, env.metadata(), task,
                evidence, expected_source=project)
            report['full_native_volume_evidence_sha256'] = evidence
            require(report['checks']['full_native_volumes']['passed'], 'Independent native volume replay rejected')
        report['terminal_task_state'] = observation['task_state']
        require(observation['task_state']['complete'] and not observation['task_state']['errors'],
                'Diagnostic observer reported invalid native evidence')
        report['terminal_objects'] = observation['objects']
        from scripts.validation.kitchen_scene_evidence import support_evidence
        report['checks']['native_support'] = support_evidence(env.metadata(), records, task,
                                                              goal_reset=not args.canonical_reset)
        if args.require_native_velocity_limits:
            summary = report['native_motion_diagnostics']
            report['checks']['raw_terminal_native_velocity'] = {
                'passed': bool(summary['window_complete'] and summary['objects'] and all(
                    row['native']['linear_over_limit_count'] == 0 and
                    row['native']['angular_over_limit_count'] == 0
                    for row in summary['objects'].values())),
                'window_s': .4, 'full_native_frames': summary['window_frames'],
                'limits': summary['limits'],
                'scope': 'Every unfiltered native velocity sample; original public limits, no pose averaging.'}
        report['images'] = {}
        for camera in ('task', 'overview'):
            frame = env.render_rgbd(camera=camera)
            from PIL import Image
            import numpy as np
            image_path = output/(camera+'.png')
            Image.fromarray(np.asarray(frame['rgb'])[..., :3].astype('uint8')).save(image_path)
            report['images'][camera] = {'path': str(image_path), 'sha256': digest(image_path)}
            dump(output/(camera+'_calibration.json'), frame['camera_calibration'])
        report['checks']['no_reset_credit'] = {'passed': all(row['score']['success'] is False
            and row['score']['status'] == 'ongoing' and row['score']['evidence_status'] == 'complete'
            for row in records), 'observations': len(records)}
        if convex_spoons:
            require(report['checks']['native_spoon_colliders']['passed'], 'Native spoon collider cooking/inventory differs')
        require(report['checks']['native_support']['passed'], 'Native goal support did not pass')
        if args.require_native_velocity_limits:
            require(report['checks']['raw_terminal_native_velocity']['passed'],
                    'Full terminal native velocities exceed original limits or evidence is incomplete')
        require(report['checks']['no_reset_credit']['passed'], 'Assembled diagnostic reset earned credit or invalid evidence')
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        if volume_archive is not None and not volume_archive.closed:
            volume_archive.close(actual_native_step=env.world.current_time_step_index, reason='probe_failure')
        try:
            final_source = verify_if_snapshot(project, required_modules=MODULES,
                                             entrypoint_file=Path(__file__))
            require(final_source is not None, 'Missing final frozen-source verification')
            dump(output/'runtime_imports_final.json', final_source)
        except BaseException as exc:
            report['passed'] = False
            report['source_verification_failure'] = str(exc)
        dump(output/'qualification.json', report)
        print(json.dumps({'passed': report['passed'], 'output': str(output)}), flush=True)
        # Kit fast shutdown exits the process. Persist evidence and status
        # first, and preserve a nonzero process exit on a rejected probe.
        if env is not None:
            if not report['passed']:
                try:
                    raise RuntimeError('Kitchen goal support probe rejected; see qualification.json')
                finally:
                    env.close()
            env.close()
    print(json.dumps({'passed': report['passed'], 'output': str(output)}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--asset-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=1.2)
    parser.add_argument('--canonical-reset', action='store_true',
        help='Read the canonical asset source reset instead of a separately authored assembled goal')
    parser.add_argument('--require-native-velocity-limits', action='store_true',
        help='Additionally require every raw terminal native velocity to meet unchanged public limits')
    args = parser.parse_args()
    require(.4 <= args.duration <= 3 and abs(args.duration/.02-round(args.duration/.02)) < 1e-9,
            'Duration must be 0.4–3 seconds in 20 ms steps')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
