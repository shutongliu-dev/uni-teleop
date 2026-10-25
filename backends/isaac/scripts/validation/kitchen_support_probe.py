"""Read-only native kitchen support probe from a separately authored goal reset.

This deliberately starts with assembled objects. It tests geometry and gravity
support, never awards manipulation qualification, and never writes object poses
after reset. Canonical public-function validation is a separate entrypoint.
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
        'canonical_reset': False, 'object_writes_after_reset': False,
        'scope': 'Separately authored assembled reset; gravity, support and fit only.',
        'checks': {}}
    env = None
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
        require(manifest['presentation'] is True and manifest['task_id'] == args.task_id,
                'Support probe requires the separately declared presentation reset')
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
        from scripts.validation.kitchen_asset_identity import diagnostic_physics_identity
        report['checks']['asset_identity'] = diagnostic_physics_identity(
            Path(report['canonical_task_usd']['path']), usd, canonical_metadata, metadata)
        dump(output/'asset_identity.json', report['checks']['asset_identity'])
        require(report['checks']['asset_identity']['passed'],
                'Diagnostic asset differs from canonical physics beyond declared reset poses')
        observation = env.reset(seed=0, initial_state=initial)
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
            score = evaluator.update(observation)
            records.append({'kind': 'step', 'observation': observation, 'score': score})
            if index % 10 == 0:
                print(json.dumps({'event': 'support_probe', 'time_s': observation['time_s']}), flush=True)
        with (output/'physics.jsonl').open('w') as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        report['native_motion_diagnostics'] = motion.summary(window_s=.4)
        dump(output/'native_motion_summary.json', report['native_motion_diagnostics'])
        motion.save_raw(output/'native_motion.jsonl')
        report['terminal_task_state'] = observation['task_state']
        require(observation['task_state']['complete'] and not observation['task_state']['errors'],
                'Diagnostic observer reported invalid native evidence')
        report['terminal_objects'] = observation['objects']
        from scripts.validation.kitchen_scene_evidence import support_evidence
        report['checks']['native_support'] = support_evidence(env.metadata(), records, task,
                                                              goal_reset=True)
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
        require(report['checks']['native_support']['passed'], 'Native goal support did not pass')
        require(report['checks']['no_reset_credit']['passed'], 'Assembled diagnostic reset earned credit or invalid evidence')
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
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
    args = parser.parse_args()
    require(.4 <= args.duration <= 3 and abs(args.duration/.02-round(args.duration/.02)) < 1e-9,
            'Duration must be 0.4–3 seconds in 20 ms steps')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
