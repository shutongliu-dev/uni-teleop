"""Bounded study initialization, public robot hold and native evidence readback.

Run from a complete source snapshot. Private evidence is inspected only after
execution; it never chooses robot actions. This does not qualify manipulation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import zipfile

from bench.isaacsim.release import digest, dump, load_evaluator, task_inputs
from bench.paths import OUTPUT_ROOT
from bench.protocol import PolicyRegistry
from bench.protocol.task_referee import project_task_rules
from bench.runtime.functions import EpisodeFunctions  # Frozen frontend import.
from bench.runtime.native import NativeWorker
from bench.runtime.service import CreateEpisodeRequest, EpisodeSession
from scenes.workrooms.study.composition import TASK_IDS
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.board_game_runtime_smoke import gpu_selection
from scripts.validation.card_room_task0 import camera_checks

MODULES = ('scripts.validation.study_task_release', 'bench.runtime.service',
    'bench.runtime.functions', 'bench.runtime.native', 'bench.protocol.task_referee',
    'scripts.validation.card_room_task0', 'scripts.validation.board_game_runtime_smoke')
KIT = 'bench/apps/workroom.headless.kit'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, *, abs_tol=1.e-8):
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=0., abs_tol=abs_tol))


def _reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def read_json(value):
    return json.loads(value, parse_constant=_reject_constant)


def ongoing(score):
    return (score.get('success') is False and score.get('referee', {}).get('state') == 'ongoing'
            and score.get('referee', {}).get('terminal') is False)


def native_coverage(private, task, duration_s, evaluator_factory):
    """Replay every frozen normalized frame and independently check its receipt."""
    root = Path(private) / 'task_runtime'
    configuration = task['evaluation']['configuration']
    dt = task['evaluation']['physics_dt_s']
    expected = round(duration_s / dt)
    require(expected > 0 and close(expected * dt, duration_s), 'Hold does not contain whole native steps')
    receipt = read_json((root / 'study_sampling_receipt.json').read_text())
    require(receipt['protocol'] == 'fresh_native_substep_contact_dual_projection_v1'
            and receipt['all_samples_consumed_once'] is True
            and type(receipt['acquisitions']) is int and type(receipt['deliveries']) is int
            and receipt['acquisitions'] == receipt['deliveries'] == expected
            and type(receipt['reset_epoch']) is int
            and receipt['raw_arrays_reused_across_native_steps'] is False
            and receipt['state_captures_reused'] is False
            and len(receipt['samples']) == expected, 'Incomplete native acquisition/delivery receipt')
    evaluator = evaluator_factory(task, trusted_normalized_input=True)
    expected_names = [f'{index:09d}.json' for index in range(expected + 1)]
    baseline = previous = None
    with zipfile.ZipFile(root / 'study_native_evidence.zip') as archive:
        require(archive.namelist() == expected_names, 'Native evidence archive has missing, duplicate or extra frames')
        for index, name in enumerate(expected_names):
            frame = read_json(archive.read(name))
            require(frame['schema'] == 'study_native_observation_v1'
                    and frame['configuration_sha256'] == configuration['configuration_sha256']
                    and frame['step_index'] == index and type(frame['step_index']) is int
                    and close(frame['time_s'], index * dt) and close(frame['physics_dt_s'], dt)
                    and set(frame['objects']) == set(task['evaluation']['body_paths'])
                    and all(frame[key] is True for key in ('sampling_complete', 'contacts_complete',
                        'physics_valid', 'robot_only_actions'))
                    and frame['object_pose_writes'] is False and frame['budget_exhausted'] is False,
                    'Incomplete, stale or foreign normalized native frame')
            if index == 0:
                baseline = frame
                require(type(frame['native_physics_step_index']) is int
                        and close(frame['native_world_time_s'], frame['native_world_time_s'])
                        and frame['contacts'] == [] and type(frame['robot_action_count']) is int
                        and frame['robot_action_count'] == 0,
                        'Invalid unscored native reset baseline')
            else:
                sample = receipt['samples'][index - 1]
                require(frame['episode_id'] == baseline['episode_id']
                        and frame['native_physics_step_index'] == baseline['native_physics_step_index'] + index
                        and type(frame['native_physics_step_index']) is int
                        # Isaac stores dt as float32. Validate each increment so
                        # its representational error cannot accumulate here.
                        and close(frame['native_world_time_s'], previous['native_world_time_s'] + dt,
                                  abs_tol=1.e-10)
                        and type(frame['robot_action_count']) is int and frame['robot_action_count'] == 1,
                        'Native frame clock, episode or action sequence is not consecutive')
                require(all(type(sample[key]) is int for key in ('sequence', 'reset_epoch',
                            'native_step_index', 'acquisition_count', 'delivery_count'))
                        and sample['sequence'] == index and sample['reset_epoch'] == receipt['reset_epoch']
                        and sample['native_step_index'] == frame['native_physics_step_index']
                        and close(sample['native_time_s'], frame['native_world_time_s'])
                        and close(sample['public_time_before_s'], previous['time_s'])
                        and close(sample['planned_public_time_s'], frame['time_s'])
                        and close(sample['consumed_public_time_s'], frame['time_s'])
                        and sample['acquisition_count'] == sample['delivery_count'] == 1
                        and sample['readonly_verified'] is True,
                        'Native frame lacks a unique same-step read-only acquisition/delivery')
            evaluator.update(frame)
            require(ongoing(evaluator.inspect_conditions()), 'Passive initialization/hold replay produced a physical verdict')
            previous = frame
    evaluator.finish('cancelled')
    # The saved score is JSON: tuple-valued geometry is serialized as arrays.
    replayed = read_json(json.dumps(evaluator.inspect_conditions(), allow_nan=False))
    saved = read_json((Path(private) / 'final_score.json').read_text())
    require(replayed == saved and ongoing(saved) and saved.get('status') == 'incomplete'
            and saved.get('reason') == 'cancelled', 'Frozen replay differs from the sealed native result')
    return {'passed': True, 'native_substeps': expected, 'normalized_frames': expected + 1,
        'physics_dt_s': dt, 'elapsed_sim_s': duration_s, 'frozen_evaluator_replay_matches': True,
        'false_success': False, 'task_success_claimed': False,
        'archive_sha256': digest(root / 'study_native_evidence.zip'),
        'receipt_sha256': digest(root / 'study_sampling_receipt.json')}


def control_coverage(metadata, records, task, duration_s):
    """Check the independent generic contact projection at control boundaries."""
    dt = metadata['physics_dt_s']
    count = round(.02 / dt)
    require(close(dt, task['evaluation']['physics_dt_s']) and close(count * dt, .02) and count > 0,
            'Native period differs from frozen task or does not divide the 20 ms control period')
    bodies = set(task['evaluation']['body_paths'])
    require(set(metadata['object_names']) == bodies, 'Native object inventory differs from frozen task')
    require(len(records) == round(duration_s / .02) + 1, 'Missing or extra control observations')
    previous_step = previous_time = None
    for index, record in enumerate(records):
        baseline = index == 0
        require(record['kind'] == ('reset' if baseline else 'step'), 'Unexpected control recording row')
        require(baseline or record['action_id'] == 'study_release_hold', 'Unexpected public action identity')
        obs, score = record['observation'], record['score']
        require(ongoing(score), 'Native reset/hold produced a physical task verdict')
        require(close(obs['sim_time_s'], index * .02) and close(obs['time_s'], index * .02)
                and set(obs['objects']) == bodies, 'Stale or incomplete control observation')
        contacts = obs['contacts']
        window = contacts['sampling']
        samples = 0 if baseline else count
        require(contacts['errors'] == [] and window['complete'] is True
                and window['protocol'] == 'native_rigid_contact_sampling_v1'
                and close(window['physics_dt_s'], dt) and window['sampled_substeps'] == samples
                and window['native_end_step_index'] - window['native_start_step_index'] == samples
                and close(window['end_sim_time_s'], obs['sim_time_s'])
                and close(window['end_sim_time_s'] - window['start_sim_time_s'], samples * dt)
                and close(window['native_end_time_s'] - window['native_start_time_s'], samples * dt),
                'Incomplete generic native contact window')
        if baseline:
            require(contacts['rigid'] == [], 'Reset invented sampled contacts')
        else:
            require(window['native_start_step_index'] == previous_step
                    and close(window['native_start_time_s'], previous_time)
                    and close(window['start_sim_time_s'], (index - 1) * .02),
                    'Generic contact windows are not consecutive')
        previous_step, previous_time = window['native_end_step_index'], window['native_end_time_s']
    return {'passed': True, 'control_observations': len(records),
        'native_substeps': round(duration_s / dt), 'physics_dt_s': dt}


def _static_requirements(value):
    return {key: val for key, val in value.items() if key not in ('referee', 'episode')}


def run(args):
    output = args.output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT), 'Store native smoke evidence under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    calls, session = [], None
    report = {'schema': 'study_task_release_public_functions_smoke.v1', 'passed': False,
        'task_id': args.task_id, 'requested_hold_s': args.duration, 'model_calls': 0,
        'task_success_claimed': False, 'http_transport_tested': False,
        'object_pose_overrides': False, 'privileged_state_used_for_actions': False,
        'scope': 'Frozen task initialization, acknowledged robot drive hold, native readback and four RGB-D cameras.',
        'started_at': datetime.now(timezone.utc).isoformat(), 'checks': {}}
    try:
        source = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
        require(source is not None and Path.cwd().resolve() == project,
                'Run from the root of a complete frozen source snapshot')
        manifest = read_json((project / 'source_snapshot_manifest.json').read_text())
        require((project / KIT).is_file() and manifest['files_sha256'].get(KIT) == digest(project / KIT)
                and Path(os.environ.get('ISAAC_BENCH_EXPERIENCE', '')).resolve() == project / KIT,
                'The headless experience must be included and selected from this frozen snapshot')
        dump(output / 'runtime_imports.json', source)
        report['gpu'] = gpu_selection(args.gpu_uuid)
        report['environment'] = {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'ISAAC_BENCH_SINGLE_CUDA_DEVICE', 'ISAAC_ALLOW_UNVALIDATED_DRIVER',
            'ISAAC_BENCH_EXPERIENCE')}
        report['source_manifest_sha256'] = digest(project / 'source_snapshot_manifest.json')
        report['release'] = str(args.release.resolve())
        report['release_manifest_sha256'] = digest(args.release / 'release_manifest.json')
        entry, scene, task, _ = task_inputs(args.release, args.task_id, seed=args.seed)
        require(task['schema_version'] == 'study-task-v1', 'Expected a frozen study task')
        report['parent_environment'] = scene['workroom']
        report['robot_configuration'] = scene['robot']
        report['robot_stand'] = scene['fixtures']
        inventory = read_json((args.release / 'asset_file_hashes.json').read_text())
        report['robot_stand_files_sha256'] = {path: row['sha256'] for path, row in inventory.items()
            if any(Path(path).is_relative_to(fixture['asset_root']) for fixture in scene['fixtures'])}
        require(report['robot_stand_files_sha256'], 'Missing actual robot stand dependency inventory')
        config = {'task_id': entry['id'], 'seed': args.seed, 'duration_s': args.duration,
                  'script_sha256': digest(Path(__file__))}
        request = CreateEpisodeRequest(task_id=entry['id'], seed=args.seed,
            agent_id='operator_study_task_release_smoke', agent_version='1',
            agent_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            track='direct_joints', budgets={'sim_time_s': args.duration + .1, 'wall_time_s': 900.,
                'max_actions': 2, 'max_model_calls': 1, 'max_action_sim_s': args.duration})
        session = EpisodeSession(request, output / 'episode', release=args.release, device='cuda:0',
                                 registry=PolicyRegistry(), worker_factory=NativeWorker)
        report['episode_id'] = session.spec.episode_id
        metadata = session.metadata
        roots = metadata['runtime_robot_base_check']['articulations']
        require(len(metadata['joint_names']) == 29 and len(roots) == 1 and roots[0]['dof_count'] == 29
                and roots[0]['fixed_base'] is True
                and set(metadata['object_names']) == set(task['evaluation']['body_paths']),
                'Expected one fixed 29 DOF robot and all frozen task bodies')
        report['checks']['robot_inventory'] = {'passed': True, 'articulations': roots,
            'physical_joint_count': 29, 'object_names': metadata['object_names']}

        def invoke(function, arguments):
            reply = session.functions.invoke({'request_id': 'study_smoke_' + str(len(calls)),
                'function': function, 'arguments': arguments}).model_dump(mode='json')
            calls.append({'function': function, 'reply': reply})
            require(reply['ok'], 'Public function failed: ' + function + ': ' + str(reply.get('error')))
            return reply['result']

        requirements = invoke('get_task', {})
        require(_static_requirements(requirements) == project_task_rules(task),
                'Public rules differ from the frozen episode task')
        dump(output / 'public_task.json', requirements)
        report['checks']['public_rules'] = {'passed': True, 'rules_sha256': requirements['public_rules_sha256']}
        dump(output / 'function_catalog.json', session.functions.catalog().model_dump(mode='json'))
        dump(output / 'robot_guide.json', invoke('get_robot', {}))
        observation = invoke('observe', {})
        dump(output / 'initial_observation.json', observation)
        report['checks']['initial_cameras'] = camera_checks(session, invoke, observation)
        require(session.clock.sim_time_s == 0. and ongoing(session.score),
                'Read-only initialization advanced time or produced a physical verdict')
        invoke('hold', {'action_id': 'study_release_hold', 'observation_sequence': observation['sequence'],
                        'duration_s': args.duration})
        deadline, last_update = time.monotonic() + 240., time.monotonic()
        while True:
            status = invoke('get_action', {'action_id': 'study_release_hold'})
            if status['status'] not in ('accepted', 'running'):
                break
            require(time.monotonic() < deadline, 'Bounded hold timed out')
            if time.monotonic() - last_update > 20:
                print(json.dumps({'event': 'holding', 'elapsed_sim_s': session.clock.sim_time_s}), flush=True)
                last_update = time.monotonic()
            time.sleep(.1)
        require(status['status'] == 'completed' and close(session.clock.sim_time_s, args.duration),
                'Hold did not complete for the requested physical duration: ' + str(status))
        observation = invoke('observe', {})
        dump(output / 'final_observation.json', observation)
        report['checks']['final_cameras'] = camera_checks(session, invoke, observation)
        require(_static_requirements(invoke('get_task', {})) == _static_requirements(requirements)
                and close(session.clock.sim_time_s, args.duration) and ongoing(session.score),
                'Read-only final queries changed time, rules or the expected ongoing verdict')
        report['hold_score'] = session.score
        invoke('finish_episode', {})
        report['episode_result'] = invoke('get_result', {})
        result = session.result.model_dump(mode='json')
        require(result['outcome'] == 'cancelled' and result['model_calls'] == 0,
                'Smoke must finish voluntarily without model calls or false success')
        require(result['physics_valid'] is True, 'Canonical physical safety audit failed')
        report['canonical_physics_valid'] = result['physics_valid']
        private = session.directory / 'private'
        records = [read_json(line) for line in (private / 'physics.jsonl').read_text().splitlines()]
        report['checks']['control_coverage'] = control_coverage(metadata, records, task, args.duration)
        report['checks']['native_coverage'] = native_coverage(private, task, args.duration, load_evaluator(args.release))
        report['evidence_sha256'] = {name: digest(private / name) for name in (
            'physics.jsonl', 'states.npz', 'metadata.json', 'task.json', 'scene.json',
            'final_score.json', 'physics_audit.json')}
        report['checks']['expected_non_success'] = {'passed': True, 'outcome': result['outcome']}
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        if session is not None:
            try:
                session.finish()
            except BaseException as exc:
                report['passed'] = False
                report['shutdown_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
                session.worker.close(force=True)
            report['native_worker_exit_code'] = session.worker.process.returncode
            report['passed'] &= session.worker.process.returncode == 0
        try:
            source = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
            require(source is not None, 'Missing final frozen-source verification')
            dump(output / 'runtime_imports_final.json', source)
        except BaseException as exc:
            report['passed'] = False
            report['source_verification_failure'] = str(exc)
        dump(output / 'public_function_calls.json', calls)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        dump(output / 'qualification.json', report)
    print(json.dumps({'output': str(output), 'passed': report['passed'], 'scope': report['scope']}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--task-id', required=True, choices=TASK_IDS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--duration', type=float, default=.3)
    args = parser.parse_args()
    if not .3 <= args.duration <= 1. or not close(args.duration / .02, round(args.duration / .02)):
        parser.error('Duration must be 0.3–1.0 seconds and an exact multiple of 20 ms')
    if not 0 <= args.seed < 2**32:
        parser.error('Seed must be an unsigned 32-bit integer')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
