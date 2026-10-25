"""Bounded nist-task release smoke through canonical public robot functions.

Run from a complete frozen project snapshot. A bounded robot target hold checks
native observation/adjudication plumbing and four real RGB-D cameras. Saved
private evidence is examined after execution and never chooses an action.
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

from bench.isaacsim.release import digest, dump, task_inputs
from bench.paths import load_json, OUTPUT_ROOT
from bench.protocol import PolicyRegistry
from bench.runtime.functions import EpisodeFunctions  # Frozen frontend import.
from bench.protocol.task_referee import project_task_rules
from bench.runtime.native import NativeWorker
from bench.runtime.service import CreateEpisodeRequest, EpisodeSession
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.board_game_runtime_smoke import gpu_selection
from scripts.validation.card_room_task0 import camera_checks


from scenes.workrooms.mechanical_assembly.nist_composition import TASK_IDS

MODULES = ('scripts.validation.mechanical_nist_release', 'bench.runtime.service',
    'bench.runtime.functions', 'bench.runtime.native', 'bench.protocol.task_referee',
    'scripts.validation.card_room_task0', 'scripts.validation.board_game_runtime_smoke')
KIT = 'bench/apps/workroom.headless.kit'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=0., abs_tol=1.e-8)


def private_coverage(metadata, records, task, duration_s):
    """Verify saved native windows and task summaries without a strategy claim."""
    identity = task['id']
    dt = metadata['physics_dt_s']
    require(type(dt) in (int, float) and math.isfinite(dt) and 0 < dt <= .02,
            'Invalid native timestep')
    substeps = round(.02/dt)
    require(close(substeps*dt, .02), 'Native integration does not divide the control period')
    require(set(metadata['object_names']) == set(task['asset_metadata']['actors']), 'Native object inventory differs')
    require(len(records) == round(duration_s/.02)+1 and records[0].get('kind') == 'reset',
            'Expected exactly one reset and all acknowledged control observations')
    previous_step = previous_native = None
    samples = 0
    for index, record in enumerate(records):
        baseline = index == 0
        require(record.get('kind') == ('reset' if baseline else 'step'), 'Unexpected recording row')
        obs, score = record['observation'], record['score']
        require(score.get('success') is False and score.get('status') == 'ongoing'
                and score.get('evidence_status') == 'complete' and score.get('referee', {}).get('state') == 'ongoing'
                and not score.get('irrecoverable_reason'), 'Reset/hold produced success or invalid task evidence')
        require(close(obs['sim_time_s'], index*.02) and close(obs['time_s'], index*.02),
                'Control observations are missing, duplicated or out of order')
        require(set(obs['objects']) == set(task['asset_metadata']['actors']), 'Task bodies disappeared from observation')
        contacts = obs['contacts']
        sampling = contacts['sampling']
        count = 0 if baseline else substeps
        require(contacts['errors'] == [] and sampling['complete'] is True
                and sampling['protocol'] == 'native_rigid_contact_sampling_v1'
                and close(sampling['physics_dt_s'], dt)
                and sampling['sampled_substeps'] == count
                and sampling['native_end_step_index']-sampling['native_start_step_index'] == count
                and close(sampling['end_sim_time_s'], obs['sim_time_s'])
                and close(sampling['end_sim_time_s']-sampling['start_sim_time_s'], count*dt)
                and close(sampling['native_end_time_s']-sampling['native_start_time_s'], count*dt),
                'Incomplete or inconsistent native contact sampling window')
        if baseline:
            require(not contacts['rigid'], 'Reset baseline must not invent sampled contacts')
        else:
            require(sampling['native_start_step_index'] == previous_step
                    and close(sampling['native_start_time_s'], previous_native)
                    and close(sampling['start_sim_time_s'], (index-1)*.02),
                    'Native sampling windows are not consecutive')
        previous_step = sampling['native_end_step_index']
        previous_native = sampling['native_end_time_s']
        samples += count
        state = obs['task_state']
        require(state.get('errors') == [], 'Task observer reported invalid native evidence')
        require(close(state['time_s'], obs['sim_time_s']), 'Task summary has a stale control time')
        require(state.get('complete') is True and state['baseline'] is baseline
                and state['native_step'] == previous_step and close(state['native_time_s'], previous_native),
                'NIST-derived mechanical observer does not match the native clock')
    require(samples == round(duration_s/dt), 'Native hold duration lacks full observation coverage')
    return {'passed': True, 'control_observations': len(records), 'native_substeps': samples,
        'physics_dt_s': dt, 'elapsed_sim_s': duration_s, 'task_id': identity,
        'false_success': False, 'task_success_claimed': False,
        'scope': 'Consecutive saved control summaries and complete native sampling receipts; robot target hold only.'}


def run(args):
    output = args.output.resolve()
    require(output.is_relative_to(OUTPUT_ROOT), 'Store smoke evidence under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    calls, session = [], None
    report = {'schema': 'mechanical_nist_release_public_functions_smoke.v1', 'passed': False,
        'task_id': args.task_id, 'requested_hold_s': args.duration, 'model_calls': 0,
        'task_success_claimed': False, 'http_transport_tested': False,
        'object_pose_overrides': False, 'privileged_state_used_for_actions': False,
        'scope': 'Frozen task initialization, public robot hold, read-only adjudication and four actual RGB-D cameras.',
        'started_at': datetime.now(timezone.utc).isoformat(), 'checks': {}}
    try:
        source = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
        require(source is not None and Path.cwd().resolve() == project,
                'Run from the root of a complete frozen source snapshot')
        manifest = load_json(project/'source_snapshot_manifest.json')
        require((project/KIT).is_file() and manifest['files_sha256'].get(KIT) == digest(project/KIT)
                and Path(os.environ.get('ISAAC_BENCH_EXPERIENCE', '')).resolve() == project/KIT,
                'The headless experience must be included and selected from this frozen snapshot')
        dump(output/'runtime_imports.json', source)
        report['gpu'] = gpu_selection(args.gpu_uuid)
        report['environment'] = {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'ISAAC_BENCH_SINGLE_CUDA_DEVICE', 'ISAAC_ALLOW_UNVALIDATED_DRIVER',
            'ISAAC_BENCH_EXPERIENCE')}
        report['source_manifest_sha256'] = digest(project/'source_snapshot_manifest.json')
        report['release'] = str(args.release.resolve())
        report['release_manifest_sha256'] = digest(args.release/'release_manifest.json')
        entry, scene, task, initial = task_inputs(args.release, args.task_id, seed=args.seed)
        require(task['schema_version'] == 'mechanical-nist-task-v1', 'Expected a published nist task')
        config = {'task_id': entry['id'], 'seed': args.seed, 'duration_s': args.duration,
                  'script_sha256': digest(Path(__file__))}
        request = CreateEpisodeRequest(task_id=entry['id'], seed=args.seed,
            agent_id='operator_mechanical_nist_release_smoke', agent_version='1',
            agent_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            track='direct_joints', budgets={'sim_time_s': args.duration+.1, 'wall_time_s': 900.,
                'max_actions': 2, 'max_model_calls': 1, 'max_action_sim_s': args.duration})
        session = EpisodeSession(request, output/'episode', release=args.release, device='cuda:0',
                                 registry=PolicyRegistry(), worker_factory=NativeWorker)
        report['episode_id'] = session.spec.episode_id
        metadata = session.metadata
        roots = metadata['runtime_robot_base_check']['articulations']
        require(len(metadata['joint_names']) == 29 and len(roots) == 1 and roots[0]['dof_count'] == 29
                and roots[0]['fixed_base'] is True and set(metadata['object_names']) == set(task['asset_metadata']['actors']),
                'Expected one fixed 29 DOF robot and the frozen task object inventory')
        report['checks']['robot_inventory'] = {'passed': True, 'articulations': roots,
            'physical_joint_count': 29, 'object_names': metadata['object_names']}

        def invoke(function, arguments):
            reply = session.functions.invoke({'request_id': 'nist_smoke_'+str(len(calls)),
                'function': function, 'arguments': arguments}).model_dump(mode='json')
            calls.append({'function': function, 'reply': reply})
            require(reply['ok'], 'Public function failed: '+function+': '+str(reply.get('error')))
            return reply['result']

        requirements = session.task_requirements()
        static = project_task_rules(task)
        require(all(requirements.get(k) == v for k, v in static.items()),
                'Public rules differ from the frozen episode task')
        dump(output/'public_task.json', requirements)
        report['checks']['public_rules'] = {'passed': True, 'schema': requirements['public_rules']['schema'],
                                           'instruction': requirements['instruction'], 'episode_seed': args.seed}
        dump(output/'function_catalog.json', session.functions.catalog().model_dump(mode='json'))
        dump(output/'robot_guide.json', invoke('get_robot', {}))
        observation = invoke('observe', {})
        dump(output/'initial_observation.json', observation)
        report['checks']['initial_cameras'] = camera_checks(session, invoke, observation)
        require(session.clock.sim_time_s == 0., 'Read-only calls advanced physical time')
        require(session.score['success'] is False, 'Reset incorrectly satisfied the task')
        invoke('hold', {'action_id': 'nist_release_hold', 'observation_sequence': observation['sequence'],
                        'duration_s': args.duration})
        deadline, last_update = time.monotonic()+240., time.monotonic()
        while True:
            status = invoke('get_action', {'action_id': 'nist_release_hold'})
            if status['status'] not in ('accepted', 'running'):
                break
            require(time.monotonic() < deadline, 'Bounded hold timed out')
            if time.monotonic()-last_update > 20:
                print(json.dumps({'event': 'holding', 'elapsed_sim_s': session.clock.sim_time_s}), flush=True)
                last_update = time.monotonic()
            time.sleep(.1)
        require(status['status'] == 'completed' and close(session.clock.sim_time_s, args.duration),
                'Hold did not complete for the requested physical duration: '+str(status))
        observation = invoke('observe', {})
        dump(output/'final_observation.json', observation)
        report['checks']['final_cameras'] = camera_checks(session, invoke, observation)
        require(all(session.task_requirements().get(k) == v for k, v in static.items()),
                'Holding changed the static public requirements')
        require(session.score['success'] is False, 'Passive hold incorrectly earned task success')
        report['final_score'] = session.score
        invoke('finish_episode', {})
        report['episode_result'] = invoke('get_result', {})
        result = session.result.model_dump(mode='json')
        require(result['outcome'] == 'cancelled' and result['model_calls'] == 0,
                'Smoke must finish voluntarily without model calls or false success')
        report['canonical_physics_valid'] = result['physics_valid']
        require(result['physics_valid'] is True, 'Canonical physical safety audit failed')
        private = session.directory/'private'
        records = [json.loads(line) for line in (private/'physics.jsonl').read_text().splitlines()]
        report['checks']['private_observer_coverage'] = private_coverage(metadata, records, task, args.duration)
        report['evidence_sha256'] = {name: digest(private/name) for name in
            ('physics.jsonl', 'states.npz', 'metadata.json', 'task.json', 'scene.json', 'final_score.json', 'physics_audit.json')}
        from scripts.validation.mechanical_nist_support import support_evidence
        report['checks']['native_support'] = support_evidence(metadata, records, task)
        require(report['checks']['native_support']['passed'], 'Native reset support validation failed')
        report['terminal_task_state'] = records[-1]['observation']['task_state']
        report['terminal_objects'] = records[-1]['observation']['objects']
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
            dump(output/'runtime_imports_final.json', source)
        except BaseException as exc:
            report['passed'] = False
            report['source_verification_failure'] = str(exc)
        dump(output/'public_function_calls.json', calls)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        dump(output/'qualification.json', report)
    print(json.dumps({'output': str(output), 'passed': report['passed'], 'scope': report['scope']}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--task-id', required=True, choices=TASK_IDS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--duration', type=float, default=1.2)
    args = parser.parse_args()
    if not .4 <= args.duration <= 3. or not close(args.duration/.02, round(args.duration/.02)):
        parser.error('Duration must be 0.4–3.0 seconds and an exact multiple of 20 ms')
    if not 0 <= args.seed < 2**32:
        parser.error('Seed must be an unsigned 32-bit integer')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
