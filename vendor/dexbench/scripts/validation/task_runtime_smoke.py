"""Bounded native task integration check through canonical HTTP handlers.

Uses an in-process ASGI HTTP client and the real isolated Isaac worker. Robot
targets derive only from public proprioception. No task strategy or model calls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from bench.paths import OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check(value, message):
    if not value:
        raise RuntimeError(message)


def optional_wall_timeout(value):
    """CLI spelling for the public Optional episode wall-time budget."""
    if value.lower() == 'none':
        return None
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected positive finite seconds or none') from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError('Expected positive finite seconds or none')
    return seconds


def wait_for_public_ready(read_state, hold, *, max_wait_sim_s, hold_duration_s=.2):
    """Advance public acknowledged-drive holds while the referee advertises settling."""
    check(math.isfinite(max_wait_sim_s) and 0 < max_wait_sim_s <= 100., 'Invalid readiness budget')
    check(math.isfinite(hold_duration_s) and .02 <= hold_duration_s <= 1., 'Invalid readiness hold duration')
    state = read_state()
    start, current = state['sim_time_s'], state['current_player_id']
    records = []
    while state['public_progress'].get('status') == 'reset_pending':
        check(state['status'] == 'active' and state['current_player_id'] == current,
              'Referee ended or changed player during reset settling')
        check(state['sim_time_s'] - start + hold_duration_s <= max_wait_sim_s + 1.e-9,
              'Public referee did not become ready within the smoke settling budget')
        before = state['sim_time_s']
        hold('reset_wait_' + str(len(records)), hold_duration_s)
        state = read_state()
        check(state['sim_time_s'] > before, 'Public settling hold did not advance acknowledged time')
        records.append({'before_sim_time_s': before, 'after_sim_time_s': state['sim_time_s'],
                        'public_status': state['public_progress'].get('status')})
    check(state['status'] == 'active' and state['current_player_id'] == current,
          'Referee ended or changed player before motion admission')
    return {'elapsed_sim_s': state['sim_time_s'] - start, 'holds': records,
            'final_public_status': state['public_progress'].get('status')}


def run(args):
    import numpy as np
    from fastapi.testclient import TestClient
    from bench.isaacsim.release import digest, task_inputs, validate
    from bench.runtime.server import create_app
    from bench.runtime.service import AgentService
    from scripts.validation.board_game_runtime_smoke import gpu_selection

    project = Path(__file__).resolve().parents[2]
    source = verify_if_snapshot(project, required_modules=(
        'scripts.validation.task_runtime_smoke', 'bench.runtime.service', 'bench.runtime.server'),
        entrypoint_file=Path(__file__))
    check(source is not None and Path.cwd().resolve() == project, 'Run from a complete frozen source root')
    output = args.output.resolve()
    check(output.is_relative_to(OUTPUT_ROOT), 'Runtime output belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'native_task_runtime_smoke.v1', 'passed': False,
        'task_id': args.task_id, 'seed': args.seed, 'model_calls': 0,
        'transport': 'in_process_ASGI_HTTP_with_real_isolated_native_worker',
        'task_success_qualified': False, 'physical_fidelity_qualified': False,
        'object_state_writes': False, 'private_state_used_for_targets': False,
        'source_snapshot': str(project), 'release': str(args.release.resolve()),
        'source_manifest_sha256': digest(project/'source_snapshot_manifest.json'),
        'release_manifest_sha256': digest(args.release/'release_manifest.json'),
        'wall_time_budget_s': args.wall_timeout_s,
        'action_wall_timeout_s': args.action_wall_timeout_s,
        'scope': 'Task release, robot control, authenticated ownership, public RGB-D and private evidence wiring only.'}
    service = session = None
    calls = []
    try:
        report['gpu'] = gpu_selection(args.gpu_uuid)
        dump(output/'runtime_imports.json', source)
        manifest = validate(args.release)
        entry, scene, task, _ = task_inputs(args.release, args.task_id, seed=args.seed)
        kind = entry.get('task_kind', manifest.get('task_kind'))
        check(kind != 'board_game', 'Use the existing tabletop smoke for board games')
        turn = kind == 'turn_game'
        service = AgentService(output/'episodes', release=args.release, device='cuda:0')
        with TestClient(create_app(service=service)) as client:
            def request(method, path, payload=None, headers=None, expected=None):
                response = client.request(method, path, json=payload, headers=headers)
                value = response.json()
                saved = deepcopy(value)
                if isinstance(saved, dict):
                    for row in saved.get('players', []):
                        row['session_token'] = '<redacted>'
                calls.append({'method': method, 'path': path, 'request': payload,
                              'status': response.status_code, 'response': saved})
                dump(output/'http_calls.json', calls)
                if expected is not None:
                    check(response.status_code == expected, f'{path}: {response.status_code} {saved}')
                else:
                    response.raise_for_status()
                return value

            request('GET', '/tasks')
            request('GET', '/interfaces?task_id='+args.task_id)
            policies = request('GET', '/policies?task_id='+args.task_id)
            check(len(policies) == 1, 'Expected one task-bound shared controller')
            policy = policies[0]
            create = {'task_id': args.task_id, 'seed': args.seed, 'robot_id': scene['robot']['hand_id'],
                'agent_id': 'native_task_integration', 'agent_version': '1',
                'agent_config_sha256': digest(Path(__file__)), 'track': 'system0',
                'policy_id': policy['policy_id'], 'policy_version': policy['version'],
                'budgets': {'sim_time_s': args.ready_timeout_s+2. if turn else 2., 'wall_time_s': args.wall_timeout_s,
                            'max_actions': math.ceil(args.ready_timeout_s/.2)+12 if turn else 12,
                            'max_model_calls': 1, 'max_action_sim_s': 1.}}
            if turn:
                request('POST', '/episodes', create, expected=403)
            print(json.dumps({'event': 'native_starting', 'task': args.task_id, 'seed': args.seed}), flush=True)
            dump(output/'progress.json', {'phase': 'native_starting'})
            created = request('POST', '/turn-games' if turn else '/episodes', create, expected=201)
            episode = created['episode_id']
            session = service.sessions[episode]
            report['episode_id'] = episode
            headers = None
            if turn:
                current = created['turn_game']['current_player_id']
                credentials = {row['binding']['player_id']: row for row in created['players']}
                selected = credentials[current]
                headers = {'Authorization': 'Bearer '+selected['session_token']}
                side = selected['binding']['hand']
                endpoint = f'/turn-games/{episode}/players/{current}'
                request('GET', endpoint+'/state', expected=403)
                request('GET', f'/episodes/{episode}/functions', expected=403)
                other = next(p for p in credentials if p != current)
                inactive = {'Authorization': 'Bearer '+credentials[other]['session_token']}
                before = session.clock.action_attempts
                request('POST', f'/turn-games/{episode}/players/{other}/call',
                    {'function': 'hold', 'arguments': {'action_id': 'denied', 'observation_sequence': 0,
                                                      'duration_s': .02}}, inactive, expected=403)
                check(session.clock.action_attempts == before, 'Inactive command spent the active player budget')
                report['player_id'], report['hand'] = current, side
            else:
                endpoint = f'/episodes/{episode}'

            def invoke(function, arguments=None):
                reply = request('POST', endpoint+'/call', {'function': function,
                    'request_id': f'integration-{len(calls)}', 'arguments': arguments or {}}, headers)
                check(reply['ok'], f'{function}: {reply.get("error")}')
                return reply['result']

            request('GET', endpoint+'/functions', headers=headers)
            if 'referee' in task.get('public_rules', {}):
                report['initial_public_task'] = invoke('get_task')
            guide = invoke('get_robot')
            initial = invoke('observe')
            side = side if turn else next(iter(guide['hands']))
            report['robot_id'] = guide['robot_id']
            report['namespaces'] = list(guide['hands'])
            report['camera_ids'] = [c['camera_id'] for c in initial['cameras']]
            for camera in initial['cameras']:
                for channel in ('rgb', 'depth'):
                    response = client.get(camera[channel+'_uri'], headers=headers)
                    check(response.status_code == 200, 'Public media retrieval failed')
                    check(hashlib.sha256(response.content).hexdigest() == camera[channel+'_sha256'], 'Media hash mismatch')
            check(len(initial['cameras']) >= 4, 'Missing configured room or robot cameras')
            geometries = initial['proprioception'].get('hand_geometries')
            geometry = geometries[side] if geometries else initial['proprioception']['geometry']
            objective = {'wrist': geometry['wrist'], 'fingertip_frame_id': guide['hands'][side]['frame_id']}
            ik = invoke('query_ik', {'observation_sequence': initial['sequence'],
                **({'hand_geometries': {side: objective}} if geometries else {'geometry': objective})})
            check(ik['solution_found'], 'Read-only robot IK failed')

            def action(name, function, arguments):
                observation = invoke('observe')
                values = dict(arguments, action_id=name, observation_sequence=observation['sequence'])
                invoke(function, values)
                deadline = time.monotonic()+args.action_wall_timeout_s
                while True:
                    result = invoke('get_action', {'action_id': name})
                    if result['status'] not in ('accepted', 'running'):
                        break
                    check(time.monotonic() < deadline, 'Action deadline expired')
                    time.sleep(.2)
                check(result['status'] == 'completed', f'Action {name}: {result}')
                after = invoke('observe')
                if function == 'hold' and arguments.get('mode') == 'acknowledged_drive':
                    before_targets = observation['proprioception']['commanded_joint_positions_rad']
                    after_targets = after['proprioception']['commanded_joint_positions_rad']
                    check(before_targets is not None and before_targets == after_targets,
                          'Fixed-drive hold changed acknowledged targets')
                    report.setdefault('acknowledged_hold_history', []).append({
                        'action_id': name, 'before_sequence': observation['sequence'],
                        'after_sequence': after['sequence'],
                        'before_targets_rad': before_targets, 'after_targets_rad': after_targets})
                invoke(function, values)
                replay = invoke('observe')
                check(after['sequence'] == replay['sequence'], 'Replayed action advanced physics')
                print(json.dumps({'event': 'action_complete', 'name': name, 'sim_time_s': after['sim_time_s']}), flush=True)
                dump(output/'progress.json', {'phase': name, 'sim_time_s': after['sim_time_s']})
                return after

            if turn:
                report['readiness_wait'] = wait_for_public_ready(
                    lambda: request('GET', endpoint+'/state', headers=headers)['turn_game'],
                    lambda name, duration: action(name, 'hold', {'duration_s': duration,
                                                               'mode': 'acknowledged_drive'}),
                    max_wait_sim_s=args.ready_timeout_s)
            action('initial_hold', 'hold', {'duration_s': .04})
            current_observation = invoke('observe')
            current_geometry = (current_observation['proprioception']['hand_geometries'][side] if geometries
                                else current_observation['proprioception']['geometry'])
            target = deepcopy(current_geometry['wrist'])
            target['position_m'][2] += .002
            moved = action('small_wrist_motion', 'move_wrist', {'hand': side, 'wrist': target,
                           'position_tolerance_m': .0005, 'timeout_sim_s': 1.})
            moved_geometry = (moved['proprioception']['hand_geometries'][side] if geometries
                              else moved['proprioception']['geometry'])
            error = float(np.linalg.norm(np.asarray(moved_geometry['wrist']['position_m'])-target['position_m']))
            report['wrist_target_error_m'] = error
            check(error < .0015, 'Measured wrist did not approach its 2 mm public target')
            action('final_hold', 'hold', {'duration_s': getattr(args, 'final_hold_duration_s', .04)})
            if 'referee' in task.get('public_rules', {}):
                report['final_public_task'] = invoke('get_task')
            report['public_elapsed_sim_s'] = invoke('observe')['sim_time_s']
            report['measured_joint_max_delta_rad'] = max(abs(v-current_observation['proprioception']['joint_positions_rad'][n])
                for n, v in moved['proprioception']['joint_positions_rad'].items())
            check(report['measured_joint_max_delta_rad'] > 1e-6, 'Native robot showed no measured motion')
            invoke('finish_episode')
            if turn:
                request('GET', endpoint+'/result', headers=headers)
            report['episode_outcome'] = session.result.outcome
            report['physics_valid'] = session.result.physics_valid
            report['worker_exit_code'] = session.worker.process.returncode
            check(session.result.physics_valid, 'Native physical validity audit failed')
            check(session.worker.process.returncode == 0, 'Worker did not exit cleanly')
            report['passed'] = True
    except BaseException as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        if service is not None:
            service.shutdown()
        dump(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ready-timeout-s', type=float, default=10.)
    parser.add_argument('--final-hold-duration-s', type=float, default=.04,
                        help='Final public hold, between .02 and 1 second; useful for release stability checks.')
    parser.add_argument('--wall-timeout-s', type=optional_wall_timeout, default=900.,
                        help='Episode wall-time budget in seconds, or none to disable only this budget; '
                             'readiness, action and outer-launcher limits still apply.')
    parser.add_argument('--action-wall-timeout-s', type=float, default=180.)
    args = parser.parse_args()
    check(math.isfinite(args.ready_timeout_s) and 0 < args.ready_timeout_s <= 100., 'Invalid readiness budget')
    check(math.isfinite(args.final_hold_duration_s) and .02 <= args.final_hold_duration_s <= 1.,
          'Invalid final hold duration')
    check(math.isfinite(args.action_wall_timeout_s) and args.action_wall_timeout_s > 0
          and (args.wall_timeout_s is None or
               (math.isfinite(args.wall_timeout_s) and args.wall_timeout_s >= args.action_wall_timeout_s)),
          'Invalid wall-time budgets')
    if os.environ.get('ISAAC_BENCH_EXPERIENCE') is None:
        os.environ['ISAAC_BENCH_EXPERIENCE'] = str(Path(__file__).resolve().parents[2]/'bench/apps/workroom.headless.kit')
    run(args)


if __name__ == '__main__':
    main()
