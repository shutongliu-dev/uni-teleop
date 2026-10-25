"""Native public HTTP smoke using only published robot state and capabilities.

This checks transport, RGB-D, read-only IK, a small joint motion/return and
shared-controller hold. It is not object manipulation or task qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def verified():
    root = Path(__file__).resolve().parents[2]
    result = verify_if_snapshot(root,
        required_modules=('scripts.validation.public_http_smoke',), entrypoint_file=Path(__file__))
    check(result is not None, 'Complete frozen source required')
    return result


def serve(args):
    import uvicorn
    from bench.runtime.server import app
    check(app.state.source_snapshot_verification is not None, 'Canonical server is not frozen')
    dump(args.evidence/'server_runtime_imports.json', app.state.source_snapshot_verification)
    # Uvicorn restores and re-emits captured signals after its graceful lifespan
    # shutdown. Consume this harness-owned SIGTERM so the receipt can be saved.
    reemitted_signals = []
    previous = signal.signal(signal.SIGTERM, lambda number, frame: reemitted_signals.append(number))
    try:
        uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)
    finally:
        dump(args.evidence/'server_shutdown.json', {
            'runtime_imports': verified(),
            'graceful_shutdown_reemitted_signals': reemitted_signals,
            'episodes': {key: {'worker_exit_code': session.worker.process.returncode,
                               'finished': session.result is not None}
                         for key, session in app.state.manager.sessions.items()}})
        signal.signal(signal.SIGTERM, previous)


def agent_readonly_probe(client, base_url, *, family, task_id, created, phase, expected_observation):
    """Exercise the actual Agent transport and context projection without actions.

    The caller owns the HTTP client and the live native episode. This optional
    check reads only the public service, never task assets or evaluator state.
    """
    from agent.model_client.bench_http import BenchHTTPClient
    from agent.model_client.function_http import FunctionHTTPClient
    from agent.model_client.function_runner import task_requirements_context
    from agent.model_client.scene_tasks import published_task, select_task

    selection = select_task(task_id, family=family)
    selection.require_runnable()
    selection.verify_release(created['spec'])
    bench = BenchHTTPClient(base_url, client=client, timeout_s=60., get_retries=0)
    published, board = published_task(selection, bench.list_tasks())
    check(not board and selection.session_kind == 'single_agent',
          'Agent read-only smoke requires an ordinary single-agent task')
    check(created['task']['public_rules'] == published['public_rules']
          and created['task']['public_rules_sha256'] == published['public_rules_sha256'],
          'Agent discovery and episode creation disagree on public rules')
    functions = FunctionHTTPClient(base_url, created['episode_id'], client=client,
                                  timeout_s=60., get_retries=0)
    functions.bind_episode(created['spec'], task=created['task'])
    calls = []

    def invoke(name):
        payload = {'request_id': f'agent-smoke-{phase}-{len(calls):02d}',
                   'function': name, 'arguments': {}}
        result = functions.call(payload)
        check(result['ok'], 'Agent '+name+' returned a rejected public query')
        calls.append({'request': payload, 'response': result})
        return result

    before = invoke('observe')
    live = invoke('get_task')
    after = invoke('observe')
    before_observation, after_observation = before['result'], after['result']
    check(before_observation == after_observation, 'Agent read-only queries changed the public observation')
    check(before_observation['sequence'] == expected_observation['sequence']
          and before_observation['sim_time_s'] == expected_observation['sim_time_s'],
          'Agent and HTTP smoke disagree on the current public observation')
    for key in ('limits', 'action_attempts', 'elapsed_sim_s', 'model_accounting'):
        check(before['context']['budget'][key] == after['context']['budget'][key],
              'Agent read-only queries changed action/model budgets')
    for reply in (before, live, after):
        context = reply['context']
        check(context['observation_sequence'] == before_observation['sequence']
              and context['observation_sim_time_s'] == before_observation['sim_time_s'],
              'Agent query response is bound to a different observation')
    static = task_requirements_context(live['result'])
    check(static == task_requirements_context(created['task']),
          'Agent model requirements changed after episode creation')
    properties = published['public_rules'].get('object_properties')
    check(properties == created['task']['public_rules'].get('object_properties')
          == live['result']['public_rules'].get('object_properties')
          == static['public_rules'].get('object_properties'),
          'Agent transport/context dropped or changed declared object properties')
    receipt = {'phase': phase, 'family': family, 'task_id': selection.task_id,
        'task_release_sha256': created['spec']['task_release_sha256'],
        'public_rules_sha256': live['result']['public_rules_sha256'],
        'observation_sequence': before_observation['sequence'],
        'sim_time_s': before_observation['sim_time_s'],
        'nominal_object_property_count': len(properties or []),
        'object_properties_sha256': hashlib.sha256(json.dumps(properties, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest() if properties else None,
        'catalog_release_verified': True, 'agent_public_parsing_verified': True,
        'static_model_requirements_verified': True, 'read_only_verified': True,
        'model_calls': 0, 'actions_submitted': 0,
        'client_modules': ['agent.model_client.bench_http.BenchHTTPClient',
                           'agent.model_client.function_http.FunctionHTTPClient']}
    return receipt, calls


def run(args):
    import httpx
    import numpy as np
    from PIL import Image

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    report = {'schema': 'native-public-http-smoke.v1', 'passed': False,
        'scope': 'Live public HTTP discovery, RGB-D, read-only IK, small robot joint motion/return and shared-policy hold; no grasp, placement or task qualification.',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'model_calls': 0,
        'private_state_used_for_targets': False, 'native_worker_module': 'bench.runtime.worker',
        'task_release': str(args.release.resolve()), 'task_id': args.task_id,
        'gpu_uuid': args.gpu_uuid, 'driver_override': '1', 'phases': [],
        'wall_time_budget_s': args.wall_time_s,
        'action_wall_timeout_s': args.action_wall_timeout_s}
    server = client = None
    episode = None
    calls = []
    agent_calls = []
    try:
        report['runtime_imports'] = verified()
        report['release_manifest_sha256'] = hashlib.sha256((args.release/'release_manifest.json').read_bytes()).hexdigest()
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        report['port'] = port
        environment = os.environ.copy()
        environment.update(BENCH_TASK_RELEASE=str(args.release.resolve()),
            BENCH_AGENT_DATA_DIR=str(output/'episodes'), BENCH_DEVICE='cuda:0',
            CUDA_VISIBLE_DEVICES=args.gpu_uuid, ISAAC_BENCH_SINGLE_CUDA_DEVICE='1',
            ISAAC_ALLOW_UNVALIDATED_DRIVER='1',
            ISAAC_BENCH_EXPERIENCE=str(root/'bench/apps/workroom.headless.kit'),
            PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-m', 'scripts.validation.public_http_smoke',
                   'serve', '--port', str(port), '--evidence', str(output)]
        report['server_command'] = command
        with (output/'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=root, env=environment,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report['server_pid'] = server.pid
        dump(output/'progress.json', {'phase': 'server_starting', 'pid': server.pid, 'port': port})
        # This client targets only the local server; host proxy settings must
        # neither intercept loopback nor add unrelated optional dependencies.
        client = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=360., trust_env=False)
        deadline = time.monotonic()+60
        while True:
            check(server.poll() is None, 'Server exited during startup')
            try:
                response = client.get('/health', timeout=1.)
                if response.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            check(time.monotonic() < deadline, 'Server health deadline expired')
            time.sleep(.2)

        def request(method, path, payload=None):
            start = time.monotonic()
            response = client.request(method, path, json=payload)
            value = response.json()
            calls.append({'method': method, 'path': path, 'request': payload,
                          'status': response.status_code, 'response': value,
                          'wall_s': time.monotonic()-start})
            dump(output/'http_calls.json', calls)
            response.raise_for_status()
            return value

        def invoke(function, arguments=None):
            value = request('POST', f'/episodes/{episode}/call', {
                'request_id': f'http-smoke-{len(calls):04d}', 'function': function,
                'arguments': arguments or {}})
            check(value['ok'], f'{function} did not succeed: {value.get("error")}')
            return value['result']

        report['health'] = request('GET', '/health')
        report['tasks'] = request('GET', '/tasks')
        report['interfaces'] = request('GET', '/interfaces?task_id='+args.task_id)
        policies = request('GET', '/policies?task_id='+args.task_id)
        policy = next(p for p in policies if p['policy_id'] == 'classical_fingertip')
        request('GET', '/functions/schema')
        create = {'task_id': args.task_id, 'agent_id': 'public_http_smoke',
            'agent_version': '1', 'agent_config_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seed': 0, 'track': 'free_tool_agent', 'observation_profile': 'vision_proprio',
            'policy_id': policy['policy_id'], 'policy_version': policy['version'],
            'budgets': {'sim_time_s': 4., 'wall_time_s': args.wall_time_s, 'max_actions': 12,
                        'max_model_calls': 1, 'max_action_sim_s': 1.}}
        print(json.dumps({'event': 'native_episode_starting', 'task_id': args.task_id}), flush=True)
        dump(output/'progress.json', {'phase': 'native_episode_starting', 'port': port})
        created = request('POST', '/episodes', create)
        episode = created['episode_id']
        report['episode_id'] = episode
        report['created'] = created
        check(created['spec']['task_id'] == args.task_id
              and created['spec']['task_release_sha256'] == report['release_manifest_sha256'],
              'Created episode differs from the requested frozen task release')
        request('GET', f'/episodes/{episode}/functions')
        guide = invoke('get_robot')
        observation = invoke('observe')
        report['initial_observation'] = observation
        check(observation['sim_time_s'] == 0., 'New public episode did not start at zero')
        if getattr(args, 'agent_family', None):
            receipt, checked_calls = agent_readonly_probe(client, f'http://127.0.0.1:{port}',
                family=args.agent_family, task_id=args.task_id, created=created,
                phase='created', expected_observation=observation)
            report.setdefault('agent_interface_checks', []).append(receipt)
            agent_calls.extend(checked_calls)
            dump(output/'agent_http_calls.json', agent_calls)
        side = next(iter(guide['hands']))
        geometries = observation['proprioception'].get('hand_geometries')
        geometry = geometries[side] if geometries else observation['proprioception']['geometry']
        target = {'wrist': geometry['wrist'], 'fingertip_frame_id': guide['hands'][side]['frame_id']}
        ik = invoke('query_ik', {'observation_sequence': observation['sequence'],
            **({'hand_geometries': {side: target}} if geometries else {'geometry': target})})
        check(ik['solution_found'] and not ik['advances_physics'] and not ik['mutates_controller'],
              'Public current-wrist IK failed or reported mutation')
        report['ik'] = ik
        invoke('get_hand_presets')
        invoke('hand_preset', {'side': side, 'preset': 'open'})
        cameras = []
        for camera in observation['cameras']:
            read = invoke('read_camera', {'observation_sequence': observation['sequence'],
                                         'camera_id': camera['camera_id']})
            check(read['camera'] == camera, 'Camera metadata disagrees with decision snapshot')
            row = {'camera_id': camera['camera_id']}
            for kind in ('rgb', 'depth'):
                response = client.get(camera[kind+'_uri'])
                response.raise_for_status()
                payload = response.content
                sha = hashlib.sha256(payload).hexdigest()
                check(sha == camera[kind+'_sha256'], 'Public media hash mismatch')
                media = output/'media'
                media.mkdir(exist_ok=True)
                path = media/(camera['camera_id']+('.png' if kind == 'rgb' else '.npy'))
                path.write_bytes(payload)
                if kind == 'rgb':
                    shape = Image.open(io.BytesIO(payload)).size
                    check(shape == (camera['width'], camera['height']), 'RGB dimensions mismatch')
                else:
                    depth = np.load(io.BytesIO(payload), allow_pickle=False)
                    check(depth.shape == (camera['height'], camera['width']), 'Depth dimensions mismatch')
                    check(np.isfinite(depth).all() and np.any(depth > 0.), 'Public depth has no finite visible surface')
                row[kind] = {'path': str(path), 'sha256': sha, 'bytes': len(payload)}
            cameras.append(row)
        report['media'] = cameras
        check(invoke('observe')['sim_time_s'] == 0., 'Read-only queries advanced physics')

        def action(name, function, arguments):
            observed = invoke('observe')
            parameters = {**arguments, 'action_id': name, 'observation_sequence': observed['sequence']}
            submitted = invoke(function, parameters)
            deadline = time.monotonic()+args.action_wall_timeout_s
            while True:
                status = invoke('get_action', {'action_id': name})
                if status['status'] not in ('accepted', 'running'):
                    break
                check(time.monotonic() < deadline, 'Action deadline expired: '+name)
                time.sleep(.2)
            check(status['status'] == 'completed', 'Action failed: '+json.dumps(status))
            after = invoke('observe')
            check(after['sequence'] > observed['sequence'], 'Acknowledged action did not publish a new observation')
            row = {'name': name, 'function': function, 'arguments': parameters,
                   'submission': submitted, 'terminal_status': status, 'observation': after}
            report['phases'].append(row)
            dump(output/'progress.json', {'phase': name, 'status': status['status'],
                                         'sim_time_s': after['sim_time_s']})
            print(json.dumps({'event': 'action_completed', 'name': name, 'time_s': after['sim_time_s']}), flush=True)
            invoke(function, parameters)
            repeat = invoke('observe')
            check(repeat['sim_time_s'] == after['sim_time_s'] and repeat['sequence'] == after['sequence'],
                  'Identical action replay advanced state')
            return after

        initial_targets = observation['proprioception']['commanded_joint_positions_rad']
        check(initial_targets is not None, 'No acknowledged native joint-drive targets in public proprioception')
        joint = guide['arm_joints'][-1]
        limits = guide['joint_limits'][joint]
        delta = args.joint_delta_rad
        if initial_targets[joint]+delta > limits['upper_rad']:
            delta = -delta
        check(limits['lower_rad'] <= initial_targets[joint]+delta <= limits['upper_rad'], 'Tiny motion exceeds public limits')
        moved_targets = {**initial_targets, joint: initial_targets[joint]+delta}
        def joints(targets):
            return {'duration_s': .4,
                    'arm_joint_positions_rad': {n: targets[n] for n in guide['arm_joints']},
                    'hand_joint_positions_rad': {n: targets[n] for n in guide['hand_joints']}}
        moved = action('small-joint-motion', 'move_joints', joints(moved_targets))
        actual_delta = moved['proprioception']['joint_positions_rad'][joint]-observation['proprioception']['joint_positions_rad'][joint]
        check(actual_delta*delta > 0. and abs(actual_delta) >= abs(delta)*.2, 'No measured joint motion in commanded direction')
        check(max(abs(moved['proprioception']['commanded_joint_positions_rad'][n]-q) for n, q in moved_targets.items()) < 1e-6,
              'Public native acknowledged drive did not reach the submitted targets')
        returned = action('return-joint-motion', 'move_joints', joints(initial_targets))
        final = action('shared-policy-hold', 'hold', {'duration_s': .12})
        check(max(abs(final['proprioception']['commanded_joint_positions_rad'][n]-q) for n, q in initial_targets.items()) < 1e-5,
              'Shared-policy hold did not retain acknowledged return targets')
        check(abs(final['proprioception']['joint_positions_rad'][joint]-observation['proprioception']['joint_positions_rad'][joint]) < .005,
              'Measured joint did not return near its public initial position')
        report['motion'] = {'joint': joint, 'command_delta_rad': delta, 'measured_delta_rad': actual_delta,
                            'return_error_rad': final['proprioception']['joint_positions_rad'][joint]-observation['proprioception']['joint_positions_rad'][joint],
                            'action_idempotency_checked': True, 'native_acknowledged_drive_targets_checked': True}
        if getattr(args, 'agent_family', None):
            receipt, checked_calls = agent_readonly_probe(client, f'http://127.0.0.1:{port}',
                family=args.agent_family, task_id=args.task_id, created=created,
                phase='before_finish', expected_observation=final)
            report.setdefault('agent_interface_checks', []).append(receipt)
            agent_calls.extend(checked_calls)
            dump(output/'agent_http_calls.json', agent_calls)
        result = invoke('finish_episode')
        report['result'] = invoke('get_result')
        check(report['result'] == result, 'Finish and result functions disagree')
        check(result['spec'] == created['spec'], 'Terminal result changed the episode contract')
        check(result['outcome'] == 'cancelled' and result['model_calls'] == 0 and result['physics_valid'],
              'Unexpected completion or invalid native physics')
        report['passed'] = True
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'detail': str(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        if client is not None:
            if episode is not None and not report.get('result'):
                try:
                    response = client.post(f'/episodes/{episode}/finish')
                    report['cleanup_finish'] = {'status': response.status_code, 'response': response.json()}
                except BaseException as error:
                    report['cleanup_error'] = str(error)
            client.close()
        if server is not None:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=45.)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10.)
                    report['forced_server_shutdown'] = True
                    report['passed'] = False
            report['server_exit_code'] = server.returncode
            if server.returncode != 0:
                report['passed'] = False
        try:
            report['runtime_imports_final'] = verified()
        except BaseException as error:
            report['passed'] = False
            report['final_verification_error'] = str(error)
        if (output/'server_shutdown.json').exists():
            shutdown = load_json(output/'server_shutdown.json')
            report['server_shutdown'] = shutdown
            if not shutdown['episodes'] or any(row['worker_exit_code'] != 0 for row in shutdown['episodes'].values()):
                report['passed'] = False
        else:
            report['passed'] = False
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        dump(output/'qualification.json', report)
    print(json.dumps({'passed': report['passed'], 'output': str(output), 'scope': report['scope']}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    worker = commands.add_parser('serve')
    worker.add_argument('--port', type=int, required=True)
    worker.add_argument('--evidence', type=Path, required=True)
    smoke = commands.add_parser('run')
    smoke.add_argument('--output', type=Path, required=True)
    smoke.add_argument('--release', type=Path, required=True)
    smoke.add_argument('--task-id', required=True)
    smoke.add_argument('--agent-family', help='Also validate catalog pin, public rules and read-only queries through the actual Agent clients')
    smoke.add_argument('--gpu-uuid', required=True)
    smoke.add_argument('--joint-delta-rad', type=float, default=.01)
    smoke.add_argument('--wall-time-s', type=float, default=2400.)
    smoke.add_argument('--action-wall-timeout-s', type=float, default=600.)
    args = parser.parse_args()
    if args.mode == 'run':
        if not args.gpu_uuid.startswith('GPU-') or not 0. < args.joint_delta_rad <= .02:
            parser.error('Select a GPU UUID and a joint displacement in (0, 0.02] rad')
        if not 0. < args.action_wall_timeout_s <= args.wall_time_s <= 2400.:
            parser.error('Action timeout must fit the positive wall budget, capped at 2400 s')
        return run(args)
    serve(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
