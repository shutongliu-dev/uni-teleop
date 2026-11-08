"""Native public HTTP smoke using only published robot state and capabilities.

This checks transport, RGB-D, read-only IK, a small public motion/return and
shared-controller hold. It is not object manipulation or task qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
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


def selected_agent_task(bench, *, family, task_id, configuration_id=None, seed=None, robot_profile=None):
    """Use the real Agent's catalog and public-release selection before creation."""
    from agent.model_client.scene_tasks import published_task, select_task, select_configuration
    from agent.model_client.seeded_tasks import select_episode_case
    # Each physical robot profile has its own catalog pin; None is the catalog default.
    selection = select_task(task_id, family=family, robot_profile=robot_profile)
    selection.require_runnable()
    published, board = published_task(selection, bench.list_tasks())
    check(not board and selection.session_kind == 'single_agent',
          'Agent smoke requires an ordinary single-agent task')
    if configuration_id is not None:
        selection = select_configuration(selection, published, configuration_id)
        published, board = published_task(selection, bench.list_tasks(
            task_id=task_id, configuration_id=configuration_id))
        check(published.get('configuration_id') == configuration_id,
              'Agent selected task projection has another configuration')
    published, _ = select_episode_case(bench, selection, published, seed)
    return selection, published


def agent_readonly_probe(client, base_url, *, family, task_id, created, phase,
                         expected_observation, create_request, controller,
                         query_timeout_s=60., configuration_id=None, robot_profile=None):
    """Exercise the actual Agent transport and context projection without actions.

    The caller owns the HTTP client and the live native episode. This optional
    check reads only the public service, never task assets or evaluator state.
    """
    from agent.model_client.bench_http import BenchHTTPClient
    from agent.model_client.function_http import FunctionHTTPClient
    from agent.model_client.function_runner import _validated_ready, task_requirements_context
    from agent.model_client.runner import canonical_hash
    from bench.protocol.schemas import PolicyDescriptor

    bench = BenchHTTPClient(base_url, client=client, timeout_s=query_timeout_s, get_retries=0)
    selection, published = selected_agent_task(bench, family=family, task_id=task_id,
                                              configuration_id=configuration_id,
                                              seed=create_request.get('seed'),
                                              robot_profile=robot_profile)
    selection.verify_release(created['spec'])
    published_rules = published.get('public_rules', {})
    created_rules = created['task'].get('public_rules', {})
    # Seeded placement adds episode targets after discovery. A declared rules
    # digest instead binds the complete structured referee contract.
    rules_match = (created_rules == published_rules if published.get('public_rules_sha256')
                   else all(created_rules.get(key) == value for key, value in published_rules.items()))
    check(rules_match and created['task'].get('public_rules_sha256') == published.get('public_rules_sha256'),
          'Agent discovery and episode creation disagree on public rules')
    pinned = PolicyDescriptor.model_validate(controller)
    spec, task = _validated_ready(created, create_request, published, False, controller=pinned)
    functions = FunctionHTTPClient(base_url, created['episode_id'], client=client,
                                  timeout_s=query_timeout_s, get_retries=0)
    functions.bind_episode(spec, task=task)
    catalog = functions.catalog()
    live_task_available = any(row['name'] == 'get_task' and row['available']
                              for row in catalog['functions'])
    calls = []

    def invoke(name):
        payload = {'request_id': f'agent-smoke-{phase}-{len(calls):02d}',
                   'function': name, 'arguments': {}}
        result = functions.call(payload)
        check(result['ok'], 'Agent '+name+' returned a rejected public query')
        calls.append({'request': payload, 'response': result})
        return result

    before = invoke('observe')
    live = invoke('get_task') if live_task_available else None
    live_task = live['result'] if live is not None else task
    after = invoke('observe')
    before_observation, after_observation = before['result'], after['result']
    check(before_observation == after_observation, 'Agent read-only queries changed the public observation')
    check(before_observation['sequence'] == expected_observation['sequence']
          and before_observation['sim_time_s'] == expected_observation['sim_time_s'],
          'Agent and HTTP smoke disagree on the current public observation')
    for key in ('limits', 'action_attempts', 'elapsed_sim_s', 'model_accounting'):
        check(before['context']['budget'][key] == after['context']['budget'][key],
              'Agent read-only queries changed action/model budgets')
    for reply in (before, *([live] if live is not None else []), after):
        context = reply['context']
        check(context['observation_sequence'] == before_observation['sequence']
              and context['observation_sim_time_s'] == before_observation['sim_time_s'],
              'Agent query response is bound to a different observation')
    static = task_requirements_context(live_task)
    check(static == task_requirements_context(created['task']),
          'Agent model requirements changed after episode creation')
    properties = published.get('public_rules', {}).get('object_properties')
    check(properties == created['task'].get('public_rules', {}).get('object_properties')
          == live_task.get('public_rules', {}).get('object_properties')
          == static.get('public_rules', {}).get('object_properties'),
          'Agent transport/context dropped or changed declared object properties')
    receipt = {'phase': phase, 'family': family, 'task_id': selection.task_id,
        'configuration_id': selection.configuration_id,
        'task_release_sha256': created['spec']['task_release_sha256'],
        'public_rules_sha256': live_task.get('public_rules_sha256'),
        'task_requirements_source': 'get_task' if live is not None else 'validated_episode_ready',
        'observation_sequence': before_observation['sequence'],
        'sim_time_s': before_observation['sim_time_s'],
        'nominal_object_property_count': len(properties or []),
        'object_properties_sha256': hashlib.sha256(json.dumps(properties, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest() if properties else None,
        'catalog_release_verified': True, 'agent_public_parsing_verified': True,
        'agent_ready_contract_verified': True,
        'controller_sha256': canonical_hash(pinned.model_dump(mode='json')),
        'create_request_sha256': canonical_hash(create_request), 'track': spec['track'],
        'static_model_requirements_verified': True, 'read_only_verified': True,
        'model_calls': 0, 'actions_submitted': 0, 'query_timeout_s': query_timeout_s,
        'client_modules': ['agent.model_client.bench_http.BenchHTTPClient',
                           'agent.model_client.function_http.FunctionHTTPClient']}
    return receipt, calls


def bind_gallery_native_samples(snapshots, states_path):
    """Bind camera timestamps to saved native states after the public actions finish."""
    import numpy as np
    states_path = Path(states_path)
    digest = hashlib.sha256(states_path.read_bytes()).hexdigest()
    with np.load(states_path, allow_pickle=False) as states:
        for snapshot in snapshots:
            matches = np.flatnonzero(np.abs(states['time_s']-snapshot['sim_time_s']) <= 1.e-8)
            check(len(matches) == 1, 'Gallery timestamp lacks a unique recorded native sample')
            index = int(matches[0])
            snapshot['native_sample'] = {'archive_path': str(states_path), 'archive_sha256': digest,
                'sample_index': index, 'sim_time_s': float(states['time_s'][index]),
                'native_physics_step_index': int(states['native_physics_step_index'][index]),
                'native_world_time_s': float(states['native_world_time_s'][index])}
    return snapshots


def run(args):
    import httpx
    import numpy as np
    from PIL import Image

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    report = {'schema': 'native-public-http-smoke.v1', 'passed': False,
        'scope': 'Live public HTTP discovery, RGB-D, read-only IK, small public robot motion/return and shared-policy hold; no grasp, placement or task qualification.',
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'model_calls': 0,
        'private_state_used_for_targets': False, 'native_worker_module': 'bench.runtime.worker',
        'task_release': str(args.release.resolve()), 'task_id': args.task_id,
        'seed': getattr(args, 'seed', 0),
        'gpu_uuid': args.gpu_uuid, 'driver_override': '1', 'phases': [],
        'wall_time_budget_s': args.wall_time_s,
        'action_wall_timeout_s': args.action_wall_timeout_s,
        'server_startup_timeout_s': args.server_startup_timeout_s,
        'http_timeout_s': args.http_timeout_s,
        'agent_query_timeout_s': args.agent_query_timeout_s,
        'hold_duration_s': args.hold_duration_s}
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
        client = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=args.http_timeout_s, trust_env=False)
        deadline = time.monotonic()+args.server_startup_timeout_s
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
        configuration_id = getattr(args, 'configuration', None)
        query = '?task_id='+args.task_id
        selected_hash = report['release_manifest_sha256']
        if configuration_id is not None:
            from agent.model_client.bench_http import BenchHTTPClient
            selection, selected_task = selected_agent_task(
                BenchHTTPClient(f'http://127.0.0.1:{port}', client=client,
                                timeout_s=args.agent_query_timeout_s, get_retries=0),
                family=getattr(args, 'agent_family', None), task_id=args.task_id,
                configuration_id=configuration_id, seed=report['seed'],
                robot_profile=getattr(args, 'robot_profile', None))
            report['configuration_selection'] = selection.metadata()
            report['selected_task'] = selected_task
            selected_hash = selected_task['task_release_sha256']
            query += '&configuration_id='+configuration_id
        report['selected_release_manifest_sha256'] = selected_hash
        report['interfaces'] = request('GET', '/interfaces'+query)
        policies = request('GET', '/policies'+query)
        candidates = [p for p in policies if p['policy_id'] in
                      ('classical_fingertip', 'classical_fingertip_instances', 'classical_fingertip_bimanual')]
        check(len(candidates) == 1, 'Expected one task-bound shared controller')
        policy = candidates[0]
        check(len(policy['supported_robot_ids']) == 1, 'Expected one frozen robot identity')
        robot_id = policy['supported_robot_ids'][0]
        multiple_hands = robot_id in ('fr3_sharpa_instances', 'dual_fr3_sharpa_w01')
        request('GET', '/functions/schema')
        create = {'task_id': args.task_id, 'agent_id': 'public_http_smoke',
            'agent_version': '1', 'agent_config_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seed': report['seed'], 'robot_id': robot_id,
            'track': 'system0' if multiple_hands or getattr(args, 'agent_family', None)
                     else 'free_tool_agent', 'observation_profile': 'vision_proprio',
            'policy_id': policy['policy_id'], 'policy_version': policy['version'],
            'budgets': {'sim_time_s': 4., 'wall_time_s': args.wall_time_s, 'max_actions': 12,
                        'max_model_calls': 1, 'max_action_sim_s': 1.}}
        if configuration_id is not None:
            create['configuration_id'] = configuration_id
        print(json.dumps({'event': 'native_episode_starting', 'task_id': args.task_id}), flush=True)
        dump(output/'progress.json', {'phase': 'native_episode_starting', 'port': port})
        created = request('POST', '/episodes', create)
        episode = created['episode_id']
        report['episode_id'] = episode
        report['created'] = created
        check(created['spec']['task_id'] == args.task_id
              and created['spec']['task_release_sha256'] == selected_hash
              and created['spec'].get('configuration_id') == configuration_id
              and created['spec']['seed'] == report['seed'],
              'Created episode differs from the requested frozen task release')
        request('GET', f'/episodes/{episode}/functions')
        guide = invoke('get_robot')
        observation = invoke('observe')
        report['initial_observation'] = observation
        check(observation['sim_time_s'] == 0., 'New public episode did not start at zero')
        if getattr(args, 'agent_family', None):
            receipt, checked_calls = agent_readonly_probe(client, f'http://127.0.0.1:{port}',
                family=args.agent_family, task_id=args.task_id, created=created,
                phase='created', expected_observation=observation,
                create_request=create, controller=policy, query_timeout_s=args.agent_query_timeout_s,
                configuration_id=configuration_id, robot_profile=getattr(args, 'robot_profile', None))
            report.setdefault('agent_interface_checks', []).append(receipt)
            agent_calls.extend(checked_calls)
            dump(output/'agent_http_calls.json', agent_calls)
        side = next(iter(guide['hands']))
        geometries = observation['proprioception'].get('hand_geometries')
        geometry = geometries[side] if geometries else observation['proprioception']['geometry']
        target = {'wrist': geometry['wrist'], 'fingertip_frame_id': guide['hands'][side]['frame_id']}
        hand_targets = {name: {'wrist': value['wrist'],
            'fingertip_frame_id': guide['hands'][name]['frame_id']} for name, value in (geometries or {}).items()}
        ik = invoke('query_ik', {'observation_sequence': observation['sequence'],
            **({'hand_geometries': hand_targets} if geometries else {'geometry': target})})
        check(ik['solution_found'] and not ik['advances_physics'] and not ik['mutates_controller'],
              'Public current-wrist IK failed or reported mutation')
        report['ik'] = ik
        presets = invoke('get_hand_presets')
        report['hand_presets'] = presets
        for preset_side in presets['sides']:
            invoke('hand_preset', {'side': preset_side, 'preset': 'open'})
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

        def gallery_snapshot(snapshot, phase):
            # Fetch the current episode's actual public PNG. Rendering and state
            # stay inside the existing native worker, never a second scene.
            frames = []
            for camera in snapshot['cameras']:
                if camera['camera_id'] not in ('overview', 'task'):
                    continue
                read = invoke('read_camera', {'observation_sequence': snapshot['sequence'],
                    'camera_id': camera['camera_id']})
                check(read['camera'] == camera and abs(camera['sim_time_s']-snapshot['sim_time_s']) < 1.e-8,
                      'Gallery camera is stale or differs from its public snapshot')
                response = client.get(camera['rgb_uri']); response.raise_for_status()
                payload = response.content; sha = hashlib.sha256(payload).hexdigest()
                check(sha == camera['rgb_sha256'], 'Gallery PNG hash differs from native public media')
                check(Image.open(io.BytesIO(payload)).size == (camera['width'], camera['height']),
                      'Gallery PNG dimensions differ')
                path = output/'gallery'/phase/(camera['camera_id']+'.png')
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
                frames.append({'camera_id': camera['camera_id'], 'png_path': str(path),
                    'png_sha256': sha, 'camera_metadata': camera})
            check({frame['camera_id'] for frame in frames} == {'overview', 'task'},
                  'Gallery requires both declared overview and task cameras')
            result = {'phase': phase, 'episode_id': episode, 'task_id': args.task_id,
                'configuration_id': configuration_id, 'task_release_sha256': selected_hash,
                'seed': report['seed'],
                'observation_sequence': snapshot['sequence'], 'sim_time_s': snapshot['sim_time_s'],
                'frames': frames, 'scope': 'Actual current native public camera PNG; no extra render process or manipulation claim.'}
            after = invoke('observe')
            check(after['sequence'] == snapshot['sequence'] and after['sim_time_s'] == snapshot['sim_time_s'],
                  'Gallery read advanced native state')
            report.setdefault('gallery_snapshots', []).append(result)

        if getattr(args, 'gallery_snapshots', False):
            gallery_snapshot(observation, 'initial')
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

        if multiple_hands or getattr(args, 'agent_family', None):
            final, motion = geometry_smoke(action, observation, guide, hold_duration_s=args.hold_duration_s)
            report['motion'] = motion
        else:
            final, motion = single_joint_smoke(action, observation, guide, args.joint_delta_rad,
                                              hold_duration_s=args.hold_duration_s)
            report['motion'] = motion
        if getattr(args, 'agent_family', None):
            receipt, checked_calls = agent_readonly_probe(client, f'http://127.0.0.1:{port}',
                family=args.agent_family, task_id=args.task_id, created=created,
                phase='before_finish', expected_observation=final,
                create_request=create, controller=policy, query_timeout_s=args.agent_query_timeout_s,
                configuration_id=configuration_id, robot_profile=getattr(args, 'robot_profile', None))
            report.setdefault('agent_interface_checks', []).append(receipt)
            agent_calls.extend(checked_calls)
            dump(output/'agent_http_calls.json', agent_calls)
        if getattr(args, 'gallery_snapshots', False):
            gallery_snapshot(final, 'final')
        result = invoke('finish_episode')
        if getattr(args, 'gallery_snapshots', False):
            bind_gallery_native_samples(report['gallery_snapshots'],
                output/'episodes'/episode/'private/states.npz')
            dump(output/'gallery/manifest.json', {'schema': 'native_public_gallery.v1',
                'snapshots': report['gallery_snapshots'],
                'scope': 'Same-episode native camera/state identity; this smoke does not perform the laundry task.'})
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


def single_joint_smoke(action, observation, guide, joint_delta_rad, *, hold_duration_s=.12):
    initial_targets = observation['proprioception']['commanded_joint_positions_rad']
    check(initial_targets is not None, 'No acknowledged native joint-drive targets in public proprioception')
    joint = guide['arm_joints'][-1]
    limits = guide['joint_limits'][joint]
    delta = joint_delta_rad
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
    final = action('shared-policy-hold', 'hold', {'duration_s': hold_duration_s})
    check(max(abs(final['proprioception']['commanded_joint_positions_rad'][n]-q) for n, q in initial_targets.items()) < 1e-5,
          'Shared-policy hold did not retain acknowledged return targets')
    check(abs(final['proprioception']['joint_positions_rad'][joint]-observation['proprioception']['joint_positions_rad'][joint]) < .005,
          'Measured joint did not return near its public initial position')
    return final, {'joint': joint, 'command_delta_rad': delta, 'measured_delta_rad': actual_delta,
                        'return_error_rad': final['proprioception']['joint_positions_rad'][joint]-observation['proprioception']['joint_positions_rad'][joint],
                        'action_idempotency_checked': True, 'native_acknowledged_drive_targets_checked': True}


def geometry_smoke(action, observation, guide, *, hold_duration_s=.12):
    """Move every declared hand in one public action using measured wrist poses."""
    from copy import deepcopy
    def hand_geometries(value):
        proprio = value['proprioception']
        return proprio.get('hand_geometries') or {next(iter(guide['hands'])): proprio['geometry']}

    geometries = hand_geometries(observation)
    check(set(geometries) == set(guide['hands']) and len(geometries) >= 1,
          'Expected all advertised hand geometries')
    multiple_hands = len(geometries) > 1

    def arguments(targets):
        return {'timeout_sim_s': 1., **({'hand_geometries': targets} if multiple_hands
                                       else {'geometry': next(iter(targets.values()))})}
    original = {name: {'wrist': deepcopy(value['wrist']),
        'fingertip_frame_id': guide['hands'][name]['frame_id'],
        'position_tolerance_m': .0005, 'orientation_tolerance_rad': .05}
        for name, value in geometries.items()}
    targets = deepcopy(original)
    for target in targets.values():
        target['wrist']['position_m'][2] += .002
    motion_name = 'simultaneous-wrist-motion' if multiple_hands else 'single-wrist-motion'
    moved = action(motion_name, 'move_geometry', arguments(targets))
    deltas = {name: hand_geometries(moved)[name]['wrist']['position_m'][2]
              - geometry['wrist']['position_m'][2] for name, geometry in geometries.items()}
    check(all(delta >= .0004 for delta in deltas.values()), 'A commanded hand did not move upward')
    action('return-'+motion_name, 'move_geometry', arguments(original))
    final = action('shared-policy-hold', 'hold', {'duration_s': hold_duration_s})
    errors = {name: math.sqrt(sum((a-b)**2 for a, b in zip(
        hand_geometries(final)[name]['wrist']['position_m'],
        geometry['wrist']['position_m']))) for name, geometry in geometries.items()}
    check(all(error < .002 for error in errors.values()), 'A hand did not return near its initial wrist position')
    return final, {'hands': sorted(geometries), 'command_delta_z_m': .002,
        'measured_delta_z_m': deltas, 'return_position_error_m': errors,
        'same_action_for_all_hands': True, 'action_idempotency_checked': True,
        'scope': 'Public wrist micro-motion for every declared hand; no object manipulation qualification.'}


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
    smoke.add_argument('--seed', type=int, default=0,
                       help='Select the episode seed; the task enforces its published case range')
    smoke.add_argument('--agent-family', help='Validate catalog pin, the full Agent ready contract and public queries on the system0 geometry track')
    smoke.add_argument('--configuration', choices=('single_arm', 'dual_same_side', 'dual_opposed'),
                       help='Select a published immutable configuration before native episode creation')
    smoke.add_argument('--robot-profile', choices=('original', 'l_mount'),
                       help='Verify the Agent catalog pin of this physical robot profile (default: catalog default)')
    smoke.add_argument('--gallery-snapshots', action='store_true',
                       help='Save actual initial/final overview and task PNGs, bound to saved native samples')
    smoke.add_argument('--gpu-uuid', required=True)
    smoke.add_argument('--joint-delta-rad', type=float, default=.01)
    smoke.add_argument('--hold-duration-s', type=float, default=.12,
                       help='Final public shared-controller hold, 20 ms increments up to 1 s')
    smoke.add_argument('--wall-time-s', type=float, default=2400.)
    smoke.add_argument('--action-wall-timeout-s', type=float, default=600.)
    smoke.add_argument('--server-startup-timeout-s', type=float, default=60.,
                       help='Host service initialization and release verification deadline')
    smoke.add_argument('--http-timeout-s', type=float, default=360.,
                       help='Per-request public HTTP timeout, including episode creation')
    smoke.add_argument('--agent-query-timeout-s', type=float, default=60.,
                       help='Actual Agent client read-only request timeout')
    args = parser.parse_args()
    if args.mode == 'run':
        if not 0 <= args.seed < 2**32:
            parser.error('Seed must be an unsigned 32-bit integer')
        if not args.gpu_uuid.startswith('GPU-') or not 0. < args.joint_delta_rad <= .02:
            parser.error('Select a GPU UUID and a joint displacement in (0, 0.02] rad')
        if (not math.isfinite(args.hold_duration_s) or not .02 <= args.hold_duration_s <= 1.
                or abs(args.hold_duration_s/.02-round(args.hold_duration_s/.02)) > 1.e-8):
            parser.error('Hold duration must be a positive multiple of 20 ms, at most 1 s')
        if not 0. < args.action_wall_timeout_s <= args.wall_time_s <= 2400.:
            parser.error('Action timeout must fit the positive wall budget, capped at 2400 s')
        if any(not math.isfinite(value) or not 0. < value <= 2400. for value in
               (args.server_startup_timeout_s, args.http_timeout_s, args.agent_query_timeout_s)):
            parser.error('Startup, HTTP and Agent query timeouts must be positive finite values up to 2400 s')
        return run(args)
    serve(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
