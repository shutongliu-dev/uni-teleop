"""Frozen-source public HTTP smoke for authenticated board and turn games.

Only public robot observations supply targets. This checks integration, not
grasping, physical move commitment, player handoff or full-game qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.public_http_smoke import check, dump, serve


def verified():
    result = verify_if_snapshot(Path(__file__).resolve().parents[2],
        required_modules=('scripts.validation.multiplayer_http_smoke',
                          'scripts.validation.public_http_smoke'),
        entrypoint_file=Path(__file__))
    check(result is not None, 'Complete frozen source required')
    return result


class PlayerProbe:
    """The same HTTP exercise can run against explicit CPU doubles in tests."""
    def __init__(self, client, output, report, kind, created, *, require_native_contacts=False,
                 action_wall_timeout_s=600.):
        self.client, self.output, self.report = client, Path(output), report
        self.kind, self.created = kind, created
        self.game_key = 'board_game' if kind == 'board' else 'turn_game'
        self.base = '/' + ('board-games' if kind == 'board' else 'turn-games')
        self.episode = created['episode_id']
        self.players = {row['binding']['player_id']: row for row in created['players']}
        self.calls = []
        self.require_native_contacts = require_native_contacts
        self.action_wall_timeout_s = action_wall_timeout_s
        self.report['phases'] = []

    def path(self, player, suffix):
        return f'{self.base}/{self.episode}/players/{player}/{suffix}'

    def headers(self, player):
        return {'Authorization': 'Bearer '+self.players[player]['session_token']}

    def request(self, method, path, payload=None, *, player=None, expected=None):
        start = time.monotonic()
        response = self.client.request(method, path, json=payload,
            headers=self.headers(player) if player else {})
        value = response.json()
        # Tokens are credentials, not diagnostic evidence.
        self.calls.append({'method': method, 'path': path, 'player': player,
            'request': payload, 'status': response.status_code, 'response': value,
            'wall_s': time.monotonic()-start})
        dump(self.output/'http_calls.json', self.calls)
        if expected is not None:
            check(response.status_code == expected,
                  f'Expected HTTP {expected}, got {response.status_code}: {path}')
        else:
            response.raise_for_status()
        return value

    def invoke(self, player, function, arguments=None, *, expected=None):
        value = self.request('POST', self.path(player, 'call'), {
            'request_id': f'multiplayer-smoke-{len(self.calls):05d}',
            'function': function, 'arguments': arguments or {}},
            player=player, expected=expected)
        if expected is None:
            check(value['ok'], f'{function} failed: {value.get("error")}')
            return value['result']
        check(not value['ok'], 'Rejected player call unexpectedly succeeded')
        return value

    def state(self, player):
        return self.request('GET', self.path(player, 'state'), player=player)

    def action(self, player, name, function, arguments):
        before = self.invoke(player, 'observe')
        parameters = {**arguments, 'action_id': name, 'observation_sequence': before['sequence']}
        submitted = self.invoke(player, function, parameters)
        deadline = time.monotonic()+self.action_wall_timeout_s
        while True:
            status = self.invoke(player, 'get_action', {'action_id': name})
            if status['status'] not in ('accepted', 'running'):
                break
            check(time.monotonic() < deadline, 'Action deadline expired: '+name)
            time.sleep(.05)
        check(status['status'] == 'completed', 'Action failed: '+json.dumps(status))
        if self.require_native_contacts:
            contact = status.get('contact_history') or {}
            check(contact.get('status') == 'available' and contact.get('sampled_substeps', 0) > 0
                  and set(contact.get('hands') or {}) == {
                      row['binding']['hand'] for row in self.players.values()},
                  'Completed action lacks complete named-player native contact history')
            check(abs(contact['sampled_substeps']*contact['physics_dt_s']-contact['duration_s']) < 1e-8,
                  'Native contact history has incomplete action-time coverage')
        after = self.invoke(player, 'observe')
        check(after['sim_time_s'] > before['sim_time_s'] and after['sequence'] > before['sequence'],
              'Action produced no new acknowledged observation')
        own = set(self.players[player]['binding']['joint_names'])
        acknowledged = before['proprioception']['commanded_joint_positions_rad']
        check(acknowledged is not None, 'Missing acknowledged drive targets')
        check(all(after['proprioception']['commanded_joint_positions_rad'][joint] == value
                  for joint, value in acknowledged.items() if joint not in own),
              'Current player changed an inactive robot drive target')
        self.invoke(player, function, parameters)
        replay = self.invoke(player, 'observe')
        check((replay['sequence'], replay['sim_time_s']) == (after['sequence'], after['sim_time_s']),
              'Identical action replay advanced physics')
        self.report['phases'].append({'name': name, 'player': player, 'function': function,
            'arguments': parameters, 'submission': submitted, 'terminal_status': status,
            'observation': after, 'inactive_drive_unchanged': True, 'idempotent_replay': True})
        dump(self.output/'progress.json', {'phase': name, 'sim_time_s': after['sim_time_s']})
        print(json.dumps({'event': 'action_completed', 'name': name,
                          'player': player, 'time_s': after['sim_time_s']}), flush=True)
        return after

    def media(self, player, observation):
        import numpy as np
        from PIL import Image
        rows = []
        for camera in observation['cameras']:
            read = self.invoke(player, 'read_camera', {
                'observation_sequence': observation['sequence'], 'camera_id': camera['camera_id']})
            check(read['camera'] == camera, 'Camera metadata differs from public observation')
            row = {'camera_id': camera['camera_id']}
            for kind in ('rgb', 'depth'):
                uri = camera[kind+'_uri']
                check(self.client.get(uri).status_code == 403, 'Game media lacks authentication')
                response = self.client.get(uri, headers=self.headers(player))
                response.raise_for_status()
                payload = response.content
                check(hashlib.sha256(payload).hexdigest() == camera[kind+'_sha256'], 'Media hash mismatch')
                directory = self.output/'media'/player
                directory.mkdir(parents=True, exist_ok=True)
                path = directory/(camera['camera_id']+('.png' if kind == 'rgb' else '.npy'))
                path.write_bytes(payload)
                if kind == 'rgb':
                    check(Image.open(io.BytesIO(payload)).size == (camera['width'], camera['height']),
                          'RGB dimensions mismatch')
                else:
                    depth = np.load(io.BytesIO(payload), allow_pickle=False)
                    check(depth.shape == (camera['height'], camera['width'])
                          and np.isfinite(depth).all() and np.any(depth > 0.), 'Invalid public depth')
                row[kind] = {'path': str(path), 'sha256': camera[kind+'_sha256'], 'bytes': len(payload)}
            rows.append(row)
        check(rows, 'Game exposed no public RGB-D cameras')
        return rows

    def exercise(self, *, expected_player=None, displacement_m=.002, reset_wait_s=3.):
        import numpy as np
        state = self.created[self.game_key]
        active = state['current_player' if self.kind == 'board' else 'current_player_id']
        check(active in self.players, 'No active player at creation')
        check(expected_player is None or active == expected_player, 'Unexpected starting player')
        peers = [player for player in self.players if player != active]
        check(len(peers) == 1, 'This smoke expects two authenticated players')
        peer = peers[0]
        self.report.update(starting_player=active, player_bindings={
            player: row['binding'] for player, row in self.players.items()})
        self.request('GET', self.path(active, 'state'), expected=403)
        self.request('GET', self.path(peer, 'state'), player=active, expected=403)
        self.request('GET', f'/episodes/{self.episode}/observation', expected=403)
        wrong = '/turn-games' if self.kind == 'board' else '/board-games'
        self.request('GET', self.path(active, 'state').replace(self.base, wrong, 1),
                     player=active, expected=404)
        observations, guides = {}, {}
        for player in self.players:
            self.request('GET', self.path(player, 'functions'), player=player)
            public = self.state(player)
            check(public[self.game_key] == state, 'Player referee receipts disagree')
            observation = self.invoke(player, 'observe')
            observations[player] = observation
            check(observation['sim_time_s'] == 0., 'Initial public time is not zero')
            guide = guides[player] = self.invoke(player, 'get_robot')
            side = self.players[player]['binding']['hand']
            wrist = observation['proprioception']['hand_geometries'][side]['wrist']
            ik = self.invoke(player, 'query_ik', {'observation_sequence': observation['sequence'],
                'hand_geometries': {side: {'wrist': wrist,
                    'fingertip_frame_id': guide['hands'][side]['frame_id']}}})
            check(ik['solution_found'] and not ik['advances_physics'] and not ik['mutates_controller'],
                  'Read-only owned-wrist IK failed')
            self.report.setdefault('read_checks', {})[player] = {
                'ik': ik, 'media': self.media(player, observation)}
        before = self.invoke(active, 'observe')
        self.invoke(peer, 'hold', {'action_id': 'inactive-denied',
            'observation_sequence': before['sequence'], 'duration_s': .02}, expected=403)
        self.invoke(active, 'move_wrist', {'action_id': 'foreign-hand-denied',
            'observation_sequence': before['sequence'], 'hand': self.players[peer]['binding']['hand'],
            'wrist': observations[peer]['proprioception']['hand_geometries'][
                self.players[peer]['binding']['hand']]['wrist'], 'timeout_sim_s': 1.}, expected=403)
        after = self.invoke(active, 'observe')
        check((after['sequence'], after['sim_time_s']) == (before['sequence'], before['sim_time_s']),
              'Read-only queries or denied commands advanced physics')
        index = 0
        while self.kind == 'turn' and state['public_progress'].get('status') == 'reset_pending':
            check(after['sim_time_s']+.2 <= reset_wait_s+1e-9, 'Reset readiness deadline expired')
            after = self.action(active, f'reset-hold-{index:02d}', 'hold',
                                {'duration_s': .2, 'mode': 'acknowledged_drive'})
            state = self.state(active)[self.game_key]
            check(state['status'] == 'active' and state['current_player_id'] == active,
                  'Reset terminated or changed the starting player')
            index += 1
        self.report['ready_state'] = state
        side = self.players[active]['binding']['hand']
        baseline = self.invoke(active, 'observe')
        original = deepcopy(baseline['proprioception']['hand_geometries'][side]['wrist'])
        target = deepcopy(original)
        target['position_m'][2] += displacement_m
        moved = self.action(active, 'owned-wrist-motion', 'move_wrist',
                            {'hand': side, 'wrist': target, 'timeout_sim_s': 2.,
                             'position_tolerance_m': .0005, 'orientation_tolerance_rad': .02})
        measured = moved['proprioception']['hand_geometries'][side]['wrist']['position_m']
        check(measured[2]-original['position_m'][2] > displacement_m*.2,
              'No measured wrist motion in commanded direction')
        returned = self.action(active, 'owned-wrist-return', 'move_wrist',
                               {'hand': side, 'wrist': original, 'timeout_sim_s': 2.,
                                'position_tolerance_m': .0005, 'orientation_tolerance_rad': .02})
        final = self.action(active, 'shared-policy-hold', 'hold',
                            {'duration_s': .12, 'mode': 'acknowledged_drive'})
        check(final['proprioception']['commanded_joint_positions_rad'] ==
              returned['proprioception']['commanded_joint_positions_rad'],
              'Acknowledged-drive hold changed the confirmed drive')
        position = final['proprioception']['hand_geometries'][side]['wrist']['position_m']
        error = float(np.linalg.norm(np.asarray(position)-original['position_m']))
        check(error < .001, 'Wrist did not return within 1 mm of its initial public pose')
        self.report['motion'] = {'hand': side, 'requested_displacement_m': displacement_m,
                                'measured_displacement_m': measured[2]-original['position_m'][2],
                                'return_position_error_m': error}
        episode = self.invoke(active, 'finish_episode')
        check(episode['spec'] == self.created['spec'], 'Terminal result changed the episode contract')
        check(episode['outcome'] == 'cancelled' and episode['physics_valid'] and episode['model_calls'] == 0,
              'Unexpected execution outcome or invalid physics')
        results = {player: self.request('GET', self.path(player, 'result'), player=player)
                   for player in self.players}
        result = results[active]
        # Preserve the actual terminal receipt even when a later smoke check
        # fails. Cleanup must not try to finish an already sealed game again.
        self.report['result'] = result
        check(results[active] == results[peer], 'Players see different final results')
        check(result['episode'] == episode, 'Player and episode results disagree')
        referee = result[self.game_key]
        if self.kind == 'board':
            check(referee['terminal'] and referee['status']['outcome'] == 'aborted'
                  and referee['status']['end_reason'] == 'cancelled'
                  and referee['winner'] is None and not referee['draw'] and referee['turn_index'] == 0,
                  'Smoke manufactured a board move or result')
        else:
            # Turn terminal reasons belong to the frozen scene referee. The
            # shared cancellation outcome is carried by EpisodeResult above.
            check(referee['status'] == 'interrupted'
                  and referee['winner_player_id'] is None and referee['loser_player_id'] is None
                  and referee['turn_index'] == 0, 'Smoke manufactured a turn or winner')


def run(args):
    import httpx
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    report = {'schema': 'native-multiplayer-http-smoke.v1', 'passed': False,
        'scope': __doc__, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'task_release': str(args.release.resolve()), 'task_id': args.task_id, 'kind': args.kind,
        'seed': args.seed, 'gpu_uuid': args.gpu_uuid, 'model_calls': 0,
        'wall_time_budget_s': args.wall_time_s, 'action_wall_timeout_s': args.action_wall_timeout_s,
        'private_state_used_for_targets': False, 'native_worker_module': 'bench.runtime.worker'}
    server = client = probe = None
    try:
        report['runtime_imports'] = verified()
        report['release_manifest_sha256'] = hashlib.sha256((args.release/'release_manifest.json').read_bytes()).hexdigest()
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        environment = os.environ.copy()
        environment.update(BENCH_TASK_RELEASE=str(args.release.resolve()),
            BENCH_AGENT_DATA_DIR=str(output/'episodes'), BENCH_DEVICE='cuda:0',
            CUDA_VISIBLE_DEVICES=args.gpu_uuid, ISAAC_BENCH_SINGLE_CUDA_DEVICE='1',
            ISAAC_ALLOW_UNVALIDATED_DRIVER='1',
            ISAAC_BENCH_EXPERIENCE=str(root/'bench/apps/workroom.headless.kit'),
            PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-m', 'scripts.validation.multiplayer_http_smoke',
                   'serve', '--port', str(port), '--evidence', str(output)]
        report.update(server_command=command, port=port)
        with (output/'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=root, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report['server_pid'] = server.pid
        client = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=360., trust_env=False)
        deadline = time.monotonic()+60.
        while True:
            check(server.poll() is None, 'Server exited during startup')
            try:
                if client.get('/health', timeout=1.).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            check(time.monotonic() < deadline, 'Server startup deadline expired')
            time.sleep(.2)
        base = '/board-games' if args.kind == 'board' else '/turn-games'
        policy_id = 'classical_fingertip_bimanual' if args.kind == 'board' else 'classical_fingertip_instances'
        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()
        report['tasks'] = get('/tasks')
        task = next(row for row in report['tasks'] if row['task_id'] == args.task_id)
        check(task['create_endpoint'] == 'POST '+base, 'Discovery advertised the wrong task route')
        report['interfaces'] = get('/interfaces?task_id='+args.task_id)
        report['game_schema'] = get(base+'/schema')
        policy = next(row for row in get('/policies?task_id='+args.task_id) if row['policy_id'] == policy_id)
        create = {'task_id': args.task_id, 'agent_id': 'multiplayer_http_smoke', 'agent_version': '1',
            'agent_config_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'seed': args.seed, 'track': 'system0', 'observation_profile': 'vision_proprio',
            'policy_id': policy_id, 'policy_version': policy['version'],
            'budgets': {'sim_time_s': 8., 'wall_time_s': args.wall_time_s, 'max_actions': 32,
                        'max_model_calls': 1, 'max_action_sim_s': 2.}}
        dump(output/'progress.json', {'phase': 'native_episode_starting', 'port': port})
        response = client.post(base, json=create)
        response.raise_for_status()
        created = response.json()
        probe = PlayerProbe(client, output, report, args.kind, created, require_native_contacts=True,
                            action_wall_timeout_s=args.action_wall_timeout_s)
        report['episode_id'] = created['episode_id']
        report['created'] = {**created, 'players': [{'binding': row['binding']} for row in created['players']]}
        check(created['spec']['task_id'] == args.task_id
              and created['spec']['task_release_sha256'] == report['release_manifest_sha256'],
              'Created game differs from the requested frozen task release')
        probe.exercise(expected_player=args.expected_player, displacement_m=args.wrist_delta_m,
                       reset_wait_s=args.reset_wait_s)
        report['passed'] = True
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'detail': str(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        if probe is not None and 'result' not in report:
            try:
                player = report.get('starting_player') or next(iter(probe.players))
                report['cleanup_finish'] = probe.invoke(player, 'finish_episode')
            except BaseException as error:
                report['cleanup_error'] = str(error)
        if client is not None:
            client.close()
        if server is not None:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=45.)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10.)
                    report.update(forced_server_shutdown=True, passed=False)
            report['server_exit_code'] = server.returncode
            if server.returncode != 0:
                report['passed'] = False
        try:
            report['runtime_imports_final'] = verified()
        except BaseException as error:
            report.update(passed=False, final_verification_error=str(error))
        if (output/'server_shutdown.json').exists():
            report['server_shutdown'] = shutdown = load_json(output/'server_shutdown.json')
            if not shutdown['episodes'] or any(row['worker_exit_code'] != 0 or not row['finished']
                                              for row in shutdown['episodes'].values()):
                report['passed'] = False
        else:
            report['passed'] = False
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        dump(output/'qualification.json', report)
    print(json.dumps({'passed': report['passed'], 'output': str(output)}), flush=True)
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
    smoke.add_argument('--kind', choices=('board', 'turn'), required=True)
    smoke.add_argument('--gpu-uuid', required=True)
    smoke.add_argument('--seed', type=int, default=0)
    smoke.add_argument('--expected-player')
    smoke.add_argument('--wrist-delta-m', type=float, default=.002)
    smoke.add_argument('--reset-wait-s', type=float, default=3.)
    smoke.add_argument('--wall-time-s', type=float, default=2400.,
                       help='Episode wall budget, bounded by the public server allocation.')
    smoke.add_argument('--action-wall-timeout-s', type=float, default=600.,
                       help='Client action wait; native substep observers can exceed real time.')
    args = parser.parse_args()
    if args.mode == 'run':
        if (not args.gpu_uuid.startswith('GPU-') or not 0 <= args.seed < 2**32
                or not 0. < args.wrist_delta_m <= .003 or not .2 <= args.reset_wait_s <= 3.
                or not math.isfinite(args.wall_time_s) or not 0 < args.wall_time_s <= 2400.
                or not math.isfinite(args.action_wall_timeout_s)
                or not 0 < args.action_wall_timeout_s <= args.wall_time_s):
            parser.error('Use a GPU UUID, uint32 seed, displacement in (0, .003] m, reset wait in [.2, 3] s, '
                         'and finite action/episode wall limits with 0 < action <= episode <= 2400 s')
        return run(args)
    serve(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
