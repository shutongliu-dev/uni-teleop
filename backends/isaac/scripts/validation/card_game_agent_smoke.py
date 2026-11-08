"""Complete public Agent runner diagnostic for one frozen expansion game.

Scripted decisions use only authenticated public context. They exercise native
baseline, intent/cancel, owned wrist micro-motion and return, not game play.
Run the four task IDs separately to retain independent process evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from agent.model_client.card_games import CARD_PROTOCOL, CARD_TASKS
from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.public_http_smoke import check, dump


def validate_action(feedback, function, *, duration=None):
    check(feedback.get('ok') is True and feedback.get('function') == function,
          'Rejected or mismatched native diagnostic action')
    value = feedback['result']
    reason = 'time_hold_completed' if function == 'hold' else 'controller_converged'
    check(value['status'] == 'completed' and value['termination_reason'] == reason
          and not value['episode_finished'] and value['elapsed_sim_s'] > 0.,
          'Native diagnostic action did not complete')
    if duration is not None:
        check(abs(value['elapsed_sim_s'] - duration) < 1.e-8, 'Hold duration was truncated')
    history = value.get('contact_history') or {}
    check(history.get('status') == 'available' and history.get('sampled_substeps', 0) > 0
          and set(history.get('hands', {})) == {'left', 'right'},
          'Native action lacks both hands contact history')
    check(abs(history['sampled_substeps'] * history['physics_dt_s'] - history['duration_s']) < 1.e-8,
          'Native action contact history does not cover all substeps')


def sample_intent(context):
    """Select a syntactically legal initial reservation from public static rules."""
    task_id = context['spec']['task_id']
    public = (context['task']['public_rules'] if 'task' in context
              else context['state']['public_rules'])
    color = context['state']['player']['color']
    pieces = public['metric_design']['pieces']
    if task_id == 'polyomino_territory':
        piece = next(row['id'] for row in pieces if row['owner'] == color and row.get('shape') == 'mono')
        return {'type': 'place_piece', 'piece_id': piece, 'anchor': [0, 0], 'rotation_deg': 0}
    if task_id == 'barrier_race':
        piece = next(row['id'] for row in pieces if row['owner'] == color and row['kind'] == 'barrier')
        return {'type': 'place_barrier', 'piece_id': piece, 'anchor': [0, 0], 'orientation': 'horizontal'}
    if task_id == 'memory_pairs':
        return {'type': 'reveal_card', 'card_id': pieces[0]['id']}
    piece = next(row['id'] for row in pieces if row['owner'] == color and row['kind'] == 'disk')
    return {'type': 'shoot_disk', 'piece_id': piece}


class CardDiagnosticModel:
    """Only model-facing public state supplies IK and motion targets."""
    def __init__(self, task_id, before_first=None):
        self.task_id, self.before_first = task_id, before_first
        self.calls = 0
        self.initial = self.moved = self.final = None
        self.target = self.intent_observation = self.inactive_targets = None
        self.trace = []

    def public_metadata(self):
        return {'model': 'deterministic_public_card_game_diagnostic',
                'protocol': 'scripted_diagnostic', 'external_model_requests': 0}

    def generate(self, prompt, turns, **kwargs):
        context = json.loads(next(row['text'] for row in turns[-1]['content'] if row['type'] == 'text'))
        index = self.calls
        self.calls += 1
        state = context['state']
        game, obs, player = state['board_game'], state['observation'], state['player']
        check(context['spec']['task_id'] == self.task_id and game['protocol'] == CARD_PROTOCOL,
              'Diagnostic received another task/protocol')
        check(game['current_player'] == player['player_id'] == 'player_0', 'Unexpected diagnostic handoff')
        check(not game['terminal'] and game['commit_count'] == 0, 'Micro-motion unexpectedly committed a game operation')
        check(len(obs['cameras']) == 5 and len({row['camera_id'] for row in obs['cameras']}) == 5,
              'Expected five distinct public RGB-D cameras')
        side = player['hand']
        geometry = obs['proprioception']['hand_geometries'][side]
        if index == 0:
            check(game['phase'] == 'awaiting_native_baseline', 'Missing explicit initial native-baseline phase')
            if self.before_first:
                self.before_first()
            own = set(player['joint_names'])
            drive = obs['proprioception'].get('commanded_joint_positions_rad')
            check(bool(drive), 'Missing acknowledged drive targets')
            self.inactive_targets = {name: value for name, value in drive.items() if name not in own}
            decision = {'function': 'hold', 'arguments': {'duration_s': .5, 'mode': 'acknowledged_drive'}}
        else:
            check(game['phase'] != 'awaiting_native_baseline', 'Native baseline did not become available')
            drive = obs['proprioception'].get('commanded_joint_positions_rad') or {}
            check(all(drive.get(name) == value for name, value in self.inactive_targets.items()),
                  'Current player changed the inactive robot drive targets')
            if 'last_feedback' in context:
                feedback = context['last_feedback']
            else:
                # Exact input projection keeps this complete feedback in the
                # immediately preceding public message; never infer a result.
                previous = turns[-2]
                check(previous['role'] == 'user' and isinstance(previous['content'], str),
                      'Missing preceding public tool feedback')
                value = json.loads(previous['content'])
                check(set(value) == {'tool_feedback'}, 'Missing exact adjacent public tool feedback')
                feedback = value['tool_feedback']
            if index == 1:
                validate_action(feedback, 'hold', duration=.5)
                self.initial = deepcopy(geometry)
                self.target = {'wrist': deepcopy(geometry['wrist']),
                    'fingertip_frame_id': side + '_hand_flange',
                    'position_tolerance_m': .0005, 'orientation_tolerance_rad': .05}
                self.intent_observation = deepcopy(obs)
                decision = {'function': 'query_ik', 'arguments': {'hand_geometries': {side: self.target}}}
            elif index == 2:
                value = feedback.get('result') or {}
                check(feedback.get('ok') and feedback.get('function') == 'query_ik'
                      and value.get('solution_found') is True and set(value.get('hands', {})) == {side}
                      and value['hands'][side]['solution_found'] is True
                      and value.get('advances_physics') is False and value.get('mutates_controller') is False,
                      'Read-only owned-hand IK did not produce a solution')
                check(obs == self.intent_observation, 'IK mutated the public observation')
                decision = {'intent': sample_intent(context)}
            elif index == 3:
                check(feedback.get('ok') and game['pending_move'] is not None
                      and obs == self.intent_observation, 'Intent did not reserve without advancing physics')
                decision = {'intent': {'type': 'cancel_intent'}}
            elif index == 4:
                check(feedback.get('ok') and feedback.get('intent') == 'cancel_intent'
                      and game['pending_move'] is None and obs == self.intent_observation,
                      'Cancel mutated physics or retained the reservation')
                moved = deepcopy(self.target)
                moved['wrist']['position_m'][2] += .002
                decision = {'function': 'move_geometry', 'arguments': {
                    'hand_geometries': {side: moved}, 'timeout_sim_s': 1.}}
            elif index == 5:
                validate_action(feedback, 'move_geometry')
                self.moved = deepcopy(geometry)
                check(geometry['wrist']['position_m'][2] - self.initial['wrist']['position_m'][2] >= .0004,
                      'Owned wrist did not move')
                decision = {'function': 'move_geometry', 'arguments': {
                    'hand_geometries': {side: self.target}, 'timeout_sim_s': 1.}}
            elif index == 6:
                validate_action(feedback, 'move_geometry')
                decision = {'function': 'hold', 'arguments': {'duration_s': .12, 'mode': 'acknowledged_drive'}}
            elif index == 7:
                validate_action(feedback, 'hold', duration=.12)
                self.final = deepcopy(geometry)
                check(math.dist(geometry['wrist']['position_m'], self.initial['wrist']['position_m']) < .002,
                      'Owned wrist did not return')
                decision = {'stop': True}
            else:
                raise RuntimeError('Unexpected additional diagnostic decision')
        self.trace.append({'index': index, 'decision': deepcopy(decision),
                           'observation_sequence': obs['sequence'], 'simulation_time_s': obs['sim_time_s']})
        return SimpleNamespace(text=json.dumps(decision),
            public_metadata=lambda: {**self.public_metadata(), 'attempts': 1})


class InactiveModel:
    def public_metadata(self):
        return {'model': 'inactive_public_card_game_diagnostic', 'external_model_requests': 0}

    def generate(self, *args, **kwargs):
        raise RuntimeError('Inactive player must not receive a decision request without physical handoff')


def probe_authentication(client, created):
    """Host-only credentials never enter the model context or returned report."""
    from agent.model_client.function_http import FunctionHTTPClient
    episode = created['episode_id']
    players = {row['binding']['player_id']: row for row in created['players']}
    active = created['board_game']['current_player']
    peer = next(key for key in players if key != active)
    headers = {key: {'Authorization': 'Bearer ' + row['session_token']} for key, row in players.items()}
    paths = {key: f'/board-games/{episode}/players/{key}' for key in players}
    checks = []
    for suffix in ('state', 'events?after=-1&limit=256'):
        for supplied in (None, headers[peer]):
            response = client.get(paths[active] + '/' + suffix, headers=supplied)
            check(response.status_code == 403, 'Missing/cross-player credential unexpectedly authorized')
            checks.append({'path': paths[active] + '/' + suffix, 'status': response.status_code})
    check(client.get(f'/episodes/{episode}/observation').status_code == 403,
          'Ordinary endpoint bypassed player authentication')
    states, pages = {}, {}
    for key, row in players.items():
        functions = FunctionHTTPClient(str(client.base_url), episode, player=row, client=client,
            game_protocol=CARD_PROTOCOL, spec=created['spec'], get_retries=0, timeout_s=600.)
        functions.bind_board_state(created)
        functions.catalog()
        states[key] = functions.state()
        pages[key] = functions.events()
        check(len(states[key]['observation']['cameras']) == 5, 'Expected five authenticated cameras for each player')
        for camera in states[key]['observation']['cameras']:
            for kind in ('rgb', 'depth'):
                uri = camera[kind + '_uri']
                check(client.get(uri).status_code == 403, 'Game camera bypassed player authentication')
                response = client.get(uri, headers=headers[key])
                check(response.status_code == 200 and hashlib.sha256(response.content).hexdigest() == camera[kind + '_sha256'],
                      'Authenticated camera bytes do not match the public hash')
        check(pages[key] == {'events': [], 'next_cursor': -1, 'has_more': False},
              'Fresh diagnostic unexpectedly exposed gameplay events')
    observation = states[peer]['observation']
    response = client.post(paths[peer] + '/call', headers=headers[peer], json={
        'request_id': 'card_smoke_inactive_motion', 'function': 'hold',
        'arguments': {'duration_s': .02, 'mode': 'acknowledged_drive',
                      'action_id': 'card_smoke_inactive_hold', 'observation_sequence': observation['sequence']}})
    value = response.json()
    check(response.status_code == 403 and value.get('ok') is False
          and value.get('error', {}).get('code') == 'permission_denied',
          'Inactive player motion was not explicitly denied')
    after = client.get(paths[active] + '/state', headers=headers[active]).json()
    check(after['observation'] == states[active]['observation'], 'Rejected motion advanced physics')
    return {'passed': True, 'rejected_requests': checks, 'inactive_motion_status': response.status_code,
            'events': pages, 'player_bindings': {key: row['binding'] for key, row in players.items()},
            'rejected_motion_observation_unchanged': True}


def run(args):
    import httpx
    from agent.model_client.bench_http import BenchHTTPClient
    from agent.model_client.function_runner import RunConfig, run as run_agent
    from agent.model_client.scene_tasks import published_task, select_task

    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    check(output.is_relative_to(OUTPUT_ROOT), 'Native evidence belongs under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'card_game_agent_smoke.v1', 'passed': False, 'task_id': args.task_id,
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'gpu_uuid': args.gpu_uuid,
        'scope': 'Complete authenticated function_runner with public-only scripted decisions, native baseline, five RGB-D cameras, IK, intent/cancel, own wrist micro-motion/return/hold and normal finalization.',
        'external_model_requests': 0, 'robot_gameplay_qualified': False, 'physics_accepted': False,
        'physical_reveal_or_operation_committed': False, 'disable_crashreporter': args.disable_crashreporter}
    server = client = None
    model = None
    http_events = []
    try:
        source = verify_if_snapshot(root, required_modules=('scripts.validation.card_game_agent_smoke',
            'agent.model_client.function_runner'), entrypoint_file=Path(__file__))
        check(source is not None, 'Complete frozen source required')
        report['runtime_imports'] = source
        selection = select_task(args.task_id, family='card_room')
        selection.require_runnable()
        release = Path(args.release or selection.source_release['path']).resolve()
        digest = hashlib.sha256((release/'release_manifest.json').read_bytes()).hexdigest()
        check(digest == selection.source_release['manifest_sha256'], 'Release does not match the task catalog pin')
        report['release'] = {'path': str(release), 'manifest_sha256': digest}
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        environment = os.environ.copy()
        environment.update(BENCH_TASK_RELEASE=str(release), BENCH_AGENT_DATA_DIR=str(output/'episodes'),
            BENCH_DEVICE='cuda:0', CUDA_VISIBLE_DEVICES=args.gpu_uuid, ISAAC_BENCH_SINGLE_CUDA_DEVICE='1',
            ISAAC_ALLOW_UNVALIDATED_DRIVER='1', ISAAC_BENCH_EXPERIENCE=str(root/'bench/apps/workroom.headless.kit'),
            PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE='1',
            ISAAC_DISABLE_CRASHREPORTER='1' if args.disable_crashreporter else '0')
        command = [sys.executable, '-B', '-m', 'scripts.validation.public_http_smoke',
                   'serve', '--port', str(port), '--evidence', str(output)]
        report['server_command'] = command
        with (output/'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        report['server_pid'] = server.pid
        def record_request(request):
            http_events.append({'method': request.method, 'path': request.url.path,
                                'query': request.url.query.decode()})
        client = httpx.Client(base_url=url, timeout=600., trust_env=False,
                             event_hooks={'request': [record_request]})
        deadline = time.monotonic() + 120.
        while True:
            check(server.poll() is None, 'Native server exited during startup')
            try:
                if client.get('/health', timeout=1.).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            check(time.monotonic() < deadline, 'Native server health deadline')
            time.sleep(.2)

        class RecordingBench(BenchHTTPClient):
            created = None
            def _json(self, method, path, **kwargs):
                value = super()._json(method, path, **kwargs)
                if method == 'POST' and path == '/board-games':
                    self.created = value  # Host-only; never persisted or included in a model request.
                return value

        bench = RecordingBench(url, client=client, timeout_s=600., get_retries=0)
        task, board = published_task(selection, bench.list_tasks())
        check(board and task['protocol'] == CARD_PROTOCOL, 'Discovery did not publish the new card-game protocol')
        schema_response = client.get('/board-games/schema', params={'protocol': CARD_PROTOCOL})
        schema_response.raise_for_status()
        report['discovery'] = {'task': task, 'schema': schema_response.json()}
        def before_first():
            check(bench.created is not None, 'Runner did not create its own game')
            report['authentication'] = probe_authentication(client, bench.created)
        model = CardDiagnosticModel(args.task_id, before_first)
        result = run_agent(RunConfig(url, args.task_id, output/'agent', family='card_room',
            seed=args.seed, sim_budget=8., wall_budget=args.wall_time_s,
            http_timeout=600., model_timeout=60., max_decisions=10, max_actions=8,
            max_status_queries=100000), bench=bench,
            models={'player_0': model, 'player_1': InactiveModel()}, source_verification=source)
        report['agent_report'] = result
        check(result['error_type'] is None and result['task_setup_error'] is None
              and result['finalization_error'] is None and not result['action_recovery_pending'],
              'Complete Agent runner reported a setup, transport or action failure')
        check(result['ready_validated'] and result['run_status'] == 'completed'
              and result['termination'] == 'model_requested_stop'
              and result['decisions'] == {'player_0': 8, 'player_1': 0}
              and result['physical_attempts'] == {'player_0': 4, 'player_1': 0},
              'Complete expected diagnostic decision/action sequence was not executed')
        game, episode = result['result']['board_game'], result['result']['episode']
        check(game['terminal'] and game['status']['outcome'] == 'aborted'
              and game['status']['end_reason'] == 'cancelled' and game['winner'] is None
              and game['commit_count'] == 0 and game['event_cursor'] == -1
              and episode['physics_valid'] and episode['outcome'] == 'cancelled' and episode['model_calls'] == 0
              and episode['spec']['task_release_sha256'] == digest,
              'Final result fabricated game success, changed release, or lost native validity')
        check(result['public_event_cursors'] == {'player_0': -1, 'player_1': -1},
              'Diagnostic unexpectedly generated gameplay events')
        for player in bench.created['players']:
            response = client.get(f'/board-games/{result["episode_id"]}/players/{player["binding"]["player_id"]}/result',
                headers={'Authorization': 'Bearer ' + player['session_token']})
            check(response.status_code == 200 and response.json() == result['result'],
                  'Players and runner disagree on the final public result')
        report['both_player_final_results_identical'] = True
        report.update(passed=True, scripted_decisions=model.calls,
            measured_delta_z_m=model.moved['wrist']['position_m'][2] - model.initial['wrist']['position_m'][2],
            return_position_error_m=math.dist(model.final['wrist']['position_m'], model.initial['wrist']['position_m']))
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'detail': str(error), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    finally:
        dump(output/'http_requests.json', http_events)
        if model is not None:
            dump(output/'scripted_decisions.json', model.trace)
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
                    report.update(passed=False, forced_server_shutdown=True)
            report['server_exit_code'] = server.returncode
            if server.returncode != 0:
                report['passed'] = False
        shutdown_path = output/'server_shutdown.json'
        if shutdown_path.exists():
            report['server_shutdown'] = shutdown = load_json(shutdown_path)
            if len(shutdown['episodes']) != 1 or any(not row['finished'] or row['worker_exit_code'] != 0
                                                   for row in shutdown['episodes'].values()):
                report['passed'] = False
        else:
            report['passed'] = False
        try:
            report['runtime_imports_final'] = verify_if_snapshot(root,
                required_modules=('scripts.validation.card_game_agent_smoke', 'agent.model_client.function_runner'),
                entrypoint_file=Path(__file__))
            check(report['runtime_imports_final'] is not None, 'Final source snapshot missing')
        except BaseException as error:
            report.update(passed=False, final_verification_error=str(error))
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        dump(output/'qualification.json', report)
    print(json.dumps({'passed': report['passed'], 'output': str(output), 'task_id': args.task_id}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', required=True, choices=sorted(CARD_TASKS))
    parser.add_argument('--release', type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wall-time-s', type=float, default=1800.)
    parser.add_argument('--disable-crashreporter', action='store_true')
    args = parser.parse_args()
    if not args.gpu_uuid.startswith('GPU-'):
        parser.error('Select an explicit GPU UUID')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
