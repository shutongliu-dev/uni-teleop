"""Real new-session HTTP lifecycle with explicitly synthetic CPU frame receipts.

These tests build immutable releases from the actual scene-v2 assets and use
the real ExpansionMatch, CardGameWorker admission, CardGameSession and HTTP
service. Only the native worker transport/physics is replaced. The constructed
frames are consumer fixtures, never native dynamics or robot qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs
from bench.paths import OUTPUT_ROOT, load_json
from bench.protocol.card_games import CardGameEventPage, CardGameResult, CardGameState
from bench.runtime.card_game_worker import CardGameWorker
from bench.runtime.card_games import CardGameSession
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.board_games.expansion_release import prepare_release
from scenes.workrooms.card_room.expansion_referee import (
    ExpansionMatch, ExpansionNativeFrame, ExpansionObjectEvidence, PLAYERS,
)
from scenes.workrooms.card_room.game_designs import TASK_IDS
from test_agent_service import FakeWorker, ManualTime, PRIVATE_MARKER, join_action


class SyntheticCardWorker(FakeWorker):
    """Private test producer; no endpoint can supply these synthetic frames."""

    def __init__(self, *, release, task_id, hand_id, **kwargs):
        assert hand_id == 'dual_fr3_sharpa_w01'
        super().__init__(hand_id='sharpa_w01_right', acknowledged_targets=True, depth=True, **kwargs)
        _, scene, task, initial = task_inputs(release, task_id)
        model = public_robot_geometry(robot_configuration(scene))
        self.arm, self.active = list(model.arm_names), list(model.names)
        self.hand = [name for name in self.active if name not in self.arm]
        self.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        self.raw['joint_velocities_rad_s'] = dict.fromkeys(self.active, 0.)
        self.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        self.initial['metadata'].update(scene_spec=scene, action_spec={
            'joint_names': self.active, 'arm_joint_names': self.arm, 'hand_joint_names': self.hand,
            'joint_limits_rad': {name: [float(side.lower[i]), float(side.upper[i])]
                                 for side in model.sides.values() for i, name in enumerate(side.names)}})
        judge = CardGameWorker.__new__(CardGameWorker)
        judge.match = ExpansionMatch(task_id, asset_metadata=task['asset_metadata'])
        judge.task = task
        judge._owners = {f'player_{i}': frozenset(model.sides[side].names)
                         for i, side in enumerate(('left', 'right'))}
        judge._drive_targets = deepcopy(initial['joint_positions_rad'])
        judge.env = SimpleNamespace(_validate_action=self._validate_command)
        judge.observer = SimpleNamespace(samples=0, unsupported_samples=0)
        judge.output, judge.closed, judge._sent_event_cursor = None, False, -1
        self.judge = judge
        self.dt = .0005
        self.objects = {}
        meta = task['asset_metadata']
        for name in meta['actors']:
            position = list(meta['initial_positions_m'][name])
            position[2] -= meta['dimensions']['reset_clearance_m']
            self.objects[name] = ExpansionObjectEvidence(tuple(position), (1., 0., 0., 0.),
                (0., 0., 0.), (0., 0., 0.), meta['masses_kg'][name]*9.81*self.dt,
                {player: 0. for player in PLAYERS}, 0.,
                robot_nonhand_contact_impulses_ns={player: 0. for player in PLAYERS})
        self.mode = 'hold'
        self._prepared_attempt = None
        self.broken = False
        self.step_replies = []
        self.initial.update(observation=deepcopy(self.raw), score=judge.score(), board_game=judge.public_state())

    def _validate_command(self, command):
        assert set(command) == {'joint_positions_rad', 'duration_s'}
        assert set(command['joint_positions_rad']) == set(self.active)
        assert command['duration_s'] == .02

    def _frames(self, duration):
        match = self.judge.match
        for _ in range(round(duration/self.dt)):
            self.judge.observer.samples += 1
            index = self.judge.observer.samples
            contacts_complete = self.mode != 'missing_evidence'
            if not match.terminal:
                for name, obj in self.objects.items():
                    self.objects[name] = replace(obj, robot_contact_impulses_ns={p: 0. for p in PLAYERS})
                if self.mode == 'physical_failure':
                    name = next(iter(self.objects))
                    self.objects[name] = replace(self.objects[name], maximum_penetration_m=.004)
                elif self.mode == 'unexpected_reveal':
                    name = next(name for name in self.objects if name not in match.rules._exposed)
                    self.objects[name] = replace(self.objects[name], quaternion_wxyz=(0., 1., 0., 0.),
                        robot_contact_impulses_ns={p: .001 if p == match.current_player else 0. for p in PLAYERS})
                elif self.mode == 'execute_pending' and match.pending:
                    intent = match.pending
                    first = self._prepared_attempt != intent.attempt_index
                    self._prepared_attempt = intent.attempt_index
                    if first:
                        assert intent.type in ('reveal_card', 'move_pawn')
                        name = intent.piece_ids[0]
                        obj = self.objects[name]
                        if intent.type == 'reveal_card':
                            obj = replace(obj, quaternion_wxyz=(0., 1., 0., 0.))
                        else:
                            q, r = intent.parameters['destination']
                            obj = replace(obj, position_m=(.65+(q-2)*.075, (r-2)*.075, .8485))
                        self.objects[name] = replace(obj, robot_contact_impulses_ns={
                            p: .001 if p == intent.player_id else 0. for p in PLAYERS})
            match.observe(ExpansionNativeFrame(index, index*self.dt, self.dt,
                                              deepcopy(self.objects), contacts_complete))

    def request(self, operation, **payload):
        if self.closed:
            raise RuntimeError('Synthetic worker is already closed')
        if operation == 'board_game':
            self.calls.append((operation, deepcopy(payload)))
            return self.judge.request(payload['command'], actor=payload.get('actor'))
        if operation == 'step':
            admitted = self.judge.authorize_step(payload['command'], payload.get('actor'))
            reply = super().request(operation, **payload)
            self.raw['robot']['joint_targets_rad'] = deepcopy(payload['command']['joint_positions_rad'])
            self._frames(payload['command']['duration_s'])
            reply.update(observation=deepcopy(self.raw), score=self.judge.score(),
                         **self.judge.acknowledge_step(payload['command'], admitted))
            self.step_replies.append(deepcopy(reply))
            return reply
        return super().request(operation, **payload)


@pytest.fixture(scope='module')
def releases():
    # Temporary synthetic releases under the output root, removed afterwards.
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='card_games_http_cpu_', dir=OUTPUT_ROOT))
    (root/'scope.json').write_text(json.dumps({
        'scope': 'CPU HTTP consumer fixtures only; no native simulation, robot or task success qualification.',
        'source_test': str(Path(__file__).resolve()), 'task_ids': list(TASK_IDS)}, indent=2)+'\n')
    paths = {}
    for task in TASK_IDS:
        path = root/task
        prepare_release(path, task)
        paths[task] = path
    yield paths
    shutil.rmtree(root)


@pytest.fixture
def game(tmp_path, releases):
    @contextmanager
    def start(task='memory_pairs'):
        workers = []
        def factory(**kwargs):
            worker = SyntheticCardWorker(**kwargs)
            workers.append(worker)
            return worker
        api = AgentService(tmp_path/task, release=releases[task], registry=default_registry(releases[task]),
                           worker_factory=factory)
        with TestClient(create_app(service=api)) as client:
            response = client.post('/board-games', json={
                'task_id': task, 'agent_id': 'synthetic_cpu', 'agent_version': 'v1',
                'agent_config_sha256': 'b'*64, 'robot_id': 'dual_fr3_sharpa_w01',
                'track': 'direct_joints', 'policy_id': None, 'policy_version': None})
            assert response.status_code == 201, response.text
            created = response.json()
            session = api.sessions[created['episode_id']]
            assert isinstance(session.board_game, CardGameSession)
            tokens = {row['binding']['player_id']: row['session_token'] for row in created['players']}
            yield SimpleNamespace(client=client, api=api, session=session, worker=workers[-1], tokens=tokens,
                                  created=created)
    return start


def endpoint(g, player='player_0'):
    return f'/board-games/{g.session.spec.episode_id}/players/{player}'


def headers(g, player='player_0'):
    return {'Authorization': 'Bearer '+g.tokens[player]}


def call(g, function, arguments=None, *, player='player_0'):
    response = g.client.post(endpoint(g, player)+'/call', headers=headers(g, player), json={
        'function': function, 'arguments': arguments or {}, 'request_id': 'synthetic/'+function})
    assert PRIVATE_MARKER not in response.text
    return response


def state(g, player='player_0'):
    response = g.client.get(endpoint(g, player)+'/state', headers=headers(g, player))
    assert response.status_code == 200, response.text
    value = response.json()['board_game']
    CardGameState.model_validate(value)
    assert PRIVATE_MARKER not in response.text and 'private_reset_pairs' not in response.text
    return value


def intent(g, command, *, player='player_0', expected=200):
    response = g.client.post(endpoint(g, player)+'/intent', headers=headers(g, player), json=command)
    assert response.status_code == expected, response.text
    return response


def hold(g, identity, duration=.02, *, player='player_0'):
    args = {'action_id': identity, 'observation_sequence': g.session.sequence, 'duration_s': duration}
    response = call(g, 'hold', args, player=player)
    assert response.status_code in (200, 202) and response.json()['ok'], response.text
    join_action(g.session)
    result = call(g, 'get_action', {'action_id': identity}, player=player)
    assert result.status_code == 200 and result.json()['ok'], result.text
    return result.json()['result'], args


def event_page(g, player='player_0', after=-1, limit=256):
    response = g.client.get(endpoint(g, player)+'/events', headers=headers(g, player),
                            params={'after': after, 'limit': limit})
    assert response.status_code == 200, response.text
    CardGameEventPage.model_validate(response.json())
    return response.json()


def result(g, player='player_0'):
    response = g.client.get(endpoint(g, player)+'/result', headers=headers(g, player))
    assert response.status_code == 200, response.text
    CardGameResult.model_validate(response.json())
    return response.json()


def steps(g):
    return [payload for operation, payload in g.worker.calls if operation == 'step']


@pytest.mark.parametrize('task', TASK_IDS)
def test_all_new_tasks_use_real_new_session_and_native_baseline_gate(game, task):
    with game(task) as g:
        assert state(g)['phase'] == 'awaiting_native_baseline'
        assert state(g)['protocol'] == 'bench_card_game_state_v1'
        command = {'polyomino_territory': {'type': 'place_piece', 'piece_id': 'black_mono', 'anchor': [0, 0]},
                   'barrier_race': {'type': 'move_pawn', 'destination': [2, 1]},
                   'memory_pairs': {'type': 'reveal_card', 'card_id': 'card_00'},
                   'disk_curling': {'type': 'shoot_disk', 'piece_id': 'black_disk_0'}}[task]
        intent(g, command, expected=422)
        assert not steps(g) and state(g)['turn_index'] == 0
        hold(g, 'baseline')
        intent(g, command)
        assert state(g)['pending_move']['type'] == command['type']
        assert event_page(g) == {'events': [], 'next_cursor': -1, 'has_more': False}


def test_same_player_reveal_epoch_stops_old_action_and_stale_native_actor_is_rejected(game):
    with game() as g:
        hold(g, 'baseline')
        intent(g, {'type': 'reveal_card', 'card_id': 'card_00'})
        g.worker.mode = 'execute_pending'
        before = len(steps(g))
        action, args = hold(g, 'first_reveal', 1.)
        assert action['status'] == 'cancelled' and action['termination_reason'] == 'board_turn_changed'
        assert action['elapsed_sim_s'] == pytest.approx(.32)
        assert len(steps(g))-before == 16
        assert state(g)['turn_index'] == 1 and state(g)['current_player'] == 'player_0'
        assert state(g)['phase'] == 'reveal_second'
        assert all(row['actor'] == {'player_id': 'player_0', 'turn_index': 0} for row in steps(g)[before:])
        count = len(steps(g))
        replay = call(g, 'hold', args)
        assert replay.status_code == 200 and replay.json()['result'] == action
        assert len(steps(g)) == count
        stale_http = call(g, 'hold', {**args, 'action_id': 'stale_observation_new_action'})
        assert stale_http.status_code == 409 and not stale_http.json()['ok']
        assert len(steps(g)) == count
        stale = g.worker.judge.request({'type': 'reveal_card', 'card_id': 'card_01'},
                                      actor={'player_id': 'player_0', 'turn_index': 0})
        assert stale['board_game_error']['code'] == 'invalid_arguments'
        with pytest.raises(ValueError, match='epoch'):
            g.worker.judge.authorize_step(steps(g)[-1]['command'], {'player_id': 'player_0', 'turn_index': 0})
        denied = call(g, 'hold', {'action_id': 'other_player', 'observation_sequence': g.session.sequence,
                                  'duration_s': .02}, player='player_1')
        assert denied.status_code == 403 and len(steps(g)) == count
        intent(g, {'type': 'reveal_card', 'card_id': 'card_01'})
        hold(g, 'second_reveal', .02)
        assert steps(g)[-1]['actor'] == {'player_id': 'player_0', 'turn_index': 1}


def test_both_players_page_identical_reveal_history_and_cache_survives_worker_close(game):
    with game() as g:
        hold(g, 'baseline')
        intent(g, {'type': 'reveal_card', 'card_id': 'card_00'})
        g.worker.mode = 'execute_pending'
        hold(g, 'reveal', 1.)
        for player in PLAYERS:
            first = event_page(g, player, limit=1)
            assert first['events'][0]['type'] == 'card_revealed'
            assert first['next_cursor'] == 0 and first['has_more']
            second = event_page(g, player, after=0, limit=1)
            assert second['events'][0]['type'] == 'operation_committed'
            assert second['next_cursor'] == 1 and not second['has_more']
        assert event_page(g, PLAYERS[0]) == event_page(g, PLAYERS[1])
        assert g.client.get(endpoint(g)+'/events').status_code == 403
        assert g.client.get(endpoint(g)+'/events', headers=headers(g, 'player_1')).status_code == 403
        finished = call(g, 'finish_episode')
        assert finished.status_code == 200, finished.text
        assert g.worker.closed and g.session.done.is_set()
        calls_before = deepcopy(g.worker.calls)
        files_before = {str(p): p.read_bytes() for p in g.session.directory.rglob('*') if p.is_file()}
        assert event_page(g, PLAYERS[0]) == event_page(g, PLAYERS[1])
        assert result(g, PLAYERS[0]) == result(g, PLAYERS[1])
        assert g.worker.calls == calls_before
        assert {str(p): p.read_bytes() for p in g.session.directory.rglob('*') if p.is_file()} == files_before


@pytest.mark.parametrize('mode,end_reason,episode_outcome,physics_valid', [
    ('unexpected_reveal', 'rule_violation', 'task_failure', True),
    ('physical_failure', 'physical_failure', 'safety_stop', False),
    ('missing_evidence', 'missing_physical_evidence', 'evaluator_invalid', False),
])
def test_terminal_referee_fault_seals_at_same_ack_without_winner(game, mode, end_reason, episode_outcome, physics_valid):
    with game() as g:
        hold(g, 'baseline')
        g.worker.mode = mode
        before = len(steps(g))
        hold(g, 'fault', 1.)
        assert len(steps(g)) == before+1  # No extra old-player command after terminal ACK.
        assert g.worker.closed and g.session.done.is_set() and g.session.life.active_action_id is None
        final = result(g)
        assert final['board_game']['status']['end_reason'] == end_reason
        assert final['board_game']['status']['outcome'] == 'aborted'
        assert final['board_game']['winner'] is None and final['board_game']['draw'] is False
        assert final['board_game']['turn_index'] == 0
        assert final['episode']['outcome'] == episode_outcome
        assert final['episode']['physics_valid'] is physics_valid
        assert g.session.raw['sim_time_s'] == g.worker.step_replies[-1]['observation']['sim_time_s']
        assert event_page(g, 'player_0') == event_page(g, 'player_1')
        if mode == 'unexpected_reveal':
            assert event_page(g)['events'][0]['details']['expected'] is False
        else:
            assert event_page(g)['events'] == []
        denied = call(g, 'hold', {'action_id': 'after_terminal',
                                  'observation_sequence': g.session.sequence, 'duration_s': .02})
        assert denied.status_code == 409 and len(steps(g)) == before+1


@pytest.mark.parametrize('budget', ['wall', 'simulation'])
def test_budget_supersedes_late_real_referee_win_and_caches_only_eligible_events(game, budget):
    with game('barrier_race') as g:
        clock = ManualTime()
        g.session.clock._monotonic, g.session.clock._start = clock, 0.
        horizon = g.session.spec.budgets.wall_time_s if budget == 'wall' else g.session.spec.budgets.sim_time_s
        hold(g, 'baseline')
        g.worker.mode = 'execute_pending'
        captured = []
        original = g.worker.request
        def delayed(operation, **payload):
            reply = original(operation, **payload)
            if operation == 'step' and reply['board_game']['status']['outcome'] == 'win':
                if budget == 'wall':
                    clock.value = horizon  # Equality already exhausts wall time.
                else:
                    # Deliberately late synthetic ACK, not a physical rollout.
                    g.worker.raw['sim_time_s'] = reply['observation']['sim_time_s'] = horizon+.01
                captured.append(deepcopy(reply))
            return reply
        g.worker.request = delayed
        history = [('player_0', [2, 1]), ('player_1', [2, 3]), ('player_0', [2, 2]),
                   ('player_1', [2, 1]), ('player_0', [2, 3]), ('player_1', [2, 0])]
        for i, (player, cell) in enumerate(history):
            intent(g, {'type': 'move_pawn', 'destination': cell}, player=player)
            before = state(g, player)
            hold(g, 'pawn_'+str(i), 1., player=player)
        assert len(captured) == 1
        final = result(g)
        assert final['episode']['outcome'] == ('wall_timeout' if budget == 'wall' else 'sim_timeout')
        assert final['board_game']['status']['outcome'] == 'truncated'
        assert final['board_game']['winner'] is None and not final['board_game']['draw']
        assert final['board_game']['turn_index'] == before['turn_index'] == 5
        assert final['board_game']['event_cursor'] == before['event_cursor'] == 4
        assert g.worker.closed and g.session.done.is_set()
        page = event_page(g)
        assert len(page['events']) == 5 and page['next_cursor'] == 4
        assert page == event_page(g, 'player_1')
        private = load_json(g.session.directory/'private/late_card_game_events.json')
        assert private['accepted_event_cursor'] == 4
        assert private['late_native_events'] == captured[0]['card_game_events']
        receipt = load_json(g.session.directory/'private/late_game_receipt.json')
        assert receipt['native_state']['winner'] == 'player_1'
        assert receipt['native_score']['success'] is True
        assert load_json(g.session.directory/'private/evaluator_evidence.json')['success'] is True
