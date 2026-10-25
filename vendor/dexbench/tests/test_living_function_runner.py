"""Living task4 client contracts over its actual frozen release and public HTTP.

The CPU worker deliberately supplies synthetic native referee transitions. These
tests establish authentication, scheduling and reporting, never tower physics or
successful robot manipulation. Task1–3 use the ordinary-release tests next door.
"""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.function_session import FunctionSession
from agent.model_client.runner import select_geometry_policy
from agent.model_client.transport import ModelGatewayError
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from bench.simulation.isaacsim_backend.robot_registry import get_robot_spec
from scenes.workrooms.catalog import task_definition
from test_agent_service import FakeWorker, PRIVATE_MARKER
from test_function_runner import ScriptedModel


PLAYERS = ('player_0', 'player_1')
SIDES = dict(zip(PLAYERS, ('front', 'rear')))


class FrozenTowerWorker(FakeWorker):
    """Keep frozen robot/task contracts; inject clearly synthetic turn receipts."""

    def __init__(self, *, hand_id, terminal=None, **kwargs):
        assert hand_id == 'fr3_sharpa_instances'
        super().__init__(hand_id='sharpa_w01_right', depth=True,
                         acknowledged_targets=True, **kwargs)
        _, scene, task, initial = task_inputs(kwargs['release'], kwargs['task_id'], seed=kwargs['seed'])
        geometry = public_robot_geometry(robot_configuration(scene))
        self.arm, self.active = list(geometry.arm_names), list(geometry.names)
        self.hand = [name for name in self.active if name not in self.arm]
        self.raw['joint_positions_rad'] = dict(initial['joint_positions_rad'])
        self.raw['joint_velocities_rad_s'] = dict.fromkeys(self.active, 0.)
        self.raw['robot'] = {'joint_targets_rad': dict(initial['joint_positions_rad'])}
        self.initial['metadata'].update(scene_spec=scene, hand_spec=get_robot_spec(scene['robot']),
            action_spec={'joint_names': self.active, 'arm_joint_names': self.arm,
                'hand_joint_names': self.hand,
                'joint_limits_rad': {joint.name: [joint.lower, joint.upper] for joint in geometry.joints}})
        self.state = {'protocol': 'turn_game_state_v1', 'game_id': task['task_id'],
            'status': 'active', 'turn_index': 0, 'current_player_id': PLAYERS[kwargs['seed'] % 2],
            'players': list(PLAYERS), 'winner_player_id': None, 'loser_player_id': None,
            'terminal_reason': None, 'violation_reason': None, 'sim_time_s': 0.,
            'public_rules': deepcopy(task['public_rules']),
            'public_progress': {'status': 'await_intent', 'selected_block': None,
                'selection_locked': False, 'completed_turns': {'front': 0, 'rear': 0},
                'collected': {}, 'legal_blocks': ['block_03', 'block_06']}}
        self.terminal = terminal
        self.step_count = 0
        self.sync_score()
        self.initial.update(observation=deepcopy(self.raw), score=deepcopy(self.score),
                            turn_game=deepcopy(self.state))

    def sync_score(self):
        self.score = {'success': self.state['status'] == 'completed', 'private_detail': PRIVATE_MARKER,
            **{key: self.state[key] for key in ('status', 'turn_index', 'winner_player_id', 'loser_player_id')}}

    def request(self, operation, **payload):
        if operation == 'turn_game':
            self.calls.append((operation, deepcopy(payload)))
            command = payload['command']
            if command['type'] == 'end_episode':
                self.state.update(status='interrupted', current_player_id=None, terminal_reason='timeout')
            else:
                assert payload['actor'] == {'player_id': self.state['current_player_id'],
                                             'turn_index': self.state['turn_index']}
                assert command['type'] == 'turn_intent' and command['name'] == 'select'
                assert set(command['arguments']) == {'block_id'}
                self.state['public_progress']['selected_block'] = command['arguments']['block_id']
            self.sync_score()
            return {'turn_game': deepcopy(self.state), 'score': deepcopy(self.score)}
        if operation == 'step':
            actor = payload['actor']
            assert actor == {'player_id': self.state['current_player_id'], 'turn_index': self.state['turn_index']}
            side = SIDES[actor['player_id']]
            for name, target in payload['command']['joint_positions_rad'].items():
                if not name.startswith((side + '_', 'fr3_' + side + '_')):
                    assert target == self.raw['robot']['joint_targets_rad'][name]
            reply = super().request(operation, **payload)
            self.raw['robot']['joint_targets_rad'] = dict(payload['command']['joint_positions_rad'])
            self.step_count += 1
            changed = self.step_count % 2 == 0 and self.terminal is None
            if changed:
                self.state['turn_index'] += 1
                self.state['current_player_id'] = PLAYERS[1 - PLAYERS.index(actor['player_id'])]
                self.state['public_progress']['completed_turns'][side] += 1
                self.state['public_progress']['selected_block'] = None
            if self.terminal:
                self.state.update(status='interrupted' if self.terminal == 'invalid_evidence' else 'completed',
                    current_player_id=None, terminal_reason=self.terminal)
                if self.terminal == 'collapse':
                    self.state.update(loser_player_id=actor['player_id'],
                        winner_player_id=PLAYERS[1 - PLAYERS.index(actor['player_id'])])
                if self.terminal == 'invalid_evidence':
                    self.state['violation_reason'] = 'invalid_evidence'
            self.state['sim_time_s'] = self.raw['sim_time_s']
            self.sync_score()
            reply.update(observation=deepcopy(self.raw), score=deepcopy(self.score),
                         turn_game=deepcopy(self.state), turn_game_turn_changed=changed)
            return reply
        return super().request(operation, **payload)


def tower_service(tmp_path, *, terminal=None):
    definition = task_definition('living_room', 'tower_extraction')
    release = Path(definition['source_release']['path'])
    workers = []

    def factory(**kwargs):
        worker = FrozenTowerWorker(terminal=terminal, **kwargs)
        workers.append(worker)
        return worker

    return AgentService(tmp_path / 'episodes', release=release,
        registry=default_registry(release), worker_factory=factory), workers, definition


def context(turns):
    return json.loads(turns[-1]['content'][0]['text'])


def test_named_instance_policy_is_explicitly_enabled_for_the_turn_runner(tmp_path):
    api, workers, _ = tower_service(tmp_path)
    with TestClient(create_app(service=api)) as http:
        policies = BenchHTTPClient('http://testserver', client=http).list_policies(task_id='tower_extraction')
    with pytest.raises(ValueError):
        select_geometry_policy(policies, 'fr3_sharpa_instances')
    selected = select_geometry_policy(policies, 'fr3_sharpa_instances', allow_instances=True)
    assert selected['policy_id'] == 'classical_fingertip_instances'
    assert not workers  # Discovery never creates a native episode.


@pytest.mark.parametrize('seed', [0, 1])
def test_tower_turns_follow_native_handoff_not_each_function_and_keep_contexts_private(tmp_path, seed):
    api, workers, definition = tower_service(tmp_path)
    first, second = PLAYERS[seed], PLAYERS[1 - seed]
    models = {
        first: ScriptedModel([
            {'function': 'private_reason_first_player', 'arguments': {}},
            {'intent': {'type': 'turn_intent', 'name': 'select', 'arguments': {'block_id': 'block_03'}}},
            {'function': 'hold', 'arguments': {'duration_s': .02}},
            {'function': 'observe', 'arguments': {}},
            {'function': 'hold', 'arguments': {'duration_s': .2}}, {'stop': True}]),
        second: ScriptedModel([
            {'intent': {'type': 'turn_intent', 'name': 'select', 'arguments': {'block_id': 'block_06'}}},
            {'function': 'hold', 'arguments': {'duration_s': .02}},
            {'function': 'hold', 'arguments': {'duration_s': .2}}]),
    }
    requests = []
    with TestClient(create_app(service=api)) as http:
        http.event_hooks['request'].append(lambda request: requests.append(
            (request.method, request.url.path, request.headers.get('Authorization'))))
        bench = BenchHTTPClient('http://testserver', client=http)
        task = bench.list_tasks()[0]
        assert task['task_id'] == 'tower_extraction' and task['task_kind'] == 'turn_game'
        assert task['create_endpoint'] == 'POST /turn-games'
        report = run(RunConfig('http://testserver', 'task4', tmp_path / 'run', family='living_room',
                              seed=seed, max_decisions=8, max_status_queries=100000), bench=bench, models=models)
    assert report['error_type'] is None and report['finalization_error'] is None, report
    assert report['turn_game'] is True and report['board_game'] is False
    assert report['run_status'] == 'completed' and report['termination'] == 'model_requested_stop'
    assert report['decisions'] == {first: 6, second: 3}
    assert report['physical_attempts'] == {first: 2, second: 2}
    assert report['result']['turn_game']['turn_index'] == 2
    assert report['result']['turn_game']['winner_player_id'] is None
    assert report['result']['turn_game']['loser_player_id'] is None
    assert not set(report['player_outcomes'].values()) & {'win', 'loss'}
    assert report['evaluation_player_outcomes'] == dict.fromkeys(PLAYERS, 'unknown')
    assert report['result_valid_for_scoring'] is False
    assert len(workers) == 1 and workers[0].closed
    steps = [row for operation, row in workers[0].calls if operation == 'step']
    assert [(row['actor']['player_id'], row['actor']['turn_index']) for row in steps] == [
        (first, 0), (first, 0), (second, 1), (second, 1)]
    assert not any(operation == 'board_game' for operation, _ in workers[0].calls)
    assert [context(turns)['state']['turn_game']['turn_index'] for turns in models[first].turns] == [0] * 5 + [2]
    assert [context(turns)['state']['turn_game']['turn_index'] for turns in models[second].turns] == [1] * 3
    assert 'private_reason_first_player' in json.dumps(models[first].turns[-1])
    assert 'private_reason_first_player' not in json.dumps(models[second].turns)
    for player, model in models.items():
        for turns in model.turns:
            value = context(turns)
            assert value['state']['turn_game']['current_player_id'] == player
            assert value['state']['player']['player_id'] == player
            assert value['state']['player']['robot_id'] == value['state']['player']['hand'] == SIDES[player]
            assert value['learning_policy']['practice_rsi_allowed']
            assert not value['learning_policy']['frozen_skills_required']
            assert 'turn_intent' in json.dumps(value['decision_schema'])
            assert 'placement_intent' not in json.dumps(value['decision_schema'])
            intents = [option['intent'] for option in value['decision_schema']['one_of'] if 'intent' in option]
            assert any(intent.get('name') == 'select' and 'block_id' in intent['arguments'] for intent in intents)
            assert 'selection_locked' in json.dumps(value['tools'])
            assert 'board_game' not in value['state']
            assert 'evaluation' not in value['task'] and 'initial_state' not in value['task']
            assert PRIVATE_MARKER not in json.dumps(turns)
    # A referee handoff cancels the unfinished primitive; it does not claim
    # that the requested motion horizon or endpoint completed.
    handoff = context(models[first].turns[-1])['last_feedback']['result']
    assert handoff['status'] == 'cancelled'
    assert handoff['termination_reason'] == 'turn_changed'
    session = next(iter(api.sessions.values()))
    tokens = {row['binding']['player_id']: row['session_token'] for row in session.turn_game.credentials()}
    for method, path, authorization in requests:
        for player, token in tokens.items():
            if '/players/' + player + '/' in path:
                assert path.startswith('/turn-games/') and authorization == 'Bearer ' + token
        assert not (method == 'POST' and path in ('/episodes', '/board-games'))
    for path in (tmp_path / 'run').rglob('*'):
        if path.suffix in ('.json', '.jsonl'):
            assert all(token not in path.read_text() for token in tokens.values())
    assert all(token not in json.dumps([model.turns for model in models.values()]) for token in tokens.values())
    assert list((tmp_path / 'run').rglob('*_depth.npy'))
    ready = load_json(tmp_path / 'run/episode_ready.json')
    request = load_json(tmp_path / 'run/create_request.json')
    assert ready['spec']['task_release_sha256'] == definition['source_release']['manifest_sha256']
    assert request['robot_id'] == 'fr3_sharpa_instances' and request['seed'] == seed


@pytest.mark.parametrize('ending,expected_status', [
    ('decision_budget', 'completed'), ('interrupt', 'interrupted'),
    ('output_budget_exhausted', 'agent_budget'), ('transport_timeout', 'infra_error'),
])
def test_tower_client_incompletion_and_faults_never_assign_a_loser(tmp_path, ending, expected_status):
    api, workers, _ = tower_service(tmp_path)
    decision = ({'function': 'observe', 'arguments': {}} if ending == 'decision_budget' else
                KeyboardInterrupt() if ending == 'interrupt' else
                ModelGatewayError('Injected model failure', category=ending))
    models = {'player_0': ScriptedModel([decision]), 'player_1': ScriptedModel([])}
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task4', tmp_path / 'run', family='living_room',
                              seed=0, max_decisions=1),
                     bench=BenchHTTPClient('http://testserver', client=http), models=models)
    assert report['run_status'] == expected_status, report
    assert report['finalization_error'] is None and report['result_source'] == 'cleanup_finish'
    assert report['result']['turn_game']['status'] == 'interrupted'
    assert report['result']['turn_game']['winner_player_id'] is None
    assert report['result']['turn_game']['loser_player_id'] is None
    assert not set(report['player_outcomes'].values()) & {'win', 'loss'}
    assert report['evaluation_player_outcomes'] == dict.fromkeys(PLAYERS, 'unknown')
    assert report['result_valid_for_scoring'] is False
    assert len(workers) == 1 and workers[0].closed and not models['player_1'].turns
    assert not any(operation == 'step' for operation, _ in workers[0].calls)


@pytest.mark.parametrize('terminal,outcomes,valid', [
    ('collapse', {'player_0': 'loss', 'player_1': 'win'}, True),
    ('equal_turn_limit', dict.fromkeys(PLAYERS, 'draw'), True),
    ('invalid_evidence', None, False),
])
def test_tower_only_reports_native_referee_wins_draws_and_abort(tmp_path, terminal, outcomes, valid):
    api, workers, _ = tower_service(tmp_path, terminal=terminal)
    models = {'player_0': ScriptedModel([{'function': 'hold', 'arguments': {'duration_s': .2}}]),
              'player_1': ScriptedModel([])}
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task4', tmp_path / 'run', family='living_room',
                              seed=0, max_decisions=2, max_status_queries=100000),
                     bench=BenchHTTPClient('http://testserver', client=http), models=models)
    assert report['error_type'] is None and report['finalization_error'] is None, report
    assert report['termination'] == 'server_finished' and report['result_source'] == 'server_result'
    assert report['result']['turn_game']['terminal_reason'] == terminal
    assert report['result_valid_for_scoring'] is valid
    if outcomes:
        assert report['player_outcomes'] == report['evaluation_player_outcomes'] == outcomes
    else:
        assert not set(report['player_outcomes'].values()) & {'win', 'loss'}
        assert report['evaluation_player_outcomes'] == dict.fromkeys(PLAYERS, 'unknown')
    assert len(workers) == 1 and workers[0].closed and not models['player_1'].turns
    assert workers[0].step_count == 1


@pytest.mark.parametrize('fault', ['wrong_rules', 'private_observation', 'initial_turn',
                                 'invalid_current_player', 'release_mismatch', 'wrong_binding'])
def test_tower_invalid_create_never_reaches_a_model_or_moves_a_robot(tmp_path, monkeypatch, fault):
    api, workers, _ = tower_service(tmp_path)
    models = {player: ScriptedModel([{'stop': True}]) for player in PLAYERS}
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        original = bench._json

        def corrupt(method, path, **kwargs):
            value = original(method, path, **kwargs)
            if method == 'POST' and path == '/turn-games':
                if fault == 'wrong_rules':
                    value['turn_game']['public_rules']['loss'] = 'Changed after task discovery'
                    value['public_rules'] = deepcopy(value['turn_game']['public_rules'])
                elif fault == 'private_observation':
                    value['observation']['privileged_object_poses'] = {'hidden': {}}
                elif fault == 'initial_turn':
                    value['turn_game']['turn_index'] = 1
                elif fault == 'invalid_current_player':
                    value['turn_game']['current_player_id'] = 'intruder'
                elif fault == 'release_mismatch':
                    value['spec']['task_release_sha256'] = '0' * 64
                elif fault == 'wrong_binding':
                    value['players'][0]['binding'].update(robot_id='rear', hand='rear')
            return value

        monkeypatch.setattr(bench, '_json', corrupt)
        report = run(RunConfig('http://testserver', 'task4', tmp_path / 'run',
                              family='living_room', seed=0), bench=bench, models=models)
        assert report['error_type'] is not None
        assert all(not model.turns for model in models.values())
        assert not (tmp_path / 'run/episode_ready.json').exists()
        assert report['result_valid_for_scoring'] is False
        assert report['evaluation_player_outcomes'] == dict.fromkeys(PLAYERS, 'unknown')
        if fault == 'wrong_binding':
            # Untrusted identities cannot authorize even host cleanup. The
            # service owns the orphan until its watchdog/shutdown ends it.
            assert report['episode_id'] is None and report['result'] is None
        elif fault == 'release_mismatch':
            # Cleanup still ends the actual episode, but its result cannot be
            # assigned to a creation spec whose frozen release was corrupted.
            assert report['result'] is None and report['result_source'] is None
            assert report['finalization_error']['error_category'] == 'invalid_response'
        else:
            assert report['result_source'] == 'cleanup_finish' and report['finalization_error'] is None
            assert report['result']['turn_game']['status'] == 'interrupted'
        if fault == 'release_mismatch':
            assert report['task_setup_error']['code'] == 'release_mismatch'
            assert report['run_status'] == 'configuration_error'
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)


def test_tower_unresolved_action_blocks_opponent_and_does_not_invent_a_result(tmp_path, monkeypatch):
    api, workers, _ = tower_service(tmp_path)
    models = {'player_0': ScriptedModel([{'function': 'hold', 'arguments': {'duration_s': .2}}]),
              'player_1': ScriptedModel([])}
    monkeypatch.setattr(FunctionSession, 'pending',
        property(lambda session: getattr(session, '_test_pending', None)))

    def unresolved(session, decision):
        session._test_pending = {'action_id': 'unknown'}
        return {'ok': False, 'pending_action': {'action_id': 'unknown'}}

    monkeypatch.setattr(FunctionSession, 'execute', unresolved)
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task4', tmp_path / 'run', family='living_room', seed=0),
                     bench=BenchHTTPClient('http://testserver', client=http), models=models)
        assert report['run_status'] == 'infra_error' and report['action_recovery_pending']
        assert report['termination'] == 'action_recovery_pending'
        assert report['result'] is None and report['result_valid_for_scoring'] is False
        assert report['player_outcomes'] == report['evaluation_player_outcomes'] == dict.fromkeys(PLAYERS, 'unknown')
        assert len(models['player_0'].turns) == 1 and not models['player_1'].turns
        assert not any(operation == 'turn_game' and payload['command']['type'] == 'end_episode'
                       for operation, payload in workers[0].calls)
    assert workers[0].closed
