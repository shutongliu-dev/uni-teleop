"""Public Task4 reset phase with real CPU controllers and synthetic referee ACKs.

This worker injects measured drift and a short reset transition solely to test
client gating. It does not simulate tower stability or qualify native physics.
"""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_http import FunctionHTTPClient, create_turn_game
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.function_session import FunctionSession
from bench.protocol.functions import ACTION_FUNCTIONS
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_agent_service import FakeWorker
from test_function_runner import ScriptedModel
from test_living_function_runner import FrozenTowerWorker, PLAYERS, context


ACK_HOLD = {'function': 'hold', 'arguments': {'duration_s': .02, 'mode': 'acknowledged_drive'}}


class ResettingTowerWorker(FrozenTowerWorker):
    def __init__(self, *, abort_reset=False, **kwargs):
        super().__init__(**kwargs)
        self.abort_reset = abort_reset
        self.reset_commands = []
        self.state['public_progress']['status'] = 'reset_pending'
        self.state['public_progress']['legal_blocks'] = []
        self.drift()
        self.initial.update(observation=deepcopy(self.raw), turn_game=deepcopy(self.state))

    def drift(self):
        for side in ('front', 'rear'):
            name = f'fr3_{side}_joint1'
            self.raw['joint_positions_rad'][name] = self.raw['robot']['joint_targets_rad'][name] - .0001

    def request(self, operation, **payload):
        if self.state['public_progress']['status'] == 'reset_pending':
            if operation == 'turn_game':
                assert payload['command']['type'] == 'end_episode', 'Client sent an intent during reset'
            if operation == 'step':
                assert payload['actor'] == {'player_id': self.state['current_player_id'], 'turn_index': 0}
                targets = payload['command']['joint_positions_rad']
                for name, target in targets.items():
                    assert np.float32(target) == np.float32(self.raw['robot']['joint_targets_rad'][name]), name
                self.reset_commands.append(deepcopy(payload))
                reply = FakeWorker.request(self, operation, **payload)
                self.raw['robot']['joint_targets_rad'] = dict(targets)
                self.drift()
                self.state['sim_time_s'] = self.raw['sim_time_s']
                if len(self.reset_commands) == 2:
                    if self.abort_reset:
                        self.state.update(status='interrupted', current_player_id=None,
                            terminal_reason='invalid_reset_not_stable', violation_reason='invalid_reset_not_stable')
                    else:
                        self.state['public_progress'].update(status='await_intent', legal_blocks=['block_03', 'block_06'])
                self.sync_score()
                reply.update(observation=deepcopy(self.raw), score=deepcopy(self.score),
                    turn_game=deepcopy(self.state), turn_game_turn_changed=False)
                return reply
        return super().request(operation, **payload)


def reset_service(tmp_path, *, abort_reset=False):
    release = Path(task_definition('living_room', 'tower_extraction')['source_release']['path'])
    workers = []

    def factory(**kwargs):
        worker = ResettingTowerWorker(abort_reset=abort_reset, **kwargs)
        workers.append(worker)
        return worker

    return AgentService(tmp_path / 'episodes', release=release,
        registry=default_registry(release), worker_factory=factory), workers


def hosts(http, tmp_path):
    bench = BenchHTTPClient('http://testserver', client=http)
    ready = create_turn_game(bench, {'task_id': 'tower_extraction', 'seed': 0,
        'agent_id': 'cpu_reset_test', 'agent_version': 'test', 'agent_config_sha256': 'a' * 64})
    result = []
    for player in ready['players']:
        client = FunctionHTTPClient('http://testserver', ready['episode_id'], player=player,
            session_kind='turn_game', client=http, spec=ready['spec'])
        client.bind_turn_state(ready['turn_game'])
        host = FunctionSession(client, output=tmp_path / player['binding']['player_id'], spec=ready['spec'],
            trial_id=player['binding']['player_id'], max_actions=40, max_calls=50, max_status_queries=100000)
        host.initialize()
        result.append(host)
    return result


def test_reset_tools_and_guard_reject_motion_and_intents_before_public_submission(tmp_path):
    api, workers = reset_service(tmp_path)
    with TestClient(create_app(service=api)) as http:
        front, rear = hosts(http, tmp_path)
        value = front.context()
        tools = {row['function']: row for row in value['tools'] if 'function' in row}
        assert set(tools) & ACTION_FUNCTIONS == {'hold'}
        assert {'observe', 'get_robot', 'query_ik', 'get_action', 'cancel_action', 'finish_episode'} <= set(tools)
        schema = tools['hold']['arguments_schema']
        assert schema['properties']['mode']['enum'] == ['acknowledged_drive']
        assert 'default' not in schema['properties']['mode'] and 'mode' in schema['required']
        assert not any('intent' in row for row in value['tools'] + value['decision_schema']['one_of'])
        assert 'reset_pending' in value['host_contract']['reset_phase']
        assert 'Read-only queries do not advance simulation' in value['host_contract']['reset_phase']
        assert 'await_intent' in value['host_contract']['reset_phase']
        assert not workers[0].reset_commands  # Initialization never advances physics for the model.
        decisions = [
            {'function': 'hold', 'arguments': {'duration_s': .02}},
            {'function': 'hold', 'arguments': {'duration_s': .02, 'mode': 'reference'}},
            {'function': 'move_wrist', 'arguments': {'hand': 'front', 'timeout_sim_s': 1.,
                'wrist': {'frame_id': 'world', 'position_m': [.65, -.4, 1.3], 'quaternion_wxyz': [1., 0., 0., 0.]}}},
            {'function': 'set_hand_targets', 'arguments': {'hand': 'front', 'timeout_sim_s': 1.,
                'hand_posture_rad': {'front_thumb_IP': .4}}},
            {'function': 'move_geometry', 'arguments': {'timeout_sim_s': 1.,
                'hand_geometries': {'front': {'hand_posture_rad': {'front_thumb_IP': .4}}}}},
            {'function': 'move_joints', 'arguments': {'duration_s': .02,
                'arm_joint_positions_rad': {'fr3_front_joint1': .01}, 'hand_joint_positions_rad': {'front_thumb_IP': .4}}},
            *({'intent': {'type': 'turn_intent', 'name': name, 'arguments': arguments}}
              for name, arguments in [('select', {'block_id': 'block_03'}), ('wait', {}), ('cancel', {})]),
        ]
        for decision in decisions:
            feedback = front.execute(decision)
            if decision.get('function') == 'move_joints':
                assert feedback['error']['code'] == 'unsupported_function', feedback
            else:
                assert feedback['error']['code'] == 'permission_denied', feedback
                assert 'reset_pending' in feedback['error']['message']
            assert feedback['pending_action'] is None
        assert rear.execute(ACK_HOLD)['error']['code'] == 'permission_denied'
        assert front.execute({'function': 'observe', 'arguments': {}})['ok']
        assert not workers[0].reset_commands and not any(operation in ('step', 'turn_game') for operation, _ in workers[0].calls)
        assert next(iter(api.sessions.values())).clock.action_attempts == 0
        assert front.execute(ACK_HOLD)['ok']
        assert front.state['turn_game']['public_progress']['status'] == 'reset_pending'
        assert front.execute({'function': 'get_action', 'arguments': {}})['ok']
        assert front.execute({'function': 'cancel_action', 'arguments': {}})['ok']
        assert len(workers[0].reset_commands) == 1
        assert front.execute(ACK_HOLD)['ok']
        ready = front.context()
        assert ready['state']['turn_game']['public_progress']['status'] == 'await_intent'
        assert ready['state']['turn_game']['turn_index'] == 0
        assert 'reset_phase' not in ready['host_contract']
        assert any('intent' in row for row in ready['tools'])
        assert 'move_wrist' in {row.get('function') for row in ready['tools']}
        assert front.execute({'intent': {'type': 'turn_intent', 'name': 'select', 'arguments': {'block_id': 'block_03'}}})['ok']


@pytest.mark.parametrize('seed', [0, 1])
def test_runner_explicit_reset_holds_then_multiple_actions_and_native_handoff(tmp_path, seed):
    api, workers = reset_service(tmp_path)
    first, second = PLAYERS[seed], PLAYERS[1 - seed]
    models = {first: ScriptedModel([
        {'intent': {'type': 'turn_intent', 'name': 'select', 'arguments': {'block_id': 'block_03'}}},
        {'function': 'hold', 'arguments': {'duration_s': .02}},
        deepcopy(ACK_HOLD), deepcopy(ACK_HOLD),
        {'intent': {'type': 'turn_intent', 'name': 'select', 'arguments': {'block_id': 'block_03'}}},
        {'function': 'hold', 'arguments': {'duration_s': .02}},
        {'function': 'hold', 'arguments': {'duration_s': .2}}]),
        second: ScriptedModel([{'function': 'observe', 'arguments': {}}, {'stop': True}])}
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task4', tmp_path / 'run', family='living_room', seed=seed,
                              max_decisions=8, max_status_queries=100000),
                     bench=BenchHTTPClient('http://testserver', client=http), models=models)
    assert report['error_type'] is None and report['finalization_error'] is None, report
    assert report['termination'] == 'model_requested_stop' and report['run_status'] == 'completed'
    assert report['decisions'] == {first: 7, second: 2}
    phases = [context(turns)['state']['turn_game']['public_progress']['status'] for turns in models[first].turns]
    assert phases == ['reset_pending'] * 4 + ['await_intent'] * 3
    assert len(workers[0].reset_commands) == 2 and workers[0].step_count == 2
    assert report['result']['turn_game']['turn_index'] == 1
    assert report['evaluation_player_outcomes'] == dict.fromkeys(PLAYERS, 'unknown')
    for player, model in models.items():
        for turns in model.turns:
            value = context(turns)
            assert value['state']['player']['player_id'] == value['state']['turn_game']['current_player_id'] == player
    assert any(turn.get('role') == 'assistant' and json.loads(turn['content']).get('intent', {}).get('name') == 'select'
               for turn in models[first].turns[-1])
    assert not any(turn.get('role') == 'assistant' and 'block_03' in str(turn['content'])
                   for turns in models[second].turns for turn in turns)
    assert workers[0].closed


@pytest.mark.parametrize('ending', ['stop', 'interrupt', 'decision_budget', 'invalid_reset'])
def test_reset_interruption_never_admits_a_turn_or_manufactures_a_loser(tmp_path, ending):
    api, workers = reset_service(tmp_path, abort_reset=ending == 'invalid_reset')
    decisions = ([deepcopy(ACK_HOLD), deepcopy(ACK_HOLD)] if ending == 'invalid_reset' else
                 [KeyboardInterrupt()] if ending == 'interrupt' else
                 [{'function': 'observe', 'arguments': {}}] if ending == 'decision_budget' else [{'stop': True}])
    models = {'player_0': ScriptedModel(decisions), 'player_1': ScriptedModel([])}
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task4', tmp_path / 'run', family='living_room', seed=0,
                              max_decisions=1 if ending == 'decision_budget' else 4, max_status_queries=100000),
                     bench=BenchHTTPClient('http://testserver', client=http), models=models)
    assert report['finalization_error'] is None, report
    assert report['result']['turn_game']['status'] == 'interrupted'
    assert report['result']['turn_game']['turn_index'] == 0
    assert report['result']['turn_game']['winner_player_id'] is None
    assert report['result']['turn_game']['loser_player_id'] is None
    assert report['evaluation_player_outcomes'] == dict.fromkeys(PLAYERS, 'unknown')
    assert report['result_valid_for_scoring'] is False
    assert not models['player_1'].turns and not workers[0].step_count
    assert len(workers[0].reset_commands) == (2 if ending == 'invalid_reset' else 0)
    assert workers[0].closed
