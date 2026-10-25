"""Real HTTP contracts with CPU worker/model doubles; no native/model qualification."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, board_outcomes, learning_policy, resolve_task, run
from agent.model_client.function_http import FunctionHTTPClient
from agent.model_client.function_session import FunctionSession
from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.responses import MicuModelClient
from agent.model_client.transport import ModelGatewayConfig, ModelGatewayError
from bench.paths import OUTPUT_ROOT, load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.isaacsim.release import task_inputs
from bench.runtime.placement_tasks import project_placement_task
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker
from test_card_room_placement_runtime import placement_release
from test_agent_service import FakeWorker, publish_targets


class ScriptedModel:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.turns = []

    def public_metadata(self):
        return {'model': 'cpu_scripted_model_double', 'protocol': 'test'}

    def generate(self, prompt, turns, **kwargs):
        self.turns.append(deepcopy(turns))
        decision = next(self.decisions)
        if isinstance(decision, BaseException):
            raise decision
        text = decision if isinstance(decision, str) else json.dumps(decision)
        return SimpleNamespace(text=text, public_metadata=lambda: {'model': 'cpu_scripted_model_double', 'attempts': 1})


@pytest.mark.parametrize('alias,expected', [('task1','tic_tac_toe_3x3'), ('task2','connect_four_5x5'), ('task3','hex7x7')])
def test_numbered_board_aliases_require_actual_published_board_kind(alias, expected):
    with pytest.raises(ValueError):
        resolve_task(alias, [{'task_id': expected}])
    task, board = resolve_task(alias, [{'task_id': expected, 'task_kind': 'board_game'}])
    assert task['task_id'] == expected and board


def test_task0_is_single_agent_and_practice_is_distinct_from_transport():
    assert resolve_task('task0', [{'task_id': 'task0', 'task_kind': 'manipulation'}])[1] is False
    with pytest.raises(ValueError):
        resolve_task('task0', [{'task_id': 'task0', 'task_kind': 'board_game'}])
    assert learning_policy('task0')['practice_rsi_allowed']
    assert learning_policy('tic_tac_toe_3x3')['practice_rsi_allowed']
    for task in ('connect_four_5x5', 'hex7x7'):
        assert learning_policy(task)['frozen_skills_required']
        assert not learning_policy(task)['practice_rsi_allowed']
        assert not learning_policy(task)['skill_mutation_tools_available']


@pytest.mark.parametrize('fault', [None, 'private_create', 'pending', 'finish_failure', 'deadline',
                                 'context_deadline', 'context_timeout', 'micu_retry', 'micu_failure'])
def test_task0_uses_seeded_ready_goals_and_ordinary_functions(placement_release, tmp_path, monkeypatch, capsys, fault):
    workers = []
    def factory(**kwargs):
        worker = FakeWorker(**kwargs, depth=True, acknowledged_targets=True, on_step=publish_targets)
        _, scene, task, initial = task_inputs(kwargs['release'], kwargs['task_id'], seed=kwargs['seed'])
        geometry = public_robot_geometry(robot_configuration(scene))
        worker.arm, worker.hand, worker.active = list(geometry.arm.names), list(geometry.hand.names), list(geometry.names)
        worker.raw['joint_positions_rad'] = deepcopy(initial['joint_positions_rad'])
        worker.raw['joint_velocities_rad_s'] = dict.fromkeys(worker.active, 0.)
        worker.raw['robot'] = {'joint_targets_rad': deepcopy(initial['joint_positions_rad'])}
        worker.initial['observation'] = deepcopy(worker.raw)
        worker.initial['metadata'].update(scene_spec=scene, action_spec={
            'joint_names': worker.active, 'arm_joint_names': worker.arm, 'hand_joint_names': worker.hand,
            'joint_limits_rad': {name: [float(geometry.lower[i]), float(geometry.upper[i])]
                                 for i, name in enumerate(geometry.names)}})
        worker.initial['task'] = project_placement_task(task)
        workers.append(worker)
        return worker
    api = AgentService(tmp_path / 'episodes', release=placement_release, registry=default_registry(placement_release),
                       worker_factory=factory)
    model = ScriptedModel([{'function': 'observe', 'arguments': {}}, {'stop': True}])
    micu, requests = None, []
    if fault in ('micu_retry', 'micu_failure'):
        from agent.model_client import responses
        monkeypatch.setattr(responses, 'time', SimpleNamespace(monotonic=responses.time.monotonic,
                                                              sleep=lambda _: None))
        def handler(request):
            requests.append(request.content)
            if fault == 'micu_failure' or len(requests) == 1:
                return httpx.Response(401 if fault == 'micu_failure' else 503,
                                      text='Bearer private-test-function-key provider-body')
            text = json.dumps(next(model.decisions))
            return httpx.Response(200, json={'model': 'gpt-6-astra', 'status': 'completed',
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}],
                'usage': {'input_tokens': 10, 'output_tokens': 4}})
        micu = MicuModelClient(ModelGatewayConfig('https://example.micuapi.ai/v1', 'private-test-function-key'),
                              'gpt-6-astra', transport=httpx.MockTransport(handler))
        def generate(prompt, turns, **kwargs):
            model.turns.append(deepcopy(turns))
            return micu.generate(prompt, turns, **kwargs)
        model.generate = generate
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        if fault == 'private_create':
            original = bench.create
            def corrupt(payload):
                ready = original(payload)
                ready['observation']['privileged_object_poses'] = {'hidden': {}}
                return ready
            monkeypatch.setattr(bench, 'create', corrupt)
        if fault == 'pending':
            monkeypatch.setattr(FunctionSession, 'pending', property(lambda session: getattr(session, '_test_pending', None)))
            def unresolved(session, decision):
                session._test_pending = {'action_id': 'unknown'}
                return {'ok': False, 'pending_action': {'action_id': 'unknown'}}
            monkeypatch.setattr(FunctionSession, 'execute', unresolved)
        if fault == 'finish_failure':
            original_call = FunctionHTTPClient.call
            def fail_finish(client, payload, **kwargs):
                if payload['function'] == 'finish_episode':
                    return {'protocol': 'bench_functions_v1', 'capability_version': '1.0.0', 'ok': False,
                            'function': 'finish_episode', 'request_id': payload['request_id'], 'context': None,
                            'error': {'code': 'infrastructure_error', 'http_status': 503, 'message': 'failure'}}
                return original_call(client, payload, **kwargs)
            monkeypatch.setattr(FunctionHTTPClient, 'call', fail_finish)
        if fault == 'deadline':
            from agent.model_client import function_runner
            def expired(_):
                raise function_runner.RunDeadline()
            monkeypatch.setattr(function_runner, '_remaining', expired)
        if fault in ('context_deadline', 'context_timeout'):
            from agent.model_client import function_runner
            clock = [function_runner.time.monotonic()]
            monkeypatch.setattr(function_runner, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
            def expired_context(session):
                clock[0] = session.deadline + 1
                category = 'request_budget_exhausted' if fault == 'context_deadline' else 'transport_timeout'
                raise BenchHTTPError('Expired context', category=category)
            monkeypatch.setattr(FunctionSession, 'context', expired_context)
        try:
            report = run(RunConfig('http://testserver', 'task0', tmp_path / 'run', seed=7,
                                  provider='micu' if micu else 'gateway', max_decisions=3,
                                  max_status_queries=100000),
                         bench=bench, models={'single': model})
        finally:
            if micu:
                micu.close()
        if micu:
            decision = tmp_path / 'run/single/decisions/0000'
            events = [json.loads(line) for line in (decision / 'model_api_events.jsonl').read_text().splitlines()]
            assert all(row['decision_index'] == 0 and row['player'] == 'single' for row in events)
            recorded = load_json(decision / 'model_request.json')
            assert all(row['request_sha256'] == recorded['expanded_messages_sha256'] for row in events)
            terminal = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
            assert terminal[0]['event'] == 'model_request_started'
            assert any(row['event'] == 'model_request_failed' for row in terminal)
            assert 'private-test-function-key' not in json.dumps(events + terminal)
            assert 'provider-body' not in json.dumps(events + terminal)
            assert not [item for item in workers[0].calls if item[0] == 'step']
            if fault == 'micu_failure':
                assert len(requests) == 1
                assert report['run_status'] == 'configuration_error' and report['task_success'] is None
                error = load_json(decision / 'model_error.json')
                assert error == {'decision_index': 0, 'error_category': 'http_status', 'http_status': 401, 'attempts': 1}
                return
            assert len(requests) == 3 and requests[0] == requests[1]
            assert [row['event'] for row in events] == ['model_request_started', 'model_request_failed',
                'model_retry_wait', 'model_retry_wait_finished', 'model_request_started', 'model_request_completed']
            assert report['decisions'] == {'single': 2}
            assert report['model_requests']['single'][0]['attempts'] == 2
        if fault == 'private_create':
            assert not model.turns and report['error_type'] == 'ActionValidationError'
            assert not (tmp_path / 'run/episode_ready.json').exists()
            assert report['finalization_error'] is None
            assert report['result_source'] == 'cleanup_finish'
            assert report['result']['outcome'] == 'cancelled'
            return
        if fault == 'context_timeout':
            assert report['error_type'] == 'BenchHTTPError'
            assert report['run_status'] == 'infra_error'
            assert report['termination'] == 'run_error'
            assert not model.turns
            return
        assert report['error_type'] is None, report
        assert report['board_game'] is False
        if fault in ('pending', 'finish_failure'):
            assert report['run_status'] == 'infra_error'
            assert report['task_success'] is None
            assert report['action_recovery_pending'] == (fault == 'pending')
            return
        if fault in ('deadline', 'context_deadline'):
            assert report['termination'] == 'client_wall_deadline'
            assert report['run_status'] == 'completed'
            assert not model.turns
            return
        request = load_json(tmp_path / 'run/create_request.json')
        assert request['robot_id'] == 'sharpa_w01_right'
        assert request['seed'] == 7
        settings = load_json(tmp_path / 'run/agent_config.json')
        assert settings['action_recovery']['max_status_queries_per_wait'] == 100000
        from agent.model_client.learning.journal import Journal
        journal = Journal(tmp_path / 'run/single/host/action_journal')
        assert journal.read()[0]['payload']['recovery_policy']['max_status_queries'] == 100000
        context = json.loads(model.turns[0][-1]['content'][0]['text'])
        assert context['task']['episode_seed'] == 7
        assert len(context['task']['goal']['target_cells']) == 3
        assert 'players' not in context['state'] and 'board_game' not in context['state']
        assert not [item for item in workers[0].calls if item[0] == 'step']


@pytest.mark.parametrize('fault,status', [(None, 'completed'), ('output_budget_exhausted', 'agent_budget'),
                                        ('transport_timeout', 'infra_error'), ('interrupt', 'interrupted'),
                                        ('private_create', 'infra_error')])
def test_board_runner_authenticates_media_hands_off_and_keeps_private_histories(tmp_path, monkeypatch, fault, status):
    release = Path(task_definition('card_room', 'tic_tac_toe_3x3')['source_release']['path'])
    workers = []
    def factory(**kwargs):
        worker = FakeBoardWorker(**kwargs)
        worker.settled_pending = True  # synthetic referee evidence, never a physical success claim
        workers.append(worker)
        return worker
    api = AgentService(tmp_path / 'episodes', release=release, registry=default_registry(release), worker_factory=factory)
    p0 = ScriptedModel([
        {'function': 'private_reason_player_0', 'arguments': {}},
        {'intent': {'type': 'placement_intent', 'piece_id': 'black_00', 'q': 0, 'r': 0}},
        {'function': 'hold', 'arguments': {'duration_s': .4}}, {'stop': True}])
    p1 = ScriptedModel([
        {'intent': {'type': 'placement_intent', 'piece_id': 'white_00', 'q': 1, 'r': 0}},
        {'function': 'hold', 'arguments': {'duration_s': .4}}])
    if fault:
        p0.decisions = iter([KeyboardInterrupt() if fault == 'interrupt'
                            else ModelGatewayError('Injected model failure', category=fault)])
    with TestClient(create_app(service=api)) as http:
        bench = BenchHTTPClient('http://testserver', client=http)
        if fault == 'private_create':
            original_json = bench._json
            def corrupt_create(method, path, **kwargs):
                reply = original_json(method, path, **kwargs)
                if method == 'POST' and path == '/board-games':
                    reply['observation']['privileged_object_poses'] = {'hidden': {}}
                return reply
            monkeypatch.setattr(bench, '_json', corrupt_create)
        report = run(RunConfig('http://testserver', 'task1', tmp_path / 'run', max_decisions=6),
                     bench=bench, models={'player_0': p0, 'player_1': p1})
        assert report['run_status'] == status, report
        if fault:
            assert report['finalization_error'] is None
            assert report['result_source'] == 'cleanup_finish'
            if fault == 'private_create':
                assert not p0.turns and not p1.turns
                assert not (tmp_path / 'run/episode_ready.json').exists()
            assert report['player_outcomes'] == {'player_0': 'aborted', 'player_1': 'aborted'}
            assert report['evaluation_player_outcomes'] == {'player_0': 'unknown', 'player_1': 'unknown'}
            assert not report['result_valid_for_scoring']
            return
        assert report['error_type'] is None, report
        assert report['finalization_error'] is None, report
        assert report['board_game']
        assert report['decisions'] == {'player_0': 4, 'player_1': 2}
        assert report['result']['board_game']['commit_count'] == 2
        assert report['player_outcomes'] == {'player_0': 'aborted', 'player_1': 'aborted'}
        assert 'private_reason_player_0' in json.dumps(p0.turns[-1])
        assert 'private_reason_player_0' not in json.dumps(p1.turns)
        for player, model in [('player_0', p0), ('player_1', p1)]:
            for turns in model.turns:
                context = json.loads(turns[-1]['content'][0]['text'])
                assert context['state']['board_game']['current_player'] == player
                assert context['state']['player']['player_id'] == player
                assert context['learning_policy']['practice_rsi_allowed']
        tokens = [row['session_token'] for row in next(iter(api.sessions.values())).board_game.credentials()]
        for path in (tmp_path / 'run').rglob('*.json'):
            assert all(token not in path.read_text() for token in tokens)
        for model in (p0, p1):
            assert all(token not in json.dumps(model.turns) for token in tokens)
        assert list((tmp_path / 'run').rglob('*_depth.npy'))
        # A later recording/runtime fault must not erase accepted referee wins,
        # while the runner separately excludes unhealthy results from scoring.
        accepted = deepcopy(report['result'])
        accepted['board_game'].update(turn_index=5, commit_count=5, winner='player_0', pending_move=None)
        accepted['board_game']['status'].update(outcome='win', end_reason=None, retryable=False)
        accepted['episode'].update(outcome='infrastructure_error', physics_valid=False)
        assert board_outcomes(accepted) == {'player_0': 'win', 'player_1': 'loss'}


def test_missing_task_never_creates_episode_or_calls_model(tmp_path):
    class Bench:
        def list_tasks(self):
            return []
    report = run(RunConfig('http://testserver', 'task3', tmp_path / 'run'), bench=Bench(), models={},
                 source_verification={'test_evidence': 'verified_before_setup'})
    assert report['episode_id'] is None
    assert report['error_type'] == 'TaskSetupError'
    assert report['task_setup_error']['code'] == 'task_not_published'
    assert report['run_status'] == 'configuration_error'
    assert report['task_success'] is None
    assert report['decisions'] == {}
    assert load_json(tmp_path / 'run/source_verification.json') == {'test_evidence': 'verified_before_setup'}


@pytest.mark.parametrize('disabled', ['none', 'null', 'disabled'])
def test_cli_preserves_explicit_disabled_time_and_output_limits(monkeypatch, tmp_path, disabled):
    from agent.model_client import function_runner
    from scripts.maintenance import freeze_project_sources
    captured = []
    monkeypatch.setattr(freeze_project_sources, 'verify_runtime_imports', lambda *args, **kwargs: {'test': True})
    def record(config, **kwargs):
        captured.append(config)
        return {'episode_id': None, 'run_status': 'completed', 'termination': 'test', 'task_setup_error': None}
    monkeypatch.setattr(function_runner, 'run', record)
    options = ['sim-budget', 'wall-budget', 'model-timeout', 'http-timeout', 'max-output-tokens']
    monkeypatch.setattr(sys, 'argv', ['function_runner', '--project-root', str(tmp_path),
        '--bench-url', 'http://testserver', '--family', 'card_room', '--task', 'task0',
        '--max-status-queries', '100000',
        '--output', f'{OUTPUT_ROOT}/cli-test-output',
        *[arg for option in options for arg in ('--' + option, disabled)]])
    function_runner.main()
    assert captured[0].family == 'card_room'
    assert len(captured) == 1
    assert all(getattr(captured[0], name.replace('-', '_')) is None for name in options)
    assert captured[0].max_decisions == 128 and captured[0].max_actions == 256
    assert captured[0].max_status_queries == 100000


@pytest.mark.parametrize('invalid', [None, True, 0, -1, 1.5])
def test_invalid_status_query_limit_rejected_before_episode_setup(tmp_path, invalid):
    with pytest.raises(ValueError, match='max_status_queries'):
        run(RunConfig('http://unused', 'task0', tmp_path / 'run', max_status_queries=invalid))
    assert not (tmp_path / 'run').exists()


def test_cli_returns_failure_with_machine_readable_scene_setup_reason(monkeypatch, tmp_path, capsys):
    from agent.model_client import function_runner
    from agent.model_client.scene_tasks import TaskSetupError
    from scripts.maintenance import freeze_project_sources
    monkeypatch.setattr(freeze_project_sources, 'verify_runtime_imports', lambda *args, **kwargs: {'test': True})
    error = TaskSetupError('task_planned').public_metadata()

    def record(config, **kwargs):
        assert config.family == 'living_room' and config.task == 'task4'
        return {'episode_id': None, 'run_status': 'configuration_error', 'termination': 'run_error',
                'task_setup_error': error}

    monkeypatch.setattr(function_runner, 'run', record)
    monkeypatch.setattr(sys, 'argv', ['function_runner', '--project-root', str(tmp_path),
        '--bench-url', 'http://unused', '--family', 'living_room', '--task', 'task4',
        '--output', f'{OUTPUT_ROOT}/cli-test-unavailable'])
    with pytest.raises(SystemExit) as result:
        function_runner.main()
    assert result.value.code == 2
    assert json.loads(capsys.readouterr().out)['task_setup_error'] == error
