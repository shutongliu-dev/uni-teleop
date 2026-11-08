"""Private model bindings over real public HTTP with CPU/scripted doubles.

No provider request, Isaac simulation or model/robot qualification occurs here.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent.model_client import function_runner
from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.player_models import load_player_models
from bench.paths import OUTPUT_ROOT, load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker


PLAYERS = ('player_0', 'player_1')


def binding_rows(providers=('gateway', 'micu')):
    return {player: {'provider': providers[i], 'model': 'exact-model-' + player,
                     'reasoning_effort': ('max', 'high')[i], 'credentials': player + '.json'}
            for i, player in enumerate(PLAYERS)}


def write_bindings(tmp_path, rows=None):
    path = tmp_path / 'private-bindings.json'
    path.write_text(json.dumps(binding_rows() if rows is None else rows))
    return path


def test_private_binding_paths_resolve_beside_real_json_without_reading_keys(tmp_path, monkeypatch):
    real = tmp_path / 'profiles'
    real.mkdir()
    config = write_bindings(real)
    alias = tmp_path / 'link.json'
    alias.symlink_to(config)
    monkeypatch.chdir(tmp_path)
    bindings = load_player_models(alias)
    for player in PLAYERS:
        assert bindings[player].credentials == real / (player + '.json')
        assert not bindings[player].credentials.exists()  # Loader never needs API keys.
        assert 'credentials' not in bindings[player].public_metadata()
        assert str(real) not in repr(bindings[player])


@pytest.mark.parametrize('fault', ['missing_player', 'extra_player', 'missing_field', 'extra_field',
    'inline_key', 'null_profile', 'empty_model', 'padded_effort', 'non_string_provider', 'wrong_top',
    'wrong_row', 'duplicate_player', 'duplicate_field', 'invalid_json'])
def test_incomplete_extra_or_ambiguous_private_bindings_are_rejected_safely(tmp_path, fault):
    rows = binding_rows()
    if fault == 'missing_player': del rows['player_1']
    elif fault == 'extra_player': rows['private-secret-value'] = deepcopy(rows['player_0'])
    elif fault == 'missing_field': del rows['player_1']['reasoning_effort']
    elif fault == 'extra_field': rows['player_0']['fallback'] = 'private-secret-value'
    elif fault == 'inline_key': rows['player_0']['api_key'] = 'private-secret-value'
    elif fault == 'null_profile': rows['player_0']['credentials'] = None
    elif fault == 'empty_model': rows['player_0']['model'] = ''
    elif fault == 'padded_effort': rows['player_0']['reasoning_effort'] = ' max '
    elif fault == 'non_string_provider': rows['player_0']['provider'] = True
    elif fault == 'wrong_top': rows = [rows]
    elif fault == 'wrong_row': rows['player_0'] = 'private-secret-value'
    path = write_bindings(tmp_path, rows)
    if fault == 'duplicate_player':
        path.write_text('{"player_0":{},"player_0":"private-secret-value","player_1":{}}')
    elif fault == 'duplicate_field':
        path.write_text('{"player_0":{"model":"a","model":"private-secret-value"},"player_1":{}}')
    elif fault == 'invalid_json': path.write_text('{private-secret-value')
    with pytest.raises(ValueError) as error:
        load_player_models(path)
    assert 'private-secret-value' not in str(error.value)
    assert str(path) not in str(error.value)


def test_binding_does_not_guess_models_or_normalize_reasoning_levels(tmp_path):
    rows = binding_rows(('future_explicit_provider', 'qwen'))
    rows['player_0']['model'] = 'Vendor/CaseSensitive-20261028'
    rows['player_0']['reasoning_effort'] = 'ultra'
    bindings = load_player_models(write_bindings(tmp_path, rows))
    assert bindings['player_0'].public_metadata() == {
        'provider': 'future_explicit_provider', 'model': 'Vendor/CaseSensitive-20261028',
        'reasoning_effort': 'ultra'}  # Factory/transport must validate actual support.


def test_invalid_binding_fails_before_any_transport_or_recording(tmp_path):
    path = write_bindings(tmp_path, {'player_0': {}})
    with pytest.raises(ValueError):
        run(RunConfig('https://must-not-connect.invalid', 'task1', tmp_path / 'run', player_models=path))
    assert not (tmp_path / 'run').exists()


def test_explicit_bindings_cannot_be_silently_overridden_by_injected_models(tmp_path):
    with pytest.raises(ValueError, match='either'):
        run(RunConfig('https://must-not-connect.invalid', 'task1', tmp_path / 'run',
                      player_models=write_bindings(tmp_path)), models={})
    assert not (tmp_path / 'run').exists()


def test_single_agent_rejects_player_bindings_before_credentials_or_episode(tmp_path, monkeypatch):
    class Bench:
        def list_tasks(self): pytest.fail('Single-agent bindings must fail before any HTTP request')
    def forbidden(*args, **kwargs): pytest.fail('Single-agent binding must not load credentials or models')
    monkeypatch.setattr(function_runner.ModelGatewayConfig, 'from_file', forbidden)
    monkeypatch.setattr(function_runner, 'configured_model', forbidden)
    report = run(RunConfig('http://unused', 'task0', tmp_path / 'run',
                          player_models=write_bindings(tmp_path)), bench=Bench())
    assert report['episode_id'] is None and report['decisions'] == {}
    assert report['run_status'] == 'configuration_error'
    assert report['task_setup_error']['code'] == 'player_models_require_multiplayer'


def test_second_client_failure_closes_first_and_does_not_log_secret_exception(tmp_path, monkeypatch):
    path = write_bindings(tmp_path)
    for player in PLAYERS:
        (tmp_path / (player + '.json')).write_text(json.dumps({
            'base_url': 'https://example.invalid/v1', 'api_key': 'private-key-' + player}))
    closed = []
    def configured_model(config, credentials):
        if config.model == 'exact-model-player_1':
            raise ValueError('private-second-client-secret at ' + str(config.credentials))
        return SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(function_runner, 'configured_model', configured_model)
    class Bench:
        def list_tasks(self): return [{'task_id': 'tic_tac_toe_3x3', 'task_kind': 'board_game'}]
    report = run(RunConfig('http://unused', 'task1', tmp_path / 'run', player_models=path), bench=Bench())
    assert closed == [True] and report['episode_id'] is None
    assert report['error_type'] == 'ValueError'
    records = '\n'.join(p.read_text() for p in (tmp_path / 'run').glob('*.json'))
    assert 'private-second-client-secret' not in records
    assert 'private-key-' not in records and str(path) not in records


class BoundModel:
    def __init__(self, player, config):
        self.player, self.config = player, config
        color = 'black' if player == 'player_0' else 'white'
        self.decisions = iter([
            {'function': 'private_reason_' + player, 'arguments': {}},
            {'intent': {'type': 'placement_intent', 'piece_id': color + '_00',
                        'q': 0 if player == 'player_0' else 1, 'r': 0}},
            {'function': 'hold', 'arguments': {'duration_s': .4}}, {'stop': True}])
        self.turns, self.calls, self.closed = [], [], False

    def public_metadata(self):
        value = {'provider': self.config.provider, 'model': self.config.model,
                 'reasoning_effort': self.config.reasoning_effort, 'protocol': 'CPU_double'}
        if self.config.provider == 'qwen': value['enable_thinking'] = True
        return value

    def generate(self, prompt, turns, **kwargs):
        self.turns.append(deepcopy(turns))
        self.calls.append(kwargs)
        assert json.loads(turns[-1]['content'][0]['text'])['state']['player']['player_id'] == self.player
        if self.config.provider == 'micu':
            assert kwargs['retry_deadline'] is not None
            kwargs['on_event']({'event': 'model_request_started', 'attempt': 1})
            kwargs['on_event']({'event': 'model_request_completed', 'attempt': 1})
        else:
            assert set(kwargs) == {'max_output_tokens', 'timeout_s'}
        return SimpleNamespace(text=json.dumps(next(self.decisions)),
                               public_metadata=lambda: {**self.public_metadata(), 'attempts': 1})

    def close(self): self.closed = True


@pytest.mark.parametrize('providers', [('gateway', 'micu'), ('micu', 'qwen')])
def test_private_bindings_construct_distinct_clients_and_follow_authenticated_turns(
        tmp_path, monkeypatch, capsys, providers):
    rows = binding_rows(providers)
    path = write_bindings(tmp_path, rows)
    secrets = {player: 'private-api-key-' + player for player in PLAYERS}
    for player in PLAYERS:
        (tmp_path / (player + '.json')).write_text(json.dumps({
            'base_url': 'https://' + player.replace('_', '-') + '.example.invalid/v1',
            'api_key': secrets[player]}))
    built = {}
    def configured_model(config, credentials):
        player = next(player for player in PLAYERS if credentials.api_key == secrets[player])
        assert config.credentials == (tmp_path / (player + '.json')).resolve()
        assert config.provider == rows[player]['provider']
        assert config.model == rows[player]['model']
        assert config.reasoning_effort == rows[player]['reasoning_effort']
        assert config.model_timeout == 17 and config.model_retries == 2
        model = BoundModel(player, config)
        built[player] = model
        return model
    monkeypatch.setattr(function_runner, 'configured_model', configured_model)
    release = Path(task_definition('card_room', 'tic_tac_toe_3x3')['source_release']['path'])
    def factory(**kwargs):
        worker = FakeBoardWorker(**kwargs)
        worker.settled_pending = True  # Synthetic evidence, never native game qualification.
        return worker
    api = AgentService(tmp_path / 'episodes', release=release, registry=default_registry(release),
                       worker_factory=factory)
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task1', tmp_path / 'run', family='card_room',
            provider='micu', model='unused-global-model', reasoning_effort='unused-global-effort',
            credentials=tmp_path / 'unused-global-profile.json', player_models=path,
            model_timeout=17, model_retries=2, max_decisions=6),
            bench=BenchHTTPClient('http://testserver', client=http))
    assert report['error_type'] is None and report['finalization_error'] is None, report
    assert report['decisions'] == {'player_0': 4, 'player_1': 3}
    assert all(model.closed for model in built.values())
    assert built['player_0'] is not built['player_1']
    assert 'private_reason_player_0' not in json.dumps(built['player_1'].turns)
    assert 'private_reason_player_1' not in json.dumps(built['player_0'].turns)
    settings = load_json(tmp_path / 'run/agent_config.json')
    assert settings['model_bindings'] == {
        player: {key: value for key, value in row.items() if key != 'credentials'}
        for player, row in rows.items()}
    for player in PLAYERS:
        assert settings['models'][player] == built[player].public_metadata()
        events = list((tmp_path / 'run' / player).rglob('model_api_events.jsonl'))
        assert bool(events) is (rows[player]['provider'] == 'micu')
        for event_file in events:
            assert all(json.loads(line)['player'] == player for line in event_file.read_text().splitlines())
    records = '\n'.join(p.read_text() for p in (tmp_path / 'run').rglob('*')
                        if p.is_file() and p.suffix in ('.json', '.jsonl')) + capsys.readouterr().out
    for secret in [*secrets.values(), str(path), 'unused-global-model', 'unused-global-effort',
                   *[str(tmp_path / (player + '.json')) for player in PLAYERS]]:
        assert secret not in records


def test_cli_keeps_private_binding_path_and_exact_global_provider(monkeypatch, tmp_path):
    from scripts.maintenance import freeze_project_sources
    captured = []
    monkeypatch.setattr(freeze_project_sources, 'verify_runtime_imports', lambda *args, **kwargs: {'test': True})
    def record(config, **kwargs):
        captured.append(config)
        return {'episode_id': None, 'run_status': 'completed', 'termination': 'test', 'task_setup_error': None}
    monkeypatch.setattr(function_runner, 'run', record)
    path = tmp_path / 'private-player-models.json'
    monkeypatch.setattr(sys, 'argv', ['function_runner', '--project-root', str(tmp_path),
        '--bench-url', 'http://unused', '--family', 'card_room', '--task', 'task6',
        '--provider', 'qwen', '--player-models', str(path),
        '--output', f'{OUTPUT_ROOT}/private-binding-cli-test'])  # dexbench: runs write under outputs/
    function_runner.main()
    assert captured[0].player_models == path and captured[0].provider == 'qwen'
    assert str(path) not in repr(captured[0])
