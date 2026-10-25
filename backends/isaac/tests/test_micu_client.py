import json

import httpx
import pytest

from agent.model_client import responses
from agent.model_client.responses import MicuModelClient
from agent.model_client.transport import ModelGatewayConfig, ModelGatewayError
from bench.paths import load_json


CONFIG = ModelGatewayConfig('https://api-slb.micuapi.ai/v1', 'private-test-key')


def completed():
    return {'model': 'gpt-6-astra', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'ready'}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 3}}


def client(handler, **kwargs):
    return MicuModelClient(CONFIG, 'gpt-6-astra', transport=httpx.MockTransport(handler), **kwargs)


def test_micu_selection_keeps_configured_endpoint_and_multiturn_images():
    def handler(request):
        assert str(request.url) == 'https://api-slb.micuapi.ai/v1/responses'
        assert request.headers['authorization'] == 'Bearer private-test-key'
        payload = json.loads(request.content)
        assert payload['model'] == 'gpt-6-astra'
        assert payload['reasoning'] == {'effort': 'max'}
        assert payload['store'] is False and payload['stream'] is True
        assert payload['instructions'] == 'public rules'
        assert [x['role'] for x in payload['input']] == ['user', 'assistant', 'user']
        assert payload['input'][1]['content'] == [{'type': 'output_text', 'text': 'previous code'}]
        assert payload['input'][2]['content'][1] == {
            'type': 'input_image', 'image_url': 'data:image/png;base64,iVBORw0KGgo=', 'detail': 'high'}
        assert 'max_output_tokens' not in payload
        return httpx.Response(200, json=completed())
    with client(handler) as model:
        result = model.generate('public rules', [
            {'role': 'user', 'content': 'old observation'},
            {'role': 'assistant', 'content': 'previous code'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'new observation'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo=', 'detail': 'high'}}]},
        ], max_output_tokens=None)
        metadata = model.public_metadata()
        assert metadata['provider'] == 'micu' and metadata['protocol'] == 'responses'
        assert result.attempts == 1 and result.usage['input_tokens'] == 10
        assert 'private-test-key' not in json.dumps(metadata)


@pytest.mark.parametrize('failure', ['connect', 'stream', 'limited', 'unavailable'])
def test_transient_failure_retries_same_request_without_provider_fallback(monkeypatch, failure):
    calls = []
    monkeypatch.setattr(responses.time, 'sleep', lambda _: None)
    def handler(request):
        calls.append((str(request.url), request.content))
        if len(calls) == 1:
            if failure == 'connect':
                raise httpx.ConnectError('private-test-key', request=request)
            if failure == 'stream':
                raise httpx.ReadTimeout('private-test-key', request=request)
            return httpx.Response(429 if failure == 'limited' else 503, text='private-test-key')
        return httpx.Response(200, json=completed())
    with client(handler, max_retries=1) as model:
        result = model.generate('rules', [{'role': 'user', 'content': 'ready'}])
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result.attempts == 2 and result.protocol == 'responses'


def test_exhausted_provider_failures_keep_typed_sanitized_metadata(monkeypatch):
    calls = []
    monkeypatch.setattr(responses.time, 'sleep', lambda _: None)
    def handler(request):
        calls.append(request.url)
        return httpx.Response(503, text='Bearer private-test-key')
    with client(handler, max_retries=2) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('rules', [{'role': 'user', 'content': 'ready'}])
    assert len(calls) == 3
    assert error.value.public_metadata() == {'error_category': 'http_status', 'http_status': 503, 'attempts': 3}
    assert 'private-test-key' not in str(error.value)


def test_invalid_request_is_not_blindly_retried(monkeypatch):
    calls = []
    def handler(request):
        calls.append(request.url)
        return httpx.Response(400, text='provider details')
    with client(handler, max_retries=3) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('rules', [{'role': 'user', 'content': 'ready'}])
    assert len(calls) == 1 and error.value.status_code == 400


def test_retry_respects_independent_episode_deadline(monkeypatch):
    now = [0.]
    monkeypatch.setattr(responses.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(responses.time, 'sleep', lambda dt: now.__setitem__(0, now[0] + dt))
    calls = []
    def handler(request):
        calls.append(request.url)
        now[0] = 2.
        raise httpx.ReadTimeout('lost stream', request=request)
    with client(handler, timeout_s=1., max_retries=3) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('rules', [{'role': 'user', 'content': 'ready'}], retry_deadline=1.)
    assert len(calls) == 1
    assert error.value.category == 'request_budget_exhausted' and error.value.attempts == 1


def test_missing_completion_and_model_mismatch_never_fall_back():
    for value, category in [(completed() | {'model': 'other'}, 'model_mismatch'), ({}, 'model_mismatch')]:
        with client(lambda _: httpx.Response(200, json=value), max_retries=3) as model:
            with pytest.raises(ModelGatewayError) as error:
                model.generate('rules', [{'role': 'user', 'content': 'ready'}])
        assert error.value.category == category and error.value.attempts == 1


def test_micu_selection_rejects_unrelated_gateway():
    with pytest.raises(ValueError, match='Micu endpoint'):
        MicuModelClient(ModelGatewayConfig('http://127.0.0.1:8788/v1', 'unused'), 'gpt-6-astra')


def test_unlimited_timeout_and_explicit_output_bound():
    def handler(request):
        assert json.loads(request.content)['max_output_tokens'] == 1024
        return httpx.Response(200, json=completed())
    with client(handler, timeout_s=None, max_retries=0) as model:
        result = model.generate('rules', [{'role': 'user', 'content': 'ready'}], max_output_tokens=1024)
    assert result.max_output_tokens == 1024


def test_runner_selects_micu_explicitly_and_archives_actual_protocol(monkeypatch, tmp_path):
    from agent.model_client import runner
    from test_online_model_runner import setup_loop
    args, bench, model = setup_loop(monkeypatch, tmp_path, [{'action': 'stop'}])
    args.provider = 'micu'
    args.credentials = tmp_path / 'external-profile.json'
    monkeypatch.setattr(runner.ModelGatewayConfig, 'from_file', lambda _: CONFIG)
    observed = []
    model.public_metadata = lambda: {'base_url': CONFIG.base_url,
        'endpoint': CONFIG.base_url + '/responses', 'protocol': 'responses', 'provider': 'micu'}
    def selected(config, model_name, **kwargs):
        observed.append((model_name, kwargs))
        return model
    monkeypatch.setattr(runner, 'MicuModelClient', selected)
    report = runner.run(args)
    assert report['error_type'] is None
    assert observed[0][1]['max_retries'] is None
    settings = load_json(args.output / 'agent_config.json')
    assert settings['provider'] == 'micu'
    assert settings['gateway']['endpoint'].endswith('/responses')
    assert settings['gateway']['protocol'] == 'responses'
    assert settings['model_retries'] is None
    assert settings['model_retry_policy'] == 'wait_10s_same_endpoint_v2'


def test_micu_requires_explicit_profile_before_constructing_client():
    from argparse import Namespace
    from agent.model_client.runner import configured_model
    args = Namespace(provider='micu', credentials=None)
    with pytest.raises(ValueError, match='explicit external'):
        configured_model(args, CONFIG)


@pytest.mark.parametrize('streamed', [True, False])
def test_provider_output_budget_is_distinct_from_api_instability(streamed):
    calls = []
    value = completed() | {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}}
    def handler(request):
        calls.append(request)
        if streamed:
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                content='data: ' + json.dumps({'type': 'response.incomplete', 'response': value}) + '\n\n')
        return httpx.Response(200, json=value)
    with client(handler, max_retries=3) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('rules', [{'role': 'user', 'content': 'ready'}], max_output_tokens=64)
    assert len(calls) == 1
    assert error.value.public_metadata()['error_category'] == 'output_budget_exhausted'


def test_numeric_usage_extensions_cannot_echo_credentials_in_keys():
    secret = 'alphanumeric_secret'
    value = completed() | {'usage': {'input_tokens': 10, secret: 1,
        'nested': {'prefix_' + secret: 2, 'safe_counter': 3}}}
    config = ModelGatewayConfig(CONFIG.base_url, secret)
    with MicuModelClient(config, 'gpt-6-astra',
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=value))) as model:
        response = model.generate('rules', [{'role': 'user', 'content': 'ready'}])
    assert secret not in json.dumps(response.public_metadata())
    assert response.usage == {'input_tokens': 10, 'nested': {'safe_counter': 3}}


def test_micu_credentials_cannot_silently_select_chat_protocol():
    from argparse import Namespace
    from agent.model_client.runner import configured_model
    with pytest.raises(ValueError, match='explicit --provider micu'):
        configured_model(Namespace(provider='gateway', credentials='external'), CONFIG)


def fake_clock(monkeypatch):
    from types import SimpleNamespace
    now, waits = [0.], []
    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(responses, 'time', SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    return now, waits


def test_default_recovers_after_many_failures_with_exact_ten_second_waits(monkeypatch):
    now, waits = fake_clock(monkeypatch)
    requests, starts, events = [], [], []
    def handler(request):
        requests.append(request.content)
        starts.append(now[0])
        return httpx.Response(503, text='private-test-key') if len(requests) <= 5 else httpx.Response(200, json=completed())
    with client(handler, timeout_s=1.) as model:
        result = model.generate('rules', [{'role': 'user', 'content': 'ready'}], on_event=events.append)
        metadata = model.public_metadata()
    assert starts == [0., 10., 20., 30., 40., 50.]
    assert waits == [10.] * 5 and len(set(requests)) == 1
    assert result.attempts == 6 and result.duration_s == 50.
    assert metadata['max_retries'] is None and metadata['timeout_scope'] == 'per_attempt'
    assert metadata['retry_delay_s'] == 10.
    assert sum(e.get('waited_s', 0.) for e in events) == 50.
    assert 'private-test-key' not in json.dumps(events)
    assert ModelGatewayError('unused', attempts=6).public_metadata()['attempts'] == 6


def test_each_timed_out_attempt_gets_a_fresh_request_timeout(monkeypatch):
    now, waits = fake_clock(monkeypatch)
    timeouts = []
    def handler(request):
        timeouts.append(request.extensions['timeout']['read'])
        if len(timeouts) == 1:
            now[0] += 2.
            raise httpx.ReadTimeout('interrupted', request=request)
        return httpx.Response(200, json=completed())
    with client(handler, timeout_s=2.) as model:
        result = model.generate('rules', [{'role': 'user', 'content': 'ready'}], retry_deadline=40.)
    assert timeouts == [2., 2.] and waits == [10.] and result.attempts == 2


def test_deadline_shortens_final_wait_without_sending_early_retry(monkeypatch):
    now, waits = fake_clock(monkeypatch)
    starts, timeouts = [], []
    def handler(request):
        starts.append(now[0])
        timeouts.append(request.extensions['timeout']['read'])
        return httpx.Response(429)
    with client(handler, timeout_s=30.) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('rules', [{'role': 'user', 'content': 'ready'}], retry_deadline=25.)
    assert starts == [0., 10., 20.] and timeouts == [25., 15., 5.]
    assert waits == [10., 10., 5.]
    assert error.value.category == 'request_budget_exhausted' and error.value.attempts == 3


def test_keyboard_interrupt_stops_wait_without_another_request(monkeypatch):
    now, _ = fake_clock(monkeypatch)
    calls, events = [], []
    def stop(seconds):
        now[0] += 3.
        raise KeyboardInterrupt
    monkeypatch.setattr(responses.time, 'sleep', stop)
    def handler(request):
        calls.append(request)
        return httpx.Response(502)
    with client(handler) as model:
        with pytest.raises(KeyboardInterrupt):
            model.generate('rules', [{'role': 'user', 'content': 'ready'}], on_event=events.append)
    assert len(calls) == 1
    assert events[-1]['event'] == 'model_retry_wait_finished' and events[-1]['waited_s'] == 3.


@pytest.mark.parametrize('status', [400, 401, 403, 404, 422])
def test_nontransient_errors_do_not_wait_forever(monkeypatch, status):
    _, waits = fake_clock(monkeypatch)
    with client(lambda _: httpx.Response(status)) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('rules', [{'role': 'user', 'content': 'ready'}])
    assert waits == [] and error.value.attempts == 1


def test_runner_retries_same_decision_then_submits_only_one_physical_action(monkeypatch, tmp_path):
    from agent.model_client import runner, interaction
    from test_online_model_runner import setup_loop
    now, waits = fake_clock(monkeypatch)
    monkeypatch.setattr(runner, 'time', responses.time)
    monkeypatch.setattr(interaction, 'time', responses.time)
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [])
    args.provider, args.credentials, args.max_decisions = 'micu', tmp_path/'external.json', 1
    monkeypatch.setattr(runner.ModelGatewayConfig, 'from_file', lambda _: CONFIG)
    requests = []
    value = completed()
    value['output'][0]['content'][0]['text'] = json.dumps({'action': 'joint_delta', 'delta_rad': {'arm': .02}, 'duration_s': .4})
    def handler(request):
        requests.append(request.content)
        return httpx.Response(503) if len(requests) <= 2 else httpx.Response(200, json=value)
    model = client(handler, timeout_s=2.)
    monkeypatch.setattr(runner, 'MicuModelClient', lambda *a, **k: model)
    args.model_timeout = 2.
    report = runner.run(args)
    assert waits == [10., 10.] and len(requests) == 3 and len(set(requests)) == 1
    assert report['error_type'] is None and report['model_call_attempts'] == 1
    assert report['model_http_attempts'] == 3 and report['model_retry_wait_s'] == 20.
    assert report['submitted_actions'] == 1 and report['completed_actions'] == 1
    events = [json.loads(line) for line in (args.output/'decisions/0000/model_api_events.jsonl').read_text().splitlines()]
    assert len({e['request_sha256'] for e in events}) == 1
    assert {e['decision_index'] for e in events} == {0}
    assert 'private-test-key' not in json.dumps(events)
    from agent.model_client.suite import model_statistics
    assert model_statistics(report)['gateway_total_tokens'] is None


def test_historical_micu_retry_policy_requires_new_plan():
    from argparse import Namespace
    from agent.model_client.runner import configured_model
    with pytest.raises(ValueError, match='new frozen plan'):
        configured_model(Namespace(provider='micu', credentials='external',
            model_retry_policy='bounded_same_endpoint_model_request_v1'), CONFIG)


@pytest.mark.parametrize('delay_stage', ['host_event', 'payload_preparation'])
def test_deadline_crossed_before_post_records_unsent_attempt(monkeypatch, delay_stage):
    now, waits = fake_clock(monkeypatch)
    events = []
    def event(value):
        events.append(value)
        if delay_stage == 'host_event' and value['event'] == 'model_request_started':
            now[0] = 6.
    if delay_stage == 'payload_preparation':
        original = responses._messages
        def prepare(*args):
            now[0] = 6.
            return original(*args)
        monkeypatch.setattr(responses, '_messages', prepare)
    with client(lambda _: pytest.fail('sent after deadline')) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('rules', [{'role': 'user', 'content': 'ready'}], retry_deadline=5., on_event=event)
    assert waits == [] and error.value.attempts == 0
    assert error.value.category == 'request_budget_exhausted'
    assert 'model_request_not_sent' in [e['event'] for e in events]


@pytest.mark.parametrize('streamed', [False, True])
def test_partial_body_cannot_trickle_past_request_timeout(monkeypatch, streamed):
    now, waits = fake_clock(monkeypatch)
    closed = []
    class SlowBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data:' if streamed else b'{'
            now[0] = 3.
            yield b'no newline yet'
            pytest.fail('continued reading after request timeout')
        def close(self):
            closed.append(True)
    def handler(_):
        return httpx.Response(200, stream=SlowBody(), headers={
            'content-type': 'text/event-stream' if streamed else 'application/json'})
    with client(handler, timeout_s=2., max_retries=0) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('rules', [{'role': 'user', 'content': 'ready'}])
    assert waits == [] and closed == [True]
    assert error.value.category == 'transport_timeout'


@pytest.mark.parametrize('stage', ['after_response', 'bench_result_read', 'before_action'])
def test_recovery_followed_by_wall_deadline_keeps_task_unknown(monkeypatch, tmp_path, stage):
    from agent.model_client import runner, interaction
    from test_online_model_runner import setup_loop
    now, waits = fake_clock(monkeypatch)
    monkeypatch.setattr(runner, 'time', responses.time)
    monkeypatch.setattr(interaction, 'time', responses.time)
    args, bench, _ = setup_loop(monkeypatch, tmp_path, [])
    args.provider, args.credentials, args.max_decisions = 'micu', tmp_path/'external.json', 1
    args.wall_budget, args.model_timeout = 15., 2.
    monkeypatch.setattr(runner.ModelGatewayConfig, 'from_file', lambda _: CONFIG)
    calls = []
    value = completed()
    value['output'][0]['content'][0]['text'] = json.dumps({'action': 'joint_delta', 'delta_rad': {'arm': .02}, 'duration_s': .4})
    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, json=value)
    model = client(handler, timeout_s=2.)
    monkeypatch.setattr(runner, 'MicuModelClient', lambda *a, **k: model)
    original_save = runner.save
    original_observation = bench.get_observation
    def save(path, item):
        original_save(path, item)
        if path.name == 'model_response.json' and stage == 'after_response': now[0] = 16.
    monkeypatch.setattr(runner, 'save', save)
    def get_result(_):
        if now[0] >= 15.:
            return bench.terminal_result('wall_timeout')
        if len(calls) == 2 and stage == 'bench_result_read': now[0] = 16.
        return None
    def get_observation(*args, **kw):
        result = original_observation(*args, **kw)
        if len(calls) == 2 and stage == 'before_action': now[0] = 16.
        return result
    bench.get_result, bench.get_observation = get_result, get_observation
    report = runner.run(args)
    assert waits == [10.] and len(calls) == 2
    assert report['submitted_actions'] == 0
    assert report['run_status'] == 'infra_error' and report['task_success'] is None
    assert report['result']['outcome'] == 'wall_timeout'
