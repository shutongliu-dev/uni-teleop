import json
from argparse import Namespace

import httpx
import pytest

from agent.model_client.qwen import QwenModelClient, QWEN_BASE_URL, QWEN_RETRY_POLICY
from agent.model_client.transport import ModelGatewayConfig, ModelGatewayError, png_data_url


MODEL = 'qwen3.8-max-0902'
SECRET = 'qwen-private-test-credential'
CONFIG = ModelGatewayConfig(QWEN_BASE_URL, SECRET)
TURNS = [{'role': 'user', 'content': 'ready'}]


def completed(**changes):
    return {'model': MODEL, 'status': 'completed', 'output': [
        {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': 'private reasoning ' + SECRET}]},
        {'type': 'message', 'role': 'assistant', 'content': [
            {'type': 'output_text', 'text': '{"action":"stop"}'}]}],
        'usage': {'input_tokens': 10, 'output_tokens': 25,
                  'output_tokens_details': {'reasoning_tokens': 20}}, **changes}


@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('effort,effective', [('none', 'none'), ('minimal', 'low'),
    ('low', 'low'), ('medium', 'medium'), ('high', 'xhigh'), ('xhigh', 'xhigh'), ('max', 'xhigh')])
def test_official_effort_mapping_and_safe_json_sse_outputs(streamed, effort, effective):
    requests = []
    def handler(request):
        requests.append(request)
        assert str(request.url) == QWEN_BASE_URL + '/responses'
        assert request.headers['authorization'] == 'Bearer ' + SECRET
        body = json.loads(request.content)
        assert body['model'] == MODEL
        assert body['reasoning'] == {'effort': effective}
        assert body['store'] is False and body['stream'] is True
        assert all(key not in body for key in ('thinking_budget', 'enable_thinking', 'max_output_tokens',
                                              'previous_response_id', 'conversation', 'tools'))
        if streamed:
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                content='data: ' + json.dumps({'type': 'response.completed', 'response': completed()}))
        return httpx.Response(200, json=completed())
    with QwenModelClient(CONFIG, MODEL, reasoning_effort=effort,
            transport=httpx.MockTransport(handler)) as model:
        result = model.generate('public rules', TURNS, max_output_tokens=None)
        metadata = model.public_metadata()
    assert len(requests) == 1 and result.attempts == 1
    assert result.text == '{"action":"stop"}'
    assert result.usage['output_tokens_details'] == {'reasoning_tokens': 20}
    assert result.reasoning_effort == effort
    for value in (metadata, result.public_metadata()):
        assert value['provider'] == 'qwen'
        assert value['requested_reasoning_effort'] == effort
        assert value['effective_reasoning_effort'] == effective
        assert value['request_parameters']['reasoning'] == {'effort': effective}
        assert value['reasoning_effort_verified_by_server'] is False
        assert value['reasoning_summary_recorded'] is False
        assert SECRET not in json.dumps(value) and 'private reasoning' not in json.dumps(value)


def test_multimodal_history_preserves_roles_and_inline_frames():
    image = png_data_url(b'\x89PNG\r\n\x1a\nfixture')
    turns = [{'role': 'user', 'content': 'previous state'},
             {'role': 'assistant', 'content': '{"action":"hold"}'},
             {'role': 'user', 'content': [{'type': 'text', 'text': 'new state'},
                 {'type': 'image_url', 'image_url': {'url': image, 'detail': 'high'}}]}]
    def handler(request):
        body = json.loads(request.content)
        assert body['instructions'] == 'public rules'
        assert body['input'] == [
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'previous state'}]},
            {'role': 'assistant', 'content': [{'type': 'output_text', 'text': '{"action":"hold"}'}]},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'new state'},
                {'type': 'input_image', 'image_url': image, 'detail': 'high'}]}]
        assert body['max_output_tokens'] == 32768
        return httpx.Response(200, json=completed())
    with QwenModelClient(CONFIG, MODEL, transport=httpx.MockTransport(handler)) as model:
        result = model.generate('public rules', turns, max_output_tokens=32768)
    assert result.public_metadata()['request_parameters']['max_output_tokens'] == 32768
    assert result.public_metadata()['output_token_scope'] == 'reasoning_and_final_answer'
    assert turns[2]['content'][1]['type'] == 'image_url'


@pytest.mark.parametrize('status', [302, 401, 429, 503])
def test_no_retry_redirect_or_provider_fallback_and_no_secret_errors(status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={'location': 'https://other.invalid'}, text=SECRET)
    with QwenModelClient(CONFIG, MODEL, transport=httpx.MockTransport(handler)) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('public rules', TURNS)
    assert len(calls) == 1 and error.value.attempts == 1
    assert error.value.status_code == status
    assert SECRET not in str(error.value) + json.dumps(error.value.public_metadata())


@pytest.mark.parametrize('streamed', [False, True])
@pytest.mark.parametrize('changes,category', [
    ({'model': 'qwen3.8-max'}, 'model_mismatch'),
    ({'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}}, 'output_budget_exhausted'),
    ({'output': [{'type': 'reasoning', 'summary': [{'text': 'no final answer'}]}]}, 'empty_response')])
def test_incomplete_different_model_and_reasoning_only_are_not_actions(streamed, changes, category):
    def handler(request):
        value = completed(**changes)
        if streamed:
            kind = 'response.incomplete' if value['status'] == 'incomplete' else 'response.completed'
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                content='data: ' + json.dumps({'type': kind, 'response': value}) + '\n\n')
        return httpx.Response(200, json=value)
    with QwenModelClient(CONFIG, MODEL, transport=httpx.MockTransport(handler)) as model:
        with pytest.raises(ModelGatewayError) as error:
            model.generate('public rules', TURNS)
    assert error.value.category == category


def test_echoed_credentials_removed_from_output_and_numeric_usage_keys():
    value = completed(output=[{'type': 'message', 'content': [{'type': 'output_text', 'text': SECRET}]}],
                      usage={SECRET: 1, 'nested': {SECRET: 2, 'safe': 3}})
    with QwenModelClient(CONFIG, MODEL,
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=value))) as model:
        result = model.generate('public rules', TURNS)
    assert result.text == '[redacted]' and result.usage == {'nested': {'safe': 3}}
    assert SECRET not in repr(result) + json.dumps(result.public_metadata())


@pytest.mark.parametrize('kwargs', [{'reasoning_effort': 'ultra'}, {'reasoning_effort': True},
    {'max_retries': 1}, {'max_retries': None}, {'max_retries': False}, {'timeout_s': 0}])
def test_invalid_settings_rejected_before_request(kwargs):
    with pytest.raises(ValueError):
        QwenModelClient(CONFIG, MODEL, **kwargs)


@pytest.mark.parametrize('cap', [True, 0, 15, 32769, 1.5])
def test_invalid_output_bound_rejected_before_request(cap):
    with QwenModelClient(CONFIG, MODEL,
            transport=httpx.MockTransport(lambda _: pytest.fail('request must not be sent'))) as model:
        with pytest.raises(ValueError):
            model.generate('public rules', TURNS, max_output_tokens=cap)


def test_client_must_not_use_other_endpoint_or_model():
    with pytest.raises(ValueError, match='endpoint'):
        QwenModelClient(ModelGatewayConfig('https://example.invalid/v1', SECRET), MODEL)
    with pytest.raises(ValueError, match='model'):
        QwenModelClient(CONFIG, 'gpt-6-astra')


def test_factory_explicit_binding_retry_contract_and_credential_boundary():
    from agent.model_client.runner import configured_model
    args = Namespace(provider='qwen', credentials='external-profile.json', model=MODEL,
                     reasoning_effort='xhigh', model_timeout=300., model_retries=None)
    with configured_model(args, CONFIG) as model:
        assert isinstance(model, QwenModelClient)
        assert model.public_metadata()['retry_policy'] == QWEN_RETRY_POLICY
    args.model_retry_policy = 'legacy_gateway_v1'
    with pytest.raises(ValueError, match='Historical retry policy'):
        configured_model(args, CONFIG)
    args.model_retry_policy = QWEN_RETRY_POLICY
    args.credentials = None
    with pytest.raises(ValueError, match='explicit external'):
        configured_model(args, CONFIG)
    args.provider = 'gateway'
    args.model_retry_policy = None
    with pytest.raises(ValueError, match='explicit --provider qwen'):
        configured_model(args, CONFIG)


def test_unknown_environment_proxies_are_not_used(monkeypatch):
    monkeypatch.setenv('ALL_PROXY', 'unsupported-proxy://127.0.0.1:1')
    with QwenModelClient(CONFIG, MODEL):
        pass  # Construction is offline and never reads any credential file.
