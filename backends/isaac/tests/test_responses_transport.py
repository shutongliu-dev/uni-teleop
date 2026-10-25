import json

import httpx
import pytest

from agent.model_client.responses import ResponsesClient
from agent.model_client.transport import ModelGatewayConfig, ModelGatewayError


def test_explicit_provider_ignores_unrelated_environment_proxy(monkeypatch):
    monkeypatch.setenv('ALL_PROXY', 'unsupported-proxy://127.0.0.1:1')
    monkeypatch.setenv('HTTPS_PROXY', 'unsupported-proxy://127.0.0.1:1')
    client = ResponsesClient(ModelGatewayConfig('https://example.invalid/v1', 'private-test'))
    client.close()  # Constructor makes no network request or proxy dependency load.


def response(**changes):
    return {'model': 'gpt-6-astra', 'status': 'completed',
            'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"q":3,"r":3}'}]}],
            'usage': {'input_tokens': 11, 'output_tokens': 9}, **changes}


def complete(handler):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = ResponsesClient(ModelGatewayConfig('https://example.invalid/v1', 'sk-private-test'), client=http)
    return client.complete(model='gpt-6-astra', reasoning_effort='max', instructions='Hex rules', prompt='A move')


def test_stream_preserves_requested_protocol_effort_and_no_storage():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == '/v1/responses'
        assert body['reasoning'] == {'effort': 'max'} and body['store'] is False
        assert body['model'] == 'gpt-6-astra' and body['stream'] is True
        data = json.dumps({'type': 'response.completed', 'response': response()})
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content='data: '+data+'\n\n')
    result = complete(handler)
    assert json.loads(result.text) == {'q': 3, 'r': 3}
    assert result.public_metadata()['reasoning_effort_verified_by_server'] is False


@pytest.mark.parametrize('value,category', [
    (response(model='other'), 'model_mismatch'),
    (response(status='incomplete'), 'invalid_finish_reason'),
    (response(output=[]), 'empty_response'),
])
def test_rejects_mismatched_or_incomplete_completion(value, category):
    with pytest.raises(ModelGatewayError) as error:
        complete(lambda request: httpx.Response(200, json=value))
    assert error.value.category == category


def test_provider_error_body_and_secret_are_not_logged_or_retried():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(403, text='Bearer sk-private-test arbitrary provider data')
    with pytest.raises(ModelGatewayError) as error:
        complete(handler)
    assert len(calls) == 1 and error.value.status_code == 403
    assert 'private-test' not in str(error.value) and 'arbitrary' not in str(error.value)


def test_truncated_stream_cannot_masquerade_as_complete_text():
    event = {'type': 'response.output_text.delta', 'delta': '{"q":3,"r":3}'}
    with pytest.raises(ModelGatewayError) as error:
        complete(lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                                content='data: '+json.dumps(event)+'\n\n'))
    assert error.value.category == 'invalid_response'


@pytest.mark.parametrize('value', [[], None, 4, 'provider secret', {}, {'type': []}])
@pytest.mark.parametrize('suffix', ['\n\n', ''])
def test_malformed_sse_event_is_sanitized_in_delimited_and_eof_paths(value, suffix):
    with pytest.raises(ModelGatewayError) as error:
        complete(lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'},
            content='data: '+json.dumps(value)+suffix))
    assert error.value.category == 'invalid_response'
    assert error.value.public_metadata()['attempts'] == 1
    assert 'provider secret' not in str(error.value)


@pytest.mark.parametrize('late_type', ['response.failed', 'error', 'response.incomplete'])
@pytest.mark.parametrize('suffix', ['\n\n', ''])
@pytest.mark.parametrize('include_done', [False, True])
def test_late_failure_overrides_completed_event_even_at_eof(late_type, suffix, include_done):
    completed = 'data: '+json.dumps({'type': 'response.completed', 'response': response()})+'\n\n'
    done = 'data: [DONE]\n\n' if include_done else ''
    failure = 'data: '+json.dumps({'type': late_type, 'message': 'Bearer sk-private-test'})+suffix
    with pytest.raises(ModelGatewayError) as error:
        complete(lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'},
            content=completed+done+failure))
    assert error.value.category == ('invalid_finish_reason' if late_type == 'response.incomplete' else 'invalid_response')
    assert 'private-test' not in str(error.value)


@pytest.mark.parametrize('output', [None, {}, 'secret provider output', [None], [[]], [{}],
    [{'type': 'message', 'content': None}], [{'type': 'message', 'content': {}}],
    [{'type': 'message', 'content': [None]}], [{'type': 'message', 'content': [[]]}],
    [{'type': 'message', 'content': [{'type': 'output_text', 'text': None}]}]])
@pytest.mark.parametrize('streamed', [False, True])
def test_malformed_output_never_raises_raw_attribute_or_type_error(output, streamed):
    value = response(output=output)
    def handler(request):
        if streamed:
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  content='data: '+json.dumps({'type': 'response.completed', 'response': value}))
        return httpx.Response(200, json=value)
    with pytest.raises(ModelGatewayError) as error:
        complete(handler)
    assert error.value.category == 'invalid_response'
    assert 'secret provider' not in str(error.value)


def test_completed_eof_event_and_done_sentinel_are_supported():
    event = 'data: '+json.dumps({'type': 'response.completed', 'response': response()})
    for text in (event, event+'\n\ndata: [DONE]'):
        value = complete(lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=text))
        assert json.loads(value.text) == {'q': 3, 'r': 3}
