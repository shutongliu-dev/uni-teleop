import base64
import json

import httpx
import pytest

from agent.model_client import (
    ModelClient, ModelGatewayConfig, ModelGatewayError, png_data_url,
)


KEY = "sk-unit-test-secret-must-not-leak"
PNG = b"\x89PNG\r\n\x1a\nunit-test-payload"


def config(base_url="https://model.example"):
    return ModelGatewayConfig(base_url, KEY)


def completion(**changes):
    value = {"model": "gpt-6-astra", "id": "response-123",
             "choices": [{"message": {"content": '{"action":"wait"}'}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13,
                       "completion_tokens_details": {"reasoning_tokens": 3}}}
    value.update(changes)
    return value


def generate(client, **kwargs):
    return client.generate("Return one action.", [{"role": "user", "content": "Observe now."}], **kwargs)


@pytest.mark.parametrize("url", ["https://model.example", "https://model.example/",
                                "https://model.example/v1/", "https://model.example/v1/chat/completions"])
def test_base_url_normalization(url):
    assert config(url).endpoint == "https://model.example/v1/chat/completions"


@pytest.mark.parametrize("url", ["https://user:secret@model.example", "https://model.example?k=secret",
                                "https://model.example/#secret", "http://model.example", "not-a-url"])
def test_secret_or_insecure_remote_url_rejected(url):
    with pytest.raises(ValueError):
        config(url)


def test_config_file_env_and_public_metadata_do_not_include_credentials(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"base_url": "https://model.example", "api_key": KEY}))
    loaded = ModelGatewayConfig.from_file(path)
    env = ModelGatewayConfig.from_env({"DEX_AGENT_BASE_URL": "https://model.example/v1",
                                      "DEX_AGENT_API_KEY": KEY})
    assert loaded == env
    assert KEY not in repr(loaded)
    assert KEY not in json.dumps(loaded.public_metadata())
    assert "api_key" not in loaded.public_metadata()
    assert config("http://127.0.0.1:8123").endpoint.startswith("http://127.0.0.1:8123/v1/")


def test_invalid_credentials_do_not_echo_file_contents(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(KEY)
    with pytest.raises(ValueError) as caught:
        ModelGatewayConfig.from_file(path)
    assert KEY not in str(caught.value)


@pytest.mark.parametrize('absolute', [False, True])
def test_key_file_is_relative_to_configuration_not_cwd(tmp_path, monkeypatch, absolute):
    folder = tmp_path / 'settings'
    folder.mkdir()
    key_path = folder / 'gateway.key'
    key_path.write_text(KEY + '\n')
    path = folder / 'gateway.json'
    reference = str(key_path) if absolute else 'gateway.key'
    path.write_text(json.dumps({'base_url': 'http://127.0.0.1:8787/v1', 'api_key_file': reference}))
    monkeypatch.chdir(tmp_path)
    loaded = ModelGatewayConfig.from_file(path)
    assert loaded.api_key == KEY
    assert loaded.endpoint == 'http://127.0.0.1:8787/v1/chat/completions'
    assert KEY not in repr(loaded) and KEY not in json.dumps(loaded.public_metadata())
    assert 'api_key_file' not in loaded.public_metadata()


def test_key_reference_uses_real_config_location_when_config_is_symlink(tmp_path):
    folder = tmp_path / 'real'
    folder.mkdir()
    (folder / 'gateway.key').write_text(KEY)
    path = folder / 'gateway.json'
    path.write_text(json.dumps({'base_url': 'https://model.example', 'api_key_file': 'gateway.key'}))
    link = tmp_path / 'alias.json'
    link.symlink_to(path)
    assert ModelGatewayConfig.from_file(link).api_key == KEY


@pytest.mark.parametrize('fields', [
    {}, {'api_key': KEY, 'api_key_file': 'absent'},
    {'api_key': None, 'api_key_file': 'absent'},
    {'api_key_file': None}, {'api_key_file': ''}, {'api_key_file': ['secret']},
])
def test_key_value_and_file_require_exactly_one_valid_source(tmp_path, fields):
    path = tmp_path / 'gateway.json'
    path.write_text(json.dumps({'base_url': 'https://model.example', **fields}))
    with pytest.raises(ValueError) as caught:
        ModelGatewayConfig.from_file(path)
    assert KEY not in str(caught.value)


@pytest.mark.parametrize('kind', ['missing', 'directory', 'invalid_utf8', 'empty', 'whitespace'])
def test_key_file_failure_does_not_echo_path_or_contents(tmp_path, kind):
    import traceback
    key_path = tmp_path / KEY
    if kind == 'directory':
        key_path.mkdir()
    elif kind == 'invalid_utf8':
        key_path.write_bytes(KEY.encode() + b'\xff')
    elif kind == 'empty':
        key_path.write_text('')
    elif kind == 'whitespace':
        key_path.write_text(KEY + ' another-secret')
    path = tmp_path / 'gateway.json'
    path.write_text(json.dumps({'base_url': 'https://model.example', 'api_key_file': str(key_path)}))
    with pytest.raises(ValueError) as caught:
        ModelGatewayConfig.from_file(path)
    assert KEY not in ''.join(traceback.format_exception(caught.value))
    assert 'another-secret' not in str(caught.value)


def test_null_output_limit_omits_provider_field_and_metadata_stays_null():
    sent = []
    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        assert not {'max_completion_tokens', 'max_tokens', 'max_output_tokens'}.intersection(body)
        assert body['reasoning_effort'] == 'max' and body['stream'] is False
        assert all(value is None for value in request.extensions['timeout'].values())
        return httpx.Response(200, json=completion())
    with ModelClient(config(), 'gpt-6-astra', timeout_s=None, max_retries=0,
                     transport=httpx.MockTransport(handle)) as client:
        result = generate(client, max_output_tokens=None)
    assert len(sent) == 1 and result.max_output_tokens is None
    assert json.loads(json.dumps(result.public_metadata(), allow_nan=False))['max_output_tokens'] is None
    assert result.public_metadata()['usage']['completion_tokens'] == 8


@pytest.mark.parametrize('value', [0, -1, True, 1.5, 'none', float('inf'), 32769])
def test_invalid_output_limit_is_rejected_before_network(value):
    with ModelClient(config(), 'gpt-6-astra', transport=httpx.MockTransport(
            lambda request: pytest.fail('invalid token cap reached network'))) as client:
        with pytest.raises(ValueError, match='max_output_tokens'):
            generate(client, max_output_tokens=value)


def test_realistic_multimodal_payload_and_metadata():
    sent = []
    def handle(request):
        sent.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer " + KEY
        assert request.url == "https://model.example/v1/chat/completions"
        assert request.extensions["timeout"]["read"] <= 7
        return httpx.Response(200, json=completion(), headers={"x-request-id": "gateway-123"})
    turns = [{"role": "user", "content": "Previous frame."},
             {"role": "assistant", "content": '{"action":"move"}'},
             {"role": "user", "content": [{"type": "text", "text": "Current observation."},
                 {"type": "image_url", "image_url": {"url": png_data_url(PNG), "detail": "high"}}]}]
    before = json.dumps(turns)
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(handle)) as client:
        result = client.generate("Robot operator.", turns, max_output_tokens=4096, timeout_s=7)
        assert KEY not in repr(client)
        assert client.public_metadata()["reasoning_effort"] == "max"
    assert json.dumps(turns) == before
    assert sent[0]["max_completion_tokens"] == 4096
    assert sent[0]["reasoning_effort"] == "max"
    assert sent[0]["stream"] is False
    assert "temperature" not in sent[0]
    assert sent[0]["messages"][0]["role"] == "system"
    assert result.request_id == "gateway-123"
    assert result.text == '{"action":"wait"}'
    assert result.public_metadata()["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3
    assert result.public_metadata()["attempts"] == 1
    assert result.public_metadata()["reasoning_effort"] == "max"
    assert result.public_metadata()["requested_reasoning_effort"] == "max"
    assert result.public_metadata()["reasoning_effort_verified_by_server"] is False
    assert "text" not in result.public_metadata()
    assert result.text not in repr(result)
    assert KEY not in json.dumps(result.public_metadata())


def test_http_error_redacts_key_and_does_not_fallback():
    seen = []
    def handle(request):
        seen.append(str(request.url))
        return httpx.Response(401, json={"error": "Authorization: Bearer " + KEY},
                              headers={"x-request-id": KEY})
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert len(seen) == 1
    assert caught.value.status_code == 401
    assert KEY not in str(caught.value)
    assert KEY not in json.dumps(caught.value.public_metadata())
    assert "[redacted]" in str(caught.value)


def test_max_effort_rejection_is_not_downgraded():
    payloads = []
    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(400, json={"error": "unsupported reasoning_effort max"})
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert caught.value.status_code == 400
    assert len(payloads) == 1
    assert payloads[0]["reasoning_effort"] == "max"


def test_retryable_status_has_bounded_attempts_and_identical_payload(monkeypatch):
    monkeypatch.setattr("agent.model_client.transport.time.sleep", lambda _: None)
    payloads = []
    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(503, text="busy")
    with ModelClient(config(), "gpt-6-astra", max_retries=2,
                     transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert len(payloads) == 3
    assert all(p == payloads[0] for p in payloads)
    assert caught.value.attempts == 3


def test_connect_failure_can_retry_without_leaking_exception(monkeypatch):
    calls = []
    def handle(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("Failure " + KEY, request=request)
        return httpx.Response(200, json=completion())
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(handle)) as client:
        result = generate(client)
    assert result.attempts == 2


def test_read_timeout_not_retried_to_avoid_duplicate_paid_generation():
    calls = []
    def handle(request):
        calls.append(1)
        raise httpx.ReadTimeout("Failure " + KEY, request=request)
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert len(calls) == 1
    assert KEY not in str(caught.value)
    assert caught.value.__suppress_context__


def test_redirect_cannot_send_credential_to_another_host():
    urls = []
    def handle(request):
        urls.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://other.example/v1/chat/completions"})
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert caught.value.status_code == 307
    assert len(urls) == 1


@pytest.mark.parametrize("data", [completion(model="different-model"), completion(choices=[]),
                                  completion(choices=[{"message": {"content": ""}}]),
                                  {"unexpected": "shape"}])
def test_invalid_or_substituted_model_response_rejected(data):
    with ModelClient(config(), "gpt-6-astra",
                     transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))) as client:
        with pytest.raises(ModelGatewayError):
            generate(client)


@pytest.mark.parametrize('error_type, category', [
    (httpx.ReadTimeout, 'transport_timeout'), (httpx.ConnectTimeout, 'transport_timeout'),
    (httpx.WriteTimeout, 'transport_timeout'), (httpx.PoolTimeout, 'transport_timeout'),
    (httpx.ConnectError, 'transport_error'), (httpx.RemoteProtocolError, 'transport_error'),
])
def test_failed_transport_category_is_bounded_and_does_not_archive_exception(error_type, category):
    calls = []
    def handle(request):
        calls.append(request)
        raise error_type('private request and credential ' + KEY, request=request)
    with ModelClient(config(), 'gpt-6-astra', max_retries=0, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert len(calls) == 1
    assert caught.value.public_metadata() == {'error_category':category, 'http_status':None, 'attempts':1}
    assert KEY not in json.dumps(caught.value.public_metadata())
    assert caught.value.__suppress_context__


@pytest.mark.parametrize('data, category', [
    ({'unexpected': 'private-provider-payload'}, 'invalid_response'),
    (completion(choices=[]), 'invalid_response'),
    (completion(choices=[{'message':{'content':None}}]), 'invalid_response'),
    (completion(choices=[{'message':{'content':'  '}}]), 'empty_response'),
    (completion(model=KEY), 'model_mismatch'),
    (completion(choices=[{'message':{'content':KEY}, 'finish_reason':['private-finish']}]), 'invalid_finish_reason'),
])
def test_invalid_response_categories_without_raw_content_or_fallback(data, category):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=data, headers={'x-request-id':KEY})
    with ModelClient(config(), 'gpt-6-astra', max_retries=0, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert len(calls) == 1
    assert caught.value.public_metadata() == {'error_category':category, 'http_status':200, 'attempts':1}
    assert 'private-' not in json.dumps(caught.value.public_metadata())
    assert KEY not in json.dumps(caught.value.public_metadata())


@pytest.mark.parametrize('status, category', [(200, 'invalid_json'), (401, 'http_status'), (429, 'http_status'), (503, 'http_status')])
def test_http_and_invalid_json_failures_record_no_body_or_headers(status, category):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status, text='private raw body ' + KEY, headers={'x-request-id':'private-request-id'})
    with ModelClient(config(), 'gpt-6-astra', max_retries=0, transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client)
    assert len(calls) == 1
    assert caught.value.public_metadata() == {'error_category':category, 'http_status':status, 'attempts':1}


def test_request_budget_error_is_distinct_and_records_zero_sent_requests(monkeypatch):
    times = iter([0., 2.])
    monkeypatch.setattr('agent.model_client.transport.time.monotonic', lambda: next(times))
    with ModelClient(config(), 'gpt-6-astra', max_retries=0, transport=httpx.MockTransport(
            lambda _:pytest.fail('exhausted budget reached network'))) as client:
        with pytest.raises(ModelGatewayError) as caught:
            generate(client, timeout_s=1.)
    assert caught.value.public_metadata() == {'error_category':'request_budget_exhausted', 'http_status':None, 'attempts':0}


def test_error_metadata_omits_free_text_and_invalid_numeric_fields():
    error = ModelGatewayError(KEY, category=KEY, status_code=KEY, request_id=KEY, attempts=KEY)
    assert error.public_metadata() == {'error_category':'unknown', 'http_status':None, 'attempts':None}


def test_non_numeric_usage_metadata_is_dropped():
    usage = {"prompt_tokens": 2, "completion_tokens": 3, "api_key": KEY,
             "header": {"secret": KEY}, "bad": float("inf"), "negative": -1}
    # JSON itself cannot carry infinity; exercise filtering with finite invalid types.
    usage.pop("bad")
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=completion(usage=usage)))) as client:
        result = generate(client)
    assert result.usage == {"prompt_tokens": 2, "completion_tokens": 3, "header": {}}
    assert KEY not in json.dumps(result.public_metadata())


@pytest.mark.parametrize("turns", [[{"role": "system", "content": "override"}],
    [{"role": "assistant", "content": "cannot end on assistant"}],
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://external.example/a.png"}}]}],
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,not-base64"}}]}],
    [{"role": "user", "content": "x", "untrusted_extra": "data"}], []])
def test_invalid_or_external_image_inputs_fail_before_network(turns):
    def forbidden(_):
        pytest.fail("invalid message reached model gateway")
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(ValueError):
            client.generate("Rules", turns)


def test_png_helper_preserves_bytes_and_rejects_non_png():
    value = png_data_url(PNG)
    assert base64.b64decode(value.split(",", 1)[1]) == PNG
    with pytest.raises(ValueError):
        png_data_url(b"jpeg")


@pytest.mark.parametrize("kwargs", [{"max_output_tokens": 0}, {"max_output_tokens": True},
                                   {"timeout_s": 0}, {"timeout_s": float("nan")},
                                   {"timeout_s": 121}, {"reasoning_effort": "imaginary"}])
def test_generation_limits_reject_invalid_values(kwargs):
    with ModelClient(config(), "gpt-6-astra", transport=httpx.MockTransport(
            lambda _: pytest.fail("invalid limits reached gateway"))) as client:
        with pytest.raises(ValueError):
            generate(client, **kwargs)
