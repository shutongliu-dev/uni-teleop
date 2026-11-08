"""Explicit gateway recovery: fake HTTP/clock only, no provider requests."""
from datetime import datetime, timezone
from email.utils import format_datetime
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.model_client import transport
from agent.model_client.runner import configured_model
from test_model_transport import KEY, completion, config


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=0., waits=[])
    def sleep(seconds):
        value.waits.append(seconds)
        value.now += seconds
    monkeypatch.setattr(transport, 'time', SimpleNamespace(
        monotonic=lambda: value.now, time=lambda: 1800000000.+value.now, sleep=sleep))
    monkeypatch.setattr(transport.random, 'uniform', lambda a, b: (a+b)/2.)
    return value


def recovering(handler, retries=None):
    return transport.ModelClient(config(), 'gpt-6-astra', timeout_s=None,
        max_retries=retries, retry_policy=transport.GATEWAY_RETRY_POLICY,
        transport=httpx.MockTransport(handler))


def generate(client, events, **kwargs):
    return client.generate('One decision.', [{'role': 'user', 'content': 'original'}],
        max_output_tokens=None, on_event=events.append, **kwargs)


@pytest.mark.parametrize('failure', [429, 502, 503, 504, httpx.ConnectError, httpx.ConnectTimeout])
def test_transient_attempts_preserve_payload_and_record_each_request(clock, failure):
    requests, events = [], []
    def handler(request):
        requests.append(request.content)
        if len(requests) == 1:
            if isinstance(failure, int):
                return httpx.Response(failure, text='Bearer '+KEY+' private-body')
            raise failure('Bearer '+KEY, request=request)
        return httpx.Response(200, json=completion())
    with recovering(handler) as client:
        value = generate(client, events)
        assert client.public_metadata().items() >= transport.gateway_retry_metadata().items()
    assert value.attempts == 2 and requests[0] == requests[1]
    payload = json.loads(requests[0])
    assert payload['model'] == 'gpt-6-astra' and payload['reasoning_effort'] == 'max'
    assert 'max_completion_tokens' not in payload
    assert value.public_metadata().items() >= transport.gateway_retry_metadata().items()
    assert [e['event'] for e in events] == ['model_request_started', 'model_request_failed',
        'model_retry_wait', 'model_retry_wait_finished', 'model_request_started', 'model_request_completed']
    assert [e['attempts'] for e in events if e['event'] == 'model_request_started'] == [1, 2]
    assert clock.waits == [3.75]
    assert KEY not in json.dumps(events) and 'private-body' not in json.dumps(events)


def test_unlimited_policy_exceeds_legacy_retry_cap_with_bounded_backoff(clock):
    requests, events = [], []
    def handler(request):
        requests.append(request.content)
        return httpx.Response(429) if len(requests) <= 8 else httpx.Response(200, json=completion())
    with recovering(handler) as client:
        assert generate(client, events).attempts == 9
    assert len(set(requests)) == 1
    assert clock.waits == [3.75, 7.5, 15., 30., 45., 45., 45., 45.]


@pytest.mark.parametrize('header,expected', [
    ('20', 20.), (format_datetime(datetime.fromtimestamp(1800000030, timezone.utc)), 30.),
    ('-1', 3.75), ('bad '+KEY, 3.75), ('0', 3.75)])
def test_retry_after_is_a_lower_bound_without_archiving_headers(clock, header, expected):
    requests, events = [], []
    def handler(request):
        requests.append(request.content)
        return (httpx.Response(429, headers={'Retry-After': header}) if len(requests) == 1
                else httpx.Response(200, json=completion()))
    with recovering(handler) as client:
        generate(client, events)
    assert clock.waits == [expected]
    assert KEY not in json.dumps(events)


def test_unsupported_long_retry_after_stops_without_retrying_early(clock):
    events = []
    with recovering(lambda _: httpx.Response(429, headers={'Retry-After': '86401'})) as client:
        with pytest.raises(transport.ModelGatewayError) as error:
            generate(client, events)
    assert error.value.attempts == 1 and not clock.waits
    assert events[-1]['retry_after_source'] == 'out_of_range'


@pytest.mark.parametrize('status', [400, 401, 403, 404, 408, 500])
def test_unlisted_http_errors_never_retry_or_fallback(clock, status):
    events, requests = [], []
    def handler(request):
        requests.append(request.content)
        return httpx.Response(status, text=KEY)
    with recovering(handler) as client:
        with pytest.raises(transport.ModelGatewayError) as error:
            generate(client, events)
    assert len(requests) == error.value.attempts == 1 and not clock.waits
    assert error.value.public_metadata().items() >= transport.gateway_retry_metadata().items()
    assert KEY not in str(error.value) and KEY not in json.dumps(events)


def test_explicit_finite_limit_retains_actual_attempts(clock):
    events = []
    with recovering(lambda _: httpx.Response(503), retries=2) as client:
        with pytest.raises(transport.ModelGatewayError) as error:
            generate(client, events)
    assert error.value.attempts == 3 and len(clock.waits) == 2
    assert error.value.public_metadata().items() >= transport.gateway_retry_metadata(2).items()


@pytest.mark.parametrize('fault', ['read_timeout', 'bad_json', 'wrong_model'])
def test_uncertain_or_invalid_completions_do_not_retry(clock, fault):
    events = []
    def handler(request):
        if fault == 'read_timeout':
            raise httpx.ReadTimeout(KEY, request=request)
        if fault == 'bad_json':
            return httpx.Response(200, text=KEY)
        return httpx.Response(200, json=completion(model='different-model'))
    with recovering(handler) as client:
        with pytest.raises(transport.ModelGatewayError) as error:
            generate(client, events)
    assert error.value.attempts == 1 and not clock.waits
    assert events[-1]['event'] == 'model_request_failed'
    assert error.value.public_metadata()['retry_policy'] == transport.GATEWAY_RETRY_POLICY


@pytest.mark.parametrize('where', ['before_start', 'after_started_event', 'retry_wait', 'response'])
def test_absolute_deadline_never_sends_an_extra_request(clock, where):
    events, requests = [], []
    def handler(request):
        requests.append(request.content)
        if where == 'response':
            clock.now = 11.
            return httpx.Response(200, json=completion())
        return httpx.Response(429, headers={'Retry-After': '60'})
    def record(event):
        events.append(event)
        if where == 'after_started_event':
            clock.now = 11.
    with recovering(handler) as client:
        with pytest.raises(transport.ModelGatewayError) as error:
            client.generate('one', [{'role': 'user', 'content': 'one'}],
                on_event=record, retry_deadline=0. if where == 'before_start' else 10.)
    assert error.value.category == 'request_budget_exhausted'
    assert error.value.attempts == len(requests) == (1 if where in ('retry_wait', 'response') else 0)
    assert sum(e['event'] == 'model_request_started' for e in events)-sum(
        e['event'] == 'model_request_not_sent' for e in events) == len(requests)
    if where == 'retry_wait':
        assert clock.waits == [10.]


def test_retry_observer_cannot_change_original_model_input(clock):
    turns, requests = [{'role': 'user', 'content': 'original'}], []
    def handler(request):
        requests.append(request.content)
        return httpx.Response(429) if len(requests) == 1 else httpx.Response(200, json=completion())
    def observer(event):
        turns[0]['content'] = 'mutated'
    with recovering(handler) as client:
        client.generate('same', turns, on_event=observer)
    assert requests[0] == requests[1] and b'original' in requests[0] and b'mutated' not in requests[0]


def test_explicit_recovery_is_required_to_change_default_configuration():
    args = SimpleNamespace(provider='gateway', model='gpt-6-astra', model_timeout=None,
                           reasoning_effort='max', model_retries=None, model_retry_policy=None)
    with configured_model(args, config()) as old:
        assert old.max_retries == 0 and 'retry_policy' not in old.public_metadata()
    args.model_retry_policy = transport.GATEWAY_RETRY_POLICY
    with configured_model(args, config()) as new:
        assert new.max_retries is None
        assert new.public_metadata().items() >= transport.gateway_retry_metadata().items()
