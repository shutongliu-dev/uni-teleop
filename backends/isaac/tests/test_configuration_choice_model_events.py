"""Pre-episode model retries must remain observable before a worker exists."""
import json
from types import SimpleNamespace
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, canonical_hash, run
from agent.model_client.responses import MicuModelClient
from agent.model_client.transport import ModelGatewayConfig
from bench.paths import load_json
from bench.runtime.server import create_app
from test_robot_configuration_selection import TASK, configured_service


@pytest.mark.parametrize('recover', [True, False])
def test_configuration_choice_records_live_retry_and_failure_events(
        tmp_path, monkeypatch, capsys, configured_service, recover):
    api, workers, _ = configured_service
    output = tmp_path / 'run'
    event_path = output / 'configuration_choice_model_api_events.jsonl'
    requests = []
    responses = iter([
        {'configuration_id': 'single_arm', 'reason': '使用已发布的原右手安装'}, {'stop': True}])
    secret = 'private-configuration-test-key'
    monkeypatch.setattr('agent.model_client.responses.time',
                        SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None))

    def handler(request):
        requests.append(request.content)
        if len(requests) == 1:
            # The record must be available while the model call is in flight,
            # not reconstructed only after the choice succeeds or fails.
            event = json.loads(event_path.read_text().splitlines()[0])
            assert event['event'] == 'model_request_started' and event['attempts'] == 1
            assert event['phase'] == 'configuration_choice' and not workers
            receipt = load_json(output / 'configuration_choice.json')
            assert receipt['status'] == 'requesting'
            assert event['request_sha256'] == canonical_hash(receipt['messages'])
            return httpx.Response(503 if recover else 401, text='Bearer ' + secret + ' provider-body')
        return httpx.Response(200, json={'model': 'gpt-6-astra', 'status': 'completed',
            'output': [{'type': 'message', 'content': [
                {'type': 'output_text', 'text': json.dumps(next(responses))}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 4}})

    model = MicuModelClient(ModelGatewayConfig('https://example.micuapi.ai/v1', secret),
                            'gpt-6-astra', transport=httpx.MockTransport(handler))
    with TestClient(create_app(service=api)) as http:
        try:
            report = run(RunConfig('http://testserver', TASK, output, family='mechanical_assembly',
                                  provider='micu', max_decisions=2),
                         bench=BenchHTTPClient('http://testserver', client=http), models={'single': model})
        finally:
            model.close()
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert all(event['phase'] == 'configuration_choice' and event['decision_index'] == 0 for event in events)
    assert len({event['request_sha256'] for event in events}) == 1
    assert report['pre_episode_model_calls'] == 1
    if recover:
        assert [event['event'] for event in events] == [
            'model_request_started', 'model_request_failed', 'model_retry_wait',
            'model_retry_wait_finished', 'model_request_started', 'model_request_completed']
        assert requests[0] == requests[1] and len(requests) == 3
        assert report['ready_validated'] and report['error_type'] is None
        assert report['decisions'] == {'single': 2}
        assert report['pre_episode_model_requests'][0]['attempts'] == 2
        control_events = [json.loads(line) for line in
                          (output / 'single/decisions/0001/model_api_events.jsonl').read_text().splitlines()]
        assert all(event['decision_index'] == 1 for event in control_events)
        assert len(workers) == 1 and workers[0].closed
        assert not any(operation == 'step' for operation, _ in workers[0].calls)
    else:
        assert [event['event'] for event in events] == ['model_request_started', 'model_request_failed']
        assert report['configuration_choice']['status'] == 'request_failed'
        assert report['error_type'] == 'ModelGatewayError' and report['episode_id'] is None
        assert report['pre_episode_model_requests'][0]['attempts'] == 1
        assert not workers and len(requests) == 1
    recorded = event_path.read_text() + json.dumps(report) + capsys.readouterr().out
    assert secret not in recorded and 'provider-body' not in recorded
