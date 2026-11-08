"""Actual frozen inputs and public HTTP clients, replacing the native worker."""
import json

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.function_runner import RunConfig, run
from bench.paths import load_json
from bench.runtime.server import create_app
from scenes.workrooms.laundry_room.rules import ALL_TASK_IDS
from test_agent_service import PRIVATE_MARKER
from test_function_runner import ScriptedModel
from test_scene_function_runner import mechanical_service


@pytest.mark.parametrize('task_id', ALL_TASK_IDS)
def test_actual_laundry_http_and_model_context_publish_frozen_goal(tmp_path, task_id):
    service, workers, verified = mechanical_service(tmp_path, task_id, family='laundry_room')
    expected = verified['task']['public_goal']
    model = ScriptedModel([{'function': 'get_task', 'arguments': {}}, {'stop': True}])
    with TestClient(create_app(service=service)) as transport:
        bench = BenchHTTPClient('http://testserver', client=transport)
        published, = bench.list_tasks()
        assert published['goal'] == expected
        assert published['public_rules_sha256'] == verified['task']['public_referee_contract_sha256']
        result = run(RunConfig('http://testserver', task_id, tmp_path/'run',
            family='laundry_room', max_decisions=3,
            configuration='dual_opposed' if verified['definition'].get('robot_configurations') else 'auto'),
            bench=bench, models={'single': model})
        assert result['error_type'] is None, result
        ready = load_json(tmp_path/'run/episode_ready.json')
        assert ready['task']['goal'] == expected
        task_reply = transport.get(f"/episodes/{result['episode_id']}/task")
        assert task_reply.status_code == 200 and task_reply.json()['goal'] == expected
    contexts = [json.loads(turns[-1]['content'][0]['text']) for turns in model.turns]
    assert len(contexts) == 2
    assert all(context['task']['goal'] == expected for context in contexts)
    assert contexts[1]['last_feedback']['result']['goal'] == expected
    assert PRIVATE_MARKER not in json.dumps(contexts)
    assert len(workers) == 1 and workers[0].closed
    assert not any(operation == 'step' for operation, _ in workers[0].calls)
