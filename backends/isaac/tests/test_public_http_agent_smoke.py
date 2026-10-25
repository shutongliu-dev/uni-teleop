"""Actual Agent parsing over ordinary HTTP; CPU workers supply no physics proof."""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from agent.model_client.scene_tasks import TaskSetupError
from bench.runtime.server import create_app
from scripts.validation.public_http_smoke import agent_readonly_probe
from test_agent_service import PRIVATE_MARKER, request
from test_laboratory_runtime import laboratory, laboratory_service


def test_catalog_selected_laboratory_public_contract_reaches_agent_without_actions(tmp_path, laboratory):
    task_id, _, _, (_, _, task, _) = laboratory
    api, workers = laboratory_service(tmp_path, laboratory)
    with TestClient(create_app(service=api)) as http:
        response = http.post('/episodes', json=request(task_id=task_id, seed=71).model_dump(mode='json'))
        assert response.status_code == 201
        created = response.json()
        for phase in ('created', 'before_finish'):
            receipt, calls = agent_readonly_probe(http, 'http://testserver', family='laboratory',
                task_id=task_id, created=created, phase=phase, expected_observation=created['observation'])
            assert receipt['catalog_release_verified'] and receipt['agent_public_parsing_verified']
            assert receipt['static_model_requirements_verified'] and receipt['read_only_verified']
            assert receipt['model_calls'] == receipt['actions_submitted'] == 0
            properties = task['public_rules']['referee'].get('object_properties', [])
            assert receipt['nominal_object_property_count'] == len(properties)
            assert calls[1]['response']['result']['public_rules'] == task['public_rules']['referee']
            assert [row['request']['function'] for row in calls] == ['observe', 'get_task', 'observe']
            assert PRIVATE_MARKER not in json.dumps(calls)
        api.get(created['episode_id']).finish()
    assert workers[0].closed
    assert all(operation != 'step' for operation, _ in workers[0].calls)


def test_agent_native_probe_rejects_release_or_creation_rule_drift(tmp_path, laboratory):
    task_id = laboratory[0]
    api, workers = laboratory_service(tmp_path, laboratory)
    with TestClient(create_app(service=api)) as http:
        created = http.post('/episodes', json=request(task_id=task_id, seed=71).model_dump(mode='json')).json()
        changed = deepcopy(created)
        changed['spec']['task_release_sha256'] = 'f'*64
        with pytest.raises(TaskSetupError, match='catalog-selected'):
            agent_readonly_probe(http, 'http://testserver', family='laboratory', task_id=task_id,
                created=changed, phase='drift', expected_observation=created['observation'])
        changed = deepcopy(created)
        changed['task']['public_rules']['thresholds']['stable_s'] = .01
        with pytest.raises(RuntimeError, match='disagree on public rules'):
            agent_readonly_probe(http, 'http://testserver', family='laboratory', task_id=task_id,
                created=changed, phase='drift', expected_observation=created['observation'])
        api.get(created['episode_id']).finish()
    assert workers[0].closed
    assert all(operation != 'step' for operation, _ in workers[0].calls)
