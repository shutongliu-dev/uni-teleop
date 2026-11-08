"""Actual Agent parsing over ordinary HTTP; CPU workers supply no physics proof."""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.scene_tasks import TaskSetupError
from bench.runtime.server import create_app
from scripts.validation.public_http_smoke import agent_readonly_probe
from test_agent_service import PRIVATE_MARKER, request
from test_laboratory_runtime import SEED, laboratory
from test_scene_function_runner import mechanical_service


def laboratory_service(tmp_path, laboratory):
    # Serve the fixture's own catalog release; each robot profile has its own pin.
    api, workers, _ = mechanical_service(tmp_path, laboratory[0], family='laboratory', release=laboratory[1])
    return api, workers


def create_agent_episode(http, task_id):
    controller, = [row for row in http.get('/policies', params={'task_id': task_id}).json()
                   if row['policy_id'] == 'classical_fingertip']
    payload = request(task_id=task_id, seed=SEED, track='system0',
                      policy_id=controller['policy_id'],
                      policy_version=controller['version']).model_dump(mode='json')
    response = http.post('/episodes', json=payload)
    assert response.status_code == 201, response.text
    return response.json(), payload, controller


def test_catalog_selected_laboratory_public_contract_reaches_agent_without_actions(tmp_path, laboratory):
    task_id, _, _, (_, _, task, _), profile = laboratory
    api, workers = laboratory_service(tmp_path, laboratory)
    with TestClient(create_app(service=api)) as http:
        created, payload, controller = create_agent_episode(http, task_id)
        for phase in ('created', 'before_finish'):
            receipt, calls = agent_readonly_probe(http, 'http://testserver', family='laboratory',
                task_id=task_id, created=created, phase=phase, expected_observation=created['observation'],
                create_request=payload, controller=controller, robot_profile=profile)
            assert receipt['catalog_release_verified'] and receipt['agent_public_parsing_verified']
            assert receipt['agent_ready_contract_verified'] and receipt['track'] == 'system0'
            assert len(receipt['controller_sha256']) == len(receipt['create_request_sha256']) == 64
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
    task_id, profile = laboratory[0], laboratory[-1]
    api, workers = laboratory_service(tmp_path, laboratory)
    with TestClient(create_app(service=api)) as http:
        created, payload, controller = create_agent_episode(http, task_id)
        changed = deepcopy(created)
        changed['spec']['task_release_sha256'] = 'f'*64
        with pytest.raises(TaskSetupError, match='catalog-selected'):
            agent_readonly_probe(http, 'http://testserver', family='laboratory', task_id=task_id,
                created=changed, phase='drift', expected_observation=created['observation'],
                create_request=payload, controller=controller, robot_profile=profile)
        # The other profile's catalog pin never verifies this physical robot.
        with pytest.raises(TaskSetupError, match='catalog-selected'):
            agent_readonly_probe(http, 'http://testserver', family='laboratory', task_id=task_id,
                created=created, phase='drift', expected_observation=created['observation'],
                create_request=payload, controller=controller,
                robot_profile='l_mount' if profile == 'original' else 'original')
        changed = deepcopy(created)
        changed['task']['public_rules']['thresholds']['stable_s'] = .01
        with pytest.raises(RuntimeError, match='disagree on public rules'):
            agent_readonly_probe(http, 'http://testserver', family='laboratory', task_id=task_id,
                created=changed, phase='drift', expected_observation=created['observation'],
                create_request=payload, controller=controller, robot_profile=profile)
        api.get(created['episode_id']).finish()
    assert workers[0].closed
    assert all(operation != 'step' for operation, _ in workers[0].calls)


@pytest.mark.parametrize('field,value', [
    ('implementation_sha256', 'e'*64), ('weights_sha256', 'd'*64),
    ('control_period_s', .04),
])
def test_native_agent_probe_checks_discovery_controller_before_public_queries(
        tmp_path, laboratory, monkeypatch, field, value):
    task_id = laboratory[0]
    api, workers = laboratory_service(tmp_path, laboratory)
    with TestClient(create_app(service=api)) as http:
        created, payload, controller = create_agent_episode(http, task_id)
        changed = deepcopy(created)
        for section in ('spec', 'capabilities'):
            changed[section]['system0'][field] = value
        if field == 'control_period_s':
            changed['capabilities']['control_period_s'] = value

        def unexpected_query(*args, **kwargs):
            pytest.fail('Untrusted ready response reached a public Agent query')

        monkeypatch.setattr('agent.model_client.function_http.FunctionHTTPClient.call', unexpected_query)
        with pytest.raises(BenchHTTPError, match='discovered controller'):
            agent_readonly_probe(http, 'http://testserver', family='laboratory', task_id=task_id,
                created=changed, phase='drift', expected_observation=created['observation'],
                create_request=payload, controller=controller, robot_profile=laboratory[-1])
        api.get(created['episode_id']).finish()
    assert workers[0].closed
    assert all(operation != 'step' for operation, _ in workers[0].calls)
