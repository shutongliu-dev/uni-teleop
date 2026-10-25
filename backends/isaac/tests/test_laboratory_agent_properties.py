"""Declared dry mass reaches Agent context; changing its contract is rejected."""
from copy import deepcopy
import json

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.function_http import FunctionHTTPClient
from agent.model_client.function_runner import task_requirements_context
from bench.protocol.task_referee import contract_sha256, project_task_rules
from scenes.workrooms.laboratory import rules
from test_function_response_bindings import episode_spec


def task_reply(task_id):
    task = rules.bind_task_rules({'id': task_id, 'public_rules': rules.public_rules(task_id)})
    public = project_task_rules(task)
    public.update(referee={'schema': 'physical_task_referee_state.v1', 'state': 'ongoing',
        'phase': next(iter(public['public_rules']['phases'])),
        'reason_codes': [next(iter(public['public_rules']['reason_descriptions']))],
        'terminal': False, 'recoverable': True}, episode={'terminal': False, 'outcome': None})
    return public


def response(public, request_id):
    return {'ok': True, 'request_id': request_id, 'function': 'get_task',
        'context': {'episode_id': 'ep1', 'observation_sequence': 9, 'observation_sim_time_s': .18,
            'episode_finished': False, 'request_recorded': True,
            'budget': {'limits': {}, 'action_attempts': 1, 'elapsed_sim_s': .18,
                'elapsed_wall_s': 1., 'remaining_wall_s': 100.}}, 'result': public}


@pytest.mark.parametrize('task_id', rules.TASK_IDS)
def test_get_task_and_repeated_model_requirements_keep_nominal_object_properties(task_id):
    public = task_reply(task_id)
    spec = episode_spec(task_id)
    def handler(request):
        payload = json.loads(request.content)
        return httpx.Response(200, json=response(public, payload['request_id']))
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        agent = FunctionHTTPClient('http://bench.test', 'ep1', client=http)
        agent.bind_episode(spec, task=public)
        result = agent.call({'request_id': 'properties', 'function': 'get_task', 'arguments': {}})['result']
    expected = rules.object_properties(task_id)
    assert result['public_rules']['object_properties'] == expected
    context = task_requirements_context(result)
    assert context['public_rules']['object_properties'] == expected
    assert not {'referee', 'episode', 'scoring'} & context.keys()
    assert 'referee' in result  # Static projection does not mutate the live receipt.
    assert all(set(row) == {'id', 'label', 'nominal_mass_kg', 'basis'} for row in expected)


@pytest.mark.parametrize('task_id', rules.TASK_IDS)
@pytest.mark.parametrize('mutation', ['mass_unhashed', 'mass_rehashed', 'omitted_rehashed', 'pose_rehashed'])
def test_bound_agent_rejects_mass_contract_drift_even_after_rehash(task_id, mutation):
    initial = task_reply(task_id)
    changed = deepcopy(initial)
    if mutation == 'omitted_rehashed':
        del changed['public_rules']['object_properties']
    elif mutation == 'pose_rehashed':
        changed['public_rules']['object_properties'][0]['position_m'] = [0., 0., 1.]
    else:
        changed['public_rules']['object_properties'][0]['nominal_mass_kg'] *= 2
    if mutation.endswith('_rehashed'):
        changed['public_rules_sha256'] = contract_sha256(changed['public_rules'])
    def handler(request):
        payload = json.loads(request.content)
        return httpx.Response(200, json=response(changed, payload['request_id']))
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        agent = FunctionHTTPClient('http://bench.test', 'ep1', client=http)
        agent.bind_episode(episode_spec(task_id), task=initial)
        with pytest.raises(BenchHTTPError) as error:
            agent.call({'request_id': 'changed', 'function': 'get_task', 'arguments': {}})
        assert error.value.category == 'invalid_response'
        changed.clear()
        changed.update(deepcopy(initial))
        assert agent.call({'request_id': 'original', 'function': 'get_task', 'arguments': {}})['result'] == initial
