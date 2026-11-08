"""Finite laundry identity and HTTP binding; CPU doubles, not native evidence."""
from copy import deepcopy
from hashlib import sha256
import json

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.function_http import FunctionHTTPClient
from agent.model_client.function_runner import _validated_ready
from agent.model_client.scene_tasks import TaskSelection, TaskSetupError
from agent.model_client.seeded_tasks import checked_reset_case, select_episode_case
from bench.protocol.laundry_reset import PublicLaundryResetCase, SCOPE, TASK_IDS, VARIATION
from bench.protocol.workroom_reset import PublicTaskRequirements
from bench.protocol.task_referee import project_task_rules
from test_task_referee_contract import declared_task
from test_living_agent_seed_contract import created, runner, frozen_inputs

HASH = 'b' * 64
PARENT = 'c' * 64
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')


def public_case(task_id='towel_flattening', seed=0, configuration='single_arm'):
    digest = sha256(json.dumps([task_id, seed, configuration]).encode()).hexdigest()
    return {'schema': 'laundry_public_reset_case.v1', 'family': 'laundry_room',
            'task_id': task_id, 'seed': seed, 'configuration_id': configuration,
            'case_sha256': digest, 'case_fingerprint': digest, 'parent_release_sha256': PARENT,
            'variation': VARIATION, 'scope': SCOPE}


def published(task_id='towel_flattening', seed=0, configuration='single_arm'):
    result = project_task_rules(declared_task(task_id))
    result.update(reset_case=public_case(task_id, seed, configuration),
                  configuration_id=configuration, task_release_sha256=HASH, episode_seed=seed)
    return result


def selection(task_id='towel_flattening', configuration='single_arm'):
    return TaskSelection('laundry_room', task_id, pin_release=True,
                        source_release={'manifest_sha256': HASH}, configuration_id=configuration)


def live_task(value):
    result = {k: deepcopy(v) for k, v in value.items()
              if k not in ('configuration_id', 'task_release_sha256', 'episode_seed')}
    result.update(referee={'schema': 'physical_task_referee_state.v1', 'state': 'ongoing',
        'phase': 'place', 'reason_codes': ['release_required'], 'terminal': False, 'recoverable': True},
        episode={'terminal': False, 'outcome': None})
    return result


class PublicBench:
    def __init__(self, value):
        self.value, self.calls = value, []

    def list_tasks(self, **kwargs):
        self.calls.append(kwargs)
        return [deepcopy(self.value)]


@pytest.mark.parametrize('task_id', TASK_IDS)
@pytest.mark.parametrize('configuration', CONFIGURATIONS)
@pytest.mark.parametrize('seed', [None, 0, 5, 9])
def test_all_current_task_installations_select_finite_seed_before_creation(task_id, configuration, seed):
    effective = 0 if seed is None else seed
    baseline, expected = published(task_id, 0, configuration), published(task_id, effective, configuration)
    bench = PublicBench(expected)
    actual, actual_seed = select_episode_case(bench, selection(task_id, configuration), baseline, seed)
    assert actual == expected and actual_seed == effective
    assert bench.calls == [{'task_id': task_id, 'configuration_id': configuration, 'seed': effective}]
    parsed = PublicTaskRequirements.model_validate(live_task(actual))
    assert parsed.model_dump(mode='json')['reset_case'] == expected['reset_case']
    assert baseline == published(task_id, 0, configuration)


@pytest.mark.parametrize('seed', [True, False, -1, 10, 1.0, '1'])
def test_invalid_seed_rejected_before_transport(seed):
    bench = PublicBench(published())
    with pytest.raises(ValueError):
        select_episode_case(bench, selection(), published(), seed)
    assert not bench.calls
    value = public_case(); value['seed'] = seed
    with pytest.raises(ValueError):
        PublicLaundryResetCase.model_validate(value)


@pytest.mark.parametrize('change', [
    {'family': 'kitchen'}, {'task_id': 'shirt_stacking'}, {'configuration_id': 'unpublished'},
    {'case_sha256': 'z'*64}, {'case_fingerprint': 'x'}, {'parent_release_sha256': 'A'*64},
    {'variation': 'mesh_scaling'}, {'scope': 'public ground truth'},
    {'initial_vertices': [[1., 2., 3.]]}, {'cloth_nodes': [[1., 2., 3.]]},
    {'static_geometry': {'private': True}}, {'xy_translation_m': [.001, 0.]},
])
def test_public_contract_rejects_private_data_or_changed_scope(change):
    value = public_case(); value.update(change)
    with pytest.raises((ValueError, BenchHTTPError)):
        checked_reset_case(value, seed=0, configuration_id='single_arm', task_id='towel_flattening')


@pytest.mark.parametrize('change', [
    lambda t: t.update(task_release_sha256='d'*64),
    lambda t: t['reset_case'].update(parent_release_sha256='d'*64),
    lambda t: t['reset_case'].update(task_id='shirt_folding'),
    lambda t: t['reset_case'].update(seed=0),
    lambda t: t['reset_case'].update(configuration_id='dual_opposed'),
    lambda t: t['reset_case'].update(schema='unknown'),
    lambda t: t.update(episode_seed=0),
    lambda t: t.pop('reset_case'),
])
def test_selected_discovery_cannot_change_frozen_identity(change):
    value = published(seed=5); change(value)
    with pytest.raises((BenchHTTPError, TaskSetupError)):
        select_episode_case(PublicBench(value), selection(), published(), 5)


def test_public_requirements_reject_reset_for_another_task():
    value = live_task(published()); value['task_id'] = 'shirt_folding'
    with pytest.raises(ValueError):
        PublicTaskRequirements.model_validate(value)


@pytest.mark.parametrize('field', ['case_sha256', 'case_fingerprint'])
def test_requerying_the_same_seed_cannot_change_its_identity(field):
    value = published(); value['reset_case'][field] = 'd'*64
    with pytest.raises(BenchHTTPError):
        select_episode_case(PublicBench(value), selection(), published(), None)


@pytest.fixture
def laundry_created(created):
    ready, request, controller = deepcopy(created)
    ready['spec']['task_id'] = request['task_id'] = 'towel_flattening'
    ready['task'] = live_task(published(seed=1))
    return ready, request, controller


@pytest.mark.parametrize('field', [None, 'case_sha256', 'case_fingerprint', 'parent_release_sha256',
                                   'task_id', 'seed', 'configuration_id', 'omitted'])
def test_ready_requires_exact_discovered_case(laundry_created, field):
    ready, request, controller = laundry_created
    if field == 'omitted':
        ready['task'].pop('reset_case')
    elif field:
        ready['task']['reset_case'][field] = {'task_id': 'shirt_folding', 'seed': 2,
            'configuration_id': 'dual_opposed'}.get(field, 'd'*64)
    if field:
        with pytest.raises((ValueError, BenchHTTPError)):
            _validated_ready(ready, request, published(seed=1), False, controller=controller)
    else:
        _, task = _validated_ready(ready, request, published(seed=1), False, controller=controller)
        assert task['reset_case'] == public_case(seed=1)


@pytest.mark.parametrize('field', [None, 'case_sha256', 'case_fingerprint', 'parent_release_sha256',
                                   'cloth_nodes', 'omitted'])
def test_every_get_task_keeps_the_ready_identity(laundry_created, field):
    ready, _, _ = laundry_created
    reply = deepcopy(ready['task'])
    if field == 'omitted':
        reply.pop('reset_case')
    elif field:
        reply['reset_case'][field] = 'd'*64
    def handle(request):
        command = json.loads(request.content)
        return httpx.Response(200, json={'ok': True, 'request_id': command['request_id'], 'function': 'get_task',
            'context': {'episode_id': ready['episode_id'], 'observation_sequence': 0,
                'observation_sim_time_s': 0., 'episode_finished': False, 'request_recorded': True,
                'budget': {'limits': {}, 'action_attempts': 0, 'elapsed_sim_s': 0.,
                           'elapsed_wall_s': 0., 'remaining_wall_s': 100.}}, 'result': reply})
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = FunctionHTTPClient('http://bench.test', ready['episode_id'], client=http, spec=ready['spec'])
        client.bind_episode(ready['spec'], task=ready['task'])
        call = {'request_id': 'requirements', 'function': 'get_task', 'arguments': {}}
        if field:
            with pytest.raises(BenchHTTPError):
                client.call(call)
        else:
            assert client.call(call)['result']['reset_case'] == public_case(seed=1)
        changed = deepcopy(ready['task']); changed['reset_case']['case_fingerprint'] = 'd'*64
        with pytest.raises(BenchHTTPError):
            client.bind_episode(ready['spec'], task=changed)
