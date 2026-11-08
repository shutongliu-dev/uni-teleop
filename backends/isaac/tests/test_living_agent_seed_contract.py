"""Public discovery/ready/function binding; synthetic CPU data, no physics/model."""
from copy import deepcopy
import json

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError, _task_query
from agent.model_client.function_http import FunctionHTTPClient
from agent.model_client.function_runner import _validated_ready
from agent.model_client.scene_tasks import TaskSelection
from agent.model_client.seeded_tasks import select_episode_case
from bench.paths import load_json
from bench.protocol import PolicyRegistry
from bench.protocol.schemas import PolicyDescriptor
from bench.protocol.task_referee import project_task_rules, contract_sha256
from test_agent_service import runner, frozen_inputs, session_for, Teacher
from test_task_referee_contract import declared_task


HASH = 'b'*64


def reset_case(seed):
    return {'schema': 'living_room_public_reset_case.v1', 'seed': seed,
        'configuration_id': 'single_arm', 'case_sha256': str(seed)*64,
        'static_geometry': {'dimensions_m': {'item': [.0665 if seed else .07, .05, .036]},
            'bodies': [{'object_id': 'box', 'local_bounds_m': [[-.15, -.12, 0], [.15, .12, .11]],
                        'nominal_mass_kg': .76 if seed else .8}],
            'goal': {'operation': 'collect_and_close', 'tray_inside_xy_m': [[.856, -.224], [1.024, -.016]],
                     'tray_floor_z_m': .786}},
        'scope': 'Static authored dimensions and goals for this episode; live object poses and native evidence remain private.'}


def public_task(seed):
    value = project_task_rules(declared_task('storage_box'))
    value['public_rules']['object_properties'] = [{'id': 'box', 'label': 'box',
        'nominal_mass_kg': .76 if seed else .8, 'basis': 'CPU protocol fixture'}]
    value['public_rules_sha256'] = contract_sha256(value['public_rules'])
    value.update(reset_case=reset_case(seed), configuration_id='single_arm', task_release_sha256=HASH)
    return value


def selection():
    return TaskSelection('living_room', 'storage_box', pin_release=True,
        source_release={'manifest_sha256': HASH}, configuration_id='single_arm')


class PublicBench:
    def __init__(self, response):
        self.response, self.calls = response, []

    def list_tasks(self, **kwargs):
        self.calls.append(kwargs)
        return deepcopy([self.response])


@pytest.mark.parametrize('seed', [None, 0, 1, 9])
def test_requested_public_seed_is_resolved_before_ready(seed):
    effective = 0 if seed is None else seed
    original, expected = public_task(0), public_task(effective)
    bench = PublicBench(expected)
    selected, actual_seed = select_episode_case(bench, selection(), original, seed)
    assert actual_seed == effective and selected == expected
    assert bench.calls == [{'task_id': 'storage_box', 'configuration_id': 'single_arm', 'seed': effective}]
    if effective:
        assert selected['public_rules_sha256'] != original['public_rules_sha256']
    assert original == public_task(0)


@pytest.mark.parametrize('seed', [True, False, -1, 10, 1.0, '1'])
def test_invalid_finite_seed_never_reaches_transport(seed):
    bench = PublicBench(public_task(0))
    with pytest.raises(ValueError):
        select_episode_case(bench, selection(), public_task(0), seed)
    assert bench.calls == []
    with pytest.raises(BenchHTTPError):
        _task_query('storage_box', seed=seed)


@pytest.mark.parametrize('field,change', [
    ('release', lambda t: t.update(task_release_sha256='c'*64)),
    ('seed', lambda t: t['reset_case'].update(seed=0)),
    ('configuration', lambda t: t['reset_case'].update(configuration_id='dual_same_side')),
    ('omitted', lambda t: t.pop('reset_case')),
    ('private', lambda t: t['reset_case'].update(actual_body_poses_m={'hidden': [1, 2, 3]})),
    ('schema', lambda t: t['reset_case'].update(schema='unknown')),
])
def test_requested_discovery_identity_mismatch_is_rejected(field, change):
    response = public_task(1); change(response)
    with pytest.raises((ValueError, BenchHTTPError)):
        select_episode_case(PublicBench(response), selection(), public_task(0), 1)


@pytest.mark.parametrize('seed', [None, 12345])
def test_legacy_task0_seed_and_discovery_remain_unchanged(seed):
    bench = PublicBench(None)
    task = {'task_id': 'task0', 'task_kind': 'manipulation', 'goal': {'type': 'template'}}
    result, effective = select_episode_case(bench, TaskSelection('card_room', 'task0'), task, seed)
    assert result == task and effective == seed and bench.calls == []


def test_task_query_encodes_seed_without_affecting_legacy_queries():
    assert _task_query(None) == ''
    assert _task_query('task0', 'single_arm') == '?task_id=task0&configuration_id=single_arm'
    with pytest.raises(BenchHTTPError):
        _task_query(None, seed=1)
    seen = []
    http = httpx.Client(transport=httpx.MockTransport(lambda request: (
        seen.append(str(request.url)) or httpx.Response(200, json=[]))))
    client = BenchHTTPClient('http://bench.test', client=http)
    client.list_tasks('storage_box', 'single_arm', seed=3)
    assert seen == ['http://bench.test/tasks?task_id=storage_box&configuration_id=single_arm&seed=3']


@pytest.fixture
def created(runner):
    make, _, _ = runner
    registry = PolicyRegistry(); registry.register(Teacher())
    api = make(registry=registry)
    _, ready = session_for(api, track='system0', policy_id='test_teacher', policy_version='v1')
    ready = deepcopy(ready)
    ready['spec'].update(task_id='storage_box', seed=1, configuration_id='single_arm', task_release_sha256=HASH)
    value = public_task(1)
    value.pop('configuration_id'); value.pop('task_release_sha256')
    value.update(referee={'schema': 'physical_task_referee_state.v1', 'state': 'ongoing',
        'phase': 'place', 'reason_codes': ['release_required'], 'terminal': False, 'recoverable': True},
        episode={'terminal': False, 'outcome': None})
    ready['task'] = value
    spec = ready['spec']
    request = {k: spec[k] for k in ('task_id', 'robot_id', 'agent_config_sha256', 'budgets',
                                  'observation_profile', 'configuration_id', 'seed')}
    request.update(policy_id='test_teacher', policy_version='v1')
    return ready, request, PolicyDescriptor.model_validate(spec['system0'])


def test_nonzero_ready_uses_selected_rules_and_preserves_old_hash_check(created):
    ready, request, controller = created
    discovered, selected = select_episode_case(PublicBench(public_task(1)), selection(), public_task(0), 1)
    assert selected == 1 and discovered['reset_case']['seed'] == 1
    _, task = _validated_ready(ready, request, public_task(1), False, controller=controller)
    assert task['reset_case']['seed'] == 1
    with pytest.raises((ValueError, BenchHTTPError)):
        _validated_ready(ready, request, public_task(0), False, controller=controller)
    bad = deepcopy(ready); bad['task']['public_rules_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='rules differ'):
        _validated_ready(bad, request, public_task(1), False, controller=controller)


@pytest.mark.parametrize('mutation', [
    lambda t: t['reset_case'].update(case_sha256='a'*64),
    lambda t: t['reset_case']['static_geometry']['bodies'][0].update(nominal_mass_kg=.8),
    lambda t: t.pop('reset_case'),
    lambda t: t['reset_case'].update(seed=2),
])
def test_ready_rejects_changed_case_or_public_geometry(created, mutation):
    ready, request, controller = created
    mutation(ready['task'])
    with pytest.raises((ValueError, BenchHTTPError)):
        _validated_ready(ready, request, public_task(1), False, controller=controller)


@pytest.mark.parametrize('fault', [None, 'hash', 'geometry', 'omitted'])
def test_every_get_task_must_retain_ready_reset_case(created, fault):
    ready, _, _ = created
    reply = deepcopy(ready['task'])
    if fault == 'hash':
        reply['reset_case']['case_sha256'] = 'a'*64
    elif fault == 'geometry':
        reply['reset_case']['static_geometry']['bodies'][0]['nominal_mass_kg'] = .8
    elif fault == 'omitted':
        reply.pop('reset_case')
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
        call = {'request_id': 'public/requirements', 'function': 'get_task', 'arguments': {}}
        if fault:
            with pytest.raises(BenchHTTPError):
                client.call(call)
        else:
            assert client.call(call)['result']['reset_case'] == ready['task']['reset_case']
        changed = deepcopy(ready['task']); changed['reset_case']['case_sha256'] = 'a'*64
        with pytest.raises(BenchHTTPError):
            client.bind_episode(ready['spec'], task=changed)


@pytest.mark.parametrize('robot_profile', ['original', 'l_mount'])
@pytest.mark.parametrize('task_id,seed', [('storage_box', 1), ('water_pouring', 2)])
def test_full_runner_nonzero_published_case_with_cpu_worker(tmp_path, monkeypatch, task_id, seed, robot_profile):
    from pathlib import Path
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    from bench.runtime.worker import public_task_projection
    from scenes.workrooms.catalog import task_definition
    from agent.model_client.function_runner import RunConfig, run
    import test_scene_function_runner as fixtures
    from test_function_runner import ScriptedModel
    # The runner pins the catalog-selected release of the requested profile;
    # serve exactly that published parameterized release.
    release = Path(task_definition('living_room', task_id, robot_profile=robot_profile)['source_release']['path'])
    if not release.is_dir():
        pytest.skip('Published immutable randomized inputs unavailable')
    # The existing CPU fixture predates reset cases; use the actual new public
    # worker projection, not a forged native-success state or private injection.
    monkeypatch.setattr(fixtures, 'project_manipulation_task', public_task_projection)
    if task_id == 'storage_box':
        # Its frozen evaluator has no state before a native baseline. This CPU
        # transport double declares an unscored opening state, never success.
        monkeypatch.setattr(fixtures, 'reset_score', lambda *args: {
            'success': False, 'reason': 'reset_baseline_not_scored',
            'referee': {'schema': 'physical_task_referee_state.v1', 'state': 'ongoing',
                        'phase': 'opening', 'reason_codes': ['reset_baseline_not_scored'],
                        'terminal': False, 'recoverable': True}})
    service, workers, _ = fixtures.mechanical_service(tmp_path, task_id, family='living_room', release=release)
    decisions = ([{'function': 'get_task', 'arguments': {}}] if task_id == 'storage_box' else [])
    decisions += [{'stop': True}]
    model = ScriptedModel(decisions)
    with TestClient(create_app(service=service)) as http:
        report = run(RunConfig('http://testserver', task_id, tmp_path/'run', family='living_room',
            configuration='single_arm', seed=seed, max_decisions=4, robot_profile=robot_profile),
            bench=BenchHTTPClient('http://testserver', client=http), models={'single': model})
    assert report['error_type'] is None, report
    assert report['run_status'] == 'completed', report
    assert len(workers) == 1
    request = load_json(tmp_path/'run/create_request.json')
    ready = load_json(tmp_path/'run/episode_ready.json')
    discovery = load_json(tmp_path/'run/episode_case_discovery.json')
    assert request['seed'] == ready['spec']['seed'] == seed
    assert discovery['reset_case'] == ready['task']['reset_case']
