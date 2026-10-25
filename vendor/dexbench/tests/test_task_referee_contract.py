"""Public-rule identity, information boundaries and canonical outcome routing."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import httpx

from bench.paths import DATA_ROOT, load_json
from bench.protocol.task_referee import contract_sha256, project_task_rules, project_referee_state
from scenes.workrooms.mechanical_assembly.referee import public_contract, bind_task_rules, verdict
from test_agent_service import (runner, frozen_inputs, FakeWorker, PRIVATE_MARKER,
                                session_for, action, join_action, AcknowledgedTeacher,
                                teacher_session, skill_action, request)
from bench.protocol import Budgets
from bench.runtime import service


def declared_task(task_id='example'):
    contract = public_contract(task_id, instruction_zh='放置并松手。',
        phases={'place': '放置', 'settling': '等待稳定', 'complete': '完成', 'invalid': '证据无效', 'failed': '规则违规'},
        reasons={'release_required': '请松手', 'completed': '已完成',
                 'bad_receipt': '接触记录不完整', 'speed_limit': '速度超界', 'forbidden_contact': '禁止接触'},
        success_conditions=['释放后稳定'], recovery=['松手后重新等待稳定'], thresholds={'dwell_s': .5})
    return bind_task_rules({'id': task_id, 'public_rules': {'referee': contract},
                            'evaluation': {'private_pose': PRIVATE_MARKER}})


def score(state='ongoing'):
    phase, reason = {
        'ongoing': ('place', 'release_required'), 'success': ('complete', 'completed'),
        'task_failure': ('failed', 'forbidden_contact'),
        'evaluator_invalid': ('invalid', 'bad_receipt'), 'safety_stop': ('failed', 'speed_limit'),
    }[state]
    return {'success': state == 'success', 'reason': PRIVATE_MARKER,
            'private_contacts': PRIVATE_MARKER, 'referee': verdict(state=state, phase=phase, reasons=[reason])}


def test_only_declared_static_contract_and_coarse_verdict_are_public():
    task = declared_task()
    public = project_task_rules(task)
    current = project_referee_state(score(), public['public_rules'])
    assert PRIVATE_MARKER not in json.dumps([public, current])
    assert current['state'] == 'ongoing' and current['recoverable'] and not current['terminal']
    public['public_rules']['recovery'].clear()
    assert task['public_rules']['referee']['recovery']
    assert project_task_rules({'id': 'old', 'public_rules': {'goal': 'old'}}) is None


@pytest.mark.parametrize('mutate', [
    lambda t: t['public_rules']['referee']['thresholds'].update(dwell_s=.001),
    lambda t: t.update(public_rules_sha256='0'*64),
    lambda t: t.update(public_referee_contract_sha256='0'*64),
    lambda t: t.update(public_rules_file='../rules.json'),
    lambda t: t.update(id='other'),
])
def test_rule_identity_drift_is_rejected(mutate):
    task = declared_task(); mutate(task)
    with pytest.raises(ValueError): project_task_rules(task)


def test_nested_unknown_private_fields_cannot_be_whitelisted_by_rehashing():
    task = declared_task()
    task['public_rules']['referee']['hidden_pose'] = PRIVATE_MARKER
    bind_task_rules(task)
    with pytest.raises(ValueError): project_task_rules(task)


@pytest.mark.parametrize('change', [
    {'phase': 'hidden_phase'}, {'reason_codes': ['undeclared_reason']},
    {'terminal': True}, {'recoverable': False}, {'hidden_pose': PRIVATE_MARKER},
])
def test_untrusted_or_inconsistent_referee_receipt_is_rejected(change):
    value = score(); value['referee'].update(change)
    with pytest.raises(ValueError): project_referee_state(value, declared_task()['public_rules']['referee'])


def install_contract(monkeypatch, frozen_inputs):
    inputs = deepcopy(frozen_inputs[1])
    inputs[2].update(declared_task(inputs[2]['id']))
    monkeypatch.setattr(service, 'task_inputs', lambda *args: deepcopy(inputs))
    original = FakeWorker.__init__
    def init(worker, **kwargs):
        original(worker, **kwargs)
        worker.score = score()
        worker.initial['score'] = deepcopy(worker.score)
        from bench.runtime.worker import public_task_projection
        worker.initial['task'] = public_task_projection(
            service.task_inputs(kwargs['release'], kwargs['task_id'])[2])
    monkeypatch.setattr(FakeWorker, '__init__', init)


@pytest.mark.parametrize('state', ['ongoing', 'success', 'task_failure', 'evaluator_invalid', 'safety_stop'])
def test_generic_episode_routes_coarse_outcomes_without_task_name_branch(
        runner, frozen_inputs, monkeypatch, state):
    install_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner
    session, reply = session_for(make(on_step=lambda worker: setattr(worker, 'score', score(state))))
    assert reply['task']['referee']['state'] == 'ongoing'
    assert PRIVATE_MARKER not in json.dumps(reply['task'])
    session.submit(action(session)); join_action(session)
    assert session.task_requirements()['referee']['state'] == state
    if state == 'ongoing':
        assert session.result is None
    else:
        assert session.result.outcome == state
    assert PRIVATE_MARKER not in json.dumps(session.task_requirements())


def test_bad_referee_ack_does_not_advance_public_time(runner, frozen_inputs, monkeypatch):
    install_contract(monkeypatch, frozen_inputs)
    make, _, _ = runner
    def corrupt(worker):
        worker.score = score()
        worker.score['referee']['phase'] = 'undeclared'
    session, _ = session_for(make(on_step=corrupt))
    before = session.raw['sim_time_s']
    session.submit(action(session)); join_action(session)
    assert session.result.outcome == 'infrastructure_error'
    assert session.raw['sim_time_s'] == before and session.sequence == 0


def test_bad_referee_ack_never_commits_controller_reference(runner, frozen_inputs, monkeypatch):
    install_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner
    teacher = AcknowledgedTeacher()
    session = teacher_session(make, teacher)
    def corrupt(worker):
        worker.score = score()
        worker.score['referee']['phase'] = 'undeclared'
    workers[-1].on_step = corrupt
    session.submit(skill_action(session)); join_action(session)
    assert session.result.outcome == 'infrastructure_error'
    assert teacher.commits == 0 and teacher.discards == 1 and not teacher.pending
    assert session.sequence == 0 and session.raw['sim_time_s'] == 0.


@pytest.mark.parametrize('outcome', ['cancelled', 'wall_timeout', 'action_budget'])
def test_runner_termination_does_not_invent_a_task_failure(runner, frozen_inputs, monkeypatch, outcome):
    install_contract(monkeypatch, frozen_inputs)
    make, clock, _ = runner
    session, _ = session_for(make(), budgets=Budgets(max_actions=1, wall_time_s=10))
    assert session.task_requirements()['episode'] == {'terminal': False, 'outcome': None}
    if outcome == 'cancelled':
        session.finish()
    elif outcome == 'wall_timeout':
        clock.value = 11.
        with pytest.raises(service.ServiceError):
            session.submit(action(session))
    else:
        with pytest.raises(service.ServiceError):
            session.submit({'action_id': 'bad'})
        with pytest.raises(service.ServiceError):
            session.submit({'action_id': 'bad_again'})
    public = session.task_requirements()
    assert public['episode'] == {'terminal': True, 'outcome': outcome}
    assert public['referee']['state'] == 'ongoing'
    assert PRIVATE_MARKER not in json.dumps(public)


def test_public_task_endpoint_exposes_rules_and_separate_episode_outcome(runner, frozen_inputs, monkeypatch):
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    install_contract(monkeypatch, frozen_inputs)
    make, _, _ = runner
    api = make()
    with TestClient(create_app(service=api)) as client:
        response = client.post('/episodes', json=request().model_dump(mode='json'))
        assert response.status_code == 201
        reply = response.json()
        url = f"/episodes/{reply['episode_id']}/task"
        published = client.get(url)
        assert published.status_code == 200 and published.json() == reply['task']
        assert published.json()['public_rules_sha256'] == project_task_rules(
            declared_task(reply['spec']['task_id']))['public_rules_sha256']
        client.post(f"/episodes/{reply['episode_id']}/finish")
        final = client.get(url)
        assert final.json()['episode'] == {'terminal': True, 'outcome': 'cancelled'}
        assert final.json()['referee']['state'] == 'ongoing'
        assert PRIVATE_MARKER not in final.text and DATA_ROOT not in final.text


def test_late_physical_success_is_visible_but_not_scored_after_timeout(runner, frozen_inputs, monkeypatch):
    install_contract(monkeypatch, frozen_inputs)
    make, clock, _ = runner
    def late_success(worker):
        worker.score = score('success')
        clock.value = 1.01
    session, _ = session_for(make(on_step=late_success), budgets=Budgets(wall_time_s=1.))
    session.submit(action(session)); join_action(session)
    public = session.task_requirements()
    assert public['referee']['state'] == 'success'
    assert public['episode'] == {'terminal': True, 'outcome': 'wall_timeout'}
    assert session.result.completion_sim_s is None


@pytest.mark.parametrize('task_id', ['parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
                                    'fastener_assembly', 'gear_train_assembly'])
def test_frozen_mechanical_contracts_match_catalog_and_http_listing(tmp_path, task_id):
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    from bench.isaacsim.release import checked, validate, load_evaluator
    from scenes.workrooms.catalog import task_definition
    definition = task_definition('mechanical_assembly', task_id)
    reference = definition['source_release']
    root = Path(reference['path'])
    checked(root/'release_manifest.json', reference['manifest_sha256'])
    manifest = validate(root, assets=False)
    task = load_json(root/'task.json')
    assert task['public_rules'] == definition['rules']['contract']
    assert load_evaluator(root)(task).task == task
    assert manifest['physics_accepted'] is False
    manager = service.AgentService(tmp_path/'api', release=root, worker_factory=FakeWorker)
    with TestClient(create_app(service=manager)) as client:
        response = client.get('/tasks')
        assert response.status_code == 200
        assert response.json() == [project_task_rules(task)]
        assert DATA_ROOT not in response.text and 'initial_objects' not in response.text
    if task_id == 'gear_train_assembly':
        assert definition['runnable'] and definition['source_release'] == definition['candidate_referee']['release']
        assert definition['evaluator']['implementation'].endswith('.gear_train_assembly.evaluator.SequenceEvaluator')


def test_existing_agent_discovers_and_reads_current_referee_without_actions(
        runner, frozen_inputs, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    from agent.model_client.function_http import FunctionHTTPClient
    from agent.model_client.function_session import FunctionSession
    install_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner
    def settling(worker):
        worker.score = score()
        worker.score['referee']['phase'] = 'settling'
    api = make(on_step=settling)
    session, ready = session_for(api)
    with TestClient(create_app(service=api)) as http:
        client = FunctionHTTPClient('http://testserver', session.spec.episode_id, client=http)
        host = FunctionSession(client, output=tmp_path/'agent', spec=ready['spec'],
                               trial_id='referee_query', max_actions=10, max_calls=10)
        host.initialize()
        assert 'get_task' in {row['function'] for row in host.context()['tools']}
        def query():
            before = (session.sequence, session.raw['sim_time_s'], session.clock.action_attempts,
                      len([op for op, _ in workers[-1].calls if op == 'step']))
            feedback = host.execute({'function': 'get_task', 'arguments': {}})
            assert feedback['ok']
            public = feedback['result']
            assert public == session.task_requirements()
            assert public['public_rules']['schema'] == 'physical_task_referee_contract.v1'
            assert public['referee']['schema'] == 'physical_task_referee_state.v1'
            assert 'schema_version' not in json.dumps(public)
            assert PRIVATE_MARKER not in json.dumps(public)
            assert before == (session.sequence, session.raw['sim_time_s'], session.clock.action_attempts,
                              len([op for op, _ in workers[-1].calls if op == 'step']))
            return public
        assert query()['referee']['phase'] == 'place'
        session.submit(action(session)); join_action(session)
        assert query()['referee']['phase'] == 'settling'
        session.finish()
        assert query()['episode'] == {'terminal': True, 'outcome': 'cancelled'}
        assert host.actions == 0


def test_old_task_does_not_advertise_referee_query(runner):
    make, _, _ = runner
    session, _ = session_for(make())
    capability = next(item for item in session.functions.catalog().functions if item.name == 'get_task')
    assert not capability.available and not capability.advances_physics and not capability.consumes_action_attempt
    reply = session.functions.invoke({'function': 'get_task', 'arguments': {}, 'request_id': 'old/query'})
    assert not reply.ok and reply.error.code == 'unsupported_capability'
    assert session.sequence == 0 and session.clock.action_attempts == 0


@pytest.mark.parametrize('drift', ['task_id', 'rules'])
def test_agent_rejects_internally_valid_task_contract_drift(
        runner, frozen_inputs, monkeypatch, tmp_path, drift):
    from fastapi.testclient import TestClient
    from bench.runtime.server import create_app
    from agent.model_client.function_http import FunctionHTTPClient
    from agent.model_client.function_session import FunctionSession
    install_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner
    api = make()
    session, ready = session_for(api)
    with TestClient(create_app(service=api)) as http:
        original = http.request
        def drifting_request(method, url, **kwargs):
            response = original(method, url, **kwargs)
            if (kwargs.get('json') or {}).get('function') != 'get_task':
                return response
            reply = response.json()
            result = reply['result']
            if drift == 'task_id':
                result['task_id'] = result['public_rules']['task_id'] = 'other_task'
            else:
                result['instruction'] = result['public_rules']['instruction_zh'] = '另一版本的规则。'
            result['public_rules_sha256'] = contract_sha256(result['public_rules'])
            return httpx.Response(response.status_code, json=reply)
        monkeypatch.setattr(http, 'request', drifting_request)
        client = FunctionHTTPClient('http://testserver', session.spec.episode_id,
                                    client=http, spec=ready['spec'])
        client.bind_episode(ready['spec'], task=ready['task'])
        host = FunctionSession(client, output=tmp_path/'agent', spec=ready['spec'],
                               trial_id='referee_query', max_actions=10, max_calls=10)
        host.initialize()
        feedback = host.execute({'function': 'get_task', 'arguments': {}})
        assert not feedback['ok'] and 'result' not in feedback
        assert feedback['error']['error_category'] == 'invalid_response'
        assert host.last_feedback == feedback
        assert host.actions == 0 and session.clock.action_attempts == 0
        assert not any(operation == 'step' for operation, _ in workers[-1].calls)
