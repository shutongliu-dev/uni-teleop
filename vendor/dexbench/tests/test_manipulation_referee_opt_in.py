"""CPU lifecycle checks for standard referee opt-in through scene projection."""
from copy import deepcopy
import json

import pytest

from bench.paths import DATA_ROOT
from bench.protocol.task_referee import project_task_rules
from bench.runtime import manipulation_tasks, service
from test_agent_service import (
    FakeWorker, PRIVATE_MARKER, action, frozen_inputs, join_action, request,
    runner, session_for,
)
from test_task_referee_contract import declared_task, score


def install_projection(monkeypatch, frozen_inputs, *, opt_in=True,
                       projection_change=None, worker_change=None, initial_state='ongoing'):
    manifest, inputs = deepcopy(frozen_inputs)
    manifest['schema_version'] = 'isaacsim-living-room-manipulation-release-v1'
    task = inputs[2]
    if opt_in:
        task.update(declared_task(task['id']))
    else:
        task['public_rules'] = {'schema': 'legacy_fixture_v1'}
    task['schema_version'] = 'living-room-manipulation-task-v1'
    monkeypatch.setattr(service, 'validate', lambda *args: deepcopy(manifest))
    monkeypatch.setattr(service, 'task_inputs', lambda *args, **kwargs: deepcopy(inputs))

    def project(value):
        public = (project_task_rules(value) if opt_in else {
            'task_id': value['id'], 'task_kind': 'manipulation',
            'create_endpoint': 'POST /episodes', 'instruction': 'Legacy fixture.',
            'public_rules': deepcopy(value['public_rules']),
        })
        if projection_change:
            projection_change(public)
        return public
    # EpisodeSession and the native worker import this same scene-owned hook.
    monkeypatch.setattr(manipulation_tasks, 'project_manipulation_task', project)
    original = FakeWorker.__init__
    def initialize(worker, **kwargs):
        original(worker, **kwargs)
        worker.initial['task'] = project(task)
        if worker_change:
            worker_change(worker.initial)
        if opt_in:
            worker.score = score(initial_state)
            worker.initial['score'] = deepcopy(worker.score)
        worker.raw['task_state'] = {'television': {'power_on': True,
            'continuous_on_s': 9.9, 'private_receipt': PRIVATE_MARKER}}
        worker.initial['observation'] = deepcopy(worker.raw)
    monkeypatch.setattr(FakeWorker, '__init__', initialize)
    return task


def test_manipulation_opt_in_publishes_read_only_rules_and_current_phase(
        runner, frozen_inputs, monkeypatch):
    task = install_projection(monkeypatch, frozen_inputs)
    make, _, workers = runner
    def settling(worker):
        worker.score = score()
        worker.score['referee']['phase'] = 'settling'
        worker.score['private_counters'] = {'continuous_on_s': 10., 'receipt': PRIVATE_MARKER}
    session, ready = session_for(make(on_step=settling))
    public = project_task_rules(task)
    assert session.public_task == workers[-1].initial['task'] == public
    assert session.referee_contract == public['public_rules']
    capability = next(row for row in session.functions.catalog().functions if row.name == 'get_task')
    assert capability.available and not capability.advances_physics and not capability.consumes_action_attempt

    def query(expected_phase):
        before = (session.sequence, session.raw['sim_time_s'], session.clock.action_attempts,
                  len(workers[-1].calls))
        response = session.functions.invoke({'function': 'get_task', 'arguments': {},
            'request_id': 'query/' + expected_phase})
        assert response.ok
        result = response.result.model_dump(mode='json', by_alias=True)
        assert result == session.task_requirements()
        assert result['referee']['phase'] == expected_phase
        assert set(result) == set(public) | {'referee', 'episode'}
        assert set(result['referee']) == {'schema', 'state', 'phase', 'reason_codes', 'terminal', 'recoverable'}
        encoded = json.dumps(result)
        for forbidden in (PRIVATE_MARKER, 'task_state', 'continuous_on_s', DATA_ROOT, 'private_counters'):
            assert forbidden not in encoded
        assert before == (session.sequence, session.raw['sim_time_s'], session.clock.action_attempts,
                          len(workers[-1].calls))
        return result
    assert query('place') == ready['task']
    session.submit(action(session)); join_action(session)
    query('settling')
    session.finish()
    assert query('settling')['episode'] == {'terminal': True, 'outcome': 'cancelled'}


@pytest.mark.parametrize('change', [
    lambda public: public.update(private_pose=PRIVATE_MARKER),
    lambda public: public['public_rules']['thresholds'].update(dwell_s=.001),
    lambda public: public.update(public_rules_sha256='0' * 64),
])
def test_opt_in_rejects_noncanonical_scene_projection_before_launch(
        runner, frozen_inputs, monkeypatch, change):
    install_projection(monkeypatch, frozen_inputs, projection_change=change)
    make, _, workers = runner
    with pytest.raises(ValueError, match='projection differs'):
        make().create(request())
    assert workers == []


@pytest.mark.parametrize('change', [
    lambda initial: initial.pop('task'),
    lambda initial: initial['task'].update(private_pose=PRIVATE_MARKER),
    lambda initial: initial['task']['public_rules']['thresholds'].update(dwell_s=.001),
])
def test_opt_in_rejects_missing_or_different_worker_initial_task(
        runner, frozen_inputs, monkeypatch, change):
    install_projection(monkeypatch, frozen_inputs, worker_change=change)
    make, _, workers = runner
    with pytest.raises(service.ServiceError, match='targets differ') as failure:
        make().create(request())
    assert failure.value.status == 503
    assert len(workers) == 1 and workers[-1].closed


@pytest.mark.parametrize('state', ['success', 'task_failure', 'evaluator_invalid', 'safety_stop'])
def test_opt_in_never_admits_terminal_reset(runner, frozen_inputs, monkeypatch, state):
    install_projection(monkeypatch, frozen_inputs, initial_state=state)
    make, _, workers = runner
    with pytest.raises(service.ServiceError, match='initial unscored reset'):
        make().create(request())
    assert workers[-1].closed


def test_opt_in_rejects_bad_phase_before_ack_and_preserves_last_public_state(
        runner, frozen_inputs, monkeypatch):
    install_projection(monkeypatch, frozen_inputs)
    make, _, _ = runner
    def corrupt(worker):
        worker.score = score()
        worker.score['referee']['phase'] = PRIVATE_MARKER
    session, _ = session_for(make(on_step=corrupt))
    previous = session.task_requirements()['referee']
    session.submit(action(session)); join_action(session)
    assert session.result.outcome == 'infrastructure_error'
    assert session.sequence == 0 and session.raw['sim_time_s'] == 0.
    assert session.task_requirements()['referee'] == previous
    assert PRIVATE_MARKER not in json.dumps(session.task_requirements())


def test_legacy_manipulation_stays_static_and_does_not_advertise_get_task(
        runner, frozen_inputs, monkeypatch):
    install_projection(monkeypatch, frozen_inputs, opt_in=False)
    make, _, workers = runner
    session, ready = session_for(make())
    assert session.referee_contract is None and session.referee_state is None
    assert ready['task'] == workers[-1].initial['task'] == session.task_requirements()
    response = session.functions.invoke({'function': 'get_task', 'arguments': {},
                                        'request_id': 'legacy/query'})
    assert not response.ok and response.error.code == 'unsupported_capability'
    assert session.sequence == 0 and session.clock.action_attempts == 0
