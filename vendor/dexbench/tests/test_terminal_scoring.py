"""Continuous results stay public only after termination, with explicit validity."""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from bench.isaacsim.release import digest, load_evaluator, task_inputs
from bench.paths import load_json
from bench.protocol import Budgets, EpisodeResult, aggregate_results
from bench.protocol.task_referee import PublicTaskRequirements, project_task_rules
from bench.protocol.task_scoring import (TerminalScoringContract, TerminalTaskScoring,
                                         project_terminal_scoring)
from bench.runtime.server import create_app
from scenes.runtime import terminal_scoring_contract
from scenes.workrooms.catalog import task_definition
from test_agent_bench_contracts import result, spec
from test_agent_service import (PRIVATE_MARKER, FakeWorker, action, join_action, request,
                                runner, frozen_inputs, session_for)
from test_scene_function_runner import mechanical_service
from test_task_referee_contract import install_contract, score as coarse_score


def declaration():
    # Deliberately generic: the bench supports metrics unrelated to tower height.
    return {'schema': 'terminal_task_scoring_contract.v1', 'mode': 'continuous',
            'objective': 'minimize', 'score_unit': 's', 'metric_units': {'duration_s': 's'}}


def receipt(**updates):
    return {'score': 2.5, 'score_unit': 's', 'score_valid': True,
            'duration_s': 2.5, 'private_objects': PRIVATE_MARKER, **updates}


def project(**updates):
    return project_terminal_scoring(receipt(**updates), declaration(),
                                    outcome='sim_timeout', physics_valid=True)


def test_only_declared_finite_numbers_are_exported():
    value = project()
    assert value.score == 2.5 and value.score_valid
    assert value.metrics == {'duration_s': 2.5}
    assert PRIVATE_MARKER not in value.model_dump_json()
    assert project_terminal_scoring(receipt(), None, outcome='cancelled', physics_valid=True) is None


@pytest.mark.parametrize('updates', [
    {'score': None}, {'score': True}, {'score': '2.5'}, {'score': float('nan')},
    {'score': float('inf')}, {'score_unit': 'm'}, {'score_valid': 'true'},
    {'score_valid': False}, {'duration_s': PRIVATE_MARKER}, {'duration_s': float('inf')},
    {'status': 'invalid'},
])
def test_bad_numeric_evidence_never_certifies_a_score(updates):
    value = project(**updates)
    assert not value.score_valid
    assert PRIVATE_MARKER not in value.model_dump_json()


@pytest.mark.parametrize('outcome,physics', [
    ('evaluator_invalid', True), ('infrastructure_error', True),
    ('safety_stop', True), ('policy_error', True), ('invalid_action', True),
    ('sim_timeout', False),
])
def test_faults_keep_previous_numeric_record_but_revoke_validity(outcome, physics):
    value = project_terminal_scoring(receipt(), declaration(), outcome=outcome, physics_valid=physics)
    assert value.score == 2.5 and value.metrics == {'duration_s': 2.5}
    assert not value.score_valid


@pytest.mark.parametrize('updates', [
    {'score': None}, {'score': float('nan')}, {'score': True},
    {'score_valid': 1}, {'metrics': {'private_pose': 1.}},
    {'metrics': {}}, {'private_pose': PRIVATE_MARKER},
])
def test_typed_result_rejects_malformed_valid_receipts(updates):
    payload = project().model_dump(mode='json') | updates
    with pytest.raises(ValidationError):
        TerminalTaskScoring.model_validate(payload)


def test_invalid_receipt_can_preserve_known_metrics_without_inventing_missing_values():
    value = project(duration_s=PRIVATE_MARKER)
    assert value.score == 2.5 and value.metrics == {} and not value.score_valid
    assert TerminalTaskScoring.model_validate(value.model_dump(mode='json')) == value
    with pytest.raises(ValidationError):
        TerminalScoringContract.model_validate(declaration() | {'private_state': PRIVATE_MARKER})


def frozen_height():
    definition = task_definition('living_room', 'tower_building')
    root = Path(definition['source_release']['path'])
    assert digest(root/'release_manifest.json') == definition['source_release']['manifest_sha256']
    task = load_json(root/'task.json')
    return root, task


def test_frozen_v5_compatibility_is_exact_and_keeps_release_and_rules_unchanged():
    root, task = frozen_height()
    before = {name: digest(root/name) for name in ('task.json', 'rules.json', 'release_manifest.json')}
    original = deepcopy(task)
    contract = terminal_scoring_contract(task)
    assert contract['score_unit'] == 'm' and contract['objective'] == 'maximize'
    assert terminal_scoring_contract(task_inputs(root, 'tower_building', seed=17)[2]) == contract
    public = project_task_rules(task, scoring_contract=contract)
    assert public['scoring_contract'] == contract
    assert public['public_rules_sha256'] == project_task_rules(task)['public_rules_sha256']
    assert task == original
    changed = deepcopy(task)
    changed['evaluation']['physics_dt_s'] = .001
    assert terminal_scoring_contract(changed) is None
    assert terminal_scoring_contract({'id': 'tower_building'}) is None
    assert before == {name: digest(root/name) for name in before}


def install_numeric_contract(monkeypatch, frozen_inputs):
    from bench.runtime import service
    install_contract(monkeypatch, frozen_inputs)
    original_inputs = service.task_inputs
    def with_scoring(*args):
        values = original_inputs(*args)
        values[2]['terminal_scoring_contract'] = declaration()
        return values
    monkeypatch.setattr(service, 'task_inputs', with_scoring)
    original_init = FakeWorker.__init__
    def init(worker, **kwargs):
        original_init(worker, **kwargs)
        worker.score.update(receipt(score=0., duration_s=0.))
        worker.initial['score'] = deepcopy(worker.score)
    monkeypatch.setattr(FakeWorker, '__init__', init)


def test_late_ack_preserves_only_the_last_budget_eligible_score(runner, frozen_inputs, monkeypatch):
    install_numeric_contract(monkeypatch, frozen_inputs)
    make, clock, workers = runner
    session, _ = session_for(make(), budgets=Budgets(wall_time_s=1.))
    workers[-1].on_step = lambda worker: worker.score.update(receipt(score=2., duration_s=2.))
    session.submit(action(session)); join_action(session)
    def late(worker):
        worker.score.update(receipt(score=99., duration_s=99.))
        clock.value = 1.01
    workers[-1].on_step = late
    session.submit(action(session, name='late')); join_action(session)
    assert session.result.outcome == 'wall_timeout'
    assert session.score['score'] == 99.
    assert session.result.scoring.score == 2. and session.result.scoring.score_valid
    evidence = load_json(session.directory/'private/evaluator_evidence.json')
    assert evidence['terminal_scoring_receipt']['score'] == 2.
    assert session.result.evaluator_evidence_sha256 == digest(session.directory/'private/evaluator_evidence.json')


def test_nonfinite_ack_score_cannot_break_finalization_or_keep_validity(runner, frozen_inputs, monkeypatch):
    install_numeric_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner
    session, _ = session_for(make())
    workers[-1].on_step = lambda worker: worker.score.update(receipt(score=float('nan')))
    session.submit(action(session)); join_action(session)
    assert session.result.outcome == 'infrastructure_error'
    assert not session.result.scoring.score_valid
    assert session.result.scoring.score == 0. and session.sequence == 0


@pytest.mark.parametrize('state', ['evaluator_invalid', 'safety_stop'])
def test_late_evidence_fault_preserves_budget_record_but_invalidates_it(
        runner, frozen_inputs, monkeypatch, state):
    install_numeric_contract(monkeypatch, frozen_inputs)
    make, clock, workers = runner
    session, _ = session_for(make(), budgets=Budgets(wall_time_s=1.))
    workers[-1].on_step = lambda worker: worker.score.update(receipt(score=2., duration_s=2.))
    session.submit(action(session)); join_action(session)
    def late_invalid(worker):
        worker.score = coarse_score(state) | receipt(score=99., duration_s=99., score_valid=False)
        clock.value = 1.01
    workers[-1].on_step = late_invalid
    session.submit(action(session, name='late_invalid')); join_action(session)
    assert session.result.outcome == 'wall_timeout'
    assert session.result.scoring.score == 2.
    assert not session.result.scoring.score_valid
    public = PublicTaskRequirements.model_validate(session.task_requirements())
    assert public.referee.state == state and not public.scoring.score_valid


@pytest.mark.parametrize('offset,expected', [(1e-12, 9.), (1e-7, 0.)])
def test_numeric_receipt_uses_the_episode_clock_simulation_boundary(
        runner, frozen_inputs, monkeypatch, offset, expected):
    install_numeric_contract(monkeypatch, frozen_inputs)
    make, _, workers = runner
    session, _ = session_for(make(), budgets=Budgets(sim_time_s=.02))
    def endpoint(worker):
        worker.score.update(receipt(score=9., duration_s=9.))
        worker.raw['sim_time_s'] += offset
    workers[-1].on_step = endpoint
    session.submit(action(session)); join_action(session)
    assert session.result.outcome == 'sim_timeout'
    assert session.result.scoring.score == expected


def height_receipt(evaluator, height, *, status='incomplete'):
    # A verified frozen evaluator supplies the adapter receipt. Native contact
    # qualification is outside these CPU public-result plumbing tests.
    return evaluator._result(status, 'height_score_active', reference_result={
        'score': height, 'score_unit': 'm', 'score_valid': status != 'invalid',
        'current_height_m': height, 'best_height_m': height,
        'best_height_cm': height*100., 'private_canary': PRIVATE_MARKER})


@pytest.mark.parametrize('height', [.025, .300])
@pytest.mark.parametrize('finish_kind', ['cancelled', 'sim_timeout', 'action_budget', 'evaluator_invalid', 'audit_invalid'])
def test_frozen_height_scores_survive_public_http_terminal_results(tmp_path, monkeypatch, height, finish_kind):
    api, workers, verified = mechanical_service(tmp_path, 'tower_building', 'living_room')
    root, task = frozen_height()
    evaluator = load_evaluator(root)(task)
    budgets = Budgets(sim_time_s=.02 if finish_kind == 'sim_timeout' else 120.,
                      max_actions=1 if finish_kind == 'action_budget' else 10)
    with TestClient(create_app(service=api)) as client:
        listed = client.get('/tasks').json()[0]
        ready_response = client.post('/episodes', json=request(
            task_id='tower_building', budgets=budgets).model_dump(mode='json'))
        assert ready_response.status_code == 201, ready_response.text
        ready = ready_response.json()
        session = api.get(ready['episode_id'])
        prefix = f"/episodes/{ready['episode_id']}"
        assert listed['scoring_contract'] == ready['task']['scoring_contract']
        assert 'scoring' not in listed and 'scoring' not in ready['task']
        assert client.get(prefix+'/result').status_code == 409
        worker = workers[-1]
        previous = worker.on_step
        def stepped(worker):
            previous(worker)
            worker.score = height_receipt(evaluator, height,
                status='invalid' if finish_kind == 'evaluator_invalid' else 'incomplete')
        worker.on_step = stepped
        command = client.post(prefix+'/call', json={
            'function': 'hold', 'request_id': 'height/hold', 'arguments': {
                'duration_s': .02, 'action_id': 'hold1', 'observation_sequence': 0}})
        assert command.status_code == 202, command.text
        join_action(session)
        if finish_kind in ('cancelled', 'action_budget', 'audit_invalid'):
            online = client.get(prefix+'/task').json()
            assert online['referee']['phase'] == 'height_recorded'
            assert 'scoring' not in online  # No online numeric feedback.
            if finish_kind == 'action_budget':
                rejected = client.post(prefix+'/call', json={
                    'function': 'hold', 'request_id': 'height/second', 'arguments': {
                        'duration_s': .02, 'action_id': 'hold2', 'observation_sequence': session.sequence}})
                assert rejected.status_code == 409
            elif finish_kind == 'audit_invalid':
                # Final native audit can invalidate a previously valid receipt.
                from bench.runtime import service
                session.metadata['native_isaacsim_app'] = True
                monkeypatch.setattr(service, 'audit_native_episode',
                                    lambda *args: {'physics_valid': False, 'status': 'failed'})
                client.post(prefix+'/finish')
            else:
                client.post(prefix+'/finish')
        response = client.get(prefix+'/result')
        assert response.status_code == 200, response.text
        terminal = response.json()
        numeric = terminal['scoring']
        assert numeric['score'] == height and numeric['score_unit'] == 'm'
        assert numeric['metrics'] == {'current_height_m': height, 'best_height_m': height,
                                     'best_height_cm': height*100.}
        assert numeric['score_valid'] == (finish_kind not in ('evaluator_invalid', 'audit_invalid'))
        assert terminal['outcome'] == ('cancelled' if finish_kind == 'audit_invalid' else finish_kind)
        queried = client.get(prefix+'/task').json()
        assert queried['scoring'] == numeric
        assert queried['episode']['terminal']
        assert PublicTaskRequirements.model_validate(queried).scoring.score == height
        invoked = client.post(prefix+'/call', json={
            'function': 'get_task', 'request_id': 'height/final', 'arguments': {}})
        assert invoked.status_code == 200 and invoked.json()['result']['scoring'] == numeric
        assert PRIVATE_MARKER not in json.dumps([listed, ready, terminal, queried, invoked.json()])
        assert 'private_canary' not in json.dumps(numeric)
        recorded = load_json(session.directory/'agent/result.json')
        assert recorded['scoring'] == numeric
        invalid_online = deepcopy(queried)
        invalid_online['episode'] = {'terminal': False, 'outcome': None}
        with pytest.raises(ValidationError):
            PublicTaskRequirements.model_validate(invalid_online)
        drifted = deepcopy(queried)
        drifted['scoring_contract']['score_unit'] = 'cm'
        with pytest.raises(ValidationError):
            PublicTaskRequirements.model_validate(drifted)


def continuous_result(index, value=2.5, **updates):
    scoring = project(score=value, duration_s=value)
    return result(spec=spec(episode_id=f'continuous{index}', seed=index),
        outcome='sim_timeout', evaluator_evidence_sha256='a'*64, scoring=scoring, **updates)


def test_continuous_aggregate_reports_scores_without_binary_failure_denominator():
    rows = [continuous_result(1, 2.), continuous_result(2, 4.)]
    summary = aggregate_results(rows)['cohorts'][0]['summary']
    assert summary['scored_episodes'] == 2
    assert summary['score_mean'] == 3. and summary['score_best'] == 2.
    assert summary['success_rate'] is None and summary['successes'] is None
    assert summary['penalized_sim_time_s'] is None
    assert summary['outcome_counts'] == {'sim_timeout': 2}


def test_continuous_invalid_score_is_auditable_but_excluded_from_aggregates():
    valid = continuous_result(1, 2.)
    invalid = continuous_result(2, 40.).model_dump(mode='json')
    invalid['scoring']['score_valid'] = False
    invalid['physics_valid'] = False
    summary = aggregate_results([valid, invalid])['cohorts'][0]['summary']
    assert summary['score_mean'] == 2. and summary['scored_episodes'] == 1
    assert summary['invalid_score_episodes'] == 1 and not summary['leaderboard_eligible']
    with pytest.raises(ValidationError):
        continuous_result(3, physics_valid=False)


def test_binary_history_and_different_continuous_contracts_never_pool():
    first = continuous_result(1)
    second = continuous_result(2).model_dump(mode='json')
    second['scoring']['objective'] = 'maximize'
    third = continuous_result(3).model_dump(mode='json')
    third['scoring']['score_unit'] = 'ms'
    old = result(spec=spec(episode_id='legacy', seed=4), outcome='sim_timeout')
    report = aggregate_results([first, second, third, old])
    assert len(report['cohorts']) == 4
    binary = next(c['summary'] for c in report['cohorts'] if c['episode_ids'] == ['legacy'])
    assert binary['success_rate'] == 0.
    assert 'scoring' not in old.model_dump(mode='json')
    assert EpisodeResult.model_validate(old.model_dump(mode='json')).scoring is None
