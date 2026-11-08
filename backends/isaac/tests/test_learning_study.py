"""Scheduler transition tests; scripted publications and unknown physical evidence.

These never count as model authorship, native success or a real RSI run.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agent.model_client.learning import runner
from agent.model_client.learning.assessment import Effort, TrialRecord, content_sha256
from agent.model_client.learning.curriculum import StudyBudget, plan_from_cases
from agent.model_client.learning.receipts import ReceiptError
from agent.model_client.player_models import PlayerModelBinding
from bench.paths import load_json
from test_learning_curriculum import prepare, inputs


def config(tmp_path):
    (tmp_path / 'private_profile.json').write_text(json.dumps({
        'base_url': 'https://fixture.invalid/v1', 'api_key': 'PRIVATE_PROFILE_CANARY'}))
    binding = {'provider': 'gateway', 'model': 'scripted_fixture', 'reasoning_effort': 'max',
               'credentials': str(tmp_path / 'private_profile.json')}
    profiles = tmp_path / 'models.json'
    profiles.write_text(json.dumps(dict.fromkeys(('player_0', 'player_1'), binding)))
    return runner.StudyConfig(gym_url='http://fixture', formal_url='http://fixture',
        gym_release=tmp_path, gym_episode_root=tmp_path / 'episodes', formal_episode_root=tmp_path / 'formal',
        output=tmp_path / 'study', player_models=profiles,
        budget=StudyBudget(practice_episodes=2, validation_batches=1))


def scripted_publication(library, index):
    revision = library.write_author('skills/observe.py', f'def run(api, args, state):\n    return {index}\n',
                                   expected_revision=library.author_revision())
    catalog = {'skills': [{'skill_id': 'observe', 'module': 'skills/observe.py', 'entrypoint': 'run',
        'description': 'CPU scheduling fixture', 'tags': [], 'input_schema': {}, 'output_schema': {}}]}
    revision = library.write_author('catalog.json', json.dumps(catalog), expected_revision=revision)
    return library.publish(revision, transaction_id=f'fixture-{index}', reason='scripted transition fixture')


def unknown_record(plan, slot, **kw):
    return TrialRecord(**slot.model_dump(), **{k: getattr(plan, k) for k in
        ('gym_id', 'parent_task_id', 'parent_release_sha256', 'public_rules_sha256',
         'condition_sha256', 'source_manifest_sha256')},
        **{k: kw[k] for k in ('candidate_event_id', 'bundle_hash', 'attempt_id',
                             'attempt_index', 'receipt_sequence', 'phase')},
        episode_id=kw['attempt_id'], execution_status='completed', outcome='unknown', physics_valid=None,
        result_valid=False, evidence_source='synthetic', temporary_skill_changes=False,
        fresh_context=True, author_history_visible=kw['phase']=='practice', cost=Effort())


@pytest.mark.parametrize('retry_policy', [None, 'gateway_transient_backoff_v1'])
def test_independent_libraries_fresh_validation_and_explicit_unqualified_freeze(tmp_path, monkeypatch, retry_policy):
    cfg = config(tmp_path)
    cfg.model_retry_policy = retry_policy
    plan = prepare(practice_episodes=2)
    monkeypatch.setattr(runner, 'prepare_study_plan', lambda *a, **k: plan)
    calls, practice_revisions = [], {}
    def episode(config, *, learning_bindings, learning_feedback=None, learning_trial=None, **kwargs):
        config.output.mkdir(parents=True)
        calls.append((config, learning_bindings, learning_feedback))
        assert config.model_retry_policy == retry_policy and config.model_retries is None
        if 'single' in learning_bindings:
            pin = learning_bindings['single']
            assert config.max_output_tokens is None and config.reasoning_effort == 'max'
            assert config.sim_budget == 45. and config.wall_budget is None
            assert learning_trial['phase'] == pin.phase
            if pin.phase == 'practice':
                key = pin.library.path
                index = practice_revisions.get(key, 0) + 1
                practice_revisions[key] = index
                scripted_publication(pin.library, index)
            else:
                assert pin.phase == 'validation' and learning_feedback is None
                assert pin.candidate_event_id and pin.bundle_sha256
        else:
            assert config.sim_budget == 120. and config.wall_budget is None
            assert set(learning_bindings) == {'player_0', 'player_1'}
            assert len({p.library.path for p in learning_bindings.values()}) == 2
            assert all(p.phase == 'evaluation' and p.candidate_event_id for p in learning_bindings.values())
            assert learning_feedback is None
        return {'episode_id': config.output.name, 'result': {'physics_valid': True},
                'run_status': 'completed', 'result_valid_for_scoring': False}
    result = runner.run_study(cfg, source_verification={'manifest_sha256': 'c'*64}, episode_runner=episode,
        receipt_collector=unknown_record, effort_collector=lambda *a, **k: Effort(),
        match_effort_collector=lambda *a, **k: Effort(model_requests=3),
        feedback_projector=lambda row, report: {'outcome': 'unknown', 'native_evidence_verified': False})
    assert len(calls) == 9  # Two independent (2 practice + 2 validation) learners and one match.
    assert len(practice_revisions) == 2 and set(practice_revisions.values()) == {2}
    assert result['status'] == 'stages_executed'
    assert result['closed_loop_verified'] is False and result['formal_match_completed'] is False
    assert result['cost']['totals']['model_requests'] == 3
    assert result['cost_accounting_complete'] is True and result['unaccounted_attempts'] == []
    assert result['cost_scope'] == 'all_started_attempts'
    for player in ('player_0', 'player_1'):
        frozen = result['learners'][player]['freeze']
        assert frozen['selection_mode'] == 'budget_fallback'
        assert frozen['qualified_for_declared_curriculum'] is False
        assert frozen['candidate_event_id'] == 'candidate-fixture-2'


def test_no_model_publication_never_gets_an_operator_fallback(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(runner, 'prepare_study_plan', lambda *a, **k: prepare(practice_episodes=2))
    def episode(config, **kwargs):
        config.output.mkdir(parents=True)
        return {'episode_id': 'fixture', 'result': {}, 'run_status': 'completed'}
    result = runner.run_study(cfg, source_verification={'manifest_sha256': 'c'*64}, episode_runner=episode,
        receipt_collector=unknown_record, effort_collector=lambda *a, **k: Effort(),
        feedback_projector=lambda *a: {'outcome': 'unknown'})
    assert result['status'] == 'no_published_candidate'
    assert not (cfg.output / 'match').exists()
    assert not list((cfg.output / 'libraries/player_0/bundles').iterdir())


@pytest.mark.parametrize('result,pending', [
    (None, {'single': 'original_action'}),
    ({'outcome': 'policy_error', 'physics_valid': True}, {}),
    ({'episode': {'outcome': 'policy_error', 'physics_valid': True}}, {}),
])
def test_pending_episode_stops_before_another_reset_or_freeze(tmp_path, monkeypatch, result, pending):
    cfg = config(tmp_path)
    monkeypatch.setattr(runner, 'prepare_study_plan', lambda *a, **k: prepare(practice_episodes=2))
    calls = []
    def episode(config, **kwargs):
        config.output.mkdir(parents=True)
        calls.append(config)
        return {'episode_id': 'fixture', 'result': result, 'action_recovery_pending': pending}
    with pytest.raises(RuntimeError, match='Unresolved episode'):
        runner.run_study(cfg, source_verification={'manifest_sha256': 'c'*64}, episode_runner=episode,
            receipt_collector=unknown_record, effort_collector=lambda *a, **k: Effort(),
            feedback_projector=lambda *a: {})
    assert len(calls) == 1
    assert load_json(cfg.output / 'study_report.json')['status'] == 'stopped_error'
    assert not list(cfg.output.glob('*/freeze_manifest.json'))


@pytest.mark.parametrize('phase', ['practice', 'validation', 'evaluation'])
@pytest.mark.parametrize('failure_at', ['episode', 'cost'])
def test_started_attempt_missing_cost_is_unknown_and_never_retried(
        tmp_path, monkeypatch, phase, failure_at):
    cfg = config(tmp_path)
    monkeypatch.setattr(runner, 'prepare_study_plan', lambda *a, **k: prepare(practice_episodes=2))
    calls, accounted = [], []
    failure_type = OSError if failure_at == 'episode' else ReceiptError

    def episode(config, *, learning_bindings, **kwargs):
        config.output.mkdir(parents=True)
        actual_phase = next(iter(learning_bindings.values())).phase
        calls.append((config.output, actual_phase))
        if failure_at == 'episode' and actual_phase == phase:
            raise OSError('PRIVATE_FAILURE_CANARY')
        if actual_phase == 'practice':
            scripted_publication(learning_bindings['single'].library, len(calls))
        return {'episode_id': config.output.name, 'result': {'physics_valid': True},
                'run_status': 'completed', 'result_valid_for_scoring': False}

    def effort(folder, **kwargs):
        assert folder == calls[-1][0]
        if failure_at == 'cost' and calls[-1][1] == phase:
            raise ReceiptError('PRIVATE_FAILURE_CANARY')
        accounted.append(folder)
        return Effort(model_requests=1)

    with pytest.raises(failure_type):
        runner.run_study(cfg, source_verification={'manifest_sha256': 'c'*64},
            episode_runner=episode, receipt_collector=unknown_record, effort_collector=effort,
            match_effort_collector=effort, feedback_projector=lambda *a: {'outcome': 'unknown'})
    expected_calls = {'practice': 1, 'validation': 3, 'evaluation': 9}[phase]
    assert len(calls) == expected_calls and len(accounted) == expected_calls-1
    report = load_json(cfg.output / 'study_report.json')
    assert report['status'] == 'stopped_error' and report['error_type'] == failure_type.__name__
    assert report['cost_accounting_complete'] is False
    assert report['cost_scope'] == 'accounted_attempts_only; unaccounted costs are unknown, not zero'
    assert report['cost']['totals']['model_requests'] == len(accounted)
    missing = report['unaccounted_attempts']
    assert len(missing) == 1 and missing[0]['phase'] == phase
    assert missing[0]['output_relative'] == calls[-1][0].relative_to(cfg.output).as_posix()
    assert missing[0]['learners'] == (['player_0', 'player_1'] if phase == 'evaluation' else ['player_0'])
    assert ('episode_id' in missing[0]) == (failure_at == 'cost')
    assert 'PRIVATE_FAILURE_CANARY' not in json.dumps(report)


@pytest.mark.parametrize('provider', ['gateway', 'micu', 'qwen'])
def test_model_condition_contains_actual_protocol_without_credentials(tmp_path, provider):
    cfg = config(tmp_path)
    profile = tmp_path/'private_profile.json'
    condition = runner.public_model_condition(PlayerModelBinding(provider, 'fixture', 'max', profile))
    protocol = 'chat/completions' if provider == 'gateway' else 'responses'
    assert condition['base_url'] == 'https://fixture.invalid/v1'
    assert condition['endpoint'] == 'https://fixture.invalid/v1/'+protocol
    assert condition['protocol'] == protocol
    assert set(condition) == {'provider', 'model', 'reasoning_effort', 'base_url', 'endpoint', 'protocol'}
    assert 'PRIVATE_PROFILE_CANARY' not in json.dumps(condition) and str(profile) not in json.dumps(condition)
    profile.write_text(json.dumps({'base_url': 'https://fixture.invalid/v1/', 'api_key': 'ROTATED_SECRET'}))
    assert runner.public_model_condition(PlayerModelBinding(provider, 'fixture', 'max', profile)) == condition


def test_explicit_gateway_recovery_is_frozen_without_changing_legacy_defaults(tmp_path):
    from agent.model_client.transport import GATEWAY_RETRY_POLICY, gateway_retry_metadata
    config(tmp_path)
    binding = PlayerModelBinding('gateway', 'fixture', 'max', tmp_path/'private_profile.json')
    legacy = runner.public_model_condition(binding)
    current = runner.public_model_condition(binding, retry_policy=GATEWAY_RETRY_POLICY)
    assert current == {**legacy, **gateway_retry_metadata(None)}
    bounded = runner.public_model_condition(binding, retry_policy=GATEWAY_RETRY_POLICY, retries=3)
    assert bounded == {**legacy, **gateway_retry_metadata(3)}
    assert content_sha256(current) != content_sha256(bounded) != content_sha256(legacy)
    with pytest.raises(ValueError, match='explicit retry policy'):
        runner.public_model_condition(binding, retries=1)
    with pytest.raises(ValueError, match='Unsupported study retry policy'):
        runner.public_model_condition(PlayerModelBinding('micu', 'fixture', 'max', binding.credentials),
                                      retry_policy=GATEWAY_RETRY_POLICY)


@pytest.mark.parametrize(('boundary', 'completed_calls'), [('practice', 1), ('validation', 2), ('evaluation', 8)])
def test_transport_change_stops_before_next_trial_or_match(tmp_path, monkeypatch, boundary, completed_calls):
    cfg = config(tmp_path)
    manifest, cases = inputs()
    def plan(_release, **kwargs):
        return plan_from_cases(manifest, cases, gym_release_sha256='d'*64, **kwargs)
    monkeypatch.setattr(runner, 'prepare_study_plan', plan)
    calls = []
    def episode(config, *, learning_bindings, **kwargs):
        config.output.mkdir(parents=True)
        calls.append(config)
        pin = learning_bindings.get('single')
        if pin and pin.phase == 'practice':
            scripted_publication(pin.library, len(calls))
        if len(calls) == completed_calls:
            (tmp_path/'private_profile.json').write_text(json.dumps({
                'base_url': 'https://changed.invalid/v1', 'api_key': 'ROTATED_SECRET'}))
        return {'episode_id': config.output.name, 'result': {'physics_valid': True},
                'run_status': 'completed', 'result_valid_for_scoring': False}
    with pytest.raises(ValueError, match='Public model transport changed'):
        runner.run_study(cfg, source_verification={'manifest_sha256': 'c'*64}, episode_runner=episode,
            receipt_collector=unknown_record, effort_collector=lambda *a, **k: Effort(model_requests=1),
            match_effort_collector=lambda *a, **k: Effort(model_requests=1),
            feedback_projector=lambda *a: {'outcome': 'unknown'})
    assert len(calls) == completed_calls
    report = load_json(cfg.output/'study_report.json')
    frozen_plan = load_json(cfg.output/'study_plan.json')
    assert report['cost_accounting_complete'] and report['unaccounted_attempts'] == []
    assert report['cost']['totals']['model_requests'] == completed_calls
    assert report['model']['base_url'] == frozen_plan['model']['base_url'] == 'https://fixture.invalid/v1'
    changed_model = {**frozen_plan['model'], 'base_url': 'https://changed.invalid/v1',
                     'endpoint': 'https://changed.invalid/v1/chat/completions'}
    changed_plan = plan(cfg.gym_release, parent_task_id=cfg.parent_task_id, model=changed_model,
        source_manifest_sha256='c'*64, budget=cfg.budget, full_curriculum=False)
    assert changed_plan.validation.condition_sha256 != frozen_plan['validation']['condition_sha256']
    assert 'PRIVATE_PROFILE_CANARY' not in json.dumps(report)+json.dumps(frozen_plan)
    assert 'ROTATED_SECRET' not in json.dumps(report)+json.dumps(frozen_plan)
    assert str(tmp_path/'private_profile.json') not in json.dumps(report)+json.dumps(frozen_plan)


def test_continuation_skips_completed_slot_keeps_feedback_and_counts_old_cost(tmp_path, monkeypatch):
    """Scheduling fixture only; the continuation module tests provenance checks."""
    cfg = config(tmp_path)
    cfg.continuation = tmp_path / 'continuation_request.json'
    cfg.continuation.write_text('{"schema":"cpu_fixture"}')
    prior = Effort(model_requests=5, native_sim_s=1.5, input_tokens=None)
    feedback = [{'assessment': {'outcome': 'unknown', 'scope': 'prior_fixture'},
                 'recent_public_executions': []}]
    manifest = {'schema': 'cpu_continuation_fixture', 'completed': ['practice_0000']}
    continued = SimpleNamespace(manifest=manifest, checkpoint_sha256=content_sha256(manifest),
        completed_slots=lambda p: ('practice_0000',) if p == 'player_0' else (),
        prior_efforts=lambda p: [prior] if p == 'player_0' else [],
        public_feedback=lambda p: list(feedback) if p == 'player_0' else [])
    def seed(checkpoint, player, library):
        assert checkpoint is continued
        return {'scope': 'CPU seed boundary fixture', 'learner': player}
    monkeypatch.setitem(sys.modules, 'agent.model_client.learning.continuation',
        SimpleNamespace(load_continuation=lambda *a: continued, seed_library=seed))
    monkeypatch.setattr(runner, 'prepare_study_plan', lambda *a, **k: prepare(practice_episodes=2))
    calls = []
    def episode(config, *, learning_bindings, learning_feedback=None, **kwargs):
        config.output.mkdir(parents=True)
        calls.append(config.output.relative_to(cfg.output).as_posix())
        pin = learning_bindings.get('single')
        if pin and pin.phase == 'practice':
            if config.output.parent.name == 'player_0':
                assert config.output.name == 'practice_0001'
                assert learning_feedback['single'][0] == feedback[0]
            scripted_publication(pin.library, len(calls))
        else:
            assert learning_feedback is None
        return {'episode_id': config.output.name, 'result': {'physics_valid': True},
                'run_status': 'completed', 'result_valid_for_scoring': False}
    summary = runner.run_study(cfg, source_verification={'manifest_sha256': 'c'*64},
        episode_runner=episode, receipt_collector=unknown_record,
        effort_collector=lambda *a, **k: Effort(model_requests=1),
        match_effort_collector=lambda *a, **k: Effort(model_requests=1),
        feedback_projector=lambda *a: {'outcome': 'unknown'})
    assert len(calls) == 8 and calls[0] == 'player_0/practice_0001'
    assert 'player_0/practice_0000' not in calls
    assert summary['cost']['totals']['model_requests'] == 8
    assert summary['inherited_cost']['totals']['model_requests'] == 5
    assert summary['cumulative_cost']['totals']['model_requests'] == 13
    assert summary['cumulative_cost']['provider_usage']['input_tokens']['complete'] is False
    assert summary['learners']['player_0']['freeze']['development_cost']['totals']['model_requests'] == 8
    assert summary['learners']['player_1']['freeze']['development_cost']['totals']['model_requests'] == 4
    assert summary['inherited_qualification'] is False
    assert summary['closed_loop_verified'] is False
    assert not (cfg.output/'player_0/practice_0000').exists()
