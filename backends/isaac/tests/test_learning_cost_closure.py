"""Closed-cost boundary fixtures; no models, network or native simulation."""
from copy import deepcopy
import json
import shutil

import pytest

from agent.model_client.learning import runner
from agent.model_client.learning.receipts import ReceiptError, collect_effort, collect_match_effort
from bench.paths import load_json
from test_learning_receipts import fixture, match_fixture, save
from test_learning_study import config, prepare


def damage(report, change, *, match=False):
    report = deepcopy(report)
    result = report['result']['episode'] if match else report['result']
    if change == 'missing_result':
        report['result'] = None
    elif change == 'missing_elapsed':
        result.pop('elapsed_sim_s')
    elif change == 'wrong_episode':
        result['spec']['episode_id'] = 'another_episode'
    elif change == 'finalization_error':
        report['finalization_error'] = {'error_category': 'transport_timeout'}
    return report


@pytest.mark.parametrize('match', [False, True], ids=['trial', 'match'])
@pytest.mark.parametrize('change', ['missing_result', 'missing_elapsed', 'wrong_episode', 'finalization_error'])
def test_unsealed_effort_is_unknown_instead_of_a_zero_cost(fixture, change, match):
    report = match_fixture(fixture) if match else fixture.report
    save(fixture.agent/'run_report.json', damage(report, change, match=match))
    original_receipt = (fixture.agent/'single/decisions/000000/model_response.json').read_bytes()
    collector = collect_match_effort if match else collect_effort
    with pytest.raises(ReceiptError):
        collector(fixture.agent)
    assert (fixture.agent/'single/decisions/000000/model_response.json').read_bytes() == original_receipt


@pytest.mark.parametrize('match', [False, True], ids=['trial', 'match'])
def test_closed_cancelled_episode_can_have_a_real_zero_clock(fixture, match):
    report = deepcopy(match_fixture(fixture) if match else fixture.report)
    result = report['result']['episode'] if match else report['result']
    result['elapsed_sim_s'] = 0.
    save(fixture.agent/'run_report.json', report)
    effort = (collect_match_effort if match else collect_effort)(fixture.agent)
    assert effort.native_sim_s == 0.
    assert effort.model_requests == (2 if match else 1)


@pytest.mark.parametrize('change', ['missing_result', 'missing_elapsed', 'wrong_episode', 'finalization_error'])
def test_real_effort_collector_leaves_unsealed_trial_unaccounted(fixture, tmp_path, monkeypatch, change):
    cfg = config(tmp_path)
    monkeypatch.setattr(runner, 'prepare_study_plan', lambda *a, **kw: prepare(practice_episodes=2))
    calls = []
    def episode(run_config, **kwargs):
        calls.append(run_config)
        shutil.copytree(fixture.agent, run_config.output)
        report = damage(fixture.report, change)
        save(run_config.output/'run_report.json', report)
        return report
    with pytest.raises(ReceiptError):
        runner.run_study(cfg, source_verification={'manifest_sha256': 'c'*64}, episode_runner=episode)
    report = load_json(cfg.output/'study_report.json')
    assert len(calls) == 1
    assert report['status'] == 'stopped_error'
    assert report['cost_accounting_complete'] is False
    assert report['cost_scope'] == 'accounted_attempts_only; unaccounted costs are unknown, not zero'
    assert report['cost']['attempts'] == 0
    assert report['unaccounted_attempts'] == [{'attempt_id': 'practice_0000', 'phase': 'practice',
        'learners': ['player_0'], 'output_relative': 'player_0/practice_0000', 'episode_id': 'ep1'}]
    assert not any(row['kind'] == 'trial_end' for row in
                   (load_json(path) for path in (cfg.output/'ledger').glob('*.json')))
    assert not list(cfg.output.glob('*/freeze_manifest.json'))
    preserved = load_json(cfg.output/'player_0/practice_0000/single/decisions/000000/model_response.json')
    assert preserved['metadata']['attempts'] == 1
    assert preserved['metadata']['usage']['input_tokens'] == 10


@pytest.mark.parametrize('created', [False, True], ids=['before_creation', 'lost_creation_response'])
def test_missing_episode_id_only_has_zero_native_cost_before_creation(tmp_path, created):
    report = dict(episode_id=None, result=None, model_requests={'single': []},
                  function_calls={}, physical_attempts={})
    save(tmp_path/'run_report.json', report)
    if created:
        save(tmp_path/'create_request.json', {'task_id': 'polyomino_territory_practice'})
        with pytest.raises(ReceiptError, match='native_effort_unavailable_without_episode_identity'):
            collect_effort(tmp_path)
    else:
        effort = collect_effort(tmp_path)
        assert effort.native_sim_s == 0. and effort.reset_attempts == 0


@pytest.mark.parametrize('match', [False, True], ids=['trial', 'match'])
@pytest.mark.parametrize('remaining', ['request', 'events', 'decision_counter'])
def test_missing_model_completion_never_becomes_zero_usage(fixture, match, remaining):
    report = deepcopy(match_fixture(fixture) if match else fixture.report)
    player = 'player_0' if match else 'single'
    directory = fixture.agent/player/'decisions/000000'
    (directory/'model_response.json').unlink()
    report['model_requests'][player] = []
    if remaining == 'request':
        save(directory/'model_request.json', {'messages': []})
    elif remaining == 'events':
        (directory/'model_api_events.jsonl').write_text('{"event":"model_request_started"}\n')
    else:
        report['decisions'] = {player: 1}
    save(fixture.agent/'run_report.json', report)
    with pytest.raises(ReceiptError, match='model_request_completion_unavailable|model_decision_receipt_count_mismatch'):
        (collect_match_effort if match else collect_effort)(fixture.agent)
