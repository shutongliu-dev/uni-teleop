"""Continuous achievements are not binary failures or outage-derived scores."""
from copy import deepcopy

import pytest

from agent.model_client.outcomes import classify_report


def report_for(height=.3, outcome='sim_timeout'):
    return {
        'termination': 'server_finished', 'result_source': 'server_result',
        'result': {'outcome': outcome, 'physics_valid': True, 'scoring': {
            'schema': 'terminal_task_scoring.v1', 'mode': 'continuous', 'objective': 'maximize',
            'score': height, 'score_unit': 'm', 'score_valid': True,
            'metrics': {'current_height_m': 0., 'best_height_m': height, 'best_height_cm': 100 * height},
            'metric_units': {'current_height_m': 'm', 'best_height_m': 'm', 'best_height_cm': 'cm'}}}}


@pytest.mark.parametrize('height', [0., .025, .3])
@pytest.mark.parametrize('outcome', ['sim_timeout', 'wall_timeout', 'action_budget', 'model_budget', 'cancelled'])
def test_valid_continuous_terminal_preserves_distinct_scores_without_binary_failure(height, outcome):
    report = report_for(height, outcome)
    original = deepcopy(report)
    result = classify_report(report)
    assert result['task_score'] == height and result['task_score_unit'] == 'm'
    assert result['task_score_valid'] and result['result_valid_for_scoring']
    assert result['task_success'] is None and result['task_outcome'] == 'unknown'
    assert result['task_outcome_reason'] == 'continuous_scoring'
    assert report == original


def test_voluntary_stop_retains_continuous_score():
    report = report_for(outcome='cancelled')
    report.update(termination='model_requested_stop', result_source='cleanup_finish')
    assert classify_report(report)['task_score_valid']


def test_normal_output_budget_retains_score_but_failed_cleanup_does_not_certify_it():
    report = report_for(outcome='cancelled')
    report.update(termination='client_output_budget', result_source='cleanup_finish',
                  error_type='ModelGatewayError',
                  run_error={'error_category': 'output_budget_exhausted'})
    result = classify_report(report)
    assert result['run_status'] == 'agent_budget' and result['task_score_valid']
    report['finalization_error'] = {'error_type': 'BenchHTTPError', 'error_category': 'transport_error'}
    assert not classify_report(report)['task_score_valid']


@pytest.mark.parametrize('cause', ['model_outage', 'cleanup_outage', 'pending', 'physics_invalid',
                                  'score_invalid', 'evaluator_invalid', 'safety_stop', 'unknown_cancel'])
def test_interrupted_or_invalid_score_is_retained_only_for_audit(cause):
    report = report_for(outcome='cancelled')
    if cause == 'model_outage':
        report.update(error_type='ModelGatewayError', run_error={'error_category': 'transport_timeout'})
    elif cause == 'cleanup_outage':
        report['finalization_error'] = {'error_type': 'BenchHTTPError', 'error_category': 'transport_error'}
    elif cause == 'pending':
        report['action_recovery_pending'] = True
    elif cause == 'physics_invalid':
        report['result']['physics_valid'] = False
    elif cause == 'score_invalid':
        report['result']['scoring']['score_valid'] = False
    elif cause in ('evaluator_invalid', 'safety_stop'):
        report['result']['outcome'] = cause
    else:
        report.update(termination='client_error', result_source='cleanup_finish')
    result = classify_report(report)
    assert result['task_score'] == .3  # The original achievement is not erased.
    assert not result['task_score_valid'] and not result['result_valid_for_scoring']
    assert result['task_outcome'] == 'unknown' and result['task_success'] is None


@pytest.mark.parametrize('change', [
    {'score': float('nan')}, {'score': None}, {'score': '0.3'}, {'score': True},
    {'mode': 'binary'}, {'private_pose': [0, 0, 0]},
])
def test_malformed_terminal_scoring_never_falls_back_to_failure(change):
    report = report_for()
    report['result']['scoring'].update(change)
    result = classify_report(report)
    assert result['run_status'] == 'infra_error' and not result['task_score_valid']
    assert result['task_outcome'] == 'unknown' and result['task_success'] is None
    assert result['task_score'] is None
