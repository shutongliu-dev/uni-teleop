"""Rejected experiment identities stay unscored when reports are reread."""
from copy import deepcopy

import pytest

from agent.model_client.outcomes import classify_report
from test_continuous_agent_outcomes import report_for
from test_model_suite import continuous_contract, continuous_result, environment, fake_runner


REJECTED = (
    {'result_validation_error': {'error_type': 'BenchHTTPError',
                                 'error_category': 'invalid_response'}},
    {'ready_validated': False, 'episode_id': 'created_episode'},
)


@pytest.mark.parametrize('rejected', REJECTED)
@pytest.mark.parametrize('continuous', [False, True])
def test_cleanup_success_cannot_rehabilitate_rejected_identity_on_reread(rejected, continuous):
    report = report_for() if continuous else {
        'result': {'outcome': 'success', 'physics_valid': True}}
    report.update(rejected, termination='model_requested_stop', result_source='cleanup_finish')
    original = deepcopy(report)
    classified = classify_report(report)
    assert classified['run_status'] == 'infra_error'
    assert classified['task_outcome'] == 'unknown' and classified['task_success'] is None
    assert not classified['result_valid_for_scoring']
    if continuous:
        assert classified['task_score'] == .3  # Available for audit only.
        assert not classified['task_score_valid']
    assert report == original
    assert classify_report({**report, **classified}) == classified


@pytest.mark.parametrize('primary', [
    {'error_type': 'ValueError', 'run_error': {'error_category': 'unknown'}},
    {'error_type': 'ModelGatewayError',
     'run_error': {'error_category': 'http_status', 'http_status': 401}},
    {'error_type': 'ModelGatewayError',
     'run_error': {'error_category': 'output_budget_exhausted'}},
])
def test_terminal_rejection_remains_infra_after_an_earlier_error(primary):
    report = {**primary, **REJECTED[0], 'result': None,
              'task_score_valid': True, 'result_valid_for_scoring': True}
    original = deepcopy(report)
    classified = classify_report(report)
    assert classified['run_status'] == 'infra_error'
    assert classified['task_success'] is None and not classified['result_valid_for_scoring']
    assert not classified['task_score_valid']
    assert report == original  # Keep the primary failure separately.


@pytest.mark.parametrize('primary,expected', [
    ({'error_type': 'TaskSetupError'}, 'configuration_error'),
    ({'error_type': 'KeyboardInterrupt'}, 'interrupted'),
    ({'error_type': 'BenchHTTPError',
      'run_error': {'error_category': 'http_status', 'http_status': 401}}, 'configuration_error'),
])
@pytest.mark.parametrize('created', [False, True])
def test_unvalidated_ready_does_not_replace_known_setup_or_interrupt_cause(primary, expected, created):
    report = {**primary, 'ready_validated': False,
              'episode_id': 'created_episode' if created else None,
              'result': {'outcome': 'success', 'physics_valid': True} if created else None}
    classified = classify_report(report)
    assert classified['run_status'] == expected
    assert classified['task_success'] is None
    if created:
        assert not classified['result_valid_for_scoring']


@pytest.mark.parametrize('flags', [{}, {'result_validation_error': None, 'ready_validated': True}])
@pytest.mark.parametrize('continuous', [False, True])
def test_accepted_and_historical_reports_keep_their_existing_labels(flags, continuous):
    report = report_for() if continuous else {
        'result': {'outcome': 'success', 'physics_valid': True}}
    baseline = classify_report(report)
    assert classify_report({**report, **flags}) == baseline
    if continuous:
        assert baseline['task_score_valid']
    else:
        assert baseline['task_success'] is True


@pytest.mark.parametrize('marker', ['result_validation_error', 'ready_validated'])
@pytest.mark.parametrize('continuous', [False, True])
def test_suite_reread_excludes_rejected_identity_even_with_successful_archived_evidence(
        environment, marker, continuous):
    from agent.model_client import suite

    _, inventory, make = environment
    if continuous:
        inventory['tasks'][0]['scoring_contract'] = continuous_contract()
    path, _ = make(modes=['joints_action'])
    execute, _ = fake_runner(environment, outcome='sim_timeout' if continuous else 'success',
                            terminal_scoring=continuous_result(.3) if continuous else None)

    def run(args):
        report = execute(args)
        report[marker] = REJECTED[0][marker] if marker == 'result_validation_error' else False
        # The cached label is not trusted when reconstructing the suite.
        report.update(task_success=True, task_score_valid=True, result_valid_for_scoring=True)
        suite.save(args.output / 'run_report.json', report)
        suite.save(args.output / 'artifact_hashes.json', {
            str(p.relative_to(args.output)): suite.digest(p)
            for p in sorted(args.output.rglob('*'))
            if p.is_file() and p.name != 'artifact_hashes.json'})
        return report

    initial = suite.run_plan(path, run_one=run)
    reread = suite.report_plan(path)
    for report in (initial, reread):
        row, = report['slots']
        assert row['status'] == 'finished' and not row['artifact_errors']
        assert row['run_status'] == 'infra_error' and row['task_success'] is None
        assert row['task_outcome'] == 'unknown' and not row['valid_recorded_episode']
        assert not row['task_score_valid'] and not row['result_valid_for_scoring']
        assert report['cohorts'][0]['valid_recorded_episodes'] == 0
        if continuous:
            assert row['task_score'] == .3
            assert report['cohorts'][0]['continuous_scores'] == []
