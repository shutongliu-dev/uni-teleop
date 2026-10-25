"""Keep infrastructure health separate from authoritative task results.

Historical reports are interpreted in memory; their archived bytes are never
rewritten. In particular, an error followed by cleanup cancellation is not a
negative task label.
"""
from __future__ import annotations


OUTCOME_POLICY = 'independent_run_and_task_outcomes_v2'
FAILURE_OUTCOMES = frozenset({'task_failure', 'sim_timeout', 'wall_timeout',
    'action_budget', 'model_budget', 'invalid_action', 'policy_error', 'safety_stop'})
NORMAL_ENDINGS = frozenset({'client_action_budget', 'client_decision_budget',
    'client_wall_deadline', 'client_output_budget', 'model_requested_stop'})
INFRA_CATEGORIES = frozenset({'request_budget_exhausted', 'transport_timeout',
    'transport_error', 'invalid_json', 'invalid_response', 'empty_response',
    'model_mismatch', 'invalid_finish_reason'})


def error_status(error_type, metadata=None):
    """Classify finite typed metadata, never exception prose or all HTTP errors."""
    metadata = metadata if isinstance(metadata, dict) else {}
    if error_type == 'KeyboardInterrupt':
        return 'interrupted'
    if error_type == 'TaskSetupError':
        return 'configuration_error'
    if error_type in ('ModelGatewayError', 'BenchHTTPError', 'FunctionRejected'):
        category = metadata.get('error_category')
        status = metadata.get('http_status')
        if category == 'output_budget_exhausted':
            return 'agent_budget'
        if isinstance(category, str) and category in INFRA_CATEGORIES:
            return 'infra_error'
        if category == 'http_status' or type(status) is int:
            if type(status) is int and (status in (408, 429) or 500 <= status <= 599):
                return 'infra_error'
            if status in (401, 403):
                return 'configuration_error'
            if type(status) is int and 400 <= status <= 499:
                return 'agent_error'
        return 'unknown_error'
    return 'agent_error' if error_type else 'completed'


def error_record(error):
    """Allow only bounded structured diagnostics into durable run records."""
    category = getattr(error, 'category', 'unknown')
    status = getattr(error, 'status_code', None)
    return {'error_type': type(error).__name__,
            'error_category': category if isinstance(category, str) and category in INFRA_CATEGORIES | {
                'http_status', 'output_budget_exhausted'} else 'unknown',
            'http_status': status if type(status) is int and 100 <= status <= 599 else None}


def _continuous_outcome(report, result, run_status):
    """Preserve public measurements without inventing a binary success label."""
    from bench.protocol.task_scoring import SCORABLE_OUTCOMES, TerminalTaskScoring

    try:
        scoring = TerminalTaskScoring.model_validate(result['scoring'])
    except (ValueError, TypeError):
        return {'outcome_policy': OUTCOME_POLICY, 'run_status': 'infra_error',
                'task_outcome': 'unknown', 'task_success': None,
                'task_outcome_reason': 'invalid_terminal_scoring',
                'task_scoring_mode': 'continuous', 'task_score': None,
                'task_score_unit': None, 'task_score_valid': False,
                'result_valid_for_scoring': False}
    native = result.get('outcome')
    normal_cancel = native == 'cancelled' and (
        report.get('result_source') in ('server_result', 'session_result')
        or report.get('termination') in NORMAL_ENDINGS | {
            'server_finished', 'server_finished_during_model_call'})
    valid = bool(scoring.score_valid and result.get('physics_valid') is True
                 and run_status in ('completed', 'agent_budget')
                 and not report.get('action_recovery_pending') and not report.get('finalization_error')
                 and (native in SCORABLE_OUTCOMES - {'cancelled'} or normal_cancel))
    return {'outcome_policy': OUTCOME_POLICY, 'run_status': run_status,
            'task_outcome': 'unknown', 'task_success': None,
            'task_outcome_reason': 'continuous_scoring' if valid else 'continuous_score_not_evaluable',
            'task_scoring_mode': scoring.mode, 'task_score': scoring.score,
            'task_score_unit': scoring.score_unit, 'task_score_valid': valid,
            'result_valid_for_scoring': valid}


def classify_report(report):
    """Derive tri-state task labels without trusting a legacy false default."""
    result = report.get('result')
    result = result if isinstance(result, dict) else {}
    termination = report.get('termination')
    error_type = report.get('error_type')
    metadata = report.get('run_error') or report.get('model_error')
    run_status = error_status(error_type, metadata)
    # An ordinary exhausted episode budget remains an assessed agent outcome.
    # Its timeout diagnostic does not independently establish an API outage.
    if not error_type and termination in NORMAL_ENDINGS:
        run_status = 'completed'
    finalization_error = report.get('finalization_error')
    finalization_error = finalization_error if isinstance(finalization_error, dict) else None
    if finalization_error and (not error_type or str(error_type).startswith('Finalization')):
        run_status = error_status(finalization_error.get('error_type'), finalization_error)
    if result.get('outcome') in ('infrastructure_error', 'evaluator_invalid'):
        run_status = 'infra_error'
    # Offline aggregation must retain explicit evidence rejection. A later
    # cleanup success cannot establish the identity of the requested experiment.
    invalid_result = report.get('result_validation_error') is not None
    invalid_ready = (report.get('ready_validated') is False
                     and (report.get('episode_id') is not None or bool(result)))
    if invalid_result or invalid_ready:
        if invalid_result or run_status == 'completed':
            run_status = 'infra_error'
        classified = (_continuous_outcome(report, result, run_status)
                      if result.get('scoring') is not None else {})
        classified.update(outcome_policy=OUTCOME_POLICY, run_status=run_status,
                          task_outcome='unknown', task_success=None,
                          task_outcome_reason=('invalid_terminal_result' if invalid_result
                                               else 'invalid_creation_response'),
                          task_score_valid=False, result_valid_for_scoring=False)
        return classified
    if result.get('scoring') is not None:
        return _continuous_outcome(report, result, run_status)
    outcome, reason = 'unknown', 'no_authoritative_terminal_result'
    native = result.get('outcome')
    if result and result.get('physics_valid') is not True:
        reason = 'physics_not_validated'
    elif native == 'success':
        outcome, reason = 'success', 'server_evaluator'
    elif native in ('cancelled', 'wall_timeout') and report.get('action_recovery_pending') is True:
        reason = 'physical_action_execution_unresolved'
    elif (native == 'wall_timeout' and report.get('result_source') == 'cleanup_finish'
          and run_status == 'infra_error'):
        reason = 'cleanup_wall_timeout_after_infrastructure_error'
    elif (native == 'wall_timeout' and error_type == 'ModelGatewayError'
          and run_status == 'infra_error'):
        # The server watchdog can expire while the client waits/retries a model
        # request. Fetching that timeout via /result does not make the outage an
        # assessed task failure. Independent success/task_failure remain valid.
        reason = 'wall_timeout_during_model_api_interruption'
    elif native in FAILURE_OUTCOMES:
        outcome, reason = 'failure', 'server_terminal'
    elif native == 'cancelled':
        # New reports identify cleanup results explicitly. Older reports with
        # an error cannot establish whether cancellation preceded that error.
        explicit_terminal = report.get('result_source') in ('server_result', 'session_result')
        if explicit_terminal or (termination in NORMAL_ENDINGS and (
                not error_type or run_status == 'agent_budget')):
            outcome, reason = 'failure', 'server_cancellation'
        elif not error_type and not finalization_error and termination in (
                'server_finished', 'server_finished_during_model_call'):
            outcome, reason = 'failure', 'server_cancellation'
        else:
            reason = 'cleanup_cancellation_or_unknown_cause'
    elif native in ('infrastructure_error', 'evaluator_invalid'):
        reason = native
    return {'outcome_policy': OUTCOME_POLICY, 'run_status': run_status,
            'task_outcome': outcome,
            'task_success': None if outcome == 'unknown' else outcome == 'success',
            'task_outcome_reason': reason}
