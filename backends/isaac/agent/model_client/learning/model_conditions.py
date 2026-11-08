"""Read-only checks of preregistered routing and actual gateway retry receipts."""
from .assessment import content_sha256
from .journal import strict_json_loads
from .receipts import _require


ROUTING_FIELDS = ('provider', 'model', 'reasoning_effort', 'base_url', 'endpoint', 'protocol')
RETRY_FIELDS = ('retry_policy', 'max_retries', 'retry_initial_delay_s', 'retry_max_delay_s',
                'retry_jitter', 'retry_after_max_s', 'retryable_http_statuses',
                'retryable_transport_errors', 'timeout_scope')
GATEWAY_POLICY = 'gateway_transient_backoff_v1'


def retry_condition(expected):
    """Legacy plans retain their original routing-only comparison."""
    fields = set(expected) - set(ROUTING_FIELDS)
    if not fields:
        return None
    _require(fields == set(RETRY_FIELDS) and expected.get('provider') == 'gateway'
             and expected.get('retry_policy') == GATEWAY_POLICY,
             'model_retry_condition_not_preregistered')
    maximum = expected['max_retries']
    _require(maximum is None or type(maximum) is int and maximum >= 0,
             'invalid_preregistered_model_retry_limit')
    return {key: expected[key] for key in RETRY_FIELDS}


def check_retry_metadata(expected, actual):
    policy = retry_condition(expected)
    if policy is None:
        _require(actual.get('retry_policy') != GATEWAY_POLICY, 'unplanned_gateway_retry_policy')
        return
    _require(all(key in actual for key in RETRY_FIELDS)
             and content_sha256({key: actual[key] for key in RETRY_FIELDS}) == content_sha256(policy),
             'model_retry_policy_changed')


def check_model_condition(expected, actual, *, reason):
    _require(all(isinstance(expected.get(key), str) and expected[key] for key in ROUTING_FIELDS),
             'model_routing_not_preregistered')
    projected = {key: actual.get(key) for key in ROUTING_FIELDS}
    if projected['provider'] is None and projected['protocol'] == 'chat/completions':
        projected['provider'] = 'gateway'
    _require(projected == {key: expected[key] for key in ROUTING_FIELDS}, reason)
    check_retry_metadata(expected, actual)


def check_retry_receipts(reader, relative, player, expected):
    """New policy checks both successful and failed logical-call completions.

    Old records lack these fields and event streams. Their original receipt
    and accounting checks remain authoritative; no new evidence is invented.
    """
    policy = retry_condition(expected)
    if policy is None:
        return
    directory = reader.root / relative / player / 'decisions'
    for folder in sorted(directory.glob('*')):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        response = folder / 'model_response.json'
        error = folder / 'model_error.json'
        if not response.exists() and not error.exists():
            _require(not (folder / 'model_request.json').exists()
                     and not (folder / 'model_api_events.jsonl').exists(),
                     'model_request_completion_unavailable')
            continue
        _require(response.exists() != error.exists(), 'ambiguous_model_completion')
        value = reader.json(str((response if response.exists() else error).relative_to(reader.root)))
        metadata = value['metadata'] if response.exists() else value
        check_retry_metadata(expected, metadata)
        attempts = metadata.get('attempts')
        _require(type(attempts) is int and attempts >= 0
                 and (policy['max_retries'] is None or attempts <= policy['max_retries'] + 1),
                 'model_retry_attempt_limit_changed')
        if response.exists():
            _require(attempts > 0 and metadata.get('model') == metadata.get('requested_model') == expected['model']
                     and metadata.get('reasoning_effort') == metadata.get('requested_reasoning_effort') == expected['reasoning_effort']
                     and metadata.get('protocol') == expected['protocol'], 'model_response_condition_changed')
        events_path = folder / 'model_api_events.jsonl'
        # The absolute deadline can expire before the first started event.
        # Only its explicit zero-send error can legitimately have no stream.
        zero_send = error.exists() and attempts == 0 and metadata.get('error_category') == 'request_budget_exhausted'
        _require(events_path.is_file() or zero_send, 'model_retry_events_missing')
        events = ([strict_json_loads(line) for line in reader.raw(str(events_path.relative_to(reader.root))).splitlines()]
                  if events_path.is_file() else [])
        _require((events or zero_send) and all(event.get('retry_policy') == policy['retry_policy'] for event in events),
                 'model_retry_event_policy_changed')
        sent = sum(event.get('event') == 'model_request_started' for event in events)
        sent -= sum(event.get('event') == 'model_request_not_sent' for event in events)
        _require(sent == attempts, 'model_event_attempt_mismatch')
