"""CPU receipt/condition checks; no provider requests or native simulation."""
import json
from types import SimpleNamespace

import pytest

from agent.model_client.learning.audit_study import _run_condition
from agent.model_client.learning.curriculum import StudyBudget
from agent.model_client.learning.model_conditions import RETRY_FIELDS
from agent.model_client.learning.receipts import _Reader, _model_effort, ReceiptError
from bench.paths import load_json
from bench.protocol.schemas import Budgets
from test_learning_receipts import fixture, formal_fixture, reseal_formal_result


RETRY = dict(retry_policy='gateway_transient_backoff_v1', max_retries=None,
    retry_initial_delay_s=5., retry_max_delay_s=60., retry_jitter='equal_jitter',
    retry_after_max_s=86400., retryable_http_statuses=[429, 502, 503, 504],
    retryable_transport_errors=['ConnectError', 'ConnectTimeout'],
    timeout_scope='per_attempt_with_absolute_deadline')
MODEL = dict(provider='gateway', model='fixture-model', reasoning_effort='max',
    protocol='chat/completions', base_url='https://fixture.invalid/v1',
    endpoint='https://fixture.invalid/v1/chat/completions')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepared(tmp_path, *, retry=True, match=False, error=False, attempts=2):
    budget = StudyBudget()
    plan = SimpleNamespace(budget=budget, model=MODEL | (RETRY if retry else {}))
    keys = ('player_0', 'player_1') if match else ('single',)
    decisions = budget.match_max_decisions_per_player if match else budget.max_decisions_per_episode
    config = dict(budgets=Budgets(sim_time_s=budget.match_sim_seconds if match else budget.sim_seconds_per_episode,
        wall_time_s=budget.match_wall_seconds if match else budget.wall_seconds_per_episode,
        max_actions=budget.match_max_actions if match else budget.max_actions_per_episode,
        max_model_calls=decisions*len(keys), max_action_sim_s=budget.max_action_sim_seconds).model_dump(mode='json'),
        max_decisions_per_player=decisions, max_function_calls_per_player=budget.max_function_calls_per_episode,
        history_mode='current_with_recall', local_geometry_math={'enabled': True}, visual_input='rgbd',
        models={key: {k: v for k, v in plan.model.items() if k != 'provider'} for key in keys})
    if match:
        config['validation_target'] = dict(commits_per_player=budget.match_commits_per_player,
            source='verified_public_operation_committed_events', pass_counts=False, game_completion_claimed=False)
    save(tmp_path/'trial/agent_config.json', config)
    completions = {}
    for key in keys:
        metadata = dict(attempts=attempts, max_output_tokens=None,
            model=MODEL['model'], requested_model=MODEL['model'], reasoning_effort='max',
            requested_reasoning_effort='max', protocol=MODEL['protocol'],
            usage=dict(input_tokens=10, output_tokens=20, output_tokens_details={'reasoning_tokens': 15}))
        if retry:
            metadata.update(RETRY)
        if error:
            metadata = {k:v for k,v in metadata.items() if k in RETRY_FIELDS or k == 'attempts'}
            metadata.update(error_category='http_status', http_status=429)
        folder = tmp_path/'trial'/key/'decisions/0000'
        completion = folder/('model_error.json' if error else 'model_response.json')
        save(completion, metadata if error else dict(metadata=metadata, text='CPU fixture'))
        completions[key] = completion
        if retry:
            events = [dict(event='model_request_started', retry_policy=RETRY['retry_policy']) for _ in range(attempts)]
            events.append(dict(event='model_request_failed' if error else 'model_request_completed',
                               retry_policy=RETRY['retry_policy'], attempts=attempts))
            (folder/'model_api_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return SimpleNamespace(root=tmp_path, plan=plan, config=config, completions=completions,
                           reader=_Reader(tmp_path, 'fixture', {}))


@pytest.mark.parametrize('match', [False, True])
@pytest.mark.parametrize('error', [False, True])
def test_preregistered_retry_checked_on_success_and_error_completions(tmp_path, match, error):
    f = prepared(tmp_path, match=match, error=error)
    _run_condition(f.reader, 'trial', f.plan, match=match)
    key = 'player_1' if match else 'single'
    value = load_json(f.completions[key])
    (value if error else value['metadata'])['max_retries'] = 0
    save(f.completions[key], value)
    with pytest.raises(ReceiptError, match='model_retry_policy_changed'):
        _run_condition(f.reader, 'trial', f.plan, match=match)


@pytest.mark.parametrize('field', RETRY_FIELDS)
def test_client_cannot_omit_a_frozen_retry_condition(tmp_path, field):
    f = prepared(tmp_path)
    del f.config['models']['single'][field]
    save(tmp_path/'trial/agent_config.json', f.config)
    with pytest.raises(ReceiptError, match='model_retry_policy_changed'):
        _run_condition(f.reader, 'trial', f.plan)


@pytest.mark.parametrize('mutation', ['missing_events', 'changed_policy', 'omitted_attempt',
                                      'wrong_model', 'wrong_effort', 'wrong_cap', 'logical_budget'])
def test_actual_retry_evidence_cannot_be_relabelled_by_the_plan(tmp_path, mutation):
    f = prepared(tmp_path)
    path = f.completions['single']; events = path.with_name('model_api_events.jsonl')
    if mutation == 'missing_events':
        events.unlink()
    elif mutation in ('changed_policy', 'omitted_attempt'):
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        if mutation == 'changed_policy': rows[0]['retry_policy'] = 'legacy_gateway_v1'
        else: rows.pop(0)
        events.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    elif mutation == 'logical_budget':
        f.config['max_decisions_per_player'] = 33
        save(tmp_path/'trial/agent_config.json', f.config)
    else:
        value = load_json(path)
        key, value_to_set = {'wrong_model': ('model', 'other'), 'wrong_effort': ('reasoning_effort', 'low'),
                             'wrong_cap': ('max_output_tokens', 4096)}[mutation]
        value['metadata'][key] = value_to_set; save(path, value)
    with pytest.raises(ReceiptError):
        _run_condition(f.reader, 'trial', f.plan)


def test_legacy_evidence_retains_original_routing_only_contract(tmp_path):
    f = prepared(tmp_path, retry=False, attempts=1)
    f.config['models']['single']['max_retries'] = 0
    save(tmp_path/'trial/agent_config.json', f.config)
    _run_condition(f.reader, 'trial', f.plan)
    f.config['models']['single'].update(RETRY)
    save(tmp_path/'trial/agent_config.json', f.config)
    with pytest.raises(ReceiptError, match='unplanned_gateway_retry_policy'):
        _run_condition(f.reader, 'trial', f.plan)


@pytest.mark.parametrize('error,attempts', [(False, 2), (True, 3), (True, 0)])
def test_failed_http_attempts_count_without_repeating_final_usage(tmp_path, error, attempts):
    f = prepared(tmp_path, error=error, attempts=attempts)
    _run_condition(f.reader, 'trial', f.plan)
    value = load_json(f.completions['single'])
    report = dict(model_requests={'single': [] if error else [value['metadata']]}, decisions={'single': 1})
    cost = _model_effort(_Reader(tmp_path/'trial', 'trial', {}), report)
    assert cost['model_logical_calls'] == 1 and cost['model_requests'] == attempts
    assert cost['input_tokens'] == cost['output_tokens'] == cost['reasoning_tokens'] == (None if attempts else 0)


def test_reasoning_tokens_remain_a_separate_subset(tmp_path):
    f = prepared(tmp_path, attempts=1)
    metadata = load_json(f.completions['single'])['metadata']
    cost = _model_effort(_Reader(tmp_path/'trial', 'trial', {}),
                         dict(model_requests={'single': [metadata]}, decisions={'single': 1}))
    assert cost['output_tokens'] == 20 and cost['reasoning_tokens'] == 15


@pytest.mark.parametrize('event_state', ['absent', 'empty', 'not_sent'])
def test_zero_send_deadline_does_not_require_an_invented_http_event(tmp_path, event_state):
    f = prepared(tmp_path, error=True, attempts=0)
    path = f.completions['single']
    value = load_json(path) | dict(error_category='request_budget_exhausted', http_status=None)
    save(path, value)
    events = path.with_name('model_api_events.jsonl')
    if event_state == 'absent': events.unlink()
    elif event_state == 'empty': events.write_text('')
    else:
        events.write_text(''.join(json.dumps(dict(event=name, retry_policy=RETRY['retry_policy']))+'\n'
            for name in ('model_request_started', 'model_request_not_sent')))
    _run_condition(f.reader, 'trial', f.plan)
    cost = _model_effort(_Reader(tmp_path/'trial', 'trial', {}),
                         dict(model_requests={'single': []}, decisions={'single': 1}))
    assert cost['model_requests'] == 0 and cost['model_logical_calls'] == 1


@pytest.mark.parametrize('logical', [32, 33])
def test_32_logical_calls_may_have_64_actual_requests_but_not_a_33rd_decision(tmp_path, logical):
    f = prepared(tmp_path, attempts=2)
    folder = f.completions['single'].parent
    response = load_json(f.completions['single'])
    event_bytes = (folder/'model_api_events.jsonl').read_bytes()
    for index in range(1, logical):
        target = folder.parent/f'{index:04d}'
        save(target/'model_response.json', response)
        (target/'model_api_events.jsonl').write_bytes(event_bytes)
    if logical == 32:
        _run_condition(f.reader, 'trial', f.plan)
    else:
        with pytest.raises(ReceiptError, match='model_logical_budget_exceeded'):
            _run_condition(f.reader, 'trial', f.plan)
    cost = _model_effort(_Reader(tmp_path/'trial', 'trial', {}),
        dict(model_requests={'single': [response['metadata']]*logical}, decisions={'single': logical}))
    assert cost['model_logical_calls'] == logical and cost['model_requests'] == logical*2
    assert cost['output_tokens'] is None


@pytest.mark.parametrize('mutation', [None, 'client', 'response', 'event'])
def test_formal_reader_checks_actual_retry_policy_independently(fixture, monkeypatch, mutation):
    from agent.model_client.learning.skills import SkillLibrary
    from agent.model_client.learning.match_receipts import collect_frozen_match_receipts
    create = SkillLibrary.create
    def retry_library(experiment_id, **kwargs):
        # This modifies only a synthetic fixture before sealing its origin.
        kwargs['contract']['model'].update(RETRY)
        return create(experiment_id, **kwargs)
    monkeypatch.setattr(SkillLibrary, 'create', staticmethod(retry_library))
    args = formal_fixture(fixture, monkeypatch)
    report = load_json(fixture.agent/'run_report.json')
    for player in ('player_0', 'player_1'):
        metadata = report['model_requests'][player][0] | RETRY
        report['model_requests'][player] = [metadata]
        folder = fixture.agent/player/'decisions/000000'
        save(folder/'model_response.json', dict(metadata=metadata, text='CPU fixture'))
        (folder/'model_api_events.jsonl').write_text(json.dumps(dict(
            event='model_request_started', retry_policy=RETRY['retry_policy']))+'\n')
    save(fixture.agent/'run_report.json', report)
    if mutation == 'client':
        config = load_json(fixture.agent/'agent_config.json')
        config['models']['player_1']['max_retries'] = 0
        reseal_formal_result(fixture, config=config)
    elif mutation == 'response':
        report['model_requests']['player_1'][0]['retry_max_delay_s'] = 1
        save(fixture.agent/'run_report.json', report)
        save(fixture.agent/'player_1/decisions/000000/model_response.json',
             dict(metadata=report['model_requests']['player_1'][0], text='CPU fixture'))
    elif mutation == 'event':
        path = fixture.agent/'player_1/decisions/000000/model_api_events.jsonl'
        path.write_text(json.dumps(dict(event='model_request_started', retry_policy='legacy_gateway_v1'))+'\n')
    diagnostic = {}
    result = collect_frozen_match_receipts(**args, diagnostics=diagnostic)
    if mutation is None:
        assert result is not None, diagnostic
    else:
        assert result is None
        assert diagnostic['reason'] == ('model_retry_event_policy_changed' if mutation == 'event' else 'model_retry_policy_changed')
