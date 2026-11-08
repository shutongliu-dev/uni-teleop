"""Exact public-text projection; no model or native simulation is invoked."""
from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.context_projection import VERSION, project_context
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.runner import canonical_hash
from agent.model_client.transport import png_data_url
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker
from test_function_runner import ScriptedModel


def user(value):
    return {'role': 'user', 'content': json.dumps(value, ensure_ascii=False)}


def fixture():
    task = {'public_rules': {'rules': ['no overlap'], 'threshold': .003},
            'public_rules_sha256': 'a'*64, 'rules': {'size': 6}, 'geometry': {'pitch': .052}}
    state = {**deepcopy(task), 'board_game': {'turn_index': 2, 'phase': 'awaiting_physical_commit'},
             'observation': {'sequence': 24, 'contacts': {'left': {'impulse': .00123}},
                             'rules': deepcopy(task['rules'])}, 'player': {'player_id': 'player_0'}}
    feedback = {'ok': False, 'function': 'move_geometry', 'result': deepcopy(state),
                'error': {'code': 'insufficient_duration', 'minimum_duration_s': 4.76},
                'contact_history': {'status': 'not_sampled', 'sampled_substeps': 0}}
    current = {'task': task, 'state': deepcopy(state), 'last_feedback': deepcopy(feedback),
               'depth_observation': {'samples_m': [[1.2345, None]]}, 'host_contract': {},
               'budgets': {'actions': 4}}
    return current, feedback


def test_only_exact_static_duplicates_and_adjacent_last_feedback_are_removed():
    current, feedback = fixture()
    prior = {'state': deepcopy(current['state']), 'last_feedback': deepcopy(feedback),
             'depth_observation': {'samples_m': [[2.3456, None]]}}
    history = [user({'tool_feedback': feedback}), user(prior),
               {'role': 'assistant', 'content': '{"function":"observe","arguments":{}}'},
               user({'tool_feedback': feedback})]
    original_history, original_current = deepcopy(history), deepcopy(current)
    projected, now, receipt = project_context(history, current)
    assert history == original_history and current == original_current
    assert len(projected) == len(history)
    assert [r['role'] for r in projected] == [r['role'] for r in history]
    assert projected[2] == history[2]
    assert now['task'] == current['task']
    assert 'last_feedback' not in now and 'last_feedback' not in json.loads(projected[1]['content'])
    for state in (now['state'], json.loads(projected[1]['content'])['state']):
        assert not {'public_rules', 'public_rules_sha256', 'rules', 'geometry'} & state.keys()
        assert state['observation'] == current['state']['observation']
        assert state['board_game'] == current['state']['board_game']
    for index in (0, 3):
        kept = json.loads(projected[index]['content'])['tool_feedback']
        assert kept['error'] == feedback['error']
        assert kept['contact_history'] == feedback['contact_history']
        assert kept['result']['observation'] == feedback['result']['observation']
    assert now['depth_observation'] == current['depth_observation']
    assert json.loads(projected[1]['content'])['depth_observation'] == prior['depth_observation']
    assert receipt['schema'] == VERSION
    assert any(row['path'] == 'current/last_feedback' for row in receipt['removed'])


@pytest.mark.parametrize('different', [True, 1.0, '1', {'changed': 1}])
def test_type_or_value_different_rules_are_never_removed(different):
    current, _ = fixture()
    current['task']['public_rules'] = {'limit': 1}
    current['state']['public_rules'] = {'limit': different}
    _, now, receipt = project_context([], current)
    assert now['state']['public_rules'] == current['state']['public_rules']
    assert not any(row['path'] == 'current/state/public_rules' for row in receipt['removed'])


@pytest.mark.parametrize('field', ['sampled_substeps', 'minimum_duration_s', 'status'])
def test_similar_but_different_feedback_keeps_all_dynamic_and_error_fields(field):
    current, feedback = fixture()
    if field == 'sampled_substeps': current['last_feedback']['contact_history'][field] = 1
    elif field == 'status': current['last_feedback']['contact_history'][field] = 'available'
    else: current['last_feedback']['error'][field] = 4.78
    history, now, _ = project_context([user({'tool_feedback': feedback})], current)
    assert 'last_feedback' in now
    assert now['last_feedback']['contact_history'] == current['last_feedback']['contact_history']
    assert now['last_feedback']['error'] == current['last_feedback']['error']
    assert json.loads(history[0]['content'])['tool_feedback']['error'] == feedback['error']


def test_boolean_and_number_feedback_are_not_conflated():
    current, feedback = fixture()
    feedback['ok'] = 1
    current['last_feedback']['ok'] = True
    _, now, _ = project_context([user({'tool_feedback': feedback})], current)
    assert now['last_feedback']['ok'] is True


@pytest.mark.parametrize('intervening', [
    user({'public_game_events': [{'sequence': 3, 'type': 'card_revealed',
                                'details': {'card_id': 'card_00', 'symbol': 'pair_2'}}]}),
    {'role': 'assistant', 'content': '{"stop":true}'}, user({'other': 'public update'})])
def test_events_or_other_messages_break_feedback_adjacency_without_any_reordering(intervening):
    current, feedback = fixture()
    original = [user({'tool_feedback': feedback}), deepcopy(intervening)]
    history, now, _ = project_context(original, current)
    assert history[1] == original[1]
    assert 'last_feedback' in now
    assert len(history) == 2


def test_missing_canonical_fields_and_non_context_messages_remain_verbatim():
    current, feedback = fixture()
    current['task'] = {'task_id': 'fixture'}
    history = [user({'tool_feedback': feedback}), {'role': 'user', 'content': 'plain public note'}]
    projected, now, receipt = project_context(history, current)
    assert projected == history and now == current and receipt['removed'] == []


def test_actual_runner_records_exact_projected_messages_and_wire_hash(tmp_path):
    release = Path(task_definition('card_room', 'tic_tac_toe_3x3')['source_release']['path'])
    def factory(**kwargs): return FakeBoardWorker(**kwargs)
    api = AgentService(tmp_path/'episodes', release=release, registry=default_registry(release),
                       worker_factory=factory)
    p0 = ScriptedModel([{'function': 'observe', 'arguments': {}}, {'stop': True}])
    p1 = ScriptedModel([])
    output = tmp_path/'run'
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task1', output, max_decisions=3),
                     bench=BenchHTTPClient('http://testserver', client=http),
                     models={'player_0': p0, 'player_1': p1})
    assert report['error_type'] is None and report['finalization_error'] is None, report
    assert not p1.turns and len(p0.turns) == 2
    for index, sent in enumerate(p0.turns):
        folder = output/'player_0/decisions'/f'{index:04d}'
        record = load_json(folder/'model_request.json')
        projection = load_json(folder/'input_projection.json')
        assert projection['schema'] == VERSION
        rebuilt = deepcopy(record['messages'])
        for message in rebuilt:
            if not isinstance(message['content'], list): continue
            for i, block in enumerate(message['content']):
                if block['type'] == 'recorded_image':
                    message['content'][i] = {'type': 'image_url', 'image_url': {
                        'url': png_data_url((output/block['path']).read_bytes())}}
        assert rebuilt == sent
        assert canonical_hash(sent) == record['expanded_messages_sha256']
    original_feedback = load_json(output/'player_0/decisions/0000/execution.json')
    last_current = json.loads(p0.turns[1][-1]['content'][0]['text'])
    previous = json.loads(p0.turns[1][-2]['content'])['tool_feedback']
    assert 'last_feedback' not in last_current
    assert previous['result'] == original_feedback['result']  # observe returns the dynamic observation itself
    assert load_json(output/'agent_config.json')['input_projection'] == VERSION
