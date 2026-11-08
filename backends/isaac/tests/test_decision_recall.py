"""CPU public-record isolation and prompt integrity; no model/native calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
import pytest

from agent.model_client.bench_http import BenchHTTPClient
from agent.model_client.capture import save
from agent.model_client.decision_recall import DecisionArchive, RecallError
from agent.model_client.function_runner import RunConfig, run
from agent.model_client.function_session import FunctionSession
from agent.model_client.recall_projection import project_recall_history, VERSION
from agent.model_client.runner import canonical_hash
from agent.model_client.transport import png_data_url
from agent.model_client.validation_feedback import controlled_validation_details
from bench.paths import load_json
from bench.runtime.controllers import default_registry
from bench.runtime.server import create_app
from bench.runtime.service import AgentService
from scenes.workrooms.catalog import task_definition
from test_board_game_service import FakeBoardWorker
from test_function_runner import ScriptedModel
from test_function_session import Client, session


def observation(sequence=0):
    return {'protocol': 'agent_bench_v1', 'episode_id': 'ep1', 'sequence': sequence,
            'sim_time_s': sequence*.02, 'proprioception': {'joint_positions_rad': {'left_arm': .12}},
            'cameras': [{'camera_id': 'task', 'depth': 'public'}]}


def record(archive, index, execution=None):
    folder = archive.root/f'{index:04d}'
    obs = observation(index)
    execution = execution if execution is not None else {'ok': True, 'function': 'observe', 'result': obs}
    for section, value in [('observation', obs), ('execution', execution), ('depth_summary', {'task': {'grid': [[1.2345]]}})]:
        save(folder/(section+'.json'), value)
    archive.register(index)
    return obs, execution


def test_recall_exact_originals_and_own_record_hashes(tmp_path):
    archive = DecisionArchive(tmp_path/'player_0', episode_id='ep1', player_id='player_0')
    obs, execution = record(archive, 0)
    before = {p.name: p.read_bytes() for p in (archive.root/'0000').iterdir()}
    value = archive.read({'decision_index': 0, 'sections': ['observation', 'execution', 'depth_summary']})
    assert value['records']['observation']['value'] == obs
    assert value['records']['execution']['value'] == execution
    assert value['advances_physics'] is False and value['historical'] is True
    for name, data in before.items():
        assert value['records'][name[:-5]]['sha256'] == hashlib.sha256(data).hexdigest()
        assert (archive.root/'0000'/name).read_bytes() == data


@pytest.mark.parametrize('arguments', [
    {'decision_index': 0, 'sections': ['observation'], 'player_id': 'player_1'},
    {'decision_index': 0, 'sections': ['observation'], 'path': '../../private'},
    {'decision_index': '../player_1', 'sections': ['observation']},
    {'decision_index': True, 'sections': ['observation']},
    {'decision_index': 0, 'sections': ['model_response']},
    {'decision_index': 0, 'sections': ['observation', 'observation']},
])
def test_recall_rejects_player_path_type_and_section_injection(tmp_path, arguments):
    archive = DecisionArchive(tmp_path/'player_0', episode_id='ep1', player_id='player_0')
    record(archive, 0)
    with pytest.raises((ValidationError, RecallError)):
        archive.read(arguments)


def test_current_future_and_other_player_are_not_registered(tmp_path):
    own = DecisionArchive(tmp_path/'player_0', episode_id='ep1', player_id='player_0')
    peer = DecisionArchive(tmp_path/'player_1', episode_id='ep1', player_id='player_1')
    record(own, 0); record(peer, 1)
    save(own.root/'0001/observation.json', observation(1))
    with pytest.raises(RecallError, match='Only completed'):
        own.read({'decision_index': 1, 'sections': ['observation']})
    with pytest.raises(ValueError, match='exact player'):
        FunctionSession(Client(board=True), output=tmp_path/'player_0', spec={'task_id': 'task1'},
                        trial_id='trial', max_actions=3, max_calls=3, decision_archive=peer)


@pytest.mark.parametrize('change', ['content', 'symlink', 'hardlink'])
def test_changed_and_linked_records_fail_closed_without_exposing_payload(tmp_path, change):
    archive = DecisionArchive(tmp_path/'player_0', episode_id='ep1', player_id='player_0')
    record(archive, 0)
    path = archive.root/'0000/observation.json'; original = path.read_bytes()
    if change == 'content': path.write_text('{"CANARY_PRIVATE":"do not return"}')
    else:
        target = tmp_path/'private.json'; target.write_bytes(original); path.unlink()
        if change == 'symlink': path.symlink_to(target)
        else: path.hardlink_to(target)
    with pytest.raises(RecallError) as error:
        archive.read({'decision_index': 0, 'sections': ['observation']})
    assert error.value.code == 'record_unavailable' and 'CANARY_PRIVATE' not in str(error.value)


def test_oversize_record_is_explicit_error_not_partial_data(tmp_path, monkeypatch):
    from agent.model_client import decision_recall
    archive = DecisionArchive(tmp_path/'player_0', episode_id='ep1', player_id='player_0')
    record(archive, 0)
    monkeypatch.setattr(decision_recall, 'MAX_RECALL_BYTES', 10)
    with pytest.raises(RecallError) as error:
        archive.read({'decision_index': 0, 'sections': ['observation']})
    assert error.value.code == 'record_too_large'


def user(value):
    return {'role': 'user', 'content': json.dumps(value)}


def test_projection_retains_decisions_events_errors_contacts_and_current_context(tmp_path):
    archive = DecisionArchive(tmp_path/'player_0', episode_id='ep1', player_id='player_0')
    action = {'ok': True, 'function': 'move_wrist', 'result': {'action_id': 'a0', 'status': 'failed',
        'diagnostic': {'code': 'tracking_stalled', 'wrist_position_error_m': .002},
        'contact_history': {'status': 'available', 'normal_impulse_sum_ns': 23.45},
        'robot_execution_state': {'maintained_joint_names': ['left_arm']}},
        'refresh_error': {'error_category': 'transport_timeout'}}
    obs0, _ = record(archive, 0, action)
    obs1, query = record(archive, 1)
    context0 = {'decision_index': 0, 'state': {'observation': obs0, 'board_game': {'turn_index': 0}},
                'depth_observation': {'cameras': {'task': {'grid': [[1.2345]]}}}, 'last_feedback': None}
    context1 = {'decision_index': 1, 'state': {'observation': obs1, 'board_game': {'turn_index': 0}},
                'depth_observation': deepcopy(context0['depth_observation']), 'last_feedback': deepcopy(action)}
    event = user({'public_game_events': [{'cursor': 0, 'type': 'card_revealed', 'details': {'symbol': 'pair_1'}}]})
    assistants = [{'role': 'assistant', 'content': '{"function":"move_wrist","arguments":{}}'},
                  {'role': 'assistant', 'content': '{"function":"observe","arguments":{}}'}]
    history = [user(context0), assistants[0], user({'tool_feedback': action}), event,
               user(context1), assistants[1], user({'tool_feedback': query})]
    current = {'decision_index': 2, 'state': {'observation': obs1}, 'depth_observation': context1['depth_observation'],
               'last_feedback': query, 'task': {}, 'host_contract': {}}
    original_history, original_current = deepcopy(history), deepcopy(current)
    projected, now, receipt = project_recall_history(history, current, archive)
    assert history == original_history and current == original_current
    assert [m for m in projected if m['role'] == 'assistant'] == assistants
    assert projected[3] == event
    assert now['state'] == current['state'] and now['last_feedback'] == query
    assert now['depth_observation'] == current['depth_observation']
    assert json.loads(projected[2]['content'])['tool_feedback'] == action
    old = json.loads(projected[0]['content'])
    assert 'historical_observation' in old['state']['observation']
    assert old['state']['board_game'] == context0['state']['board_game']
    assert old['depth_observation']['source']['sha256'] == archive.reference(0)['records']['depth_summary']['sha256']
    assert json.loads(projected[6]['content'])['tool_feedback']['result']['historical_observation']['record']['source'] == 'decision:0001/execution'
    assert receipt['schema'] == VERSION and receipt['changes']


def test_recalled_values_available_now_but_history_keeps_receipt(tmp_path):
    archive = DecisionArchive(tmp_path/'player_0', episode_id='ep1', player_id='player_0')
    obs, _ = record(archive, 0)
    recalled = {'ok': True, 'function': 'read_decision', 'result': archive.read({'decision_index': 0, 'sections': ['observation']})}
    record(archive, 1, recalled)
    history = [user({'decision_index': 1, 'state': {'observation': observation(1)}}),
               {'role': 'assistant', 'content': '{"function":"read_decision","arguments":{}}'}, user({'tool_feedback': recalled})]
    projected, now, _ = project_recall_history(history, {'last_feedback': recalled, 'task': {}}, archive)
    assert now['last_feedback']['result']['records']['observation']['value'] == obs
    old = json.loads(projected[-1]['content'])['tool_feedback']['result']
    assert 'value' not in old['records']['observation'] and old['records']['observation']['sha256']
    assert archive.read({'decision_index': 0, 'sections': ['observation']})['records']['observation']['value'] == obs


def test_recall_counts_function_only_and_never_queries_bench(tmp_path):
    archive = DecisionArchive(tmp_path/'run', episode_id='ep1', player_id='player_0')
    record(archive, 0)
    host, client = session(tmp_path, Client(board=True), decision_archive=archive)
    def forbidden(): raise AssertionError('Recall must not refresh the Bench')
    client.state = forbidden
    before = deepcopy(client.calls)
    result = host.execute({'function': 'read_decision', 'arguments': {'decision_index': 0, 'sections': ['observation']}})
    assert result['ok'] and host.calls == 1 and host.actions == 0 and not client.posts
    assert client.calls == before
    assert load_json(tmp_path/'run/calls/000001.json') == result


def test_quaternion_error_reports_safe_field_and_constraint_without_relaxation(tmp_path):
    host, client = session(tmp_path, Client(board=True))
    result = host.execute({'function': 'move_wrist', 'arguments': {'hand': 'left', 'timeout_sim_s': 5.,
        'wrist': {'frame_id': 'world', 'position_m': [.7, -.3, 1.1],
                  'quaternion_wxyz': [.14054, -.53804, .81916, .14054]}}})
    error = result['error']
    assert error['code'] == 'invalid_arguments' and not client.posts
    assert error['validation_errors'][0]['field_path'] == ['wrist']
    assert 'unit quaternion' in error['validation_errors'][0]['reason']
    assert '.14054' not in json.dumps(error)


def test_validation_does_not_echo_unknown_field_values_or_exception_text():
    class Input(BaseModel):
        model_config = ConfigDict(extra='forbid')
        number: float
        @model_validator(mode='after')
        def fail(self):
            raise ValueError('CANARY_EXCEPTION_SECRET')
    for supplied in ({'number': 'CANARY_VALUE', 'CANARY_KEY': 1}, {'number': 1}):
        with pytest.raises(ValidationError) as caught: Input.model_validate(supplied)
        result = controlled_validation_details(caught.value, Input.model_json_schema())
        assert 'CANARY' not in json.dumps(result)


def test_intent_validation_exposes_only_public_field_locations(tmp_path):
    host, client = session(tmp_path, Client(board=True))
    result = host.execute({'intent': {'type': 'placement_intent', 'piece_id': 'black_0', 'r': 0}})
    assert result['error']['code'] == 'invalid_arguments' and not client.intents
    assert any(item['field_path'][-1] == 'q' for item in result['error']['validation_errors'])
    assert host.actions == 0


def test_local_numeric_function_uses_same_call_budget_without_http(tmp_path):
    class Number(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        number: float
    host, client = session(tmp_path, local_functions={'compute_fixture': Number},
        local_execute=lambda name, args: {'value': args['number']*2}, max_calls=1)
    client.state = lambda: (_ for _ in ()).throw(AssertionError('No Bench access'))
    assert any(t.get('function') == 'compute_fixture' for t in host._tools())
    result = host.execute({'function': 'compute_fixture', 'arguments': {'number': 2.}})
    assert result['result'] == {'value': 4.} and host.actions == 0 and not client.posts
    again = host.execute({'function': 'compute_fixture', 'arguments': {'number': 2.}})
    assert again['error']['code'] == 'budget_exhausted' and host.calls == 1


def test_actual_geometry_helpers_dispatch_locally_without_private_state(tmp_path):
    from agent.model_client.geometry_math import FUNCTION_INPUTS, execute
    host, client = session(tmp_path, local_functions=FUNCTION_INPUTS, local_execute=execute)
    client.state = lambda: (_ for _ in ()).throw(AssertionError('No Bench access'))
    result = host.execute({'function': 'compute_rotation', 'arguments': {'quaternion_wxyz': [0., 0., 0., 1.000006565]}})
    assert result['ok'] and result['result']['defined']
    assert result['result']['result']['quaternion_wxyz'] == [0., 0., 0., 1.]
    assert result['result']['advances_physics'] is False and host.calls == 1 and host.actions == 0
    rejected = host.execute({'function': 'compute_vector', 'arguments': {'operation': 'scale', 'a': [1., 0., 0.], 'scale': 1.e9}})
    assert rejected['error']['code'] == 'invalid_arguments' and not client.posts


def test_query_motion_owns_hand_injects_frame_and_sequence_without_action_budget(tmp_path):
    host, client = session(tmp_path, Client(board=True))
    tools = {t['function']: t for t in host._tools() if 'function' in t}
    properties = tools['query_motion']['arguments_schema']['properties']
    assert set(properties['hand_geometries']['properties']) == {'left'}
    assert 'geometry' not in properties and 'observation_sequence' not in properties
    args = {'hand_geometries': {'left': {'hand_posture_rad': {'left_finger': .1}}}, 'requested_duration_s': 6.}
    assert host.execute({'function': 'query_motion', 'arguments': args})['ok']
    payload = client.calls[-1]['arguments']
    assert payload['observation_sequence'] == 0 and payload['hand_geometries']['left']['fingertip_frame_id'] == 'left_hand_flange'
    assert payload['requested_duration_s'] == 6. and host.actions == 0 and not client.posts
    calls = len(client.calls)
    client.public['board_game']['current_player'] = 'player_1'
    assert host.execute({'function': 'query_motion', 'arguments': args})['error']['code'] == 'permission_denied'
    assert len(client.calls) == calls and host.actions == 0


@pytest.mark.parametrize('args', [
    {'hand_geometries': {'right': {'hand_posture_rad': {'left_finger': .1}}}},
    {'hand_geometries': {'left': {'hand_posture_rad': {'left_finger': .1}, 'fingertip_frame_id': 'right_hand_flange'}}},
])
def test_query_motion_rejects_opponent_hand_or_frame(tmp_path, args):
    host, client = session(tmp_path, Client(board=True)); before = len(client.calls)
    result = host.execute({'function': 'query_motion', 'arguments': args})
    assert result['error']['code'] == 'permission_denied' and len(client.calls) == before and host.actions == 0


@pytest.mark.parametrize('math_enabled', [False, True])
def test_real_runner_recall_and_recorded_wire_hash(tmp_path, math_enabled):
    release = Path(task_definition('card_room', 'tic_tac_toe_3x3')['source_release']['path'])
    api = AgentService(tmp_path/'episodes', release=release, registry=default_registry(release), worker_factory=FakeBoardWorker)
    decisions = [{'function': 'observe', 'arguments': {}},
        {'function': 'read_decision', 'arguments': {'decision_index': 0, 'sections': ['observation', 'execution', 'depth_summary']}},
        {'function': 'hold', 'arguments': {'duration_s': .2}}, {'stop': True}]
    if math_enabled: decisions.insert(0, {'function': 'compute_rotation', 'arguments': {'axis_xyz': [0., 0., 1.], 'angle_rad': 0.}})
    model = ScriptedModel(decisions)
    output = tmp_path/'run'
    with TestClient(create_app(service=api)) as http:
        report = run(RunConfig('http://testserver', 'task1', output, max_decisions=6, history_mode='current_with_recall', geometry_math=math_enabled),
                     bench=BenchHTTPClient('http://testserver', client=http), models={'player_0': model, 'player_1': ScriptedModel([])})
    assert report['error_type'] is None and report['finalization_error'] is None, report
    assert report['function_calls']['player_0'] == 3+math_enabled and report['physical_attempts']['player_0'] == 1
    original = load_json(output/'player_0/decisions/0000/observation.json')
    recalled = load_json(output/'player_0/decisions'/f'{1+math_enabled:04d}'/'execution.json')
    assert recalled['result']['records']['observation']['value'] == original
    for index, sent in enumerate(model.turns):
        record = load_json(output/'player_0/decisions'/f'{index:04d}'/'model_request.json')
        rebuilt = deepcopy(record['messages'])
        for message in rebuilt:
            if not isinstance(message['content'], list): continue
            for i, block in enumerate(message['content']):
                if block['type'] == 'recorded_image':
                    message['content'][i] = {'type': 'image_url', 'image_url': {'url': png_data_url((output/block['path']).read_bytes())}}
        assert rebuilt == sent and canonical_hash(sent) == record['expanded_messages_sha256']
        current = json.loads(sent[-1]['content'][0]['text'])
        assert current['decision_index'] == index and 'proprioception' in current['state']['observation']
        assert any(t.get('function') == 'compute_rotation' for t in current['tools']) is math_enabled
    assert load_json(output/'agent_config.json')['history_mode'] == 'current_with_recall'
