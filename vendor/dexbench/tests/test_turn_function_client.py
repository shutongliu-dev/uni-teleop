"""Authenticated Agent turn transport/session checks using an explicit CPU referee double."""
from copy import deepcopy
import json

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError
from agent.model_client.function_http import FunctionHTTPClient, create_turn_game
from agent.model_client.function_session import FunctionSession
from agent.model_client.function_session import _turn_intent_tools
from bench.protocol.turn_games import TurnIntent
from test_agent_service import PRIVATE_MARKER, frozen_inputs
from test_turn_game_service import game


def clients(g):
    result = [FunctionHTTPClient('http://testserver', g.created['episode_id'],
                              player=player, session_kind='turn_game', client=g.client)
            for player in g.created['players']]
    for client in result:
        client.bind_turn_state(g.created['turn_game'])
    return result


def mocked_bench(created):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(201, json=created)
    return BenchHTTPClient('http://bench.test', client=httpx.Client(transport=httpx.MockTransport(handler))), requests


def test_create_turn_game_binds_public_task_and_host_credentials(game):
    with game() as g:
        bench, requests = mocked_bench(g.created)
        received = []
        created = create_turn_game(bench, g.payload, on_created=received.append)
        assert created['players'] == g.created['players']
        assert created['public_rules'] == created['turn_game']['public_rules']
        assert received == [g.created]
        assert len(requests) == 1 and requests[0].url.path == '/turn-games'
        assert 'authorization' not in requests[0].headers
        assert PRIVATE_MARKER not in json.dumps(created)


@pytest.mark.parametrize('mutation', [
    lambda v: v['players'][1].update(session_token=v['players'][0]['session_token']),
    lambda v: v['players'][1]['binding'].update(player_id='alpha'),
    lambda v: v['players'][0]['binding'].update(player_id='../other'),
    lambda v: v['players'][1]['binding'].update(robot_id='front', hand='front'),
    lambda v: v['players'][1]['binding']['joint_names'].append(v['players'][0]['binding']['joint_names'][0]),
    lambda v: v['players'][0]['binding']['joint_names'].pop(),
    lambda v: v['turn_game'].update(players=['alpha', 'other']),
    lambda v: v['turn_game'].update(game_id='other'),
    lambda v: v['turn_game'].update(turn_index=1),
    lambda v: v['turn_game'].update(sim_time_s=.02),
    lambda v: v['public_rules'].update(unannounced=True),
    lambda v: v['spec'].update(task_id='other'),
    lambda v: v.update(private_state=PRIVATE_MARKER),
])
def test_create_turn_game_rejects_misbound_or_malformed_response_without_token_leak(game, mutation):
    with game() as g:
        value = deepcopy(g.created)
        mutation(value)
        bench, requests = mocked_bench(value)
        with pytest.raises(BenchHTTPError) as error:
            create_turn_game(bench, g.payload)
        assert error.value.category == 'invalid_response'
        assert all(token not in str(error.value) for token in g.tokens.values())
        assert len(requests) == 1


def test_turn_clients_scope_catalog_media_intents_actions_and_result(game):
    with game() as g:
        original_headers = dict(g.client.headers)
        front, rear = clients(g)
        for client, side, player_id in ((front, 'front', 'alpha'), (rear, 'rear', 'beta')):
            assert client.session_kind == 'turn_game'
            assert client.binding['hand'] == side
            assert '/turn-games/' in client.catalog()['call_endpoint']
            state = client.state()
            assert state['player']['player_id'] == player_id
            assert client.game_result() is None
            frames = client.download_cameras(client.episode_id, state['observation'])
            assert frames[0]['png_bytes'] and frames[0]['depth_npy_bytes']
            with pytest.raises(BenchHTTPError):
                client._request_headers('/board-games/' + client.episode_id + '/players/' + player_id + '/state')
        assert dict(g.client.headers) == original_headers
        state = front.intent({'type': 'turn_intent', 'name': 'select', 'arguments': {'slot': 1}})
        assert state['turn_game']['public_progress'] == {'selected': 1}
        assert state['turn_game']['turn_index'] == 0
        with pytest.raises(BenchHTTPError):
            rear.intent({'type': 'turn_intent', 'name': 'select', 'arguments': {'slot': 2}})
        with pytest.raises(BenchHTTPError):
            front.intent({'type': 'placement_intent', 'piece_id': 'black_00', 'q': 0, 'r': 0})
        front.finish(front.episode_id)
        result = rear.game_result()
        assert result['turn_game']['status'] == 'interrupted'
        assert result['turn_game']['winner_player_id'] is None
        assert result['turn_game']['loser_player_id'] is None
        assert result['episode']['outcome'] == 'cancelled'


@pytest.mark.parametrize('mutation', [
    lambda v: v['player'].update(player_id='beta'),
    lambda v: v['turn_game'].update(game_id='other'),
    lambda v: v['turn_game'].update(players=['alpha', 'other']),
    lambda v: v['turn_game']['public_rules'].update(hidden_override=True),
    lambda v: v['turn_game'].update(current_player_id='beta'),
    lambda v: v['turn_game'].update(sim_time_s=1.),
    lambda v: v['observation'].update(episode_id='other'),
    lambda v: v.update(private_state=PRIVATE_MARKER),
])
def test_turn_state_rejects_changed_identity_rules_clock_or_private_fields(game, mutation):
    with game() as g:
        client = clients(g)[0]
        state = client.state()
        mutation(state)
        client.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=state)))
        with pytest.raises(BenchHTTPError) as error:
            client.state()
        assert error.value.category == 'invalid_response'


def test_creation_state_binding_rejects_changed_rules_on_first_state(game):
    with game() as g:
        client = clients(g)[0]
        state = g.session.turn_game.public('alpha')
        state['turn_game']['public_rules'] = {'intents': ['unauthorized']}
        client.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=state)))
        with pytest.raises(BenchHTTPError) as error:
            client.state()
        assert error.value.category == 'invalid_response'


@pytest.mark.parametrize('offset,accepted', [(5.e-10, True), (1.e-9, True), (2.e-9, False)])
def test_turn_state_clock_uses_bench_nanosecond_tolerance(game, offset, accepted):
    with game() as g:
        client = clients(g)[0]
        state = g.session.turn_game.public('alpha')
        state['turn_game']['sim_time_s'] += offset
        client.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=state)))
        if accepted:
            assert client.state()['turn_game']['sim_time_s'] == offset
        else:
            with pytest.raises(BenchHTTPError) as error:
                client.state()
            assert error.value.category == 'invalid_response'


@pytest.mark.parametrize('mutation', [
    lambda v: v['episode']['spec'].update(episode_id='other'),
    lambda v: v['turn_game']['public_rules'].update(changed=True),
    lambda v: v['turn_game'].update(players=['alpha', 'other']),
    lambda v: v['turn_game'].update(status='completed', winner_player_id='alpha', loser_player_id=None),
])
def test_turn_result_rejects_mismatched_or_incomplete_adjudication(game, mutation):
    with game() as g:
        client = clients(g)[0]
        client.finish(client.episode_id)
        result = g.session.turn_game.result('alpha')
        mutation(result)
        client.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=result)))
        with pytest.raises(BenchHTTPError) as error:
            client.game_result()
        assert error.value.category == 'invalid_response'


def test_turn_session_keeps_current_player_until_native_handoff_and_scopes_queries(game, tmp_path):
    with game() as g:
        front, rear = [FunctionSession(client, output=tmp_path / side, spec=g.created['spec'],
                      trial_id=side, max_actions=10, max_calls=20)
                      for side, client in zip(('front', 'rear'), clients(g))]
        for host in (front, rear):
            context = host.initialize()
            assert PRIVATE_MARKER not in json.dumps(context)
            assert all(token not in json.dumps(context) for token in g.tokens.values())
            assert context['decision_schema']['one_of'][1]['intent']['type'] == 'turn_intent'
            tools = {tool['function']: tool['arguments_schema'] for tool in context['tools'] if 'function' in tool}
            side = host.binding['hand']
            assert tools['move_wrist']['properties']['hand']['enum'] == [side]
            assert set(tools['query_ik']['properties']['hand_geometries']['properties']) == {side}
        hold = {'function': 'hold', 'arguments': {'duration_s': .02}}
        assert rear.execute(hold)['error']['code'] == 'permission_denied'
        assert front.execute({'intent': {'type': 'turn_intent', 'name': 'select', 'arguments': {'slot': 1}}})['ok']
        assert front.execute(hold)['ok']
        assert front.state['turn_game']['current_player_id'] == 'alpha'
        foreign = front.execute({'function': 'hand_preset', 'arguments': {'side': 'rear', 'preset': 'open'}})
        assert not foreign['ok']
        g.worker.switch_next = True
        result = front.execute({'function': 'hold', 'arguments': {'duration_s': .1}})
        assert result['ok'] and result['result']['status'] == 'cancelled'
        assert result['result']['termination_reason'] == 'turn_changed'
        assert front.pending is None and front.state['turn_game']['current_player_id'] == 'beta'
        assert front.execute(hold)['error']['code'] == 'permission_denied'
        rear.context()
        assert rear.execute(hold)['ok']
        assert rear.state['turn_game']['current_player_id'] == 'beta'


@pytest.mark.parametrize('session_kind,player', [
    ('turn_game', None), ('episode', {}), ('unknown', None),
])
def test_invalid_session_kind_rejected_before_network(session_kind, player):
    with pytest.raises(BenchHTTPError):
        FunctionHTTPClient('http://bench.test', 'ep1', player=player, session_kind=session_kind)


def test_tower_intent_adapter_describes_required_public_block_selection():
    tool, examples = _turn_intent_tools('tower_extraction')
    example = examples[0]['intent']
    assert example == {'type': 'turn_intent', 'name': 'select',
                       'arguments': {'block_id': 'ID from state.turn_game.public_progress.legal_blocks'}}
    for row in examples:
        assert TurnIntent.model_validate(row['intent']).model_dump(mode='json') == row['intent']
    schema = tool['intent']
    assert schema['properties']['name']['enum'] == ['select', 'cancel', 'push', 'pull', 'regrasp', 'release', 'wait']
    assert set(schema['properties']['arguments']['properties']) == {'block_id'}
    assert schema['properties']['arguments']['additionalProperties'] is False
    selection = schema['allOf'][0]
    assert selection['if']['properties']['name']['const'] == 'select'
    assert selection['then']['required'] == ['arguments']
    assert selection['then']['properties']['arguments']['required'] == ['block_id']
    assert 'selection_locked' in tool['scope'] and 'without physical motion' in tool['scope']
    generic, examples = _turn_intent_tools('another_turn_task')
    assert 'enum' not in generic['intent']['properties']['name']
    assert examples[0]['intent']['name'] == 'advertised intent'
