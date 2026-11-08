"""Public Agent card-game contracts with CPU doubles, not physical play evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.card_games import CARD_PROTOCOL, CARD_TASKS, CARD_INTENTS, PublicEventDelivery, intent_tools
from agent.model_client.function_http import FunctionHTTPClient
from agent.model_client.function_session import FunctionSession
from agent.model_client.runner import canonical_hash
from agent.model_client.scene_tasks import TaskSetupError, _selection, published_task
from bench.protocol.card_games import INTENT_ADAPTER
from test_function_http_client import PLAYER, TOKEN
from test_function_session import Client


def public_state(task_id='memory_pairs'):
    return {'board_game': {'protocol': CARD_PROTOCOL, 'task_id': task_id,
        'turn_index': 0, 'round_index': 0, 'current_player': 'player_0', 'terminal': False,
        'winner': None, 'draw': False, 'pending_move': None, 'settlement_dwell_s': 0.,
        'status': {'outcome': 'ongoing', 'end_reason': None, 'recovery_required': False,
                   'evidence_unavailable': False, 'reason': 'await_intent', 'retryable': True},
        'commit_count': 0, 'phase': 'await_intent', 'event_cursor': -1},
        'public_rules': {'task_id': task_id, 'test_fixture': 'public_rules_only'},
        'public_rules_sha256': canonical_hash({'task_id': task_id, 'test_fixture': 'public_rules_only'}),
        'observation': Client().obs, 'player': deepcopy(PLAYER['binding'])}


def client_for(state, handler=None):
    client = FunctionHTTPClient('http://bench.test', 'ep1', player=PLAYER, game_protocol=CARD_PROTOCOL,
        client=httpx.Client(transport=httpx.MockTransport(handler or (lambda _: httpx.Response(200, json=state)))))
    client.bind_board_state(state)
    return client


def revealed(sequence=0, *, card_id='card_00', symbol='pair_0', expected=True):
    return {'sequence': sequence, 'task_id': 'memory_pairs', 'type': 'card_revealed',
            'player_id': 'player_0', 'turn_index': sequence, 'native_substep_index': sequence + 1,
            'simulation_time_s': (sequence + 1) * .0005,
            'details': {'card_id': card_id, 'symbol': symbol, 'expected': expected}}


@pytest.mark.parametrize('task_id', sorted(CARD_TASKS))
def test_task_specific_tools_and_real_robot_functions(tmp_path, task_id):
    client = Client(board=True)
    client.public.update(public_state(task_id))
    client.public['player'] = client.binding
    client.public['observation'] = client.obs
    host = FunctionSession(client, output=tmp_path / task_id, spec={'task_id': task_id},
                           trial_id='player_0', max_actions=8, max_calls=30)
    host.initialize()
    context = host.context()
    tools = context['tools']
    schema = next(row['intent'] for row in tools if 'intent' in row)
    assert set(schema['discriminator']['mapping']) == set(CARD_INTENTS[task_id])
    assert {'move_wrist', 'set_hand_targets', 'hold'} <= {row.get('function') for row in tools}
    for example in intent_tools(task_id, 'white')[1]:
        parsed = INTENT_ADAPTER.validate_python(example['intent'])
        assert parsed.type in CARD_INTENTS[task_id]
        assert getattr(parsed, 'piece_id', 'white_').startswith('white_')
    # Intent declares an operation and never submits a motion or moves a body.
    intent = intent_tools(task_id)[1][0]
    assert host.execute(intent)['ok']
    assert client.intents == [intent['intent']]
    assert client.posts == []
    feedback = host.execute({'function': 'hold', 'arguments': {'duration_s': .02}})
    assert feedback['ok'], feedback
    assert len(client.posts) == 1 and client.posts[0]['function'] == 'hold'
    illegal = {'type': 'shoot_disk', 'piece_id': 'black_disk_0'} if task_id != 'disk_curling' else {'type': 'reveal_card', 'card_id': 'card_00'}
    assert not host.execute({'intent': illegal})['ok']
    assert len(client.intents) == 1


def test_same_player_second_reveal_uses_explicit_player_not_epoch_parity(tmp_path):
    state = public_state()
    client = client_for(state)
    state['board_game'].update(turn_index=1, commit_count=1, event_cursor=0, phase='await_second_reveal')
    assert client.state()['board_game']['current_player'] == 'player_0'
    fake = Client(board=True)
    fake.public.update(deepcopy(state), player=fake.binding)
    host = FunctionSession(fake, output=tmp_path / 'epoch', spec={'task_id': 'memory_pairs'},
                           trial_id='player_0', max_actions=4, max_calls=12)
    host.initialize()
    assert host.execute({'intent': {'type': 'reveal_card', 'card_id': 'card_01'}})['ok']
    state['board_game']['current_player'] = 'player_1'  # Same epoch cannot silently change owner.
    with pytest.raises(BenchHTTPError):
        client.state()


def test_card_baseline_opens_only_after_a_real_explicit_hold(tmp_path):
    client = Client(board=True)
    client.public.update(public_state(), player=client.binding, observation=client.obs)
    client.public['board_game']['phase'] = 'awaiting_native_baseline'
    host = FunctionSession(client, output=tmp_path / 'baseline', spec={'task_id': 'memory_pairs'},
                           trial_id='player_0', max_actions=8, max_calls=20)
    host.initialize()
    context = host.context()
    assert not any('intent' in row for row in context['tools'])
    functions = {row.get('function') for row in context['tools']}
    assert {'hold', 'get_robot', 'finish_episode'} <= functions
    assert 'move_wrist' not in functions
    assert not host.execute({'intent': {'type': 'reveal_card', 'card_id': 'card_00'}})['ok']
    assert not host.execute({'function': 'hold', 'arguments': {'duration_s': .5}})['ok']
    assert client.posts == []
    assert host.execute({'function': 'hold', 'arguments': {'duration_s': .5, 'mode': 'acknowledged_drive'}})['ok']
    assert len(client.posts) == 1
    # A successful hold alone never overrides the referee; its new phase does.
    assert not any('intent' in row for row in host.context()['tools'])
    client.public['board_game']['phase'] = 'reveal_first'
    assert any('intent' in row for row in host.context()['tools'])
    assert host.execute({'intent': {'type': 'reveal_card', 'card_id': 'card_00'}})['ok']


@pytest.mark.parametrize('fault', ['old_geometry', 'private_state', 'private_pose', 'changed_rules', 'wrong_protocol', 'epoch_regression'])
def test_card_public_state_fails_closed(fault):
    state = public_state()
    state['board_game'].update(turn_index=2, commit_count=2, round_index=1, event_cursor=3)
    client = client_for(state)
    if fault == 'old_geometry': state['geometry'] = {}
    if fault == 'private_state': state['private_reset_pairs'] = {'card_00': 'pair_0'}
    if fault == 'private_pose': state['observation']['privileged_object_poses'] = {'card_00': {}}
    if fault == 'changed_rules':
        state['public_rules']['test_fixture'] = 'changed'
        state['public_rules_sha256'] = canonical_hash(state['public_rules'])
    if fault == 'wrong_protocol': state['board_game']['protocol'] = 'bench_board_game_state_v1'
    if fault == 'epoch_regression': state['board_game'].update(turn_index=1, commit_count=1)
    with pytest.raises(BenchHTTPError) as error:
        client.state()
    assert error.value.category == 'invalid_response'


def test_authenticated_event_transport_has_exact_scope_and_public_fields():
    seen = []
    event = revealed(expected=False)
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={'events': [event], 'next_cursor': 0, 'has_more': False})
    client = client_for(public_state(), handler)
    assert client.events()['events'] == [event]
    assert seen[0].url.path == '/board-games/ep1/players/player_0/events'
    assert str(seen[0].url.query, 'ascii') == 'after=-1&limit=256'
    assert seen[0].headers['authorization'] == 'Bearer ' + TOKEN
    for path in ('/board-games/ep1/players/player_1/events?after=-1&limit=256',
                 client._endpoint + '/events?after=-1&limit=257',
                 client._endpoint + '/events?after=-1&limit=256&private=true'):
        with pytest.raises(BenchHTTPError): client._request_headers(path)
    event['details']['private_reset_pairs'] = {'card_01': 'pair_0'}
    with pytest.raises(BenchHTTPError): client.events()


@pytest.mark.parametrize('fault', ['skip', 'duplicate', 'empty_more', 'bad_cursor', 'wrong_task'])
def test_event_transport_rejects_missing_or_misbound_events(fault):
    value = {'events': [revealed()], 'next_cursor': 0, 'has_more': False}
    if fault == 'skip': value['events'][0]['sequence'] = value['next_cursor'] = 1
    if fault == 'duplicate': value['events'].append(revealed())
    if fault == 'empty_more': value.update(events=[], has_more=True)
    if fault == 'bad_cursor': value['next_cursor'] = 2
    if fault == 'wrong_task': value['events'][0]['task_id'] = 'disk_curling'
    client = client_for(public_state(), lambda _: httpx.Response(200, json=value))
    with pytest.raises(BenchHTTPError): client.events()


def event_client(log, calls):
    def events(*, after, limit):
        calls.append((after, limit))
        rows = deepcopy(log[after + 1:after + 1 + limit])
        cursor = after + len(rows)
        return {'events': rows, 'next_cursor': cursor, 'has_more': cursor < len(log) - 1}
    return SimpleNamespace(events=events)


def test_every_transient_reveal_reaches_both_persistent_histories_once():
    log = [revealed(i, card_id=f'card_{i % 12:02d}', expected=i % 2 == 0) for i in range(300)]
    calls = {'player_0': [], 'player_1': []}
    clients = {key: event_client(log, value) for key, value in calls.items()}
    history = {key: [{'role': 'assistant', 'content': key + '_private_strategy'}] for key in clients}
    delivery = PublicEventDelivery(clients)
    delivery.drain(history, through=299)
    delivery.drain(history, through=299)
    for key in clients:
        assert calls[key] == [(-1, 256), (255, 256), (299, 256)]
        rows = [row for message in history[key][1:] for row in json.loads(message['content'])['public_game_events']]
        assert rows == log and delivery.events[key] == log
        assert ('player_1' if key == 'player_0' else 'player_0') + '_private_strategy' not in json.dumps(history[key])
    # Later turn/context snapshots need no current face-up card to retain all reveals.
    assert any(not row['details']['expected'] for row in delivery.events['player_1'])


@pytest.mark.parametrize('fault', ['missing', 'different'])
def test_both_players_must_receive_the_same_complete_public_log(fault):
    other = [] if fault == 'missing' else [revealed(symbol='pair_1')]
    clients = {'player_0': event_client([revealed()], []), 'player_1': event_client(other, [])}
    history = {key: [] for key in clients}
    delivery = PublicEventDelivery(clients)
    with pytest.raises(BenchHTTPError): delivery.drain(history, through=0)
    assert delivery.events['player_0'] == [revealed()]  # Earlier receipt is not discarded on failure.


@pytest.mark.parametrize('task_id', sorted(CARD_TASKS))
def test_card_selection_requires_its_own_published_protocol_and_preserves_planned_gate(task_id):
    definition = {'family': 'card_room', 'task_id': task_id, 'status': 'implemented',
                  'rules': {'contract': {'competitive': True}}}
    selected = _selection(definition, pin_release=False)
    assert selected.session_kind == 'board_game' and selected.adapter_available
    assert selected.players == ('player_0', 'player_1')
    task = {'task_id': task_id, 'task_kind': 'board_game', 'protocol': CARD_PROTOCOL,
            'schema_endpoint': 'GET /board-games/schema?protocol=' + CARD_PROTOCOL}
    assert published_task(selected, [task]) == (task, True)
    task['protocol'] = 'bench_board_game_state_v1'
    with pytest.raises(TaskSetupError): published_task(selected, [task])
    definition['status'] = 'planned'
    with pytest.raises(TaskSetupError) as error: _selection(definition, pin_release=False).require_runnable()
    assert error.value.code == 'task_planned'
