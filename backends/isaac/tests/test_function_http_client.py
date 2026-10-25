"""Public-only agent transport checks; CPU doubles do not qualify native play."""
from copy import deepcopy
import json

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPClient, BenchHTTPError
from agent.model_client.function_http import FunctionHTTPClient, FunctionRejected, create_board_game
from test_agent_service import runner, frozen_inputs, session_for, join_action, publish_targets
from test_bench_http_client import camera_observation, rgbd_observation
from test_board_game_service import game
from bench.runtime.server import create_app
from fastapi.testclient import TestClient


TOKEN = 'secret_test_player_token_' + 'a' * 32
PLAYER = {'binding': {'player_id': 'player_0', 'color': 'black', 'hand': 'left',
                     'robot_id': 'left', 'joint_names': ['left_joint']}, 'session_token': TOKEN}


def transport(handler, *, player=None, **kwargs):
    return FunctionHTTPClient('http://bench.test', 'ep1', player=player,
        client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def context():
    return {'episode_id': 'ep1', 'observation_sequence': 2, 'observation_sim_time_s': .4,
        'episode_finished': False, 'request_recorded': True,
        'budget': {'limits': {}, 'action_attempts': 0, 'elapsed_sim_s': .4,
                   'elapsed_wall_s': 1., 'remaining_wall_s': 100.}}


def success(request, result):
    return {'ok': True, 'request_id': request['request_id'], 'function': request['function'],
            'context': context(), 'result': result}


def failure(request=None, *, code='permission_denied', status=403):
    return {'ok': False, 'request_id': None if request is None else request['request_id'],
            'function': None if request is None else request['function'],
            'context': None if request is None else context(),
            'error': {'code': code, 'http_status': status, 'message': TOKEN + ' private details'}}


def query():
    return {'request_id': 'test/observation', 'function': 'observe', 'arguments': {}}


def action_status(action_id='motion', status='accepted'):
    return {'action_id': action_id, 'status': status, 'sim_time_s': .4,
            'elapsed_sim_s': 0., 'episode_finished': False}


def test_ordinary_functions_use_episode_route_and_validate_observation(camera_observation):
    observation, _ = camera_observation
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=success(json.loads(request.content), observation))
    client = transport(handler)
    assert client.state()['observation']['episode_id'] == 'ep1'
    assert client.binding is None
    assert seen[0].url.path == '/episodes/ep1/call'
    assert 'authorization' not in seen[0].headers


@pytest.mark.parametrize('task_id', ['tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'])
def test_authenticated_board_state_catalog_media_and_pending_result(game, task_id):
    with game(task_id) as g:
        before = dict(g.client.headers)
        clients = [FunctionHTTPClient('http://testserver', g.created['episode_id'],
                    player=player, client=g.client) for player in g.created['players']]
        for index, client in enumerate(clients):
            catalog = client.catalog()
            assert '/board-games/' in catalog['call_endpoint']
            assert {'query_ik', 'hand_preset', 'move_wrist', 'set_hand_targets'} <= {
                row['name'] for row in catalog['functions']}
            state = client.state()
            assert state['player']['player_id'] == f'player_{index}'
            assert state['board_game']['task_id'] == task_id
            assert state['public_rules']['revision'] == 'recoverable_physical_moves_v2'
            frames = client.download_cameras(client.episode_id, state['observation'])
            assert frames and frames[0]['png_bytes'] and frames[0]['depth_npy_bytes']
            assert client.game_result() is None
            assert TOKEN not in repr(client)
        assert dict(g.client.headers) == before
        owned = clients[0].binding
        owned['joint_names'].clear()
        assert clients[0].binding['joint_names']


def test_board_intent_and_owned_action_run_through_public_functions(game):
    with game() as g:
        client = FunctionHTTPClient('http://testserver', g.created['episode_id'],
                                    player=g.created['players'][0], client=g.client)
        state = client.intent({'type': 'placement_intent', 'piece_id': 'black_00', 'q': 0, 'r': 0})
        assert state['board_game']['pending_move']['piece_id'] == 'black_00'
        status = client.submit(client.episode_id, {'action_id': 'motion',
            'observation_sequence': state['observation']['sequence'], 'function': 'hold',
            'arguments': {'duration_s': .02}})
        assert status['status'] == 'accepted'
        join_action(g.session)
        assert client.get_action(client.episode_id, 'motion')['status'] == 'completed'
        client.intent({'type': 'cancel_placement'})
        assert client.state()['board_game']['pending_move'] is None
        assert client.get_result(client.episode_id) is None
        client.finish(client.episode_id)
        result = client.game_result()
        assert result['board_game']['status']['outcome'] == 'aborted'
        assert result['episode']['outcome'] == 'cancelled'


def test_ordinary_robot_api_does_not_require_board_player(runner):
    make, _, _ = runner
    api = make(acknowledged_targets=True, on_step=publish_targets)
    session, created = session_for(api)
    with TestClient(create_app(service=api)) as http:
        client = FunctionHTTPClient('http://testserver', created['episode_id'], client=http)
        assert '/episodes/' in client.catalog()['call_endpoint']
        observation = client.get_observation(client.episode_id)
        client.submit(client.episode_id, {'action_id': 'motion',
            'observation_sequence': observation['sequence'], 'function': 'hold',
            'arguments': {'duration_s': .02}})
        join_action(session)
        assert client.get_action(client.episode_id, 'motion')['status'] == 'completed'
        assert client.get_result(client.episode_id) is None
        client.finish(client.episode_id)
        assert client.get_result(client.episode_id)['outcome'] == 'cancelled'


@pytest.mark.parametrize('function', ['get_result', 'finish_episode'])
@pytest.mark.parametrize('field,value', [
    (None, None), ('task_id', 'other_task'), ('task_release_sha256', 'f' * 64),
    ('evaluator_sha256', 'f' * 64), ('physics_profile_sha256', 'f' * 64),
    ('scene_sha256', 'f' * 64), ('agent_config_sha256', 'f' * 64),
    ('robot_id', 'other_robot'), ('seed', 12345),
])
def test_bound_episode_rejects_terminal_spec_drift(runner, function, field, value):
    make, _, _ = runner
    session, ready = session_for(make())
    session.finish()
    def handler(request):
        reply = session.functions.invoke(json.loads(request.content)).model_dump(mode='json')
        assert reply['ok']
        if field is not None:
            assert reply['result']['spec'][field] != value
            reply['result']['spec'][field] = value
        return httpx.Response(200, json=reply)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = FunctionHTTPClient('http://bench.test', ready['episode_id'],
                                    client=http, spec=ready['spec'])
        payload = {'request_id': 'result_identity', 'function': function, 'arguments': {}}
        if field is None:
            assert client.call(payload)['result']['spec'] == ready['spec']
        else:
            with pytest.raises(BenchHTTPError) as failure:
                client.call(payload)
            assert failure.value.category == 'invalid_response'


def test_episode_binding_cannot_be_replaced_or_mutated_by_the_caller(runner):
    make, _, _ = runner
    api = make()
    session, ready = session_for(api)
    with TestClient(create_app(service=api)) as http:
        client = FunctionHTTPClient('http://testserver', ready['episode_id'],
                                    client=http, spec=ready['spec'])
        different = deepcopy(ready['spec'])
        different['budgets']['max_actions'] += 1
        with pytest.raises(BenchHTTPError) as failure:
            client.bind_episode(different)
        assert failure.value.category == 'invalid_response'
        ready['spec']['seed'] = 12345
        assert client.finish(client.episode_id)['spec'] == session.spec.model_dump(mode='json')


def test_two_players_share_client_without_mutating_headers(rgbd_observation):
    observation, png, depth, _ = rgbd_observation
    requests = []
    def handler(request):
        requests.append(request)
        binary = '/depth/' in request.url.path
        return httpx.Response(200, content=depth if binary else png,
            headers={'content-type': 'application/octet-stream' if binary else 'image/png'})
    shared = httpx.Client(transport=httpx.MockTransport(handler))
    other = {'binding': {'player_id': 'player_1', 'color': 'white', 'hand': 'right',
             'robot_id': 'right', 'joint_names': ['right_joint']}, 'session_token': 'b' * 40}
    for player in (PLAYER, other):
        client = FunctionHTTPClient('http://bench.test', 'ep1', player=player, client=shared)
        client.download_cameras('ep1', observation)
        assert [r.headers['authorization'] for r in requests[-2:]] == ['Bearer ' + player['session_token']] * 2
        assert player['session_token'] not in repr(client)
    assert 'authorization' not in shared.headers


@pytest.mark.parametrize('operation', [
    lambda c, o: c.get_observation('other'), lambda c, o: c.get_action('other', 'move'),
    lambda c, o: c.cancel_action('other', 'move'), lambda c, o: c.submit('other', {}),
    lambda c, o: c.download_cameras('other', o), lambda c, o: c.finish('other'),
    lambda c, o: c.get_result('other'),
])
def test_bound_episode_rejects_cross_episode_calls_before_network(camera_observation, operation):
    client = transport(lambda _: pytest.fail('cross-episode request reached network'), player=PLAYER)
    with pytest.raises(BenchHTTPError):
        operation(client, camera_observation[0])


@pytest.mark.parametrize('uri', ['//evil.test/image', 'https://evil.test/image',
    '/episodes/other/images/3/overview', '/episodes/ep1/images/3/overview?token=bad'])
def test_authenticated_media_never_follows_untrusted_location(camera_observation, uri):
    observation, _ = camera_observation
    observation['cameras'][0]['rgb_uri'] = uri
    client = transport(lambda _: pytest.fail('untrusted location reached network'), player=PLAYER)
    with pytest.raises(BenchHTTPError):
        client.download_cameras('ep1', observation)


def test_authenticated_redirect_rejected_without_forwarding_credentials(camera_observation):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={'Location': 'https://evil.test/steal'})
    client = transport(handler, player=PLAYER)
    with pytest.raises(BenchHTTPError):
        client.download_cameras('ep1', camera_observation[0])
    assert len(requests) == 1 and requests[0].url.host == 'bench.test'


@pytest.mark.parametrize('code,status,definite', [('stale_observation', 409, True),
    ('permission_denied', 403, True), ('infrastructure_error', 503, False),
    ('action_conflict', 409, False), ('episode_busy', 409, False)])
def test_typed_function_failures_are_clean_and_never_retried(code, status, definite):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(status, json=failure(json.loads(request.content), code=code, status=status))
    client = transport(handler, player=PLAYER)
    response = client.call(query())
    assert not response['ok'] and response['error']['code'] == code
    assert TOKEN not in json.dumps(response)
    assert len(requests) == 1
    with pytest.raises(FunctionRejected) as error:
        client.submit('ep1', {'action_id': 'motion', 'observation_sequence': 2,
                             'function': 'hold', 'arguments': {'duration_s': .02}})
    assert error.value.definite_no_action is definite
    assert error.value.status_code == status
    assert TOKEN not in str(error.value) + repr(error.value.response)
    assert len(requests) == 2


def test_predispatch_permission_failure_is_typed_but_partial_identity_is_invalid():
    client = transport(lambda _: httpx.Response(403, json=failure()), player=PLAYER)
    assert client.call(query())['error']['code'] == 'permission_denied'
    malformed = failure()
    malformed['request_id'] = 'other'
    client = transport(lambda _: httpx.Response(403, json=malformed), player=PLAYER)
    with pytest.raises(BenchHTTPError) as error:
        client.call(query())
    assert error.value.category == 'invalid_response'


@pytest.mark.parametrize('mutate,status', [
    (lambda r: r.update(request_id='different'), 200),
    (lambda r: r.update(function='get_robot'), 200),
    (lambda r: r.update(context=None), 200),
    (lambda r: r['context'].update(episode_id='different'), 200),
    (lambda r: r['result'].update(episode_id='different'), 200),
    (lambda r: r['result'].update(sequence=10), 200),
    (lambda r: r.update(private_state={'secret': TOKEN}), 200),
    (lambda r: None, 202), (lambda r: None, 503),
])
def test_malformed_or_misbound_function_response_is_not_accepted(camera_observation, mutate, status):
    response = success(query(), camera_observation[0])
    mutate(response)
    client = transport(lambda _: httpx.Response(status, json=response))
    with pytest.raises(BenchHTTPError) as error:
        client.call(query())
    assert error.value.category == 'invalid_response' and TOKEN not in str(error.value)


@pytest.mark.parametrize('status', [200, 400, 503])
def test_failure_http_status_must_match_envelope(status):
    client = transport(lambda _: httpx.Response(status, json=failure(query(), status=409)))
    with pytest.raises(BenchHTTPError) as error:
        client.call(query())
    assert error.value.category == 'invalid_response'


@pytest.mark.parametrize('status,body', [(502, {'detail': TOKEN}),
                                      (200, {'ok': True}), (422, [TOKEN])])
def test_invalid_responses_never_leak_body(status, body):
    client = transport(lambda _: httpx.Response(status, json=body))
    with pytest.raises(BenchHTTPError) as error:
        client.call(query())
    assert TOKEN not in str(error.value)


def test_post_transport_timeout_is_once_and_secret_free():
    requests = []
    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout(TOKEN, request=request)
    client = transport(handler, player=PLAYER)
    with pytest.raises(BenchHTTPError) as error:
        client.call(query())
    assert error.value.category == 'transport_timeout'
    assert TOKEN not in str(error.value) and len(requests) == 1


def test_status_and_cancel_are_public_calls_and_check_action_identity():
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        result = action_status(status='completed')
        if payload['function'] == 'cancel_action':
            result['cancellation_requested'] = True
        return httpx.Response(200, json=success(payload, result))
    client = transport(handler, player=PLAYER)
    assert client.get_action('ep1', 'motion')['status'] == 'completed'
    assert client.cancel_action('ep1', 'motion')['cancellation_requested']
    assert [request['function'] for request in requests] == ['get_action', 'cancel_action']
    with pytest.raises(BenchHTTPError):
        client.get_action('ep1', 'another_action')


@pytest.mark.parametrize('change', [
    lambda value: value.update(private_objects={}),
    lambda value: value['player'].update(joint_names=['another_joint']),
    lambda value: value['observation'].update(episode_id='different'),
    lambda value: value.update(public_rules_sha256='0' * 64),
    lambda value: value['geometry'].update(pitch_m=9.),
])
def test_board_state_wrapper_rejects_wrong_binding_private_fields_or_rules(game, change):
    with game() as g:
        source = FunctionHTTPClient('http://testserver', g.created['episode_id'],
                                    player=g.created['players'][0], client=g.client).state()
        change(source)
        client = FunctionHTTPClient('http://bench.test', g.created['episode_id'],
            player=g.created['players'][0], client=httpx.Client(transport=httpx.MockTransport(
                lambda _: httpx.Response(200, json=source))))
        with pytest.raises(BenchHTTPError):
            client.state()


def test_create_board_game_validates_credentials_without_logging_or_global_auth(game):
    with game() as g:
        seen = []
        def handler(request):
            seen.append(request)
            return httpx.Response(201, json=g.created)
        bench = BenchHTTPClient('http://bench.test', client=httpx.Client(transport=httpx.MockTransport(handler)))
        created = create_board_game(bench, g.request)
        assert created['players'] == g.created['players']
        assert len(seen) == 1 and seen[0].url.path == '/board-games'
        assert 'authorization' not in seen[0].headers
        with pytest.raises(BenchHTTPError):
            create_board_game(bench, {**g.request, 'task_id': 'task0'})
        assert len(seen) == 1
        bad = deepcopy(g.created)
        bad['players'][1]['session_token'] = bad['players'][0]['session_token']
        bench.client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(201, json=bad)))
        with pytest.raises(BenchHTTPError) as error:
            create_board_game(bench, g.request)
        assert all(p['session_token'] not in str(error.value) for p in g.created['players'])
