"""Every public contact-history version survives the Agent feedback boundary."""
from copy import deepcopy
import json

import httpx
import pytest

from agent.model_client.bench_http import BenchHTTPError
from agent.model_client.program import execute_program
from agent.model_client.visibility import feedback_view
from bench.sensors.contact_history import ContactHistory
from bench.sensors.contact_schema import parse_contact_history
from test_bimanual_contact_history import dual_config, dual_metadata, sampled_window
from test_contact_history import contact
from test_instance_contact_history import instance_config
from test_function_http_client import transport, success


@pytest.fixture(params=['dual_config', 'instance_config'])
def multiple_hand_history(request):
    config = request.getfixturevalue(request.param)
    side = config.get('instance_names', ['left'])[0]
    history = ContactHistory(config)
    history.begin('motion', 0.)
    history.observe(sampled_window(rows=[contact(config, side + '_index_elastomer', impulse=.002)]))
    value = history.snapshot()
    assert value['status'] == 'available'
    assert value['hands'][side]['regions']['index_pad']['normal_impulse_sum_ns'] == .002
    return value


def status(history):
    return {'action_id': 'motion', 'status': 'completed', 'sim_time_s': .02,
            'elapsed_sim_s': .02, 'episode_finished': False, 'contact_history': history}


@pytest.mark.parametrize('tactile', ['on', 'off'])
def test_valid_multiple_hand_feedback_preserves_public_partitions(multiple_hand_history, tactile):
    value = status(multiple_hand_history)
    original = deepcopy(value)
    projected = feedback_view(value, tactile_input=tactile)
    assert value == original
    assert projected == (value if tactile == 'on' else {
        key: item for key, item in value.items() if key != 'contact_history'})
    assert 'private_' not in json.dumps(projected)


def test_program_minimum_feedback_retains_multiple_hand_history_after_later_query(multiple_hand_history):
    value = status(multiple_hand_history)
    result = execute_program('hold(.02)\nobserve()', lambda name, args: (
        {'feedback': value} if name == 'hold' else {'observation': {}}))
    assert result['status'] == 'completed'
    assert result['primitive_results'][0]['feedback']['contact_history'] == multiple_hand_history


def test_program_preserves_declared_eight_instance_summary(instance_config):
    history = ContactHistory(instance_config)
    history.begin('motion', 0.)
    history.observe(sampled_window())
    value = history.snapshot()
    hand = value['hands']['front']
    value['instance_names'] = ['robot_' + str(index) for index in range(8)]
    value['hands'] = {name: deepcopy(hand) for name in value['instance_names']}
    value = parse_contact_history(value).model_dump(mode='json')
    assert len(json.dumps(value).encode()) > 16384
    result = execute_program('hold(.02)', lambda *_: {'feedback': status(value)})
    assert result['primitive_results'][0]['feedback']['contact_history'] == value


@pytest.mark.parametrize('damage', ['action_id', 'private_field', 'wrong_namespace'])
@pytest.mark.parametrize('tactile', ['on', 'off'])
def test_invalid_multiple_hand_feedback_still_rejects_before_ablation(multiple_hand_history, damage, tactile):
    value = deepcopy(multiple_hand_history)
    if damage == 'action_id':
        value['action_id'] = 'another_motion'
    elif damage == 'private_field':
        value['hands'][next(iter(value['hands']))]['private_pose'] = 'PRIVATE_CANARY'
    else:
        side = next(iter(value['hands']))
        value['hands']['not_a_published_hand'] = value['hands'].pop(side)
    with pytest.raises(BenchHTTPError) as caught:
        feedback_view(status(value), tactile_input=tactile)
    assert caught.value.category == 'invalid_response'
    assert 'PRIVATE_CANARY' not in str(caught.value)
    result = execute_program('hold(.02)', lambda *_: {'feedback': status(value)})
    assert 'contact_history' not in result['primitive_results'][0]['feedback']


@pytest.mark.parametrize('function', ['get_action', 'cancel_action', 'hold'])
@pytest.mark.parametrize('matching', [False, True])
def test_function_transport_binds_contact_to_the_requested_action(multiple_hand_history, function, matching):
    value = deepcopy(multiple_hand_history)
    if not matching:
        value['action_id'] = 'another_motion'
    result = status(value)
    if function == 'cancel_action':
        result['cancellation_requested'] = False
    def handler(request):
        return httpx.Response(200, json=success(json.loads(request.content), result))
    client = transport(handler)
    arguments = {'action_id': 'motion'}
    if function == 'hold':
        arguments.update(observation_sequence=2, duration_s=.02)
    payload = {'request_id': 'contact_binding', 'function': function, 'arguments': arguments}
    if matching:
        # Querying an earlier completed action retains its own interval, even
        # after the episode clock in the response context has advanced to .4s.
        reply = client.call(payload)
        assert reply['result']['contact_history'] == value
    else:
        with pytest.raises(BenchHTTPError) as caught:
            client.call(payload)
        assert caught.value.category == 'invalid_response'
