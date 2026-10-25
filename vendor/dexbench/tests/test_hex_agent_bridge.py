"""Two-context bridge tests use fake Responses clients; no API/native execution."""
from dataclasses import replace
import json

import pytest

from scenes.board_games.hex import ActuatorOwnership, HexMatch, HexPlayerProtocol, SMOKE_MODE
from scripts.validation.hex_model_smoke import HexAgentBridge, HexBridgeError, parse_hex_decision
from agent.model_client.transport import ModelGatewayError, ModelResponse

PNG = b'\x89PNG\r\n\x1a\nunit-test-bytes-not-a-render'


def snapshot(**changes):
    return {'snapshot_id': 'snapshot-1', 'simulation_time_s': 0., 'rgb': 'public/task_rgb.png',
            'depth': 'public/task_depth.npy', 'calibration': {'intrinsics_3x3': [500., 0., 320., 0., 500., 240., 0., 0., 1.],
            'world_from_camera': {'frame_id': 'world', 'position_m': [0., 0., 1.],
                                  'quaternion_wxyz': [1., 0., 0., 0.]}}, **changes}


def command(player='player_0', **changes):
    side = 'left' if player == 'player_0' else 'right'
    return {'type': 'joint_targets', 'robot_id': side,
            'joint_positions': {f'fr3_{side}_joint7': .002}, 'duration_s': .02, **changes}


def decision(player='player_0'):
    color = 'red' if player == 'player_0' else 'blue'
    return {'placement_intent': {'type': 'placement_intent', 'piece_id': color+'_00', 'q': 3, 'r': 3},
            'joint_targets': command(player)}


def response(value):
    return ModelResponse(text=json.dumps(value) if isinstance(value, dict) else value, model='gpt-6-astra',
                         usage={'input_tokens': 123, 'output_tokens': 45, 'total_tokens': 168,
                                'output_tokens_details': {'reasoning_tokens': 33}, 'secret': 'must-not-leak'},
                         request_id='must-not-leak', duration_s=.5, attempts=1, finish_reason='completed',
                         requested_model='gpt-6-astra', max_output_tokens=8192, reasoning_effort='max',
                         protocol='responses')


class FakeClient:
    def __init__(self, value):
        self.response = response(value)
        self.calls = []
        self.error = None
        self.during_request = None

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.during_request:
            self.during_request()
        if self.error:
            raise self.error
        return self.response


def setup():
    match = HexMatch()
    protocol = HexPlayerProtocol(match, (
        ActuatorOwnership('player_0', 'left', ('fr3_left_joint7', 'left_thumb')),
        ActuatorOwnership('player_1', 'right', ('fr3_right_joint7', 'right_thumb'))), observation_mode=SMOKE_MODE)
    clients = {player: FakeClient(decision(player)) for player in match.rules.players}
    return HexAgentBridge(protocol, clients), clients


def test_two_contexts_same_images_and_no_readiness_execution():
    bridge, clients = setup()
    public = snapshot(private_object_poses={'secret': 'must-not-leak'}, opponent_reasoning='must-not-leak')
    red = bridge.propose('player_0', public, joint_command=command(), images=(PNG,))
    blue = bridge.propose('player_1', public, joint_command=command('player_1'), images=(PNG,), readiness=True)
    assert bridge.protocol.match.pending is None
    assert not bridge.protocol.match.rules.board
    red_prompt = json.loads(clients['player_0'].calls[0]['prompt'])
    blue_prompt = json.loads(clients['player_1'].calls[0]['prompt'])
    assert red_prompt['own_player_history'] == blue_prompt['own_player_history'] == []
    assert red_prompt['request']['context_id'] != blue_prompt['request']['context_id']
    assert red_prompt['request']['public_observation']['snapshot'] == blue_prompt['request']['public_observation']['snapshot']
    assert red_prompt['request']['snapshot_image_sha256'] == blue_prompt['request']['snapshot_image_sha256']
    assert clients['player_0'].calls[0]['images'] == clients['player_1'].calls[0]['images'] == (PNG,)
    assert 'must-not-leak' not in clients['player_0'].calls[0]['prompt']
    assert 'unexecuted' in blue.evidence()['status']
    calls = []
    with pytest.raises(HexBridgeError, match='readiness_proposal_cannot_execute'):
        bridge.execute(blue, published_snapshot=public, execute_joint_targets=calls.append)
    assert calls == []
    result = bridge.execute(red, published_snapshot=public, execute_joint_targets=calls.append)
    assert len(calls) == 1
    assert calls[0]['player_id'] == 'player_0'
    assert calls[0]['robot_id'] == 'left'
    assert calls[0]['joint_positions'] == {'fr3_left_joint7': .002}
    assert bridge.protocol.match.pending.player_id == 'player_0'
    assert bridge.protocol.match.rules.turn_index == 0
    assert result['pending_move_remains'] and not result['logical_move_committed']
    assert not result['model_manipulation_score_claimed']
    assert 'must-not-leak' not in json.dumps(result)
    assert result['model_response']['usage']['input_tokens'] == 123
    assert result['model_response']['duration_s'] == .5
    assert result['model_response']['model'] == 'gpt-6-astra'
    assert not result['model_response']['reasoning_effort_verified_by_server']


def test_run_turn_calls_model_once_and_discards_private_callback_return():
    bridge, clients = setup()
    result = bridge.run_turn(snapshot(), joint_command=command(), images=(PNG,),
                             execute_joint_targets=lambda _: {'private_native_state': 'must-not-leak'})
    assert result['joint_command_dispatched']
    assert 'must-not-leak' not in json.dumps(result)
    assert len(clients['player_0'].calls) == 1
    assert not clients['player_1'].calls
    call = clients['player_0'].calls[0]
    assert call['model'] == 'gpt-6-astra'
    assert call['reasoning_effort'] == 'max'
    assert call['max_output_tokens'] == 8192


def test_rgb_then_quantized_depth_image_order_is_explicit():
    bridge, clients = setup()
    public = snapshot()
    public['calibration']['depth_scale_m'] = .01
    bridge.propose('player_0', public, joint_command=command(), images=(PNG, PNG+b'depth'))
    call = clients['player_0'].calls[0]
    attached = json.loads(call['prompt'])['request']['attached_images']
    assert attached[0]['index'] == 0 and attached[0]['kind'] == 'rgb'
    assert attached[1]['index'] == 1 and attached[1]['kind'] == 'depth_visualization'
    assert attached[1]['meters_per_nonzero_code'] == .01
    assert attached[1]['saturation_depth_m'] == pytest.approx(2.55)
    assert attached[1]['zero_code'] == 'invalid'
    assert 'Attached image 0 is RGB' in call['instructions']
    assert call['images'] == (PNG, PNG+b'depth')


def test_request_advertises_static_canonical_own_token_catalog_even_after_use():
    bridge, clients = setup()
    match = bridge.protocol.match
    # Logical fixture only: test the public catalog, not physical commitment.
    match.rules.play('player_0', 0, 0)
    match.rules.play('player_1', 1, 0)
    match._committed.update(red_24=(0, 0), blue_24=(1, 0))
    bridge.propose('player_0', snapshot(), joint_command=command())
    bridge.propose('player_1', snapshot(), joint_command=command('player_1'), readiness=True)
    for player, color in [('player_0', 'red'), ('player_1', 'blue')]:
        call = clients[player].calls[0]
        request = json.loads(call['prompt'])['request']
        assert request['own_piece_ids'] == [f'{color}_{i:02d}' for i in range(25)]
        assert f'{color}_24' in request['own_piece_ids']
        assert request['piece_id_format'] == color+'_NN with exactly two digits NN=00..24'
        assert 'exact string from request.own_piece_ids' in call['instructions']
        assert all(isinstance(item, str) for item in request['own_piece_ids'])
        assert 'includes used tokens; no poses or availability' in request['piece_catalog_scope']


def test_current_player_authentication_precedes_model_request():
    bridge, clients = setup()
    with pytest.raises(HexBridgeError, match='player_not_current'):
        bridge.propose('player_1', snapshot(), joint_command=command('player_1'))
    assert not clients['player_1'].calls


@pytest.mark.parametrize('bad', [
    '```json\n{}\n```', '{} trailing', '[]',
    '{"placement_intent":{},"placement_intent":{},"joint_targets":{}}',
    '{"placement_intent":{},"joint_targets":{"x":NaN}}',
    '{"placement_intent":{},"joint_targets":{},"reason":"must-not-leak"}',
])
def test_strict_json_rejects_prose_duplicates_constants_and_extra_fields(bad):
    with pytest.raises(HexBridgeError):
        parse_hex_decision(bad)


@pytest.mark.parametrize('mutation', ['wrong_robot', 'changed_target', 'opponent_token', 'wrong_identity', 'boolean_cell'])
def test_invalid_model_decisions_are_atomic_before_reservation(mutation):
    bridge, clients = setup()
    value = decision()
    if mutation == 'wrong_robot':
        value['joint_targets']['robot_id'] = 'right'
    elif mutation == 'changed_target':
        value['joint_targets']['joint_positions']['fr3_left_joint7'] = 1.
    elif mutation == 'opponent_token':
        value['placement_intent']['piece_id'] = 'blue_00'
    elif mutation == 'wrong_identity':
        value['placement_intent']['player_id'] = 'player_1'
    else:
        value['placement_intent']['q'] = True
    clients['player_0'].response = response(value)
    with pytest.raises(HexBridgeError):
        bridge.propose('player_0', snapshot(), joint_command=command())
    assert bridge.protocol.match.pending is None
    assert not bridge.protocol.match.rules.board
    assert bridge.history('player_0') == []
    assert len(clients['player_0'].calls) == 1


@pytest.mark.parametrize('kind,category', [
    ('unknown', 'unknown_piece_identity'), ('opponent', 'wrong_piece_owner'),
    ('used', 'already_committed'), ('occupied', 'occupied_cell'), ('json', 'invalid_model_json')])
def test_rejection_keeps_completed_response_metadata_without_raw_values(kind, category):
    bridge, clients = setup()
    match = bridge.protocol.match
    value = decision()
    if kind == 'unknown':
        value['placement_intent']['piece_id'] = 'must-not-leak'
    elif kind == 'opponent':
        value['placement_intent']['piece_id'] = 'blue_00'
    elif kind == 'used':
        match._committed['red_00'] = (0, 0)
    elif kind == 'occupied':
        match.rules._board[3, 3] = 'blue'
    else:
        value = 'must-not-leak malformed JSON'
    clients['player_0'].response = response(value)
    with pytest.raises(HexBridgeError) as captured:
        bridge.propose('player_0', snapshot(), joint_command=command())
    assert captured.value.category == category
    summary = bridge.summary()
    entry = summary['contexts'][0]['completed_responses'][0]
    assert entry['decision_validation'] == 'rejected'
    assert entry['rejection_category'] == category
    assert entry['model_response']['model'] == 'gpt-6-astra'
    assert entry['model_response']['usage']['input_tokens'] == 123
    assert entry['model_response']['duration_s'] == .5
    assert entry['request_number'] == 1
    assert summary['contexts'][0]['validated_proposals'] == 0
    assert 'must-not-leak' not in json.dumps(summary)
    assert match.pending is None
    entry['model_response']['model'] = 'tampered'
    assert bridge.summary()['contexts'][0]['completed_responses'][0]['model_response']['model'] == 'gpt-6-astra'


def test_opponent_history_is_never_carried_to_other_context_and_getter_is_copy():
    bridge, clients = setup()
    bridge.propose('player_0', snapshot(), joint_command=command())
    bridge.propose('player_0', snapshot(), joint_command=command())
    bridge.propose('player_1', snapshot(), joint_command=command('player_1'), readiness=True)
    assert len(json.loads(clients['player_0'].calls[1]['prompt'])['own_player_history']) == 1
    assert json.loads(clients['player_1'].calls[0]['prompt'])['own_player_history'] == []
    history = bridge.history('player_0')
    history[0]['request']['public_observation']['player_id'] = 'tampered'
    assert bridge.history('player_0')[0]['request']['public_observation']['player_id'] == 'player_0'
    assert not bridge.summary()['cross_player_history_shared']


def test_shared_snapshot_id_cannot_hide_changed_images_or_private_calibration():
    bridge, clients = setup()
    bridge.propose('player_0', snapshot(), joint_command=command(), images=(PNG,))
    with pytest.raises(HexBridgeError, match='snapshot_content_changed'):
        bridge.propose('player_1', snapshot(), joint_command=command('player_1'), images=(PNG+b'other',), readiness=True)
    with pytest.raises(HexBridgeError, match='invalid_public_calibration'):
        bridge.propose('player_1', snapshot(calibration={'private_object_poses': []}),
                       joint_command=command('player_1'), images=(PNG,), readiness=True)
    assert not clients['player_1'].calls


def test_stale_state_and_replay_rejected_before_callback():
    bridge, _ = setup()
    proposal = bridge.propose('player_0', snapshot(), joint_command=command())
    calls = []
    with pytest.raises(HexBridgeError, match='stale_turn_or_snapshot'):
        bridge.execute(proposal, execute_joint_targets=calls.append, published_snapshot=snapshot(snapshot_id='new'))
    assert not calls
    assert bridge.protocol.match.pending is None
    bridge.execute(proposal, execute_joint_targets=calls.append, published_snapshot=snapshot())
    with pytest.raises(HexBridgeError, match='unknown_or_consumed_proposal'):
        bridge.execute(proposal, execute_joint_targets=calls.append, published_snapshot=snapshot())
    assert len(calls) == 1


def test_changed_match_during_model_wait_rejected_without_retry():
    bridge, clients = setup()
    clients['player_0'].during_request = lambda: bridge.protocol.match.begin_move('player_0', 'red_01', 2, 2)
    with pytest.raises(HexBridgeError, match='match_changed_during_model_request'):
        bridge.propose('player_0', snapshot(), joint_command=command())
    assert len(clients['player_0'].calls) == 1
    assert not bridge.history('player_0')


def test_provider_error_and_model_fallback_do_not_leak_or_retry():
    bridge, clients = setup()
    clients['player_0'].error = RuntimeError('provider body contains must-not-leak')
    with pytest.raises(HexBridgeError, match='model_request_failed_no_fallback') as captured:
        bridge.propose('player_0', snapshot(), joint_command=command())
    assert 'must-not-leak' not in str(captured.value)
    assert len(clients['player_0'].calls) == 1
    clients['player_0'].error = None
    clients['player_0'].response = replace(clients['player_0'].response, model='fallback-model')
    with pytest.raises(HexBridgeError, match='unexpected_model_response_contract'):
        bridge.propose('player_0', snapshot(), joint_command=command())
    assert len(clients['player_0'].calls) == 2
    assert not bridge.protocol.match.pending


@pytest.mark.parametrize('category,status', [('http_status', 401), ('http_status', 404),
                                             ('transport_timeout', None), ('transport_error', None)])
def test_gateway_failure_metadata_preserves_safe_category_and_status(category, status):
    bridge, clients = setup()
    clients['player_0'].error = ModelGatewayError(
        'Authorization bearer must-not-leak', category=category, status_code=status,
        request_id='must-not-leak', attempts=1)
    with pytest.raises(HexBridgeError) as captured:
        bridge.propose('player_0', snapshot(), joint_command=command())
    metadata = captured.value.public_metadata()
    assert metadata == {'category': 'model_request_failed_no_fallback',
                        'gateway_error': {'error_category': category, 'http_status': status, 'attempts': 1}}
    assert 'must-not-leak' not in json.dumps(metadata)
    assert 'must-not-leak' not in str(captured.value)
    assert len(clients['player_0'].calls) == 1
    assert not bridge.protocol.match.pending
    metadata['gateway_error']['error_category'] = 'tampered'
    assert captured.value.public_metadata()['gateway_error']['error_category'] == category


def test_callback_error_keeps_pending_move_and_blocks_automatic_replay():
    bridge, _ = setup()
    proposal = bridge.propose('player_0', snapshot(), joint_command=command())

    def fail(_):
        raise RuntimeError('private native must-not-leak')

    with pytest.raises(HexBridgeError, match='joint_execution_failed_no_replay') as captured:
        bridge.execute(proposal, published_snapshot=snapshot(), execute_joint_targets=fail)
    assert 'must-not-leak' not in str(captured.value)
    assert bridge.protocol.match.pending is not None
    assert not bridge.protocol.match.rules.board
    with pytest.raises(HexBridgeError, match='unknown_or_consumed_proposal'):
        bridge.execute(proposal, published_snapshot=snapshot(), execute_joint_targets=fail)
    assert bridge.history('player_0')[0]['execution_status'] == 'callback_failed_pending_move_retained'
