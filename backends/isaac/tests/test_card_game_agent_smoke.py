"""CPU checks for the public-only native diagnostic; no native qualification."""
from copy import deepcopy
import json

import pytest

from bench.protocol.card_games import INTENT_ADAPTER
from scripts.validation.card_game_agent_smoke import CardDiagnosticModel, sample_intent, validate_action


def action(function='hold', duration=.5):
    return {'ok': True, 'function': function, 'result': {
        'status': 'completed', 'termination_reason': 'time_hold_completed' if function == 'hold' else 'controller_converged',
        'episode_finished': False, 'elapsed_sim_s': duration,
        'contact_history': {'status': 'available', 'hands': {'left': {}, 'right': {}},
                            'sampled_substeps': 1000, 'physics_dt_s': .0005, 'duration_s': .5}}}


@pytest.mark.parametrize('change', [
    lambda value: value.update(ok=False),
    lambda value: value['result'].update(status='failed'),
    lambda value: value['result'].update(termination_reason='convergence_timeout'),
    lambda value: value['result'].update(episode_finished=True),
    lambda value: value['result'].update(elapsed_sim_s=.4),
    lambda value: value['result']['contact_history'].update(status='unavailable'),
    lambda value: value['result']['contact_history'].update(hands={'left': {}}),
    lambda value: value['result']['contact_history'].update(sampled_substeps=999),
])
def test_failed_or_incompletely_sampled_native_action_cannot_pass(change):
    good = action()
    validate_action(good, 'hold', duration=.5)
    bad = deepcopy(good)
    change(bad)
    with pytest.raises(RuntimeError): validate_action(bad, 'hold', duration=.5)


def context(task_id='memory_pairs'):
    pieces = {
        'polyomino_territory': [{'id': 'black_mono', 'owner': 'black', 'kind': 'polyomino', 'shape': 'mono'}],
        'barrier_race': [{'id': 'black_barrier_0', 'owner': 'black', 'kind': 'barrier'}],
        'memory_pairs': [{'id': 'card_00', 'owner': None, 'kind': 'memory_card'}],
        'disk_curling': [{'id': 'black_disk_0', 'owner': 'black', 'kind': 'disk'}],
    }
    return {'spec': {'task_id': task_id}, 'last_feedback': None,
        'state': {'board_game': {'protocol': 'bench_card_game_state_v1', 'current_player': 'player_0',
            'terminal': False, 'commit_count': 0, 'phase': 'awaiting_native_baseline', 'pending_move': None},
            'public_rules': {'metric_design': {'pieces': pieces[task_id]}},
            'player': {'player_id': 'player_0', 'color': 'black', 'hand': 'left', 'joint_names': ['left_joint']},
            'observation': {'sequence': 0, 'sim_time_s': 0.,
                'cameras': [{'camera_id': str(i)} for i in range(5)],
                'proprioception': {'commanded_joint_positions_rad': {'left_joint': 0., 'right_joint': 0.},
                    'hand_geometries': {'left': {'wrist': {'position_m': [.4, -.2, 1.1],
                                                         'orientation_wxyz': [1., 0., 0., 0.]}}}}}}}


def generate(model, value):
    turns = [{'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(value)}]}]
    return json.loads(model.generate('test', turns).text)


@pytest.mark.parametrize('task_id', ['polyomino_territory', 'barrier_race', 'memory_pairs', 'disk_curling'])
@pytest.mark.parametrize('projected', [False, True])
def test_scripted_public_context_generates_full_diagnostic_sequence(task_id, projected):
    value = context(task_id)
    model = CardDiagnosticModel(task_id)
    if projected:
        from agent.model_client.context_projection import project_context
        original_generate = model.generate
        history = []
        def projected_generate(prompt, turns, **kwargs):
            current = json.loads(turns[-1]['content'][0]['text'])
            current['task'] = {'public_rules': deepcopy(current['state']['public_rules'])}
            if current['last_feedback'] is not None:
                history.append({'role': 'user', 'content': json.dumps({'tool_feedback': current['last_feedback']})})
            projected_history, projected_context, _ = project_context(history, current)
            response = original_generate(prompt, projected_history + [{'role': 'user', 'content': [
                {'type': 'text', 'text': json.dumps(projected_context)}]}], **kwargs)
            history.extend([{'role': 'user', 'content': json.dumps(current)},
                            {'role': 'assistant', 'content': response.text}])
            return response
        model.generate = projected_generate
    assert generate(model, value) == {'function': 'hold', 'arguments': {'duration_s': .5, 'mode': 'acknowledged_drive'}}
    obs, game = value['state']['observation'], value['state']['board_game']
    game['phase'] = 'reveal_first' if task_id == 'memory_pairs' else 'awaiting_intent'
    obs.update(sequence=1, sim_time_s=.5)
    value['last_feedback'] = action()
    assert generate(model, value)['function'] == 'query_ik'
    value['last_feedback'] = {'ok': True, 'function': 'query_ik', 'result': {
        'solution_found': True, 'hands': {'left': {'solution_found': True}},
        'advances_physics': False, 'mutates_controller': False}}
    intent = generate(model, value)['intent']
    assert INTENT_ADAPTER.validate_python(intent).type == sample_intent(value)['type']
    game['pending_move'] = {'type': intent['type']}
    value['last_feedback'] = {'ok': True, 'intent': intent['type']}
    assert generate(model, value) == {'intent': {'type': 'cancel_intent'}}
    game['pending_move'] = None
    value['last_feedback'] = {'ok': True, 'intent': 'cancel_intent'}
    move = generate(model, value)
    assert move['function'] == 'move_geometry'
    assert set(move['arguments']['hand_geometries']) == {'left'}
    assert move['arguments']['hand_geometries']['left']['wrist']['position_m'][2] == pytest.approx(1.102)
    obs['proprioception']['hand_geometries']['left']['wrist']['position_m'][2] += .002
    value['last_feedback'] = action('move_geometry', .1)
    assert generate(model, value)['function'] == 'move_geometry'
    obs['proprioception']['hand_geometries']['left']['wrist']['position_m'][2] -= .002
    assert generate(model, value)['arguments'] == {'duration_s': .12, 'mode': 'acknowledged_drive'}
    value['last_feedback'] = action('hold', .12)
    assert generate(model, value) == {'stop': True}
    assert model.calls == 8 and model.public_metadata()['external_model_requests'] == 0


@pytest.mark.parametrize('fault', ['baseline_missing', 'cameras_missing', 'inactive_drive_changed'])
def test_public_diagnostic_stops_on_wrong_native_evidence(fault):
    value = context()
    model = CardDiagnosticModel('memory_pairs')
    if fault == 'baseline_missing':
        value['state']['board_game']['phase'] = 'reveal_first'
    elif fault == 'cameras_missing':
        value['state']['observation']['cameras'].pop()
    else:
        generate(model, value)
        value['state']['board_game']['phase'] = 'reveal_first'
        value['state']['observation']['proprioception']['commanded_joint_positions_rad']['right_joint'] = .01
        value['last_feedback'] = action()
    with pytest.raises(RuntimeError): generate(model, value)
