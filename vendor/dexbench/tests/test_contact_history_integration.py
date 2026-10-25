"""Exercise real lifecycle/client paths with explicit CPU sensor doubles."""
from copy import deepcopy
import json

import pytest

from bench.paths import load_json
from bench.sensors.contact_history import ContactHistory
from bench.sensors.contact_schema import ContactHistoryObservation
from bench.runtime import service
from agent.model_client import runner as model_runner
from agent.model_client.visibility import feedback_view
from test_agent_service import frozen_inputs, runner, session_for, action, join_action
from test_contact_history import metadata, config, contact, window
from test_multimode_runner import loop
from test_online_model_runner import setup_loop


def test_service_retains_contacts_missed_by_final_window_and_freezes_old_actions(runner, monkeypatch, config):
    make, _, _ = runner
    monkeypatch.setattr(service, 'configure_contact_history', lambda *_: deepcopy(config))
    def step(worker):
        number = round(worker.raw['sim_time_s'] / .02)
        rows = [contact(config, tick=5), contact(config, 'right_index_elastomer', tick=5)] if number == 1 else []
        worker.raw.update(window(number, rows))
    session, _ = session_for(make(on_step=step))
    payload = action(session, 'first', .06)
    session.submit(payload); join_action(session)
    first = session.status('first')['contact_history']
    ContactHistoryObservation.model_validate(first)
    assert first['sampled_substeps'] == 120
    assert first['pad_coactivity']['thumb+index']['contact_substeps'] == 1
    assert first['regions']['thumb_pad']['current_contact_substeps'] == 0
    session.submit(payload)  # Idempotent retry must not reset history.
    assert session.status('first')['contact_history'] == first
    session.submit(action(session, 'second', .02)); join_action(session)
    second = session.status('second')['contact_history']
    assert second['sampled_substeps'] == 40 and second['start_sim_time_s'] == .06
    assert second['regions']['thumb_pad']['contact_substeps'] == 0
    assert session.status('first')['contact_history'] == first
    assert 'private_' not in json.dumps(first)


@pytest.mark.parametrize('terminal', [False, True])
def test_terminal_step_and_invalid_gap_are_not_dropped(runner, monkeypatch, config, terminal):
    make, _, _ = runner
    monkeypatch.setattr(service, 'configure_contact_history', lambda *_: deepcopy(config))
    def step(worker):
        number = round(worker.raw['sim_time_s'] / .02)
        worker.raw.update(window(number, [contact(config, 'right_index_DP', window=number)]))
        if terminal:
            worker.score['success'] = True
        elif number == 2:
            worker.raw['contacts']['errors'] = ['private_sensor_canary']
    session, _ = session_for(make(on_step=step))
    session.submit(action(session, 'move', .06)); join_action(session)
    feedback = session.status('move')
    history = feedback['contact_history']
    ContactHistoryObservation.model_validate(history)
    assert 'private_sensor_canary' not in json.dumps(history)
    if terminal:
        assert feedback['termination_reason'] == 'task_terminated'
        assert history['sampled_substeps'] == 40
        assert history['regions']['index_shell']['contact_substeps'] == 1
    else:
        assert history['status'] == 'invalid'
        assert history['sampled_substeps'] == 40
        assert history['sim_time_s'] == .06
        assert history['regions'] is None


def has_history(value):
    if isinstance(value, dict):
        return 'contact_history' in value or any(has_history(v) for v in value.values())
    if isinstance(value, list):
        return any(has_history(v) for v in value)
    return False


@pytest.mark.parametrize('track', ['joints_action', 'fingertip_action', 'fingertip_tools', 'fingertip_code'])
@pytest.mark.parametrize('tactile', ['on', 'off'])
def test_all_four_clients_apply_contact_history_ablation(monkeypatch, tmp_path, config, track, tactile):
    mode = track.split('_')[-1]
    if track == 'joints_action':
        args, bench, model = setup_loop(monkeypatch, tmp_path, [
            {'action': 'hold', 'duration_s': .4}, {'action': 'stop'}], rgbd=True)
        args.interaction_mode = 'action'
    else:
        responses = {
            'action': [{'action': 'hold', 'duration_s': .4}, {'action': 'stop'}],
            'tools': [{'tool': 'hold', 'arguments': {'duration_s': .4}}, {'tool': 'stop', 'arguments': {}}],
            'code': ['state = hold(duration_s=0.4)\nstate = observe()', 'stop()'],
        }[mode]
        args, bench, model = loop(monkeypatch, tmp_path, responses, mode=mode)
    args.tactile_input = tactile
    original_status = bench.get_action
    def status(episode, action_id, **kwargs):
        result = original_status(episode, action_id, **kwargs)
        history = ContactHistory(config)
        number = round(bench.obs['sim_time_s'] / .02)
        history.begin(action_id, (number-1)*.02)
        history.observe(window(number, [contact(config, 'right_index_DP', window=number)]))
        return {**result, 'contact_history': history.snapshot()}
    bench.get_action = status
    report = model_runner.run(args)
    assert report['error_type'] is None and report['submitted_actions'] == 1
    second = json.loads(model.requests[1][1][-1]['content'][0]['text'])
    assert has_history(second['previous_action_feedback']) == (tactile == 'on')
    if mode == 'code':
        program = load_json(args.output/'decisions/0000/program_execution.json')
        assert has_history(program['trace']) == (tactile == 'on')
    originals = list(args.output.glob('decisions/**/execution_source.json'))
    assert originals and all(has_history(load_json(path)) for path in originals)


def test_feedback_rejects_wrong_action_and_private_extra(config):
    from agent.model_client.bench_http import BenchHTTPError
    history = ContactHistory(config); history.begin('one', 0.); history.observe(window())
    value = {'action_id': 'two', 'status': 'completed', 'sim_time_s': .02, 'contact_history': history.snapshot()}
    with pytest.raises(BenchHTTPError):
        feedback_view(value)
    value['action_id'] = 'one'
    value['contact_history']['object_id'] = 'private_object_canary'
    with pytest.raises(BenchHTTPError):
        feedback_view(value)
