"""Per-primitive contact history survives later tools and resource truncation."""
from copy import deepcopy
import json

import pytest
from bench.paths import load_json
from bench.sensors.contact_history import ContactHistory
from agent.model_client import runner as model_runner
from agent.model_client.program import ProgramLimits, execute_program
from test_contact_history import metadata, config, contact, window
from test_multimode_runner import loop


def has_history(value):
    if isinstance(value, dict):
        return 'contact_history' in value or any(has_history(v) for v in value.values())
    if isinstance(value, list):
        return any(has_history(v) for v in value)
    return False


@pytest.mark.parametrize('tactile', ['on', 'off'])
def test_two_code_primitives_keep_each_history_after_observe(monkeypatch, tmp_path, config, tactile):
    args, bench, model = loop(monkeypatch, tmp_path,
        ['hold(duration_s=0.4)\nhold(duration_s=0.4)\nobserve()', 'stop()'], mode='code')
    args.tactile_input = tactile
    original_status = bench.get_action
    histories = []
    def status(episode, action_id, **kwargs):
        result = original_status(episode, action_id, **kwargs)
        number = round(bench.obs['sim_time_s'] / .02)
        history = ContactHistory(config)
        history.begin(action_id, (number-1)*.02)
        link = 'right_index_DP' if not histories else 'right_middle_elastomer'
        history.observe(window(number, [contact(config, link, window=number)]))
        histories.append(history.snapshot())
        return {**result, 'contact_history': deepcopy(histories[-1])}
    bench.get_action = status
    report = model_runner.run(args)
    assert report['error_type'] is None and report['submitted_actions'] == 2
    prompt = json.loads(model.requests[1][1][-1]['content'][0]['text'])
    feedback = prompt['previous_action_feedback']
    assert len(feedback['primitive_results']) == 2
    program = load_json(args.output/'decisions/0000/program_execution.json')
    for index in range(2):
        returned = feedback['primitive_results'][index]['feedback']
        assert returned['action_id'] == histories[index]['action_id']
        if tactile == 'on':
            assert returned['contact_history'] == histories[index]
            assert program['primitive_results'][index]['feedback']['contact_history'] == histories[index]
        else:
            assert not has_history(returned)
    if tactile == 'off':
        assert not has_history(feedback) and not has_history(program)
    sources = list(args.output.glob('decisions/**/execution_source.json'))
    assert len(sources) == 2
    assert all(has_history(load_json(p)) for p in sources)


def test_history_survives_post_execution_trace_limit(config):
    history = ContactHistory(config);history.begin('first', 0.)
    history.observe(window(1, [contact(config, 'right_index_DP', window=1)]))
    feedback = {'status':'completed', 'action_id':'first', 'elapsed_sim_s':.02,
                'sim_time_s':.02, 'contact_history':history.snapshot()}
    calls = []
    def invoke(name, arguments):
        calls.append(name)
        return {'feedback':deepcopy(feedback), 'observation':{'camera_note':'ordinary payload'}}
    result = execute_program('hold(duration_s=0.02)', invoke,
        limits=ProgramLimits(max_runtime_s=None, max_trace_bytes=1024))
    assert calls == ['hold'] and result['status'] == 'budget_exceeded'
    assert result['error']['code'] == 'trace_bytes_limit'
    assert result['primitive_results'][0]['feedback']['contact_history'] == feedback['contact_history']


@pytest.mark.parametrize('bad', ['private_extra', 'wrong_action', 'wrong_shape'])
def test_minimum_history_cannot_copy_private_or_mismatched_payload(config, bad):
    history = ContactHistory(config);history.begin('first', 0.);history.observe(window())
    payload=history.snapshot()
    if bad == 'private_extra':payload['object_id']='private_canary'
    elif bad == 'wrong_action':payload['action_id']='second'
    else:payload['regions']['index_pad']['contact_substeps']=True
    result = execute_program('hold(duration_s=0.02)',
        lambda *_:{'feedback':{'status':'completed','action_id':'first','contact_history':payload}},
        limits=ProgramLimits(max_runtime_s=None))
    minimum = result['primitive_results'][0]['feedback']
    assert minimum['status'] == 'completed' and 'contact_history' not in minimum
    assert 'private_canary' not in json.dumps(minimum)
