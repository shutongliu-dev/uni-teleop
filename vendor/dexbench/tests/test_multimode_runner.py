"""Cross-layer multi-mode tests. Fake physics is not native acceptance."""
from copy import deepcopy
import json

import pytest

from agent.model_client import runner
from bench.paths import load_json
from test_online_model_runner import setup_loop, Response


def loop(monkeypatch,tmp_path,responses,*,mode='code',rgbd=True):
    args,bench,model=setup_loop(monkeypatch,tmp_path,responses,fingertip=True,rgbd=rgbd)
    args.interaction_mode=mode
    args.tactile_input='on'
    args.max_actions=4
    args.max_program_actions=3
    args.max_tool_calls_per_decision=8
    original=model.generate
    def generate(*a,**kw):
        response=original(*a,**kw)
        value=json.loads(response.text)
        return Response(value) if isinstance(value,str) else response
    model.generate=generate
    return args,bench,model


def test_code_moves_use_fresh_geometry_and_separate_primitive_budget(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,[
        "state = observe()\nfor i in range(2):\n    state = move_geometry(wrist_delta_m=[0,0,0.01],duration_s=3.0)\n    if state['feedback']['status'] != 'completed':\n        break",
        "stop(reason='inspected new images')"])
    report=runner.run(args)
    assert report['error_type'] is None
    assert report['model_call_attempts']==2 and report['submitted_actions']==2
    assert report['program_multiple_actions_exercised']
    assert report['online_model_decision_loop_exercised']
    assert report['model_input_observation_sequences']==[0,40]
    assert [x['geometry']['wrist']['position_m'][2] for x in bench.commands]==pytest.approx([1.01,1.02])
    assert bench.create_requests[0]['budgets']['max_actions']==4
    assert bench.create_requests[0]['budgets']['max_model_calls']==2
    program=load_json(args.output/'decisions/0000/program_execution.json')
    assert program['move_calls']==2 and program['tool_calls']==3
    assert program['trace'][-1]['result']['observation']['sequence']==40
    assert not report['task_success']


def test_code_can_branch_on_failed_move_without_script_takeover(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,[
        "state = move_geometry(wrist_delta_m=[0,0,0.01],duration_s=3.0)\nif state['feedback']['status'] == 'failed':\n    state = move_geometry(wrist_delta_m=[0,0,0.005],duration_s=3.0)"])
    bench.reject_geometry_once=True
    report=runner.run(args)
    assert report['error_type'] is None and report['submitted_actions']==2
    assert report['completed_actions']==1 and len(bench.commands)==2
    assert bench.commands[1]['observation_sequence']==0


def test_code_primitive_limit_stops_extra_moves(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,[
        "for i in range(3):\n    move_geometry(wrist_delta_m=[0,0,0.005],duration_s=3.0)"])
    args.max_actions=2
    report=runner.run(args)
    assert report['error_type'] is None and len(bench.commands)==2
    assert report['primitive_attempts']==2 and report['submitted_actions']==2
    assert report['program_status_counts']=={'tool_error':1}
    assert not report['task_success']


def test_code_prevalidation_prevents_partial_execution(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,[
        "move_geometry(wrist_delta_m=[0,0,0.01],duration_s=3.0)\nimport os"])
    report=runner.run(args)
    assert not bench.commands and report['submitted_actions']==0
    assert report['program_status_counts']=={'rejected':1}


def test_tool_actions_preserve_current_feedback_and_images(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,[
        {'tool':'move_geometry','arguments':{'wrist_delta_m':[0.,0.,.01],'duration_s':3.}},
        {'tool':'observe','arguments':{}}],mode='tools')
    report=runner.run(args)
    assert report['error_type'] is None and report['completed_actions']==1
    assert report['tool_calls']==2 and report['read_calls']==1
    second=json.loads(model.requests[1][1][-1]['content'][0]['text'])
    assert second['previous_action_feedback']['result']['observation']['sequence']==20
    assert second['previous_action_feedback']['result']['feedback']['status']=='completed'
    assert sum(x['type']=='image_url' for x in model.requests[1][1][-1]['content'])==4


def test_rgb_point_query_is_rejected_before_depth_tool(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,[
        {'tool':'point_at','arguments':{'camera_id':'task','u':0,'v':0}},
        {'tool':'stop','arguments':{}}],mode='tools')
    args.visual_input='rgb'
    report=runner.run(args)
    assert report['error_type'] is None and not bench.commands
    first=json.loads(model.requests[0][1][-1]['content'][0]['text'])
    assert first['depth_observation'] is None and first['capabilities']['depth_available'] is False
    assert sum(x['type']=='image_url' for x in model.requests[0][1][-1]['content'])==2
    second=json.loads(model.requests[1][1][-1]['content'][0]['text'])
    assert second['previous_action_feedback']['status']=='rejected_by_client'


def test_tactile_off_removes_readings_from_model_and_code(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,["state = observe()\nstop()"])
    args.tactile_input='off'
    # A measured 20 ms window is not the unsampled reset snapshot.
    bench.obs.update(sequence=1, sim_time_s=.02)
    bench.obs['tactile']={'protocol':'rigid_fingertip_contact_proxy_v1','mode':'simulated_rigid_contact_proxy',
        'status':'available','sim_time_s':.02,'window_start_sim_s':0.,'window_duration_s':.02,
        'window_completeness':'assumed_from_native_step_contract','force_calibrated':False,
        'frame_id':'scalar_invariant','coverage':{'rigid':True},
        'fingers':{'thumb':{'status':'valid','rigid_contact_any':True,
                            'normal_impulse_sum_ns':.0246,'mean_normal_load_n':1.23}}}
    for finger in ('index','middle','ring','little'):
        bench.obs['tactile']['fingers'][finger]={'status':'valid','rigid_contact_any':False,
                                               'normal_impulse_sum_ns':0.,'mean_normal_load_n':0.}
    report=runner.run(args)
    assert report['error_type'] is None
    text=json.loads(model.requests[0][1][-1]['content'][0]['text'])
    assert 'tactile' not in text['observation']
    assert text['capabilities']['tactile_available'] is False
    program=load_json(args.output/'decisions/0000/program_execution.json')
    assert 'tactile' not in program['trace'][0]['result']['observation']
    assert 'tactile' in load_json(args.output/'episode_ready.json')['observation']


@pytest.mark.parametrize('mode',['action','tools','code'])
def test_all_interactions_reject_stale_model_context(monkeypatch,tmp_path,mode):
    response={'action':'move_geometry','wrist_delta_m':[0,0,.01],'duration_s':3.}
    if mode=='tools':response={'tool':'move_geometry','arguments':{k:v for k,v in response.items() if k!='action'}}
    if mode=='code':response='move_geometry(wrist_delta_m=[0,0,0.01],duration_s=3.0)'
    args,bench,model=loop(monkeypatch,tmp_path,[response],mode=mode)
    generate=model.generate
    def stale(*a,**kw):
        out=generate(*a,**kw);bench.obs['sequence']+=1;return out
    model.generate=stale
    report=runner.run(args)
    assert report['error_type'] is None and not bench.commands
    assert report['decision_status_counts']=={'rejected_by_client':1}


def test_wrong_experiment_pin_is_rejected_before_model(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,['stop()'])
    args.expected_spec_pins={'task_release_sha256':'f'*64}
    report=runner.run(args)
    assert report['error_type']=='ValueError' and not model.requests and not bench.commands
    assert bench.finished


def test_tool_transport_error_is_reported_as_client_error(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,[
        'move_geometry(wrist_delta_m=[0,0,.01],duration_s=3.0)'])
    def fail(*args,**kw):raise RuntimeError('private-transport-error-token')
    bench.submit=fail
    report=runner.run(args)
    assert report['error_type']=='RuntimeError'
    assert report['submitted_actions']==1 and not report['task_success']
    for path in args.output.rglob('*.json'):
        assert 'private-transport-error-token' not in path.read_text()
