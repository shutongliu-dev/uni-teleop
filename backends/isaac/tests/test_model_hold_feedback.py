"""Timed primitive and safe feedback contracts; CPU fakes are not physics acceptance."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from bench.protocol.schemas import ActionDiagnostic, System0Action
from agent.model_client import runner as model_runner
from agent.model_client.actions import ActionValidationError, build_joint_action
from agent.model_client.geometry_actions import build_geometry_action, parse_geometry_action
from agent.model_client.program import ProgramLimits, execute_program
from agent.model_client.tools import parse_tool_call
from test_model_geometry_actions import public_geometry_inputs
from test_model_actions import public_inputs
from test_multimode_runner import loop
from test_online_model_runner import setup_loop


@pytest.mark.parametrize('changes',[
    {'hold_duration_s':None}, {'hold_duration_s':0.}, {'hold_duration_s':True},
    {'hold_duration_s':float('nan')}, {'timeout_sim_s':.2},
    {'object_effect':'hold'}, {'target':{'object':'hidden'}},
    {'geometry':{'hand_posture_rad':{'joint':0.}}},
    {'skill':'move_geometry'},
])
def test_system_hold_rejects_ambiguous_or_invalid_contract(changes):
    payload=dict(action_id='a',observation_sequence=0,skill='hold',hold_duration_s=.3,timeout_sim_s=.3)
    payload.update(changes)
    with pytest.raises(ValidationError):
        System0Action(**payload)


@pytest.mark.parametrize('duration',[.02,.3,5.])
def test_geometry_hold_is_explicit_no_ik_target_and_preserves_input(public_geometry_inputs,duration):
    original=deepcopy(public_geometry_inputs)
    command=build_geometry_action({'action':'hold','duration_s':duration},*public_geometry_inputs,
        action_id='h',expected_sequence=2)
    assert command['skill']=='hold' and command['geometry'] is None
    assert command['hold_duration_s']==command['timeout_sim_s']==duration
    assert public_geometry_inputs==original


@pytest.mark.parametrize('duration',[.01,.03,5.02])
def test_geometry_hold_respects_period_and_action_budget(public_geometry_inputs,duration):
    with pytest.raises(ActionValidationError):
        build_geometry_action({'action':'hold','duration_s':duration},*public_geometry_inputs,
            action_id='h',expected_sequence=2)


def test_geometry_hold_requires_advertised_skill(public_geometry_inputs):
    obs,caps,spec=public_geometry_inputs
    caps['system0']['skills'].remove('hold');spec['system0']['skills'].remove('hold')
    with pytest.raises(ActionValidationError):
        build_geometry_action({'action':'hold','duration_s':.3},obs,caps,spec,action_id='h',expected_sequence=2)


@pytest.mark.parametrize('duration',[.02,.3,5.])
def test_joint_hold_has_same_timed_duration_range(public_inputs,duration):
    obs,caps,spec=public_inputs
    spec['budgets']['max_action_sim_s']=5.
    spec['budgets']['sim_time_s']=None
    action=build_joint_action({'action':'hold','duration_s':duration},obs,caps,spec,
        action_id='h',expected_sequence=obs['sequence'])
    assert action['duration_s']==duration
    assert action['arm_joint_positions_rad']|action['hand_joint_positions_rad']==obs['proprioception']['joint_positions_rad']


@pytest.mark.parametrize('extra',[{'wrist_delta_m':[0,0,0]}, {'hold_duration_s':.3}, {'target':'object'}])
def test_hold_cannot_hide_a_geometry_or_oracle_request(extra):
    with pytest.raises(ActionValidationError):
        parse_geometry_action({'action':'hold','duration_s':.3,**extra})
    with pytest.raises(ActionValidationError):
        parse_tool_call({'tool':'hold','arguments':{'duration_s':.3,**extra}})


@pytest.mark.parametrize('mode',['joints','action','tools','code'])
def test_all_modes_hold_counts_one_primitive_and_advances_requested_time(monkeypatch,tmp_path,mode):
    if mode in ('tools','code'):
        text='hold(duration_s=0.3)' if mode=='code' else {'tool':'hold','arguments':{'duration_s':.3}}
        args,bench,model=loop(monkeypatch,tmp_path,[text],mode=mode)
    else:
        args,bench,model=setup_loop(monkeypatch,tmp_path,[{'action':'hold','duration_s':.3}],fingertip=mode=='action')
    before=bench.obs['sim_time_s']
    report=model_runner.run(args)
    assert report['error_type'] is None
    assert report['primitive_attempts']==report['submitted_actions']==report['completed_actions']==1
    assert bench.obs['sim_time_s']-before==pytest.approx(.3)
    assert len(bench.commands)==1
    if mode!='joints':assert bench.commands[0]['skill']=='hold'


def test_hold_and_move_share_program_move_budget():
    calls=[]
    def invoke(name,arguments):
        calls.append(name)
        return {'feedback':{'status':'completed','action_id':str(len(calls))}}
    result=execute_program('hold(duration_s=0.3)\nmove_geometry(wrist_delta_m=[0,0,.001],duration_s=1)\nhold(duration_s=0.3)',
        invoke,limits=ProgramLimits(max_move_calls=2))
    assert result['status']=='budget_exceeded' and calls==['hold','move_geometry']
    assert result['move_calls']==2 and len(result['primitive_results'])==2


def test_positional_hold_duration_has_identical_callback_contract():
    calls=[]
    result=execute_program('hold(.3, reason="maintain")',lambda name,args:calls.append((name,args)))
    assert result['status']=='completed'
    assert calls==[('hold',{'duration_s':.3,'reason':'maintain'})]


@pytest.mark.parametrize('source',['hold()', 'hold(.3,duration_s=.3)', 'hold(.3,.4)',
                                   'hold(duration_s=.3,wrist_delta_m=[0,0,0])'])
def test_hold_signature_errors_precede_any_callback(source):
    calls=[]
    result=execute_program(source,lambda *args:calls.append(args))
    assert result['status']=='rejected' and calls==[]


def test_early_failed_primitive_survives_later_success_and_observe_in_next_model_input(monkeypatch,tmp_path):
    args,bench,model=loop(monkeypatch,tmp_path,[
        'move_geometry(wrist_delta_m=[0,0,.001],duration_s=1)\nmove_geometry(wrist_delta_m=[0,0,.001],duration_s=1)\nobserve()',
        'stop()'])
    bench.reject_geometry_once=True
    report=model_runner.run(args)
    assert report['error_type'] is None
    data=json.loads(model.requests[1][1][-1]['content'][0]['text'])
    feedback=data['previous_action_feedback']
    assert feedback['last_result']['feedback']['status']=='completed'
    assert [x['feedback']['status'] for x in feedback['primitive_results']]==['failed','completed']
    assert len({x['feedback']['action_id'] for x in feedback['primitive_results']})==2


@pytest.mark.parametrize('mode',['tools','code'])
def test_safe_public_validation_reason_survives_each_mode(monkeypatch,tmp_path,mode):
    decision=('move_geometry(wrist_delta_m=[.06,0,0],duration_s=1)' if mode=='code' else
              {'tool':'move_geometry','arguments':{'wrist_delta_m':[.06,0,0],'duration_s':1}})
    stop='stop()' if mode=='code' else {'tool':'stop','arguments':{}}
    args,bench,model=loop(monkeypatch,tmp_path,[decision,stop],mode=mode)
    report=model_runner.run(args)
    assert report['error_type'] is None and not bench.commands
    feedback=json.loads(model.requests[1][1][-1]['content'][0]['text'])['previous_action_feedback']
    public=feedback['error']['public_error'] if mode=='code' else feedback['error']
    assert public=={'code':'action_validation_error','message':'wrist translation exceeds the per-decision limit'}
    if mode=='code':
        assert feedback['primitive_results'][0]['error']['public_error']==public


def test_private_callback_exception_stays_sanitized():
    def invoke(*_):raise RuntimeError('PRIVATE backend payload or key')
    result=execute_program('hold(duration_s=.3)',invoke)
    assert result['error']['code']=='tool_callback_failed'
    assert 'PRIVATE' not in json.dumps(result)


def test_executed_primitive_feedback_survives_total_trace_budget_fault():
    calls=[]
    def invoke(name,args):
        calls.append(name)
        return {'feedback':{'action_id':'executed_hold_17','status':'failed',
            'diagnostic':{'code':'convergence_timeout'},'elapsed_sim_s':.3},
            'observation':{'bulk':['x'*4000]*16}}
    result=execute_program('for i in range(16):\n observe()\nhold(.3)',invoke)
    assert calls==['observe']*16+['hold']
    assert result['status']=='budget_exceeded' and result['error']['code']=='trace_bytes_limit'
    assert 'result' not in result['trace'][-1]
    primitive=result['primitive_results'][0]
    assert primitive['call_status']=='error'
    assert primitive['feedback']['action_id']=='executed_hold_17'
    assert primitive['feedback']['status']=='failed'
    assert primitive['feedback']['diagnostic']['code']=='convergence_timeout'
    assert primitive['feedback']['elapsed_sim_s']==.3
    assert 'bulk' not in json.dumps(result['primitive_results'])


@pytest.mark.parametrize('observation',[{'too_large':'x'*100}, {'malformed':object()}])
def test_primitive_minimum_feedback_survives_invalid_whole_result(observation):
    feedback={'action_id':'real_action','status':'completed','elapsed_sim_s':.3,
              'private':'DO_NOT_PROPAGATE'}
    result=execute_program('hold(.3)',lambda *_:{'feedback':feedback,'observation':observation},
        limits=ProgramLimits(max_string_length=50))
    assert result['status'] in ('tool_error','budget_exceeded')
    minimum=result['primitive_results'][0]['feedback']
    assert minimum=={'action_id':'real_action','status':'completed','elapsed_sim_s':.3,'diagnostic':None}
    assert 'DO_NOT_PROPAGATE' not in json.dumps(result['primitive_results'])


@pytest.mark.parametrize('changes',[{'code':'arbitrary private text'}, {'object_pose':[1,2,3]},
    {'joint_error_rad':-1.}, {'joint_error_rad':float('nan')}, {'fingertip_errors_m':{'secret':1.}}])
def test_public_diagnostic_has_no_private_or_unbounded_fields(changes):
    with pytest.raises(ValidationError):ActionDiagnostic(code='trajectory_limit_guard',**changes) if 'code' not in changes else ActionDiagnostic(**changes)
