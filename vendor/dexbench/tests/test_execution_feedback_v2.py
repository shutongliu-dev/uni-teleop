"""CPU public-contract checks; fake worker responses do not establish physics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from bench.paths import DATA_ROOT
from bench.protocol.schemas import (ActionEvent, HandGoalSemantics, PolicyDescriptor,
    RobotExecutionState)
from agent.model_client import runner as model_runner
from agent.model_client.program import execute_program
from test_agent_service import (Teacher, TimedHoldTeacher, action, frozen_inputs,
    join_action, runner, session_for, skill_action, teacher_session)
from test_classical_service import JointFeedbackPolicy, setup_session
from test_hand_goal_model_contract import descriptor
from test_multimode_runner import loop
from test_online_model_runner import setup_loop


def state(at=0., **changes):
    return dict(protocol='robot_execution_state_v1',sim_time_s=at,
        completion_scope='requested_geometry',objective_status='reached',
        reference_stationary=True,posture_objective_joint_names=[],
        recovery_fixed_drive_joint_names=[],maintained_joint_error_rad={}) | changes


def v2_descriptor():
    # Obtain the advertised contract from the actual new controller, no simulator.
    from bench.control.fingertip import ClassicalFingertipPolicy
    policy=ClassicalFingertipPolicy(f'{DATA_ROOT}/assets',[.4,0,.77])
    return policy.descriptor.model_dump(mode='json')


def test_legacy_event_and_descriptor_decode_without_invented_terminal_evidence():
    event=ActionEvent(action_id='old',status='completed',sim_time_s=.2,wall_time_s=1.)
    assert event.termination_reason is None and event.robot_execution_state is None
    assert isinstance(PolicyDescriptor.model_validate(descriptor()).hand_goal_semantics,HandGoalSemantics)


@pytest.mark.parametrize('value',[0,1,True,'false',None])
def test_v2_measured_reference_claim_requires_exact_false(value):
    d=v2_descriptor();d['hand_goal_semantics']['measured_reference']['raw_feedback_projected']=value
    with pytest.raises(ValidationError):PolicyDescriptor.model_validate(d)


@pytest.mark.parametrize('mode',['action','tools','code'])
def test_v2_prompt_exact_advertised_scope_and_legacy_branch(mode):
    d=v2_descriptor();PolicyDescriptor.model_validate(d)
    prompt=model_runner.model_system_prompt('fingertip',mode,d)
    assert 'persistent recovery fixed-drive' in prompt and '0.03 rad' in prompt
    assert 'posture on the same finger' in prompt and 'not a hardware safety limit' in prompt
    assert model_runner.HAND_REFERENCE_RULES not in prompt
    old=model_runner.model_system_prompt('fingertip',mode,descriptor())
    assert model_runner.HAND_REFERENCE_RULES in old and 'persistent recovery fixed-drive' not in old
    joints=model_runner.model_system_prompt('joints','action',d)
    assert 'persistent recovery fixed-drive' not in joints and 'direct_duration_completed' in joints


@pytest.mark.parametrize('changes',[
    {'reference_stationary':0},{'sim_time_s':True},
    {'maintained_joint_error_rad':{'j':float('nan')}},
    {'maintained_joint_error_rad':{'j':-.1}},
    {'posture_objective_joint_names':['j','j']},{'object_position':[1,2,3]},
])
def test_execution_state_rejects_ambiguous_types_or_private_fields(changes):
    with pytest.raises(ValidationError):RobotExecutionState.model_validate(state(**changes))


def test_service_state_filters_capability_names_clock_and_residual_envelope(runner):
    make,_,_=runner;s,_=session_for(make());j=next(iter(s.capabilities.joint_limits))
    good=state(maintained_joint_error_rad={j:.1})
    assert s._safe_execution_state(good).model_dump(mode='json')==good
    for bad in [state(.02),state(maintained_joint_error_rad={'object_secret':.1}),
                state(maintained_joint_error_rad={j:100.})]:
        assert s._safe_execution_state(bad) is None
    mutable=RobotExecutionState.model_validate(good)
    mutable.maintained_joint_error_rad[j]=float('nan')
    assert s._safe_execution_state(mutable) is None


def test_direct_interval_and_controller_final_check_have_distinct_reasons(runner):
    make,_,_=runner;s,_=session_for(make())
    s.submit(action(s));join_action(s)
    assert s.status('move_1')['termination_reason']=='direct_duration_completed'
    p=JointFeedbackPolicy(reach_after_last_step=True)
    original=p.step
    def with_state(obs):
        result=original(obs)
        result.robot_execution_state=RobotExecutionState.model_validate(state(obs.sim_time_s))
        return result
    p.step=with_state
    s=setup_session(make,p);s.submit(skill_action(s));join_action(s)
    feedback=s.status('skill_1')
    assert feedback['termination_reason']=='controller_converged'
    assert feedback['robot_execution_state']['sim_time_s']==pytest.approx(.04)
    assert feedback['elapsed_sim_s']==pytest.approx(.04)


def test_task_success_interrupt_does_not_call_policy_again_or_claim_hold_completed(runner):
    make,_,workers=runner
    p=TimedHoldTeacher();p.step_count=0;original=p.step
    def counted(obs):
        p.step_count+=1
        result=original(obs)
        result.robot_execution_state=RobotExecutionState.model_validate(state(obs.sim_time_s))
        return result
    p.step=counted
    s=teacher_session(make,p)
    workers[-1].on_step=lambda w:w.score.update(success=True)
    s.submit(dict(kind='system0',action_id='h',observation_sequence=0,skill='hold',
        hold_duration_s=.3,timeout_sim_s=.3));join_action(s)
    fb=s.status('h')
    assert p.step_count==1 and fb['status']=='completed'
    assert fb['termination_reason']=='task_terminated' and fb['robot_execution_state'] is None
    assert fb['elapsed_sim_s']==pytest.approx(.02) and s.result.outcome=='success'


@pytest.mark.parametrize('reject',[False,True])
def test_failed_and_rejected_are_not_convergence(runner,reject):
    make,_,_=runner;p=JointFeedbackPolicy(reject_first=reject)
    s=setup_session(make,p);s.submit(skill_action(s));join_action(s)
    fb=s.status('skill_1')
    assert fb['status']=='failed'
    assert fb['termination_reason']==('begin_rejected' if reject else 'controller_failed')


def test_real_timed_hold_gets_maintenance_reason(runner):
    make,_,_=runner;s=teacher_session(make,TimedHoldTeacher())
    s.submit(dict(kind='system0',action_id='h',observation_sequence=0,skill='hold',
        hold_duration_s=.3,timeout_sim_s=.3));join_action(s)
    assert s.status('h')['termination_reason']=='time_hold_completed'
    assert s.status('h')['elapsed_sim_s']==pytest.approx(.3)


def test_post_effect_trace_limit_preserves_new_typed_minimum_feedback():
    fb={'status':'completed','action_id':'h','termination_reason':'task_terminated',
        'robot_execution_state':None,'elapsed_sim_s':.02}
    result=execute_program('for i in range(16):\n observe()\nhold(.3)',
        lambda *_:{'feedback':fb,'observation':{'bulk':['x'*4000]*16}})
    assert result['status']=='budget_exceeded' and result['error']['code']=='trace_bytes_limit'
    assert result['primitive_results'][0]['feedback']==fb|{'diagnostic':None}
    fb.update(termination_reason='controller_failed',robot_execution_state=state(.3,objective_status='not_reached'))
    result=execute_program('hold(.3)',lambda *_:{'feedback':fb})
    assert result['primitive_results'][0]['feedback']['robot_execution_state']==fb['robot_execution_state']


@pytest.mark.parametrize('mode',['joints','action','tools','code'])
def test_four_modes_forward_reason_and_state_to_next_model_and_count_without_changing_budget(monkeypatch,tmp_path,mode):
    if mode in ('tools','code'):
        requests=['hold(.3)','stop()'] if mode=='code' else [
            {'tool':'hold','arguments':{'duration_s':.3}},{'tool':'stop','arguments':{}}]
        args,bench,model=loop(monkeypatch,tmp_path,requests,mode=mode)
    else:
        args,bench,model=setup_loop(monkeypatch,tmp_path,[{'action':'hold','duration_s':.3},
            {'action':'stop'}],fingertip=mode=='action')
    old=bench.get_action
    extra={'termination_reason':'direct_duration_completed' if mode=='joints' else 'time_hold_completed',
        'robot_execution_state':None if mode=='joints' else state(.3,completion_scope='timed_reference_maintenance')}
    bench.get_action=lambda *a,**kw:old(*a,**kw)|deepcopy(extra)
    report=model_runner.run(args)
    assert report['error_type'] is None and report['model_call_attempts']==2
    assert report['primitive_attempts']==report['submitted_actions']==report['completed_actions']==1
    assert report['primitive_termination_reason_counts']=={extra['termination_reason']:1}
    fb=json.loads(model.requests[1][1][-1]['content'][0]['text'])['previous_action_feedback']
    if mode=='code':fb=fb['primitive_results'][0]['feedback']
    elif mode=='tools':fb=fb['result']['feedback']
    assert fb['termination_reason']==extra['termination_reason']
    assert fb['robot_execution_state']==extra['robot_execution_state']
