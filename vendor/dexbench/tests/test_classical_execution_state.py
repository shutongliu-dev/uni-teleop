"""Requested completion, fixed-drive recovery and pure public status; CPU only.

Synthetic robot loads are regression inputs, not new physical evidence.
"""
from pathlib import Path
from types import SimpleNamespace
import pickle
import numpy as np
import pytest
from bench.paths import DATA_ROOT
from bench.protocol.schemas import AgentObservation, JointLimit, System0Action
import bench.control.fingertip as c
ASSETS = Path(f'{DATA_ROOT}/assets')

def obs(q,time=0.,episode='test',sequence=None):
 return AgentObservation(episode_id=episode,sequence=round(time/.02) if sequence is None else sequence,sim_time_s=time,instruction='test public task',profile='vision_proprio',proprioception={'joint_positions_rad':q,'joint_velocities_rad_s':dict.fromkeys(q,0.)},cameras=[])


def policy(episode='test',base=(0.,0.,.77)):
 p=c.ClassicalFingertipPolicy(ASSETS,list(base));g=p.geometry
 p.reset(SimpleNamespace(episode_id=episode,observation_profile='vision_proprio'),SimpleNamespace(control_period_s=.02,arm_joints=g.arm.names,hand_joints=g.hand.names,joint_limits={j.name:JointLimit(lower_rad=j.lower,upper_rad=j.upper) for j in g.joints}))
 return p


def home(g):
 q={j.name:float(np.clip(0.,j.lower,j.upper)) for j in g.joints};q.update(zip(g.arm.names,[0.,-.785,0.,-2.356,0.,1.57,.785]));q['right_thumb_CMC_AA']=.08;return q


def action(o,geometry=None,hold=None):
 return System0Action(action_id='unit',observation_sequence=o.sequence,skill='hold' if hold else 'move_geometry',timeout_sim_s=hold or 5.,hold_duration_s=hold,geometry=geometry)


def execute(p,o):
 r=p.step(o)
 if r.status=='running':p.commit_pending_step()
 return r


def seed_nominal(p,q):
 p.previous=np.array([q[n] for n in p.geometry.names]);p.velocity=np.zeros_like(p.previous);p.goal=p.previous.copy();p._has_goal=True


def test_omitted_loaded_hand_is_not_requested_pose_completion():
 p=policy();q=home(p.geometry);seed_nominal(p,q);loaded=q|{'right_thumb_CMC_AA':.02};o=obs(loaded)
 p.begin(action(o,{'wrist':p.public_geometry(loaded)['wrist']}),o)
 assert p.goal[p.geometry.names.index('right_thumb_CMC_AA')]==.08
 assert p._reached(loaded)
 bad=dict(loaded);bad['fr3_joint1']+=.1;assert not p._reached(bad)
 assert p._convergence_diagnostic(loaded,5.).joint_name is None


@pytest.mark.parametrize('kind',['tip','posture'])
def test_selected_loaded_objective_still_fails(kind):
 p=policy();q=home(p.geometry);seed_nominal(p,q);o=obs(q)
 target=p.public_geometry(q)
 geo=({'fingertip_positions_m':{'thumb':target['fingertip_positions_m']['thumb']}} if kind=='tip' else {'hand_posture_rad':{'right_thumb_CMC_AA':.08}})
 p.begin(action(o,geo),o)
 assert not p._reached(q|{'right_thumb_CMC_AA':-.08})


def test_tip_plus_posture_remains_redundancy_reference():
 p=policy();q=home(p.geometry);o=obs(q)
 tip=p.public_geometry(q)['fingertip_positions_m']['thumb']
 p.begin(action(o,{'fingertip_positions_m':{'thumb':tip},'hand_posture_rad':{'right_thumb_CMC_AA':.081}}),o)
 assert p._posture_objective_names==()
 assert p._reached(q)


@pytest.mark.parametrize('sign',[-1,1])
def test_raw_admissibility_is_per_tick_and_commands_strict(sign):
 p=policy();q=home(p.geometry);name='right_ring_PIP';j=p.geometry.by_name[name];edge=j.lower if sign<0 else j.upper
 q[name]=edge+sign*.0008;o=obs(q);raw=dict(q)
 p.begin(action(o,hold=.4),o);r=execute(p,o)
 assert q==raw and p.public_geometry(q)==p.geometry.public_geometry(raw)
 assert all(p.geometry.by_name[n].lower<=v<=p.geometry.by_name[n].upper for n,v in r.command.joint_positions_rad.items())
 broken=q|{name:edge+sign*.03001};r=p.step(obs(broken,.02))
 assert r.status=='failed' and r.diagnostic.code=='invalid_robot_state' and r.command is None


def test_explicit_posture_does_not_inherit_measured_allowance():
 p=policy();q=home(p.geometry);o=obs(q)
 with pytest.raises(ValueError):p.begin(action(o,{'hand_posture_rad':{'right_ring_PIP':-.0005}}),o)


def test_no_compensated_drive_ratchet_through_recovery_holds_and_wrist():
 p=policy();q=home(p.geometry);name='right_thumb_CMC_AA';q[name]=.003305643331259489
 ref=q|{name:.11987570959206513};seed_nominal(p,ref);p.integral[:]=.012
 now=0.;anchor=p.previous.copy()
 for _ in range(4):
  p.cancel('synthetic_loaded_failure');o=obs(q,now);p.begin(action(o,hold=.4),o)
  for tick in range(21):
   r=execute(p,obs(q,now+tick*.02))
   if tick<20:
    assert r.status=='running';np.testing.assert_array_equal(p.previous,anchor)
   else:assert r.status=='completed'
  now+=.4
 o=obs(q,now);p.begin(action(o,{'wrist':p.public_geometry(q)['wrist']}),o)
 assert p._fixed_drive_mask[p.geometry.names.index(name)]
 for tick in range(251):
  r=execute(p,obs(q,now+tick*.02));assert p.previous[p.geometry.names.index(name)]==anchor[p.geometry.names.index(name)]
  if r.status!='running':break
 assert r.status=='completed'
 assert p.integral[p.geometry.names.index(name)]==0


def test_explicit_reselection_leaves_fixed_drive_without_command_jump():
 p=policy();q=home(p.geometry);seed_nominal(p,q);p._fixed_drive_mask[:]=True;p._fixed_drive_goal=p.previous.copy();o=obs(q)
 p.begin(action(o,{'hand_posture_rad':{'right_thumb_CMC_AA':.10}}),o)
 idx=p.geometry.names.index('right_thumb_CMC_AA');assert not p._fixed_drive_mask[idx]
 before=p.previous.copy();v=p.velocity.copy();r=execute(p,o)
 assert r.status=='running' and np.max(abs(p.velocity-v))<=.06+1e-10
 assert np.max(abs(p.previous-before))<=.02+1e-10


def test_pending_cancel_and_rejected_begin_preserve_drive_modes():
 p=policy();q=home(p.geometry);o=obs(q);p.begin(action(o,{'hand_posture_rad':{'right_thumb_CMC_AA':.15}}),o)
 committed=p.previous.copy();p.step(o);p.cancel('pending_cancel')
 np.testing.assert_array_equal(p.previous,committed)
 before=pickle.dumps(p.__dict__)
 with pytest.raises(ValueError):p.begin(action(o,{'fingertip_positions_m':{'thumb':[9,9,9]}}),o)
 assert pickle.dumps(p.__dict__)==before
 p.begin(action(o,hold=.4),o);assert np.all(p._fixed_drive_mask)


def test_zero_step_accepted_goal_cancel_never_becomes_recovery_anchor():
 p=policy();q=home(p.geometry);seed_nominal(p,q);anchor=p.previous.copy();o=obs(q)
 p.begin(action(o,{'hand_posture_rad':{'right_thumb_CMC_AA':.2}}),o)
 assert p.goal[p.geometry.names.index('right_thumb_CMC_AA')]==.2
 p.cancel('cancel_before_first_step');p.begin(action(o,hold=.4),o)
 np.testing.assert_array_equal(p.goal,anchor);assert np.all(p._fixed_drive_mask)


def test_nonfinite_mid_action_is_safe_typed_failure_without_command():
 p=policy();q=home(p.geometry);o=obs(q);p.begin(action(o,hold=.4),o);execute(p,o)
 broken=obs(q,.02);broken.proprioception.joint_positions_rad['right_ring_PIP']=float('nan')
 r=p.step(broken);assert r.status=='failed' and r.diagnostic.code=='invalid_robot_state' and r.command is None


def test_recovery_hold_all_arm_then_omitted_wrist_restores_arm_tracking():
 p=policy();q=home(p.geometry);seed_nominal(p,q);p.cancel('unit_failure');o=obs(q);p.begin(action(o,hold=.02),o)
 assert np.all(p._fixed_drive_mask);execute(p,o);assert execute(p,obs(q,.02)).status=='completed'
 o=obs(q,.02);p.begin(action(o,{'hand_posture_rad':{'right_index_PIP':.05}}),o)
 assert not np.any(p._fixed_drive_mask[:len(p.geometry.arm.names)])
 assert p._fixed_drive_mask[p.geometry.names.index('right_thumb_CMC_AA')]
 assert not p._fixed_drive_mask[p.geometry.names.index('right_index_PIP')]
 assert p.requested_wrist[:3,3]==pytest.approx(p.public_geometry(q)['wrist']['position_m'])


def test_pure_state_reports_time_and_omitted_load_without_claiming_grasp():
 p=policy();q=home(p.geometry);seed_nominal(p,q);loaded=q|{'right_thumb_CMC_AA':.02};o=obs(loaded)
 p.begin(action(o,{'wrist':p.public_geometry(loaded)['wrist']}),o)
 before=pickle.dumps(p.__dict__);state=p.execution_state(o)
 assert pickle.dumps(p.__dict__)==before
 assert state.sim_time_s==0 and state.objective_status=='reached'
 assert state.completion_scope=='requested_geometry' and state.reference_stationary is True
 assert state.maintained_joint_error_rad['right_thumb_CMC_AA']==pytest.approx(.06)
 assert set(state.maintained_joint_error_rad)==set(p.geometry.hand.names)
 assert state.posture_objective_joint_names==[] and state.recovery_fixed_drive_joint_names==[]
 assert set(state.model_dump())=={'protocol','sim_time_s','completion_scope','objective_status','reference_stationary','posture_objective_joint_names','recovery_fixed_drive_joint_names','maintained_joint_error_rad'}


def test_pending_reference_velocity_not_reported_as_executed():
 p=policy();q=home(p.geometry);o=obs(q);p.begin(action(o,{'hand_posture_rad':{'right_thumb_CMC_AA':.1}}),o)
 before=p.velocity.copy();result=p.step(o)
 assert result.status=='running' and np.array_equal(before,np.zeros(29)) and np.any(abs(p.velocity)>1e-10)
 assert result.robot_execution_state.sim_time_s==0 and result.robot_execution_state.reference_stationary is True
 snapshot=pickle.dumps(p.__dict__);assert p.execution_state(o).reference_stationary is True;assert pickle.dumps(p.__dict__)==snapshot
 p.commit_pending_step();next_obs=obs(result.command.joint_positions_rad,.02)
 assert p.execution_state(next_obs).sim_time_s==.02 and p.execution_state(next_obs).reference_stationary is False


def test_terminal_hold_is_time_maintenance_not_objective_reached():
 p=policy();q=home(p.geometry);o=obs(q);p.begin(action(o,hold=.4),o)
 for i in range(20):assert execute(p,obs(q,i*.02)).status=='running'
 result=p.step(obs(q,.4));state=result.robot_execution_state
 assert result.status=='completed' and state.sim_time_s==.4 and state.reference_stationary is True
 assert state.completion_scope=='timed_reference_maintenance' and state.objective_status=='not_checked'
 assert set(state.maintained_joint_error_rad)==set(q)


def test_terminal_move_reports_actual_goal_reached_and_empty_selected_maintenance():
 p=policy();q=home(p.geometry);o=obs(q);tips=p.public_geometry(q)['fingertip_positions_m'];p.begin(action(o,{'fingertip_positions_m':tips}),o)
 result=execute(p,o);assert result.status=='running'
 result=p.step(obs(q,.02));assert result.status=='completed'
 assert result.robot_execution_state.objective_status=='reached'
 assert result.robot_execution_state.maintained_joint_error_rad=={}


def test_recovery_all29_fixed_then_geometry_arm_and_selected_exit():
 p=policy();q=home(p.geometry);seed_nominal(p,q);p.cancel('test');o=obs(q);p.begin(action(o,hold=.4),o)
 state=p.execution_state(o);assert set(state.recovery_fixed_drive_joint_names)==set(q)
 p.cancel('test');p.begin(action(o,{'hand_posture_rad':{'right_thumb_CMC_AA':.08}}),o)
 state=p.execution_state(o);thumb={j.name for j in p.geometry.finger_joints['thumb']}
 assert set(state.recovery_fixed_drive_joint_names)==set(p.geometry.hand.names) - thumb
 assert state.posture_objective_joint_names==['right_thumb_CMC_AA']
 assert set(state.maintained_joint_error_rad)==set(p.geometry.hand.names) - thumb


@pytest.mark.parametrize('bad_value',[float('nan'),-.031])
def test_invalid_raw_feedback_terminal_state_never_reports_convergence(bad_value):
 p=policy();q=home(p.geometry);o=obs(q);p.begin(action(o,hold=.4),o)
 # model_copy deliberately exercises a faulty producer beyond normal wire validation.
 broken=o.model_copy(deep=True);broken.proprioception.joint_positions_rad['right_ring_PIP']=bad_value
 r=p.step(broken);assert r.status=='failed' and r.diagnostic.code=='invalid_robot_state'
 assert r.robot_execution_state.objective_status=='not_checked'
 assert r.robot_execution_state.maintained_joint_error_rad=={}


def test_same_tip_posture_not_falsely_listed_as_joint_objective():
 p=policy();q=home(p.geometry);o=obs(q);tip=p.public_geometry(q)['fingertip_positions_m']['thumb'];p.begin(action(o,{'fingertip_positions_m':{'thumb':tip},'hand_posture_rad':{'right_thumb_CMC_AA':.081,'right_index_PIP':.1}}),o)
 state=p.execution_state(o);assert state.posture_objective_joint_names==['right_index_PIP']
 selected={j.name for fng in ['thumb','index'] for j in p.geometry.finger_joints[fng]}
 assert set(state.maintained_joint_error_rad)==set(p.geometry.hand.names)-selected
 assert state.objective_status=='not_reached'

