"""Controller lifecycle boundaries; native geometry is tested separately."""
from copy import deepcopy

import pytest

from bench.protocol import PolicyRegistry
from bench.protocol.policies import PolicyStep
from bench.protocol.schemas import GeometryObjective
from bench.control.fingertip import GeometryIKError
from test_agent_service import (Teacher, frozen_inputs, join_action, runner,
                                session_for, skill_action)


class JointFeedbackPolicy(Teacher):
    def __init__(self, *, reach_after_last_step=False, reject_first=False):
        super().__init__(terminal='running')
        self.descriptor=self.descriptor.model_copy(update={
            'observation_profile':'vision_proprio','camera_observation_mode':'decision_only'})
        self.reach_after_last_step=reach_after_last_step
        self.reject_first=reject_first

    def begin(self, action, observation):
        super().begin(action,observation)
        if self.reject_first:
            self.reject_first=False
            raise GeometryIKError('Requested target has no verified IK solution')

    def step(self, observation):
        if self.reach_after_last_step and observation.sim_time_s>=.04-1e-9:
            self.observations.append(observation)
            return PolicyStep(status='completed')
        return super().step(observation)


def setup_session(make, policy):
    registry=PolicyRegistry();registry.register(policy)
    return session_for(make(registry=registry),track='system0',
        policy_id='test_teacher',policy_version='v1')[0]


@pytest.mark.parametrize('reached',[False,True])
def test_decision_only_controller_uses_final_feedback_without_extra_step_or_camera(runner,reached):
    make,_,workers=runner
    policy=JointFeedbackPolicy(reach_after_last_step=reached)
    session=setup_session(make,policy)
    session.submit(skill_action(session));join_action(session)
    assert session.result is None
    assert session.status('skill_1')['status']==('completed' if reached else 'failed')
    assert session.sequence==2
    assert len([c for c in workers[-1].calls if c[0]=='step'])==2
    assert len([c for c in workers[-1].calls if c[0]=='cameras'])==2
    assert all(not obs.cameras for obs in policy.observations[1:])
    assert session.snapshot.sim_time_s==pytest.approx(.04)


def test_geometry_rejection_keeps_episode_available_for_new_decision(runner):
    make,_,workers=runner
    policy=JointFeedbackPolicy(reject_first=True,reach_after_last_step=True)
    session=setup_session(make,policy)
    session.submit(skill_action(session));join_action(session)
    assert session.result is None and session.sequence==0
    assert session.status('skill_1')['status']=='failed'
    assert not [c for c in workers[-1].calls if c[0]=='step']
    retry=deepcopy(skill_action(session));retry['action_id']='retry'
    session.submit(retry);join_action(session)
    assert session.status('retry')['status']=='completed'
    assert session.result is None


def test_fingertips_are_independent_from_contact_and_force_objectives():
    value=GeometryObjective(fingertip_positions_m={'thumb':[.02,.04,.12]})
    assert value.fingertip_frame_id=='right_hand_flange'
    assert not value.contacts
    with pytest.raises(ValueError):
        GeometryObjective(fingertip_positions_m={'pinky':[.02,.04,.12]})
