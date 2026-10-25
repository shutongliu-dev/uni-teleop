"""Declared robot goal priors must remain typed and visible through negotiation."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from bench.protocol.schemas import PolicyDescriptor,RobotCapabilities


def descriptor():
    return {'policy_id':'classical_fingertip', 'version':'candidate', 'implementation_sha256':'a'*64,
            'trained':False, 'robot_posture_profile':{
                'profile_id':'sharpa_selected_finger_goal', 'max_distal_minus_proximal_rad':.5,
                'joint_pairs':[{'distal_joint':'right_index_DIP', 'proximal_joint':'right_index_PIP'}]}}


def test_public_policy_describes_robot_constraint_and_preserves_legacy_empty_default():
    value = descriptor()
    row = PolicyDescriptor.model_validate(value).model_dump(mode='json')
    assert row['robot_posture_profile']['max_distal_minus_proximal_rad'] == .5
    assert row['robot_posture_profile']['scope'] == 'requested_finger_goals_only'
    assert row['robot_posture_profile']['grasp_success_guarantee'] is False
    value.pop('robot_posture_profile')
    assert PolicyDescriptor.model_validate(value).robot_posture_profile is None


@pytest.mark.parametrize('bad', [True, '0.5', float('nan'), float('inf'), -.1])
def test_robot_difference_bound_is_finite_nonnegative_numeric(bad):
    value = descriptor()
    value['robot_posture_profile']['max_distal_minus_proximal_rad'] = bad
    with pytest.raises(ValidationError):
        PolicyDescriptor.model_validate(value)


def test_robot_prior_rejects_unknown_oracle_payload_and_degenerate_joint_pair():
    value = descriptor()
    bad = deepcopy(value)
    bad['robot_posture_profile']['joint_pairs'][0]['private_object_pose'] = [1,2,3]
    with pytest.raises(ValidationError):
        PolicyDescriptor.model_validate(bad)
    value['robot_posture_profile']['joint_pairs'][0]['proximal_joint'] = 'right_index_DIP'
    with pytest.raises(ValidationError):
        PolicyDescriptor.model_validate(value)


@pytest.mark.parametrize('field',['measured_state_projected','grasp_success_guarantee'])
@pytest.mark.parametrize('value',[0,0.,1,True,'false',None])
def test_posture_claims_require_exact_false_not_falsey_coercion(field,value):
    payload=descriptor()
    payload['robot_posture_profile'][field]=value
    with pytest.raises(ValidationError,match='literal boolean false'):
        PolicyDescriptor.model_validate(payload)


@pytest.mark.parametrize('pair',[{'proximal_joint':'fr3_joint1','distal_joint':'right_index_DIP'},
                              {'proximal_joint':'unknown','distal_joint':'right_index_DIP'}])
def test_posture_pairs_must_reference_advertised_hand_joints(pair):
    payload=descriptor();payload['robot_posture_profile']['joint_pairs']=[pair]
    with pytest.raises(ValidationError,match='advertised hand joints'):
        RobotCapabilities(robot_id='test',robot_asset_sha256='b'*64,arm_joints=['fr3_joint1'],
            hand_joints=['right_index_PIP','right_index_DIP'],
            joint_limits={n:{'lower_rad':-2.,'upper_rad':2.} for n in ['fr3_joint1','right_index_PIP','right_index_DIP']},
            tracks=['direct_joints','system0'],system0=payload)


def test_posture_profile_reaches_model_capabilities_without_projection():
    from agent.model_client.runner import decision_text
    from agent.model_client.geometry_actions import GeometryActionLimits
    from agent.model_client.visibility import capability_view
    payload=descriptor();payload['robot_posture_profile'].update(measured_state_projected=False,grasp_success_guarantee=False)
    caps={'system0':PolicyDescriptor.model_validate(payload).model_dump(mode='json'),
          'depth_available':True,'tactile_available':False,'tactile_mode':'unavailable'}
    encoded=decision_text({},capability_view(caps,visual_input='rgbd',tactile_input='on'),{},GeometryActionLimits(),None,2)
    import json
    assert json.loads(encoded)['capabilities']['system0']['robot_posture_profile']==caps['system0']['robot_posture_profile']
