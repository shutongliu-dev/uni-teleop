from copy import deepcopy

import pytest

from scenes.workrooms.robot_revision import REVISION_SCHEMA
from scenes.workrooms.robot_revision.inputs import canonical_hash, derive_inputs


@pytest.fixture
def inputs():
    initial = {'joint_positions_rad': {'fr3_joint1': .2, 'finger1': .1},
               'cloths_vertices_m': {'cloth': [[.1, .2, .3]]}}
    case = {'schema': 'workroom_reset_case.v1', 'case_fingerprint': '1'*64,
            'physical_geometry_fingerprint': '2'*64,
            'actual_body_poses_m': {'piece': {'position_m': [.1, .2, .3], 'quaternion_wxyz': [1., 0., 0., 0.]}}}
    case['case_sha256'] = canonical_hash(case)
    task = {'id': 'parts_sorting', 'initial_state': deepcopy(initial), 'reset_case': case,
            'public_rules': {'threshold': .004}, 'public_goal': {'target': [1., 0., .8]}}
    scene = {'robot': {'hand_id': 'sharpa_w01_right', 'assembly_version': 'old',
                'base_position_m': [0., 0., .8], 'mount_position_m': [0., 0., 0.],
                'mount_quaternion_wxyz': [1., 0., 0., 0.], 'arm_home_rad': [0.]*7,
                'self_collisions': True, 'arm_kp': 2000.},
             'workroom': {'sha256': '3'*64}, 'physics': {'dt': .0005},
             'objects': [{'id': 'piece', 'position_m': [.1, .2, .3]}],
             'workroom_reset_case': {k:case[k] for k in ('schema', 'case_sha256', 'case_fingerprint')},
             'native_task_bundle': {'observer': {'configuration': deepcopy(task), 'sha256': '4'*64}}}
    robot = deepcopy(scene['robot']); robot['assembly_version'] = 'new'; robot['arm_home_rad'][0] = .3
    revision = {'schema': REVISION_SCHEMA, 'revision_id': 'test_revision',
        'parent_robot_sha256': canonical_hash(scene['robot']), 'robot': robot,
        'joint_positions_rad': {'fr3_joint1': .3, 'finger1': .1}, 'robot_asset_manifest_sha256': '5'*64}
    return scene, task, initial, revision


def test_only_robot_ready_and_case_identity_change(inputs):
    scene, task, initial, revision = inputs; saved = deepcopy(inputs)
    s, t, i = derive_inputs(*inputs)
    assert inputs == saved
    assert s['robot'] == revision['robot']
    assert i['cloths_vertices_m'] == initial['cloths_vertices_m']
    assert i['joint_positions_rad'] == revision['joint_positions_rad']
    assert t['initial_state'] == i
    assert t['public_rules'] == task['public_rules'] and t['public_goal'] == task['public_goal']
    assert s['objects'] == scene['objects'] and s['physics'] == scene['physics'] and s['workroom'] == scene['workroom']
    before, after = task['reset_case'], t['reset_case']
    assert after['actual_body_poses_m'] == before['actual_body_poses_m']
    assert after['physical_geometry_fingerprint'] == before['physical_geometry_fingerprint']
    assert after['case_sha256'] != before['case_sha256'] and after['case_fingerprint'] != before['case_fingerprint']
    assert after['case_sha256'] == canonical_hash({k:v for k,v in after.items() if k != 'case_sha256'})
    assert after['robot_revision']['parent_case_sha256'] == before['case_sha256']
    assert s['native_task_bundle']['observer']['configuration'] == t
    assert s['native_task_bundle']['observer']['sha256'] == scene['native_task_bundle']['observer']['sha256']
    assert s['workroom_reset_case']['case_sha256'] == after['case_sha256']
    t['reset_case']['actual_body_poses_m']['piece']['position_m'][0] = 99
    assert inputs == saved


@pytest.mark.parametrize('field,value', [('base_position_m', [.01, 0., .8]),
    ('hand_id', 'another_hand'), ('arm_kp', 2100.), ('self_collisions', 1)])
def test_robot_installation_controller_and_chirality_are_protected(inputs, field, value):
    inputs[3]['robot'][field] = value
    with pytest.raises(ValueError, match='protected'):
        derive_inputs(*inputs)


@pytest.mark.parametrize('change', ['missing_joint', 'extra_joint', 'nan', 'bool', 'wrong_parent', 'bad_asset_pin'])
def test_revision_identity_and_ready_inventory_fail_closed(inputs, change):
    revision = inputs[3]
    if change == 'missing_joint': del revision['joint_positions_rad']['finger1']
    if change == 'extra_joint': revision['joint_positions_rad']['unowned'] = 0.
    if change == 'nan': revision['joint_positions_rad']['fr3_joint1'] = float('nan')
    if change == 'bool': revision['joint_positions_rad']['fr3_joint1'] = True
    if change == 'wrong_parent': revision['parent_robot_sha256'] = '0'*64
    if change == 'bad_asset_pin': revision['robot_asset_manifest_sha256'] = 'x'*64
    with pytest.raises(ValueError): derive_inputs(*inputs)


@pytest.mark.parametrize('summary', ['workroom_reset_case', 'living_room_reset_case', 'laundry_reset_case'])
def test_family_case_summary_follows_new_identity(inputs, summary):
    scene = inputs[0]; scene[summary] = scene.pop('workroom_reset_case')
    s, t, _ = derive_inputs(*inputs)
    assert s[summary]['case_sha256'] == t['reset_case']['case_sha256']
    assert s[summary]['case_fingerprint'] == t['reset_case']['case_fingerprint']


def test_bad_source_case_or_duplicate_task_reset_rejected(inputs):
    inputs[1]['reset_case']['case_sha256'] = '0'*64
    with pytest.raises(ValueError, match='case identity'): derive_inputs(*inputs)
    inputs[1]['reset_case']['case_sha256'] = canonical_hash({k:v for k,v in inputs[1]['reset_case'].items() if k != 'case_sha256'})
    inputs[1]['initial_state']['cloths_vertices_m']['cloth'][0][0] = 9
    with pytest.raises(ValueError, match='initial-state'): derive_inputs(*inputs)


def test_fixed_private_projection_hook_preserved(inputs):
    inputs[0]['native_task_bundle']['observer']['configuration'] = {'projection': 'private_worker_only'}
    scene, _, _ = derive_inputs(*inputs)
    assert scene['native_task_bundle'] == inputs[0]['native_task_bundle']


def test_practice_requires_explicit_adapter(inputs):
    inputs[1]['practice_case'] = {}
    with pytest.raises(ValueError, match='Practice'): derive_inputs(*inputs)


def test_namespaced_front_rear_right_hand_identity_is_preserved(inputs):
    scene, _, _, revision = inputs
    scene['robot']['hand_id'] = 'fr3_sharpa_instances'
    scene['robot']['instances'] = {s:{'hand_id':'sharpa_w01_right', 'assembly_version':'old',
        'base_position_m': [0.,sign*.6,.8]} for s,sign in [('front',-1),('rear',1)]}
    revision['robot'] = deepcopy(scene['robot'])
    for inst in revision['robot']['instances'].values(): inst['assembly_version'] = 'new'
    revision['parent_robot_sha256'] = canonical_hash(scene['robot'])
    result, _, _ = derive_inputs(*inputs)
    assert result['robot']['instances']['rear']['hand_id'] == 'sharpa_w01_right'
    revision['robot']['instances']['rear']['base_position_m'][1] += .01
    with pytest.raises(ValueError, match='installation/chirality'): derive_inputs(*inputs)


def test_new_full_per_side_mounts_are_explicit_and_owned(inputs):
    inputs[3]['robot']['mounts_by_side'] = {'right': {
        'translation_m': [0.,0.,.018], 'quaternion_wxyz':[1.,0.,0.,0.], 'rpy_rad':[0.,0.,0.]}}
    scene, _, _ = derive_inputs(*inputs)
    assert scene['robot']['mounts_by_side'] == inputs[3]['robot']['mounts_by_side']
    inputs[3]['robot']['mounts_by_side']['left'] = deepcopy(inputs[3]['robot']['mounts_by_side']['right'])
    with pytest.raises(ValueError, match='ownership'): derive_inputs(*inputs)
