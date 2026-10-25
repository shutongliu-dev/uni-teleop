"""Adversarial trajectory tests; CPU results do not certify native contact physics."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bench.paths import load_json
from scenes.manipulation.evaluator import SimpleTaskEvaluator
from bench.isaacsim.release import digest, dump, load_evaluator, task_inputs, validate


def task(mode='push'):
    return {'schema_version':'simple-manipulation-task-v1','id':'01_cube_'+mode,
        'title_zh':'移动方块','steps':[{'id':mode,'instruction_zh':'按公开目标移动方块'}],
        'public_goal':{'object_id':'cube','operation':mode,'target_position_m':[.45,.2,.78],
                       'frame_id':'world','description':'桌面目标'},
        'evaluation':{'mode':mode,'object':'cube','initial_position_m':[.45,.1,.78],
            'goal_position_m':[.45,.2,.78],'initial_tolerance_m':.005,'goal_xy_radius_m':.03,
            'goal_z_tolerance_m':.006,'minimum_displacement_m':.05,'minimum_lift_m':.08,
            'hold_s':.3,'lift_stage_hold_s':.12,'maximum_speed_m_s':.04,
            'maximum_angular_speed_rad_s':.3,'maximum_motion_speed_m_s':1.5,
            'maximum_tilt_rad':.7,'minimum_contact_duration_s':.04,'max_observation_gap_s':.041,
            'physics_dt_s':.0005,'control_period_s':.02}}


class Trajectory:
    def __init__(self, mode='push'):
        self.evaluator=SimpleTaskEvaluator(task(mode))
        self.time=-.02
        self.position=np.array([.45,.1,.78])
        self.last=None

    def observe(self, position=None, *, hand=False, table=True, impulse=.002,
                robot_actor='/World/Robot/right_index_DP', dt=.02, edits=None, ticks=None):
        self.time=round(self.time+dt,10)
        if position is not None:
            self.position=np.asarray(position,dtype=float)
        rows=[]
        for actor,active in [(robot_actor,hand),('/World/Table',table)]:
            if active and self.time:
                for tick in (range(1, 41) if ticks is None else ticks):
                    rows.append({'actor0':'/World/Objects/cube','actor1':actor,'time_s':self.time-.02+tick*.0005,
                                 'point_count':2,'impulse_magnitude_sum_ns':impulse,'min_separation_m':-.0001,
                                 'impulse_vector_ns':[0.,0.,impulse], 'nonzero_impulse_normal_count':1 if impulse else 0,
                                 'normal_length_max_abs_error':0., 'impulse_components':'normal_only'})
        start = self.time - .02 if self.time else 0.
        sampling = {'protocol':'native_rigid_contact_sampling_v1','physics_dt_s':.0005,
                    'start_sim_time_s':start,'end_sim_time_s':self.time,
                    'native_start_time_s':start,'native_end_time_s':self.time,
                    'native_start_step_index':round(start/.0005),'native_end_step_index':round(self.time/.0005),
                    'sampled_substeps':40 if self.time else 0,'complete':True}
        obs={'time_s':self.time,'last_duration_s':.02 if self.time else 0.,
            'metrics':{'state_finite':True},'objects':{'cube':{
            'position_m':self.position.tolist(),'quaternion_wxyz':[1,0,0,0],
            'linear_velocity_m_s':[0,0,0],'angular_velocity_rad_s':[0,0,0]}},
            'contacts':{'source':'PhysX stage-bound rigid contact tensor; normal impulses','rigid':rows,'errors':[],
                        'sampling':sampling}}
        if edits:
            edits(obs)
        self.last=deepcopy(obs)
        return self.evaluator.update(obs)

    def move(self, goal, **kwargs):
        start=self.position.copy()
        for alpha in np.linspace(0,1,51)[1:]:
            result=self.observe(start+(np.asarray(goal)-start)*alpha,**kwargs)
        return result

    def hold(self, steps=20, **kwargs):
        for _ in range(steps):
            result=self.observe(**kwargs)
        return result


def push_trajectory(**contact):
    run=Trajectory()
    run.observe()
    run.move([.45,.2,.78],hand=True,**contact)
    return run


def lifted_trajectory(mode='lift_hold'):
    run=Trajectory(mode)
    run.observe()
    run.move([.45,.1,.88],hand=True,table=False)
    return run


def test_realistic_push_requires_interaction_motion_release_and_stability():
    run=push_trajectory()
    assert not run.evaluator.last_result['success']
    result=run.hold()
    assert result['success'] and result['hand_contact_intervals']>=2
    assert result['table_support_contact'] and not result['hand_contact']


def test_two_single_substep_touches_do_not_credit_two_entire_control_windows():
    run = Trajectory()
    run.observe()
    run.observe(hand=True, ticks=[40])
    result = run.observe(hand=True, ticks=[1])
    assert result['hand_contact_duration_s'] == pytest.approx(.001)
    assert result['hand_contact_substeps'] == 2
    assert result['hand_contact_intervals'] == 2
    assert not result['interaction_observed']


def test_multiple_hand_pairs_share_one_duration_but_sum_distinct_impulses():
    run = Trajectory()
    run.observe()
    def second_finger(obs):
        row = deepcopy(obs['contacts']['rigid'][0])
        row['actor1'] = '/World/Robot/right_thumb_DP'
        obs['contacts']['rigid'].append(row)
    result = run.observe(hand=True, ticks=[40], edits=second_finger)
    assert result['hand_contact_duration_s'] == pytest.approx(.0005)
    assert result['hand_contact_normal_impulse_ns'] == pytest.approx(.004)


@pytest.mark.parametrize('reverse', [False, True])
def test_duplicate_pair_fails_closed_instead_of_double_counting(reverse):
    run = Trajectory()
    run.observe()
    def duplicate(obs):
        row = deepcopy(obs['contacts']['rigid'][0])
        if reverse:
            row['actor0'], row['actor1'] = row['actor1'], row['actor0']
        obs['contacts']['rigid'].append(row)
    result = run.observe(hand=True, edits=duplicate)
    assert result['reason'].startswith('invalid_observation:duplicated native contact pair')
    assert result['hand_contact_duration_s'] == 0
    assert not run.hold(hand=True)['success']


@pytest.mark.parametrize('mutation', [
    lambda o: o['contacts'].pop('sampling'),
    lambda o: o['contacts']['sampling'].update(complete=False),
    lambda o: o['contacts']['sampling'].update(sampled_substeps=39),
    lambda o: o['contacts']['sampling'].update(sampled_substeps=True),
    lambda o: o['contacts']['sampling'].update(native_end_step_index=41),
    lambda o: o['contacts']['sampling'].update(native_end_time_s=.0205),
    lambda o: o['contacts']['sampling'].update(physics_dt_s=.001),
    lambda o: o['contacts']['sampling'].update(start_sim_time_s=.0005),
    lambda o: o.update(last_duration_s=.04),
    lambda o: o['contacts']['rigid'][0].update(time_s=.0007),
    lambda o: o['contacts']['rigid'][0].update(time_s=0.),
    lambda o: o['contacts']['rigid'][0].update(time_s=.0205),
    lambda o: o['contacts']['rigid'][0].update(point_count=True),
    lambda o: o['contacts']['rigid'][0].update(impulse_components='combined'),
    lambda o: o['contacts']['rigid'][0].update(impulse_magnitude_sum_ns=True),
])
def test_substep_evidence_requires_complete_sampling_and_aligned_clocks(mutation):
    run = Trajectory()
    run.observe()
    result = run.observe(hand=True, edits=mutation)
    assert result['reason'].startswith('invalid_observation:')
    assert result['hand_contact_duration_s'] == 0


def test_old_task_without_explicit_physics_dt_cannot_use_new_scoring():
    value = task()
    del value['evaluation']['physics_dt_s']
    with pytest.raises(ValueError, match='physics_dt_s'):
        SimpleTaskEvaluator(value)


def test_full_dense_contact_windows_preserve_the_real_40_ms_threshold():
    run = Trajectory()
    run.observe()
    assert not run.observe(hand=True)['interaction_observed']
    result = run.observe(hand=True)
    assert result['hand_contact_substeps'] == 80
    assert result['hand_contact_duration_s'] == pytest.approx(.04)
    assert result['interaction_observed']


def test_idle_cannot_succeed_even_with_hand_contact():
    run=Trajectory()
    run.observe(hand=True)
    assert not run.hold(100,hand=True)['success']


def test_initial_state_at_goal_is_permanently_rejected():
    run=Trajectory()
    first=run.observe([.45,.2,.78])
    assert first['reason'].startswith('invalid_observation:initial state')
    run.observe([.45,.1,.78])
    assert not run.hold(100)['success']


@pytest.mark.parametrize('kwargs',[{'robot_actor':'/World/Robot/fr3_link7'}, {'impulse':0.}])
def test_arm_contact_or_zero_impulse_cannot_count_as_grasp(kwargs):
    run=push_trajectory(**kwargs)
    assert not run.hold()['success']
    assert run.evaluator.last_result['hand_contact_intervals']==0


def test_prior_hand_touch_cannot_credit_an_arm_push_even_after_arm_releases():
    run=Trajectory()
    run.observe()
    run.hold(2,hand=True)
    run.move([.45,.2,.78],hand=True,robot_actor='/World/Robot/fr3_link7')
    still_touching=run.hold(hand=True,robot_actor='/World/Robot/fr3_link7')
    assert still_touching['interaction_observed']
    assert still_touching['any_robot_contact'] and not still_touching['hand_contact']
    assert not still_touching['success']
    released=run.hold()
    assert not released['any_robot_contact']
    assert released['non_hand_robot_contact_violated']
    assert released['non_hand_robot_contact_time_s'] is not None
    assert released['reason']=='non_hand_robot_contact_violated'
    assert not released['success']


def test_arm_support_at_terminal_goal_is_not_a_released_push():
    run=push_trajectory()
    result=run.hold(hand=True,robot_actor='/World/Robot/fr3_link8')
    assert result['goal_geometry'] and result['stable'] and result['interaction_observed']
    assert result['any_robot_contact'] and not result['hand_contact']
    assert not result['success']


def test_zero_impulse_arm_pair_does_not_invent_a_contact_violation():
    run=push_trajectory()
    result=run.hold(hand=True,robot_actor='/World/Robot/fr3_link7',impulse=0.)
    # The fixture gives table zero impulse too; it therefore cannot satisfy
    # support until a subsequent valid table-only interval.
    assert not result['any_robot_contact'] and not result['non_hand_robot_contact_violated']
    assert run.hold()['success']


def test_table_only_drift_cannot_count_as_agent_push():
    run=Trajectory()
    run.observe()
    run.move([.45,.2,.78],table=True)
    assert not run.hold()['success']


def test_transient_goal_crossing_and_held_object_do_not_count_as_placed():
    run=push_trajectory()
    assert not run.hold(10)['success']
    run.move([.45,.3,.78],hand=True)
    assert not run.hold()['success']
    run=push_trajectory()
    assert not run.hold(40,hand=True)['success']


def test_table_support_is_required_even_if_final_pose_matches():
    run=push_trajectory()
    assert not run.hold(40,table=False)['success']


def test_success_is_revoked_after_physical_goal_is_lost():
    run=push_trajectory()
    assert run.hold()['success']
    assert not run.move([.45,.26,.78])['success']


def test_lift_requires_stable_in_hand_hold_and_cannot_pass_free_fall():
    run=lifted_trajectory()
    assert run.hold(hand=True,table=False)['success']
    assert not run.observe(hand=False,table=False)['success']
    free=lifted_trajectory()
    assert not free.hold(table=False)['success']


def test_pick_place_requires_ordered_lift_transport_and_release():
    run=lifted_trajectory('pick_place')
    assert not run.hold(10,hand=True,table=False)['success']
    assert run.evaluator.last_result['lift_stage_completed']
    run.move([.45,.2,.88],hand=True,table=False)
    run.move([.45,.2,.78],hand=True,table=False)
    assert run.hold()['success']


def test_pushing_to_goal_cannot_pass_pick_place():
    run=Trajectory('pick_place')
    run.observe()
    run.move([.45,.2,.78],hand=True)
    assert not run.hold()['success']


def test_lift_then_drop_and_push_cannot_substitute_for_in_hand_transport():
    run=lifted_trajectory('pick_place')
    run.hold(10,hand=True,table=False)
    run.move([.45,.1,.78],table=False)
    run.move([.45,.2,.78],hand=True)
    assert not run.hold()['success']
    assert not run.evaluator.last_result['transport_stage_completed']


def test_velocity_or_tip_over_blocks_terminal_stability():
    for edit in [lambda obs:obs['objects']['cube'].update(linear_velocity_m_s=[.06,0,0]),
                 lambda obs:obs['objects']['cube'].update(angular_velocity_rad_s=[0,.4,0]),
                 lambda obs:obs['objects']['cube'].update(quaternion_wxyz=[.70710678,.70710678,0,0])]:
        run=push_trajectory()
        assert not run.hold(40,edits=edit)['success']


def test_teleport_is_rejected_even_with_false_zero_reported_velocity():
    run=Trajectory()
    run.observe()
    result=run.observe([.45,.2,.78],hand=True)
    assert result['reason'].startswith('invalid_observation:object trajectory')


def test_long_observation_gap_permanently_rejects_incomplete_episode_evidence():
    run=push_trajectory()
    run.hold(5)
    assert run.observe(dt=.4)['reason'].startswith('invalid_observation:native contact sampling window')
    assert not run.hold(10)['success']
    assert not run.hold(10)['success']


@pytest.mark.parametrize('mutation',[
    lambda obs:obs.pop('contacts'),
    lambda obs:obs['contacts'].update(source='proximity'),
    lambda obs:obs['contacts'].update(errors=['overflow']),
    lambda obs:obs['objects']['cube'].pop('linear_velocity_m_s'),
    lambda obs:obs['objects']['cube'].update(position_m=[float('nan'),0,0]),
    lambda obs:obs['contacts']['rigid'][0].update(time_s=-1.),
])
def test_missing_corrupted_or_stale_physics_evidence_fails_closed(mutation):
    run=Trajectory()
    run.observe()
    result=run.observe(hand=True,edits=mutation)
    assert result['reason'].startswith('invalid_observation:')
    assert not result['success']


def test_input_is_not_mutated_and_timestamps_are_strict():
    run=Trajectory()
    run.observe()
    obs=deepcopy(run.last)
    obs['time_s']=.02
    before=deepcopy(obs)
    run.evaluator.update(obs)
    assert obs==before
    with pytest.raises(ValueError,match='strictly increasing'):
        run.evaluator.update(obs)


@pytest.mark.parametrize('mode',['push','lift_hold','pick_place'])
def test_each_native_worker_result_is_strict_json_serializable(mode):
    run=Trajectory(mode)
    samples=[run.observe(),run.observe(hand=True),run.move([.45,.2,.78],hand=True),run.hold()]
    for result in samples:
        # Native AF_UNIX IPC and physics.jsonl use strict stdlib JSON, not a
        # NumPy-aware development encoder. np.bool_ previously broke step 1.
        assert json.loads(json.dumps(result,allow_nan=False))==result


@pytest.fixture
def independent_release(tmp_path):
    root=tmp_path/'release'
    root.mkdir()
    value=task()
    files={}
    for relative,data in [('tasks/task.json',value),('tasks/scene.json',{'objects':[]}),
                          ('asset_file_hashes.json',{})]:
        dump(root/relative,data)
        files[relative]={'sha256':digest(root/relative)}
    evaluator=Path(__file__).resolve().parents[1]/'scenes/manipulation/evaluator.py'
    (root/'evaluator').mkdir()
    (root/'evaluator/sequence_evaluator.py').write_bytes(evaluator.read_bytes())
    files['evaluator/sequence_evaluator.py']={'sha256':digest(evaluator)}
    dump(root/'physics_profile.json',{'isaacsim_options':{}})
    dump(root/'catalog.json',{})
    manifest={'schema_version':'isaacsim-simple-manipulation-release-v1','source_release':None,
        'original_release_modified':False,'files':files,'physics_profile':'physics_profile.json',
        'physics_profile_sha256':digest(root/'physics_profile.json'),
        'asset_catalog':{'path':str(root/'catalog.json'),'sha256':digest(root/'catalog.json')},
        'tasks':[{'id':value['id'],'task':'tasks/task.json','scene':'tasks/scene.json'}]}
    dump(root/'release_manifest.json',manifest)
    return root


def test_independent_release_loads_pinned_evaluator_and_rejects_changed_bytes(independent_release):
    root=independent_release
    validate(root)
    assert load_evaluator(root)(task()).mode=='push'
    with pytest.raises(ValueError,match='reset-at-goal'):
        task_inputs(root,task()['id'],'qualification_probe')
    file=root/'tasks/task.json'
    file.write_text(file.read_text()+'\n')
    with pytest.raises(ValueError,match='Frozen input changed'):
        validate(root)


def test_tasks_route_exposes_only_public_intent_not_private_evaluator(independent_release):
    from bench.runtime.server import create_app
    manager=SimpleNamespace(release=independent_release,shutdown=lambda:None)
    with TestClient(create_app(service=manager)) as client:
        response=client.get('/tasks')
    assert response.status_code==200
    row=response.json()[0]
    assert set(row)=={'task_id','instruction','goal'}
    assert row['goal']==task()['public_goal']
    assert 'maximum_speed' not in response.text and 'hold_s' not in response.text


def test_server_release_selection_is_operator_environment_only(monkeypatch,tmp_path):
    from bench.runtime import server
    captured={}
    monkeypatch.setenv('BENCH_TASK_RELEASE',str(tmp_path/'chosen'))
    def factory(directory,**options):
        captured.update(options)
        return SimpleNamespace()
    monkeypatch.setattr(server,'AgentService',factory)
    server.create_app(data_dir=tmp_path/'episodes')
    assert captured['release']==tmp_path/'chosen'


@pytest.fixture
def contract_trajectory(monkeypatch):
    """Exercise task-stage composition separately from the detailed grasp tests."""
    from scenes.manipulation import humanlike

    class ContractProbe:
        def __init__(self, config):
            self.reset()

        def reset(self):
            pass

        def update(self, observation):
            result = {'valid': True, 'passed': True, 'instantaneous_pass': True,
                      'grasp_attempted': True, 'pose_valid': True, 'reason': 'satisfied',
                      'hold_elapsed_s': .2, 'metrics': {}}
            result.update(observation.get('_contract_test_result', {}))
            return result

    monkeypatch.setattr(humanlike, 'HumanlikeGraspEvaluator', ContractProbe)

    def create(mode):
        run = Trajectory()
        value = task(mode)
        value['evaluation']['humanlike_grasp'] = {'test_contract': True}
        if mode in ('lift_hold', 'pick_place'):
            value['evaluation']['humanlike_lift_start_m'] = .04
        if mode == 'push':
            value['evaluation']['push_grasp_violation_hold_s'] = .12
        if mode == 'grasp_hold':
            value['evaluation']['goal_position_m'] = [.45, .1, .78]
        run.evaluator = SimpleTaskEvaluator(value)
        return run

    return create


def contract_result(**fields):
    return lambda observation: observation.update(_contract_test_result=fields)


def test_tabletop_grasp_cannot_omit_the_frozen_contract():
    with pytest.raises(ValueError, match='humanlike contract or a pinch-force rule'):
        SimpleTaskEvaluator(task('grasp_hold'))


def test_tabletop_grasp_needs_no_displacement_but_does_need_sustained_pose_and_contacts(contract_trajectory):
    run = contract_trajectory('grasp_hold')
    run.observe()
    assert not run.hold(hand=True, edits=contract_result(passed=False, instantaneous_pass=False))['success']
    result = run.hold(hand=True)
    assert result['success'] and result['maximum_displacement_m'] == 0
    assert not run.observe(hand=True, edits=contract_result(passed=False, instantaneous_pass=False))['success']


@pytest.mark.parametrize('contact', [{'hand': False}, {'hand': True, 'table': False}])
def test_pose_alone_or_airborne_contact_cannot_pass_tabletop_grasp(contract_trajectory, contact):
    run = contract_trajectory('grasp_hold')
    run.observe()
    assert not run.hold(**contact)['success']


def test_tabletop_grasp_cannot_be_preceded_by_lift_or_far_displacement(contract_trajectory):
    for target in ([.45, .1, .8], [.45, .16, .78]):
        run = contract_trajectory('grasp_hold')
        run.observe()
        run.move(target, hand=True)
        run.move([.45, .1, .78], hand=True)
        assert not run.hold(hand=True)['success']


def test_lifting_before_grasp_is_established_is_an_irreversible_violation(contract_trajectory):
    run = contract_trajectory('lift_hold')
    run.observe()
    run.move([.45, .1, .88], hand=True, table=False,
             edits=contract_result(passed=False, instantaneous_pass=True, reason='acquiring'))
    result = run.hold(hand=True, table=False)
    assert not result['success'] and result['grasp_constraint_violated']
    assert result['grasp_violation_reason'] == 'grasp_not_established_before_lift'


def test_losing_grasp_then_restoring_terminal_posture_cannot_pass_lift(contract_trajectory):
    run = contract_trajectory('lift_hold')
    run.observe()
    run.move([.45, .1, .88], hand=True, table=False)
    assert run.hold(hand=True, table=False)['success']
    broken = run.observe(hand=True, table=False,
        edits=contract_result(passed=False, instantaneous_pass=False, reason='back_of_hand'))
    stamp = broken['grasp_violation_time_s']
    result = run.hold(hand=True, table=False)
    assert not result['success'] and result['grasp_constraint_violated']
    assert result['grasp_violation_time_s'] == stamp


def test_carry_contract_stays_active_below_lift_threshold_until_table_support(contract_trajectory):
    run = contract_trajectory('lift_hold')
    run.observe()
    run.move([.45, .1, .88], hand=True, table=False)
    run.move([.45, .1, .80], hand=True, table=False)
    result = run.observe(hand=True, table=False,
                         edits=contract_result(passed=False, instantaneous_pass=False))
    assert result['carry_grasp_active'] and result['grasp_constraint_violated']


def test_clean_table_support_allows_release_after_valid_carry(contract_trajectory):
    run = contract_trajectory('pick_place')
    run.observe()
    run.move([.45, .1, .88], hand=True, table=False)
    run.hold(10, hand=True, table=False)
    run.move([.45, .2, .88], hand=True, table=False)
    run.move([.45, .2, .78], hand=True, table=False)
    result = run.hold(edits=contract_result(passed=False, instantaneous_pass=False))
    assert result['success'] and not result['grasp_constraint_violated']
    assert not result['carry_grasp_active']


def test_carry_violation_cannot_be_hidden_by_successful_released_placement(contract_trajectory):
    run = contract_trajectory('pick_place')
    run.observe()
    run.move([.45, .1, .88], hand=True, table=False)
    run.hold(10, hand=True, table=False)
    run.observe(hand=True, table=False,
                edits=contract_result(passed=False, instantaneous_pass=False))
    run.move([.45, .2, .88], hand=True, table=False)
    run.move([.45, .2, .78], hand=True, table=False)
    assert not run.hold()['success']
    assert run.evaluator.last_result['grasp_constraint_violated']


def test_observation_gap_during_carry_cannot_hide_grasp_violation(contract_trajectory):
    run = contract_trajectory('lift_hold')
    run.observe()
    run.move([.45, .1, .88], hand=True, table=False)
    run.observe(hand=True, table=False, dt=.2)
    result = run.hold(hand=True, table=False)
    assert not result['success']
    assert result['reason'].startswith('invalid_observation:native contact sampling window')


def test_missing_humanlike_evidence_fails_closed_and_cannot_recover(contract_trajectory):
    run = contract_trajectory('grasp_hold')
    run.observe()
    invalid = run.observe(hand=True, edits=contract_result(valid=False, reason='missing_measured_fk'))
    assert invalid['reason'].startswith('invalid_observation:humanlike grasp evidence:')
    assert not run.hold(hand=True)['success']


def test_push_does_not_require_a_pinch_or_power_grasp(contract_trajectory):
    run = contract_trajectory('push')
    edit = contract_result(passed=False, instantaneous_pass=False, grasp_attempted=False)
    run.observe(edits=edit)
    run.move([.45, .2, .78], hand=True, edits=edit)
    assert run.hold(edits=edit)['success']


def test_sustained_nonconforming_grasp_cannot_be_hidden_by_successful_push(contract_trajectory):
    run = contract_trajectory('push')
    edit = contract_result(passed=False, instantaneous_pass=False, grasp_attempted=True)
    run.observe()
    run.hold(7, hand=True, edits=edit)
    run.move([.45, .2, .78], hand=True)
    result = run.hold()
    assert not result['success'] and result['grasp_constraint_violated']
    assert result['grasp_violation_reason'] == 'sustained_nonconforming_grasp_during_push'


def test_brief_grasp_closing_and_release_transitions_do_not_invalidate_push(contract_trajectory):
    run = contract_trajectory('push')
    invalid_grasp = contract_result(passed=False, instantaneous_pass=False, grasp_attempted=True)
    single_finger = contract_result(passed=False, instantaneous_pass=False, grasp_attempted=False)
    run.observe()
    for _ in range(3):
        run.hold(3, hand=True, edits=invalid_grasp)
        run.observe(hand=True, edits=single_finger)
    run.move([.45, .2, .78], hand=True, edits=single_finger)
    result = run.hold(edits=single_finger)
    assert result['success'] and not result['grasp_constraint_violated']


def test_four_task_spec_keeps_original_physics_thresholds_assets_and_initialization():
    from scenes.manipulation.release import PREVIOUS_SIMPLE_RELEASE, specifications
    original = validate(PREVIOUS_SIMPLE_RELEASE)
    prior = {entry['id']: entry for entry in original['tasks']}
    specs = specifications()
    assert {task['id'] for _, task in specs} == set(prior) | {'04_cube_grasp_hold'}
    for scene, value in specs:
        if value['id'] in prior:
            entry = prior[value['id']]
            old_task = load_json(PREVIOUS_SIMPLE_RELEASE / entry['task'])
            old_scene = load_json(PREVIOUS_SIMPLE_RELEASE / entry['scene'])
            for key, threshold in old_task['evaluation'].items():
                assert value['evaluation'][key] == threshold
            assert scene['robot'] == old_scene['robot']
            assert scene['objects'] == old_scene['objects']
            assert scene['isaacsim'] == old_scene['isaacsim']
        if value['evaluation']['mode'] != 'push':
            assert value['evaluation']['humanlike_grasp']
            assert '拇指' in value['steps'][0]['instruction_zh']
            assert '反折' in value['steps'][0]['instruction_zh']
        if value['evaluation']['mode'] in ('lift_hold', 'pick_place'):
            assert '途中违反后不能靠最后摆正补救' in value['steps'][0]['instruction_zh']
    assert digest(PREVIOUS_SIMPLE_RELEASE / 'release_manifest.json') == '8be901d08e8e0b677d24dcc323e12f1f4661d554280f8e4e23f17fbb8b51849a'


def test_latched_episode_violations_are_reported_as_irrecoverable():
    run=Trajectory()
    run.observe()
    assert run.evaluator.last_result['irrecoverable_reason'] is None
    result=run.move([.45,.1,.78+.05],hand=True,table=False)
    assert result['irrecoverable_reason']=='episode_lift_limit_exceeded'
    # Returning the object does not erase a latched episode violation.
    result=run.move([.45,.1,.78],hand=True)
    assert result['irrecoverable_reason']=='episode_lift_limit_exceeded'


def test_arm_contact_is_irrecoverable_but_an_ordinary_push_is_not():
    run=push_trajectory()
    assert run.evaluator.last_result['irrecoverable_reason'] is None
    result=run.observe(hand=True,robot_actor='/World/Robot/fr3_link7')
    assert result['irrecoverable_reason']=='non_hand_robot_contact_violated'
