"""Adversarial physical receipts for two-hand cooperation, not simulated grasps."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from bench.paths import DATA_ROOT, load_json
from scenes.workrooms.mechanical_assembly.cooperative_composition import build_task_inputs,robot_inputs
from scenes.workrooms.mechanical_assembly.cooperative_release import initial_state
from scenes.workrooms.mechanical_assembly.cooperative_evaluation import NativeTaskObserver,SequenceEvaluator
from scenes.workrooms.mechanical_assembly.cooperative_rules import evaluation_configuration
from scenes.workrooms.mechanical_assembly.nist_native_evidence import fixed

@pytest.fixture(scope='module',params=('historical_right_instances','anatomical_left_right'))
def candidate(request):
    if request.param=='historical_right_instances':return build_task_inputs()
    return build_task_inputs(asset_manifest=Path(f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/cooperative_insertion/v3/manifest.json'))


def hand_path(task,index,link='index_distal',*,arm=False):
    p=task['asset_metadata']['task_parameters'];side=p['robot_instances'][index]
    scope=p.get('robot_stage_scopes',{}).get(side,side)
    name='fr3_'+side+'_link7' if arm else side+'_'+link
    return '/World/Robot/'+scope+'/'+name


def fixture(candidate):
    task=deepcopy(candidate['task'])
    task['evaluation']=evaluation_configuration(task['id'],task['asset_metadata'],physics_dt_s=.002,room_support_paths=task['room_support_paths'])
    obs=NativeTaskObserver(task);ev=SequenceEvaluator(task)
    frame=dict(time_s=0.,native_time_s=0.,native_step=0,baseline=True,physics_dt_s=.002,
        objects={r['id']:fixed(r['initial_pose_xyzw']) for r in task['asset_metadata']['objects']},contacts=[])
    obs.on_frame(frame);ev.update(dict(time_s=0.,task_state=obs.on_observe()))
    return task,obs,ev,frame


def contact(a,b,impulse=.001,vector=(0.,0.,.001)):
    return dict(actor0=a,actor1=b,time_s=0.,impulse_magnitude_sum_ns=impulse,
        impulse_vector_ns=list(vector),min_separation_m=0.)


def step(obs,ev,f,count=1):
    for _ in range(count):
        f.update(baseline=False,time_s=f['time_s']+.002,native_time_s=f['native_time_s']+.002,native_step=f['native_step']+1)
        for row in f['contacts']:row['time_s']=f['time_s']
        obs.on_frame(f)
        if f['native_step']%10==0:ev.update(dict(time_s=f['time_s'],task_state=obs.on_observe()))
    return obs.on_observe()


def assemble(task,obs,ev,f,*,same=False,arm=False,zero=False,no_motion=False):
    paths=task['asset_metadata']['actors']
    first=hand_path(task,0,'thumb_distal',arm=arm)
    second=hand_path(task,0 if same else 1)
    handforce=0. if zero else .001
    f['objects']['sleeve']['position_m']=[.65,0.,.84]
    for i in range(80):
        depth=.05 if no_motion else .012+.040*min(i/60,1.)
        f['objects']['peg']['position_m']=[.65,0.,.90-depth]
        f['contacts']=[contact(paths['sleeve'],first,handforce,(0.,0.,handforce)),
            contact(paths['peg'],second,handforce,(0.,0.,handforce)),
            contact(paths['peg'],paths['sleeve'])]
        step(obs,ev,f)


def tray(task,obs,ev,f,*,support=True,release=True):
    paths=task['asset_metadata']['actors'];fixtures=task['asset_metadata']['fixtures']
    for name,pose in task['asset_metadata']['diagnostic_goal_poses'].items():f['objects'][name]=fixed(pose)
    f['contacts']=[contact(paths['peg'],paths['sleeve'])]
    if support:f['contacts'].append(contact(paths['sleeve'],fixtures['collection_tray']))
    if not release:f['contacts'].append(contact(paths['peg'],hand_path(task,1)))
    return step(obs,ev,f,200)


def test_original_robots_fixed_room_and_task_owned_dressing(candidate):
    scene,task=candidate['scene'],candidate['task']
    assert len(initial_state(scene,task)['joint_positions_rad'])==58
    if task['asset_metadata']['task_parameters']['robot_instances']==['left','right']:
        assert scene['robot']['hand_id']=='dual_fr3_sharpa_w01'
        assert set(scene['robot']['arm_base_poses'])=={'left','right'}
    else:
        assert scene['robot']['hand_id']=='fr3_sharpa_instances'
        assert set(scene['robot']['instances'])=={'front','rear'}
        assert all(v['hand_id']=='sharpa_w01_right' for v in scene['robot']['instances'].values())
    assert scene['task_layers'][0]['rigid_bodies']=={}
    assert set(scene['native_task_bundle']['rigid_bodies'])=={'sleeve','peg'}


def test_real_two_hand_progress_then_tray_release_passes(candidate):
    task,obs,ev,f=fixture(candidate);assemble(task,obs,ev,f)
    assert obs.on_observe()['detail']['cooperation_confirmed']
    tray(task,obs,ev,f)
    assert ev.last_result['success']

@pytest.mark.parametrize('variant',[{'same':True},{'arm':True},{'zero':True},{'no_motion':True}])
def test_mere_hand_or_arm_contact_cannot_fake_cooperation(candidate,variant):
    task,obs,ev,f=fixture(candidate);assemble(task,obs,ev,f,**variant);tray(task,obs,ev,f)
    assert not ev.last_result['success']
    assert not obs.on_observe()['detail']['cooperation_confirmed']

@pytest.mark.parametrize('options',[{'support':False},{'release':False}])
def test_no_final_credit_without_actual_tray_support_and_release(candidate,options):
    task,obs,ev,f=fixture(candidate);assemble(task,obs,ev,f);tray(task,obs,ev,f,**options)
    assert not ev.last_result['success']


def test_goal_pose_alone_does_not_count(candidate):
    task,obs,ev,f=fixture(candidate);tray(task,obs,ev,f)
    assert not ev.last_result['success']
    assert obs.on_observe()['events']==[]


def test_contact_break_during_adjustment_is_recoverable(candidate):
    task,obs,ev,f=fixture(candidate);assemble(task,obs,ev,f)
    f['contacts']=[];step(obs,ev,f,20)
    assert ev.last_result['status']=='ongoing'
    tray(task,obs,ev,f);assert ev.last_result['success']


def test_missing_native_frames_are_unknown_not_failure(candidate):
    task,obs,ev,f=fixture(candidate)
    f['native_step']+=1;step(obs,ev,f,9)
    assert ev.last_result['status']=='unknown'
    assert not ev.last_result['physical_failure']


def test_desktop_is_static_and_does_not_cross_work_rectangle():
    from pxr import Usd,UsdPhysics,UsdGeom
    from scenes.workrooms.mechanical_assembly.desktop_dressing import ASSET_ROOT
    meta=load_json(ASSET_ROOT/'metadata.json');stage=Usd.Stage.Open(str(ASSET_ROOT/'desktop.usda'))
    assert meta['group_count']==31
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    for row in meta['groups']:
        x=row['initial_pose_xyzw'][0];low,high=row['local_bounds_m']
        assert x+low[0]>=.86 and x+high[0]<=1.12

@pytest.mark.parametrize('change',[
    lambda s,t:next(iter(s['robot']['arm_home_rad_by_side'].values())).pop(),
    lambda s,t:s['mechanical_cooperative_provenance'].update(engineering_probe=True),
])
def test_frozen_candidate_rejects_truncated_robot_or_diagnostic_reset(candidate,change):
    values=deepcopy(candidate);change(values['scene'],values['task'])
    with pytest.raises(ValueError):initial_state(values['scene'],values['task'])


def test_retouch_after_single_hand_insertion_cannot_retroactively_earn_progress(candidate):
    task,obs,ev,f=fixture(candidate);paths=task['asset_metadata']['actors']
    f['objects']['sleeve']['position_m']=[.65,0.,.84]
    f['objects']['peg']['position_m']=[.65,0.,.888]
    hands=[contact(paths['sleeve'],hand_path(task,0,'thumb_distal')),
           contact(paths['peg'],hand_path(task,1))]
    f['contacts']=deepcopy(hands);step(obs,ev,f,50)
    f['contacts']=[]
    f['objects']['peg']['position_m']=[.65,0.,.848]
    step(obs,ev,f,10)
    f['contacts']=deepcopy(hands)+[contact(paths['peg'],paths['sleeve'])]
    step(obs,ev,f,50)
    assert not obs.on_observe()['detail']['cooperation_confirmed']
    assert obs.on_observe()['detail']['cooperative_insertion_progress_m']==0.
    tray(task,obs,ev,f);assert not ev.last_result['success']


def test_submillimetre_contact_jitter_is_not_accumulated_as_insertion(candidate):
    task,obs,ev,f=fixture(candidate);paths=task['asset_metadata']['actors']
    f['objects']['sleeve']['position_m']=[.65,0.,.84]
    f['contacts']=[contact(paths['sleeve'],hand_path(task,0,'thumb_distal')),
        contact(paths['peg'],hand_path(task,1)),contact(paths['peg'],paths['sleeve'])]
    for i in range(500):
        f['objects']['peg']['position_m']=[.65,0.,.850+(.00005 if i%2 else 0.)]
        step(obs,ev,f)
    assert obs.on_observe()['detail']['cooperative_insertion_progress_m']<.0001
    assert not obs.on_observe()['detail']['cooperation_confirmed']


@pytest.mark.parametrize('withdrawal_contact',['none','one_hand','reattached'])
def test_uncooperative_withdrawals_debit_prior_insertion(candidate,withdrawal_contact):
    task,obs,ev,f=fixture(candidate);paths=task['asset_metadata']['actors']
    f['objects']['sleeve']['position_m']=[.65,0.,.84]
    hands=[contact(paths['sleeve'],hand_path(task,0,'thumb_distal')),
           contact(paths['peg'],hand_path(task,1))]
    mating=[contact(paths['peg'],paths['sleeve'])]
    # 170 pushes span only 50 micrometres, but used to earn 8.5 mm because
    # the returning segment lacked consecutive cooperative hand contacts.
    for _ in range(170):
        f['objects']['peg']['position_m']=[.65,0.,.850]
        f['contacts']=deepcopy(hands+mating);step(obs,ev,f)
        f['objects']['peg']['position_m'][2]-=.00005;step(obs,ev,f)
        f['contacts']=deepcopy(mating+(hands[:1] if withdrawal_contact=='one_hand' else []))
        if withdrawal_contact=='reattached':
            step(obs,ev,f)  # Reattachment itself must not hide the retreat.
            f['contacts']=deepcopy(hands+mating)
        f['objects']['peg']['position_m'][2]+=.00005;step(obs,ev,f)
    detail=obs.on_observe()['detail']
    assert detail['cooperative_insertion_progress_m']==pytest.approx(0.,abs=1e-12)
    assert not detail['cooperation_confirmed']
    tray(task,obs,ev,f)
    assert not ev.last_result['success']


@pytest.mark.parametrize('alignment_lost',[False,True])
def test_regrasp_keeps_retained_progress_and_allows_fresh_insertion(candidate,alignment_lost):
    task,obs,ev,f=fixture(candidate);paths=task['asset_metadata']['actors']
    f['objects']['sleeve']['position_m']=[.65,0.,.84]
    hands=[contact(paths['sleeve'],hand_path(task,0,'thumb_distal')),
           contact(paths['peg'],hand_path(task,1))]
    mating=[contact(paths['peg'],paths['sleeve'])]
    def at(depth,rows,count=1):
        f['objects']['peg']['position_m']=[.65,0.,.90-depth]
        f['contacts']=deepcopy(rows);step(obs,ev,f,count)
    at(.025,hands+mating,50)
    for i in range(1,21):at(.025+i*.0002,hands+mating)
    at(.029,mating,10)  # Stationary regrasp preserves the first 4 mm.
    assert obs.on_observe()['detail']['cooperative_insertion_progress_m']==pytest.approx(.004)
    at(.027,mating)  # Unheld 2 mm retreat debits existing credit.
    assert obs.on_observe()['detail']['cooperative_insertion_progress_m']==pytest.approx(.002)
    if alignment_lost:
        f['objects']['peg']['position_m'][0]+=.02
        f['contacts']=deepcopy(hands+mating);step(obs,ev,f,10)
        assert not obs.on_observe()['detail']['aligned']
    at(.027,hands+mating)
    for i in range(1,31):at(.027+i*.0002,hands+mating)
    assert obs.on_observe()['detail']['cooperative_insertion_progress_m']==pytest.approx(.008)
    assert obs.on_observe()['detail']['cooperation_confirmed']
    tray(task,obs,ev,f)
    assert ev.last_result['success']
