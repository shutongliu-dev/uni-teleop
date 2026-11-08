"""CPU rule/asset regression only; native tool contact has separate evidence."""
import numpy as np
import pytest
from pxr import Usd, UsdPhysics

from scenes.workrooms.laboratory import rules
from scenes.workrooms.laboratory.evaluation import NativeTaskObserver
from scenes.workrooms.laboratory.tasks.tube_clamp_transfer.assets import author_task
from test_laboratory_evaluation import setup


def test_tube_is_free_body_and_spring_never_connects_to_it():
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage)
    joint=UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(meta['joints']['jaw_slide']))
    targets=[str(p) for rel in (joint.GetBody0Rel(),joint.GetBody1Rel()) for p in rel.GetTargets()]
    assert set(targets)=={meta['actors']['clamp_frame'],meta['actors']['clamp_jaw']}
    assert meta['actors']['tube'] not in targets
    drive=UsdPhysics.DriveAPI(joint.GetPrim(),'linear')
    assert drive.GetStiffnessAttr().Get()==200.
    assert drive.GetTargetPositionAttr().Get()==pytest.approx(-.018)
    assert set(meta['actors'])=={'clamp_frame','clamp_jaw','tube'}
    assert len(meta['interaction_targets'])==6
    assert meta['physics_accepted'] is False


def synthetic_case(*, missing_jaw=False, direct_hand=False):
    task,frame=setup('tube_clamp_transfer');observer=NativeTaskObserver(task)
    observer.on_frame(frame)
    assert observer.on_observe()['released_stable_s']==0
    meta=task['asset_metadata'];actors={**meta['actors'],**meta['fixtures']}
    def contact(a,b,vec):
        return {'actor0':actors.get(a,a),'actor1':actors.get(b,b),'time_s':frame['time_s'],
                'impulse_vector_ns':vec,'impulse_magnitude_sum_ns':float(np.linalg.norm(vec)),
                'min_separation_m':0.}
    for tick in range(1,1001):
        frame.update(baseline=False,time_s=tick*.0005,native_time_s=tick*.0005,native_step=tick)
        if tick<250:
            tube=[.48 if tick<200 else .92,.57,.9063 if tick<200 else 1.0063]
            frame['objects']['tube']['position_m']=tube
            frame['objects']['clamp_frame']['position_m']=[tube[0]-.06,tube[1]+.003,tube[2]]
            frame['objects']['clamp_jaw']['position_m']=[tube[0]-.06,tube[1]-.007,tube[2]]
            pairs=[contact('tube','clamp_frame',[0,.0008,0])]
            if not missing_jaw:pairs.append(contact('tube','clamp_jaw',[0,-.0008,0]))
            if direct_hand:pairs.append(contact('tube','/World/Robot/left_index',[0,0,.0002]))
        else:
            for name in ('clamp_frame','clamp_jaw'):
                original=next(o for o in meta['objects'] if o['id']==name)['initial_pose_xyzw']
                frame['objects'][name]['position_m']=original[:3]
            frame['objects']['tube']['position_m']=[.92,.57,.9063]
            pairs=[contact('tube','receiver_holder',[0,0,.0002]),contact('clamp_frame','tool_rest',[0,0,.0008])]
        frame['contacts']=pairs
        observer.on_frame(frame)
    return observer.on_observe()


def test_real_two_pad_history_and_released_support_are_required_by_rules():
    result=synthetic_case()
    assert result['complete'],result['errors']
    assert result['detail']['tube_carried']
    assert result['detail']['tube_seated'] and result['detail']['tool_returned']
    assert result['released_stable_s']>=.3


@pytest.mark.parametrize('kwargs',[{'missing_jaw':True},{'direct_hand':True}])
def test_single_pad_or_direct_hand_transport_cannot_earn_tool_operation(kwargs):
    result=synthetic_case(**kwargs)
    assert result['complete'],result['errors']
    assert not result['detail']['tube_carried']
    assert result['released_stable_s']==0.


def test_no_contact_static_destination_is_not_an_operation():
    task,frame=setup('tube_clamp_transfer');observer=NativeTaskObserver(task);observer.on_frame(frame)
    for tick in range(1,1001):
        frame.update(baseline=False,time_s=tick*.0005,native_time_s=tick*.0005,native_step=tick)
        frame['objects']['tube']['position_m']=[.92,.57,.9063]
        observer.on_frame(frame)
    result=observer.on_observe()
    assert result['complete'] and not result['detail']['tube_carried'] and result['released_stable_s']==0.


def test_return_tray_contains_closed_tool_and_clears_receiver_handle_path():
    stage=Usd.Stage.CreateInMemory();meta=author_task(stage)
    tray=meta['fixture_metadata']['tool_rest']
    lo,hi=np.asarray(tray['local_bounds_m'])+tray['initial_position_m']
    for obj in meta['objects']:
        if obj['id']=='tube':continue
        bounds=np.asarray(obj['local_bounds_m'])+obj['initial_position_m']
        assert np.all(bounds[0,:2]>=lo[:2])
        assert np.all(bounds[1,:2]<=hi[:2])
    # Nominal receiver tool handle starts at y .571-.018, outside the lip.
    assert .571-.018-hi[1]>=.0029


@pytest.mark.parametrize('tool_degrees,tube_degrees,expected',[(45,45,True),(90,90,True),(45,0,False)])
def test_common_tilt_is_allowed_but_relative_axis_mismatch_is_rejected(tool_degrees,tube_degrees,expected):
    from scenes.workrooms.laboratory.native_evidence import rotation
    task,frame=setup('tube_clamp_transfer');observer=NativeTaskObserver(task);observer.on_frame(frame)
    def quaternion(degrees):
        a=np.radians(degrees)/2
        return [float(np.cos(a)),float(np.sin(a)),0.,0.]
    tool_q=quaternion(tool_degrees);tube_q=quaternion(tube_degrees);R=rotation(tool_q)
    origin=np.array([.50,.55,1.15]);meta=task['asset_metadata']
    frame.update(baseline=False,time_s=.0005,native_time_s=.0005,native_step=1)
    for name,offset,q in [('clamp_frame',[0,0,0],tool_q),('clamp_jaw',[0,-.010,0],tool_q),('tube',[.060,-.003,0],tube_q)]:
        frame['objects'][name].update(position_m=(origin+R@offset).tolist(),quaternion_wxyz=q)
    frame['contacts']=[{'actor0':meta['actors']['tube'],'actor1':meta['actors'][name],
        'time_s':.0005,'impulse_vector_ns':(R@[0,sign*.0008,0]).tolist(),
        'impulse_magnitude_sum_ns':.0008,'min_separation_m':0.}
        for name,sign in [('clamp_frame',1),('clamp_jaw',-1)]]
    observer.on_frame(frame);state=observer.on_observe()
    assert state['complete'],state['errors']
    assert state['detail']['clamp_holding_tube'] is expected
