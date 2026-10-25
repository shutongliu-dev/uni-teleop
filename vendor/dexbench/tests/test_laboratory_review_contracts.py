"""Independent CPU referee probes. Synthetic evidence is not native task success."""
from copy import deepcopy
import importlib
import math
import pytest
from pxr import Usd
from scenes.workrooms.laboratory import rules
from scenes.workrooms.laboratory.evaluation import NativeTaskObserver, SequenceEvaluator
from scenes.workrooms.laboratory.native_evidence import fixed


def setup(task_id):
    meta=importlib.import_module(f'scenes.workrooms.laboratory.tasks.{task_id}.assets').author_task(Usd.Stage.CreateInMemory())
    task={'schema_version':'laboratory-manipulation-task-v1','id':task_id,'episode_seed':0,
        'evaluation':rules.evaluation_configuration(task_id,meta,room_support_paths={'table':['/World/Geometry/worktop/Collider']}),
        'public_rules':rules.public_rules(task_id),'asset_metadata':meta}
    frame={'baseline':True,'time_s':0.,'native_time_s':0.,'native_step':0,'physics_dt_s':.0005,
        'objects':{r['id']:fixed(r['initial_pose_xyzw']) for r in meta['objects']},'contacts':[],'joints':{},'particle_sets':{}}
    if task_id=='pipette_transfer':
        from scenes.workrooms.laboratory.tasks.pipette_transfer.liquid import ReducedLiquidDynamics
        provider=ReducedLiquidDynamics(task);provider.on_frame(frame);frame['reduced_dynamics']=provider.on_observe()
    observer=NativeTaskObserver(task);observer.on_frame(frame)
    assert observer.on_observe()['complete']
    return task,meta,frame,observer


def contacts(meta,pairs,t):
    paths={**meta['actors'],**meta['fixtures'],'table':'/World/Room/Geometry/worktop/Collider', 'robot_hand':'/World/Robot/Hand/Collider'}
    return [{'actor0':paths[a],'actor1':paths[b],'time_s':t,'impulse_magnitude_sum_ns':.001,
        'impulse_vector_ns':[0.,0.,.001],'min_separation_m':0.} for a,b in pairs]


def step(frame,observer,meta,pairs):
    tick=frame['native_step']+1
    frame.update(baseline=False,time_s=tick*.0005,native_time_s=tick*.0005,native_step=tick,
        contacts=contacts(meta,pairs,tick*.0005))
    if 'reduced_dynamics' in frame:frame['reduced_dynamics'].update(native_step=tick,native_time_s=tick*.0005)
    observer.on_frame(frame)
    result=observer.on_observe()
    assert result['complete'],result['errors']
    return result


def test_positive_sort_and_permuted_rack_goals_are_achievable_in_referee():
    for task_id in ('sample_sorting','vial_racking'):
        task,m,f,o=setup(task_id);p=m['task_parameters'];pairs=[]
        if task_id=='sample_sorting':
            for name,goal in p['targets'].items():
                f['objects'][name]=fixed(goal['goal_pose_xyzw']);pairs.append((name,goal['fixture']))
        else:
            names=list(f['objects'])
            for name,goal in zip(names,list(reversed(p['holes']))):
                f['objects'][name]=fixed(goal['goal_pose_xyzw']);pairs.append((name,goal['fixture']))
        for _ in range(700):result=step(f,o,m,pairs)
        assert result['released_stable_s']>=.3
        assert result['phase']=='settling'


def test_cap_open_park_three_start_reclose_and_home_is_achievable():
    task,m,f,o=setup('cap_placement');p=m['task_parameters'];thread=p['thread']
    b=list(p['holder_center_m']);f['objects']['sample_bottle']=fixed([*b,0,0,0,1])
    f['objects']['cap']=fixed([b[0],b[1],b[2]+thread['closed_cap_z_m'],0,0,0,1])
    pairs=[('sample_bottle','bottle_holder'),('cap','sample_bottle')]
    step(f,o,m,pairs)
    for i in range(1,50):
        yaw=thread['required_turn_rad']*i/49
        z=thread['closed_cap_z_m']+.035*i/49
        f['objects']['cap']=fixed([b[0],b[1],b[2]+z,0,0,math.sin(yaw/2),math.cos(yaw/2)])
        result=step(f,o,m,[('sample_bottle','bottle_holder')])
    assert result['detail']['cap_opened']
    f['objects']['cap']=fixed([*p['cap_tray_target']['center_m'],0,0,0,1])
    for _ in range(700):result=step(f,o,m,[('sample_bottle','bottle_holder'),('cap','cap_tray')])
    assert result['detail']['cap_parked']
    yaw=2*math.pi/3
    f['objects']['cap']=fixed([b[0],b[1],b[2]+thread['closed_cap_z_m'],0,0,math.sin(yaw/2),math.cos(yaw/2)])
    result=step(f,o,m,pairs);assert result['detail']['cap_reclosed']
    b=p['bottle_home']['center_m'];f['objects']['sample_bottle']=fixed([*b,0,0,0,1])
    f['objects']['cap']=fixed([b[0],b[1],b[2]+thread['closed_cap_z_m'],0,0,math.sin(yaw/2),math.cos(yaw/2)])
    for _ in range(700):result=step(f,o,m,[('sample_bottle','table'),('cap','sample_bottle')])
    assert result['released_stable_s']>=.3


def test_pipette_closed_balance_supported_return_can_reach_referee_goal():
    task,m,f,o=setup('pipette_transfer')
    for name in ('pipette','plunger'):f['objects'][name]['position_m'][0]+=.03
    step(f,o,m,[])
    for name in ('pipette','plunger'):f['objects'][name]['position_m'][0]-=.03
    water=f['reduced_dynamics'];water.update(source_ml=60.,receivers_ml={'receiver_a':10.,'receiver_b':10.},
        aspirated_ml=20.,dispensed_ml=20.,accounted_total_ml=80.)
    pairs=[('source','table'),('receiver_a','table'),('receiver_b','table'),('pipette','pipette_cradle')]
    for _ in range(700):result=step(f,o,m,pairs)
    assert result['released_stable_s']>=.3
    assert result['detail']['tool_returned']


def test_pipette_forged_closed_flag_with_bad_sum_is_invalid():
    task,m,f,o=setup('pipette_transfer')
    f['reduced_dynamics']['receivers_ml']['receiver_a']=10.
    f.update(baseline=False,time_s=.0005,native_time_s=.0005,native_step=1)
    f['reduced_dynamics'].update(native_step=1,native_time_s=.0005)
    o.on_frame(f);result=o.on_observe()
    assert not result['complete']
    assert 'account' in result['errors'][0]


def test_weighing_mass_allows_horizontal_roll_but_requires_cup_load_support():
    task,m,f,o=setup('solid_weighing');p=m['task_parameters']
    selected=p['pellet_ids'][:12]
    scoop_initial=deepcopy(f['objects']['scoop'])
    sx,sy,sz=scoop_initial['position_m']
    # Complete an observable loaded-scoop translation before depositing.
    for i,name in enumerate(selected):
        f['objects'][name]['position_m']=[sx+(.01*(i//3)-.015),sy+.01*(i%3-1),sz+.006]
    step(f,o,m,[(name,'scoop') for name in selected])
    for k in ('scoop',*selected):f['objects'][k]['position_m'][0]+=.03
    step(f,o,m,[(name,'scoop') for name in selected])
    f['objects']['scoop']=scoop_initial
    f['objects']['weighing_cup']['position_m']=[.71,2.075,.9255]
    bases={name:[.695+.01*(i//3),2.075+.01*(i%3-1),.9325] for i,name in enumerate(selected)}
    for name,position in bases.items():f['objects'][name]['position_m']=position.copy()
    base_pairs=[('weighing_cup','balance_platform'),('scoop','scoop_rest')]
    base_pairs += [(name,'source_tray') for name in p['pellet_ids'][12:]]
    # Mere correct containment and cup/platform support must not weigh a set
    # of unsupported grains; later add their actual load path into the cup.
    for _ in range(650):result=step(f,o,m,base_pairs)
    assert result['detail']['cup_sample_mass_g']==pytest.approx(6.)
    assert not result['detail']['cup_grains_supported']
    assert result['released_stable_s']==0
    peak=0.
    for i in range(700):
        horizontal=.0015*math.sin(i*.025)
        for name,position in bases.items():
            f['objects'][name]['position_m']=[position[0]+horizontal,position[1],position[2]]
        result=step(f,o,m,base_pairs+[(name,'weighing_cup') for name in selected])
        peak=max(peak,o.pose_speed[selected[0]]['linear_m_s'])
    assert peak>.03
    assert result['detail']['cup_grains_supported']
    assert result['released_stable_s']>=.3


def test_weighing_carrier_needs_loaded_motion_and_hand_touch_revokes_it():
    task,m,f,o=setup('solid_weighing');name=m['task_parameters']['pellet_ids'][0]
    sx,sy,sz=f['objects']['scoop']['position_m']
    f['objects'][name]['position_m']=[sx,sy,sz+.006]
    for _ in range(5):result=step(f,o,m,[(name,'scoop')])
    assert name not in result['detail']['scooped_ids']
    for key in ('scoop',name):f['objects'][key]['position_m'][0]+=.019
    result=step(f,o,m,[(name,'scoop')]);assert name not in result['detail']['scooped_ids']
    for key in ('scoop',name):f['objects'][key]['position_m'][0]+=.002
    result=step(f,o,m,[(name,'scoop')]);assert name in result['detail']['scooped_ids']
    result=step(f,o,m,[(name,'scoop'),(name,'robot_hand')])
    assert name not in result['detail']['scooped_ids']
    result=step(f,o,m,[(name,'scoop')]);assert name not in result['detail']['scooped_ids']
    for key in ('scoop',name):f['objects'][key]['position_m'][0]+=.021
    result=step(f,o,m,[(name,'scoop')]);assert name in result['detail']['scooped_ids']
