from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from scenes.workrooms.living_room.composition import AUTHORING_TASK_IDS, build_task_inputs, stand_usda
from scenes.workrooms.living_room.evaluation import SequenceEvaluator
from scenes.workrooms.living_room.native_evidence import raw_contacts, cavity_contains, normalized_contacts, extrema
from scenes.workrooms.living_room.layer_release import initial_state
from scenes.workrooms.living_room.runtime import LivingTaskRuntime


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    root=tmp_path_factory.mktemp('living_inputs')
    result={}
    for task in AUTHORING_TASK_IDS:
        path=root/(task+'.usda');path.write_text(stand_usda(task))
        result[task]=build_task_inputs(task,stand_path=path)
    return result


@pytest.mark.parametrize('task',AUTHORING_TASK_IDS)
def test_composition_preserves_candidate_and_controller(inputs,task):
    value=inputs[task];scene=value['scene'];spec=value['task']
    assert initial_state(scene,spec)==value['initial_state']
    assert scene['isaacsim']['physics_dt_s']==(.00025 if task=='tower_extraction' else .0005)
    assert scene['task_layers'][0]['target_prim_path']=='/World/LivingTask'
    has_particles=any(layer.get('particle_sets') for layer in scene['task_layers'])
    assert scene.get('native_readback_profile','cuda_tensors_v1')==(
        'gpu_solver_usd_host_v1' if has_particles else 'cuda_tensors_v1')
    assert scene['living_task_provenance']['original_robot_dynamics_and_gains_preserved'] is True
    assert spec['physics_accepted'] is False
    if task=='tower_extraction':
        assert set(scene['robot']['instances'])=={'front','rear'}
        assert len(value['initial_state']['joint_positions_rad'])==58
        assert {name:row['mount_link'] for name,row in scene['cameras'].items() if row['kind']=='link'}=={
            'front_wrist':'front_hand_flange','rear_wrist':'rear_hand_flange'}
    else:
        assert scene['robot']['base_position_m']==[.65,-.88,.78]


def test_unknown_explicit_camera_mount_rejects_before_native_start(inputs):
    value=inputs['tower_extraction'];scene=deepcopy(value['scene'])
    scene['cameras']['front_wrist']['mount_link']='right_hand_flange'
    with pytest.raises(ValueError,match='unknown declared robot link'):
        initial_state(scene,value['task'])


def test_legacy_remote_authoring_is_removed_before_creating_files(tmp_path):
    from scenes.workrooms.living_room.layer_release import prepare_task_release
    with pytest.raises(ValueError, match='legacy remote authoring is removed'):
        build_task_inputs('remote_operation')
    output = tmp_path/'removed_remote'
    with pytest.raises(ValueError, match='legacy remote authoring is removed'):
        prepare_task_release('remote_operation', output)
    assert not output.exists()


def test_signed_contact_duplicate_and_disagreement():
    arrays=[np.array([[.2],[-.2],[0.]]),np.array([[0.,0.,1.],[0.,0.,1.],[0.,0.,0.]]),
            np.array([[0.,0.,1.],[0.,0.,1.],[0.,0.,0.]]),np.array([[0.],[0.],[0.]]),
            np.array([[1],[1]]),np.array([[0],[1]])]
    rows=raw_contacts(arrays,['a','b'],[['b'],['a']])
    assert len(rows)==1 and rows[0]['impulse_on_a_n_s']==[0.,0.,.2]
    arrays[0][1]=-.3
    with pytest.raises(ValueError,match='disagree'):raw_contacts(arrays,['a','b'],[['b'],['a']])


@pytest.mark.parametrize('game',[False,True])
def test_room_worktop_contact_has_explicit_support_role_without_guessing_unknown_props(game):
    paths={'block':'/World/LivingTask/Blocks/block'}
    row={'a':paths['block'],'b':'/World/Room/Geometry/worktop/Collider',
         'point_m':[.65,0.,.78],'impulse_on_a_n_s':[0.,0.,.0004],'separation_m':0.}
    mapped=normalized_contacts([row],paths,game=game)
    assert mapped==[{**row,'a':'block','b':'table'}]
    for unknown in ('/World/Room/Geometry/worktop_extra/Collider','/World/Props/worktop/Collider'):
        if game:
            with pytest.raises(ValueError,match='Unmapped'):
                normalized_contacts([{**row,'b':unknown}],paths,game=game)
        else:
            assert normalized_contacts([{**row,'b':unknown}],paths,game=game)[0]['b'].startswith('external/')


def test_first_invalid_evidence_reason_is_preserved_after_later_samples(inputs):
    scorer=SequenceEvaluator(inputs['tower_extraction']['task'])
    first=scorer.update({'schema':'living_native_evidence_v1','complete':False})
    assert first['status']=='invalid'
    later=scorer.update({'schema':'living_native_evidence_v1','complete':True})
    assert later==first and scorer.previous is None


@pytest.mark.parametrize('qa,qb',[
    # Actual adjacent native Task4 orientations: numpy dot rounds differently
    # from the normalized reference observation's scalar dot at this angle.
    ([0.9999999999998448,-5.569254426517591e-7,1.1502974572861637e-8,7.95410493026215e-9],
     [0.9999999999991467,-1.3058347576577715e-6,-1.0979798403800213e-8,3.750990629211156e-8]),
    # A nonzero endpoint rotation must not disappear into a near-one deadband.
    ([1.,0.,0.,0.],[1.,5.e-8,0.,0.]),
])
@pytest.mark.parametrize('dt',[.0005,.00025])
def test_native_interval_angular_bound_contains_normalized_endpoint_motion(qa,qb,dt):
    from scenes.workrooms.living_room.native_evidence import quaternion
    from scenes.workrooms.living_room.tasks.tower_extraction.evaluator import _angle
    frames=[{'time_s':t,'objects':{'block':{'position_m':[0.,0.,1.],
             'quaternion_wxyz':q,'linear_velocity_m_s':[0.,0.,0.],
             'angular_velocity_rad_s':[0.,0.,0.]}}} for t,q in [(0.,qa),(dt,qb)]]
    required=_angle(quaternion(qa),quaternion(qb))/dt
    assert required>0.
    assert extrema(frames,'block')[1]>=required


def test_cavity_preserves_spout_and_excludes_wall():
    profile=[[.006,0,0],[.006,.045,0],[.075,.049,0],[.096,.042,.02],[.102,.040,.030]]
    inside=cavity_contains([[.060,0,.091],[0,.060,.091],[0,0,.11],[0,0,.04]],profile)
    assert inside.tolist()==[True,False,False,True]


def test_game_admission_requires_turn_and_inactive_drive_hold(inputs):
    runtime=LivingTaskRuntime.__new__(LivingTaskRuntime)
    runtime.task=inputs['tower_extraction']['task'];runtime.evaluator=SequenceEvaluator(runtime.task)
    runtime.evaluator.game.status='await_intent';runtime.actions=0;runtime.executed=[]
    names=list(runtime.task['initial_state']['joint_positions_rad'])
    from types import SimpleNamespace
    targets=np.array([[runtime.task['initial_state']['joint_positions_rad'][n] for n in names]],dtype=np.float32)
    runtime.env=SimpleNamespace(joint_names=names,_targets=targets,time_s=0.)
    command={'joint_positions_rad':dict(zip(names,targets[0].tolist())),'duration_s':.02}
    actor={'player_id':'player_0','turn_index':0}
    assert runtime.authorize_step(command,actor)==0 and runtime.executed==[]
    with pytest.raises(ValueError):runtime.authorize_step(command,{'player_id':'player_0','turn_index':1})
    bad=deepcopy(command);bad['joint_positions_rad']['fr3_rear_joint1']+=.1
    with pytest.raises(ValueError,match='Inactive'):runtime.authorize_step(bad,actor)


@pytest.mark.parametrize('status,joint',[
    ('await_intent','fr3_rear_joint6'),
    ('reset_pending','fr3_front_joint6'),
    ('reset_pending','fr3_rear_joint6'),
])
def test_game_drive_guard_compares_native_float32_at_urdf_boundary(inputs,status,joint):
    from types import SimpleNamespace
    runtime=LivingTaskRuntime.__new__(LivingTaskRuntime)
    runtime.task=inputs['tower_extraction']['task'];runtime.evaluator=SequenceEvaluator(runtime.task)
    runtime.evaluator.game.status=status;runtime.actions=0;runtime.executed=[]
    names=list(runtime.task['initial_state']['joint_positions_rad'])
    command={'joint_positions_rad':deepcopy(runtime.task['initial_state']['joint_positions_rad']),
             'duration_s':.02}
    command['joint_positions_rad'][joint]=4.6216  # Original FR3 joint6 upper bound.
    targets=np.array([[command['joint_positions_rad'][n] for n in names]],dtype=np.float32)
    runtime.env=SimpleNamespace(joint_names=names,_targets=targets,time_s=0.)
    actor={'player_id':'player_0','turn_index':0}
    index=names.index(joint)
    assert abs(command['joint_positions_rad'][joint]-float(targets[0,index]))>1.e-7
    assert runtime.authorize_step(command,actor)==0 and runtime.executed==[]
    changed=deepcopy(command)
    changed['joint_positions_rad'][joint]=float(np.nextafter(targets[0,index],np.float32(-np.inf)))
    with pytest.raises(ValueError,match='Inactive'):
        runtime.authorize_step(changed,actor)


def tower_baseline(task):
    c=task['evaluation']['reference_configuration'];objects={}
    if task['id']=='tower_extraction':
        g=c['geometry'];origin=np.array([*g['center_xy_m'],g['table_top_z_m']])
        initial={name:{'position_m':(origin+np.asarray(row['center_h'])*g['block_size_m'][2]).tolist(),
                       'quaternion_wxyz':row['quaternion_wxyz']} for name,row in c['blocks'].items()}
    else:
        initial=c['initial_objects']
    for name,row in initial.items():
        objects[name]={**deepcopy(row),'linear_velocity_m_s':[0.,0.,0.],'angular_velocity_rad_s':[0.,0.,0.]}
    return {'schema':'living_native_evidence_v1','complete':True,'episode_id':'test','step_index':0,'time_s':0.,
            'objects':objects,'joints':{},'particle_sets':{},'raw_contacts':[],'robot_action_count':0}


@pytest.mark.parametrize('task_id',['tower_building','tower_extraction'])
def test_tower_reset_cannot_award_and_missing_substep_invalidates(inputs,task_id):
    task=inputs[task_id]['task'];scorer=SequenceEvaluator(task);frame=tower_baseline(task)
    if task_id=='tower_extraction':
        frame['hand_clearance_m']={name:{'front':1.,'rear':1.} for name in frame['objects']}
    result=scorer.update(frame)
    assert result['status']!='invalid' and not result['success']
    frame['step_index']=1;frame['time_s']=task['evaluation']['physics_dt_s']
    result=scorer.update(frame)
    assert result['status']!='invalid' and not result['success']
    frame['step_index']=3;frame['time_s']=3*task['evaluation']['physics_dt_s']
    assert scorer.update(frame)['status']=='invalid'


@pytest.mark.parametrize('purpose',['default','render','proxy','guide'])
def test_collision_clearance_uses_authored_relative_geometry_then_native_pose(purpose):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    from types import SimpleNamespace
    from scenes.workrooms.living_room.runtime import CollisionClearance
    stage=Usd.Stage.CreateInMemory();links={}
    for side,x in [('front',0.),('rear',3.)]:
        root='/World/Robot/'+side+'/'+side+'_hand'
        body=UsdGeom.Xform.Define(stage,root);body.AddTranslateOp().Set(Gf.Vec3d(x,0,0))
        cube=UsdGeom.Cube.Define(stage,root+'/collider');cube.CreateSizeAttr(.1)
        cube.CreatePurposeAttr(purpose)
        cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim());links[side+'_hand']=root
    # Actual native front body is one metre away from its authored pose.
    state={'body_q':np.array([[1.,0,0,0,0,0,1],[3.,0,0,0,0,0,1]])}
    env=SimpleNamespace(stage=stage,body_names=list(links),scene_info={'robot_link_paths':links},capture_state=lambda:state)
    clearance=CollisionClearance(env).measure(state,{'block':{'position_m':[1.,0,0],'quaternion_wxyz':[1.,0,0,0]}},[.025]*3)
    assert clearance['block']['front']==0.
    assert clearance['block']['rear']>1.9
