"""Source-bound object dynamics probe for the storage-box task.

Explicit bounded wrenches act on lid/catches and the collected item only.
No robot is present, no task success is claimed, and no post-baseline state
setter is used. Raw native contacts, body state and requested loads are saved.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
from bench.isaacsim.release import digest, dump, task_inputs, _frozen_evaluator_module
from bench.paths import OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.board_game_runtime_smoke import gpu_selection
from scripts.validation.living_tower_interactions import array, fast_raw_contacts

PROJECT=Path(__file__).resolve().parents[2]
CAPACITY=8192


def blend(t,start,end):
    a=float(np.clip((t-start)/(end-start),0.,1.));return a*a*(3-2*a)


def targets(t):
    """Negative lock test, real opening/transfer, unlatched negative, closing."""
    if t<.6: return 'settle',None,None,None
    if t<1.2: return 'locked_lift',None,'locked_probe',None
    catch=95.*blend(t,1.2,2.7)
    lid=70.*blend(t,2.7,4.2)
    phase='unlock' if t<2.7 else 'open'
    item=None
    if 4.2<=t<8.7:
        item=[.65+.29*blend(t,5.7,7.2),-.12,.809+.181*blend(t,4.2,5.7)-.184*blend(t,7.2,8.7)]
        phase='item_lift' if t<5.7 else 'item_transfer' if t<7.2 else 'item_lower'
    if 8.7<=t<10.2: phase='close_lid';lid=70.*(1-blend(t,8.7,10.2))
    if 10.2<=t<10.7: return 'unlatched_release',None,None,None
    if 10.7<=t<12.2: phase='close_catches';catch=95.*(1-blend(t,10.7,12.2));lid=0.
    if t>=12.2: return 'released',None,None,None
    return phase,catch,lid,item


def aggregate(rows,t):
    groups=defaultdict(list)
    for row in rows: groups[row['a'],row['b']].append(row)
    return [{'actor0':a,'actor1':b,'time_s':t,'point_count':len(v),
             'impulse_vector_ns':np.sum([r['impulse_on_a_n_s'] for r in v],axis=0).tolist(),
             'impulse_magnitude_sum_ns':float(sum(np.linalg.norm(r['impulse_on_a_n_s']) for r in v)),
             'min_separation_m':min(r['separation_m'] for r in v)} for (a,b),v in groups.items()]


def run(args):
    output=args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT): raise ValueError('Use shared experiment storage')
    source=verify_if_snapshot(PROJECT,required_modules=('scripts.validation.storage_box_interactions',),entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve()!=PROJECT: raise ValueError('Use a complete frozen source as cwd')
    output.mkdir(parents=True,exist_ok=False)
    report={'schema':'storage_box_native_interactions.v1','passed':False,'case':args.case,
        'source':source,'source_manifest_sha256':digest(PROJECT/'source_snapshot_manifest.json'),
        'gpu':gpu_selection(args.gpu_uuid),'robot_present':False,'robot_manipulation_qualified':False,
        'task_success_claimed':False,'physics_accepted':False,'model_calls':0,'post_baseline_state_writes':0}
    app=None
    try:
        _,scene,task,_=task_inputs(args.release,'storage_box')
        report['release']={'path':str(args.release.resolve()),'manifest_sha256':digest(args.release/'release_manifest.json')}
        frozen=_frozen_evaluator_module(args.release.resolve())
        observer=frozen.NativeTaskObserver(task);scorer=frozen.SequenceEvaluator(task)
        dt=scene['isaacsim']['physics_dt_s'];duration=13.5 if args.case=='cycle' else 2.
        report.update(physics_dt_s=dt,duration_s=duration,physics_options=scene['isaacsim'])
        from isaacsim import SimulationApp
        override=os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER')=='1'
        report['driver_validation_override']=override
        app=SimulationApp({'headless':True,'width':64,'height':64,'multi_gpu':False,
            'active_cuda_gpus':[0],'max_gpu_count':1,'physics_gpu':0,'fast_shutdown':True,
            'extra_args':['--/rtx/verifyDriverVersion/enabled=false'] if override else []},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        from isaacsim.core.api import World
        from pxr import PhysxSchema
        from bench.simulation.isaacsim_backend.workroom import add_workroom
        from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle,joint_positions
        from bench.simulation.isaacsim_backend.static_mesh_contacts import adapt_static_meshes,FixedActorGuard
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=.02,backend='numpy')
        context=world.get_physics_context();context.enable_gpu_dynamics(True);context.set_broadphase_type('GPU');context.set_solver_type('TGS')
        api=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(context.prim_path))
        api.CreateEnableCCDAttr(False);api.CreateGpuMaxRigidContactCountAttr(2**20)
        api.CreateGpuMaxRigidPatchCountAttr(2**18);api.CreateGpuFoundLostAggregatePairsCapacityAttr(2**14)
        api.CreateGpuTotalAggregatePairsCapacityAttr(2**18)
        room=add_workroom(stage,scene['workroom'],surface_materials=scene['isaacsim']['workroom_surface_materials'])
        bundle=add_bundle(stage,scene['native_task_bundle'],physics_scene_path=context.prim_path)
        paths={n:v['path'] for n,v in bundle['objects'].items()}
        adapted=adapt_static_meshes(stage,scene['isaacsim']['static_mesh_contact_mode'])
        fixed_paths=bundle['fixtures'].get('native_bundle_fixed',{}).get('collision_paths',[])
        static=sorted({adapted['collider_actor_paths'].get(p,p) for p in room['contact_filter_paths']+fixed_paths}|set(adapted['kinematic_body_paths']))
        stage.Export(str(output/'initial.usda'));dump(output/'scene.json',scene);dump(output/'task.json',task)
        world.reset();sim=world.physics_sim_view
        bodies=sim.create_rigid_body_view(list(paths.values()));reverse={p:n for n,p in paths.items()}
        names=[reverse[p] for p in bodies.prim_paths]
        original=[[ *bundle['objects'][n]['position_m'],*bundle['objects'][n]['quaternion_wxyz'][1:],bundle['objects'][n]['quaternion_wxyz'][0]] for n in names]
        bodies.set_transforms(np.asarray(original,dtype=np.float32),np.arange(len(names),dtype=np.int32))
        bodies.set_velocities(np.zeros((len(names),6),dtype=np.float32),np.arange(len(names),dtype=np.int32))
        views={n:sim.create_rigid_body_view(p) for n,p in paths.items()}
        report['native_masses_kg']={n:float(array(v.get_masses()).reshape(-1)[0]) for n,v in views.items()}
        report['reset_boundary_writes']={'poses':len(names),'velocities':len(names),'after_baseline':0}
        filters=[*paths.values(),*static]
        contact=sim.create_rigid_contact_view(list(paths.values()),filter_patterns=[filters]*len(paths),max_contact_data_count=CAPACITY)
        sensors=list(contact.sensor_paths);mapped=list(contact.filter_paths)
        if mapped and isinstance(mapped[0],str): mapped=[mapped]*len(sensors)
        if any(set(m)!=set(filters) for m in mapped): raise ValueError('Native contact inventory differs')
        report['contact_mapping']={'sensors':sensors,'filters':mapped,'capacity':CAPACITY}
        fixed=sim.create_rigid_body_view(adapted['kinematic_body_paths']) if adapted['kinematic_body_paths'] else None
        guard=FixedActorGuard(list(fixed.prim_paths),array(fixed.get_transforms())) if fixed else None
        start=float(world.current_time);start_step=int(world.current_time_step_index)
        previous=None;checks_rows=[];scores=[];max_penetration=0.;peak_force=peak_torque=0.;force_calls=0
        stage_times={};locked_max_angle=0.;locked_contact_impulse=0.
        progress=time.monotonic()
        with gzip.open(output/'native_frames.jsonl.gz','xt') as stream:
            for tick in range(round(duration/dt)+1):
                commands={};phase='baseline'
                if tick:
                    phase,catch_goal,lid_goal,item_goal=targets((tick-1)*dt) if args.case=='cycle' else ('reset',None,None,None)
                    q=frozen.angles(previous['objects']); rb=np.asarray(frozen.rotation(previous['objects']['box']['quaternion_wxyz']))
                    for n in ('lid','left_catch','right_catch','item'):
                        force=np.zeros(3);torque=np.zeros(3);active=False
                        vel=np.asarray(previous['objects'][n]['angular_velocity_rad_s'])
                        if n=='lid' and lid_goal is not None:
                            active=True
                            if lid_goal=='locked_probe': torque=rb[:,0]*-.25
                            else:
                                err=math.radians(-lid_goal+q['lid']);speed=float(np.dot(vel,rb[:,0]))
                                tau=np.clip(.65*err-.10*speed-.12*9.81*.12*math.cos(math.radians(q['lid'])),-.40,.40)
                                torque=rb[:,0]*tau
                        elif n in ('left_catch','right_catch') and catch_goal is not None:
                            active=True;sign=1. if n=='left_catch' else -1.
                            speed=float(np.dot(vel,rb[:,2]))*sign
                            torque=rb[:,2]*sign*np.clip(.09*math.radians(catch_goal-q[n])-.012*speed,-.08,.08)
                        elif n=='item' and item_goal is not None:
                            active=True
                            state=previous['objects'][n]
                            force=np.clip(400.*(np.asarray(item_goal)-state['position_m'])-8.*np.asarray(state['linear_velocity_m_s'])+[0,0,.08*9.81],-4.,4.)
                            # Only keep the item's original upright orientation; no position writes.
                            rot=np.asarray(frozen.rotation(state['quaternion_wxyz']))
                            error=.5*np.array([rot[2,1]-rot[1,2],rot[0,2]-rot[2,0],rot[1,0]-rot[0,1]])
                            torque=np.clip(-.20*error-.02*vel,-.04,.04)
                        if active:
                            views[n].apply_forces_and_torques_at_position(np.asarray([force],np.float32),np.asarray([torque],np.float32),None,np.array([0],np.int32),True)
                            commands[n]={'force_world_n':force.tolist(),'torque_world_nm':torque.tolist()}
                            force_calls+=1;peak_force=max(peak_force,float(np.linalg.norm(force)));peak_torque=max(peak_torque,float(np.linalg.norm(torque)))
                    world.step(render=False)
                    if int(world.current_time_step_index)!=start_step+tick or abs(float(world.current_time)-start-tick*dt)>2.e-6:
                        raise ValueError('Native physical clock discontinuity')
                if guard: guard.check(list(fixed.prim_paths),array(fixed.get_transforms()))
                poses,velocities=array(bodies.get_transforms()),array(bodies.get_velocities())
                if not np.isfinite(poses).all() or not np.isfinite(velocities).all(): raise ValueError('Nonfinite native body state')
                objects={n:{'position_m':poses[i,:3].tolist(),'quaternion_wxyz':[float(poses[i,6]),*poses[i,3:6].tolist()],
                    'linear_velocity_m_s':velocities[i,:3].tolist(),'angular_velocity_rad_s':velocities[i,3:].tolist()} for i,n in enumerate(names)}
                raw=[]
                if tick:
                    arrays=[array(v) for v in contact.get_contact_data(dt=1.)]
                    raw=fast_raw_contacts(arrays,sensors,mapped)
                    max_penetration=max(max_penetration,max((-r['separation_m'] for r in raw),default=0.))
                frame={'baseline':tick==0,'time_s':tick*dt,'native_time_s':float(world.current_time),
                    'native_step':tick,'physics_dt_s':dt,'objects':objects,'joints':joint_positions(objects,bundle['joints']),
                    'contacts':aggregate(raw,tick*dt),'particle_sets':{},'engineering_commands':commands,'phase':phase,'raw_contacts':raw}
                observer.on_frame(frame);observed=observer.on_observe()
                if not observed['complete']: raise ValueError(str(observed['errors']))
                frame['observed']=observed
                stage_times.setdefault(phase,{'time_s':tick*dt,'native_step':tick,'objects':objects})
                if phase=='locked_lift':
                    locked_max_angle=max(locked_max_angle,observed['angles_deg']['lid'])
                    for row in raw:
                        if paths['lid'] in (row['a'],row['b']) and any(paths[n] in (row['a'],row['b']) for n in ('left_catch','right_catch')):
                            locked_contact_impulse+=float(np.linalg.norm(row['impulse_on_a_n_s']))
                for name,condition in [('catches_unlocked',all(observed['angles_deg'][n]>=85 for n in ('left_catch','right_catch'))),('lid_open',observed['opened']),('item_collected',observed['transferred']),('reclosed_stable',observed['physical_released_stable_s']>=.3)]:
                    if condition: stage_times.setdefault(name,{'time_s':tick*dt,'native_step':tick,'objects':objects})
                stream.write(json.dumps(frame,separators=(',',':'),allow_nan=False)+'\n')
                if tick%40==0:
                    scores.append(scorer.update({'time_s':tick*dt,'task_state':observed}))
                    checks_rows.append({'engineering_phase':phase,**observed})
                previous=frame
                if time.monotonic()-progress>20.:
                    print(json.dumps({'tick':tick,'time_s':tick*dt,'phase':phase,'angles':observed['angles_deg'],'collected':observed['collected']}),flush=True);progress=time.monotonic()
        tail=[v for v in checks_rows if v['time_s']>=duration-.5-1.e-9]
        checks={'complete_native_steps':tick==round(duration/dt),'no_deep_penetration':max_penetration<=.003,
            'no_robot_credit':all(not s['success'] for s in scores),'referee_evidence_complete':all(s['referee']['state']!='evaluator_invalid' for s in scores)}
        if args.case=='cycle':
            locked=[v for v in checks_rows if v['engineering_phase']=='locked_lift']
            unlatched=[v for v in checks_rows if v['engineering_phase']=='unlatched_release' and v['time_s']>10.3]
            checks.update(locked_catches_physically_block_lid=bool(locked) and locked_max_angle<5. and locked_contact_impulse>0.,
                opened_after_unlock=previous['observed']['opened'],real_item_collection=previous['observed']['transferred'],
                unlatched_closed_lid_not_complete=bool(unlatched) and all(v['collected'] and not v['closed'] and v['physical_released_stable_s']==0 for v in unlatched),
                complete_released_stability=all(v['physical_released_stable_s']>=.3 and v['closed'] and v['collected'] for v in tail))
        else: checks.update(reset_no_mechanism_progress=all(not v['opened'] and not v['transferred'] for v in checks_rows))
        report.update(passed=all(bool(v) for v in checks.values()),checks={k:bool(v) for k,v in checks.items()},native_steps=tick,
            native_frames=tick+1,maximum_penetration_m=max_penetration,peak_requested_force_n=peak_force,
            peak_requested_torque_nm=peak_torque,force_calls=force_calls,final_state=previous['observed'],
            fixed_scenery=guard.report() if guard else None,stage_times=stage_times,locked_lid_max_angle_deg=locked_max_angle,
            locked_lid_catch_impulse_sum_ns=locked_contact_impulse,scope='Real object mechanisms under declared bounded loads; zero robot/game/task credit.')
        dump(output/'observations.json',checks_rows);dump(output/'scores.json',scores)
        stage_labels={
            'baseline':['left_catch_closed','right_catch_closed','lid_handle_closed'],
            'catches_unlocked':['left_catch_open','right_catch_open','lid_handle_closed'],
            'item_lift':['item_supply','lid_handle_open','left_catch_open','right_catch_open'],
            'item_collected':['item_collection','lid_handle_open'],
            'reclosed_stable':['item_collection','lid_handle_closed','left_catch_closed','right_catch_closed']}
        workspace_stages=[]
        for label,labels in stage_labels.items():
            if label not in stage_times: continue
            row=stage_times[label]
            workspace_stages.append({'label':label,'sim_time_s':row['time_s'],'trace_frame_index':row['native_step'],
                'body_poses':[{'id':n,'prim_path':paths[n],'position_m':b['position_m'],'quaternion_wxyz':b['quaternion_wxyz']}
                              for n,b in row['objects'].items()], 'target_labels':labels})
        dump(output/'workspace_stages.json',{'workspace_stages':workspace_stages,
            'native_trace_files':[{'path':str(output/'native_frames.jsonl.gz'),'sha256':digest(output/'native_frames.jsonl.gz')}],
            'shared_approach_candidates_m':[[.65,-.12,1.20],[.65,.20,1.05],[.80,-.12,1.15]],
            'scope':'Offline geometric approach candidates only; not robot trajectory or native manipulation qualification.'})
        report['evidence_sha256']={n:digest(output/n) for n in ('initial.usda','scene.json','task.json','native_frames.jsonl.gz','observations.json','scores.json','workspace_stages.json')}
        verify_if_snapshot(PROJECT,required_modules=('scripts.validation.storage_box_interactions',),entrypoint_file=Path(__file__))
    except BaseException as exc:
        traceback.print_exc();report['error']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        try:
            dump(output/'qualification.json',report);print(json.dumps({'passed':report['passed'],'output':str(output),'error':report.get('error'),'checks':report.get('checks')}),flush=True)
        finally:
            if app: app.close()
    return 0 if report['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--case',choices=('reset','cycle'),required=True)
    raise SystemExit(run(parser.parse_args()))
