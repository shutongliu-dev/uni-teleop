"""Native passive clamp/tube contact experiments with explicitly bounded tool loads.

No robot is instantiated. Only the clamp frame and thumb jaw receive declared
engineering forces. The tube remains a free body and cannot be attached or
assigned a pose after the reset baseline. Every physics step is archived.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time
import traceback
import numpy as np

from bench.isaacsim.release import digest, dump, task_inputs, _frozen_evaluator_module
from bench.paths import OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.board_game_runtime_smoke import gpu_selection
from scripts.validation.living_tower_interactions import array, fast_raw_contacts
from scenes.workrooms.laboratory.native_evidence import rotation

PROJECT = Path(__file__).resolve().parents[2]
CAPACITY = 4096


def target(t, initial, *, open_only=False):
    # These are engineering tool targets, not hidden task success positions.
    keys = [(0.5,initial), (1.3,[.70,.50,1.080]), (2.5,[.42,.571,1.080]),
            (3.3,[.42,.571,.9065]), (3.8,[.42,.571,.9065]),
            (4.8,[.42,.571,1.0065]), (6.8,[.86,.571,1.0065]),
            (7.8,[.86,.571,.9073]), (8.4,[.86,.571,.9073]),
            (9.4,[.86,.571,1.080]), (10.8,[.70,.50,1.080]), (12.,initial)]
    if t < .5 or t >= 12.:
        return None
    for (a,pa),(b,pb) in zip(keys,keys[1:]):
        if a <= t <= b:
            u=np.clip((t-a)/(b-a),0,1);u=u*u*(3-2*u)
            position=(1-u)*np.asarray(pa)+u*np.asarray(pb)
            break
    opened = open_only or t < 3.3 or 7.8 <= t < 9.4
    return position, .022 if opened else None


def read_contact_rows(view, sensors, mapped, t):
    native = view.get_contact_data(dt=1.)
    counts,starts=array(native[4]),array(native[5]); active=counts>0
    used=int(np.max((starts+counts)[active],initial=0))
    if used>=CAPACITY: raise ValueError('Native contact buffer reached capacity')
    arrays=[array(v[:used+1]) for v in native[:4]]+[counts,starts]
    raw=fast_raw_contacts(arrays,sensors,mapped)
    grouped=defaultdict(list)
    for row in raw: grouped[row['a'],row['b']].append(row)
    rows=[]
    for (a,b),points in grouped.items():
        vectors=np.asarray([p['impulse_on_a_n_s'] for p in points],float)
        rows.append({'actor0':a,'actor1':b,'time_s':t,'point_count':len(points),
            'impulse_vector_ns':vectors.sum(axis=0).tolist(),
            'impulse_magnitude_sum_ns':float(np.linalg.norm(vectors,axis=1).sum()),
            'min_separation_m':min(p['separation_m'] for p in points)})
    return rows,raw


def run(args):
    output=args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT): raise ValueError('Use shared datasets')
    output.mkdir(parents=True,exist_ok=False)
    report={'schema':'laboratory_clamp_object_interactions.v1','passed':False,'case':args.case,
        'robot_present':False,'robot_manipulation_qualified':False,'physics_accepted':False,
        'task_success_claimed':False,'model_calls':0,'post_baseline_pose_writes':0,
        'source_root':str(PROJECT),'task_id':'tube_clamp_transfer'}
    app=None
    try:
        source=verify_if_snapshot(PROJECT,required_modules=('scripts.validation.laboratory_clamp_interactions',),entrypoint_file=Path(__file__))
        if source is None or Path.cwd().resolve()!=PROJECT: raise ValueError('Run frozen source from its root')
        report['source']=source;report['source_manifest_sha256']=digest(PROJECT/'source_snapshot_manifest.json')
        report['gpu']=gpu_selection(args.gpu_uuid)
        _,scene,task,_=task_inputs(args.release,'tube_clamp_transfer')
        report['release']={'path':str(args.release.resolve()),'manifest_sha256':digest(args.release/'release_manifest.json')}
        meta=task['asset_metadata'];dt=scene['isaacsim']['physics_dt_s']
        duration={'canonical':1.6,'transfer':15.2,'open_transport':5.2}[args.case]
        report.update(physics_dt_s=dt,duration_s=duration,physics_options=scene['isaacsim'])
        frozen=_frozen_evaluator_module(args.release.resolve());observer=frozen.NativeTaskObserver(task)
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'width':64,'height':64,'multi_gpu':False,
            'active_cuda_gpus':[0],'max_gpu_count':1,'physics_gpu':0,'fast_shutdown':True,
            'extra_args':['--/rtx/verifyDriverVersion/enabled=false']},experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from pxr import PhysxSchema
        from bench.simulation.isaacsim_backend.workroom import add_workroom
        from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle
        from bench.simulation.isaacsim_backend.static_mesh_contacts import adapt_static_meshes, FixedActorGuard
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        world=World(stage_units_in_meters=1.,physics_dt=dt,rendering_dt=.02,backend='torch',device='cuda:0')
        context=world.get_physics_context();context.enable_gpu_dynamics(True);context.set_broadphase_type('GPU');context.set_solver_type('TGS')
        api=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(context.prim_path))
        api.CreateEnableCCDAttr(False);api.CreateGpuMaxRigidContactCountAttr(2**20)
        api.CreateGpuMaxRigidPatchCountAttr(2**18);api.CreateGpuFoundLostAggregatePairsCapacityAttr(2**14)
        api.CreateGpuTotalAggregatePairsCapacityAttr(2**18)
        if 'enable_external_forces_every_iteration' in scene['isaacsim']:
            api.CreateEnableExternalForcesEveryIterationAttr(scene['isaacsim']['enable_external_forces_every_iteration'])
        room=add_workroom(stage,scene['workroom'],surface_materials=scene['isaacsim']['workroom_surface_materials'])
        bundle=add_bundle(stage,scene['native_task_bundle'],physics_scene_path=context.prim_path)
        paths={name:row['path'] for name,row in bundle['objects'].items()}
        adapted=adapt_static_meshes(stage,scene['isaacsim']['static_mesh_contact_mode'])
        static=sorted({adapted['collider_actor_paths'].get(p,p) for p in room['contact_filter_paths']+
            [p for row in bundle['fixtures'].values() for p in row['collision_paths']]}|set(adapted['kinematic_body_paths']))
        stage.Export(str(output/'initial.usda'));dump(output/'task.json',task);dump(output/'scene.json',scene)
        world.reset();sim=world.physics_sim_view
        bodies=sim.create_rigid_body_view(list(paths.values()));reverse={p:n for n,p in paths.items()}
        names=[reverse[p] for p in bodies.prim_paths]
        device=bodies.get_transforms().device;indices=torch.arange(len(names),dtype=torch.int32,device=device)
        specs={row['id']:row for row in meta['objects']}
        bodies.set_transforms(torch.tensor([specs[n]['initial_pose_xyzw'] for n in names],dtype=torch.float32,device=device),indices)
        bodies.set_velocities(torch.zeros((len(names),6),dtype=torch.float32,device=device),indices)
        report['reset_boundary_writes']={'poses':len(names),'velocities':len(names),'scope':'Before baseline only, restore declared asset after World.reset warmup.'}
        filters=[*paths.values(),*static]
        contact_view=sim.create_rigid_contact_view(list(paths.values()),filter_patterns=[filters for _ in paths],max_contact_data_count=CAPACITY)
        sensors=list(contact_view.sensor_paths);mapped=[list(r) for r in contact_view.filter_paths]
        report['contact_mapping']={'sensors':sensors,'filters':mapped[0],'room_unobserved':room['contact_unobserved_paths']}
        fixed_view=sim.create_rigid_body_view(adapted['kinematic_body_paths']) if adapted['kinematic_body_paths'] else None
        guard=FixedActorGuard(fixed_view.prim_paths,array(fixed_view.get_transforms())) if fixed_view else None
        views={n:sim.create_rigid_body_view(paths[n]) for n in ('clamp_frame','clamp_jaw')}
        one=torch.tensor([0],dtype=torch.int32,device=device)
        report['native_masses_kg']={n:float(m) for n,m in zip(names,array(bodies.get_masses()).reshape(-1))}
        report['engineering_boundary']={'controlled_bodies':['clamp_frame','clamp_jaw'],
            'tube_external_wrench':False,'tube_constraints':False,'frame_force_axis_limit_n':20.,
            'frame_torque_axis_limit_nm':.6,'jaw_open_force_limit_n':12.,
            'frame_position_stiffness_n_m':5000.,'frame_position_damping_n_s_m':30.,
            'frame_orientation_stiffness_nm_rad':4.,'frame_orientation_damping_nm_s_rad':.08,
            'jaw_position_stiffness_n_m':600.,'jaw_position_damping_n_s_m':4.,
            'passive_spring_targets_changed':False,'complete_release_time_s':12.,
            'jaw_opening_reaction':'Equal opposite wrench on clamp frame at measured jaw COM; explicit tool-only thumb/palm boundary.'}
        t0=float(world.current_time);s0=int(world.current_time_step_index)
        saved=[];previous=None;max_pen=0.;peak_force=0.;lift=0.;bilateral_steps=0;last_print=time.monotonic()
        with gzip.open(output/'native_frames.jsonl.gz','wt') as stream:
            for tick in range(round(duration/dt)+1):
                loads={}
                if tick:
                    planned=None if args.case=='canonical' else target((tick-1)*dt,specs['clamp_frame']['initial_pose_xyzw'][:3],open_only=args.case=='open_transport')
                    if planned is not None:
                        pos,q_open=planned;frame=previous['objects']['clamp_frame'];jaw=previous['objects']['clamp_jaw']
                        v=np.asarray(frame['linear_velocity_m_s']);w=np.asarray(frame['angular_velocity_rad_s'])
                        q=np.asarray(frame['quaternion_wxyz']);q=q if q[0]>=0 else -q
                        force=np.clip(5000*(pos-frame['position_m'])-30*v+[0,0,.145*9.81],-20,20)
                        torque=np.clip(-8*q[1:]-.08*w,-.6,.6)
                        loads['clamp_frame']={'force_n':force.tolist(),'torque_nm':torque.tolist(),'target_m':pos.tolist()}
                        if q_open is not None:
                            gap=np.asarray(jaw['position_m'])-frame['position_m']
                            relative_v=np.asarray(jaw['linear_velocity_m_s'])-v
                            opening=float(np.clip(600*(q_open-gap[1])-4*relative_v[1],-12,12))
                            loads['clamp_jaw']={'force_n':[0,opening,0],'torque_nm':[0,0,0],'target_slide_m':q_open}
                            jaw_force=np.array([0.,opening,0.])
                            frame_com=np.asarray(frame['position_m'])+rotation(frame['quaternion_wxyz'])@specs['clamp_frame']['center_of_mass_m']
                            jaw_com=np.asarray(jaw['position_m'])+rotation(jaw['quaternion_wxyz'])@specs['clamp_jaw']['center_of_mass_m']
                            reaction_torque=-np.cross(jaw_com-frame_com,jaw_force)
                            loads['clamp_frame'].update(control_force_n=force.tolist(),control_torque_nm=torque.tolist(),
                                jaw_reaction_force_n=(-jaw_force).tolist(),jaw_reaction_torque_nm=reaction_torque.tolist(),
                                force_n=np.clip(force-jaw_force,-20,20).tolist(),
                                torque_nm=np.clip(torque+reaction_torque,-.6,.6).tolist())
                        for name,load in loads.items():
                            views[name].apply_forces_and_torques_at_position(force_data=torch.tensor([load['force_n']],dtype=torch.float32,device=device),
                                torque_data=torch.tensor([load['torque_nm']],dtype=torch.float32,device=device),position_data=None,indices=one,is_global=True)
                            peak_force=max(peak_force,float(np.linalg.norm(load['force_n'])))
                    world.step(render=False)
                if guard:guard.check(fixed_view.prim_paths,array(fixed_view.get_transforms()))
                poses=array(bodies.get_transforms());vel=array(bodies.get_velocities())
                frame={'baseline':tick==0,'time_s':tick*dt,'native_time_s':float(world.current_time)-t0,
                    'native_step':int(world.current_time_step_index)-s0,'physics_dt_s':dt,
                    'objects':{},'loads':loads}
                if frame['native_step']!=tick:raise ValueError('Missing native step')
                for name,p,v in zip(names,poses,vel):
                    frame['objects'][name]={'position_m':p[:3].tolist(),'quaternion_wxyz':[float(p[6]),*p[3:6].tolist()],
                        'linear_velocity_m_s':v[:3].tolist(),'angular_velocity_rad_s':v[3:].tolist()}
                contacts,raw=([],[]) if tick==0 else read_contact_rows(contact_view,sensors,mapped,tick*dt)
                frame['contacts'],frame['raw_contacts']=contacts,raw
                max_pen=max(max_pen,max((-r['separation_m'] for r in raw),default=0.))
                observer.on_frame(frame);state=observer.on_observe()
                if state['errors']:raise ValueError('Frozen observer error: '+str(state['errors']))
                frame['task_state']=state
                lift=max(lift,frame['objects']['tube']['position_m'][2]-specs['tube']['initial_pose_xyzw'][2])
                bilateral_steps+=int(state.get('detail',{}).get('bilateral_pad_contact',False))
                stream.write(json.dumps(frame,separators=(',',':'),allow_nan=False)+'\n')
                if tick%40==0:saved.append(frame)
                previous=frame
                if time.monotonic()-last_print>25.:
                    print(json.dumps({'time_s':tick*dt,'detail':state.get('detail'),'tube_z':frame['objects']['tube']['position_m'][2]}),flush=True);last_print=time.monotonic()
        tail=[r for r in saved if r['time_s']>=duration-.5-1.e-9]
        final=previous['task_state'];checks={'complete_native_steps':tick==round(duration/dt),'no_deep_contact':max_pen<=.005,
            'observer_complete':final['complete'] and not final['errors'],'no_safety_stop':not final['safety_stop'],
            'no_task_objects_fell':not final['terminal_failure']}
        if args.case=='transfer':
            checks.update(real_passive_tube_lift=lift>=.035,bilateral_contact_dwell=bilateral_steps*dt>=.05,
                measured_grip_transport=final['detail']['tube_carried'],
                final_supported_released_stable=final['released_stable_s']>=.5,
                complete_final_release=all(not r['loads'] for r in tail))
        else:
            checks.update(no_false_transport=not any(r['task_state']['detail']['tube_carried'] for r in saved),
                          no_false_success=final['released_stable_s']==0.)
            if args.case=='open_transport':checks['open_tool_cannot_carry_tube']=lift<.010
        dump(output/'checkpoints.json',saved)
        workspace_stages=[]
        stage_plan=[('reset',0.,['tool_handle','jaw_thumb_paddle','tube_pickup','receiver','tool_return'])]
        if args.case=='transfer':
            stage_plan += [('source_open',3.20,['tool_handle','jaw_thumb_paddle','tube_pickup']),
                ('source_closed',3.70,['tool_handle','jaw_thumb_paddle','tube_pickup']),
                ('lifted',4.70,['tool_handle','tube_lift']),('receiver_approach',6.70,['tool_handle','receiver']),
                ('receiver_seated',7.70,['tool_handle','receiver']),('release_open',8.30,['tool_handle','jaw_thumb_paddle','receiver']),
                ('parked',14.70,['tool_return','jaw_thumb_paddle'])]
        for label,t,labels in stage_plan:
            measured=min(saved,key=lambda f:abs(f['time_s']-t))
            workspace_stages.append({'label':label,'sim_time_s':measured['time_s'],
                'trace_frame_index':measured['native_step'], 'target_labels':labels,
                'body_poses':[{'id':n,'prim_path':paths[n],'position_m':o['position_m'],
                    'quaternion_wxyz':o['quaternion_wxyz']} for n,o in measured['objects'].items()]})
        report['workspace_stages']=workspace_stages
        report['raw_trace']={'path':str(output/'native_frames.jsonl.gz'),'sha256':digest(output/'native_frames.jsonl.gz')}

        report.update(passed=all(checks.values()),checks=checks,native_steps=tick,native_frames=tick+1,
            maximum_penetration_m=max_pen,maximum_tube_lift_m=lift,bilateral_contact_duration_s=bilateral_steps*dt,
            peak_requested_force_norm_n=peak_force,final_state=final,final_objects=previous['objects'],
            fixed_scenery=guard.report() if guard else None,
            evidence_sha256={n:digest(output/n) for n in ('initial.usda','scene.json','task.json','native_frames.jsonl.gz','checkpoints.json')},
            scope='Object-only engineering forces on tool bodies, real passive tube contact/transport/release. No robot grasp or ordinary episode success.')
    except BaseException as exc:
        traceback.print_exc();report['error']={'type':type(exc).__name__,'message':str(exc)}
    finally:
        try:
            dump(output/'qualification.json',report)
            print(json.dumps({'passed':report['passed'],'output':str(output),'checks':report.get('checks'),'error':report.get('error')}),flush=True)
        finally:
            if app is not None:app.close()
    return 0 if report['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--case',choices=('canonical','transfer','open_transport'),required=True)
    raise SystemExit(run(p.parse_args()))


if __name__=='__main__':main()
