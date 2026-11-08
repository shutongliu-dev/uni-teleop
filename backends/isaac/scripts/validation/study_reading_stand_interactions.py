"""Native stand mechanisms with declared bounded engineering wrenches.

No robot is present and no task-policy success is claimed. The raw stream stores
all native rigid states and signed contacts; every control is explicit and stops
before the released terminal window.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
from bench.isaacsim.release import digest, dump
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT=Path(__file__).resolve().parents[2]
DT=.0005


def array(x):
    return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()


def smooth(a,b,t):
    t=float(np.clip(t,0,1));return np.asarray(a)+(np.asarray(b)-np.asarray(a))*(3*t*t-2*t*t*t)


def run(args):
    source=verify_if_snapshot(PROJECT,required_modules=('scripts.validation.study_reading_stand_interactions',),entrypoint_file=Path(__file__))
    if source is None:raise ValueError('Run from frozen full source')
    from scripts.validation.board_game_runtime_smoke import gpu_selection
    gpu=gpu_selection(os.environ.get('CUDA_VISIBLE_DEVICES',''))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'schema':'study_reading_stand_native_probe.v1','passed':False,'case':args.case,'source_snapshot':source,
        'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_selection':gpu,'physics_dt_s':DT,'robot_present':False,
        'physics_accepted':False,'robot_manipulation_qualified':False,'task_policy_success_claimed':False,
        'model_calls':0,'post_baseline_pose_or_velocity_writes':False}
    app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'width':640,'height':480,'multi_gpu':False,'active_cuda_gpus':[0],
            'max_gpu_count':1,'physics_gpu':0,'fast_shutdown':True,
            'extra_args':['--/rtx/verifyDriverVersion/enabled=false']},experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        import torch
        from isaacsim.core.api import World
        from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema
        from scipy.spatial.transform import Rotation
        from scenes.workrooms.study.task_scene import candidate_inputs
        from scenes.workrooms.study.scene import environment_reference
        from scenes.workrooms.study.reading_stand_rules import configuration,panel_pose,on_panel
        from scenes.workrooms.study.reading_stand_evaluation import SequenceEvaluator
        from scenes.workrooms.study.native_observer import raw_contacts
        candidate,manifest,meta,deps=candidate_inputs('reading_stand');config=configuration()
        report['asset_manifest_sha256']=candidate['manifest_sha256'];report['asset_root']=str(Path(candidate['initial_usd']).parent)
        report['asset_metadata_sha256']=digest(Path(candidate['initial_usd']).parent/'metadata.json')
        ref=environment_reference();report['fixed_room']=ref
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        world=World(stage_units_in_meters=1.,physics_dt=DT,rendering_dt=1/30,backend='torch',device='cuda:0',physics_prim_path='/World/PhysicsScene')
        pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('TGS')
        stage.DefinePrim('/World/Room','Xform').GetReferences().AddReference(ref['usd_path'],ref['source_prim'])
        stage.DefinePrim('/World/StudyTask','Xform').GetReferences().AddReference(candidate['initial_usd'],'/World/StudyTask')
        actors=manifest['bodies'];initial={n:deepcopy(o['initial_pose']) for n,o in config['objects'].items()}
        if args.case in ('load','unlocked','withdraw','unseated'):
            initial['panel']=panel_pose(55.4)
            if args.case!='unseated':
                initial['book']={'position_m':on_panel([.10,0.,.0145],55.4),'quaternion_wxyz':panel_pose(55.4)['quaternion_wxyz']}
            if args.case!='unlocked':initial['support_pin']['position_m'][1]=0.
        for n,pose in initial.items():
            xf=UsdGeom.Xformable(stage.GetPrimAtPath(actors[n]));q=pose['quaternion_wxyz']
            for op in xf.GetOrderedXformOps():
                if op.GetOpType()==UsdGeom.XformOp.TypeTranslate:op.Set(Gf.Vec3d(*pose['position_m']))
                elif op.GetOpType()==UsdGeom.XformOp.TypeOrient:op.Set(Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))
        fixed=[str(p.GetPath()) for root in ('/World/Room','/World/StudyTask/Fixtures') for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.CollisionAPI)]
        for path in actors.values():PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        stage.Export(str(out/'initial.usda'));dump(out/'configuration.json',config)
        world.reset();sim=world.physics_sim_view
        views={n:sim.create_rigid_body_view(p) for n,p in actors.items()}
        device=views['panel'].get_transforms().device;indices=torch.tensor([0],dtype=torch.int32,device=device)
        for n,v in views.items():
            p=initial[n];q=p['quaternion_wxyz']
            v.set_transforms(torch.tensor([[*p['position_m'],*q[1:],q[0]]],device=device,dtype=torch.float32),indices)
            v.set_velocities(torch.zeros((1,6),device=device),indices)
        filters=sorted(fixed+list(actors.values()))
        contact_view=sim.create_rigid_contact_view(list(actors.values()),filter_patterns=[filters for _ in actors],max_contact_data_count=8192)
        sensors=list(contact_view.sensor_paths);mapped=[list(r) for r in contact_view.filter_paths]
        report['native_sensor_paths']=sensors;report['native_filter_paths']=mapped
        def actor(path):
            for n,p in actors.items():
                if path==p or path.startswith(p+'/'):return n
            if path=='/World/Room/Geometry/worktop/Collider':return 'table'
            if path=='/World/Room/Geometry/room_floor/Collider':return 'floor'
            return 'external/'+path.strip('/').replace('/','_')
        evaluator=SequenceEvaluator(config,trusted_normalized_input=True)
        sequential=args.case=='sequential_cycle'
        cycle=args.case in ('cycle','sequential_cycle')
        duration={'reset':2.,'load':2.,'unlocked':2.,'withdraw':3.4,'cycle':9.,'sequential_cycle':13.,'unseated':2.}[args.case]
        report['duration_s']=duration;report['reset_fixture']=initial
        report['stage_times_s']=({'folded':.35,'deployed_unlocked':2.35,'pin_inserted_released_panel':3.9,'loaded_all_released':8.8} if cycle else {'initial':0.,'released_terminal':duration-.1})
        if sequential:report['stage_times_s']={'folded':.35,'deployed_unlocked':3.35,'pin_inserted_released_panel':7.9,'loaded_all_released':12.8}
        report['reset_write_scope']='Declared USD reset fixtures, then exact same reset states restored after World.reset warmup and before baseline; no later pose/velocity writes.'
        report['native_masses_kg']={n:array(v.get_masses())[0].item() for n,v in views.items()}
        report['engineering_boundary']={'case':args.case,'panel_torque_limit_nm':2.5,'pin_force_limit_n':6.,
            'book_force_per_axis_limit_n':6.,'book_torque_per_axis_limit_nm':.2,
            'force_end_s':11.5 if sequential else 7.5 if cycle else 2.2 if args.case=='withdraw' else 0.,
            'scope':'Explicit object-only mechanism test; no robot or legal task-action credit.'}
        report['engineering_boundary']['sequential_single_object_control']=sequential
        report['engineering_boundary']['withdraw_friction_bias_n']=2. if args.case=='withdraw' else 0.
        previous=None;metrics=[];peak_force=0.;peak_torque=0.;penetration=0.;forces_count=0;last=time.monotonic()
        book_initial=np.array(initial['book']['position_m']);goal=np.array(on_panel([.115,0.,.016]))
        book_target_rotation=Rotation.from_quat([0.,-math.sin(math.radians(55)/2),0.,math.cos(math.radians(55)/2)])
        with gzip.open(out/'native_frames.jsonl.gz','wt',compresslevel=1) as stream:
            for step in range(round(duration/DT)+1):
                t=step*DT;wrenches={}
                if step:
                    control_t=(step-1)*DT
                    def apply(name,force=None,torque=None):
                        f=np.zeros(3) if force is None else np.asarray(force,float)
                        q=np.zeros(3) if torque is None else np.asarray(torque,float)
                        views[name].apply_forces_and_torques_at_position(force_data=torch.tensor([f.tolist()],dtype=torch.float32,device=device),
                            torque_data=torch.tensor([q.tolist()],dtype=torch.float32,device=device),position_data=None,indices=indices,is_global=True)
                        wrenches[name]={'force_n':f.tolist(),'torque_nm':q.tolist()}
                    if cycle:
                        if .4<=control_t<3.4:
                            state=previous['objects']['panel'];r=Rotation.from_quat(state['quaternion_wxyz'][1:]+state['quaternion_wxyz'][:1])
                            theta=math.atan2(r.as_matrix()[2,0],r.as_matrix()[0,0]);target=float(smooth(0.,math.radians(78. if sequential else 62.),(control_t-.4)/2.))
                            torque=-12.*(target-theta)-.8*state['angular_velocity_rad_s'][1]-.68*9.81*.151*math.cos(theta)
                            apply('panel',torque=[0.,float(np.clip(torque,-2.5,2.5)),0.])
                        pin_start=3.4 if sequential else 2.4
                        if pin_start<=control_t<pin_start+1.:
                            s=previous['objects']['support_pin'];target=float(smooth(.28,0.,control_t-pin_start))
                            f=np.clip(70*(target-s['position_m'][1])-3*s['linear_velocity_m_s'][1],-6.,6.)
                            apply('support_pin',force=[0.,f,0.])
                        book_t=control_t-(4. if sequential else 0.)
                        if 4.<=book_t<7.5:
                            s=previous['objects']['book']
                            if book_t<4.8:target=smooth(book_initial,[book_initial[0],book_initial[1],1.20],(book_t-4.)/.8)
                            elif book_t<5.8:target=smooth([book_initial[0],book_initial[1],1.20],[goal[0],0.,1.20],book_t-4.8)
                            else:target=smooth([goal[0],0.,1.20],goal,(book_t-5.8)/1.7)
                            force=np.clip(160*(target-np.array(s['position_m']))-12*np.array(s['linear_velocity_m_s'])+[0.,0.,.22*9.81],-6.,6.)
                            alpha=float(smooth(0.,1.,(book_t-4.8)/1.));rt=Rotation.from_rotvec(book_target_rotation.as_rotvec()*alpha)
                            actual=Rotation.from_quat(s['quaternion_wxyz'][1:]+s['quaternion_wxyz'][:1])
                            torque=np.clip(2.*(rt*actual.inv()).as_rotvec()-.07*np.array(s['angular_velocity_rad_s']),-.2,.2)
                            apply('book',force=force,torque=torque)
                    elif args.case=='withdraw' and .8<=control_t<2.2:
                        s=previous['objects']['support_pin'];target=float(smooth(0.,.28,control_t-.8))
                        apply('support_pin',force=[0.,np.clip(70*(target-s['position_m'][1])-3*s['linear_velocity_m_s'][1]+2.,-6.,6.),0.])
                    if sequential and len(wrenches)>1:raise AssertionError('Sequential probe may control only one object per frame')
                    for w in wrenches.values():
                        peak_force=max(peak_force,float(np.linalg.norm(w['force_n'])));peak_torque=max(peak_torque,float(np.linalg.norm(w['torque_nm'])))
                    forces_count+=bool(wrenches)
                    world.step(render=False)
                objects={};raw=[]
                for n,v in views.items():
                    p=array(v.get_transforms())[0];vel=array(v.get_velocities())[0]
                    objects[n]={'position_m':p[:3].tolist(),'quaternion_wxyz':[float(p[6]),*p[3:6].tolist()],
                        'linear_velocity_m_s':vel[:3].tolist(),'angular_velocity_rad_s':vel[3:].tolist()}
                if step:
                    buffers=contact_view.get_contact_data(dt=1.)
                    counts,starts=array(buffers[4]),array(buffers[5]);active=counts>0
                    used=int(np.max((starts+counts)[active],initial=0))
                    if used>=8192:raise ValueError('Native contact capacity reached')
                    copied=[array(x[:used+1]) for x in buffers[:4]]+[counts,starts]
                    raw=raw_contacts(copied,sensors,mapped)
                normal=[{**r,'a':actor(r['a']),'b':actor(r['b'])} for r in raw]
                states={n:evaluator._state(s) for n,s in objects.items()}
                parsed=evaluator._contacts({'contacts':normal})
                measurement=evaluator.physical_conditions(states,parsed,None if previous is None else {n:evaluator._state(s) for n,s in previous['objects'].items()})
                penetration=max(penetration,max((-r['separation_m'] for r in raw),default=0.))
                frame={'step':step,'time_s':t,'objects':objects,'raw_contacts':raw,'contacts':normal,
                    'engineering_wrenches':wrenches,'conditions':measurement}
                stream.write(json.dumps(frame,separators=(',',':'),allow_nan=False)+'\n')
                metrics.append({'time_s':t,**measurement,'external_wrench':bool(wrenches)})
                previous=frame
                if time.monotonic()-last>25:
                    print(json.dumps({'native_time_s':t,'case':args.case}),flush=True);last=time.monotonic()
        tail=[r for r in metrics if r['time_s']>=duration-.4-1.e-9]
        checks={'complete_frames':len(metrics)==round(duration/DT)+1,'no_deep_penetration':penetration<.002,
                'terminal_all_engineering_forces_removed':not any(r['external_wrench'] for r in tail)}
        if args.case in ('cycle','sequential_cycle','load'):
            checks['complete_400ms_locked_loaded_released_stability']=all(r['conditions_passed'] for r in tail)
        elif args.case=='reset':checks['initial_never_solved']=not any(r['conditions_passed'] for r in metrics)
        elif args.case=='unseated':
            checks['panel_locked_without_book']=all(r['objects']['panel']['geometry'] and r['objects']['panel']['supported'] for r in tail)
            checks['unseated_book_never_accepted']=not any(r['conditions_passed'] for r in metrics)
            checks['book_remains_on_table']=abs(previous['objects']['book']['position_m'][2]-.788)<.003
        else:
            checks['unlocked_panel_collapses']=metrics[-1]['panel_angle_deg']<20.
            checks['unlocked_never_stably_accepted']=not any(r['conditions_passed'] for r in tail)
        if cycle:
            checks['actual_opening']=max(r['panel_angle_deg'] for r in metrics)>(75. if sequential else 60.)
            if sequential:checks['single_object_control_at_every_frame']=True
            checks['actual_pin_travel']=max(r['pin_y_m'] for r in metrics)-min(r['pin_y_m'] for r in metrics)>.25
        if args.case=='withdraw':
            checks['loaded_before_pin_withdrawal']=all(r['conditions_passed'] for r in metrics if .4<=r['time_s']<.8)
            checks['actual_pin_fully_withdrawn']=metrics[-1]['pin_y_m']>=.273
        report.update(passed=all(checks.values()),checks=checks,native_steps=round(duration/DT),native_frames=len(metrics),
            terminal_frames=len(tail),maximum_penetration_m=penetration,peak_force_norm_n=peak_force,peak_torque_norm_nm=peak_torque,
            external_force_steps=forces_count,final_conditions=metrics[-1],
            evidence_sha256={n:digest(out/n) for n in ('native_frames.jsonl.gz','initial.usda','configuration.json')})
    except BaseException as error:
        traceback.print_exc();report['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        try:
            dump(out/'qualification.json',report);print(json.dumps({'passed':report['passed'],'output':str(out),'checks':report.get('checks'),'error':report.get('error')}),flush=True)
        finally:
            if app is not None:app.close()
    return 0 if report['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True,choices=('reset','load','unlocked','withdraw','cycle','sequential_cycle','unseated'));p.add_argument('--output',required=True,type=Path)
    raise SystemExit(run(p.parse_args()))


if __name__=='__main__':main()
