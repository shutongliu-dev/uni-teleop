"""Independent native laboratory object/contact probes; no robot or hand policy.

Canonical and goal modes test separately declared gravity resets. Thread mode
applies bounded tool wrench only to the independent cap. Bottle remains passive;
cap Z is never prescribed and its measured axial lead must follow real contact.
"""
from __future__ import annotations
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import traceback
import numpy as np
from bench.paths import OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROJECT=Path(__file__).resolve().parents[2]


def dump(path,data): path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def array(x): return x.detach().cpu().numpy().copy() if hasattr(x,'detach') else np.asarray(x).copy()
def yaw(q):
    x,y,z,w=q
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def run(args):
    source=verify_if_snapshot(PROJECT, required_modules=('scripts.validation.laboratory_core_interactions',), entrypoint_file=Path(__file__))
    if source is None: raise ValueError('Run from a complete frozen source snapshot')
    output=args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT): raise ValueError('Evidence belongs in shared datasets')
    output.mkdir(parents=True,exist_ok=False)
    report={'schema':'laboratory_core_interactions.v1','passed':False,'task_id':args.task,
        'case':args.case,'source_snapshot':str(PROJECT),'source':source,'physics_dt_s':.0005,
        'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'model_calls':0,
        'robot_present':False,'hand_object_interaction_qualified':False,'task_success_claimed':False,
        'physics_force_calibration_claimed':False,
        'scope':'Independent object support/fit and bounded cap-tool contact only.'}
    app=world=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'width':640,'height':480,'multi_gpu':False,
            'active_cuda_gpus':[0],'max_gpu_count':1,'physics_gpu':0,'fast_shutdown':True,
            'extra_args':['--/rtx/verifyDriverVersion/enabled=false']},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        import torch
        omni.usd.get_context().new_stage()
        stage=omni.usd.get_context().get_stage()
        world=World(stage_units_in_meters=1.,physics_dt=.0005,rendering_dt=1/30,
            backend='torch',device='cuda:0',physics_prim_path='/World/PhysicsScene')
        pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('TGS')
        # Tables reproduce the two task support planes without room dressing.
        for label,y,sy in [('Island',0.,1.25),('Rear',2.09,.30)]:
            cube=UsdGeom.Cube.Define(stage,'/World/'+label)
            cube.CreateSizeAttr(1);cube.AddTranslateOp().Set(Gf.Vec3d(.65,y,.875))
            cube.AddScaleOp().Set(Gf.Vec3f(2.4,sy,.05));UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        module=importlib.import_module('scenes.workrooms.laboratory.tasks.'+args.task+'.assets')
        meta=module.author_task(stage,presentation=args.case in ('goal','insertion'))
        if args.case=='insertion':
            holes={h['object_id']:h for h in meta['task_parameters']['holes']}
            for entry in meta['objects']:
                hole=holes[entry['id']]
                position=[*hole['center_m'][:2],hole['bottom_z_m']+.075]
                op=UsdGeom.Xformable(stage.GetPrimAtPath(entry['body_path'])).GetOrderedXformOps()[0]
                op.Set(Gf.Vec3d(*position));entry['initial_position_m']=position
                entry['initial_pose_xyzw'][:3]=position
            report['insertion_protocol']={'kind':'gravity-only release',
                'initial_bottom_above_top_rim_m':.015,'initial_bottom_above_support_floor_m':.075,'pose_writes_after_reset':False,
                'external_forces_or_targets':False,'robot_present':False}
        if args.task=='cap_placement' and args.case!='canonical':
            holder=meta['task_parameters']['holder_center_m']
            for entry in meta['objects']:
                if entry['id']=='sample_bottle': position=[holder[0],holder[1],holder[2]+.0003]
                elif args.case=='goal': position=[.86,2.06,.9133]
                else: position=[holder[0],holder[1],holder[2]+.0003+module.CLOSED_CAP_Z]
                op=UsdGeom.Xformable(stage.GetPrimAtPath(entry['body_path'])).GetOrderedXformOps()[0]
                op.Set(Gf.Vec3d(*position));entry['initial_position_m']=position
                entry['initial_pose_xyzw'][:3]=position
        for name,path in meta['actors'].items():
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0)
        dump(output/'metadata.json',meta);stage.Export(str(output/'initial.usda'))
        world.reset()
        sim=world.physics_sim_view
        views={name:sim.create_rigid_body_view(path) for name,path in meta['actors'].items()}
        contacts={name:sim.create_rigid_contact_view(path,filter_patterns=[c for fixture in meta['fixture_metadata'].values() for c in fixture['collider_paths']]+
            [p for n,p in meta['actors'].items() if n!=name]+['/World/Island','/World/Rear'],
            max_contact_data_count=8192) for name,path in meta['actors'].items()}
        report['contact_filter_paths']={n:list(v.filter_paths) for n,v in contacts.items()}
        device=next(iter(views.values())).get_transforms().device
        indices=torch.tensor([0],dtype=torch.int32,device=device)
        dt=.0005;duration=15. if args.case=='thread' else 3. if args.case=='insertion' else 1.5
        report['duration_s']=duration
        if args.case=='thread':
            report['tool_boundary']={'body':'cap','kind':'bounded wrench',
              'torque_limit_nm':.006,'radial_force_limit_n':1.,'constant_downforce_n':.04,
              'yaw_rate_rad_s':.60,'cap_z_prescribed':False,
              'reentry':'Cap Z bounded-force hold at entrance until angle < 99 degrees; free Z inside thread','bottle_actuation':False,
              'protocol':'0.5 s settle; 0.5 s 0.4 N locked pull; 5.5 s turn to 110 degrees; 1 s lift; 1 s return to thread entrance; 5.5 s clockwise; 1 s release'}
        rows=[]
        for tick in range(round(duration/dt)):
            time=tick*dt;load=None
            if args.case=='thread' and .5<=time<14.:
                cap=array(views['cap'].get_transforms())[0];bottle=array(views['sample_bottle'].get_transforms())[0]
                velocity=array(views['cap'].get_velocities())[0]
                relative_angle=yaw(cap[3:])-yaw(bottle[3:])
                target_angle=module.THREAD_TURN if 1.<=time<8.5 else 0.
                target_rate=float(np.clip(4*(target_angle-relative_angle),-.60,.60))
                force=np.clip(80*(bottle[:3]-cap[:3])-.5*velocity[:3],-1.,1.);force[2]=-.04
                if time<1.: force[2]=.4
                if 6.5<=time<8.5 or (8.5<=time<14. and relative_angle>math.radians(99)):
                    # Hold the independent cap at the groove entrance until
                    # its lugs reenter; then release axial tool control.
                    relative_target=module.CLOSED_CAP_Z+(.05 if time<7.5 else .0053)
                    force[2]=float(np.clip(80*(bottle[2]+relative_target-cap[2])-.8*velocity[2]+.025*9.81,-1.,1.))
                # An independent tool centers the cap and maintains upright
                # orientation, but neither prescribes nor drives axial travel.
                q=cap[3:];tilt=np.asarray([2*q[0]*q[3],2*q[1]*q[3]])
                torque=np.array([*np.clip(-.03*tilt-.002*velocity[3:5],-.004,.004),
                                 np.clip(.012*(target_rate-velocity[5]),-.006,.006)])
                views['cap'].apply_forces_and_torques_at_position(
                    force_data=torch.tensor([force],dtype=torch.float32,device=device),
                    torque_data=torch.tensor([torque],dtype=torch.float32,device=device),
                    position_data=None,indices=indices,is_global=True)
                load={'force_n':force.tolist(),'torque_nm':torque.tolist()}
            world.step(render=False)
            if tick%10==0 or tick==round(duration/dt)-1:
                row={'time_s':(tick+1)*dt,'native_step':tick+1,'objects':{},'load':load}
                for name,view in views.items():
                    row['objects'][name]={'pose_xyzw':array(view.get_transforms())[0].tolist(),
                        'velocity':array(view.get_velocities())[0].tolist(),
                        'contact_forces_n':array(contacts[name].get_contact_force_matrix(dt=dt))[0].tolist()}
                rows.append(row)
        with (output/'samples.jsonl').open('w') as f:
            for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
        final=rows[-1]['objects'];report['final']=final
        checks={}
        for name,entry in final.items():
            p=entry['pose_xyzw'];v=entry['velocity']
            forces=np.asarray(entry['contact_forces_n'])
            checks[name+'_finite_supported']={'passed':bool(np.isfinite(p+v).all() and p[2]>.899 and np.sum(np.maximum(forces[:,2],0.))>.01),
                'position_m':p[:3],'speed_m_s':float(np.linalg.norm(v[:3])),
                'angular_speed_rad_s':float(np.linalg.norm(v[3:])),
                'contact_force_norm_n':float(np.linalg.norm(forces)),
                'upward_contact_force_n':float(np.sum(np.maximum(forces[:,2],0.)))}
            checks[name+'_released_rest']={'passed':bool(np.linalg.norm(v[:3])<.02 and np.linalg.norm(v[3:])<.15)}
        if args.task=='vial_racking' and args.case in ('goal','insertion'):
            checks['true_hole_insertion']={'passed':all(abs(final[h['object_id']]['pose_xyzw'][2]-h['bottom_z_m'])<.003
                and np.linalg.norm(np.array(final[h['object_id']]['pose_xyzw'][:2])-h['center_m'][:2])<.003
                for h in meta['task_parameters']['holes'])}
        if args.case=='insertion':
            for hole in meta['task_parameters']['holes']:
                name=hole['object_id']
                initial=next(o for o in meta['objects'] if o['id']==name)['initial_position_m']
                crosses=[r for r in rows if r['objects'][name]['pose_xyzw'][2]<hole['rim_z_m']-.005]
                descent=initial[2]-final[name]['pose_xyzw'][2]
                checks[name+'_gravity_crossed_rim_and_seated']={
                    'passed':bool(initial[2]>hole['rim_z_m']+.014 and crosses and descent>.070),
                    'initial_bottom_z_m':initial[2],'top_rim_z_m':hole['rim_z_m'],
                    'first_below_rim_time_s':crosses[0]['time_s'] if crosses else None,
                    'measured_descent_m':descent,
                    'minimum_bottom_z_m':min(r['objects'][name]['pose_xyzw'][2] for r in rows)}
            checks['no_actuation_after_reset']={'passed':all(r['load'] is None for r in rows)}
        if args.case=='thread':
            baseline=next(r for r in rows if r['time_s']>=.49)
            opened=next(r for r in rows if r['time_s']>=4.0)
            def relative(row):
                c=row['objects']['cap']['pose_xyzw'];b=row['objects']['sample_bottle']['pose_xyzw']
                return yaw(c[3:])-yaw(b[3:]),c[2]-b[2]
            ba,bz=relative(baseline);oa,oz=relative(opened)
            expected=(oa-ba)*module.PITCH/(2*math.pi)
            checks['real_thread_axial_lead']={'passed':bool(oa-ba>.8 and oz-bz>.002 and abs((oz-bz)-expected)<.0015),
                 'relative_turn_rad':oa-ba,'measured_rise_m':oz-bz,'expected_rise_m':expected}
            pulled=next(r for r in rows if r['time_s']>=.99)
            _,pz=relative(pulled)
            checks['closed_cap_resists_straight_pull']={'passed':pz-bz<.002,'rise_under_0_4n_pull_m':pz-bz}
            lifted=next(r for r in rows if r['time_s']>=7.49)
            la,lz=relative(lifted)
            checks['cap_fully_removed']={'passed':bool(la>1.80 and lz-bz>.035),'relative_angle_rad':la,'rise_m':lz-bz}
            fa,fz=relative(rows[-1]);checks['reclosed_height']={'passed':abs(fz-bz)<.002 and abs(fa-ba)<.12,'relative_final_angle_rad':fa,'relative_final_z_m':fz,'baseline_z_m':bz}
        report['checks']=checks;report['passed']=all(c['passed'] for c in checks.values())
    except BaseException as e:
        traceback.print_exc();report['error']={'type':type(e).__name__,'message':str(e)}
    finally:
        dump(output/'qualification.json',report)
        print(json.dumps({'passed':report['passed'],'output':str(output)}),flush=True)
        if app is not None:
            if not report['passed']:
                try: raise RuntimeError('Independent laboratory probe did not pass')
                finally: app.close()
            app.close()
    return 0 if report['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=['sample_sorting','vial_racking','cap_placement'])
    p.add_argument('--case',required=True,choices=['canonical','goal','thread','insertion'])
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.case=='thread' and args.task!='cap_placement':p.error('Thread probe only applies to cap closure')
    if args.case=='insertion' and args.task!='vial_racking':p.error('Insertion probe only applies to vial racking')
    raise SystemExit(run(args))

if __name__=='__main__': main()
