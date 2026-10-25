"""Frozen native piston/vessel interactions with explicitly reduced liquid.

Only the independent barrel receives a prescribed engineering pose boundary;
the dynamic plunger receives a bounded axial force. Vessels are passive native
bodies. There is no robot, grasp policy, model call or task-score claim.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np

from bench.paths import OUTPUT_ROOT

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def timeline(area):
    q20=20e-6/area;q5=5e-6/area
    home=[.83,.53,.92315];high=[.83,.53,1.15];source_high=[.62,.53,1.15]
    source=[.62,.53,.91315];a=[.56,2.08,1.00];b=[.88,2.08,1.00]
    return [
        ('initial_hold',.3,home,0.),('lift_tool',.7,high,0.),
        ('air_pull',.5,high,q5),('air_expel',.5,high,0.),
        ('source_approach',.5,source_high,0.),('immerse',.9,source,0.),
        ('aspirate',1.7,source,q20),('aspiration_stop',.3,source,q20),
        ('withdraw',.9,source_high,q20),('cross_benches',3.,[.56,2.08,1.15],q20),
        ('receiver_a_approach',.7,a,q20),('dispense_a',1.2,a,q20/2),('stop_a',.5,a,q20/2),
        ('receiver_b_approach',1.,b,q20/2),('dispense_b',1.2,b,0.),('stop_b',.5,b,0.),
        ('raise',.6,[.88,2.08,1.15],0.),('cross_return',3.,high,0.),
        ('return_cradle',.8,home,0.),('released_rest',.5,home,0.)]


def run(args):
    snapshot=verify_if_snapshot(PROJECT,required_modules=('scripts.maintenance.freeze_project_sources',),entrypoint_file=Path(__file__))
    if snapshot is None:raise ValueError('Native validation requires a complete frozen source snapshot')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu.startswith('GPU-') or ',' in gpu:raise ValueError('Select one GPU by UUID')
    output=args.output.resolve()
    if not str(output).startswith(OUTPUT_ROOT + '/'):raise ValueError('Shared datasets output required')
    output.mkdir(parents=True,exist_ok=False)
    report={'schema':'laboratory_pipette_interactions.v1','passed':False,'scope':'Object interactions, not hand/robot manipulation',
        'created_utc':datetime.now(timezone.utc).isoformat(),'source_snapshot':snapshot,'source_root':str(PROJECT),
        'gpu_uuid':gpu,'physics_dt_s':.0005,'model_calls':0,'robot_actions':0,
        'fluid_model':'Explicit reduced finite-volume piston component; no PBD or native water-weight force claim',
        'engineering_boundary':'Prescribed independent barrel pose/velocity except released rest; dynamic plunger axial PD force limited to 2 N',
        'support_scope':'Two rigid laboratory table-top proxies at original 0.90 m height; fixed room dressing is not loaded'}
    dump(output/'plan.json',report)
    app=None;code=1
    try:
        from isaacsim import SimulationApp
        allow=os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER')=='1'
        app=SimulationApp({'headless':True,'width':800,'height':600,'multi_gpu':False,'active_cuda_gpus':[0],
            'max_gpu_count':1,'physics_gpu':0,'renderer':'RayTracedLighting','fast_shutdown':True,
            'extra_args':['--/rtx/verifyDriverVersion/enabled=false'] if allow else []},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        from isaacsim.core.api import World
        from pxr import Gf,UsdGeom,UsdPhysics
        from scenes.workrooms.laboratory.tasks.pipette_transfer.assets import author_task,PISTON_AREA_M2
        from scenes.workrooms.laboratory.tasks.pipette_transfer.liquid import ReducedLiquidDynamics
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
        root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim())
        for name,center,size in [('Island',(.65,0.,.85),(2.40,1.25,.10)),('Rear',(.72,2.095,.85),(.64,.21,.10))]:
            cube=UsdGeom.Cube.Define(stage,'/World/'+name);cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center));cube.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        metadata=author_task(stage);dump(output/'asset_metadata.json',metadata)
        phases=timeline(PISTON_AREA_M2);report['timeline']=phases
        report['asset_metadata_sha256']=digest(output/'asset_metadata.json')
        world=World(stage_units_in_meters=1.,physics_dt=.0005,rendering_dt=1/30,backend='numpy',physics_prim_path='/World/PhysicsScene')
        context=world.get_physics_context();context.enable_gpu_dynamics(True);context.set_broadphase_type('GPU');context.set_solver_type('TGS')
        stage.Export(str(output/'initial.usda'));world.reset()
        sim=world.physics_sim_view
        views={name:sim.create_rigid_body_view(path) for name,path in metadata['actors'].items()}
        if any(v.count!=1 for v in views.values()):raise RuntimeError('Native body count mismatch')
        masses={name:float(np.asarray(v.get_masses()).reshape(-1)[0]) for name,v in views.items()}
        report['native_dry_masses_kg']=masses
        provider=ReducedLiquidDynamics({'asset_metadata':metadata})
        names=list(views);frames=[];water=[];commands=[];phase_indices=[];times=[];native_steps=[]
        initial_poses={name:np.asarray(view.get_transforms())[0].copy() for name,view in views.items()}
        def frame(baseline=False):
            objects={};poses=[]
            for name,view in views.items():
                pose=np.asarray(view.get_transforms())[0].copy();vel=np.asarray(view.get_velocities())[0].copy()
                objects[name]={'position_m':pose[:3].tolist(),'quaternion_wxyz':[float(pose[6]),*pose[3:6].tolist()],
                    'linear_velocity_m_s':vel[:3].tolist(),'angular_velocity_rad_s':vel[3:].tolist()}
                poses.append([*pose,*vel])
            result={'baseline':baseline,'objects':objects,'native_step':int(world.current_time_step_index),
                'native_time_s':float(world.current_time),'time_s':float(world.current_time),'physics_dt_s':.0005}
            provider.on_frame(result)
            return result,poses,provider.on_observe()
        _,_,initial=frame(True)
        start=np.asarray(phases[0][2]);qstart=0.;one=np.array([0],np.int32)
        max_force=0.;max_transverse=0.;max_inventory_residual=0.;phase_results={}
        for phase_index,(name,duration,target,qtarget) in enumerate(phases):
            target=np.asarray(target);count=round(duration/.0005);begin_state=provider.on_observe()
            for i in range(count):
                u=(i+1)/count;s=u*u*(3-2*u);ds=6*u*(1-u)/duration
                position=start+(target-start)*s;velocity=(target-start)*ds
                qcommand=qstart+(qtarget-qstart)*s
                barrel=np.asarray(views['pipette'].get_transforms())[0]
                plunger=np.asarray(views['plunger'].get_transforms())[0]
                vp=np.asarray(views['plunger'].get_velocities())[0]
                vb=np.asarray(views['pipette'].get_velocities())[0]
                # All commanded poses belong to the independent barrel only.
                if name!='released_rest':
                    views['pipette'].set_transforms(np.array([[*position,0.,0.,0.,1.]],np.float32),one)
                    views['pipette'].set_velocities(np.array([[*velocity,0.,0.,0.]],np.float32),one)
                q=float(plunger[2]-barrel[2]-.028)
                qd=float(vp[2]-vb[2])
                force=float(np.clip(1000.*(qcommand-q)-.7*qd+masses['plunger']*9.81,-2.,2.)) if name!='released_rest' else 0.
                if name!='released_rest':
                    views['plunger'].apply_forces_and_torques_at_position(np.array([[0.,0.,force]],np.float32),np.zeros((1,3),np.float32),None,one,True)
                world.step(render=False)
                observed,poses,state=frame()
                max_force=max(max_force,abs(force));max_inventory_residual=max(max_inventory_residual,abs(state['inventory_residual_ml']))
                dx=np.asarray(observed['objects']['plunger']['position_m'])-np.asarray(observed['objects']['pipette']['position_m'])
                max_transverse=max(max_transverse,float(np.linalg.norm(dx[:2])))
                frames.append(poses);times.append(world.current_time);native_steps.append(world.current_time_step_index)
                water.append([state['source_ml'],state['tool_ml'],state['receivers_ml']['receiver_a'],state['receivers_ml']['receiver_b'],
                    state['in_flight_ml'],state['cumulative_spill_ml'],state['piston_q_m'],state['outlet_flow_ml_s']])
                commands.append([*position,qcommand,force]);phase_indices.append(phase_index)
            phase_results[name]={'begin':begin_state,'end':state,'native_step':int(world.current_time_step_index)}
            print(json.dumps({'phase':name,'time_s':world.current_time,'source_ml':state['source_ml'],'tool_ml':state['tool_ml'],
                'receivers_ml':state['receivers_ml'],'spill_ml':state['cumulative_spill_ml'],'q_m':state['piston_q_m']}),flush=True)
            start=target;qstart=qtarget
        np.savez_compressed(output/'native_samples.npz',time_s=np.asarray(times),native_step=np.asarray(native_steps),
            body_state_xyzw_velocity=np.asarray(frames),water_state=np.asarray(water),commands=np.asarray(commands),phase_index=np.asarray(phase_indices))
        state=provider.on_observe();final_poses={name:np.asarray(view.get_transforms())[0].copy() for name,view in views.items()}
        receiver_drift={name:float(np.linalg.norm(final_poses[name][:3]-initial_poses[name][:3])) for name in ('source','receiver_a','receiver_b')}
        checks={'native_steps_complete':bool(np.all(np.diff(native_steps)==1)),
            'all_finite':bool(np.isfinite(frames).all() and np.isfinite(water).all()),
            'air_pull_no_water':phase_results['air_pull']['end']['tool_ml']<.01,
            'aspirate_near_20ml':18<=phase_results['aspiration_stop']['end']['tool_ml']<=22,
            'receiver_a_10_plusminus_2_ml':8<=state['receivers_ml']['receiver_a']<=12,
            'receiver_b_10_plusminus_2_ml':8<=state['receivers_ml']['receiver_b']<=12,
            'spill_within_2ml':state['cumulative_spill_ml']<=2,
            'flight_settled':state['in_flight_ml']<.01,
            'inventory_closed':state['inventory_closed'] and max_inventory_residual<1e-7,
            'vessels_stay_supported':max(receiver_drift.values())<.005,
            'native_plunger_transverse_constraint':max_transverse<.001,
            'bounded_plunger_force':max_force<=2.,
            'tool_return_near_cradle':float(np.linalg.norm(final_poses['pipette'][:3]-initial_poses['pipette'][:3]))<.012}
        report.update(status='completed',passed=all(checks.values()),checks=checks,final=state,phase_results=phase_results,
            native_steps=len(times),native_duration_s=float(world.current_time),max_transverse_error_m=max_transverse,
            max_inventory_residual_ml=max_inventory_residual,maximum_plunger_force_n=max_force,vessel_drift_m=receiver_drift,
            arrays={'path':str(output/'native_samples.npz'),'sha256':digest(output/'native_samples.npz'),
                'body_order':names,'body_components':['x','y','z','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz'],
                'water_components':['source_ml','tool_ml','receiver_a_ml','receiver_b_ml','in_flight_ml','spill_ml','q_m','outlet_flow_ml_s'],
                'command_components':['barrel_target_x','barrel_target_y','barrel_target_z','piston_target_q_m','piston_force_n']},
            initial_usd_sha256=digest(output/'initial.usda'),robot_manipulation_qualified=False)
        code=0 if report['passed'] else 1
    except Exception:
        report.update(status='exception',exception=traceback.format_exc());traceback.print_exc()
    finally:
        report['process_exit_code']=code;dump(output/'result.json',report)
        if app:
            app.close(exit_code=code)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    sys.exit(run(parser.parse_args()))
