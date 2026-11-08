"""Sealed native rigid-cell sponge freeflight and independently calibrated support.

All member q/qd, native mass/COM/inertia and contact forces are archived. Only a
world-space external force on the actual green pad COM is applied in freeflight.
No member state or joint target is written after reset. Diagnostic, not robot
manipulation qualification. Original static task geometry stays present.
"""
from __future__ import annotations
import argparse,json,os,traceback
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from bench.isaacsim.release import checked,digest,dump
from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation import elastic_sponge_metrics as metrics
from scripts.validation import elastic_sponge_compression as compression
from scripts.validation import elastic_sponge_wiping as wiping_probe
from scripts.validation.elastic_sponge_clearance import certify_stage
from scenes.workrooms.kitchen.wiping import ContactWiping
from scenes.workrooms.kitchen.rules import thresholds
from scripts.validation.sponge_freeflight_metrics import force_for_step
from scripts.validation.sponge_support_loadcell import (author_instrument,native,json_value,contact_row,
    static_collision_bounds,bbox_gap,native_offsets,instrument_properties,PLATE,WEIGHT,SCENE,CONFIG)
PROJECT=Path(__file__).resolve().parents[2]
MODULES=('scripts.validation.elastic_sponge_probe','scripts.validation.elastic_sponge_metrics','scripts.validation.sponge_support_loadcell','scripts.validation.elastic_sponge_compression','scripts.validation.elastic_sponge_wiping','scenes.workrooms.kitchen.wiping','scripts.validation.elastic_sponge_clearance')
TABLE='/World/FreeflightTableBoundary'


def load_asset(path):
    manifest=load_json(path);usd=path.parent/'task.usda';file=path.parent/'metadata.json'
    checked(usd,manifest['usd_sha256']);checked(file,manifest['metadata_sha256']);meta=load_json(file)
    if meta.get('sponge_physics_profile')!='elastic_cell_lattice_v1':raise ValueError('Explicit lattice candidate required')
    if len(meta['actors'])!=28 or len(meta['elastic_composites']['sponge']['links'])!=63:raise ValueError('Expected28 bodies and63 elastic links')
    return manifest,usd,meta


def author(stage,case,manifest,usd,meta,placement='source'):
    from pxr import Gf,Sdf,UsdGeom,UsdPhysics,UsdShade
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    free=case=='freeflight';rest=case=='rest';wipe=case=='wiping';elastic=case in ('freeflight','elastic','compression','rest','wiping')
    if not free and not rest and not wipe:author_instrument(stage,'rigid' if case=='rigid' else 'empty',manifest,usd,meta)
    if elastic:
        stage.DefinePrim(manifest['root_prim_path'],'Xform').GetReferences().AddReference(str(usd),manifest['root_prim_path'])
        origin=np.asarray(next(x for x in meta['objects'] if x['id']=='sponge')['initial_pose_xyzw'][:3])
        shift=np.array([0.,0.,.3]) if free else (np.zeros(3) if rest else np.array([.6,-.8,.9003])-origin)
        if wipe:shift=np.r_[wiping_probe.center(meta)+[-.08,-.019],.9003]-origin
        for member,path in meta['actors'].items():
            prim=stage.GetPrimAtPath(path);attr=prim.GetAttribute('xformOp:translate')
            attr.Set(Gf.Vec3d(*(np.asarray(attr.Get(),float)+shift)))
            if rest and placement=='goal':
                pose=meta['goals']['body_poses_xyzw'][member];attr.Set(Gf.Vec3d(*pose[:3]))
                prim.GetAttribute('xformOp:orient').Set(Gf.Quatf(pose[6],Gf.Vec3f(*pose[3:6])))
            prim.AddAppliedSchema('PhysxContactReportAPI');prim.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float,custom=False).Set(0.)
    if free or rest or wipe:
        table=UsdGeom.Cube.Define(stage,TABLE);table.CreateSizeAttr(1.)
        table.AddTranslateOp().Set(Gf.Vec3d(.45,-.95,.8775));table.AddScaleOp().Set(Gf.Vec3f(2.8,.8,.045))
        UsdPhysics.CollisionAPI.Apply(table.GetPrim());table.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
        for key,value in [('contactOffset',.002),('restOffset',0.)]:table.GetPrim().CreateAttribute('physxCollision:'+key,Sdf.ValueTypeNames.Float,custom=False).Set(value)
    if wipe:
        material=UsdShade.Material.Define(stage,'/World/WipingInstrumentContact');api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(.4);api.CreateDynamicFrictionAttr(.4);api.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(TABLE)).Bind(material,materialPurpose='physics')
    physics=UsdPhysics.Scene.Define(stage,SCENE);physics.CreateGravityDirectionAttr(Gf.Vec3f(0.,0.,-1.));physics.CreateGravityMagnitudeAttr(9.81)
    return {'placement':placement if rest else None,'member_translation_m':shift.tolist() if elastic and not (rest and placement=='goal') else None,'static_task_geometry_modified':False,'post_reset_state_or_target_writes':0}


def mass_receipt(view,specs):
    mass=native(view.get_masses()).reshape(-1).astype(float);com=native(view.get_coms()).astype(float)
    inertia=native(view.get_inertias()).reshape(-1,3,3).astype(float)
    errors=[]
    for i,row in enumerate(specs):
        rot=Rotation.from_quat(row['principal_axes_xyzw']).as_matrix();expected=rot@np.diag(row['diagonal_inertia_kg_m2'])@rot.T
        for name,actual,target,rtol,atol in [('mass',mass[i],row['mass_kg'],2e-6,1e-10),
            ('com',com[i,:3],row['center_of_mass_m'],1e-5,2e-8),('inertia',inertia[i],expected,2e-4,1e-10)]:
            if not np.allclose(actual,target,rtol=rtol,atol=atol):errors.append({'body':row['id'],'property':name,'expected':np.asarray(target).tolist()})
    return mass,com[:,:3],{'paths':list(view.prim_paths),'masses_kg':mass.tolist(),'com_poses_local':com.tolist(),'inertias_kg_m2':inertia.tolist(),'native_offsets':native_offsets(view),'property_mismatches':errors}



def reference(path,case,identity):
    if path is None:return None
    value=load_json(path)
    if value.get('identity')!=identity or value.get('case')!=case or value.get('passed') is not True:raise ValueError('Passing same source/asset/solver calibration required')
    for name,sha in value['evidence_sha256'].items():checked(path.parent/name,sha)
    return value


def run(args):
    source=verify_if_snapshot(PROJECT,required_modules=MODULES,entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve()!=PROJECT:raise ValueError('Run only from complete frozen source')
    sf=PROJECT/'source_snapshot_manifest.json';inventory=load_json(sf)['files_sha256']
    if sf.stat().st_mode&0o222 or any((PROJECT/p).stat().st_mode&0o222 for p in inventory):raise ValueError('Seal source before native launch')
    manifest,usd,meta=load_asset(args.asset_manifest)
    identity={'source_manifest_sha256':digest(sf),'asset_manifest_sha256':digest(args.asset_manifest),'solver':args.solver,'dt_s':.0005,'placement':args.placement if args.case=='rest' else None}
    empty=reference(args.empty_report,'empty',identity);rigid=reference(args.rigid_report,'rigid',identity)
    if args.case in ('rigid','elastic') and empty is None:raise ValueError('Empty calibration required')
    if args.case=='elastic' and rigid is None:raise ValueError('Known rigid calibration required')
    output=args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):raise ValueError('Evidence belongs in shared datasets')
    output.mkdir(parents=True,exist_ok=False)
    report={'schema':'elastic_sponge_native_probe.v1','case':args.case,'passed':False,'source':source,'identity':identity,'asset_manifest':str(args.asset_manifest.resolve()),
        'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'model_calls':0,'render_calls':0,'post_reset_state_writes':0,'post_reset_joint_target_writes':0,
        'scope':'Physical instrument diagnostics only; native mass per rigid member, no FEM pseudo-state, no robot success claim.',
        'calibrations':{k:({'path':str(p),'sha256':digest(p)} if p else None) for k,p in [('empty',args.empty_report),('rigid',args.rigid_report)]}}
    keys=('time_s','native_step','body_q','body_qd','body_contact_force_n','plate_q','plate_qd','force_world_n','force_position_world_m','platen_q','platen_qd')
    raw={k:[] for k in keys};rows=[];chunks=[];saved=0;app=None;code=1
    free=args.case=='freeflight';rest=args.case=='rest';compress=args.case=='compression';wipe=args.case=='wiping';elastic=args.case in ('freeflight','elastic','compression','rest','wiping');steps=400 if free else (3200 if rest else (6000 if compress else (wiping_probe.STEPS if wipe else 1200)))
    def flush():
        nonlocal saved
        if len(rows)==saved:return
        d=output/'native_chunks';d.mkdir(exist_ok=True);f=d/f'frames_{len(chunks):06d}.npz'
        np.savez_compressed(f,**{k:np.asarray(v[saved:]) for k,v in raw.items()});chunks.append({'path':str(f),'first_frame':saved,'last_frame':len(rows)-1,'sha256':digest(f)});saved=len(rows)
        dump(output/'native_chunks_manifest.json',{'frames':saved,'chunks':chunks,'trajectory_complete':saved==steps+1})
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'multi_gpu':False,'active_cuda_gpus':[0],'max_gpu_count':1,'physics_gpu':0,'fast_shutdown':True,
            'extra_args':['--/rtx/verifyDriverVersion/enabled=false'] if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER')=='1' else []},experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd,torch
        from isaacsim.core.api import World
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage();report['pre_reset_overrides']=author(stage,args.case,manifest,usd,meta,args.placement)
        if wipe:report['wiping_instrument']={'plan_native_steps':wiping_probe.PLAN,'bounded_force_n_per_axis':.4,'position_gain_n_m':30.,'velocity_gain_ns_m':1.4,'vertical_force_n':0.,'torque_nm':[0.,0.,0.],'force_point':'Actual preceding pad COM','support_material':{'static_friction':.4,'dynamic_friction':.4,'restitution':0.},'scope':'Original engineering bounded pad force law and soil credit; not robot success.'}
        if compress:report['platen_instrument']=compression.author_platen(stage,SCENE)
        world=World(stage_units_in_meters=1.,physics_dt=.0005,rendering_dt=1/30,backend='torch',device='cuda:0',physics_prim_path=SCENE)
        context=world.get_physics_context();context.enable_gpu_dynamics(True);context.set_broadphase_type('GPU');context.set_solver_type(args.solver)
        paths=list(meta['actors'].values()) if elastic else ([WEIGHT] if args.case=='rigid' else [])
        obstacles=static_collision_bounds(stage,paths+([] if free or rest or wipe else [PLATE])+([compression.PLATEN] if compress else []));report['static_collision_bounds']=obstacles
        scene=stage.GetPrimAtPath(SCENE);report['scene_properties']={n:json_value(scene.GetAttribute(n).Get()) for n in ('physxScene:solverType','physxScene:enableGPUDynamics','physics:gravityMagnitude','physics:gravityDirection')}
        spring=instrument_properties(stage) if not free and not rest and not wipe else None;report['spring_properties']=spring
        stage.Export(str(output/'initial.usda'));world.reset();sim=world.physics_sim_view
        if int(sim.device_ordinal)<0:raise RuntimeError('GPU physics required')
        report['native_backend']={'device':str(sim.device),'ordinal':int(sim.device_ordinal)}
        bodies=sim.create_rigid_body_view(manifest['root_prim_path']+'/Actors/*' if elastic else WEIGHT) if paths else None
        specs=[];mass=np.empty(0);com=np.empty((0,3));padindex=None
        if bodies is not None:
            if set(bodies.prim_paths)!=set(paths):raise ValueError('Native bulk member inventory mismatch')
            if elastic:
                bypath={x['body_path']:x for x in meta['objects']};specs=[bypath[p] for p in bodies.prim_paths];padindex=list(bodies.prim_paths).index(meta['actors']['sponge'])
            else:
                size=np.array([.12,.078,.045]);corners=np.array([[x,y,z] for x in (-.06,.06) for y in (-.039,.039) for z in (-.0225,.0225)])
                specs=[{'id':'known_rigid','body_path':WEIGHT,'mass_kg':.0792,'center_of_mass_m':[0.,0.,0.],'principal_axes_xyzw':[0.,0.,0.,1.],'diagonal_inertia_kg_m2':(.0792/12*(np.sum(size**2)-size**2)).tolist(),'collision_support_points_m':corners.tolist()}]
            mass,com,report['native_members']=mass_receipt(bodies,specs)
            if report['native_members']['property_mismatches']:raise ValueError('Native mass properties differ; exact readback retained')
            np.testing.assert_allclose(mass.sum(),.0792,rtol=2e-6,atol=1e-9)
        plate=sim.create_rigid_body_view(PLATE) if not free and not rest and not wipe else None
        if plate is not None:np.testing.assert_allclose(native(plate.get_masses()),.2,rtol=2e-6)
        contact=sim.create_rigid_contact_view(manifest['root_prim_path']+'/Actors/*' if elastic else WEIGHT,max_contact_data_count=16384) if bodies else None
        platecontact=sim.create_rigid_contact_view(PLATE,max_contact_data_count=16384) if plate else None
        platen=sim.create_rigid_body_view(compression.PLATEN) if compress else None
        platencontact=sim.create_rigid_contact_view(compression.PLATEN,max_contact_data_count=16384) if compress else None
        padcontact=sim.create_rigid_contact_view(meta['actors']['sponge'],max_contact_data_count=16384) if rest or wipe else None
        if platen:
            report['native_platen']={'mass_kg':float(native(platen.get_masses()).reshape(-1)[0]),'com':native(platen.get_coms()).tolist(),'inertia':native(platen.get_inertias()).tolist()}
            np.testing.assert_allclose(report['native_platen']['mass_kg'],compression.MASS,rtol=2e-6)
        forceview=sim.create_rigid_body_view(meta['actors']['sponge']) if free or wipe else None
        one=torch.tensor([0],device='cuda:0',dtype=torch.int32);tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0')
        locals=[np.asarray(s['collision_support_points_m'],float) for s in specs]
        wiper=ContactWiping(meta,thresholds('surface_wiping'),.0005) if wipe else None
        if compress:
            central=next(i for i,row in enumerate(specs) if row.get('grid_index')==[1,1,2])
            top_mask=np.isclose(locals[central][:,2]+specs[central]['composite_rest_center_m'][2],.045,atol=1e-8)
            if top_mask.sum()<3:raise ValueError('Central top-face readback required')
        links=meta['elastic_composites']['sponge']['links'] if elastic else [];ids={s['id']:i for i,s in enumerate(specs)}
        dump(output/'native_initialization.json',report);initial_time=float(world.current_time)
        def sample(index,force,position):
            q=native(bodies.get_transforms()) if bodies else np.empty((0,7));v=native(bodies.get_velocities()) if bodies else np.empty((0,6))
            cf=native(contact.get_net_contact_forces(.0005)).reshape(-1,3) if contact else np.empty((0,3))
            pq=native(plate.get_transforms())[0] if plate else np.empty(0);pv=native(plate.get_velocities())[0] if plate else np.empty(0)
            pq2=native(platen.get_transforms())[0] if platen else np.empty(0);pv2=native(platen.get_velocities())[0] if platen else np.empty(0)
            values=(float(world.current_time)-initial_time,int(world.current_time_step_index),q,v,cf,pq,pv,force,position,pq2,pv2)
            for key,value in zip(keys,values):raw[key].append(value)
            row={'relative_step':index,'time_s':values[0],'native_step':values[1]};rows.append(row)
            cloud=[Rotation.from_quat(pose[3:]).apply(vertices)+pose[:3] for pose,vertices in zip(q,locals)]
            if cloud:
                full=np.concatenate(cloud);lo,hi=full.min(0),full.max(0);row['body_bounds_m']=[lo.tolist(),hi.tolist()]
                row['minimum_static_gap_m']=min((bbox_gap(lo,hi,np.asarray(x['bounds_m'])) for x in obstacles),default=None)
                if wipe:row['minimum_non_support_gap_m']=min((bbox_gap(lo,hi,np.asarray(x['bounds_m'])) for x in obstacles if x['path']!=TABLE),default=None)
                if plate:
                    local=(full-pq[:3])@Rotation.from_quat(pq[3:]).as_matrix();row['lowest_above_plate_top_m']=float(local[:,2].min()-.01);row['plate_xy_clearance_m']=float((np.array([.1,.075])-abs(local[:,:2])).min())
            if padcontact:row['pad_contact']=contact_row(padcontact,.0005,index==0)
            if wipe:
                actual=row['pad_contact'];loaded=TABLE in actual['other_actor_paths'] and actual['net_contact_force_n'][2]>0.
                row['coverage']=wiper.update({'position_m':q[padindex,:3].tolist(),'quaternion_wxyz':[float(q[padindex,6]),*q[padindex,3:6].tolist()]},loaded_contact=loaded,baseline=index==0)
                row['phase'],desired,speed=wiping_probe.target(index,meta);row['desired_xy_m']=desired.tolist();row['desired_velocity_xy_m_s']=speed.tolist()
            if platencontact:
                row['platen_contact']=contact_row(platencontact,.0005,index==0)
                row['central_top_in_pad_frame_m']=float(np.median(((cloud[central][top_mask]-q[padindex,:3])@Rotation.from_quat(q[padindex,3:]).as_matrix())[:,2]))
                row['platen_bottom_minus_highest_sponge_m']=float(pq2[2]-.005-hi[2])
                row['platen_weight_balance_force_n']=compression.MASS*9.81 if index else 0.
                row['additional_platen_force_n']=compression.force_for_step(index)[1] if index else 0.
            if platecontact:row['plate_contact']=contact_row(platecontact,.0005,index==0)
            if links:
                rotations=Rotation.from_quat(q[:,3:].astype(float));centers=rotations.apply(com)+q[:,:3]
                ratio=[];angles=[]
                for link in links:
                    i,j=ids[link['body0']],ids[link['body1']];rest=np.linalg.norm(np.array(specs[i]['composite_rest_center_m'])-specs[j]['composite_rest_center_m'])
                    ratio.append(float(np.linalg.norm(centers[i]-centers[j])/rest));angles.append(float(np.linalg.norm((rotations[i].inv()*rotations[j]).as_rotvec())))
                row['minimum_neighbor_center_distance_ratio']=min(ratio);row['maximum_neighbor_center_distance_ratio']=max(ratio);row['maximum_relative_neighbor_rotation_rad']=max(angles)
            if len(rows)-saved>=100:flush()
            return row
        with (output/'native_frames.jsonl').open('x') as stream:
            stream.write(json.dumps(sample(0,np.zeros(3),np.zeros(3)),allow_nan=False)+'\n')
            for index in range(1,steps+1):
                force=force_for_step(index).astype(np.float32) if free else np.zeros(3,np.float32);position=np.zeros(3,np.float32)
                if free:
                    pose=native(forceview.get_transforms())[0];position=(Rotation.from_quat(pose[3:]).apply(com[padindex])+pose[:3]).astype(np.float32)
                    if force.any() and forceview.apply_forces_and_torques_at_position(tensor(force[None]),tensor(np.zeros((1,3))),tensor(position[None]),one,True) is False:raise RuntimeError('Native force API rejected wrench')
                if wipe:
                    pose=native(forceview.get_transforms())[0];velocity=native(forceview.get_velocities())[0]
                    force,phase,desired,speed=wiping_probe.force(index,meta,pose,velocity);position=(Rotation.from_quat(pose[3:]).apply(com[padindex])+pose[:3]).astype(np.float32)
                    if force.any() and forceview.apply_forces_and_torques_at_position(tensor(force[None]),tensor(np.zeros((1,3))),tensor(position[None]),one,True) is False:raise RuntimeError('Wiping native pad force API rejected')
                if compress:
                    force,extra=compression.force_for_step(index);position=native(platen.get_transforms())[0,:3].copy()
                    if platen.apply_forces_and_torques_at_position(tensor(force[None]),tensor(np.zeros((1,3))),tensor(position[None]),one,True) is False:raise RuntimeError('Platen force API rejected')
                world.step(render=False);stream.write(json.dumps(sample(index,force,position),allow_nan=False)+'\n')
        data={k:np.asarray(v) for k,v in raw.items()}
        if free:checks,measurement,derived=metrics.freeflight(data,mass,com,padindex)
        elif wipe:checks,measurement,derived=wiping_probe.evaluate(data,rows,mass,padindex,TABLE)
        elif compress:checks,measurement,derived=compression.evaluate(data,rows,mass)
        elif rest:
            checks,measurement=compression.quiet(data,2400);force=np.sum(data['body_contact_force_n'].astype(float),axis=1);mean=force[2400:].mean(0);expected=float(mass.sum()*9.81)
            checks.update(contact_weight_within_15_percent=abs(mean[2]-expected)<=.15*expected,source_support_contact=all(r['pad_contact']['point_count']>0 for r in rows[2400:]),native_steps_consecutive=np.all(np.diff(data['native_step'])==1),dt_matches=np.allclose(np.diff(data['time_s']),float(np.float32(.0005)),atol=1e-10,rtol=0))
            measurement.update(mean_actual_support_force_n=mean.tolist(),expected_weight_n=expected,complete_steps=3200,quiet_window_s=.4,placement=args.placement);derived={'support_force_n':force}
        else:checks,measurement,derived=metrics.support(data,mass,com,empty)
        checks.update(gravity_matches=np.isclose(report['scene_properties']['physics:gravityMagnitude'],9.81,atol=1e-5) and np.allclose(report['scene_properties']['physics:gravityDirection'],[0.,0.,-1.]),all_raw_finite=all(np.isfinite(v).all() for v in data.values()),solver_matches=report['scene_properties']['physxScene:solverType']==args.solver,
            asset_unchanged=digest(usd)==manifest['usd_sha256'] and digest(args.asset_manifest.parent/'metadata.json')==manifest['metadata_sha256'])
        if bodies and not rest and not wipe:
            checks['no_other_static_contact']=all(r['minimum_static_gap_m'] is None or r['minimum_static_gap_m']>(.01 if free else .005) for r in rows)
        if free:checks['all_member_contact_forces_zero']=bool(np.max(abs(data['body_contact_force_n']))<=1e-8)
        elif bodies and not rest and not wipe:
            tail=rows[-801:];checks.update(load_not_through_plate=min(r['lowest_above_plate_top_m'] for r in rows)>=-.001,
                load_supported=all(abs(r['lowest_above_plate_top_m'])<=.001 and r['plate_xy_clearance_m']>=.005 for r in tail),
                actual_plate_contact=sum(r['plate_contact']['point_count']>0 for r in tail)>=.95*len(tail))
        if links:checks['no_gross_lattice_distortion']=all(r['minimum_neighbor_center_distance_ratio']>=.5 and r['maximum_neighbor_center_distance_ratio']<=1.5 and r['maximum_relative_neighbor_rotation_rad']<=.5 for r in rows)
        if wipe:
            proof,certificates=certify_stage(stage,specs,data['body_q'],obstacles,excluded_paths=(TABLE,),threshold=.005)
            report['actual_convex_clearance']=proof
            np.savez_compressed(output/'clearance_certificates.npz',**certificates)
            checks['no_other_static_contact']=proof['passed']
        if not free and not rest and not wipe:checks['spring_unchanged']=instrument_properties(stage)==spring
        np.savez_compressed(output/'derived.npz',**derived);report.update(checks={k:bool(v) for k,v in checks.items()},measurements=measurement,passed=bool(all(checks.values())));code=0 if report['passed'] else 1
    except BaseException:report['exception']=traceback.format_exc();traceback.print_exc()
    finally:
        try:
            flush();np.savez_compressed(output/'native_states.npz',**{k:np.asarray(v) for k,v in raw.items()})
            with (output/'native_frames.jsonl').open('w') as stream:
                for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
            report['raw_frames']=len(rows);report['evidence_sha256']={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='qualification.json'}
        except BaseException:report['archive_exception']=traceback.format_exc();code=1;report['passed']=False
        finally:
            report['process_exit_code']=code;dump(output/'qualification.json',report)
            if app is not None:app.close(exit_code=code)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('freeflight','empty','rigid','elastic','compression','rest','wiping'),required=True)
    parser.add_argument('--placement',choices=('source','goal'),default='source')
    parser.add_argument('--solver',choices=('TGS','PGS'),required=True)
    parser.add_argument('--asset-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--empty-report',type=Path);parser.add_argument('--rigid-report',type=Path)
    raise SystemExit(run(parser.parse_args()))
