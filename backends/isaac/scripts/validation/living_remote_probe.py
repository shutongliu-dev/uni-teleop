"""Native remote mechanism evidence using separate dynamic contact probes.

No robot policy, button pose writes, spring target changes or scored task.
Only reset poses differ between the three explicitly labelled experiments.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from bench.paths import load_json, OUTPUT_ROOT

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.maintenance.freeze_project_sources import verify_if_snapshot

PROBE = dict(radius_m=.004, mass_kg=.020, stiffness_n_m=1200., damping_ns_m=10., force_cap_n=3.)
PRESS_M, RELEASE_M = -.0014, -.0003
TABLE_PATH = '/World/Geometry/worktop/Collider'
TIMELINE = [
    ('settle', .50, .012, .012),
    ('hover_approach', .25, .0007, .0007), ('hover_hold', .30, .0007, .0007),
    ('hover_retract', .25, .012, .012),
    ('weak_approach', .25, -.0006, -.0006), ('weak_hold', .30, -.0006, -.0006),
    ('weak_retract', .25, .012, .012), ('weak_released', .35, .012, .012),
    ('power_approach', .30, -.003, .012), ('power_hold', .35, -.003, .012),
    ('power_retract', .30, .012, .012), ('power_released', .35, .012, .012),
    ('confirm_approach', .30, .012, -.003), ('confirm_hold', .35, .012, -.003),
    ('confirm_retract', .30, .012, .012), ('confirm_released', .35, .012, .012),
    ('repeat_approach', .30, -.003, -.003), ('repeat_hold', .35, -.003, -.003),
    ('repeat_retract', .30, .012, .012), ('repeat_released', .50, .012, .012),
]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def rotation(xyzw):
    q=np.asarray(xyzw,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<.99:
        raise ValueError('Invalid native quaternion')
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def schedule(time_s):
    start, previous = 0., np.array([.012, .012])
    for name, duration, power, confirm in TIMELINE:
        target = np.array([power, confirm])
        if time_s < start+duration-1e-10:
            t = np.clip((time_s-start)/duration, 0., 1.)
            blend = t*t*(3-2*t)
            return name, previous+(target-previous)*blend
        start += duration
        previous = target
    return TIMELINE[-1][0], previous


def bounded_force(position, velocity, target):
    force = PROBE['stiffness_n_m']*(np.asarray(target)-position)-PROBE['damping_ns_m']*velocity
    force[2] += PROBE['mass_kg']*9.81
    return force*min(1., PROBE['force_cap_n']/max(float(np.linalg.norm(force)), 1e-12))


def joint_state(body, button, anchor):
    p0, p1 = np.array(body['pose_xyzw'][:3]), np.array(button['pose_xyzw'][:3])
    r0, r1 = rotation(body['pose_xyzw'][3:]), rotation(button['pose_xyzw'][3:])
    d = r0.T @ (p1-p0)-anchor
    v0, v1 = np.array(body['velocity']), np.array(button['velocity'])
    qdot = (r0.T @ (v1[:3]-v0[:3]-np.cross(v0[3:], p1-p0)))[2]
    angle = np.arccos(np.clip((np.trace(r0.T@r1)-1)/2, -1., 1.))
    return {'q_m': float(d[2]), 'qdot_m_s': float(qdot),
            'transverse_error_m': float(np.linalg.norm(d[:2])), 'angle_error_rad': float(angle)}


def pair_info(contacts, first, second):
    for row in contacts:
        if row['sensor'] == first and row['filter'] == second:
            return row
    return {'count': 0, 'normal_impulse_l1_ns': 0., 'normal_impulse_vector_ns': [0., 0., 0.], 'min_separation_m': None}


def button_events(rows, paths):
    """One contact-backed press per release cycle, at native sample times."""
    events={name:[] for name in ('power','confirm')}
    for name in events:
        state='unarmed'
        for row in rows:
            q=row['joints'][name]['q_m']
            contact=pair_info(row['contacts'],paths['probe_'+name],paths[name])
            kind=None
            if state in ('unarmed','pressed') and q>=RELEASE_M and contact['count']==0:
                kind='armed' if state=='unarmed' else 'released';state='armed'
            elif state=='armed' and q<=PRESS_M and contact['normal_impulse_l1_ns']>1e-9:
                kind='pressed';state='pressed'
            if kind:
                events[name].append({'event':kind,'tick':row['tick'],'native_time_s':row['time_s'],
                    'phase':row['phase'],'q_m':q,'contact_count':contact['count'],
                    'normal_impulse_l1_ns':contact['normal_impulse_l1_ns']})
    return events


def assess(rows, case, dt, paths, *, table_path=TABLE_PATH):
    if not rows:
        return {'passed': False, 'reason': 'missing native samples'}
    tail = rows[-round(.3/dt):]
    checks = {
        'complete_steps': all(b['tick']==a['tick']+1 and abs(b['time_s']-a['time_s']-dt)<1e-7 for a,b in zip(rows, rows[1:])),
        'joint_limits': all(-.00215 <= j['q_m'] <= .00015 for r in rows for j in r['joints'].values()),
        'joint_transverse_constraints': all(j['transverse_error_m']<.00015 and j['angle_error_rad']<.01 for r in rows for j in r['joints'].values()),
        'contact_penetration': all(c['min_separation_m']>=-.0005 for r in rows for c in r['contacts']),
        'tail_rest': all(np.linalg.norm(b['velocity'][:3])<.002 and np.linalg.norm(b['velocity'][3:])<.05
                         for r in tail for n,b in r['bodies'].items() if not n.startswith('probe_')),
    }
    details = {}
    if case == 'press':
        events=button_events(rows,paths);details['button_events']=events
        for name in ('power', 'confirm'):
            checks[name+'_two_distinct_press_release_cycles']=(
                sum(e['event']=='pressed' for e in events[name])==2 and
                sum(e['event']=='released' for e in events[name])==2)
            for phase in (name+'_hold', 'repeat_hold'):
                selected = [r for r in rows if r['phase']==phase]
                credited = [r for r in selected if r['joints'][name]['q_m']<=PRESS_M and
                    pair_info(r['contacts'], paths['probe_'+name], paths[name])['normal_impulse_l1_ns']>1e-9]
                longest=current=0;previous_tick=None
                for r in credited:
                    current=current+1 if previous_tick is not None and r['tick']==previous_tick+1 else 1
                    longest=max(longest,current);previous_tick=r['tick']
                checks[name+'_'+phase+'_contact_press'] = bool(selected) and longest*dt >= .05
                details[name+'_'+phase] = {'contact_pressed_duration_s':len(credited)*dt,
                    'longest_continuous_contact_pressed_s':longest*dt,
                    'min_q_m':min((r['joints'][name]['q_m'] for r in selected), default=None)}
            for phase in ('weak_released', name+'_released', 'repeat_released'):
                selected = [r for r in rows if r['phase']==phase][-round(.3/dt):]
                checks[name+'_'+phase] = len(selected)*dt >= .29 and all(
                    r['joints'][name]['q_m']>=RELEASE_M and
                    pair_info(r['contacts'], paths['probe_'+name], paths[name])['count']==0 for r in selected)
            for phase in ('hover_hold', 'weak_hold'):
                selected = [r for r in rows if r['phase']==phase]
                checks[name+'_'+phase+'_no_trigger'] = bool(selected) and all(r['joints'][name]['q_m']>PRESS_M for r in selected)
                touching = any(pair_info(r['contacts'], paths['probe_'+name], paths[name])['normal_impulse_l1_ns']>1e-9 for r in selected)
                checks[name+'_'+phase+'_expected_contact'] = touching if phase=='weak_hold' else not touching
            other = 'confirm' if name=='power' else 'power'
            selected = [r for r in rows if r['phase'].startswith(other+'_')]
            checks[name+'_no_other_button_trigger'] = bool(selected) and all(r['joints'][name]['q_m']>PRESS_M for r in selected)
    else:
        checks['buttons_remain_released' if case == 'reset' else 'buttons_released_after_settling'] = all(
            j['q_m']>=RELEASE_M for r in (rows if case == 'reset' else tail) for j in r['joints'].values())
        if case == 'holder':
            checks['drop_does_not_trigger_buttons'] = all(
                j['q_m']>PRESS_M for r in rows for j in r['joints'].values())
        if case == 'reset':
            checks['buttons_clear_table'] = all(r['button_min_z_m'][n]-.78>.0005 and
                pair_info(r['contacts'], paths[n], table_path)['count']==0 for r in rows for n in ('power','confirm'))
            support_pairs = [('body', table_path)]
        else:
            support_pairs = [('body', paths['holder']), ('holder', table_path)]
            checks['body_clear_table'] = all(pair_info(r['contacts'], paths['body'], table_path)['count']==0 for r in rows)
        for name, support in support_pairs:
            contact_rows = [pair_info(r['contacts'], paths[name], support) for r in tail]
            witnessed=[r for r in rows if pair_info(r['contacts'],paths[name],support)['normal_impulse_l1_ns']>1e-9]
            quiet_witnessed=[r for r in witnessed if np.linalg.norm(r['bodies'][name]['velocity'][:3])<.002 and
                             np.linalg.norm(r['bodies'][name]['velocity'][3:])<.05]
            last=witnessed[-1] if witnessed else None
            continuation=[]
            for r in tail:
                c=pair_info(r['contacts'],paths[name],support)
                measured=c['count']>0
                body=r['bodies'][name]
                sleeping=(last is not None and body['is_sleeping'] and r['tick']>=last['tick'] and
                    np.linalg.norm(np.array(body['pose_xyzw'][:3])-last['bodies'][name]['pose_xyzw'][:3])<2e-5 and
                    np.linalg.norm(rotation(body['pose_xyzw'][3:])-rotation(last['bodies'][name]['pose_xyzw'][3:]))<.001 and
                    not r['probe_commands'] and not any(c['sensor']==paths[name] and c['filter']!=support for c in r['contacts']))
                if support==paths.get('holder'):
                    sleeping=sleeping and r['bodies']['holder']['is_sleeping']
                continuation.append(measured or sleeping)
            checks[name+'_support_contact_or_confirmed_sleep_continuation']=(len(quiet_witnessed)*dt>=.20 and all(continuation))
            observed_tail=witnessed[-round(.25/dt):]
            reactions=[pair_info(r['contacts'],paths[name],support)['normal_impulse_vector_ns'] for r in observed_tail]
            details[name+'_support'] = {'tail_contact_duration_s':sum(c['count']>0 for c in contact_rows)*dt,
                'low_speed_observed_support_duration_s':len(quiet_witnessed)*dt,
                'last_observed_support_tick':None if last is None else last['tick'],
                'native_sleep_tail_duration_s':sum(r['bodies'][name]['is_sleeping'] for r in tail)*dt,
                'mean_last_observed_normal_impulse_over_dt_n':None if not reactions else (np.mean(reactions,axis=0)/dt).tolist(),
                'scope':'Measured contact before native sleep plus unchanged unforced sleeping pose; no reaction force is imputed to missing contacts'}
    details['joint_extrema'] = {name:{'min_q_m':min(r['joints'][name]['q_m'] for r in rows),
        'max_q_m':max(r['joints'][name]['q_m'] for r in rows),'final_q_m':rows[-1]['joints'][name]['q_m']} for name in ('power','confirm')}
    return {'passed':all(checks.values()), 'checks':checks, 'details':details,
            'threshold_scope':'Explicit engineering thresholds; not a calibrated task scorer'}


def run(args):
    snapshot = verify_if_snapshot(PROJECT, required_modules=('scripts.maintenance.freeze_project_sources',),
                                  entrypoint_file=Path(__file__))
    if snapshot is None:
        raise ValueError('A complete frozen project snapshot is required')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not gpu.startswith('GPU-') or ',' in gpu:
        raise ValueError('Select one physical GPU by UUID')
    if args.dt not in (.0005,1/960,1/1920):
        raise ValueError('Use declared 0.5 ms, 1/960 or 1/1920 s timestep')
    output=args.output.resolve()
    if not str(output).startswith(OUTPUT_ROOT + '/'):
        raise ValueError('Evidence must be under shared datasets')
    output.mkdir(parents=True,exist_ok=False)
    seconds=sum(x[1] for x in TIMELINE) if args.case=='press' else 2.
    report={'schema':'living_remote_native_probe.v1','case':args.case,'source_root':str(PROJECT),
        'source_snapshot':snapshot,'gpu_uuid':gpu,'dt_s':args.dt,'requested_duration_s':seconds,
        'created_utc':datetime.now(timezone.utc).isoformat(),'status':'initializing','passed':False,
        'native_manipulation_qualified':False,'model_calls':0,'robot_actions':0,
        'scope':'Independent dynamic probe mechanism/reset/support evidence, not FR3/SHARPA task execution',
        'probe_parameters':PROBE if args.case=='press' else None,
        'press_timeline':TIMELINE if args.case=='press' else None}
    dump(output/'plan.json',report)
    app=None;code=1;rows=[]
    try:
        from isaacsim import SimulationApp
        allow=os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER')=='1'
        report['isaac_allow_unvalidated_driver']=allow
        app=SimulationApp({'headless':True,'width':640,'height':480,'multi_gpu':False,
            'active_cuda_gpus':[0],'max_gpu_count':1,'physics_gpu':0,'renderer':'RayTracedLighting',
            'fast_shutdown':True,'extra_args':['--/rtx/verifyDriverVersion/enabled=false'] if allow else []},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import carb
        import omni.usd
        import omni.physx
        from omni.physxfabric import get_physx_fabric_interface
        from isaacsim.core.api import World
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
        from scenes.workrooms.living_room.scene import environment_reference, ENVIRONMENT_ROOT
        from scenes.workrooms.living_room.scene_candidates import scene_candidate
        from bench.simulation.isaacsim_backend.workroom import verify_reference
        candidate=scene_candidate('remote_operation')
        selected_scene = selected_task = None
        if args.release is not None:
            from bench.isaacsim.release import task_inputs
            _, selected_scene, selected_task, _ = task_inputs(args.release, 'remote_operation')
            bundle = selected_scene['native_task_bundle']
            if (bundle['usd_path'] != candidate['initial_usd']
                    or bundle['usd_sha256'] != candidate['initial_usd_sha256']
                    or selected_scene['physics']['dt_s'] != args.dt
                    or selected_scene['isaacsim'].get('solver_type', 'TGS') != 'TGS'):
                raise ValueError('Probe must match the selected release asset, timestep and solver')
            report['selected_release'] = {'path': str(args.release.resolve()),
                'manifest_sha256': digest(args.release/'release_manifest.json'),
                'physics': selected_scene['physics'], 'isaacsim': selected_scene['isaacsim'],
                'scope': 'Object-only experiment; frozen object asset, room, support materials and timestep. Robot omitted.'}
            dump(output/'selected_task.json', selected_task)
        for path_key,hash_key in [('manifest_path','manifest_sha256'),('initial_usd','initial_usd_sha256')]:
            if digest(candidate[path_key])!=candidate[hash_key]:raise RuntimeError('Pinned candidate hash mismatch')
        metadata_path=Path(candidate['initial_usd']).with_name('metadata.json')
        manifest=load_json(Path(candidate['manifest_path']))
        if digest(metadata_path)!=manifest['metadata_sha256']:raise RuntimeError('Pinned metadata hash mismatch')
        metadata=load_json(metadata_path)
        reference=environment_reference();verify_reference(reference)
        if selected_scene is not None and (reference != selected_scene['workroom'] or metadata != selected_task['asset_metadata']):
            raise ValueError('Selected release metadata/environment differs from probe assets')
        env_manifest=load_json(ENVIRONMENT_ROOT/'manifest.json')
        report.update(asset_candidate=candidate,asset_metadata=metadata,parent_environment=reference,
            runtime_version=(Path(os.environ['ISAAC_PATH'])/'VERSION').read_text().strip())
        report['source_snapshot_runtime']=verify_if_snapshot(PROJECT,
            required_modules=('scenes.workrooms.living_room.scene_candidates',),entrypoint_file=Path(__file__))
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim())
        table_path = TABLE_PATH
        if selected_scene is None:
            root.GetPrim().GetReferences().AddReference(env_manifest['scene'],'/World')
        else:
            from bench.simulation.isaacsim_backend.workroom import add_workroom
            room = add_workroom(stage, reference, surface_materials=selected_scene['isaacsim']['workroom_surface_materials'])
            report['workroom'] = room
            table_path = room['supports']['table']
        root.GetPrim().GetReferences().AddReference(candidate['initial_usd'],'/World')
        UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
        report['table_path'] = table_path
        if not stage.GetPrimAtPath(table_path).HasAPI(UsdPhysics.CollisionAPI):raise RuntimeError('Missing original worktop collider')
        paths={'body':metadata['remote']['body_path'],'holder':metadata['holder']['body_path'],
               **metadata['remote']['button_paths']}
        if args.case!='reset':
            rig=UsdGeom.XformCommonAPI(stage.GetPrimAtPath(metadata['remote']['assembly_path']))
            xyz=(.49,-.12,.7905) if args.case=='press' else (.82,.075,.8034)
            rig.SetTranslate(Gf.Vec3d(*xyz));rig.SetRotate(Gf.Vec3f(0.,0.,-12. if args.case=='press' else 0.))
            report['reset_override']={'remote_translation_m':xyz,'remote_rotation_xyz_deg':[0.,0.,-12. if args.case=='press' else 0.],
                'scope':'Before native initialization only; holder release starts 6 mm above its liner, no pose writes after reset'}
        joint_info={}
        for name,definition in metadata['buttons'].items():
            joint=UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(definition['joint_path']))
            drive=UsdPhysics.DriveAPI(joint.GetPrim(),'linear')
            joint_info[name]={'anchor':list(joint.GetLocalPos0Attr().Get()),'joint_path':definition['joint_path'],
                'target_position':drive.GetTargetPositionAttr().Get(),'target_velocity':drive.GetTargetVelocityAttr().Get(),
                'stiffness_n_m':drive.GetStiffnessAttr().Get(),'damping_ns_m':drive.GetDampingAttr().Get(),
                'limits_m':[joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()]}
            if joint_info[name]['target_position']!=0 or joint_info[name]['target_velocity']!=0:raise RuntimeError('Nonzero button drive target')
        report['joint_contract']=joint_info
        cache=UsdGeom.XformCache();probe_targets={};button_points={}
        for name in ('power','confirm'):
            transform=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(paths[name]))
            button_points[name]=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(paths[name]+'/Cap')).GetPointsAttr().Get(),dtype=float)
            if args.case=='press':
                center=np.asarray(transform.ExtractTranslation());center[2]+=.002+PROBE['radius_m']
                probe_targets[name]=center
                path='/World/IndependentProbes/'+name;paths['probe_'+name]=path
                sphere=UsdGeom.Sphere.Define(stage,path);sphere.CreateRadiusAttr(PROBE['radius_m'])
                sphere.AddTranslateOp().Set(Gf.Vec3d(*(center+[0.,0.,.012])))
                UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim());UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
                UsdPhysics.MassAPI.Apply(sphere.GetPrim()).CreateMassAttr(PROBE['mass_kg'])
                collision=PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
                collision.CreateContactOffsetAttr(.0001);collision.CreateRestOffsetAttr(0.)
        for path in paths.values():
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        world=World(stage_units_in_meters=1.,physics_dt=args.dt,rendering_dt=1/30,backend='numpy',physics_prim_path='/World/PhysicsScene')
        context=world.get_physics_context();context.enable_gpu_dynamics(True);context.set_broadphase_type('GPU');context.set_solver_type('TGS')
        if selected_scene is not None:
            physics=PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(context.prim_path))
            physics.CreateEnableCCDAttr(False)
            physics.CreateGpuMaxRigidContactCountAttr(2**20)
            physics.CreateGpuMaxRigidPatchCountAttr(2**18)
            physics.CreateGpuFoundLostAggregatePairsCapacityAttr(2**14)
            physics.CreateGpuTotalAggregatePairsCapacityAttr(2**18)
        settings=carb.settings.get_settings()
        def sync():
            omni.physx.get_physx_interface().update_transformations(False,True,True,False)
            fabric=get_physx_fabric_interface();fabric.force_update(args.dt,world.current_time);fabric.save_to_usd()
        for key,value in {'updateToUsd':True,'updateVelocitiesToUsd':True,'suppressReadback':False,'outputVelocitiesLocalSpace':False}.items():settings.set_bool('/physics/'+key,value)
        stage.Export(str(output/'initial.usda'));world.reset()
        sim=world.physics_sim_view
        views={name:sim.create_rigid_body_view(path) for name,path in paths.items()}
        if any(v.count!=1 for v in views.values()):raise RuntimeError('Native body inventory mismatch')
        if any(not isinstance(v.get_transforms(),np.ndarray) for v in views.values()):
            raise RuntimeError('Probe wrench arrays require the declared numpy native backend')
        sensors=list(paths.values());filters=sensors+[table_path]
        contact=sim.create_rigid_contact_view(sensors,filter_patterns=[filters]*len(sensors),max_contact_data_count=16384)
        sensor_paths=list(contact.sensor_paths);filter_paths=list(contact.filter_paths)
        if filter_paths and isinstance(filter_paths[0],str):filter_paths=[filter_paths]*len(sensor_paths)
        if (len(sensor_paths)!=contact.sensor_count or len(filter_paths)!=contact.sensor_count or
            set(sensor_paths)!=set(sensors) or any(len(row)!=contact.filter_count or set(row)!=set(filters) for row in filter_paths)):
            raise RuntimeError('Contact mapping mismatch')
        report['contact_mapping']={'sensors':sensor_paths,'filters':filter_paths,'capacity':contact.max_contact_data_count,
            'components':'normal impulses only; dt=1 scalar and signed normal preserved, including zero impulse contacts',
            'array_packing':'native_indices identify original buffers; unused slots are excluded, original counts/starts retained'}
        report['native_masses_kg']={n:float(array(v.get_masses()).reshape(-1)[0]) for n,v in views.items()}
        report['paths']=paths
        stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        if stage_id<=0 or stage_id!=omni.usd.get_context().get_stage_id():
            raise RuntimeError('Native sleep query stage ID differs from the live USD context')
        encoded_paths={name:PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path)) for name,path in paths.items()}
        physics_interface=omni.physx.get_physx_simulation_interface()
        report['sleep_query']={'method':'IPhysxSimulation.is_sleeping(stage_id, encoded_body_path)','stage_id':stage_id,
                              'encoded_body_paths':encoded_paths}
        start=float(world.current_time);previous=start;start_step=int(world.current_time_step_index)
        baseline_bodies={n:{'pose_xyzw':array(v.get_transforms())[0].tolist(),
            'velocity':array(v.get_velocities())[0].tolist()} for n,v in views.items()}
        dump(output/'baseline.json', {'tick':0, 'time_s':start, 'bodies':baseline_bodies,
            'joints':{n:joint_state(baseline_bodies['body'],baseline_bodies[n],np.array(joint_info[n]['anchor']))
                for n in ('power','confirm')}, 'contacts':[]})
        with (output/'samples.jsonl').open('x') as stream,(output/'raw_contacts.jsonl').open('x') as raw_stream:
            for tick in range(1,round(seconds/args.dt)+1):
                phase,offsets=schedule((tick-1)*args.dt) if args.case=='press' else (args.case,np.zeros(2))
                commands={}
                for index,name in enumerate(('power','confirm')):
                    if args.case!='press':break
                    view=views['probe_'+name];pose=array(view.get_transforms())[0];velocity=array(view.get_velocities())[0]
                    target=probe_targets[name]+[0.,0.,offsets[index]]
                    force=bounded_force(pose[:3],velocity[:3],target)
                    view.apply_forces_and_torques_at_position(np.asarray([force],dtype=np.float32),None,None,np.array([0],np.int32),True)
                    commands[name]={'force_world_n':force.tolist(),'target_center_m':target.tolist()}
                world.step(render=False)
                time=float(world.current_time)
                if abs(time-previous-args.dt)>1e-7 or int(world.current_time_step_index)!=start_step+tick:raise RuntimeError('Missing native substep')
                previous=time
                bodies={n:{'pose_xyzw':array(v.get_transforms())[0].tolist(),'velocity':array(v.get_velocities())[0].tolist(),
                           'is_sleeping':bool(physics_interface.is_sleeping(stage_id,encoded_paths[n]))} for n,v in views.items()}
                if any(b['is_sleeping'] and np.linalg.norm(b['velocity'])>1e-5 for b in bodies.values()):
                    raise RuntimeError('Native sleep query contradicts native tensor velocity')
                joint_rows={n:joint_state(bodies['body'],bodies[n],np.array(joint_info[n]['anchor'])) for n in ('power','confirm')}
                raw=[array(v) for v in contact.get_contact_data(dt=1.)]
                scalars,points,normals,separation,counts,starts=raw
                active=np.argwhere(counts>0);used=max((int(starts[i,j])+int(counts[i,j]) for i,j in active),default=0)
                if used>=contact.max_contact_data_count:raise RuntimeError('Native contact buffer at capacity')
                valid_indices=np.array(sorted({k for i,j in active for k in range(int(starts[i,j]),int(starts[i,j])+int(counts[i,j]))}),dtype=int)
                if any(not np.isfinite(v[valid_indices]).all() for v in (scalars,points,normals,separation)):
                    raise RuntimeError('Nonfinite native contact sample')
                loaded=valid_indices[np.abs(scalars[valid_indices].reshape(-1))>1e-12]
                if len(loaded) and not np.allclose(np.linalg.norm(normals[loaded],axis=1),1.,atol=.01,rtol=0):
                    raise RuntimeError('Native contact normal is not a unit vector')
                contacts=[]
                for i,j in active:
                    begin,count=int(starts[i,j]),int(counts[i,j]);f=scalars[begin:begin+count].reshape(-1);normal=normals[begin:begin+count]
                    contacts.append({'sensor':sensor_paths[i],'filter':filter_paths[i][j],'count':count,
                        'normal_impulse_l1_ns':float(np.abs(f).sum()),'normal_impulse_vector_ns':(f[:,None]*normal).sum(axis=0).tolist(),
                        'min_separation_m':float(separation[begin:begin+count].min())})
                raw_stream.write(json.dumps({'tick':tick,'time_s':time,'native_indices':valid_indices.tolist(),'scalar_ns':scalars[valid_indices].reshape(-1).tolist(),
                    'points_m':points[valid_indices].tolist(),'normals':normals[valid_indices].tolist(),'separation_m':separation[valid_indices].reshape(-1).tolist(),
                    'counts':counts.tolist(),'starts':starts.tolist()},allow_nan=False)+'\n')
                min_z={n:float((button_points[n]@rotation(bodies[n]['pose_xyzw'][3:]).T+np.array(bodies[n]['pose_xyzw'][:3]))[:,2].min()) for n in ('power','confirm')}
                row={'tick':tick,'time_s':time,'elapsed_s':time-start,'phase':phase,'bodies':bodies,'joints':joint_rows,
                     'button_min_z_m':min_z,'probe_commands':commands,'contacts':contacts}
                stream.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                if tick%round(.5/args.dt)==0:
                    stream.flush();raw_stream.flush();print(json.dumps({'tick':tick,'phase':phase,'joints':joint_rows}),flush=True)
                if max(np.linalg.norm(v['velocity'][:3]) for v in bodies.values())>1.:
                    raise RuntimeError('Diagnostic speed exceeded 1 m/s; preserving raw failure')
                if any(v['pose_xyzw'][2]<.75 for v in bodies.values()):raise RuntimeError('Task body fell below table')
        report['native_state_before_export']=rows[-1]['bodies']
        world.pause();sync();stage.Flatten().Export(str(output/'final.usda'))
        report['native_state_after_export']={n:{'pose_xyzw':array(v.get_transforms())[0].tolist(),
                                               'velocity':array(v.get_velocities())[0].tolist()} for n,v in views.items()}
        report['export_scope']='One Fabric save only after all measured steps and world.pause; never used during live physics'
        cache=UsdGeom.XformCache();agreement={}
        for name,path in paths.items():
            matrix=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            actual=array(views[name].get_transforms())[0].astype(float)
            q=matrix.ExtractRotationQuat();usd_quat=np.array([*q.GetImaginary(),q.GetReal()])
            agreement[name]={'position_m':float(np.linalg.norm(np.array(matrix.ExtractTranslation())-actual[:3])),
                'orientation_rad':float(2*np.arccos(np.clip(abs(np.dot(usd_quat/np.linalg.norm(usd_quat),actual[3:]/np.linalg.norm(actual[3:]))),-1,1)))}
        if any(x['position_m']>1e-5 or x['orientation_rad']>1e-4 for x in agreement.values()):raise RuntimeError('Final USD/native mismatch')
        for data in joint_info.values():
            drive=UsdPhysics.DriveAPI(stage.GetPrimAtPath(data['joint_path']),'linear')
            if drive.GetTargetPositionAttr().Get()!=0 or drive.GetTargetVelocityAttr().Get()!=0:raise RuntimeError('Button target changed')
        if digest(candidate['initial_usd'])!=candidate['initial_usd_sha256']:
            raise RuntimeError('Pinned source asset changed during diagnostic')
        assessment=assess(rows,args.case,args.dt,paths,table_path=table_path)
        report['native_motion_consistency']={}
        for name in paths:
            positions=np.array([r['bodies'][name]['pose_xyzw'][:3] for r in rows])
            speeds=np.array([np.linalg.norm(r['bodies'][name]['velocity'][:3]) for r in rows])
            delta=np.linalg.norm(np.diff(positions,axis=0),axis=1)
            report['native_motion_consistency'][name]={'maximum_native_linear_speed_m_s':float(speeds.max()),
                'maximum_pose_difference_speed_m_s':float(delta.max()/args.dt),
                'moving_over_1um_with_zero_velocity_count':int(np.count_nonzero((delta>1e-6)&(speeds[1:]<1e-12)))}
        report.update(status='completed',passed=assessment['passed'],assessment=assessment,
            native_steps=len(rows),native_time_advance_s=previous-start,final_usd_native_agreement=agreement,
            button_drive_targets_unchanged=True,source_asset_sha256_after=digest(candidate['initial_usd']),
            artifact_sha256={p.name:digest(p) for p in output.glob('*') if p.name not in ('report.json','plan.json')})
        code=0 if assessment['passed'] else 2
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),completed_native_steps=len(rows))
        traceback.print_exc()
    finally:
        report['exit_code']=code;dump(output/'report.json',report)
        if app is not None:app.close(exit_code=code)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('reset','press','holder'),required=True)
    parser.add_argument('--dt',type=float,default=1/960)
    parser.add_argument('--release',type=Path,help='Bind the exact frozen task asset, environment, support materials and timestep')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    return run(args)


if __name__=='__main__':raise SystemExit(main())
