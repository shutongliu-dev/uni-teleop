"""Passive native spring-plate cross-check of sponge weight, with no state control.

Run empty, known-rigid, then FEM from the same sealed source. A world-to-plate
Prismatic joint has a fixed force-mode spring rest position; no target, pose,
velocity or node value is written after reset. This instrument is independent
of the contact-force buffer, but still uses the same PhysX solver.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from bench.isaacsim.release import checked, digest, dump
from bench.paths import load_json, OUTPUT_ROOT
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scenes.workrooms.kitchen.sponge_dynamics import VolumeGeometry

PROJECT = Path(__file__).resolve().parents[2]
PLATE = '/World/LoadPlate'
JOINT = '/World/LoadPlateSpring'
WEIGHT = '/World/CalibrationWeight'
SCENE = '/World/PhysicsScene'
CONFIG = {'plate_mass_kg': .2, 'plate_size_m': [.20,.15,.02],
    'plate_rest_center_m': [.6,-.8,.89], 'initial_top_m': .9,
    'spring_stiffness_n_m': 2000., 'spring_damping_ns_m': 45.,
    'joint_limits_m': [-.01,.01], 'joint_target_position_m': 0.,
    'joint_target_velocity_m_s': 0., 'known_mass_kg': .0792,
    'known_size_m': [.12,.078,.045], 'initial_load_bottom_m': .9003,
    'dt_s': .0005, 'duration_s': .6, 'tail_s': .4,
    'plate_contact_offset_m': .002, 'weight_contact_offset_m': .0002, 'instrument_rest_offset_m': 0.,
    'offset_scope': 'Declared independent instrument parameters; not claimed identical to canonical room support.',
    'solver': 'TGS', 'static_friction': .4, 'dynamic_friction': .4,
    'gravity_m_s2': 9.81, 'weight_relative_tolerance': .15,
    'limit_clearance_m': .005, 'quiet_linear_speed_m_s': .025,
    'quiet_angular_speed_rad_s': .25, 'tail_plate_position_range_m': .00005, 'static_correction_fraction': .05}


def instrument_config(solver='TGS'):
    if solver not in ('TGS','PGS'):raise ValueError('Instrument solver must be TGS or PGS')
    return {**CONFIG,'solver':solver}


def tail_mask(native_steps,config=CONFIG):
    """Select an exact number of native-step intervals despite float32 clock drift."""
    steps=np.asarray(native_steps)
    if steps.ndim!=1 or not len(steps):raise ValueError('A nonempty native step vector is required')
    return steps>=steps[-1]-round(config['tail_s']/config['dt_s'])


def native(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


def json_value(value):
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if np.isfinite(value) else str(value)
    try:
        return [json_value(x) for x in value]
    except TypeError:
        return str(value)


def load_asset(path):
    manifest=load_json(path)
    usd,meta=path.parent/'task.usda',path.parent/'metadata.json'
    checked(usd,manifest['usd_sha256']);checked(meta,manifest['metadata_sha256'])
    record=load_json(meta)
    if record.get('task_id')!='surface_wiping' or set(record.get('volume_bodies',{}))!={'foam'}:
        raise ValueError('An exact bonded FEM sponge manifest is required')
    return manifest,usd,record


def author_instrument(stage,case,manifest,usd,meta,config=CONFIG):
    """CPU authoring only. Must be called before native reset."""
    from pxr import Gf,Sdf,UsdGeom,UsdPhysics,UsdShade
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    material=UsdShade.Material.Define(stage,'/World/LoadPlateMaterial')
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(config['static_friction'])
    api.CreateDynamicFrictionAttr(config['dynamic_friction'])
    api.CreateRestitutionAttr(0.)
    def contact_report(prim):
        applied=prim.GetMetadata('apiSchemas')
        names=list(applied.GetAppliedItems()) if applied else []
        if 'PhysxContactReportAPI' not in names:names.append('PhysxContactReportAPI')
        prim.SetMetadata('apiSchemas',Sdf.TokenListOp.Create(prependedItems=names))
        prim.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float,custom=False).Set(0.)
    def box(path,center,size,mass):
        cube=UsdGeom.Cube.Define(stage,path);cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center));cube.AddScaleOp().Set(Gf.Vec3f(*size))
        body=UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        body.CreateKinematicEnabledAttr(False);body.CreateSimulationOwnerRel().SetTargets([Sdf.Path(SCENE)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        applied=list(cube.GetPrim().GetMetadata('apiSchemas').GetAppliedItems())
        cube.GetPrim().SetMetadata('apiSchemas',Sdf.TokenListOp.Create(prependedItems=applied+['PhysxCollisionAPI']))
        for key,value in [('physxCollision:contactOffset',config['plate_contact_offset_m'] if path==PLATE else config['weight_contact_offset_m']),('physxCollision:restOffset',config['instrument_rest_offset_m'])]:
            cube.GetPrim().CreateAttribute(key,Sdf.ValueTypeNames.Float,custom=False).Set(value)
        mp=UsdPhysics.MassAPI.Apply(cube.GetPrim());mp.CreateMassAttr(mass)
        mp.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        ix=mass/12*np.array([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
        mp.CreateDiagonalInertiaAttr(Gf.Vec3f(*ix));mp.CreatePrincipalAxesAttr(Gf.Quatf(1.))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material,materialPurpose='physics')
        contact_report(cube.GetPrim())
        return cube
    box(PLATE,config['plate_rest_center_m'],config['plate_size_m'],config['plate_mass_kg'])
    joint=UsdPhysics.PrismaticJoint.Define(stage,JOINT)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(PLATE)])
    joint.CreateAxisAttr().Set('Z')
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*config['plate_rest_center_m']))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.));joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.))
    joint.CreateLowerLimitAttr().Set(config['joint_limits_m'][0])
    joint.CreateUpperLimitAttr().Set(config['joint_limits_m'][1])
    joint.CreateExcludeFromArticulationAttr().Set(True)
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
    for method,value in [('CreateTypeAttr','force'),('CreateStiffnessAttr',config['spring_stiffness_n_m']),
        ('CreateDampingAttr',config['spring_damping_ns_m']),('CreateTargetPositionAttr',0.),
        ('CreateTargetVelocityAttr',0.),('CreateMaxForceAttr',float('inf'))]:
        getattr(drive,method)().Set(value)
    load_path=None
    if case=='rigid':
        size=config['known_size_m'];center=[.6,-.8,config['initial_load_bottom_m']+size[2]/2]
        box(WEIGHT,center,size,config['known_mass_kg']);load_path=WEIGHT
    elif case=='fem':
        root=stage.DefinePrim(manifest['root_prim_path'],'Xform')
        root.GetReferences().AddReference(str(usd),manifest['root_prim_path'])
        sponge=next(r for r in meta['objects'] if r['id']=='sponge')
        old=np.asarray(sponge['initial_pose_xyzw'][:3])
        shift=np.array([.6,-.8,config['initial_load_bottom_m']])-old
        for path in (meta['actors']['sponge'],meta['volume_bodies']['foam']['prim_path']):
            attr=stage.GetPrimAtPath(path).GetAttribute('xformOp:translate')
            if not attr or attr.Get() is None:raise ValueError('Explicit material-layer translation required')
            attr.Set(Gf.Vec3d(*(np.asarray(attr.Get())+shift)))
        load_path=meta['actors']['sponge']
        contact_report(stage.GetPrimAtPath(load_path))
    elif case!='empty':
        raise ValueError('Unknown loadcell case')
    return load_path


def instrument_properties(stage):
    prim=stage.GetPrimAtPath(JOINT)
    names=('physics:axis','physics:lowerLimit','physics:upperLimit','physics:localPos0','physics:localPos1',
        'physics:localRot0','physics:localRot1','physics:excludeFromArticulation',
        'drive:linear:physics:type','drive:linear:physics:stiffness','drive:linear:physics:damping',
        'drive:linear:physics:targetPosition','drive:linear:physics:targetVelocity','drive:linear:physics:maxForce')
    return {'joint_type':prim.GetTypeName(),'attributes':{n:json_value(prim.GetAttribute(n).Get()) for n in names},
        'body0':[str(x) for x in prim.GetRelationship('physics:body0').GetTargets()],
        'body1':[str(x) for x in prim.GetRelationship('physics:body1').GetTargets()],
        'plate_mass_attr_kg':stage.GetPrimAtPath(PLATE).GetAttribute('physics:mass').Get()}


def contact_row(view,dt,baseline=False):
    """Copy native buffers before another getter reuses its storage."""
    f,p,n,s,c,start,ids=view.get_raw_contact_data(dt=1.)
    count,begin=int(native(c).reshape(-1)[0]),int(native(start).reshape(-1)[0])
    if count>=view.max_contact_data_count:raise RuntimeError('Contact buffer may be saturated')
    f=native(f)[begin:begin+count].reshape(-1);p=native(p)[begin:begin+count]
    n=native(n)[begin:begin+count];s=native(s)[begin:begin+count].reshape(-1)
    idcopy=ids[begin:begin+count].detach().cpu().clone()
    paths=view.get_other_actor_paths_from_ids(idcopy) if count else []
    if len(paths)!=count or any(not path for path in paths):raise RuntimeError('Native contact actor-ID path resolution failed')
    vector=(f.astype(float)[:,None]*n.astype(float)).sum(0)
    net=native(view.get_net_contact_forces(dt)).reshape(-1,3)[0]
    matrix=native(view.get_contact_force_matrix(dt)).tolist() if view.filter_count else None
    return {'point_count':count,'scalar_impulses_ns':f.tolist(),'points_world_m':p.tolist(),
        'normals':n.tolist(),'separations_m':s.tolist(),'other_actor_ids':native(idcopy).tolist(),
        'other_actor_paths':list(paths),'summed_normal_impulse_ns':vector.tolist(),
        'net_contact_force_n':net.tolist(),'filtered_contact_force_matrix_n':matrix,
        'filtered_matrix_availability':'native' if view.filter_count else 'unavailable_without_filter_prims',
        'finite':bool(all(np.isfinite(a).all() for a in (f,p,n,s,net))),
        'point_sum_to_net_force_error_n':float(np.linalg.norm(vector/dt-net)),
        'point_sum_to_filtered_matrix_error_n':float(np.linalg.norm(vector/dt-np.asarray(matrix).reshape(-1,3).sum(0))) if matrix is not None else None,
        'baseline':baseline}


def static_collision_bounds(stage,exclude):
    from pxr import UsdGeom,UsdPhysics
    cache=UsdGeom.BBoxCache(0.,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy,UsdGeom.Tokens.guide],useExtentsHint=False)
    rows=[]
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or any(path==x or path.startswith(x+'/') for x in exclude):continue
        bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo,hi=np.array(bounds.GetMin()),np.array(bounds.GetMax())
        if np.isfinite([lo,hi]).all():rows.append({'path':path,'bounds_m':[lo.tolist(),hi.tolist()]})
    return rows


def bbox_gap(lower,upper,other):
    return float(np.linalg.norm(np.maximum(np.maximum(other[0]-upper,lower-other[1]),0.)))


def evaluate_raw(raw,case,reference=None,calibration=None,foam_spec=None,config=CONFIG):
    time=np.asarray(raw['time_s']);p=np.asarray(raw['plate_q']);v=np.asarray(raw['plate_qd'])
    tail=tail_mask(raw['native_step'],config)
    tail_intervals=tail[:-1]&tail[1:]
    coordinate=p[:,2]-config['plate_rest_center_m'][2]
    spring=-config['spring_stiffness_n_m']*coordinate
    spring_damped=spring-config['spring_damping_ns_m']*v[:,2]
    adjacent=np.diff(p[:,2])/np.diff(time)
    mean_z=float(p[tail,2].mean());median_z=float(np.median(p[tail,2]))
    momentum=config['plate_mass_kg']*v[:,:3]
    load_checks={};load_measurements={}
    if case!='empty':
        lp,lv=np.asarray(raw['load_q']),np.asarray(raw['load_qd'])
        load_speed=np.linalg.norm(lv[tail,:3],axis=1)
        load_pose_speed=np.linalg.norm(np.diff(lp[:,:3],axis=0)/np.diff(time)[:,None],axis=1)
        load_angular=np.linalg.norm(lv[tail,3:],axis=1)
        orientation=Rotation.from_quat(lp[:,3:])
        load_pose_angular=np.linalg.norm((orientation[:-1].inv()*orientation[1:]).as_rotvec(),axis=1)/np.diff(time)
        load_measurements={'tail_max_native_load_speed_m_s':float(load_speed.max()),
            'tail_max_adjacent_load_speed_m_s':float(load_pose_speed[tail_intervals].max()),
            'tail_max_native_load_angular_speed_rad_s':float(load_angular.max()),
            'tail_max_adjacent_load_angular_speed_rad_s':float(load_pose_angular[tail_intervals].max())}
        load_checks={'load_raw_quiet':bool(load_speed.max()<config['quiet_linear_speed_m_s']),
            'load_adjacent_quiet':bool(load_pose_speed[tail_intervals].max()<config['quiet_linear_speed_m_s']),
            'load_angular_quiet':bool(load_angular.max()<config['quiet_angular_speed_rad_s']),
            'load_adjacent_angular_quiet':bool(load_pose_angular[tail_intervals].max()<config['quiet_angular_speed_rad_s'])}
        if case=='rigid':momentum+=config['known_mass_kg']*lv[:,:3]
        else:
            if foam_spec is None:raise ValueError('FEM momentum review requires frozen mass and rest tetrahedra')
            rest=np.asarray(foam_spec['rest_points_m']);tet=np.asarray(foam_spec['tetrahedra']);t=rest[tet]
            vol=np.einsum('ij,ij->i',np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),t[:,3]-t[:,0])/6
            if np.any(vol<=0):raise ValueError('Nonpositive rest tetrahedra')
            lumped=np.zeros(len(rest));np.add.at(lumped,tet.reshape(-1),np.repeat(vol/vol.sum()*foam_spec['total_mass_kg']/4,4))
            momentum+=(config['known_mass_kg']-foam_spec['total_mass_kg'])*lv[:,:3]
            momentum+=np.einsum('n,tnc->tc',lumped,raw['foam_qd'])
    tail_ids=np.flatnonzero(tail)
    rate=(momentum[tail_ids[-1],2]-momentum[tail_ids[0],2])/(time[tail_ids[-1]]-time[tail_ids[0]])
    damping_mean=float(-config['spring_damping_ns_m']*v[tail,2].mean())
    balance=float(spring_damped[tail].mean()-rate)
    limits=config['joint_limits_m'];clearance=np.minimum(coordinate-limits[0],limits[1]-coordinate)
    measured={'tail_frames':int(tail.sum()),'tail_intervals':int(tail_intervals.sum()),
        'tail_elapsed_s':float(time[tail][-1]-time[tail][0]),'tail_selection':'last round(tail_s/dt_s) native-step intervals including both endpoints','tail_mean_plate_z_m':mean_z,'tail_median_plate_z_m':median_z,
        'tail_mean_spring_elastic_force_n':float(spring[tail].mean()),
        'tail_mean_spring_force_with_damping_n':float(spring_damped[tail].mean()),
        'tail_mean_damping_force_n':damping_mean,'nominal_total_momentum_rate_n':float(rate),
        'total_weight_from_spring_and_momentum_n':balance, **load_measurements,
        'tail_max_native_plate_speed_m_s':float(np.linalg.norm(v[tail,:3],axis=1).max()),
        'tail_max_adjacent_plate_z_speed_m_s':float(abs(adjacent[tail_intervals]).max()),
        'tail_plate_z_range_m':float(np.ptp(p[tail,2])),
        'minimum_joint_limit_clearance_m':float(clearance.min()),
        'maximum_plate_xy_drift_m':float(np.linalg.norm(p[:,:2]-np.array(config['plate_rest_center_m'][:2]),axis=1).max()),
        'maximum_plate_rotation_rad':float(np.linalg.norm(Rotation.from_quat(p[:,3:]).as_rotvec(),axis=1).max()),
        'expected_total_weight_n':(config['plate_mass_kg']+(0 if case=='empty' else config['known_mass_kg']))*config['gravity_m_s2']}
    checks={'all_raw_states_finite':all(np.isfinite(x).all() for x in raw.values()),
        'complete_planned_steps':len(time)==round(config['duration_s']/config['dt_s'])+1,
        'native_steps_consecutive':bool(np.all(np.diff(raw['native_step'])==1)),
        'time_steps_match':bool(np.allclose(np.diff(time),config['dt_s'],rtol=2e-6,atol=1e-10)),
        'away_from_joint_limits':bool(clearance.min()>=config['limit_clearance_m']),
        'locked_xy':measured['maximum_plate_xy_drift_m']<1e-5,
        'locked_rotations':measured['maximum_plate_rotation_rad']<1e-3,
        'quiet_native_plate':measured['tail_max_native_plate_speed_m_s']<config['quiet_linear_speed_m_s'],
        'quiet_adjacent_plate':measured['tail_max_adjacent_plate_z_speed_m_s']<config['quiet_linear_speed_m_s'],
        'plate_tail_position_range':measured['tail_plate_z_range_m']<=config['tail_plate_position_range_m']}
    checks.update(load_checks)
    expected_total=measured['expected_total_weight_n']
    checks['mean_damping_small_for_static_weighing']=abs(damping_mean)<=config['static_correction_fraction']*expected_total
    checks['momentum_rate_small_for_static_weighing']=abs(rate)<=config['static_correction_fraction']*expected_total
    checks['total_weight_from_full_balance']=abs(balance-expected_total)<=config['weight_relative_tolerance']*expected_total
    if case=='empty':
        expected=config['plate_mass_kg']*config['gravity_m_s2']
        checks['empty_spring_supports_declared_plate_mass']=abs(measured['tail_mean_spring_elastic_force_n']-expected)<=config['weight_relative_tolerance']*expected
    else:
        if reference is None:raise ValueError('A completed empty calibration is required')
        delta=config['spring_stiffness_n_m']*(reference['measurements']['tail_mean_plate_z_m']-mean_z)
        expected=config['known_mass_kg']*config['gravity_m_s2']
        measured.update(tare_subtracted_spring_load_n=delta,expected_added_weight_n=expected,
            added_weight_relative_error=abs(delta-expected)/expected)
        delta_balance=balance-reference['measurements']['total_weight_from_spring_and_momentum_n']
        damping_difference=damping_mean-reference['measurements']['tail_mean_damping_force_n']
        momentum_difference=float(rate)-reference['measurements']['nominal_total_momentum_rate_n']
        measured.update(tare_subtracted_full_balance_load_n=delta_balance,
            tare_damping_difference_n=damping_difference,tare_momentum_rate_difference_n=momentum_difference)
        checks['tare_damping_correction_small']=abs(damping_difference)<=config['static_correction_fraction']*expected
        checks['tare_momentum_correction_small']=abs(momentum_difference)<=config['static_correction_fraction']*expected
        checks['tare_full_balance_weight_within_15_percent']=abs(delta_balance-expected)<=config['weight_relative_tolerance']*expected
        checks['empty_calibration_passed']=bool(reference['passed'])
        checks['spring_added_weight_within_15_percent']=abs(delta-expected)<=config['weight_relative_tolerance']*expected
        if case=='fem':
            if calibration is None:raise ValueError('A completed known rigid calibration is required')
            checks['known_rigid_calibration_passed']=bool(calibration['passed'])
            speed=np.linalg.norm(raw['foam_qd'][tail],axis=2)
            q=np.asarray(raw['foam_q']);pose_speed=np.linalg.norm(np.diff(q.astype(float),axis=0)/np.diff(time)[:,None,None],axis=2)
            measured.update(tail_max_raw_foam_speed_m_s=float(speed.max()),
                tail_max_adjacent_foam_speed_m_s=float(pose_speed[tail_intervals].max()))
            checks['foam_raw_quiet']=bool(speed.max()<config['quiet_linear_speed_m_s'])
            checks['foam_adjacent_quiet']=bool(pose_speed[tail_intervals].max()<config['quiet_linear_speed_m_s'])
    return checks,measured


def stable_source_identity(source):
    """Cross-process source identity; retain PID separately in the full receipt."""
    required=('snapshot_root','snapshot_manifest_sha256','entrypoint','imported_project_modules','required_modules')
    if not all(k in source for k in required):
        raise ValueError('Frozen source identity lacks root, manifest, entrypoint or actual imports')
    return {'snapshot_root':source['snapshot_root'],
        'snapshot_manifest_sha256':source['snapshot_manifest_sha256'],
        'entrypoint':source['entrypoint'],
        'required_modules':sorted(source['required_modules']),
        'imported_project_modules':source['imported_project_modules'],
        'project_package_paths':source.get('project_package_paths',{}),
        'verified':source.get('verified'), 'runtime_imports_verified':source.get('runtime_imports_verified')}


def collision_offsets(stage,paths):
    """Composed attributes plus authored status; absent is not an assumed default."""
    from pxr import UsdPhysics
    result=[]
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not any(path==x or path.startswith(x+'/') for x in paths):continue
        attrs={}
        for key in ('physxCollision:contactOffset','physxCollision:restOffset'):
            attr=prim.GetAttribute(key)
            attrs[key]={'composed_value_m':json_value(attr.Get()) if attr else None,
                'authored':bool(attr and attr.HasAuthoredValueOpinion())}
        result.append({'path':path,'attributes':attrs})
    return result


def native_offsets(view):
    result={'body_paths':list(view.prim_paths),'count':int(view.count),'max_shapes':int(view.max_shapes)}
    for name in ('get_contact_offsets','get_rest_offsets'):
        try:result[name]={'available':True,'values_m':native(getattr(view,name)()).tolist()}
        except Exception as exc:result[name]={'available':False,'error':str(exc)}
    return result


def reference_report(path,case,source,manifest_hash,config=CONFIG):
    if path is None:return None
    result=load_json(path)
    if (result.get('schema')!='sponge_passive_loadcell.v1' or result.get('case')!=case
        or result.get('instrument')!=config or stable_source_identity(result.get('source',{}))!=stable_source_identity(source)
        or result.get('asset_manifest_sha256')!=manifest_hash):
        raise ValueError('Calibration must match exact source, asset, instrument and requested case')
    if result.get('passed') is not True:raise ValueError('Preceding calibration did not pass')
    for name,sha in result['evidence_sha256'].items():checked(path.parent/name,sha)
    return result


def run(args):
    config=instrument_config(args.solver)
    source=verify_if_snapshot(PROJECT,required_modules=('scripts.validation.sponge_support_loadcell','scenes.workrooms.kitchen.sponge_dynamics'),entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve()!=PROJECT:raise ValueError('Run from the complete frozen source root')
    source={**source,'snapshot_manifest_sha256':digest(PROJECT/'source_snapshot_manifest.json')}
    if args.case!='empty' and args.empty_report is None:raise ValueError('Rigid/FEM cases require --empty-report')
    if args.case=='fem' and args.rigid_report is None:raise ValueError('FEM case requires --rigid-report')
    output=args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):raise ValueError('Evidence belongs in shared datasets')
    output.mkdir(parents=True,exist_ok=False)
    manifest,usd,meta=load_asset(args.asset_manifest)
    manifest_hash=digest(args.asset_manifest)
    empty=reference_report(args.empty_report,'empty',source,manifest_hash,config)
    rigid=reference_report(args.rigid_report,'rigid',source,manifest_hash,config)
    report={'schema':'sponge_passive_loadcell.v1','case':args.case,'passed':False,'source':source,
        'asset_manifest':str(args.asset_manifest.resolve()),'asset_manifest_sha256':manifest_hash,
        'instrument':config,'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'scope':'Passive calibrated spring displacement cross-check; same engine, independent of contact force buffer. Not robot task or material qualification.',
        'post_reset_state_writes':0,'post_reset_joint_target_writes':0,'render_calls':0,'model_calls':0,
        'calibration_inputs':{k:({'path':str(p.resolve()),'sha256':digest(p)} if p else None)
            for k,p in [('empty',args.empty_report),('rigid',args.rigid_report)]}}
    raw={k:[] for k in ('time_s','native_step','plate_q','plate_qd','load_q','load_qd','foam_q','foam_qd')}
    rows=[];chunks=[];archived=0;app=None;code=1
    def flush():
        nonlocal archived
        stop=len(raw['time_s'])
        if stop==archived:return
        folder=output/'native_chunks';folder.mkdir(exist_ok=True)
        path=folder/f'frames_{len(chunks):06d}.npz'
        np.savez_compressed(path,**{k:np.asarray(v[archived:stop]) for k,v in raw.items()})
        chunks.append({'path':str(path),'first_frame':archived,'last_frame':stop-1,'sha256':digest(path)})
        archived=stop;dump(output/'native_chunks_manifest.json',{'frames':stop,'chunks':chunks,'trajectory_complete':False})
    try:
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':True,'multi_gpu':False,'active_cuda_gpus':[0],
            'max_gpu_count':1,'physics_gpu':0,'fast_shutdown':True,'extra_args':
            ['--/rtx/verifyDriverVersion/enabled=false'] if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER')=='1' else []},
            experience=str(PROJECT/'bench/apps/workroom.headless.kit'))
        import omni.usd
        from isaacsim.core.api import World
        from pxr import UsdPhysics
        omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
        load_path=author_instrument(stage,args.case,manifest,usd,meta,config)
        world=World(stage_units_in_meters=1.,physics_dt=config['dt_s'],rendering_dt=1/30,
            backend='torch',device='cuda:0',physics_prim_path=SCENE)
        pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type(config['solver'])
        exclusions=[PLATE]+([load_path] if load_path else [])
        foam=meta['volume_bodies']['foam']
        if args.case=='fem':exclusions.append(foam['prim_path'])
        obstacles=static_collision_bounds(stage,exclusions)
        report['other_collision_bounds']=obstacles
        report['composed_collision_offsets']=collision_offsets(stage,exclusions)
        report['actual_spring_properties']=instrument_properties(stage)
        sceneprim=stage.GetPrimAtPath(SCENE)
        report['actual_scene_properties']={n:json_value(sceneprim.GetAttribute(n).Get()) for n in
            ('physxScene:solverType','physxScene:enableGPUDynamics','physxScene:enableExternalForcesEveryIteration','physics:gravityMagnitude','physics:gravityDirection')}
        stage.Export(str(output/'initial.usda'));world.reset();sim=world.physics_sim_view
        report['native_tensor_backend']={'frontend':'torch','device_ordinal':int(sim.device_ordinal),'device':str(sim.device)}
        if int(sim.device_ordinal)<0:raise RuntimeError('Native GPU simulation view required; CPU fallback rejected')
        plate=sim.create_rigid_body_view(PLATE)
        load=sim.create_rigid_body_view(load_path) if load_path else None
        volume=sim.create_volume_deformable_body_view(foam['prim_path']) if args.case=='fem' else None
        contact=sim.create_rigid_contact_view(PLATE,filter_patterns=[load_path] if load_path else [],max_contact_data_count=8192)
        loadcontact=sim.create_rigid_contact_view(load_path,filter_patterns=[PLATE],max_contact_data_count=8192) if load_path else None
        if plate.count!=1 or (load is not None and load.count!=1) or (volume is not None and volume.count!=1):
            raise ValueError('Expected exactly one native body for each instrument component')
        report['native_rigid_shape_offsets']={'plate':native_offsets(plate)}
        if load is not None:report['native_rigid_shape_offsets']['load']=native_offsets(load)
        report['native_plate_mass_kg']=float(native(plate.get_masses()).reshape(-1)[0])
        if load is not None:report['native_load_rigid_mass_kg']=float(native(load.get_masses()).reshape(-1)[0])
        geom=None
        if volume is not None:
            bodyprim=stage.GetPrimAtPath(foam['prim_path'])
            report['composed_deformable_properties']={n:json_value(bodyprim.GetAttribute(n).Get()) for n in (
                'omniphysics:mass','omniphysics:kinematicEnabled','omniphysics:deformableBodyEnabled',
                'physxDeformableBody:linearDamping','physxDeformableBody:settlingDamping',
                'physxDeformableBody:settlingThreshold','physxDeformableBody:sleepThreshold',
                'physxDeformableBody:solverPositionIterationCount')}
            report['composed_pad_kinematic']=stage.GetPrimAtPath(load_path).GetAttribute('physics:kinematicEnabled').Get()
            report['authored_attachment_inventory']=[{'path':str(p.GetPath()),'type':p.GetTypeName(),
                'enabled':json_value(p.GetAttribute('omniphysics:attachmentEnabled').Get()),
                'src0':[str(x) for x in p.GetRelationship('omniphysics:src0').GetTargets()],
                'src1':[str(x) for x in p.GetRelationship('omniphysics:src1').GetTargets()]}
                for p in stage.Traverse() if p.GetTypeName().endswith('Attachment')]
            report['kinematic_target_getter_used']=False
            report['kinematic_target_getter_boundary']='The public upstream tensor getter can read a library-owned buffer before any setter establishes native targets; it is not used as evidence of native pin flags. No target setter is called.'
            tets=native(volume.get_simulation_element_indices()).reshape(-1,4)
            if not np.array_equal(tets,foam['tetrahedra']):raise ValueError('Native FEM tetrahedra changed')
            geom=VolumeGeometry(foam['rest_points_m'],foam['tetrahedra'])
            mv=sim.create_deformable_material_view(foam['material']['material_path'])
            if mv.count!=1:raise ValueError('Expected one effective native FEM material')
            report['native_material']={n:float(native(getattr(mv,g)()).reshape(-1)[0]) for n,g in
                [('youngs_modulus_pa','get_youngs_modulus'),('poissons_ratio','get_poissons_ratio'),('dynamic_friction','get_dynamic_friction')]}
            if any(not np.isclose(v,foam['material'][n],rtol=1e-5,atol=1e-7) for n,v in report['native_material'].items()):raise ValueError('Effective FEM material changed')
            report['native_volume_paths']=list(volume.prim_paths)
            report['native_simulation_mesh_paths']=list(volume.simulation_mesh_prim_paths)
        dump(output/'native_initialization.json',report)
        initial_time=float(world.current_time);initial_step=int(world.current_time_step_index)
        support=np.asarray(next(x for x in meta['objects'] if x['id']=='sponge')['collision_support_points_m'])
        def sample(baseline=False):
            p=native(plate.get_transforms())[0];v=native(plate.get_velocities())[0]
            lp=native(load.get_transforms())[0] if load is not None else np.empty(0)
            lv=native(load.get_velocities())[0] if load is not None else np.empty(0)
            q=native(volume.get_simulation_nodal_positions())[0] if volume is not None else np.empty((0,3))
            qd=native(volume.get_simulation_nodal_velocities())[0] if volume is not None else np.empty((0,3))
            # Archive the offending raw frame before derived validation can fail.
            for key,value in [('time_s',float(world.current_time)-initial_time),('native_step',int(world.current_time_step_index)),
                ('plate_q',p),('plate_qd',v),('load_q',lp),('load_qd',lv),('foam_q',q),('foam_qd',qd)]:raw[key].append(value)
            row={'time_s':raw['time_s'][-1],'native_step':raw['native_step'][-1],'baseline':baseline,
                'plate_contact':contact_row(contact,config['dt_s'],baseline),
                'load_contact':contact_row(loadcontact,config['dt_s'],baseline) if loadcontact else None}
            plate_points=np.array([[sx,sy,sz] for sx in (-.1,.1) for sy in (-.075,.075) for sz in (-.01,.01)])
            plate_points=Rotation.from_quat(p[3:]).apply(plate_points)+p[:3]
            moving=[plate_points]
            rows.append(row)
            if args.case=='fem':
                measure=geom.sample(q,qd);r=Rotation.from_quat(lp[3:])
                anchors=r.apply(foam['attachment']['rigid_local_positions_m'])+lp[:3]
                measure['maximum_material_bond_error_m']=float(np.linalg.norm(q[np.asarray(foam['attachment']['node_indices'])]-anchors,axis=1).max())
                padpoints=r.apply(support)+lp[:3];moving.extend([q,padpoints])
                row['foam']=measure
                local=(q-p[:3])@Rotation.from_quat(p[3:]).as_matrix()
                row['foam_lowest_above_plate_top_m']=float(local[:,2].min()-.01)
                local_load=(padpoints-p[:3])@Rotation.from_quat(p[3:]).as_matrix()
                row['load_lowest_above_plate_top_m']=float(local_load[:,2].min()-.01)
                row['load_plate_xy_clearance_m']=float((np.array([.1,.075])-abs(local_load[:,:2])).min())
            elif args.case=='rigid':
                corners=np.array([[sx,sy,sz] for sx in (-.06,.06) for sy in (-.039,.039) for sz in (-.0225,.0225)])
                corners=Rotation.from_quat(lp[3:]).apply(corners)+lp[:3];moving.append(corners)
                local_load=(corners-p[:3])@Rotation.from_quat(p[3:]).as_matrix()
                row['load_lowest_above_plate_top_m']=float(local_load[:,2].min()-.01)
                row['load_plate_xy_clearance_m']=float((np.array([.1,.075])-abs(local_load[:,:2])).min())
            points=np.concatenate(moving);lo,hi=points.min(0),points.max(0)
            row['minimum_other_collider_bbox_gap_m']=min((bbox_gap(lo,hi,np.asarray(o['bounds_m'])) for o in obstacles),default=None)
            if len(raw['time_s'])-archived>=256:flush()
            return row
        with (output/'native_frames.jsonl').open('x') as stream:
            row=sample(True);stream.write(json.dumps(row,allow_nan=False)+'\n')
            for _ in range(round(config['duration_s']/config['dt_s'])):
                world.step(render=False);row=sample();stream.write(json.dumps(row,allow_nan=False)+'\n')
                if args.case=='fem' and (row['foam']['inverted_tetrahedron_count'] or row['foam']['maximum_material_bond_error_m']>.004):
                    stream.flush();raise RuntimeError('FEM inverted or detached; raw failure retained')
        data={k:np.asarray(v) for k,v in raw.items()}
        checks,measure=evaluate_raw(data,args.case,empty,rigid,foam if volume is not None else None,config)
        checks['actual_solver_matches_requested']=report['actual_scene_properties']['physxScene:solverType']==config['solver']
        checks['native_plate_mass_matches']=bool(np.isclose(report['native_plate_mass_kg'],config['plate_mass_kg'],rtol=2e-6))
        if load is not None:
            expected=config['known_mass_kg'] if args.case=='rigid' else next(x for x in meta['objects'] if x['id']=='sponge')['mass_kg']
            checks['native_load_mass_matches']=bool(np.isclose(report['native_load_rigid_mass_kg'],expected,rtol=2e-6))
        contacts=[c for r in rows for c in (r['plate_contact'],r['load_contact']) if c is not None]
        checks['native_contact_data_finite']=all(c['finite'] for c in contacts)
        measure['maximum_point_sum_vs_native_net_force_error_n']=max(c['point_sum_to_net_force_error_n'] for c in contacts)
        measure['maximum_point_sum_vs_filtered_matrix_error_n']=max((c['point_sum_to_filtered_matrix_error_n'] for c in contacts if c['point_sum_to_filtered_matrix_error_n'] is not None),default=None)
        checks['contact_buffers_agree']=measure['maximum_point_sum_vs_native_net_force_error_n']<1e-4 and (measure['maximum_point_sum_vs_filtered_matrix_error_n'] is None or measure['maximum_point_sum_vs_filtered_matrix_error_n']<1e-4)
        checks['native_offsets_available']=all(v[name]['available'] for v in report['native_rigid_shape_offsets'].values() for name in ('get_contact_offsets','get_rest_offsets'))
        checks['native_offsets_match_declared_instrument']=checks['native_offsets_available'] and all(
            np.allclose(v['get_contact_offsets']['values_m'],config['plate_contact_offset_m'] if key=='plate' else config['weight_contact_offset_m'],rtol=1e-5,atol=1e-8)
            and np.allclose(v['get_rest_offsets']['values_m'],0.,atol=1e-8) for key,v in report['native_rigid_shape_offsets'].items())
        checks['spring_properties_unchanged']=instrument_properties(stage)==report['actual_spring_properties']
        checks['no_other_static_collider_nearby']=all(r['minimum_other_collider_bbox_gap_m'] is None or r['minimum_other_collider_bbox_gap_m']>=.005 for r in rows)
        checks['no_unexpected_rigid_contacts']=all(set(r['plate_contact']['other_actor_paths'])<=({load_path} if load_path else set())
            and (r['load_contact'] is None or set(r['load_contact']['other_actor_paths'])<={PLATE}) for r in rows)
        if load is not None:
            checks['load_not_through_plate']=min(r['load_lowest_above_plate_top_m'] for r in rows)>=-.001
            terminal=[r for r,selected in zip(rows,tail_mask([r['native_step'] for r in rows],config)) if selected]
            checks['load_supported_on_plate']=all(abs(r['load_lowest_above_plate_top_m'])<=.001 and r['load_plate_xy_clearance_m']>=.005 for r in terminal)
            checks['terminal_native_rigid_contact_present']=sum(r['load_contact']['point_count']>0 for r in terminal)>=.95*len(terminal)
        if args.case=='fem':
            checks.update(no_inverted_tets=all(r['foam']['inverted_tetrahedron_count']==0 for r in rows),
                no_gross_distortion=all(r['foam']['minimum_tetrahedron_volume_ratio']>=.1 and r['foam']['minimum_edge_length_ratio']>=.5 and r['foam']['maximum_edge_length_ratio']<=1.5 for r in rows),
                material_bond_intact=max(r['foam']['maximum_material_bond_error_m'] for r in rows)<=.001)
        report.update(checks={k:bool(v) for k,v in checks.items()},measurements=measure,passed=all(checks.values()),native_steps=len(rows)-1)
        code=0 if report['passed'] else 1
    except BaseException:
        report['exception']=traceback.format_exc();traceback.print_exc()
    finally:
        try:
            flush()
            path=output/'native_states.npz';arrays={k:np.asarray(v) for k,v in raw.items()}
            if args.case=='fem':arrays.update(rest_points_m=np.asarray(meta['volume_bodies']['foam']['rest_points_m']),tetrahedra=np.asarray(meta['volume_bodies']['foam']['tetrahedra']))
            np.savez_compressed(path,**arrays)
            report['raw_frames']=len(raw['time_s'])
            report['evidence_sha256']={name:digest(output/name) for name in ('initial.usda','native_frames.jsonl','native_states.npz') if (output/name).exists()}
        except BaseException:
            report['archive_exception']=traceback.format_exc();code=1;report['passed']=False
        finally:
            report['process_exit_code']=code
            try:
                # Re-emit all successfully read contact rows even if later geometry validation failed.
                if rows:
                    with (output/'native_frames.jsonl').open('w') as saved:
                        for row in rows:saved.write(json.dumps(row,allow_nan=False)+'\n')
                    report.setdefault('evidence_sha256',{})['native_frames.jsonl']=digest(output/'native_frames.jsonl')
                dump(output/'qualification.json',report)
                print(json.dumps({'case':args.case,'passed':report['passed'],'output':str(output)}),flush=True)
            finally:
                if app is not None:app.close(exit_code=code)
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver',choices=('TGS','PGS'),default='TGS',help='Explicit instrument solver; all calibration cases must match')
    parser.add_argument('--case',choices=('empty','rigid','fem'),required=True)
    parser.add_argument('--asset-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--empty-report',type=Path)
    parser.add_argument('--rigid-report',type=Path)
    raise SystemExit(run(parser.parse_args()))
