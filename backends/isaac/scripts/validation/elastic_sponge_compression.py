"""Passive dynamic platen with explicit bounded external loading, no pose writes."""
import numpy as np
from scipy.spatial.transform import Rotation
PLATEN='/World/ElasticCompressionPlaten'
JOINT='/World/ElasticCompressionGuide'
MASS=.1
CENTER=np.array([.6,-.8,.9528])
SIZE=np.array([.06,.04,.01])
STIFFNESS=200.
DAMPING=4.


def force_for_step(step):
    if not 0<=step<=6000:raise ValueError('Compression has6000 integration steps')
    if step<=800:load=0.
    elif step<=2000:
        u=(step-800)/1200;load=-3*u*u*(3-2*u)
    elif step<=2800:load=-3.
    elif step<=4000:
        u=(step-2800)/1200;load=-3*(1-u*u*(3-2*u))
    else:load=0.
    return np.array([0.,0.,MASS*9.81+load],dtype=np.float32),float(load)


def author_platen(stage,scene):
    from pxr import Gf,Sdf,UsdGeom,UsdPhysics
    cube=UsdGeom.Cube.Define(stage,PLATEN);cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(Gf.Vec3d(*CENTER));cube.AddScaleOp().Set(Gf.Vec3f(*SIZE))
    p=cube.GetPrim();rb=UsdPhysics.RigidBodyAPI.Apply(p);rb.CreateKinematicEnabledAttr(False);rb.CreateSimulationOwnerRel().SetTargets([Sdf.Path(scene)])
    UsdPhysics.CollisionAPI.Apply(p);p.AddAppliedSchema('PhysxCollisionAPI');p.AddAppliedSchema('PhysxContactReportAPI');p.AddAppliedSchema('PhysxRigidBodyAPI')
    for key,value in [('physxCollision:contactOffset',.0002),('physxCollision:restOffset',0.),('physxContactReport:threshold',0.),('physxRigidBody:linearDamping',0.),('physxRigidBody:angularDamping',0.)]:p.CreateAttribute(key,Sdf.ValueTypeNames.Float,custom=False).Set(value)
    mp=UsdPhysics.MassAPI.Apply(p);mp.CreateMassAttr(MASS);mp.CreateCenterOfMassAttr(Gf.Vec3f(0.));mp.CreatePrincipalAxesAttr(Gf.Quatf(1.));mp.CreateDiagonalInertiaAttr(Gf.Vec3f(*(MASS/12*(np.sum(SIZE**2)-SIZE**2))))
    joint=UsdPhysics.PrismaticJoint.Define(stage,JOINT);joint.CreateBody1Rel().SetTargets([Sdf.Path(PLATEN)])
    joint.CreateAxisAttr('Z');joint.CreateLocalPos0Attr(Gf.Vec3f(*CENTER));joint.CreateLocalPos1Attr(Gf.Vec3f(0.));joint.CreateLocalRot0Attr(Gf.Quatf(1.));joint.CreateLocalRot1Attr(Gf.Quatf(1.))
    joint.CreateLowerLimitAttr(-.03);joint.CreateUpperLimitAttr(.03);joint.CreateExcludeFromArticulationAttr(True)
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear');drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(STIFFNESS);drive.CreateDampingAttr(DAMPING);drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.);drive.CreateMaxForceAttr(float('inf'))
    return {'mass_kg':MASS,'size_m':SIZE.tolist(),'initial_center_m':CENTER.tolist(),'passive_guide_stiffness_n_m':STIFFNESS,'passive_guide_damping_ns_m':DAMPING,
        'fixed_platen_weight_balance_force_n':MASS*9.81,'extra_downward_load_n':3.,'max_compression_m':.01,'min_compression_m':.00025,
        'rebound_residual_gate_m':'max(.0002,20%loadedcompression)','phases_steps':{'settle':[0,800],'press':[800,2000],'loaded_hold':[2000,2800],'withdraw':[2800,4000],'recover':[4000,6000]},
        'boundary':'Dynamic native platen; worldZ prismatic guide only, passive fixed spring; no sponge/world or sponge/platen joint. Fixed force balances only platen self-weight, not sponge. External load is separate.'}


def quiet(raw,first):
    q=np.asarray(raw['body_q'],float);v=np.asarray(raw['body_qd'],float);t=np.asarray(raw['time_s'],float);dt=np.diff(t)
    linear=np.linalg.norm(v[first:,:,:3],axis=2);angular=np.linalg.norm(v[first:,:,3:],axis=2)
    pose=np.linalg.norm(np.diff(q[:,:,:3],axis=0)/dt[:,None,None],axis=2)
    r=Rotation.from_quat(q[:,:,3:].reshape(-1,4)).as_matrix().reshape(len(q),q.shape[1],3,3)
    rel=np.einsum('tnji,tnjk->tnik',r[:-1],r[1:]);speed=np.linalg.norm(Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec(),axis=1).reshape(len(dt),q.shape[1])/dt[:,None]
    values={'maximum_raw_linear_speed_m_s':float(linear.max()),'maximum_raw_angular_speed_rad_s':float(angular.max()),'maximum_pose_linear_speed_m_s':float(pose[first:].max()),'maximum_pose_angular_speed_rad_s':float(speed[first:].max())}
    return {'all_native_body_linear_quiet':linear.max()<.025,'all_native_body_angular_quiet':angular.max()<.25,'all_pose_body_linear_quiet':pose[first:].max()<.025,'all_pose_body_angular_quiet':speed[first:].max()<.25},values


def evaluate(raw,rows,masses):
    if len(rows)!=6001 or not np.all(np.diff(raw['native_step'])==1):raise ValueError('6000 complete compression steps required')
    height=np.array([x['central_top_in_pad_frame_m'] for x in rows]);base=float(height[600:801].mean());loaded=float(height[2600:2801].mean());recovered=float(height[5200:].mean());compression=base-loaded;residual=abs(recovered-base)
    checks,measure=quiet(raw,5200)
    q=np.asarray(raw['platen_q'],float);v=np.asarray(raw['platen_qd'],float);force=np.asarray(raw['force_world_n'],float)
    checks.update(dt_matches=np.allclose(np.diff(raw['time_s']),float(np.float32(.0005)),atol=1e-10,rtol=0),
        minimum_real_compression=compression>=.00025,bounded_real_compression=float(base-height.min())<=.01,rebound=residual<=max(.0002,.2*compression),
        platen_contact_on_loaded_sponge=all(any('/Actors/sponge_cell_' in p for p in r['platen_contact']['other_actor_paths']) for r in rows[2600:2801]),
        platen_released=all(r['platen_contact']['point_count']==0 and r['platen_bottom_minus_highest_sponge_m']>=.0005 for r in rows[5200:]),
        actual_springplate_contact=all(r['plate_contact']['point_count']>0 for r in rows[5200:]),
        platen_guide_limits_clear=np.max(abs(q[:,2]-CENTER[2]))<.025,
        platen_guide_xy_locked=np.max(np.linalg.norm(q[:,:2]-CENTER[:2],axis=1))<1e-5,
        platen_final_quiet=np.max(np.linalg.norm(v[5200:,:3],axis=1))<.025,
        load_not_through_plate=min(r['lowest_above_plate_top_m'] for r in rows)>=-.001,
        frozen_external_force_schedule=np.array_equal(force,np.array([np.zeros(3,np.float32)]+[force_for_step(i)[0] for i in range(1,6001)])))
    plateq=np.asarray(raw['plate_q'],float);platev=np.asarray(raw['plate_qd'],float);spring=-2000*(plateq[:,2]-.89)-45*platev[:,2]
    support_increment=float(spring[2600:2801].mean()-spring[600:801].mean())
    contact_force=float(np.mean([r['platen_contact']['net_contact_force_n'][2] for r in rows[2600:2801]]))
    guide=-STIFFNESS*(q[:,2]-CENTER[2])-DAMPING*v[:,2]
    times=np.asarray(raw['time_s'],float);platen_rate=MASS*(v[2800,2]-v[2600,2])/(times[2800]-times[2600])
    expected_contact=platen_rate+3.-float(guide[2600:2801].mean())
    checks['compression_load_reaches_support']=abs(support_increment-contact_force)<=.15*max(abs(contact_force),1e-9)
    checks['platen_external_force_balance']=abs(contact_force-expected_contact)<=.15*max(abs(expected_contact),1e-9)
    measure.update(support_increment_n=support_increment,actual_platen_contact_z_n=contact_force,expected_platen_contact_z_n=expected_contact,loaded_mean_guide_reaction_n=float(guide[2600:2801].mean()),baseline_height_m=base,loaded_height_m=loaded,recovered_height_m=recovered,loaded_compression_m=compression,maximum_compression_m=float(base-height.min()),rebound_residual_m=residual,
        loaded_mean_support_spring_force_n=float(spring[2600:2801].mean()),recovered_mean_support_spring_force_n=float(spring[5200:].mean()),
        loaded_mean_actual_platen_contact_force_n=np.mean([r['platen_contact']['net_contact_force_n'] for r in rows[2600:2801]],axis=0).tolist(),
        native_sponge_mass_kg=float(sum(masses)),fixed_platen_self_weight_force_n=MASS*9.81,extra_loaded_force_n=-3.)
    return checks,measure,{'central_top_height_m':height,'support_spring_force_n':spring}
