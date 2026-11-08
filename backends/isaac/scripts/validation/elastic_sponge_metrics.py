"""Float64 momentum and passive spring-scale reductions for actual rigid members."""
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.validation.sponge_freeflight_metrics import force_for_step
from scripts.validation.sponge_support_loadcell import CONFIG


def composite(raw, masses, local_com):
    q=np.asarray(raw['body_q'],float);v=np.asarray(raw['body_qd'],float);m=np.asarray(masses,float)
    if q.shape[1:]!=(len(m),7) or v.shape!=(*q.shape[:2],6):raise ValueError('Complete native member inventory required')
    if not all(np.isfinite(x).all() for x in (q,v,m)) or np.any(m<=0):raise ValueError('Finite actual masses and states required')
    r=Rotation.from_quat(q[:,:,3:].reshape(-1,4))
    points=r.apply(np.tile(np.asarray(local_com,float),(len(q),1))) .reshape(*q.shape[:2],3)+q[:,:,:3]
    return np.einsum('n,tnc->tc',m,v[:,:,:3]),np.einsum('n,tnc->tc',m,points)/m.sum(),points


def freeflight(raw,masses,coms,pad_index):
    p,center,points=composite(raw,masses,coms);t=np.asarray(raw['time_s'],float);dt=np.diff(t)
    if len(t)!=401 or not np.all(np.diff(raw['native_step'])==1) or not np.allclose(dt,float(np.float32(.0005)),atol=1e-10,rtol=0):raise ValueError('400 consecutive native steps required')
    f=np.asarray(raw['force_world_n'],float);expected=np.array([force_for_step(i) for i in range(401)],dtype=np.float32).astype(float)
    if not np.array_equal(f,expected):raise ValueError('Frozen .4N pulse schedule required')
    positions=np.asarray(raw['force_position_world_m'],float)
    if not np.allclose(positions[1:],points[:-1,pad_index].astype(np.float32),atol=2e-7,rtol=0):raise ValueError('Force must act at actual preceding native padCOM')
    impulse=np.sum(f[101:201]*dt[100:200,None],axis=0);dp=p[200]-p[100]
    error=abs(dp[0]-impulse[0])/impulse[0];release=abs(p[400,0]-p[200,0])/max(abs(p[200,0]),1e-15)
    gravity=[];motion=[]
    for name,start,end in [('fall',0,100),('pulse',100,200),('release',200,400),('all',0,400)]:
        expected=-9.81*np.sum(masses)*(t[end]-t[start]);actual=p[end,2]-p[start,2]
        integrated=np.sum(.5*(p[start:end]+p[start+1:end+1])*dt[start:end,None],axis=0)/np.sum(masses)
        displacement=center[end]-center[start]
        gravity.append({'phase':name,'expected_ns':float(expected),'actual_ns':float(actual),'relative_error':float(abs(actual-expected)/abs(expected))})
        motion.append({'phase':name,'com_displacement_m':displacement.tolist(),'integrated_com_velocity_m':integrated.tolist()})
    checks={'pulse_momentum_within_15_percent':error<=.15,'release_momentum_within_5_percent':release<=.05,
        'gravity_momentum_within_15_percent':all(x['relative_error']<=.15 for x in gravity),
        'unforced_transverse_momentum':np.max(abs(p[:,1]-p[0,1]))<=.15*impulse[0],
        'no_early_x_impulse':abs(p[100,0]-p[0,0])<=.15*impulse[0]}
    for axis in (0,2):
        selected=[x for x in motion if axis==2 or x['phase'] in ('pulse','release')]
        checks['com_q_matches_qd_'+str(axis)]=all(abs(x['com_displacement_m'][axis]-x['integrated_com_velocity_m'][axis])<=.15*max(abs(x['integrated_com_velocity_m'][axis]),1e-12) for x in selected)
    return checks,{'actual_impulse_ns':impulse.tolist(),'pulse_delta_p_ns':dp.tolist(),'pulse_relative_x_error':float(error),'release_relative_x_error':float(release),'gravity':gravity,'motion':motion,'native_total_mass_kg':float(np.sum(masses))},{'momentum_ns':p,'com_world_m':center}


def support(raw,masses,coms,empty=None):
    t=np.asarray(raw['time_s'],float);q=np.asarray(raw['plate_q'],float);v=np.asarray(raw['plate_qd'],float)
    if len(t)!=1201 or not np.all(np.diff(raw['native_step'])==1):raise ValueError('1200 consecutive native steps required')
    tail=np.asarray(raw['native_step'])>=raw['native_step'][-1]-800;interval=tail[:-1]&tail[1:];dt=np.diff(t)
    p=.2*v[:,:3]
    if len(masses):p+=composite(raw,masses,coms)[0]
    coord=q[:,2]-.89;elastic=-2000*coord;damp=-45*v[:,2];rate=(p[-1,2]-p[400,2])/(t[-1]-t[400]);balance=float((elastic+damp)[tail].mean()-rate)
    expected=(.2+sum(masses))*9.81;clearance=np.minimum(coord+.01,.01-coord)
    checks={'dt_matches':np.allclose(dt,float(np.float32(.0005)),atol=1e-10,rtol=0),'plate_quiet':np.max(np.linalg.norm(v[tail,:3],axis=1))<.025,
        'plate_adjacent_quiet':np.max(abs(np.diff(q[:,2])/dt)[interval])<.025,'plate_tail_range':np.ptp(q[tail,2])<=.00005,
        'plate_xy_locked':np.max(np.linalg.norm(q[:,:2]-[.6,-.8],axis=1))<1e-5,'plate_rotation_locked':np.max(np.linalg.norm(Rotation.from_quat(q[:,3:]).as_rotvec(),axis=1))<1e-3,
        'away_from_joint_limits':clearance.min()>=.005,'damping_small':abs(damp[tail].mean())<=.05*expected,'momentum_rate_small':abs(rate)<=.05*expected,
        'total_weight_within_15_percent':abs(balance-expected)<=.15*expected}
    measurement={'full_balance_n':balance,'elastic_mean_n':float(elastic[tail].mean()),'damping_mean_n':float(damp[tail].mean()),'nominal_momentum_rate_n':float(rate),'expected_total_weight_n':float(expected),'tail_plate_range_m':float(np.ptp(q[tail,2]))}
    if len(masses):
        if empty is None or empty['passed'] is not True:raise ValueError('Passing same-source empty calibration required')
        delta=balance-empty['measurements']['full_balance_n'];elastic_delta=measurement['elastic_mean_n']-empty['measurements']['elastic_mean_n'];weight=sum(masses)*9.81
        checks.update(tare_weight_within_15_percent=abs(delta-weight)<=.15*weight,tare_elastic_weight_within_15_percent=abs(elastic_delta-weight)<=.15*weight,
            tare_damping_small=abs(measurement['damping_mean_n']-empty['measurements']['damping_mean_n'])<=.05*weight,
            tare_momentum_small=abs(rate-empty['measurements']['nominal_momentum_rate_n'])<=.05*weight)
        bq=np.asarray(raw['body_q'],float);bv=np.asarray(raw['body_qd'],float);speed=np.linalg.norm(bv[tail,:,:3],axis=2);angular=np.linalg.norm(bv[tail,:,3:],axis=2)
        pose_speed=np.linalg.norm(np.diff(bq[:,:,:3],axis=0)/dt[:,None,None],axis=2)
        rots=Rotation.from_quat(bq[:,:,3:].reshape(-1,4));rmat=rots.as_matrix().reshape(len(t),len(masses),3,3)
        rel=np.einsum('tnji,tnjk->tnik',rmat[:-1],rmat[1:]);omega=np.linalg.norm(Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec(),axis=1).reshape(1200,len(masses))/dt[:,None]
        checks.update(all_native_body_linear_quiet=speed.max()<.025,all_native_body_angular_quiet=angular.max()<.25,
            all_pose_body_linear_quiet=pose_speed[interval].max()<.025,all_pose_body_angular_quiet=omega[interval].max()<.25)
        measurement.update(tare_full_balance_n=delta,tare_elastic_n=elastic_delta,expected_added_weight_n=float(weight),relative_tare_error=float(abs(delta-weight)/weight),
            maximum_tail_body_speed_m_s=float(speed.max()),maximum_tail_body_angular_speed_rad_s=float(angular.max()))
    return checks,measurement,{'momentum_ns':p,'spring_force_n':elastic,'spring_damping_n':damp}
