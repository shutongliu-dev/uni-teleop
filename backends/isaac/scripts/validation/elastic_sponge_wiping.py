"""Original bounded pad-force wiping fixture, with unchanged scene soil scoring."""
import numpy as np
from scripts.validation.elastic_sponge_compression import quiet
PLAN=(('settle',800),('forward',4800),('side_step',1600),('backward',4800),('release',1200))
STEPS=sum(n for _,n in PLAN)


def center(metadata):
    return np.mean([r['center_world_m'][:2] for r in metadata['goals']['soil_cells'] if r['patch']=='soil_1'],axis=0)


def target(step,metadata):
    middle=center(metadata);start=middle+[-.08,-.019];begin=0
    for phase,count in PLAN:
        end=begin+count
        destination={'forward':middle+[.08,-.019],'side_step':middle+[.08,.019],'backward':middle+[-.08,.019]}.get(phase,start)
        if step<=end:
            u=max(0.,(step-begin)/count);duration=count*.0005
            return phase,start+(destination-start)*u*u*(3.-2*u),(destination-start)*6*u*(1-u)/duration
        start=destination;begin=end
    raise ValueError('Wiping step exceeds frozen plan')


def force(step,metadata,pose,velocity):
    phase,position,speed=target(step,metadata);f=np.zeros(3,np.float32)
    if phase not in ('settle','release'):f[:2]=np.clip(30.*(position-np.asarray(pose)[:2])+1.4*(speed-np.asarray(velocity)[:2]),-.4,.4)
    return f,phase,position,speed


def evaluate(raw,rows,masses,pad_index,table):
    if len(rows)!=STEPS+1 or not np.all(np.diff(raw['native_step'])==1):raise ValueError('Complete native bounded wiping trace required')
    checks,measure=quiet(raw,STEPS-800);f=np.asarray(raw['force_world_n'],float)
    checks.update(dt_matches=np.allclose(np.diff(raw['time_s']),float(np.float32(.0005)),atol=1e-10,rtol=0),
        bounded_horizontal_force=bool(np.max(abs(f[:,:2]))<=float(np.float32(.4)) and np.all(f[:,2]==0.)),
        first_patch_cleaned=rows[-1]['coverage']['patch_coverage']['soil_1']>=.60,
        untouched_third_patch_not_cleaned=rows[-1]['coverage']['patch_coverage']['soil_3']==0.,
        no_credit_at_reset=rows[0]['coverage']['coverage_fraction']==0.,
        load_not_through_table=min(r['body_bounds_m'][0][2] for r in rows)>=.899,
        real_final_support=all(table in r['pad_contact']['other_actor_paths'] and r['pad_contact']['net_contact_force_n'][2]>0 for r in rows[-801:]))
    net=np.sum(np.asarray(raw['body_contact_force_n'],float),axis=1);mean=net[-801:].mean(0);expected=float(sum(masses)*9.81)
    checks['recovered_actual_support_weight']=abs(mean[2]-expected)<=.15*expected
    q=np.asarray(raw['body_q'],float);travel=np.linalg.norm(q[:,pad_index,:2]-q[0,pad_index,:2],axis=1)
    measure.update(final_coverage=rows[-1]['coverage'],maximum_pad_excursion_m=float(travel.max()),mean_final_support_force_n=mean.tolist(),expected_weight_n=expected,
        scope='Bounded external pad force engineering diagnostic; soil credit is actual contact travel, never robot manipulation success.')
    return checks,measure,{'whole_sponge_support_force_n':net,'pad_excursion_m':travel}
