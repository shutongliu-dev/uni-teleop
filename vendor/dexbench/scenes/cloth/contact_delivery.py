"""Pure mandatory validation of the recorded once-per-substep cloth contact delivery.

Missing legacy delivery evidence is unqualified; nothing is synthesized.
No simulator, controller, private observation projection, or scoring changes.
"""
import math
import numpy as np
from bench.simulation.isaacsim_backend.contact_sample import PROTOCOL

def require(condition,message):
    if not condition:raise ValueError(message)

def exact_int(value,expected,name):require(type(value) is int and value==expected,'Invalid '+name)

def finite_equal(a,b,name):
    require(type(a) in (float,int) and type(b) in (float,int) and math.isfinite(a) and math.isfinite(b) and a==b,'Invalid '+name)

def validate_delivery(seal,trace,states,dt):
    require(seal.get('schema')=='cloth_agent_substep_physics_observer_v1' and seal.get('status')=='passed','Observer not passed')
    require(type(dt) in (float,int) and math.isfinite(dt) and abs(dt-.0005)<1e-10,'Unexpected native dt')
    delivery=seal.get('contact_delivery');require(isinstance(delivery,dict),'Missing contact_delivery')
    require(delivery.get('protocol')==PROTOCOL,'Wrong contact protocol')
    require(delivery.get('all_samples_consumed_once') is True,'Incomplete contact delivery')
    for field in ('raw_arrays_reused_across_native_steps','state_captures_reused'):require(delivery.get(field) is False,'Unexpected '+field)
    require(isinstance(trace,list) and len(trace)>=1,'Missing observer trace')
    n=len(trace)-1
    exact_int(seal.get('existing_steps_observed'),n,'observed steps')
    exact_int(delivery.get('acquisitions'),n,'acquisitions');exact_int(delivery.get('deliveries'),n,'deliveries')
    epoch=delivery.get('reset_epoch');require(type(epoch) is int and epoch>=1,'Invalid reset epoch')
    rows=delivery.get('samples');require(isinstance(rows,list) and len(rows)==n,'Missing per-substep delivery ledger')
    for name in ('native_physics_step_index','native_world_time_s','time_s'):
        require(name in states and np.asarray(states[name]).shape==(n+1,) and np.isfinite(states[name]).all(),'Invalid captured '+name)
    for i,(row,observed) in enumerate(zip(rows,trace[1:]),1):
        require(isinstance(row,dict),'Invalid delivery row')
        for field,want in [('sequence',i),('reset_epoch',epoch),('acquisition_count',1),('delivery_count',1),('native_step_index',int(states['native_physics_step_index'][i]))]:exact_int(row.get(field),want,field)
        exact_int(observed.get('native_step_index'),row['native_step_index'],'trace step')
        exact_int(observed.get('observed_steps'),i,'observed step')
        exact_int(row['native_step_index'],int(states['native_physics_step_index'][0])+i,'ordered native step')
        finite_equal(row.get('native_time_s'),float(states['native_world_time_s'][i]),'captured native time')
        finite_equal(row['native_time_s'],observed.get('native_time_s'),'trace native time')
        finite_equal(row.get('public_time_before_s'),float(states['time_s'][i]),'captured public time')
        finite_equal(row['public_time_before_s'],observed.get('public_time_s_at_callback'),'trace public time')
        finite_equal(row.get('planned_public_time_s'),row['public_time_before_s']+dt,'planned public time')
        finite_equal(row.get('consumed_public_time_s'),row['planned_public_time_s'],'consumed public time')
        require(row.get('readonly_verified') is True and observed.get('readonly_preserved') is True,'Readonly sample not verified')
        require(observed.get('rigid_contact_getter_completed') is True,'Incomplete contact sample')
    return {'protocol':PROTOCOL,'status':'passed','native_substeps_verified':n,'exactly_one_acquisition_and_delivery_per_substep':True,'scope':'CPU verification of captured per-substep handoff evidence, not a new native measurement.'}
