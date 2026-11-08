"""Stop further finger closure after measured local opposing cloth witnesses.

This pure planner changes no state and creates no attachment. Its one-shot
targets either reduce preload or hold the original command; CaptureGate must independently
observe actual joint stability and a continuous sampled 100ms material window.
"""
from __future__ import annotations

import copy
import hashlib
import numpy as np

from scripts.validation.laundry_material_topology import face_edge_neighbors, local_material_patch
from scripts.validation.laundry_closure_measurement import (
    validate_closure_measurement, measured_closure_requirement)


class ContactClosureStop:
    def __init__(self, side, triangles, rest_vertices, body_names, joint_names,
                 joint_limits, *, actuated_joint_names=None, preload_fraction=.2,
                 target_policy='reduce_preload_v1', geometry_profile='shared_edge_v1',
                 closure_measurement_profile='strict_v1', planned_open_tip_gap_m=None,
                 material_capture_region=None,contact_normal_profile='facet_v1',
                 contact_closure_observation_policy='close_only_v1',contact_witness_profile='centroid_v1'):
        from scripts.validation.laundry_contact_observation import validate_normal_profile,validate_observation_policy
        validate_normal_profile(contact_normal_profile)
        validate_observation_policy(contact_closure_observation_policy,True,target_policy)
        self.contact_normal_profile=contact_normal_profile
        from scripts.validation.laundry_contact_observation import validate_witness_profile
        validate_witness_profile(contact_witness_profile, contact_normal_profile)
        self.contact_witness_profile = contact_witness_profile
        self.observation_policy=contact_closure_observation_policy
        validate_closure_measurement(closure_measurement_profile, planned_open_tip_gap_m)
        self.closure_measurement_profile=closure_measurement_profile
        self.planned_open_tip_gap_m=planned_open_tip_gap_m
        if side not in ('left','right') or not np.isfinite(preload_fraction) or not 0<=preload_fraction<=.2:
            raise ValueError('Original side and bounded zero-to-20% preload required')
        if target_policy not in ('reduce_preload_v1','hold_command_v2'):
            raise ValueError('Unknown explicit contact target policy')
        if geometry_profile not in ('shared_edge_v1','adjacent_patch_v3','angled_adjacent_patch_v4'):
            raise ValueError('Unknown explicit closure geometry profile')
        self.geometry_profile=geometry_profile
        self.normal_dot_max=-.25 if geometry_profile=='angled_adjacent_patch_v4' else -.5
        self.side=side;self.tri=np.asarray(triangles,np.int64).copy()
        self.rest=np.asarray(rest_vertices,float).copy();self.bodies=list(body_names);self.joints=list(joint_names)
        if (self.tri.ndim!=2 or self.tri.shape[1]!=3 or self.rest.ndim!=2 or self.rest.shape[1]!=3
                or not np.isfinite(self.rest).all() or self.tri.min()<0 or self.tri.max()>=len(self.rest)):
            raise ValueError('Original finite material topology required')
        from scripts.validation.laundry_capture_region import MaterialCaptureRegion
        self.material_region = (None if material_capture_region is None else
                                MaterialCaptureRegion(material_capture_region,self.tri,self.rest))
        self.face_neighbors=face_edge_neighbors(self.tri) if geometry_profile!='shared_edge_v1' else None
        self.thumb=self.bodies.index(side+'_thumb_fingertip');self.index=self.bodies.index(side+'_index_fingertip')
        actuated=set(self.joints if actuated_joint_names is None else actuated_joint_names)
        self.fingers={n:i for i,n in enumerate(self.joints) if n in actuated and
                      (n.startswith(side+'_thumb_') or n.startswith(side+'_index_'))}
        if not self.fingers:raise ValueError('Original actuated thumb/index joints required')
        self.limits={n:tuple(map(float,joint_limits[n])) for n in self.fingers}
        if any(len(v)!=2 or not np.isfinite(v).all() or v[0]>=v[1] for v in self.limits.values()):
            raise ValueError('Finite original joint limits required')
        self.alpha=float(preload_fraction);self.target_policy=target_policy;self.receipt=None

    @property
    def latched(self):
        return self.receipt is not None

    def observe(self, time_s, diagnostic, robot_body_q, joint_q, commanded_joint_q,
                initial_open_tip_gap_m, *, cloth_q, allowed_material_vertex_ids=None,
                observation_phase='close'):
        if observation_phase not in (('close','closed_hold') if self.observation_policy=='close_and_hold_v1' else ('close',)):
            raise ValueError('Closure observation phase is outside the explicit policy')
        if self.latched:return copy.deepcopy(self.receipt)
        q=np.asarray(cloth_q,float);body=np.asarray(robot_body_q,float)
        actual=np.asarray(joint_q,float);target=np.asarray(commanded_joint_q,float)
        if (q.shape!=self.rest.shape or body.shape!=(len(self.bodies),7)
                or actual.shape!=(len(self.joints),) or target.shape!=actual.shape
                or not all(np.isfinite(a).all() for a in (q,body,actual,target))
                or not np.isfinite(time_s) or not np.isfinite(initial_open_tip_gap_m)):
            raise ValueError('Fresh finite native states and inventories required')
        reasons=[]
        identity={'body_q_float64_sha256':hashlib.sha256(body.astype('<f8').tobytes()).hexdigest(),
                  'cloth_q_float64_sha256':hashlib.sha256(q.astype('<f8').tobytes()).hexdigest()}
        if diagnostic.get('state_identity')!=identity:reasons.append('stale_geometry')
        if diagnostic.get('contact_normal_profile','facet_v1')!=self.contact_normal_profile:
            reasons.append('contact_normal_profile_mismatch')
        if diagnostic.get('contact_witness_profile','centroid_v1')!=self.contact_witness_profile:
            reasons.append('contact_witness_profile_mismatch')
        if (diagnostic.get('side')!=self.side or diagnostic.get('contact_force_measured') is not False
                or diagnostic.get('distal_group_severe_penetration') is not False):
            reasons.append('invalid_or_deeply_penetrating_distal_geometry')
        expected_profile=self.geometry_profile if self.geometry_profile=='angled_adjacent_patch_v4' else None
        if diagnostic.get('engineering_geometry_profile')!=expected_profile:
            reasons.append('engineering_geometry_profile_mismatch')
        gap=float(np.linalg.norm(body[self.thumb,:3]-body[self.index,:3]))
        closing=float(initial_open_tip_gap_m)-gap
        measurement=measured_closure_requirement(self.closure_measurement_profile,
            initial_open_tip_gap_m,self.planned_open_tip_gap_m)
        if not measurement['open_baseline_valid']:
            reasons.append('narrow_opening_baseline_outside_20_to_30mm')
        if closing<measurement['minimum_required_closure_m']:
            reasons.append('insufficient_measured_closure')
        allowed=None if allowed_material_vertex_ids is None else set(map(int,allowed_material_vertex_ids))
        candidates=[]
        diameter=lambda a:float(np.linalg.norm(a[:,None]-a[None,:],axis=2).max())
        for variant in [diagnostic]+diagnostic.get('distal_contact_diagnostics',[]):
            pair=variant.get('surface_pair',{});multi=variant.get('multilayer_diagnostic',{})
            if (set(pair)!={'thumb','index'} or any(v not in ('elastomer','DP') for v in pair.values())
                    or variant.get('severe_penetration_pads') or not multi.get('plausible')
                    or variant.get('contact_normal_profile','facet_v1')!=self.contact_normal_profile
                    or variant.get('contact_witness_profile','centroid_v1')!=self.contact_witness_profile
                    or variant.get('engineering_geometry_profile')!=expected_profile):continue
            for witness in multi.get('paired_witnesses',[]):
                a,b=int(witness['thumb']['triangle_id']),int(witness['index']['triangle_id'])
                if a==b or min(a,b)<0 or max(a,b)>=len(self.tri):continue
                topology=local_material_patch(self.tri,a,b,face_neighbors=self.face_neighbors)
                if topology is None:continue
                ids=np.unique(self.tri[topology['support_triangle_ids']])
                if len(ids)!=(4 if topology['relation']=='shared_edge' else 5):continue
                region = (None if self.material_region is None else self.material_region.check(
                    witness,topology,q,allowed_material_vertex_ids))
                if region is not None:
                    if not region['eligible']:continue
                elif allowed is not None and not set(map(int,ids))<=allowed:continue
                world_diameter=diameter(q[ids]);material_diameter=diameter(self.rest[ids])
                if (world_diameter>.035 or material_diameter>.035
                        or not np.isfinite(witness['opposing_normal_dot'])
                        or float(witness['opposing_normal_dot'])>self.normal_dot_max):continue
                candidates.append({'vertex_ids':ids.tolist(),'triangle_ids':[a,b],
                    'surface_pair':dict(pair),'witnesses':copy.deepcopy(witness),
                    'world_diameter_m':world_diameter,'material_diameter_m':material_diameter})
                if self.geometry_profile in ('adjacent_patch_v3','angled_adjacent_patch_v4'):
                    candidates[-1]['material_topology']=topology
                if region is not None:candidates[-1]['material_region']=region
        if not candidates:reasons.append('no_shared_or_single_bridge_local_opposing_witness'
            if self.geometry_profile in ('adjacent_patch_v3','angled_adjacent_patch_v4') else 'no_shared_edge_local_opposing_witness')
        result={'schema':'laundry_contact_closure_stop.v1','stop_requested':not reasons,
            'latched':False,'finger_joint_targets_rad':{},'reasons':reasons,
            'evidence':{'time_s':float(time_s),'actual_tip_gap_m':gap,
                'measured_tip_closure_m':closing,'initial_open_tip_gap_m':float(initial_open_tip_gap_m),
                'state_identity':identity},'attachment_enabled':False,'cloth_state_written':False,
            'capture_candidate':False,'friction_grasp_qualified':False,
            'scope':'One-shot original finger position targets only; actual stable capture gate remains mandatory.'}
        if self.geometry_profile in ('adjacent_patch_v3','angled_adjacent_patch_v4'):
            result.update(schema=('laundry_contact_closure_stop.v4' if self.geometry_profile=='angled_adjacent_patch_v4'
                                  else 'laundry_contact_closure_stop.v3'),geometry_profile=self.geometry_profile)
            if self.geometry_profile=='angled_adjacent_patch_v4':
                result['maximum_opposing_normal_dot']=self.normal_dot_max
        if self.closure_measurement_profile!='strict_v1':
            result.update(closure_measurement_profile=self.closure_measurement_profile,
                          closure_measurement=measurement)
        if self.material_region is not None:result['material_region_profile']='region_support_v1'
        if self.contact_normal_profile!='facet_v1':result['contact_normal_profile']=self.contact_normal_profile
        if self.contact_witness_profile!='centroid_v1':result['contact_witness_profile']=self.contact_witness_profile
        if self.observation_policy!='close_only_v1':
            result.update(contact_closure_observation_policy=self.observation_policy,observation_phase=observation_phase)
        if reasons:return result
        chosen=min(candidates,key=lambda c:(c['world_diameter_m'],c['triangle_ids'],str(c['surface_pair'])))
        outputs={};joints={}
        for name,index in self.fingers.items():
            lo,hi=self.limits[name]
            if not lo<=target[index]<=hi:
                raise ValueError('Original requested finger target exceeds original limits')
            # Freezing the command prevents additional planned closure without
            # backing the target away from its existing loaded position.
            value=(float(target[index]) if self.target_policy=='hold_command_v2' else
                   float(np.clip(actual[index]+self.alpha*(target[index]-actual[index]),lo,hi)))
            outputs[name]=value;joints[name]={'actual_rad':float(actual[index]),
                'previous_command_rad':float(target[index]),'stopped_command_rad':value,
                'original_limits_rad':[lo,hi]}
        result.update(latched=True,finger_joint_targets_rad=outputs)
        result['evidence'].update(chosen,preload_fraction=self.alpha,finger_joints=joints,
            further_closure_stopped=True,normal_force_measured=False,
            sampled_material_stability_proven=False)
        if self.target_policy=='hold_command_v2':
            if self.geometry_profile=='shared_edge_v1':result['schema']='laundry_contact_closure_stop.v2'
            result['evidence'].update(target_policy=self.target_policy,preload_fraction=1.,
                original_position_commands_held_exactly=True,drive_stiffness_changed=False)
        if self.observation_policy!='close_only_v1':
            result['evidence']['existing_closed_hold_command_frozen_unchanged']=(observation_phase=='closed_hold')
        self.receipt=copy.deepcopy(result)
        return result
