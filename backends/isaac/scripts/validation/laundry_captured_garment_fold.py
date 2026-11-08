"""Recorded garment folds with actual fingertip capture and declared local aid.

Each new edge must be captured through original thumb/index joint drives before
any local constraint is enabled. Constraints follow measured wrists, with fixed
material offsets. This script does not qualify friction-only grasp, a policy,
or task success. The default run verifies only the first folding phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from bench.paths import load_json
from scripts.validation.laundry_contact_pinch import plan as pinch_plan, interpolate_posture
from scripts.validation.laundry_closure_timing import bind_closure_timing, closure_hand_posture
from scripts.validation.laundry_capture_options import validate_hand_options
from scripts.validation.laundry_pinch_slot import pinch_slot_translation,slot_entry_bounds
from scripts.validation.laundry_engineering_motion import minimum_jerk
from scripts.validation.laundry_capture_settings import phase_capture_settings, planned_capture_openings
from scripts.validation.laundry_pivot_reference import validate_pivot_reference_options, bind_pivot_reference
from scripts.validation.laundry_arc_options import phase_peak_heights, arc_height_report
from scripts.validation.laundry_closure_feed import (
    closure_wrists, preflight_edge_feed,
    observed_garment_offset, edge_feed_plan, ClosureFeedProgress)
from scripts.validation.laundry_release_hand import (
    release_hand_profile, release_target, interpolate_release_target,
    opening_postures as release_opening_postures, retain_released_targets)
from scripts.validation.laundry_phase_motions import (
    validate_phase_motions, phase_release_settings, phase_park_transition,
    phase_released_dwell, cuff_boundary_descriptions, observed_corner_bottom, CartesianParkTransition)
from scripts.validation.laundry_trousers_sequence import reference_phase_indices, reference_phase_index
from scripts.validation.laundry_prepare_timing import (
    BEFORE_PARK, AFTER_PARK, phase_prepare_timing, prepare_at_stage)
from scripts.validation.laundry_released_joint_plan import (
    OBSERVED_SLEEVE_PROFILE, phase_released_joint_plan, load_released_joint_plan,
    rebind_released_start, plan_observed_released_press)
from scripts.validation.laundry_single_garment_native import (
    DATASETS, PROJECT, NativeArrays, advance, dump, pin, robot_readback)


def _frame(task, points, inverse=False):
    from scenes.workrooms.laundry_room.garment_folding_assets import transform_fold_frame
    frame = task['asset_metadata']['design'].get('fold_coordinate_frame')
    return transform_fold_frame(np.asarray(points), frame, inverse=inverse) if frame else np.asarray(points).copy()


def phase_edges(task):
    """Only genuine garment boundaries; no interior stationary holding clamp."""
    if task['id'] == 'trousers_folding':
        return [[{'feature': 'waistband', 'material_x_m': -.15, 'inward_xy': [0., -1.]},
                 {'feature': 'left_cuff', 'material_x_m': -.143, 'inward_xy': [0., 1.]}],
                [{'feature': 'paired_cuffs', 'inward_xy': [0., 1.]}]]
    if task['id'] == 'shirt_folding':
        return [[{'feature': 'right_cuff', 'inward_xy': [-1., 0.]}],
                [{'feature': 'left_cuff', 'inward_xy': [1., 0.]}],
                [{'feature': 'lower_hem', 'material_x_quantile': .05, 'inward_xy': [0., 1.]},
                 {'feature': 'lower_hem', 'material_x_quantile': .95, 'inward_xy': [0., 1.]}]]
    raise ValueError('Expected an unfolded single shirt or trousers folding task')


def execution_phase_edges(task, options):
    original = phase_edges(task)
    sequence = reference_phase_indices(options, task['id'])
    return original if sequence is None else [original[index] for index in sequence]


def edge_pick(task, current_q, description):
    spec = task['asset_metadata']['cloths'][0]
    rest = np.asarray(spec['rest_vertices_m'], float)
    q = np.asarray(current_q, float)
    if q.shape != rest.shape or not np.isfinite(q).all():
        raise ValueError('Full actual native garment q required')
    features = spec['boundary_features']
    name = description['feature']
    ids = np.unique(np.concatenate([features['left_cuff'],features['right_cuff']])) if name == 'paired_cuffs' else np.asarray(features[name],int)
    corner_keys=[key for key in ('material_x_corner','material_y_corner') if key in description]
    if len(corner_keys)>1:raise ValueError('Select one material boundary axis for a corner')
    corner = description.get(corner_keys[0]) if corner_keys else None
    if corner:
        if corner not in ('minimum','maximum'):raise ValueError('Unknown material corner extreme')
        axis=0 if corner_keys[0]=='material_x_corner' else 1
        extreme = float(rest[ids,axis].min() if corner=='minimum' else rest[ids,axis].max())
        ids=ids[np.abs(rest[ids,axis]-extreme)<1.e-7]
    elif 'material_x_m' in description:
        # A fixed count can mix neighbouring X rows and silently move the
        # contact TCP by half a mesh cell. Preserve one exact material level,
        # including its front/back counterparts, and deterministic tie-break.
        levels=np.unique(rest[ids,0]);level=levels[np.argmin(abs(levels-description['material_x_m']))]
        ids=ids[abs(rest[ids,0]-level)<1.e-8]
    elif 'material_x_quantile' in description:
        x = np.quantile(rest[ids,0],description['material_x_quantile'])
        ids = ids[np.argsort(abs(rest[ids,0]-x),kind='stable')[:4]]
    else:
        center = np.median(q[ids,:2],axis=0)
        ids = ids[np.argsort(np.linalg.norm(q[ids,:2]-center,axis=1),kind='stable')[:4]]
    highest=int(ids[np.argmax(q[ids,2])]) if corner else None
    edge = q[highest].copy() if corner else np.median(q[ids],axis=0)
    material = rest[highest].copy() if corner else np.median(rest[ids],axis=0)
    # Opposite folded cuffs can be world-near but materially far. Retain the
    # union of each seed's material neighbourhood; CaptureGate still requires
    # its own small shared-edge patch and never welds the two cuffs together.
    allowed = np.flatnonzero(np.min(np.linalg.norm(rest[:,None]-rest[ids][None],axis=2),axis=1)<=.030)
    # Frame rotation only, not a fictitious change to the physical vertices.
    direction = np.r_[description['inward_xy'],0.]
    world_direction = _frame(task,np.array([np.zeros(3),direction]))
    inward = world_direction[1,:2]-world_direction[0,:2]
    if not task['asset_metadata']['design'].get('fold_coordinate_frame'):
        angle=int(task['asset_metadata'].get('source_yaw_quarter_turns',0))*np.pi/2
        c,s=np.cos(angle),np.sin(angle)
        inward=np.array([[c,-s],[s,c]])@inward
    inward /= np.linalg.norm(inward)
    return {'feature':name, 'seed_vertex_ids':ids.tolist(),'natural_corner_vertex_id':highest,
        'allowed_material_vertex_ids':allowed.tolist(), 'edge_world_m':edge.tolist(),
        'edge_material_m':material.tolist(), 'inward_world_xy':inward.tolist(),
        'pinch_tcp_xy_m':(edge[:2]+.006*inward).tolist()}


def selected_phase_edges(descriptions, configured):
    """An explicit boundary subset is a partial manipulation experiment."""
    selected=configured.get('edge_indices',list(range(len(descriptions))))
    if (not isinstance(selected,list) or not selected or
            any(type(i) is not int or not 0<=i<len(descriptions) for i in selected) or
            len(set(selected))!=len(selected)):
        raise ValueError('Explicit edge indices must select distinct existing boundaries')
    return [descriptions[i] for i in selected],selected


def prepare_phase(release, task, scene, current_q, index, options, *, only_sides=None):
    validate_hand_options(options)
    validate_phase_motions(options, task['id'])
    validate_pivot_reference_options(options, task['id'])
    from scripts.validation.laundry_capture_region import validate_region_option
    validate_region_option(options,task.get('id'),towel_adapter=False)
    reference_index = reference_phase_index(options, index, task['id'])
    descriptions = phase_edges(task)[reference_index]
    profile=options.get('pickup_profile','boundary_midpoint')
    if profile not in ('boundary_midpoint','outer_corners','waist_corner','cuff_corner','shirt_cuff_corners'):raise ValueError('Unknown explicit pickup profile')
    if profile in ('outer_corners','waist_corner','cuff_corner') and task['id']=='trousers_folding' and index==0:
        descriptions=[({'feature':'waistband','material_x_corner':'minimum','inward_xy':[1.,-1.]}
                       if profile!='cuff_corner' else descriptions[0]),
                      ({'feature':'left_cuff','material_x_corner':'minimum','inward_xy':[1.,1.]}
                       if profile!='waist_corner' else descriptions[1])]
    if profile=='shirt_cuff_corners':
        if task['id']!='shirt_folding':raise ValueError('Sleeve corners require a shirt folding task')
        if index<2:
            descriptions=[{'feature':'right_cuff' if index==0 else 'left_cuff',
                'material_y_corner':'minimum','inward_xy':[-1. if index==0 else 1.,1.]}]
    configured = options.get('phases',{}).get(str(index),{})
    descriptions=cuff_boundary_descriptions(task['id'],reference_index,descriptions,configured,pickup_profile=profile)
    total_edges=len(descriptions)
    descriptions,edge_indices=selected_phase_edges(descriptions,configured)
    picks = [edge_pick(task,current_q,d) for d in descriptions]
    if len(picks)==2:
        # Opposed installation: the northern pickup goes to the northern arm.
        order = np.argsort([p['edge_world_m'][1] for p in picks])
        sides = ['']*2;sides[int(order[0])]='right';sides[int(order[1])]='left'
    else:
        sides = ['left' if picks[0]['edge_world_m'][1] > -.55 else 'right']
        if task['id']=='trousers_folding' and index==1:sides=['left']
    sides = configured.get('sides',sides)
    if len(sides)!=len(picks) or len(set(sides))!=len(sides):
        raise ValueError('One distinct original hand per moving garment edge')
    table_z = float(scene['table']['position_m'][2]+scene['table']['half_extents_m'][2])
    for j,(pick,side) in enumerate(zip(picks,sides)):
        if only_sides is not None and side not in only_sides:continue
        override=configured.get('hands',{}).get(side,{})
        bottom_z,height_receipt=observed_corner_bottom(pick,override,table_z)
        if 'pinch_inward_offset_m' in override:
            pick.update(observed_garment_offset(task['asset_metadata']['cloths'][0],current_q,
                pick,descriptions[j]['inward_xy'],override['pinch_inward_offset_m'],
                **({'profile':override['pinch_tangent_profile']} if 'pinch_tangent_profile' in override else {})))
        feed=edge_feed_plan(pick,override)
        inward=np.asarray(pick['inward_world_xy']);sign=1. if side=='right' else -1.
        # R[:,1]=[sin(yaw),-cos(yaw),0] for roll=pi. Thumb is +Y on right, -Y on left.
        yaw=float(np.rad2deg(np.arctan2(sign*inward[0],-sign*inward[1])) % 360.)
        # The symmetric alternative puts the index inside the edge. Both
        # remain actual opposing fingers and require the identical native gate.
        angles=[override['yaw_deg']] if 'yaw_deg' in override else [yaw,(yaw+180.)%360.]
        failures=[];contact=None
        for angle in angles:
            try:
                arguments=(angle,
                    override.get('pitch_deg',60.),bottom_z,
                    override.get('close_distance_m',.010),override.get('closure_lift_m',0.),
                    override.get('capture_rise_m',0.),override.get('closure_lift_start_fraction',0.),
                    override.get('open_distance_m',.060 if task['asset_metadata']['cloths'][0]['physical_thickness_m']>=.0017 else .045))
                candidate=pinch_plan(release,side,pick['pinch_tcp_xy_m'],*arguments)
                if override.get('pinch_slot_offset_m',0.) != 0.:
                    from bench.control.factory import public_robot_geometry,robot_configuration
                    model=public_robot_geometry(robot_configuration(scene)).sides[side]
                    centering=pinch_slot_translation(model,side,candidate,override['pinch_slot_offset_m'])
                    centering['combined_entry']=slot_entry_bounds(pick,centering,feed)
                    original_xy=np.asarray(pick['pinch_tcp_xy_m'],dtype=float)
                    shifted_xy=(original_xy+np.asarray(centering['translation_world_m'])[:2]).tolist()
                    # A translated cached pose or seed is not a new IK solution.
                    candidate=pinch_plan(release,side,shifted_xy,*arguments)
                    centering.update(original_pinch_tcp_xy_m=original_xy.tolist(),
                                     commanded_pinch_tcp_xy_m=shifted_xy,approach_ik_replanned=True)
                    candidate['slot_centering']=centering
                    pick['pinch_tcp_xy_m']=shifted_xy
                contact=candidate
                contact['inside_finger']='thumb' if np.cos(np.deg2rad(angle-yaw))>0. else 'index'
                break
            except ValueError as error:failures.append({'yaw_deg':angle,'message':str(error)})
        if contact is None:raise ValueError('No original-hand edge approach IK: '+json.dumps(failures))
        if height_receipt is not None:contact['observed_corner_hand_height']=height_receipt
        if feed is not None:contact['closure_feed']=feed
        bind_closure_timing(contact,override)
        pick['rejected_approach_candidates']=failures
        pick.update(side=side,contact_plan=contact,edge_index=edge_indices[j])
        pivot_reference = bind_pivot_reference(task, pick, configured)
        if pivot_reference is not None:pick['pivot_material_reference'] = pivot_reference
    result={'phase_index':index,'pickup_profile':profile,
        'selected_edge_indices':edge_indices,'all_phase_edges_selected':len(edge_indices)==total_edges,
        'edges':[p for p in picks if 'contact_plan' in p],
        'observed_q_sha256':hashlib.sha256(np.asarray(current_q,dtype='<f8').tobytes()).hexdigest(),
        'reference_fold':task['asset_metadata']['design']['fold_sequence_axis_hinge_sign_radius_m'][reference_index],
        'scope':'Boundary pickup and discrete approach IK plan; no capture/collision/task qualification.'}
    if reference_phase_indices(options, task['id']) is not None:
        result.update(execution_phase_index=index, original_reference_phase_index=reference_index,
            trousers_phase_sequence=options['trousers_phase_sequence'], public_fold_goal_changed=False)
    if 'folded_cuff_boundary' in configured:
        result['observed_boundary_override']={'profile':configured['folded_cuff_boundary'],
            'description':descriptions[0], 'source':'Original named cuff boundary and fresh corner; no remote panel membership added'}
    elif 'sleeve_cuff_corner' in configured:
        result['observed_boundary_override']={'profile':'sleeve_cuff_'+configured['sleeve_cuff_corner'],
            'description':descriptions[0], 'source':'Original named sleeve cuff extreme and fresh corner; material IDs unchanged'}
    return result


def captured_arc(task, phase, cloth_q, captured_ids, *, release_height=.025, peak=.12):
    """Reflect the captured material, leaving a free fall above the target layers."""
    if not .02<=release_height<=.03 or not .04<=peak<=.15:
        raise ValueError('Use20–30mm free release and40–150mm slow arc')
    q=_frame(task,np.asarray(cloth_q,float),inverse=True)
    ref=_frame(task,np.asarray(task['asset_metadata']['cloths'][0]['unfolded_reference_vertices_m']),inverse=True)
    axis,hinge,sign,_=phase['reference_fold'];axis=int(axis)
    stationary=sign*(ref[:,axis]-hinge)<-.04
    if not np.any(stationary):raise ValueError('No stationary material to locate the observed fold line')
    hinge=float(hinge+np.median(q[stationary,axis]-ref[stationary,axis]))
    ids=np.asarray(captured_ids,int);start=q[ids].mean(0);end=start.copy()
    thickness=float(task['asset_metadata']['cloths'][0]['physical_thickness_m'])
    end[axis]=2*hinge-start[axis]+sign*2.5*thickness
    support=np.flatnonzero(np.linalg.norm(q[:,:2]-end[:2],axis=1)<.035)
    support=np.setdiff1d(support,ids)
    if not len(support):raise ValueError('No actual garment support at reflected target')
    support_z=float(np.quantile(q[support,2],.9))
    if support_z>.99:raise ValueError('Observed target layers are not table-supported')
    end[2]=support_z+release_height
    middle=(start+end)/2;middle[2]=max(start[2],end[2])+2*peak
    return {'controls_m':_frame(task,np.asarray([start,middle,end])).tolist(),
        'captured_vertex_ids':ids.tolist(),'observed_hinge_m':hinge,
        'observed_target_support_z_m':support_z,'release_height_m':release_height,
        'actual_wrist_orientation':'fixed; only actual joint-driven wrist translation transports captured offsets',
        'free_cloth_state_written':False}


def bezier_position(controls, amount):
    a,b,c=np.asarray(controls,float);s=float(amount)
    return (1-s)**2*a+2*s*(1-s)*b+s*s*c


def capture_lift_arc(cloth_q, captured_ids):
    """A measured 5cm pickup diagnostic, without a task-fold target."""
    ids=np.asarray(captured_ids,int);q=np.asarray(cloth_q,float)
    if not 2<=len(ids)<=32 or len(set(map(int,ids)))!=len(ids) or not np.isfinite(q).all():
        raise ValueError('Measured local capture patch required for lift probe')
    start=q[ids].mean(0);end=start+[0.,0.,.05]
    return {'controls_m':np.array([start,(start+end)/2,end]).tolist(),
        'captured_vertex_ids':ids.tolist(),'kind':'capture_lift_probe','lift_height_m':.05,
        'free_cloth_state_written':False,'scope':'Local captured-material pickup and release; no task fold attempted.'}


def planned_duration(count, options, final_free):
    hand_profile=release_hand_profile(options)
    validate_phase_motions(options)
    total=.5+final_free
    for i in range(count):
        phase=options.get('phases',{}).get(str(i),{});order=phase.get('capture_order')
        hands=phase.get('hands',{})
        if order is not None:
            if not order or len(set(order))!=len(order) or not set(order)<={'left','right'}:
                raise ValueError('Explicit capture order must contain distinct original sides')
            total+=16.5+6.5*(len(order)-1)+sum(bool(hands.get(s,{}).get('capture_rise_m',0.)) for s in order)
        else:total+=16.5+float(any(h.get('capture_rise_m',0.) for h in hands.values()))
    if not any('release_settings' in options.get('phases',{}).get(str(i),{}) for i in range(count)):
        if options.get('retreat_profile','base_directed_v1')=='finger_axis_exit_v2':total+=count
        if hand_profile in ('distal_extend_v1','full_spread_extend_v2'):total+=count
    else:
        for i in range(count):
            settings=phase_release_settings(options,i)
            total+=int(settings['retreat_profile']=='finger_axis_exit_v2')
            total+=int(settings['release_hand_profile'] in ('distal_extend_v1','full_spread_extend_v2'))
    for i in range(count):
        transition=phase_park_transition(options,i)
        if transition is not None:total+=transition['duration_s']
        total+=phase_released_dwell(options,i)
        released_plan=phase_released_joint_plan(options,i)
        if released_plan is not None:total+=released_plan['duration_s']
    return total


def capture_batches(phase, options):
    sides=[edge['side'] for edge in phase['edges']]
    order=options.get('phases',{}).get(str(phase['phase_index']),{}).get('capture_order')
    if order is None:return [sides]
    if len(order)!=len(sides) or set(order)!=set(sides):
        raise ValueError('Capture order must cover each intended hand exactly once')
    return [[side] for side in order]


def capture_approach(contact_plan, settings):
    offset=np.asarray(settings.get('hover_offset_m',[0.,0.,.10]),float)
    if (offset.shape!=(3,) or not np.isfinite(offset).all() or not .01<=offset[2]<=.20
            or np.linalg.norm(offset[:2])>.20 or np.linalg.norm(offset)>.25):
        raise ValueError('Bounded finite overhead approach offset required')
    original=contact_plan['arm_ik'][0]['joint_positions_rad']
    seed=settings.get('hover_arm_joint_positions_rad',original)
    if set(seed)!=set(original) or not all(np.isfinite(float(v)) for v in seed.values()):
        raise ValueError('Hover seed must name exactly this original arm joint inventory')
    return {'offset_m':offset.tolist(),'arm_joint_positions_rad':{n:float(v) for n,v in seed.items()}}


def rebound_descent_wrists(previous_hover, fresh_contact, amount):
    """Keep the reached hover; interpolate XYZ to newly observed cloth only."""
    if (not previous_hover or set(previous_hover)!=set(fresh_contact)
            or not np.isfinite(amount) or not 0.<=amount<=1.):
        raise ValueError('Matching current-batch wrists and bounded descent fraction required')
    result={}
    for side in previous_hover:
        start,end=np.asarray(previous_hover[side],float),np.asarray(fresh_contact[side],float)
        if (start.shape!=(4,4) or end.shape!=(4,4) or not np.isfinite([start,end]).all()
                or not np.allclose(start[3],[0.,0.,0.,1.],atol=1e-12,rtol=0.)
                or not np.allclose(end[3],[0.,0.,0.,1.],atol=1e-12,rtol=0.)
                or not np.allclose(start[:3,:3],end[:3,:3],atol=1e-8,rtol=0.)):
            raise ValueError('Cloth rebind must preserve the original wrist orientation')
        pose=start.copy();pose[:3,3]+=amount*(end[:3,3]-start[:3,3]);result[side]=pose
    return result


def preflight_rebound_descent(previous_hover, fresh_contact, seed, solve_targets):
    """Read-only 76-point original-arm IK continuation before any new descent."""
    preview=dict(seed);samples=[]
    for index in range(76):
        time_s=index*.02;amount=minimum_jerk(time_s,0.,1.5)
        wrists=rebound_descent_wrists(previous_hover,fresh_contact,amount)
        targets=solve_targets(wrists,dict(preview))
        if set(targets)!=set(seed) or not np.isfinite(list(targets.values())).all():
            raise ValueError('Preflight changed the original finite joint inventory')
        preview=dict(targets)
        samples.append({'time_s':time_s,'fraction':float(amount),
            'wrist_world_matrices':{s:m.tolist() for s,m in wrists.items()},
            'joint_targets_rad':dict(preview)})
    return {'schema':'laundry_rebound_descent_preflight.v1','sample_count':len(samples),
        'samples':samples,'scope':'Original IK continuation at20ms; no native step, state write, contact or collision qualification.'}


def actual_ik_frames(body_q, body_names, models, sides):
    """Read the public IK frame; attachment hand_wrist is a separate frame."""
    from bench.evaluation.cloth.cloth_robot_collision import transform_xyzw
    return {side:transform_xyzw(np.asarray(body_q)[body_names.index(models[side].frame_id)])
            for side in sides}


def retreat_translation(model, flange_world, amount):
    """Withdraw toward the installed arm base, including same-side layouts."""
    direction=np.asarray(model.arm_base)[:2,3]-np.asarray(flange_world)[:2,3]
    length=float(np.linalg.norm(direction))
    if length<1.e-6:raise ValueError('No horizontal base direction for withdrawal')
    return np.r_[.05*direction/length,.06]*float(amount)


def finger_axis_exit(body_q, body_names, side):
    """Withdraw from a fully opened original distal index, then lift separately."""
    q=np.asarray(body_q,float)
    if side not in ('left','right') or q.shape!=(len(body_names),7) or not np.isfinite(q).all():
        raise ValueError('Finite actual original hand body inventory required')
    tip=q[body_names.index(side+'_index_fingertip'),:3]
    distal=q[body_names.index(side+'_index_DP'),:3]
    axis=tip-distal;length=float(np.linalg.norm(axis))
    if not .005<=length<=.08:raise ValueError('Expected actual original distal index length')
    delta=-.05*axis/length
    if delta[2]<-1.e-6:
        raise ValueError('This actual finger-axis withdrawal points into the table')
    return {'translation_m':delta.tolist(),'actual_index_dp_position_m':distal.tolist(),
        'actual_index_tip_position_m':tip.tolist(),'distance_m':.05,
        'scope':'Measured distal-finger withdrawal direction; native contact and separation remain independently checked.'}


def independent_cloth_streams(moving_id,positions,velocities):
    """Additional named native cloths retain independent positions/velocities."""
    if set(positions)!=set(velocities) or moving_id not in positions:
        raise ValueError('Matching complete independent cloth inventories required')
    result={}
    for name,q in positions.items():
        if name==moving_id:continue
        if not name.replace('_','').isalnum() or np.asarray(q).shape!=np.asarray(velocities[name]).shape:
            raise ValueError('Invalid independent cloth stream')
        result.update({'cloth_q_'+name:q,'cloth_qd_'+name:velocities[name],
                       'control_mask_'+name:np.zeros(len(q),np.uint8)})
    return result


def close_native_with_status(env, exit_code):
    """Keep failure active while Kit closes so shutdown cannot report success."""
    if exit_code:
        try:raise RuntimeError('Captured manipulation failed; original evidence remains in native_report.json')
        finally:env.close()
    env.close()


def command_joint_targets(env, values, assists=(), phase=None):
    """Send original joint positions, preserving any measured finger stop."""
    if any(not env.joint_limits[n][0]<=values[n]<=env.joint_limits[n][1] for n in values):
        raise ValueError('Original joint limit exceeded')
    values=dict(values)
    for assist in assists:
        held=assist.held_joint_targets(phase)
        if any(n not in env.actuated_joint_names or not env.joint_limits[n][0]<=v<=env.joint_limits[n][1]
               for n,v in held.items()):
            raise ValueError('Contact stop may only change bounded original actuated positions')
        values.update(held)
    expanded=env._expand_mimic_targets({n:float(values[n]) for n in env.actuated_joint_names})
    env._targets=np.asarray([[expanded[n] for n in env.joint_names]],np.float32)
    env._robot_view.set_dof_position_targets(env._tensor(env._targets),env._one)
    return {**values,**expanded}


def opening_capture_targets(assists, phase, start_targets, planned_targets, amount, time_s,
                            *, hand_profile='pinch_open_v1', release_targets=None):
    """Disable assistance before the first opening drive command is submitted."""
    if hand_profile=='full_spread_extend_v2' and (release_targets is None or
            any(n not in release_targets for e in phase['edges']
                for n in e['contact_plan']['postures'][0]['hand_posture_rad'])):
        raise ValueError('Full-spread opening requires the same complete previewed endpoint')
    result=dict(planned_targets)
    for assist in assists:
        assist.release(time_s, 'Opening phase releases the local constraint before stepping.')
        edge=next(e for e in phase['edges'] if e['side']==assist.side)
        postures=edge['contact_plan']['postures']
        if hand_profile in ('distal_extend_v1','full_spread_extend_v2'):
            result.update(interpolate_release_target(start_targets,
                (release_target(assist.side,postures,hand_profile) if hand_profile=='distal_extend_v1'
                 else {n:release_targets[n] for n in postures[0]['hand_posture_rad']}),amount))
        elif hand_profile=='pinch_open_v1':
            result.update(assist.opening_joint_targets(postures,start_targets,amount))
        else:raise ValueError('Unknown explicit release hand profile')
    return result


def native(args, *, task_adapter=None):
    from scripts.maintenance.freeze_project_sources import verify_if_snapshot
    entry=Path(task_adapter.__file__) if task_adapter else Path(__file__)
    required=task_adapter.MODULE_NAME if task_adapter else 'scripts.validation.laundry_captured_garment_fold'
    source=verify_if_snapshot(PROJECT,required_modules=(required,),entrypoint_file=entry)
    if source is None:raise ValueError('Use a complete frozen source snapshot')
    from bench.isaacsim.release import validate
    release=args.release.resolve();manifest=validate(release)
    task,scene,initial=[load_json(release/n) for n in ('task.json','scene.json','initial_state.json')]
    if manifest['task_id']!=task['id']:raise ValueError('Validated release task identity mismatch')
    phase_definitions=task_adapter.phase_edges if task_adapter else phase_edges
    prepare=task_adapter.prepare_phase if task_adapter else prepare_phase
    if task_adapter:task_adapter.validate_scene(task,scene)
    if scene.get('named_cloth_initialization')!='author-only' or (task_adapter is None and len(scene.get('cloths',[]))!=1):
        raise ValueError('One authored original garment, no native cloth reset setters')
    if scene['robot']['hand_id']!='dual_fr3_sharpa_w01':raise ValueError('Initial runner requires the actual dual original hands')
    options=load_json(args.options) if args.options else {}
    if 'trousers_phase_sequence' in options and task_adapter is not None:
        raise ValueError('Explicit sequential cuffs require the original garment runner')
    if ('flatten_phase_sequence' in options and getattr(task_adapter,'MODULE_NAME',None)
            != 'scripts.validation.laundry_captured_towel_tasks'):
        raise ValueError('Explicit corner unfolding requires the towel adapter')
    from scripts.validation.laundry_capture_region import validate_region_option
    validate_region_option(options,task['id'],towel_adapter=(getattr(task_adapter,'MODULE_NAME',None)==
        'scripts.validation.laundry_captured_towel_tasks'))
    validate_hand_options(options, towel=getattr(task_adapter, 'MODULE_NAME', None)
                          == 'scripts.validation.laundry_captured_towel_tasks')
    validate_pivot_reference_options(options, task['id'], garment_adapter=task_adapter is None)
    validate_phase_motions(options,task['id'],towel=getattr(task_adapter,'MODULE_NAME',None)==
        'scripts.validation.laundry_captured_towel_tasks')
    if task_adapter is not None and any({'folded_cuff_boundary','sleeve_cuff_corner'} & set(p)
            for p in options.get('phases',{}).values()):
        raise ValueError('Explicit garment boundary overrides require the original garment runner')
    if task_adapter is not None and hasattr(task_adapter, 'validate_options'):
        task_adapter.validate_options(task, options)
    if task_adapter is not None and hasattr(task_adapter, 'execution_phase_edges'):
        phase_definitions=lambda task: task_adapter.execution_phase_edges(task,options)
    elif task_adapter is None:
        phase_definitions=lambda task: execution_phase_edges(task,options)
    phase_capture_settings(options)  # Validate the legacy global controls before startup.
    rebind_before_descent=options.get('rebind_before_descent',False)
    if type(rebind_before_descent) is not bool:
        raise ValueError('rebind_before_descent must be an explicit boolean')
    transport_mode=options.get('transport_mode','fold')
    if transport_mode not in ('fold','capture_lift_probe'):raise ValueError('Unknown recorded transport mode')
    release_profile=options.get('release_profile','fixed_v1')
    hand_profile=release_hand_profile(options)
    if release_profile not in ('fixed_v1','cooked_hand_clearance_v2','cooked_opening_lift_v3'):
        raise ValueError('Unknown explicit cloth release placement profile')
    if transport_mode!='fold' and release_profile!='fixed_v1':
        raise ValueError('The lift diagnostic retains its original release trajectory')
    retreat_profile=options.get('retreat_profile','base_directed_v1')
    if retreat_profile not in ('base_directed_v1','finger_axis_exit_v2'):
        raise ValueError('Unknown explicit release retreat profile')
    count=args.max_phases
    if transport_mode=='capture_lift_probe' and count!=1:raise ValueError('Lift probe contains one capture phase')
    if not 1<=count<=len(phase_definitions(task)):raise ValueError('Select a positive available phase count')
    peak_heights=phase_peak_heights(options,count,args.peak_height)
    capture_configurations=[phase_capture_settings(options,i) for i in range(count)]
    release_configurations=[phase_release_settings(options,i) for i in range(count)]
    park_transitions=[phase_park_transition(options,i) for i in range(count)]
    preparation_timings=[phase_prepare_timing(options,i) for i in range(count)]
    released_dwells=[phase_released_dwell(options,i) for i in range(count)]
    released_joint_plans=[phase_released_joint_plan(options,i) for i in range(count)]
    released_joint_rows=[None if setting is None else load_released_joint_plan(setting,
        {'scene.json':scene,'task.json':task,'initial_state.json':initial},list(initial['joint_positions_rad']))
        for setting in released_joint_plans]
    planned_openings=[planned_capture_openings(options,i,settings)
                      for i,settings in enumerate(capture_configurations)]
    if (not np.isfinite(args.final_free) or not .5<=args.final_free<=30.
            or not np.isfinite(args.peak_height) or not .04<=args.peak_height<=.15
            or not 1<=args.media_fps<=30):
        raise ValueError('Finite final-free0.5–30s, peak40–150mm, and media1–30fps required')
    out=args.output.resolve()
    if not out.is_relative_to(DATASETS) or out.exists():raise ValueError('Fresh shared dataset output required')
    out.mkdir(parents=True);dt=float(scene['physics']['dt_s'])
    report={'schema':getattr(task_adapter,'REPORT_SCHEMA','laundry_captured_towel_task.v1') if task_adapter else 'laundry_captured_garment_fold.v1','complete':False,'source':source,
        'source_manifest':pin(PROJECT/'source_snapshot_manifest.json'),
        'release_manifest_validation':'Complete source and asset inventory validation before native initialization.',
        'inputs':{n:pin(release/n) for n in ('task.json','scene.json','initial_state.json','release_manifest.json')},
        'options':options,'transport_mode':transport_mode,'requested_phases':count,'phase_plans':[], 'captured_arcs':[], 'action_events':[],
        'declared_local_capture_assistance':True,'collision_filtering':False,
        'cloth_position_velocity_writes':0,'task_success_claimed':False,'friction_grasp_qualified':False,
        'scope':'Scripted native captured-assistance folding diagnostic; partial phases are explicitly identified.'}
    report.update(arc_height_report(options,peak_heights))
    if any('capture_settings' in options.get('phases',{}).get(str(i),{}) for i in range(count)):
        report['phase_capture_settings']=[{'phase_index':i,**settings} for i,settings in enumerate(capture_configurations)]
    if any('release_settings' in options.get('phases',{}).get(str(i),{}) for i in range(count)):
        report['phase_release_settings']=[{'phase_index':i,**settings} for i,settings in enumerate(release_configurations)]
    env=writer=capture=contacts=None;assists=[];current=[];exit_code=1
    # Each phase has park2s, open/hover2s, descend1.5s, close2s, optional rise1s,
    # closed-hold1s, fold5s, open1s (explicit distal extension2s), retreat1s and passive1s.
    duration=planned_duration(count,options,args.final_free)
    steps=round(duration/dt)
    if abs(steps*dt-duration)>1.e-8:raise ValueError('Requested durations must land on native time steps')
    try:
        from bench.control.factory import public_robot_geometry,robot_configuration
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv,_np
        from bench.evaluation.cloth.cloth_robot_collision import collect_colliders,transform_xyzw
        from scripts.validation.laundry_contact_assist import CapturedPatchAssist
        from scripts.validation.laundry_material_settle import mesh_data,usd_material_readback
        env=FR3DexterousIsaacEnv(scene);env.reset(initial_state=initial)
        models=public_robot_geometry(robot_configuration(scene)).sides
        state=robot_readback(env);park={n:float(np.clip(v,*env.joint_limits[n])) for n,v in zip(env.joint_names,state['joint_q'])};seed=dict(park)
        spec=next(c for c in scene['cloths'] if c['id']=='towel');view=env._cloth_views[spec['id']]
        cloth=lambda:_np(view.get_simulation_nodal_positions())[0]
        all_cloths=lambda:{name:_np(v.get_simulation_nodal_positions())[0] for name,v in env._cloth_views.items()}
        all_velocities=lambda:{name:_np(v.get_simulation_nodal_velocities())[0] for name,v in env._cloth_views.items()}
        report.update(joint_names=env.joint_names,body_names=env.body_names,native_initialization=env.named_cloth_initialization_receipt,
            actual_control='Original native joint position drives and gravity feedforward; original thumb/index joints close and open.',
            usd_material_readback=usd_material_readback(env,spec,mesh_data(spec,task)))
        report['all_cloth_material_readback']={c['id']:usd_material_readback(env,c,mesh_data(c,task)) for c in scene['cloths']}
        report['cloth_stream_fields']={c['id']:{'q':'cloth_q'+('' if c['id']=='towel' else '_'+c['id']),
            'qd':'cloth_qd'+('' if c['id']=='towel' else '_'+c['id']),
            'mask':'control_mask'+('' if c['id']=='towel' else '_'+c['id'])} for c in scene['cloths']}
        export=collect_colliders(env);dump(out/'robot_colliders.json',export);report['robot_colliders']=pin(out/'robot_colliders.json')
        env.stage.Export(str(out/'initial.usda'));report['initial_stage']=pin(out/'initial.usda')
        writer=NativeArrays(out/'states',steps+1);contacts=(out/'rigid_contacts.jsonl').open('x');env._begin_contact_sampling()
        start_step,start_time=env.world.current_time_step_index,env.world.current_time
        step,phase_index,action_index,pairs,impulse=0,-1,-1,0,0.;action='settle';heartbeat=time.monotonic();capture_sides=set()
        if args.media_output:
            from scripts.media.laundry_cloth_frames import LaundryClothFrameCapture
            capture=LaundryClothFrameCapture(stage=env.stage,world=env.world,app=env.app,
                cloth_views=env._cloth_views,cloth_specs=scene['cloths'],triangles=env.cloth_triangles,
                cloth_paths={k:v['path'] for k,v in env.scene_info['cloths'].items()},
                output=args.media_output,cameras={k:scene['cameras'][k] for k in ('task','overview')})
        def photograph(tag):
            before=robot_readback(env);q_before={n:q.copy() for n,q in all_cloths().items()};v_before={n:q.copy() for n,q in all_velocities().items()}
            capture.capture(tag,camera_ids=('task','overview'))
            if any(not np.array_equal(v,robot_readback(env)[k]) for k,v in before.items()):raise RuntimeError('Rendering changed native robot state')
            q_after=all_cloths();v_after=all_velocities()
            if (any(not np.array_equal(q,q_after[n]) for n,q in q_before.items()) or
                    any(not np.array_equal(q,v_after[n]) for n,q in v_before.items())):
                raise RuntimeError('Rendering changed native cloth state')
        def record():
            positions=all_cloths();velocities=all_velocities();q=positions[spec['id']]
            mask=np.zeros(len(q),np.uint8);enabled=np.zeros(2,np.uint8);poses=np.tile([0.,0.,0.,0.,0.,0.,1.],(2,1))
            for a in assists:
                own=a.mask(len(q))
                if np.any(mask&own):raise RuntimeError('Two hands attempted to constrain the same material')
                mask|=own
                if a.active:enabled[('left','right').index(a.side)]=1;poses[('left','right').index(a.side)]=a.anchor_pose
            others=independent_cloth_streams(spec['id'],positions,velocities)
            writer.append({**robot_readback(env),**others,'cloth_q':q,'cloth_qd':velocities[spec['id']],
                'control_mask':mask,'assist_enabled':enabled,'assist_anchor_pose_xyzw':poses,
                'time_s':np.asarray(step*dt),'native_step':np.asarray(step,np.int64),
                'world_step':np.asarray(env.world.current_time_step_index,np.int64),'world_time_s':np.asarray(env.world.current_time),
                'phase_index':np.asarray(phase_index,np.int64),'action_index':np.asarray(action_index,np.int64),
                'joint_target':env._targets[0].copy(),'rigid_contact_pair_count':np.asarray(pairs,np.int64),
                'rigid_contact_impulse_ns':np.asarray(impulse,np.float64)})
        def command(values):
            nonlocal seed
            seed=command_joint_targets(env,values,current,action)
        def solve(wrists,closed,*,warm=None,selected=None,closure_timing=False):
            result=dict(seed if warm is None else warm)
            for edge in phase['edges']:
                side=edge['side']
                if selected is not None and side not in selected:continue
                model=models[side];offset=model.wrist_offset if model.wrist_offset is not None else np.eye(4)
                target=np.linalg.inv(model.arm_base)@wrists[side]@np.linalg.inv(offset)
                ik=model.arm.inverse(target,model.arm_tip,result,iterations=180,multi_starts=0,position_tolerance=.0005,rotation_tolerance=.005)
                if not ik['converged']:raise ValueError('Actual pinch-frame path IK failed: '+side+' '+json.dumps(ik))
                result.update(ik['joint_positions_rad'])
                result.update(closure_hand_posture(edge['contact_plan'],closed) if closure_timing
                    else interpolate_posture(edge['contact_plan']['postures'],closed))
            return result
        def tick():
            nonlocal step,pairs,impulse,heartbeat
            state=robot_readback(env);q=cloth()
            stopped=False
            for a in current:
                if a.side in capture_sides:
                    stopped=a.observe_closure(action,step*dt,state,q,env._targets[0]) or stopped
            # Apply the measured stop before CaptureGate sees the actual target,
            # and before a single subsequent native physics step is advanced.
            if stopped:command(seed)
            for a in current:
                if a.active or a.side in capture_sides:a.update(action,step*dt,state,q,env._targets[0])
            env._current_step_time=env.time_s+env.dt;advance(env);env._contact_reports.clear();env._read_rigid_contacts()
            pairs=len(env._contact_reports);impulse=float(sum(x['impulse_magnitude_sum_ns'] for x in env._contact_reports))
            if pairs:contacts.write(json.dumps({'native_step':step+1,'contacts':env._contact_reports})+'\n')
            step+=1;record()
            if capture and int(step*dt*args.media_fps+1e-9)>int((step-1)*dt*args.media_fps+1e-9):photograph(f'step_{step:06d}')
            if time.monotonic()-heartbeat>20:
                print(json.dumps({'event':'captured_fold_progress','phase':phase_index,'action':action,'step':step,'time_s':step*dt}),flush=True);heartbeat=time.monotonic()
        def run_action(name,seconds,controller=None):
            nonlocal action,action_index
            action=name;action_index+=1
            event={'phase_index':phase_index,'action_index':action_index,'action':name,
                'first_native_step':step,'begin_time_s':step*dt,'requested_duration_s':seconds,
                'capture_sides':sorted(capture_sides)}
            report['action_events'].append(event)
            for local in range(round(seconds/dt)):
                if controller and local%round(.02/dt)==0:controller(minimum_jerk((local+1)*dt,0.,seconds))
                tick()
            event.update(last_native_step=step,end_time_s=step*dt,complete=True)
            print(json.dumps({'event':'captured_fold_action_complete','phase':phase_index,'action':name,'time_s':step*dt}),flush=True)
        record()
        if capture:photograph('initial')
        run_action('settle',.5)
        for phase_index in range(count):
            if any(a.active for a in assists):raise RuntimeError('Previous phase retained a constraint')
            current=[]
            release_settings=release_configurations[phase_index]
            release_profile=release_settings['release_profile'];hand_profile=release_settings['release_hand_profile']
            retreat_profile=release_settings['retreat_profile']
            timing=preparation_timings[phase_index]
            phase=None
            if timing==AFTER_PARK:capture_sides=set()
            def prepare_current_phase():
                nonlocal phase
                observed_q=cloth()
                phase=prepare(release,task,scene,observed_q,phase_index,options)
                report['phase_plans'].append(json.loads(json.dumps(phase)));dump(out/f'phase_{phase_index:02d}_plan.json',phase)
                openings=planned_openings[phase_index]
                if openings and set(openings)!={e['side'] for e in phase['edges']}:
                    raise ValueError('Narrow opening settings must cover exactly the selected actual hands')
                for edge in phase['edges']:
                    measurement_options={}
                    if openings:
                        actual_plan_open=edge['contact_plan']['postures'][0]['nominal_tip_distance_m']
                        if abs(actual_plan_open-openings[edge['side']])>1.e-9:
                            raise ValueError('Original pinch plan changed the declared narrow opening')
                        measurement_options['planned_open_tip_gap_m']=float(actual_plan_open)
                    a=CapturedPatchAssist(env,scene,edge['side'],export,out,name=f'phase_{phase_index:02d}_{edge["side"]}',
                        allowed_material_vertex_ids=edge['allowed_material_vertex_ids'],cloth_id=spec['id'],
                        attachment_material_mode=options.get('attachment_material_mode','captured_patch'),
                        **capture_configurations[phase_index],**measurement_options,
                        **({'material_capture_region':edge['material_capture_region']}
                           if 'material_capture_region' in edge else {}),
                        **({'pivot_material_reference':edge['pivot_material_reference']}
                           if 'pivot_material_reference' in edge else {}));current.append(a);assists.append(a)
                if timing==AFTER_PARK:
                    report.setdefault('phase_preparation_events',[]).append({
                        'phase_index':phase_index,'profile':timing,
                        'observed_native_time_s':step*dt,'observed_native_step':step,
                        'observed_cloth_q_float64_sha256':hashlib.sha256(
                            np.asarray(observed_q,dtype='<f8').tobytes()).hexdigest(),
                        'prior_assists_released':True,'capture_sides_during_park':[],
                        'scope':'Fresh planning and new assist construction after released Cartesian/via_park; original capture guards remain unchanged.'})
                return phase
            prepare_at_stage(timing,BEFORE_PARK,prepare_current_phase,
                previous_assists=assists,current=current,capture_sides=capture_sides)
            # Return both arms to their original known park before changing edge.
            capture_sides=set()
            transition_settings=park_transitions[phase_index]
            if transition_settings is not None:
                if any(a.active for a in assists):raise RuntimeError('Park transition requires all cloth constraints released')
                side=transition_settings['side'];model=models[side];observed=robot_readback(env)
                wrist=actual_ik_frames(observed['robot_body_q'],env.body_names,models,[side])[side]
                park_wrist=model.arm.forward(park,model.arm_base)[model.arm_tip]@model.wrist_offset
                actual_joints=dict(zip(env.joint_names,map(float,observed['joint_q'])))
                hand_names=[joint.name for joint in model.hand.active]
                path=CartesianParkTransition(wrist,park_wrist,
                    {n:actual_joints[n] for n in hand_names},{n:park[n] for n in hand_names},
                    peak_height_m=transition_settings['peak_height_m'])
                def park_targets(u,warm):
                    target,hand=path.targets(u)
                    ik=model.arm.inverse(np.linalg.inv(model.arm_base)@target@np.linalg.inv(model.wrist_offset),
                        model.arm_tip,warm,iterations=180,multi_starts=0,
                        position_tolerance=.0005,rotation_tolerance=.005)
                    if not ik['converged']:raise ValueError('Released wrist-to-park IK failed: '+json.dumps(ik))
                    return {**warm,**ik['joint_positions_rad'],**hand}
                preview=dict(seed);seconds=transition_settings['duration_s']
                for local in range(round(seconds/.02)+1):
                    preview=park_targets(minimum_jerk(local*.02,0.,seconds),preview)
                report.setdefault('park_transition_plans',[]).append({'phase_index':phase_index,
                    'side':side,'observed_native_time_s':step*dt,'observed_native_step':step,
                    'duration_s':seconds,'constraints_released':True,
                    'preflight_control_samples':round(seconds/.02)+1,**path.receipt()})
                run_action('cartesian_to_park',seconds,lambda u:command(park_targets(u,seed)))
            begin=dict(seed);run_action('via_park',2.,lambda u:command({n:begin[n]+u*(park[n]-begin[n]) for n in begin}))
            prepare_at_stage(timing,AFTER_PARK,prepare_current_phase,
                previous_assists=assists,current=current,capture_sides=capture_sides)
            for batch_index,batch in enumerate(capture_batches(phase,options)):
                capture_sides=set(batch)
                rebound=prepare(release,task,scene,cloth(),phase_index,options,only_sides=capture_sides)
                rebound['observed_native_time_s']=step*dt
                report.setdefault('capture_replans',[]).append(rebound)
                replacement={e['side']:e for e in rebound['edges']}
                phase['edges']=[replacement.get(e['side'],e) for e in phase['edges']]
                for a in current:
                    if a.side in replacement:
                        a.allowed_ids=set(replacement[a.side]['allowed_material_vertex_ids'])
                        if 'pivot_material_reference' in replacement[a.side]:
                            a.set_pivot_material_reference(replacement[a.side]['pivot_material_reference'])
                selected_edges=[e for e in phase['edges'] if e['side'] in capture_sides]
                base={e['side']:np.asarray(e['contact_plan']['wrist_world_matrix']) for e in selected_edges}
                settings=options.get('phases',{}).get(str(phase_index),{}).get('hands',{})
                approaches={e['side']:capture_approach(e['contact_plan'],settings.get(e['side'],{})) for e in selected_edges}
                report.setdefault('approach_paths',[]).append({'phase_index':phase_index,
                    'begin_time_s':step*dt,'sides':approaches,
                    'scope':'Joint interpolation to explicit seeded hover, then fixed-orientation XYZ descent; actual native tracking recorded separately.'})
                def raised(amount):
                    value={s:m.copy() for s,m in base.items()}
                    for side,m in value.items():m[:3,3]+=np.asarray(approaches[side]['offset_m'])*amount
                    return value
                warm=dict(seed)
                for approach in approaches.values():warm.update(approach['arm_joint_positions_rad'])
                previous_hover=raised(1.)
                hover=solve(previous_hover,0.,warm=warm,selected=capture_sides);begin=dict(seed)
                run_action('open_hover',2.,lambda u:command({n:begin[n]+u*(hover[n]-begin[n]) for n in begin}))
                if rebind_before_descent:
                    observed_q=cloth().copy()
                    rebound=prepare(release,task,scene,observed_q,phase_index,options,only_sides=capture_sides)
                    rebound.update(observed_native_time_s=step*dt,observed_native_step=step,
                        observed_cloth_q_float64_sha256=hashlib.sha256(np.asarray(observed_q,dtype='<f8').tobytes()).hexdigest(),
                        reason='Fresh cloth after open_hover; descend from the previous hover target without moving to a new hover.')
                    report.setdefault('capture_replans',[]).append(rebound)
                    replacement={e['side']:e for e in rebound['edges']}
                    if set(replacement)!=capture_sides:raise ValueError('Descent rebind changed the current capture batch')
                    phase['edges']=[replacement.get(e['side'],e) for e in phase['edges']]
                    for a in current:
                        if a.side in replacement:
                            a.allowed_ids=set(replacement[a.side]['allowed_material_vertex_ids'])
                            if 'pivot_material_reference' in replacement[a.side]:
                                a.set_pivot_material_reference(replacement[a.side]['pivot_material_reference'])
                    selected_edges=[e for e in phase['edges'] if e['side'] in capture_sides]
                    base={e['side']:np.asarray(e['contact_plan']['wrist_world_matrix']) for e in selected_edges}
                    preflight=preflight_rebound_descent(previous_hover,base,seed,
                        lambda wrists,preview:solve(wrists,0.,warm=preview,selected=capture_sides))
                    report.setdefault('descent_preflights',[]).append({**preflight,
                        'phase_index':phase_index,'batch_index':batch_index,'begin_time_s':step*dt})
                    run_action('descend',1.5,lambda u:command(solve(
                        rebound_descent_wrists(previous_hover,base,u),0.,selected=capture_sides)))
                else:
                    run_action('descend',1.5,lambda u:command(solve(raised(1-u),0.,selected=capture_sides)))
                feed_progress=ClosureFeedProgress()
                has_feed=any('closure_feed' in e['contact_plan'] for e in selected_edges)
                timings={e['side']:e['contact_plan']['closure_timing_profile'] for e in selected_edges
                         if e['contact_plan'].get('closure_timing_profile','synchronous_v1')!='synchronous_v1'}
                if has_feed or timings:
                    rows=preflight_edge_feed(base,selected_edges,seed,
                        lambda wrists,u,warm:solve(wrists,u,warm=warm,selected=capture_sides,closure_timing=True))
                    report.setdefault('closure_feed_preflights' if has_feed else 'closure_timing_preflights',[]).append({'phase_index':phase_index,
                        'capture_sides':sorted(capture_sides),'observed_native_time_s':step*dt,
                        'sample_count':len(rows),'samples':rows,'native_steps_advanced':0,
                        **({'closure_timing_profiles':timings} if timings else {}),
                        'scope':'Coupled original wrist/finger IK before entering the unchanged native capture gate.'})
                    report.setdefault('closure_feed_stops',[]).append({'phase_index':phase_index,
                        'capture_sides':sorted(capture_sides),'events':feed_progress.events})
                def closure(u):
                    fractions=feed_progress.fractions(selected_edges,current,u,step*dt)
                    command(solve(closure_wrists(base,selected_edges,u,feed_fractions=fractions),u,
                        selected=capture_sides,closure_timing=True))
                run_action('close',2.,closure)
                if any(e['contact_plan']['capture_rise_m'] for e in selected_edges):
                    def seating(u):
                        fractions=feed_progress.fractions(selected_edges,current,1.,step*dt)
                        wrists=closure_wrists(base,selected_edges,1.,feed_fractions=fractions)
                        for e in selected_edges:
                            wrists[e['side']][2,3]+=u*e['contact_plan']['capture_rise_m']
                        command(solve(wrists,1.,selected=capture_sides))
                    run_action('capture_raise',1.,seating)
                run_action('closed_hold',1.)
                if not all(a.transport_ready for a in current if a.side in capture_sides):
                    raise RuntimeError('Each intended garment edge must be captured before any transport; capture failed: '+','.join(batch))
            capture_sides={a.side for a in current}
            actual=robot_readback(env);q=cloth();wrists0=actual_ik_frames(actual['robot_body_q'],env.body_names,models,[a.side for a in current])
            arcs={a.side:(capture_lift_arc(q,a.trajectory_reference_ids()) if transport_mode=='capture_lift_probe' else
                         task_adapter.captured_arc(task,phase,q,a.trajectory_reference_ids(),release_height=.025,peak=peak_heights[phase_index],
                                                   all_cloth_q=all_cloths()) if task_adapter else
                         captured_arc(task,phase,q,a.trajectory_reference_ids(),release_height=.025,peak=peak_heights[phase_index])) for a in current}
            transport_hand_targets={}
            preview_release_targets={}
            if release_profile in ('cooked_hand_clearance_v2','cooked_opening_lift_v3'):
                if release_profile=='cooked_opening_lift_v3':
                    from scripts.validation.laundry_dynamic_release import place_arc
                else:
                    from scripts.validation.laundry_release_clearance import place_arc
                preview_start=(dict(zip(env.joint_names,map(float,env._targets[0])))
                               if hand_profile in ('distal_extend_v1','full_spread_extend_v2') else seed)
                for a in current:
                    edge=next(e for e in phase['edges'] if e['side']==a.side)
                    if hand_profile in ('distal_extend_v1','full_spread_extend_v2'):
                        transport_hand_targets.update({n:preview_start[n]
                            for n in edge['contact_plan']['postures'][0]['hand_posture_rad']})
                    opening_sweep=release_opening_postures(a,edge['contact_plan']['postures'],
                        preview_start,hand_profile,model=models[a.side])
                    if hand_profile=='full_spread_extend_v2':
                        preview_release_targets.update(opening_sweep[-1]['hand_posture_rad'])
                    arcs[a.side]=place_arc(arcs[a.side],side=a.side,model=models[a.side],
                        postures=opening_sweep,
                        actual_joint_positions=dict(zip(env.joint_names,map(float,actual['joint_q']))),
                        actual_flange=wrists0[a.side],body_q=actual['robot_body_q'],body_names=env.body_names,
                        colliders=export,cloth_q=np.concatenate(list(all_cloths().values())),
                        table_z=task['asset_metadata']['table_top_z_m'],
                        thickness=float(spec['isaacsim']['cloth_thickness_m']),peak=peak_heights[phase_index])
            for a in current:
                if a.attachment_material_mode == 'captured_pivot_v1':
                    arcs[a.side].update(attachment_material_mode='captured_pivot_v1',
                        trajectory_reference_material_ids=a.trajectory_reference_ids(),
                        constraint_material_vertex_ids=list(a.ids))
            report['captured_arcs'].append({'phase_index':phase_index,'sides':arcs,
                'actual_ik_frame_start_xyzw':{a.side:{'body_name':models[a.side].frame_id,
                    'pose':actual['robot_body_q'][env.body_names.index(models[a.side].frame_id)].tolist()} for a in current},
                'attachment_frames':{a.side:a.side+'_hand_wrist' for a in current},
                'frame_scope':'IK follows the public model frame (hand_flange); local assistance follows the independently measured hand_wrist.'})
            def transport(u):
                targets={s:m.copy() for s,m in wrists0.items()}
                for s,m in targets.items():
                    controls=arcs[s]['controls_m'];m[:3,3]+=bezier_position(controls,u)-np.asarray(controls[0])
                command(retain_released_targets(solve(targets,1.),transport_hand_targets))
            # Preflight every20ms of this actual-bound transport before moving.
            preview=dict(seed)
            for local in range(251):
                u=minimum_jerk(local*.02,0.,5.);targets={s:m.copy() for s,m in wrists0.items()}
                for s,m in targets.items():m[:3,3]+=bezier_position(arcs[s]['controls_m'],u)-np.asarray(arcs[s]['controls_m'][0])
                preview=retain_released_targets(solve(targets,1.,warm=preview),transport_hand_targets)
            run_action('capture_lift' if transport_mode=='capture_lift_probe' else 'fold',5.,transport)
            # All constraints are disabled at the first opening tick. Actual
            # finger drives open above the goal; the cloth then falls freely.
            release_wrists=actual_ik_frames(robot_readback(env)['robot_body_q'],env.body_names,models,[a.side for a in current])
            released_targets={}
            if hand_profile in ('distal_extend_v1','full_spread_extend_v2'):
                opening_start=dict(zip(env.joint_names,map(float,env._targets[0])))
                for a in current:
                    edge=next(e for e in phase['edges'] if e['side']==a.side)
                    opened=({n:preview_release_targets[n]
                             for n in edge['contact_plan']['postures'][0]['hand_posture_rad']}
                            if hand_profile=='full_spread_extend_v2' else
                            release_target(a.side,edge['contact_plan']['postures'],hand_profile))
                    if any(opening_start[n]!=preview_start[n] for n in opened):
                        raise ValueError('Held hand commands changed after the cooked opening sweep')
                    released_targets.update(opened)
                report.setdefault('release_hand_plans',[]).append({'phase_index':phase_index,
                    'profile':hand_profile,'observed_native_time_s':step*dt,
                    'start_original_joint_targets_rad':{n:opening_start[n] for n in released_targets},
                    'end_original_joint_targets_rad':dict(released_targets),
                    'opening_duration_s':2.,'cooked_opening_samples':1001 if release_profile=='cooked_opening_lift_v3' else 61,
                    'interpolation':'single synchronous joint interpolation with the action time law',
                    'endpoint_held_through':['release_exit','retreat','free'],
                    'scope':'Original drive targets; actual finger motion and cloth separation require native review.'})
                if release_profile=='cooked_opening_lift_v3':
                    from scripts.validation.laundry_dynamic_release import rebind_opening,opening_lift_offset
                    observed=robot_readback(env);lift_plans={}
                    for a in current:
                        edge=next(e for e in phase['edges'] if e['side']==a.side)
                        names=edge['contact_plan']['postures'][0]['hand_posture_rad']
                        lift_plans[a.side]=rebind_opening(arcs[a.side]['release_clearance'],
                            side=a.side,model=models[a.side],
                            start_targets={n:opening_start[n] for n in names},
                            opened={n:released_targets[n] for n in names},
                            actual_joint_positions=dict(zip(env.joint_names,map(float,observed['joint_q']))),
                            actual_flange=release_wrists[a.side],body_q=observed['robot_body_q'],
                            body_names=env.body_names,colliders=export)
                    def lifting_wrists(u):
                        value={s:m.copy() for s,m in release_wrists.items()}
                        for side,m in value.items():m[2,3]+=opening_lift_offset(lift_plans[side],u)
                        return value
                    preview=dict(seed)
                    for local in range(101):
                        u=minimum_jerk(local*.02,0.,2.)
                        preview=retain_released_targets(solve(lifting_wrists(u),1-u,warm=preview),
                            interpolate_release_target(opening_start,released_targets,u))
                    report.setdefault('dynamic_release_plans',[]).append({'phase_index':phase_index,
                        'profile':release_profile,'observed_native_time_s':step*dt,
                        'observed_native_step':step,'opening_duration_s':2.,
                        'preflight_control_samples':101,'sides':lift_plans,
                        'scope':'Original hand/wrist joint commands only; actual release and cloth separation audited independently.'})
                    def dynamic_opening(u):
                        # Detach before submitting either finger or rising-wrist targets.
                        for a in current:a.release(step*dt,'Dynamic opening releases local constraint before robot motion.')
                        command(retain_released_targets(solve(lifting_wrists(u),1-u),
                            interpolate_release_target(opening_start,released_targets,u)))
                    run_action('open',2.,dynamic_opening)
                    release_wrists=actual_ik_frames(robot_readback(env)['robot_body_q'],env.body_names,models,[a.side for a in current])
                else:
                    run_action('open',2.,lambda u:command(opening_capture_targets(current,phase,
                        opening_start,solve(release_wrists,1-u),u,step*dt,hand_profile=hand_profile,release_targets=released_targets)))
            elif capture_configurations[phase_index]['contact_closure_stop']:
                opening_start=dict(seed)
                run_action('open',1.,lambda u:command(opening_capture_targets(current,phase,
                    opening_start,solve(release_wrists,1-u),u,step*dt)))
            else:
                run_action('open',1.,lambda u:command(solve(release_wrists,1-u)))
            if retreat_profile=='finger_axis_exit_v2':
                opened=robot_readback(env)
                exit_start=actual_ik_frames(opened['robot_body_q'],env.body_names,models,[a.side for a in current])
                exits={a.side:finger_axis_exit(opened['robot_body_q'],env.body_names,a.side) for a in current}
                report.setdefault('release_exit_plans',[]).append({'phase_index':phase_index,
                    'observed_native_time_s':step*dt,'sides':exits,'constraints_released_before_exit':not any(a.active for a in current)})
                def exit_targets(u):
                    value={s:m.copy() for s,m in exit_start.items()}
                    for s,m in value.items():m[:3,3]+=u*np.asarray(exits[s]['translation_m'])
                    return value
                preview=dict(seed)
                for local in range(51):
                    preview=retain_released_targets(solve(exit_targets(minimum_jerk(local*.02,0.,1.)),0.,warm=preview),released_targets)
                run_action('release_exit',1.,lambda u:command(retain_released_targets(solve(exit_targets(u),0.),released_targets)))
                release_wrists=actual_ik_frames(robot_readback(env)['robot_body_q'],env.body_names,models,[a.side for a in current])
            def retreat(u):
                targets={s:m.copy() for s,m in release_wrists.items()}
                for s,m in targets.items():
                    m[:3,3]+=(np.array([0.,0.,.06])*u if retreat_profile=='finger_axis_exit_v2' else
                              retreat_translation(models[s],release_wrists[s],u))
                command(retain_released_targets(solve(targets,0.),released_targets))
            run_action('retreat',1.,retreat);run_action('free',1.)
            if released_dwells[phase_index]:
                if any(a.active for a in assists):
                    raise RuntimeError('Additional released dwell cannot retain a cloth constraint')
                report.setdefault('additional_released_dwells',[]).append({'phase_index':phase_index,
                    'begin_native_time_s':step*dt,'duration_s':released_dwells[phase_index],
                    'all_attachments_disabled':True,
                    'scope':'New free simulation with unchanged joint targets, not checkpoint playback.'})
                run_action('released_dwell',released_dwells[phase_index])
            settings=released_joint_plans[phase_index]
            if settings is not None:
                if any(a.active for a in assists):
                    raise RuntimeError('Ordinary released motion cannot retain any cloth constraint')
                current=[];capture_sides=set()
                observed=robot_readback(env)
                actual=dict(zip(env.joint_names,map(float,observed['joint_q'])))
                original_rows,provenance=released_joint_rows[phase_index]
                if settings['profile']==OBSERVED_SLEEVE_PROFILE:
                    from scripts.validation.laundry_press_preflight import make_press_preflight
                    positions=all_cloths();masks={name:np.zeros(len(q),np.uint8) for name,q in positions.items()}
                    for a in assists:masks[a.cloth_id]|=a.mask(len(positions[a.cloth_id]))
                    enabled=[a.active for a in assists];observed_q=positions[spec['id']].copy()
                    preflight_path=out/f'phase_{phase_index:02d}_observed_press_preflight.json'
                    checker=make_press_preflight(scene,initial,models,export,observed_q,
                        env.cloth_triangles[spec['id']],actual_joints=actual)
                    def preflight(samples):
                        review=checker(samples);dump(preflight_path,review);return review
                    rows,fresh_plan=plan_observed_released_press(settings,provenance,
                        {'scene.json':scene,'task.json':task,'initial_state.json':initial},
                        actual,models,export,observed_q,masks,enabled,preflight)
                    after=robot_readback(env)
                    if (any(not np.array_equal(value,after[key]) for key,value in observed.items())
                            or any(not np.array_equal(q,all_cloths()[name]) for name,q in positions.items())):
                        raise RuntimeError('Observed press planning changed the native state')
                    observed_path=out/f'phase_{phase_index:02d}_observed_press_plan.json';dump(observed_path,fresh_plan)
                    receipt={'profile':settings['profile'],'actual_start_joint_positions_rad':actual,
                        'observation':fresh_plan['observation'],'observed_plan':pin(observed_path),
                        'fresh_collision_preflight':pin(preflight_path),
                        'replanned_control_intervals':len(rows)-1,'all_later_references_unchanged':False,
                        'maximum_original_joint_velocity_fraction':fresh_plan['maximum_original_joint_velocity_fraction'],
                        'attachments':False,'cloth_state_writes':0,
                        'scope':'Entire ordinary vertical press freshly planned from actual released sleeve geometry; '
                                'template motions provide no inherited path, physics or task qualification.'}
                else:
                    rows,receipt=rebind_released_start(original_rows,settings,actual,models)
                receipt.update(phase_index=phase_index,begin_native_step=step,
                    begin_native_time_s=step*dt,duration_s=settings['duration_s'],**provenance)
                plan_path=out/f'phase_{phase_index:02d}_released_joint_plan.json'
                dump(plan_path,{'receipt':receipt,'samples':rows})
                report.setdefault('released_joint_plans',[]).append({**receipt,'executed_plan':pin(plan_path)})
                commands=iter(row['joint_positions_rad'] for row in rows[:-1])
                run_action('released_robot_press',settings['duration_s'],lambda _:command(next(commands)))
        current=[];run_action('free',args.final_free)
        if any(a.active for a in assists):raise RuntimeError('Final free window retained assistance')
        if step!=steps or env.world.current_time_step_index-start_step!=steps:raise RuntimeError('Native step count mismatch')
        if abs(env.world.current_time-start_time-duration)>4e-6:raise RuntimeError('Native time mismatch')
        sampling=env._contact_sampling
        if sampling['sampled_substeps']!=steps:raise RuntimeError('Incomplete rigid contact acquisition')
        sampling['complete']=True;report['native_rigid_contact_sampling']=dict(sampling)
        env.stage.Export(str(out/'final.usda'));report['final_stage']=pin(out/'final.usda')
        if capture:
            photograph('final');report['videos']={k:capture.encode_video(camera_id=k,fps=args.media_fps) for k in ('task','overview')}
        report.update(complete=True,native_steps=steps,duration_s=duration,
            all_task_folding_phases_executed=transport_mode=='fold' and count==len(phase_definitions(task)),
            original_edge_set_executed=all(p.get('all_phase_edges_selected',True) for p in report['phase_plans']),
            final_free_window_s=args.final_free)
        exit_code=0
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()};traceback.print_exc()
    finally:
        # Abort disables constraints; it never claims the incomplete trajectory
        # or its unsimulated free window as a completed release.
        for a in assists:
            if a.active:a.release(env.time_s,'Abort cleanup; no subsequent native free-window claim')
            a.close()
        report['capture_assistance']=[{'side':a.side,'name':a.name,'events':a.events,
            'active_at_end':a.active,'diagnostics':pin(a.diagnostics_path)} for a in assists]
        if any(a.closure_stop is not None for a in assists):
            report['contact_closure_stops']=[{'side':a.side,'name':a.name,
                'receipt':a.closure_stop.receipt,'diagnostics':pin(a.closure_diagnostics_path)}
                for a in assists if a.closure_stop is not None]
        if env and getattr(env,'_contact_sampling',None):
            report['native_rigid_contact_sampling']=dict(env._contact_sampling)
        if contacts:contacts.close();report['rigid_contacts']=pin(out/'rigid_contacts.jsonl')
        if writer:report['states']=writer.close()
        if capture:capture.close()
        dump(out/'native_report.json',report);print(json.dumps({'event':'captured_fold_finished','complete':report['complete'],'output':str(out)}),flush=True)
        if env:close_native_with_status(env,exit_code)
    return exit_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--options',type=Path);p.add_argument('--max-phases',type=int,default=1)
    p.add_argument('--peak-height',type=float,default=.12);p.add_argument('--final-free',type=float,default=2.)
    p.add_argument('--media-output',type=Path);p.add_argument('--media-fps',type=int,default=4)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if a.plan_only:
        task,scene=[load_json(a.release/n) for n in ('task.json','scene.json')]
        q=np.asarray(scene['cloths'][0]['initial_vertices_m'])
        options=load_json(a.options) if a.options else {}
        phase_peak_heights(options,a.max_phases,a.peak_height)
        if a.output.exists():raise FileExistsError(a.output)
        a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,prepare_phase(a.release,task,scene,q,0,options));return 0
    return native(a)


if __name__=='__main__':raise SystemExit(main())
