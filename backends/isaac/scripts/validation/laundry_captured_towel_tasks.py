"""Native towel unfold/fold/stack using measured two-finger edge capture.

Only moving towel boundary material can receive the declared wrist-frame local
aid. Stacking retains both independent physical cloth streams, including the
uncontrolled base. This runner never changes free cloth coordinates or filters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from bench.paths import load_json
from scripts.validation.laundry_contact_pinch import plan as pinch_plan
from scripts.validation.laundry_single_garment_native import dump
from scripts.validation.laundry_closure_feed import edge_feed_plan
from scripts.validation.laundry_closure_timing import bind_closure_timing
from scripts.validation.laundry_capture_options import validate_hand_options

MODULE_NAME='scripts.validation.laundry_captured_towel_tasks'
TASKS=('towel_flattening','towel_folding','towel_stacking')
SEQUENTIAL_FLATTEN='sequential_corners_then_alignment_v1'


def validate_scene(task,scene):
    if task['id'] not in TASKS:raise ValueError('Explicit towel flatten/fold/stack task required')
    names=[c['id'] for c in scene['cloths']]
    expected=['towel','towel_base'] if task['id']=='towel_stacking' else ['towel']
    if sorted(names)!=sorted(expected):raise ValueError('Required independent towel inventory changed')
    if task['id']=='towel_flattening' and task['asset_metadata'].get('reset_profile')!='towel_partial_edge_fold_v1':
        raise ValueError('Flatten demo requires the declared12cm partial flap reset')


def phase_edges(task):
    if task['id'] not in TASKS:raise ValueError('Towel task required')
    # Flattening also requires the public position/heading after unfolding.
    # A first-phase-only diagnostic must not claim the whole task executed.
    return [['negative_y_corner','positive_y_corner']] * (2 if task['id']=='towel_flattening' else 1)


def execution_phase_edges(task, options):
    """Keep repeated real regrasp separate from the two semantic operations."""
    validate_options(task, options)
    if 'flatten_phase_sequence' not in options:
        return phase_edges(task)
    return [['positive_y_corner'], ['negative_y_corner'],
            ['negative_y_corner', 'positive_y_corner']]


def task_phase_index(task, options, index):
    if type(index) is not int or not 0 <= index < len(execution_phase_edges(task, options)):
        raise ValueError('Unavailable towel manipulation phase')
    return (0, 0, 1)[index] if 'flatten_phase_sequence' in options else index



def validate_options(task, options):
    """Make an optional single-corner unfold explicit before native startup.

    Indices address the original two material boundaries, before hand filtering.
    Boundary-script coverage is diagnostic; public task success still requires
    the unchanged final shape, placement, release and stability conditions.
    """
    validate_hand_options(options, towel=True)
    sequential = 'flatten_phase_sequence' in options
    if sequential:
        if (task.get('id') != 'towel_flattening'
                or options['flatten_phase_sequence'] != SEQUENTIAL_FLATTEN
                or options.get('transport_mode', 'fold') != 'fold'):
            raise ValueError('Sequential corner unfolding requires the explicit flattening fold profile')
        phases = options.get('phases', {})
        if set(phases) != {'0', '1', '2'}:
            raise ValueError('Sequential flattening requires exactly three explicit execution phases')
        if (phases['0'].get('edge_indices') != [1]
                or phases['1'].get('edge_indices') != [0]
                or 'edge_indices' in phases['2']):
            raise ValueError('Unfold positive-Y then negative-Y corners, then align both leading corners')
    for index, configured in options.get('phases', {}).items():
        if 'edge_indices' not in configured:
            continue
        if task.get('id') != 'towel_flattening' or index not in (('0','1') if sequential else ('0',)):
            raise ValueError('Explicit towel edge selection is only available for initial flap unfolding')
        selected = configured['edge_indices']
        if (not isinstance(selected, list) or not selected
                or any(type(i) is not int or i not in (0, 1) for i in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('Flap edge_indices must select distinct original boundaries 0 and/or 1')
        sides = configured.get('sides', ['right', 'left'])
        if (not isinstance(sides, list) or len(sides) != 2
                or set(sides) != {'left', 'right'}):
            raise ValueError('Original two-boundary side mapping must be left and right')
        order = configured.get('capture_order')
        if order is not None and (not isinstance(order, list)
                or len(order) != len(selected)
                or set(order) != {sides[i] for i in selected}):
            raise ValueError('Capture order must cover exactly the explicitly selected flap hands')


def towel_picks(task,current_q,index=0):
    spec=next(c for c in task['asset_metadata']['cloths'] if c['id']=='towel')
    rest=np.asarray(spec['rest_vertices_m'],float);q=np.asarray(current_q,float)
    if q.shape!=rest.shape or not np.isfinite(q).all():raise ValueError('Complete current moving towel q required')
    task_id=task['id'];theta=.18 if task_id=='towel_flattening' else 0.
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    x=(rest[:,0].min() if index==0 else rest[:,0].max()) if task_id=='towel_flattening' else (rest[:,0].max() if task_id=='towel_folding' else 0.)
    result=[]
    for upper in (False,True):
        y=float(rest[:,1].max() if upper else rest[:,1].min())
        seed=int(np.argmin(np.linalg.norm(rest[:,:2]-[x,y],axis=1)))
        if np.linalg.norm(rest[seed,:2]-[x,y])>1.e-7:raise ValueError('Required exact material edge absent from mesh')
        allowed=np.flatnonzero(np.linalg.norm(rest-rest[seed],axis=1)<=.030)
        if task_id=='towel_flattening':allowed=allowed[(rest[allowed,0]<-.18) if index==0 else (rest[allowed,0]>.24)]
        if task_id=='towel_stacking':allowed=allowed[abs(rest[allowed,0])<=.025]
        # The flap's free negative-X edge has already reflected to the right
        # of its hinge. All three pickup edges therefore face inward toward-X.
        inward=rotation@np.array([-1.,-1. if upper else 1.]);inward/=np.linalg.norm(inward)
        if task_id=='towel_flattening' and index==1:
            # Pull the positive-X free edge toward the displaced public goal.
            # Pushing the former negative-X flap would merely compress it.
            # Estimate orientation from actual neighbouring material.
            neighbors=[int(np.argmin(np.linalg.norm(rest[:,:2]-p,axis=1))) for p in
                       ([x-.02,y],[x,y+(-.02 if upper else .02)])]
            vectors=q[neighbors,:2]-q[seed,:2]
            lengths=np.linalg.norm(vectors,axis=1)
            if np.any(lengths<.005):raise ValueError('Unfolded boundary neighbours are collapsed')
            inward=np.sum(vectors/lengths[:,None],axis=0);inward/=np.linalg.norm(inward)
        edge=q[seed].copy()
        result.append({'side':'left' if upper else 'right','seed_vertex_ids':[seed],
            'allowed_material_vertex_ids':allowed.tolist(),'edge_world_m':edge.tolist(),
            'edge_material_m':rest[seed].tolist(),'inward_world_xy':inward.tolist(),
            'pinch_tcp_xy_m':(edge[:2]+.006*inward).tolist(),
            'feature':('alignment_leading_corner' if index else 'partial_flap_free_corner') if task_id=='towel_flattening' else
                      ('positive_x_corner' if task_id=='towel_folding' else 'connected_crease_end')})
    return result


def observed_pinch_offset(task, current_q, pick, offset_m):
    """Move the TCP along the observed corner bisector without moving material.

    The local surface directions come from existing material neighbours. This
    opt-in never expands the capture region or substitutes an interior seed.
    """
    if (isinstance(offset_m, bool) or not isinstance(offset_m,(int,float)) or
            not np.isfinite(offset_m) or not -.010<=offset_m<=.010):
        raise ValueError('Explicit pinch inward offset must be finite and within +/-10mm')
    spec=next(c for c in task['asset_metadata']['cloths'] if c['id']=='towel')
    rest=np.asarray(spec['rest_vertices_m'],float);q=np.asarray(current_q,float)
    if q.shape!=rest.shape or not np.isfinite(q).all() or len(pick['seed_vertex_ids'])!=1:
        raise ValueError('One real towel boundary seed and complete actual q required')
    seed=int(pick['seed_vertex_ids'][0]);material=rest[seed,:2]
    dx=.02 if np.isclose(material[0],rest[:,0].min(),atol=1.e-8) else -.02
    dy=-.02 if np.isclose(material[1],rest[:,1].max(),atol=1.e-8) else .02
    wanted=[material+[dx,0.],material+[0.,dy]]
    neighbors=[int(np.argmin(np.linalg.norm(rest[:,:2]-v,axis=1))) for v in wanted]
    if any(np.linalg.norm(rest[i,:2]-v)>1.e-7 for i,v in zip(neighbors,wanted)):
        raise ValueError('Required original nearby material samples are absent')
    directions=q[neighbors,:2]-q[seed,:2];lengths=np.linalg.norm(directions,axis=1)
    if np.any(lengths<.005):raise ValueError('Observed local towel directions are collapsed')
    inward=np.sum(directions/lengths[:,None],axis=0)
    if np.linalg.norm(inward)<.25:raise ValueError('Observed local inward direction is ambiguous')
    inward/=np.linalg.norm(inward)
    return {**pick,'inward_world_xy':inward.tolist(),
        'pinch_tcp_xy_m':(q[seed,:2]+float(offset_m)*inward).tolist(),
        'pinch_inward_offset_m':float(offset_m),
        'pinch_direction_basis':{'kind':'actual_material_neighbour_corner_bisector',
            'seed_vertex_id':seed,'neighbour_vertex_ids':neighbors,
            'source':'fresh actual cloth q; original material identities; XY projection'},
        'allowed_material_vertex_ids':list(pick['allowed_material_vertex_ids'])}


def prepare_phase(release,task,scene,current_q,index,options,*,only_sides=None):
    validate_options(task, options)
    from scripts.validation.laundry_capture_region import validate_region_option,stack_region_descriptor
    region_profile=validate_region_option(options,task.get('id'),towel_adapter=True)
    validate_scene(task,scene)
    semantic_index=task_phase_index(task,options,index)
    picks=towel_picks(task,current_q,semantic_index);thickness=float(task['asset_metadata']['physical_thickness_m'])
    if region_profile=='region_support_v1':
        for pick in picks:pick['material_capture_region']=stack_region_descriptor(pick)
    configured=options.get('phases',{}).get(str(index),{})
    if 'sides' in configured:
        if len(configured['sides'])!=2 or set(configured['sides'])!={'left','right'}:raise ValueError('Two distinct original hands required')
        for p,s in zip(picks,configured['sides']):p['side']=s
    selection = {}
    if 'edge_indices' in configured:
        indices = list(configured['edge_indices'])
        picks = [picks[i] for i in indices]
        selection = {'selected_edge_indices': indices,
            'all_phase_edges_selected': len(indices) == 2,
            'partial_boundary_use': {'original_boundary_count': 2,
                'selected_boundary_count': len(indices),
                'selected_sides': [p['side'] for p in picks],
                'partial': len(indices) < 2,
                'scope': 'Explicit initial flap boundary subset; both task phases and unchanged public final judgement remain required.'}}
        if only_sides is not None and not set(only_sides) <= {p['side'] for p in picks}:
            raise ValueError('Cannot rebind an unselected flap hand')
    table=float(task['asset_metadata']['table_top_z_m'])
    for pick in picks:
        if only_sides is not None and pick['side'] not in only_sides:continue
        side=pick['side'];sign=1. if side=='right' else -1.
        setting=configured.get('hands',{}).get(side,{})
        if 'pinch_inward_offset_m' in setting or 'corner_tangent_sign' in setting:
            pick.update(observed_pinch_offset(task,current_q,pick,setting.get('pinch_inward_offset_m',.006)))
        feed=edge_feed_plan(pick,setting)
        n=np.asarray(pick['inward_world_xy'])
        yaw=float(np.rad2deg(np.arctan2(sign*n[0],-sign*n[1]))%360.)
        relative=setting.get('yaw_offset_deg')
        if 'corner_tangent_sign' in setting:
            angles=[(yaw+90.*setting['corner_tangent_sign'])%360.]
        elif 'yaw_offset_deg' in setting:
            if ('yaw_deg' in setting or isinstance(relative,bool) or not isinstance(relative,(int,float))
                    or not np.isfinite(relative) or abs(relative)>30.):
                raise ValueError('Use one finite relative edge yaw within30 degrees or an absolute yaw')
            angles=[(yaw+relative)%360.]
        else:angles=[setting['yaw_deg']] if 'yaw_deg' in setting else [yaw,(yaw+180)%360.]
        bottom=max(table+.0003,pick['edge_world_m'][2]-thickness/2+.0003)
        failed=[];contact=None
        for angle in angles:
            try:
                contact=pinch_plan(release,side,pick['pinch_tcp_xy_m'],angle,
                    setting.get('pitch_deg',60.),setting.get('bottom_z_m',bottom),
                    setting.get('close_distance_m',.010),setting.get('closure_lift_m',0.),
                    setting.get('capture_rise_m',0.),setting.get('closure_lift_start_fraction',0.),
                    setting.get('open_distance_m',.060))
                contact['inside_finger']=('tangent' if 'corner_tangent_sign' in setting else
                    'thumb' if np.cos(np.deg2rad(angle-yaw))>0. else 'index');break
            except ValueError as error:failed.append({'yaw_deg':angle,'message':str(error)})
        if contact is None:raise ValueError('No original towel-edge approach IK: '+json.dumps(failed))
        if 'yaw_offset_deg' in setting:
            contact['yaw_definition']={'basis':'current observed inward direction',
                'edge_normal_yaw_deg':yaw,'offset_deg':float(relative),'commanded_yaw_deg':float(angles[0])}
        if 'corner_tangent_sign' in setting:
            contact['yaw_definition']={'basis':'fresh observed material-neighbour corner bisector',
                'edge_normal_yaw_deg':yaw,'corner_tangent_sign':setting['corner_tangent_sign'],
                'offset_deg':90.*setting['corner_tangent_sign'],'commanded_yaw_deg':float(angles[0])}
        if feed is not None:contact['closure_feed']=feed
        bind_closure_timing(contact,setting)
        pick.update(contact_plan=contact,rejected_approach_candidates=failed)
    if 'flatten_phase_sequence' in options:
        selection.update(task_phase_index=semantic_index, flatten_phase_sequence=SEQUENTIAL_FLATTEN,
            task_operation='unfold' if semantic_index==0 else 'align')
        if 'partial_boundary_use' in selection:
            selection['partial_boundary_use']['scope'] = (
                'One original flap corner per execution phase; both corners and a separate '
                'positive-X alignment recapture are required. Public final judgement is unchanged.')
    return {'phase_index':index,'task_id':task['id'],'edges':[p for p in picks if 'contact_plan' in p],
        'scope':'Real material boundary/crease pickup, nominal discrete IK only; native capture remains mandatory.',
        **selection}


def captured_arc(task,phase,cloth_q,captured_ids,*,release_height=.025,peak=.12,all_cloth_q=None):
    if not .02<=release_height<=.03 or not .04<=peak<=.15:raise ValueError('Bounded arc/free-release height required')
    spec=next(c for c in task['asset_metadata']['cloths'] if c['id']=='towel')
    rest=np.asarray(spec['rest_vertices_m'],float);q=np.asarray(cloth_q,float);ids=np.asarray(captured_ids,int)
    if q.shape!=rest.shape or not np.isfinite(q).all():raise ValueError('Complete actual moving towel required')
    start=q[ids].mean(0);end=start.copy();thickness=float(task['asset_metadata']['physical_thickness_m'])
    table=float(task['asset_metadata']['table_top_z_m']);receipt={}
    if task['id']=='towel_flattening':
        if phase.get('task_phase_index',phase.get('phase_index',0))==0:
            if np.any(rest[ids,0]>=-.18):raise ValueError('Captured material is not within the12cm moving flap')
            theta=.18;rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            fixed=rest[:,0]>-.14
            translation=np.median(q[fixed,:2]-rest[fixed,:2]@rotation.T,axis=0)
            end[:2]=rest[ids,:2].mean(0)@rotation.T+translation
            receipt.update(observed_stationary_translation_xy_m=translation.tolist(),unfold_heading_rad=theta,
                target_basis='Original planar material embedding fitted to stationary native towel; no shape setter.')
        else:
            if np.any(rest[ids,0]<=.24):raise ValueError('Alignment requires recaptured positive-X leading boundary')
            target=np.asarray(task['asset_metadata']['target_center_xy_m'],float)
            end[:2]=rest[ids,:2].mean(0)+target
            receipt.update(public_target_center_xy_m=target.tolist(),target_heading_rad=0.,
                target_basis='Only the newly captured boundary patch moves toward its public planar goal; remaining cloth is dynamic.')
        end[2]=table+thickness/2+release_height
    elif task['id']=='towel_folding':
        if np.any(rest[ids,0]<.24):raise ValueError('Captured material is not on the positive-X moving edge')
        fixed=rest[:,0]<-.04;hinge=float(np.median(q[fixed,0]-rest[fixed,0]))
        end[0]=2*hinge-start[0]+2.5*thickness
        support=np.flatnonzero(np.linalg.norm(q[:,:2]-end[:2],axis=1)<.04)
        support=np.setdiff1d(support,ids)
        if not len(support):raise ValueError('No actual stationary towel beneath reflected target')
        support_z=float(np.quantile(q[support,2],.9));end[2]=support_z+release_height
        receipt.update(observed_hinge_x_m=hinge,observed_target_support_z_m=support_z)
    elif task['id']=='towel_stacking':
        if np.any(abs(rest[ids,0])>.025):raise ValueError('Only connected crease-end material may be captured')
        if not all_cloth_q or 'towel_base' not in all_cloth_q:raise ValueError('Independent actual base towel required')
        base=np.asarray(all_cloth_q['towel_base'],float)
        if not np.isfinite(base).all():raise ValueError('Nonfinite actual base')
        center=lambda v:(np.quantile(v[:,:2],.02,axis=0)+np.quantile(v[:,:2],.98,axis=0))/2
        translation=center(base)-center(q)
        base_top=float(np.quantile(base[:,2],.95));moving_bottom=float(np.quantile(q[:,2],.05))
        dz=base_top+thickness/2+release_height-moving_bottom
        if dz<0 or dz>.10:raise ValueError('Stack release unsupported or already vertically displaced')
        end[:2]+=translation;end[2]+=dz
        receipt.update(observed_translation_xy_m=translation.tolist(),base_top_z_m=base_top,
            moving_bottom_z_m=moving_bottom,whole_fold_translation_z_m=dz,
            base_state_used=True,base_controlled=False)
    else:raise ValueError('Unsupported towel task')
    middle=(start+end)/2;middle[2]=max(start[2],end[2])+2*peak
    return {'controls_m':np.array([start,middle,end]).tolist(),'captured_vertex_ids':ids.tolist(),
        'release_height_m':release_height,'target_shape_written':False,'free_cloth_state_written':False,
        'actual_wrist_orientation':'fixed public flange trajectory; assistance follows actual hand_wrist',**receipt}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--options',type=Path);p.add_argument('--peak-height',type=float,default=.08)
    p.add_argument('--final-free',type=float,default=2.);p.add_argument('--media-output',type=Path)
    p.add_argument('--media-fps',type=int,default=4);p.add_argument('--plan-only',action='store_true')
    p.add_argument('--max-phases',type=int,default=1)
    args=p.parse_args()
    if args.plan_only:
        task,scene=[load_json(args.release/n) for n in ('task.json','scene.json')]
        moving=next(c for c in scene['cloths'] if c['id']=='towel')
        options=load_json(args.options) if args.options else {}
        if args.output.exists():raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        dump(args.output,prepare_phase(args.release,task,scene,np.asarray(moving['initial_vertices_m']),0,options));return 0
    from scripts.validation.laundry_captured_garment_fold import native
    return native(args,task_adapter=sys.modules[__name__])


if __name__=='__main__':raise SystemExit(main())
