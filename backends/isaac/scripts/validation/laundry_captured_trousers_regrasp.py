"""Capture, release and regrasp trousers through three measured operations.

First move the cuff across the longitudinal crease, then freshly capture the
waist to complete that same crease, and finally capture the exposed paired
cuffs for the transverse fold. Every operation uses the common native runner,
original finger drives and actual capture gate. The public two-fold goal and
cloth state are unchanged. This is a scripted candidate, not task qualification.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

from bench.paths import load_json
from scripts.validation import laundry_captured_garment_fold as garment
from scripts.validation.laundry_capture_options import validate_hand_options
from scripts.validation.laundry_single_garment_native import dump

MODULE_NAME='scripts.validation.laundry_captured_trousers_regrasp'
REPORT_SCHEMA='laundry_captured_garment_fold.v1'
OPERATIONS=((0,1,'longitudinal_cuff'),(0,0,'longitudinal_waist_regrasp'),
            (1,0,'transverse_paired_cuffs'))


def validate_scene(task,scene):
    if task['id']!='trousers_folding' or [c['id'] for c in scene['cloths']]!=['towel']:
        raise ValueError('Regrasp adapter requires one original trousers cloth')
    if len(task['asset_metadata']['design']['fold_sequence_axis_hinge_sign_radius_m'])!=2:
        raise ValueError('The unchanged public longitudinal and transverse folds are required')


def phase_edges(task):
    if task['id']!='trousers_folding':raise ValueError('Trousers task required')
    original=garment.phase_edges(task)
    return [[copy.deepcopy(original[reference][edge])] for reference,edge,_ in OPERATIONS]


def prepare_phase(release,task,scene,current_q,index,options,*,only_sides=None):
    validate_hand_options(options)
    from scripts.validation.laundry_capture_region import validate_region_option
    validate_region_option(options,task.get('id'),towel_adapter=False)
    validate_scene(task,scene)
    if type(index) is not int or not 0<=index<len(OPERATIONS):
        raise ValueError('Select one of the three actual regrasp operations')
    reference,edge,label=OPERATIONS[index]
    mapped=copy.deepcopy(options)
    configured=copy.deepcopy(options.get('phases',{}).get(str(index),{}))
    if 'edge_indices' in configured and configured['edge_indices']!=[edge]:
        raise ValueError('Operation boundary cannot be changed by a conflicting subset')
    configured['edge_indices']=[edge]
    mapped['phases']={str(reference):configured}
    result=garment.prepare_phase(release,task,scene,current_q,reference,mapped,only_sides=only_sides)
    result.update(phase_index=index,execution_phase_index=index,
        original_reference_phase_index=reference,original_reference_edge_indices=[edge],
        operation=label,execution_adapter=MODULE_NAME,
        public_fold_goal_changed=False,
        scope='Fresh native boundary capture per operation; waist regrasp completes the original first crease. No cloth state write or task qualification.')
    return result


def captured_arc(task,phase,cloth_q,captured_ids,*,release_height=.025,peak=.12,all_cloth_q=None):
    if phase['phase_index']==1:
        return waist_alignment_arc(task,cloth_q,captured_ids,release_height=release_height,peak=peak)
    # Other operations retain the exact original public crease. Only the
    # freshly captured material moves through the measured wrist trajectory.
    return garment.captured_arc(task,phase,cloth_q,captured_ids,
                                release_height=release_height,peak=peak)


def waist_alignment_arc(task,cloth_q,captured_ids,*,release_height=.025,peak=.12):
    """Align the picked waist with its actual stationary material counterpart.

    A prior cuff pull can move/rotate the unpicked waist. Reflecting its world
    coordinates about a global line would then miss the other leg. Material
    identity supplies the original fold correspondence without setting a shape.
    """
    if not .02<=release_height<=.03 or not .04<=peak<=.15:
        raise ValueError('Bounded original fold arc and free-release height required')
    spec=task['asset_metadata']['cloths'][0]
    rest=np.asarray(spec['rest_vertices_m'],float);q=np.asarray(cloth_q,float)
    ids=np.asarray(captured_ids,int)
    if (q.shape!=rest.shape or not np.isfinite(q).all() or ids.ndim!=1 or not 2<=len(ids)<=32
            or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=len(rest))):
        raise ValueError('Complete actual trousers and a measured local captured patch required')
    if np.any(rest[ids,0]>=-.04) or np.any(rest[ids,1]<.30):
        raise ValueError('Waist regrasp requires the original moving-side upper boundary region')
    desired=rest[ids].copy();desired[:,0]*=-1.
    stationary=np.flatnonzero(rest[:,0]>.04)
    distances=np.linalg.norm(rest[stationary][None]-desired[:,None],axis=2)
    counterparts=stationary[np.argmin(distances,axis=1)]
    errors=np.min(distances,axis=1)
    if np.max(errors)>.015:raise ValueError('No original symmetric waist material correspondence within15mm')
    start=q[ids].mean(0);end=q[counterparts].mean(0)
    support=stationary[np.linalg.norm(q[stationary,:2]-end[:2],axis=1)<.035]
    if not len(support):raise ValueError('No actual stationary waist beneath the aligned target')
    support_z=float(np.quantile(q[support,2],.9))
    if support_z>.99:raise ValueError('Actual stationary waist is not table-supported')
    end[2]=support_z+release_height
    middle=(start+end)/2;middle[2]=max(start[2],end[2])+2*peak
    return {'controls_m':np.array([start,middle,end]).tolist(),
        'captured_vertex_ids':ids.tolist(),'counterpart_vertex_ids':counterparts.tolist(),
        'counterpart_material_errors_m':errors.tolist(),
        'target_basis':'Actual stationary waist at the original material-X mirror correspondence',
        'observed_target_support_z_m':support_z,'release_height_m':release_height,
        'actual_wrist_orientation':'fixed; native wrist translation carries only the measured local grip',
        'public_fold_goal_changed':False,'free_cloth_state_written':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--options',type=Path);p.add_argument('--max-phases',type=int,default=1)
    p.add_argument('--peak-height',type=float,default=.12);p.add_argument('--final-free',type=float,default=2.)
    p.add_argument('--media-output',type=Path);p.add_argument('--media-fps',type=int,default=4)
    p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if a.plan_only:
        task,scene=[load_json(a.release/n) for n in ('task.json','scene.json')]
        options=load_json(a.options) if a.options else {}
        if a.output.exists():raise FileExistsError(a.output)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        dump(a.output,prepare_phase(a.release,task,scene,np.asarray(scene['cloths'][0]['initial_vertices_m']),0,options))
        return 0
    return garment.native(a,task_adapter=sys.modules[__name__])


if __name__=='__main__':raise SystemExit(main())
