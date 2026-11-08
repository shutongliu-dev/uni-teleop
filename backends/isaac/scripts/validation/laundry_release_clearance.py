"""Place captured cloth using the measured hand and its full opening envelope.

This only plans a joint-driven wrist trajectory. The unpressed native cloth
remains dynamic. Its future fold thickness is a conservative allowance, so
actual cloth/robot contact and released shape still require native review.
"""
from __future__ import annotations

import numpy as np

from bench.evaluation.cloth.cloth_robot_collision import transform_xyzw,apply_transform


def release_envelope(start, end, hand_points, cloth_q, *, table_z, thickness,
                     model_error_m=0., moving_layers=2, clearance_m=.002):
    start,end,hand,q=[np.asarray(x,float) for x in (start,end,hand_points,cloth_q)]
    if (start.shape!=(3,) or end.shape!=(3,) or hand.ndim!=2 or hand.shape[1]!=3
            or q.ndim!=2 or q.shape[1]!=3 or not len(hand)
            or not all(np.isfinite(x).all() for x in (start,end,hand,q))):
        raise ValueError('Finite full hand-opening envelope and actual cloth required')
    if not 0<=model_error_m<=.002 or not .001<=clearance_m<=.005 or moving_layers not in (1,2):
        raise ValueError('Bounded measured kinematic error and physical clearance required')
    relative=hand-start
    lo=relative[:,:2].min(0)+end[:2]-thickness/2
    hi=relative[:,:2].max(0)+end[:2]+thickness/2
    inside=np.all((q[:,:2]>=lo)&(q[:,:2]<=hi),axis=1)
    surface=max(table_z,float(q[inside,2].max()+thickness/2) if inside.any() else table_z)
    minimum_relative_z=float(relative[:,2].min())
    # Allow the incoming fabric layers and measured FK mismatch in addition
    # to geometric separation. The 1mm allowance covers small servo/contact
    # motion; it is not permission to overwrite a folded shape.
    margin=clearance_m+model_error_m+.001
    new_z=surface+moving_layers*thickness+margin-minimum_relative_z
    if new_z-surface>.08:
        raise ValueError('The hand cannot place this patch close to the observed support')
    return {'end_z_m':new_z,'target_support_surface_z_m':surface,
        'support_node_count':int(inside.sum()),'hand_opening_projected_bounds_xy_m':[lo.tolist(),hi.tolist()],
        'minimum_hand_z_relative_to_capture_m':minimum_relative_z,
        'incoming_layer_allowance_m':moving_layers*thickness,'geometric_clearance_m':clearance_m,
        'measured_kinematic_point_error_m':model_error_m,'servo_contact_allowance_m':.001,
        'predicted_minimum_opening_hand_surface_z_m':new_z+minimum_relative_z,
        'cloth_state_written':False,'scope':'Pre-transport full cooked opening envelope above observed support plus incoming-layer allowance; not a continuous native contact certificate.'}


def place_arc(arc, *, side, model, postures, actual_joint_positions, actual_flange,
              body_q, body_names, colliders, cloth_q, table_z, thickness, peak):
    rows=[c for c in colliders['colliders'] if c['body_name'].startswith(side+'_')]
    if not rows:raise ValueError('Original cooked hand collision inventory required')
    body_indices={n:i for i,n in enumerate(body_names)}
    local={}
    for i,row in enumerate(rows):
        points=np.concatenate([np.asarray(p['conservative_outer_vertices'],float) for p in row['pieces']])
        local[i]=apply_transform(points,np.asarray(row['mesh_to_body_column_matrix'],float))
    native=[];planned=[];maximum_error=0.
    actual_fk=model.hand.forward(actual_joint_positions)
    for i,row in enumerate(rows):
        name=row['body_name'];observed=apply_transform(local[i],transform_xyzw(body_q[body_indices[name]]))
        predicted=apply_transform(local[i],actual_flange@actual_fk[name])
        maximum_error=max(maximum_error,float(np.linalg.norm(observed-predicted,axis=1).max()))
        native.append(observed)
    for posture in postures:
        fk=model.hand.forward({**actual_joint_positions,**posture['hand_posture_rad']})
        planned.extend(apply_transform(local[i],actual_flange@fk[row['body_name']]) for i,row in enumerate(rows))
    envelope=np.concatenate(native+planned)
    controls=np.asarray(arc['controls_m'],float)
    receipt=release_envelope(controls[0],controls[-1],envelope,cloth_q,
        table_z=table_z,thickness=thickness,model_error_m=maximum_error)
    controls[-1,2]=receipt['end_z_m'];controls[1]=(controls[0]+controls[-1])/2
    controls[1,2]=max(controls[0,2],controls[-1,2])+2*peak
    return {**arc,'controls_m':controls.tolist(),
        'release_height_m':float(controls[-1,2]-receipt['target_support_surface_z_m']),
        'release_height_basis':'Measured cooked hand full opening envelope and actual target cloth support',
        'release_clearance':{**receipt,'cooked_hand_collider_count':len(rows),
            'opening_posture_count':len(postures),'source_and_actual_envelope_points':len(envelope)}}
