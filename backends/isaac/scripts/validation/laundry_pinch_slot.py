"""Explicit small horizontal centering of an original two-finger approach."""
from __future__ import annotations

import math

import numpy as np


def validate_slot_offset(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > .005):
        raise ValueError('pinch_slot_offset_m must be finite and within +/-5mm')
    return float(value)


def pinch_slot_translation(model, side, contact_plan, offset_m):
    """Return a commanded XY translation, never a cloth or body-state edit.

    Use the original 61-pose pinch path's middle pose to define the horizontal
    thumb-to-index fingertip axis. The caller must replan IK at the returned TCP.
    This planned axis is not evidence that material is actually captured.
    """
    offset = validate_slot_offset(offset_m)
    if side not in ('left', 'right'):
        raise ValueError('Expected an original left or right hand')
    poses = contact_plan['postures']
    if len(poses) != 61:
        raise ValueError('Slot centering requires the original61-pose pinch path')
    wrist = np.asarray(contact_plan['wrist_world_matrix'], dtype=float)
    if (wrist.shape != (4, 4) or not np.isfinite(wrist).all()
            or not np.allclose(wrist[3], [0., 0., 0., 1.], atol=1.e-10)
            or not np.allclose(wrist[:3, :3].T @ wrist[:3, :3], np.eye(3), atol=1.e-8)
            or not np.isclose(np.linalg.det(wrist[:3, :3]), 1., atol=1.e-8)):
        raise ValueError('Slot centering requires a finite rigid public-flange pose')
    forward = model.hand.forward(poses[30]['hand_posture_rad'])
    tips = {finger: (wrist @ np.asarray(forward[side+'_'+finger+'_fingertip']))[:3, 3]
            for finger in ('thumb', 'index')}
    delta = tips['index']-tips['thumb']
    delta[2] = 0.
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length < .001:
        raise ValueError('Original fingertip slot has no defined horizontal axis')
    axis = delta/length
    return {'schema': 'laundry_pinch_slot_center.v1', 'offset_m': offset,
            'reference_posture_index': 30,
            'reference_tip_positions_world_m': {k: v.tolist() for k, v in tips.items()},
            'horizontal_thumb_to_index_axis_world': axis.tolist(),
            'translation_world_m': (offset*axis).tolist(),
            'scope': 'Bounded original-hand command translation; replan IK and verify native capture separately.'}


def slot_entry_bounds(pick, centering, feed=None):
    """Validate combined slot and entry geometry against the actual edge."""
    direction=np.asarray(pick['inward_world_xy'],float)
    edge=np.asarray(pick['edge_world_m'],float)
    tcp=np.asarray(pick['pinch_tcp_xy_m'],float)
    shift=np.asarray(centering['translation_world_m'],float)
    if (direction.shape!=(2,) or edge.shape!=(3,) or tcp.shape!=(2,) or shift.shape!=(3,)
            or not all(np.isfinite(x).all() for x in (direction,edge,tcp,shift))
            or abs(np.linalg.norm(direction)-1.)>1.e-8 or abs(shift[2])>1.e-12):
        raise ValueError('Combined slot entry needs a finite observed unit inward direction and XY shift')
    start=float((tcp+shift[:2]-edge[:2])@direction)
    end=start
    if feed is not None:
        travel=np.asarray(feed['translation_world_m'],float)
        if travel.shape!=(3,) or not np.isfinite(travel).all() or abs(travel[2])>1.e-12:
            raise ValueError('Combined slot entry requires a finite horizontal closure feed')
        end+=float(travel[:2]@direction)
        if not (-.010-1.e-12<=start<=1.e-12 and -1.e-12<=end<=.010+1.e-12):
            raise ValueError('Combined slot/feed must start outside within10mm and finish within10mm inside')
    elif not -.010-1.e-12<=start<=.010+1.e-12:
        raise ValueError('Combined slot center must remain within10mm of the observed edge')
    inward=float(shift[:2]@direction)
    tangent=float(shift[:2]@np.array([-direction[1],direction[0]]))
    return {'effective_start_inward_offset_m':start,'effective_end_inward_offset_m':end,
            'slot_inward_component_m':inward,'slot_tangent_component_m':tangent,
            'start_inward_bounds_m':[-.010,0. if feed is not None else .010],
            'end_inward_bounds_m':[0.,.010] if feed is not None else [-.010,.010],
            'basis':'Original observed material edge and unit inward direction; combined commanded XY geometry'}
