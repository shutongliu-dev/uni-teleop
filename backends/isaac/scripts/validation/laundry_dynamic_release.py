"""Opt-in low placement followed by real wrist lift during full hand opening.

All geometry is from original cooked robot surfaces. These helpers only plan
robot targets; they have no simulator, attachment or cloth-state setter.
"""
from __future__ import annotations
import numpy as np
from bench.evaluation.cloth.cloth_robot_collision import apply_transform, transform_xyzw
from scripts.validation.laundry_release_hand import interpolate_release_target

PROFILE = 'cooked_opening_lift_v3'
SAMPLES = 1001


def height_envelope(minimum_z, support_z, *, thickness, model_error_m):
    minimum = np.asarray(minimum_z, float)
    if (minimum.shape != (SAMPLES,) or not np.isfinite(minimum).all()
            or not np.isfinite([support_z, thickness, model_error_m]).all()
            or not 0 < thickness <= .01 or not 0 <= model_error_m <= .002):
        raise ValueError('1001 finite cooked opening minima and physical margins required')
    required = support_z + 2*thickness + .003 + model_error_m - minimum
    profile = np.maximum.accumulate(required)
    if np.ptp(profile) > .08:
        raise ValueError('Opening requires more than80mm real wrist lift')
    return required, profile


def opening_lift_offset(plan, amount):
    if not np.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError('Finite opening fraction in [0,1] required')
    values = np.asarray(plan['wrist_lift_m'], float)
    if (values.shape != (SAMPLES,) or not np.isfinite(values).all()
            or values.min() < 0 or np.any(np.diff(values) < 0)):
        raise ValueError('A finite nonnegative monotone cooked wrist lift is required')
    # Upper endpoint bins never interpolate below a sampled opening envelope.
    return float(values[min(int(np.ceil(amount*(SAMPLES-1)-1.e-12)), SAMPLES-1)])


def _geometry(*, side, model, start_targets, opened, actual_joint_positions,
              actual_flange, body_q, body_names, colliders):
    rows = [r for r in colliders['colliders'] if r['body_name'].startswith(side+'_')]
    if not rows:
        raise ValueError('Original cooked hand inventory required')
    local = [apply_transform(np.concatenate([np.asarray(p['conservative_outer_vertices'], float)
              for p in r['pieces']]), np.asarray(r['mesh_to_body_column_matrix'], float)) for r in rows]
    actual_fk = model.hand.forward(actual_joint_positions)
    error = 0.; native = []
    for row, vertices in zip(rows, local):
        observed = apply_transform(vertices, transform_xyzw(body_q[body_names.index(row['body_name'])]))
        predicted = apply_transform(vertices, actual_flange@actual_fk[row['body_name']])
        error = max(error, float(np.linalg.norm(observed-predicted, axis=1).max()))
        native.append(observed)
    if error > .002:
        raise ValueError('Actual hand FK mismatch exceeds original2mm clearance bound')
    minima = []; lo = []; hi = []
    for u in np.linspace(0., 1., SAMPLES):
        hand = interpolate_release_target(start_targets, opened, u)
        if (any(not j.lower <= hand[j.name] <= j.upper for j in model.hand.active)
                or any(model.hand_posture_prior.violation(f, hand) for f in ('thumb', 'index'))):
            raise ValueError('Dynamic opening violates original hand bounds or posture prior')
        fk = model.hand.forward({**actual_joint_positions, **hand})
        points = np.concatenate([apply_transform(v, actual_flange@fk[r['body_name']]) for r, v in zip(rows, local)])
        relative = points-actual_flange[:3, 3]
        minima.append(float(relative[:, 2].min()));lo.append(relative[:, :2].min(0));hi.append(relative[:, :2].max(0))
    native_relative = np.concatenate(native)-actual_flange[:3, 3]
    # At placement the original actual closed surface also needs clearance.
    minima[0] = min(minima[0], float(native_relative[:, 2].min()))
    lo.append(native_relative[:, :2].min(0));hi.append(native_relative[:, :2].max(0))
    return np.array(minima), np.min(lo, axis=0), np.max(hi, axis=0), error, len(rows)


def place_arc(arc, *, side, model, postures, actual_joint_positions, actual_flange,
              body_q, body_names, colliders, cloth_q, table_z, thickness, peak):
    controls = np.asarray(arc['controls_m'], float).copy()
    start, opened = [dict(p['hand_posture_rad']) for p in (postures[0], postures[-1])]
    minima, lo, hi, error, count = _geometry(side=side, model=model, start_targets=start,
        opened=opened, actual_joint_positions=actual_joint_positions, actual_flange=actual_flange,
        body_q=body_q, body_names=body_names, colliders=colliders)
    end_wrist_xy = actual_flange[:2, 3] + controls[-1, :2]-controls[0, :2]
    lo += end_wrist_xy-thickness/2;hi += end_wrist_xy+thickness/2
    q = np.asarray(cloth_q, float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError('Finite actual pretransport cloth support required')
    inside = np.all((q[:, :2] >= lo)&(q[:, :2] <= hi), axis=1)
    support = max(table_z, float(q[inside, 2].max()+thickness/2) if inside.any() else table_z)
    required, monotone = height_envelope(minima, support, thickness=thickness, model_error_m=error)
    controls[-1, 2] = required[0] + controls[0, 2]-actual_flange[2, 3]
    if controls[-1, 2]-support > .08:
        raise ValueError('Closed hand cannot place this captured material near support')
    controls[1] = (controls[0]+controls[-1])/2
    controls[1, 2] = max(controls[0, 2], controls[-1, 2])+2*peak
    receipt = {'profile': PROFILE, 'target_support_surface_z_m': support,
        'support_node_count': int(inside.sum()), 'projected_opening_bounds_xy_m': [lo.tolist(), hi.tolist()],
        'opening_posture_count': SAMPLES, 'cooked_hand_collider_count': count,
        'required_wrist_z_m': required.tolist(), 'monotone_wrist_z_m': monotone.tolist(),
        'physical_thickness_m': thickness, 'incoming_layer_allowance_m': 2*thickness,
        'geometric_clearance_m': .002, 'servo_contact_allowance_m': .001,
        'measured_kinematic_point_error_m': error, 'closed_placement_capture_z_m': float(controls[-1, 2]),
        'start_original_joint_targets_rad': start, 'end_original_joint_targets_rad': opened,
        'cloth_state_written': False, 'scope': 'Original closed cooked hand low placement; frozen pretransport support, real wrist lifting during opening still requires native review.'}
    return {**arc, 'controls_m': controls.tolist(), 'release_height_m': float(controls[-1, 2]-support),
        'release_height_basis': 'Actual closed cooked hand with a separate released opening lift',
        'release_clearance': receipt}


def rebind_opening(placement, *, side, model, start_targets, opened,
                   actual_joint_positions, actual_flange, body_q, body_names, colliders):
    if placement.get('profile') != PROFILE:
        raise ValueError('Explicit dynamic cooked placement receipt required')
    if (placement['start_original_joint_targets_rad'] != start_targets
            or placement['end_original_joint_targets_rad'] != opened):
        raise ValueError('Opening changed the planned held or fully-open original targets')
    minima, _, _, error, count = _geometry(side=side, model=model, start_targets=start_targets,
        opened=opened, actual_joint_positions=actual_joint_positions, actual_flange=actual_flange,
        body_q=body_q, body_names=body_names, colliders=colliders)
    required, monotone = height_envelope(minima, placement['target_support_surface_z_m'],
        thickness=placement['physical_thickness_m'], model_error_m=error)
    lift = np.maximum(0., monotone-actual_flange[2, 3])
    if lift.max() > .08:
        raise ValueError('Actual release pose needs more than80mm wrist lift')
    return {'profile': PROFILE, 'actual_start_wrist_world_matrix': actual_flange.tolist(),
        'opening_posture_count': SAMPLES, 'cooked_hand_collider_count': count,
        'required_wrist_z_m': required.tolist(), 'wrist_lift_m': lift.tolist(),
        'target_support_surface_z_m': placement['target_support_surface_z_m'],
        'measured_kinematic_point_error_m': error,
        'start_original_joint_targets_rad': dict(start_targets),
        'end_original_joint_targets_rad': dict(opened),
        'interpolation': 'Upper endpoint bins of1001-pose monotone cooked envelope; real wrist Z only.',
        'cloth_state_written': False}
