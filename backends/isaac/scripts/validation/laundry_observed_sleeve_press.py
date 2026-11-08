"""Plan ordinary vertical palm presses at the currently observed sleeve peaks.

This module has no simulator or attachment access. It does not reuse the cloth
shape, successful response, or collision qualification of a template execution.
Every newly planned joint sample needs the supplied original-geometry preflight.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np

from bench.evaluation.cloth.cloth_robot_collision import apply_transform
from scenes.workrooms.laundry_room.garment_folding_assets import transform_fold_frame
from scripts.validation.laundry_engineering_motion import minimum_jerk

PROFILE = 'observed_sleeve_vertical_press_v1'
CONTROL_DT_S = .02
STAGES = (
    ('actual_released_state_to_original_park', 2.),
    ('extend_all_fingers_at_park', 1.),
    ('ordinary_joints_to_palm_hover', 2.),
    ('approach_above_observed_sleeve_peak', 1.5),
    ('slow_physical_palm_compression', 4.),
    ('physical_palm_hold', .5),
    ('vertical_release_of_palm_contact', 2.),
    ('return_open_hands_to_original_park', 2.),
    ('restore_original_park_hand_posture', 1.),
    ('fully_released_settle_before_fresh_body_prepare', 1.),
)


def observed_sleeve_peaks(task, cloth_q, control_masks, assist_enabled):
    """Select by original material identity; ties select the lowest vertex ID."""
    meta = task['asset_metadata']
    if meta.get('garment_kind') != 'shirt' or len(meta['cloths']) != 1:
        raise ValueError('Observed sleeve pressing requires one original shirt')
    row = meta['cloths'][0]
    q = np.asarray(cloth_q, dtype=float)
    flat = np.asarray(row['unfolded_reference_vertices_m'], dtype=float)
    rest = np.asarray(row['rest_vertices_m'], dtype=float)
    if q.shape != rest.shape or q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError('Finite actual positions must cover the original cloth inventory')
    if flat.shape != q.shape or not np.isfinite(flat).all():
        raise ValueError('Original unfolded material reference is required')
    if not isinstance(control_masks, dict) or set(control_masks) != {row['id']}:
        raise ValueError('All task cloth control masks are required')
    mask = np.asarray(control_masks[row['id']])
    enabled = np.asarray(assist_enabled)
    if (mask.shape != (len(q),) or not np.isfinite(mask).all() or mask.any()
            or enabled.ndim != 1 or not np.isfinite(enabled).all() or enabled.any()):
        raise ValueError('Every cloth and historical assist must already be released')
    design = meta['design']
    frame = design.get('fold_coordinate_frame')
    material = transform_fold_frame(flat, frame, inverse=True) if frame is not None else flat
    peaks = {}
    for side, fold in zip(('left', 'right'), design['fold_sequence_axis_hinge_sign_radius_m'][:2]):
        axis, hinge, sign, _ = fold
        if int(axis) not in (0, 1) or sign not in (-1, 1):
            raise ValueError('Original planar sleeve fold definitions are required')
        ids = np.flatnonzero(sign * (material[:, int(axis)] - hinge) > .015)
        if not len(ids):
            raise ValueError('Original sleeve material region cannot be empty')
        vertex = int(ids[np.argmax(q[ids, 2])])
        peaks[side] = {'vertex_id': vertex, 'world_position_m': q[vertex].tolist(),
                       'original_material_position_m': rest[vertex].tolist(),
                       'material_vertex_ids': ids.tolist()}
    return peaks, {'actual_cloth_q_sha256': hashlib.sha256(q.astype('<f8').tobytes()).hexdigest(),
                   'all_cloth_masks_zero': True, 'all_assists_disabled': True,
                   'peak_selection': 'Original sleeve material; maximum observed Z; lowest-ID tie break'}


def _hand_vertices(model, export, posture):
    fk = model.hand.forward(posture)
    pieces = []
    for row in export['colliders']:
        if row['body_name'] not in fk:
            continue
        for piece in row['pieces']:
            vertices = apply_transform(np.asarray(piece['conservative_outer_vertices'], float),
                                       np.asarray(row['mesh_to_body_column_matrix'], float))
            pieces.append(apply_transform(vertices, fk[row['body_name']]))
    if not pieces:
        raise ValueError('Original cooked whole-hand geometry is required')
    return np.concatenate(pieces)


def relocated_press_targets(template, peaks, models, export, cloth_q):
    """Retain the reviewed palm orientation/height; relocate only to fresh XY."""
    if (template.get('planned_duration_s') != 17. or template.get('attachments') is not False
            or template.get('cloth_state_writes') != 0
            or [(x['label'], x['duration_s']) for x in template['phase_endpoints']] != list(STAGES)):
        raise ValueError('The original ordinary 17-second vertical-press template is required')
    targets = {}
    for side, model in models.items():
        if side not in ('left', 'right'):
            raise ValueError('This observed two-sleeve planner requires the original dual arms')
        selected = template['selected_sides'][side]
        old = np.asarray(selected['world_flange'], float)
        if (old.shape != (4, 4) or not np.isfinite(old).all()
                or not np.allclose(old[3], [0, 0, 0, 1], atol=1.e-12, rtol=0)
                or not np.allclose(old[:3, :3].T @ old[:3, :3], np.eye(3), atol=1.e-9, rtol=0)):
            raise ValueError('A finite original rigid palm pose is required')
        press = old.copy()
        press[:2, 3] += (np.asarray(peaks[side]['world_position_m'])[:2]
                         - np.asarray(template['peaks'][side]['peak_world'])[:2])
        zero = {name: 0. for name in model.hand.names}
        local = _hand_vertices(model, export, zero)
        minimum = float(apply_transform(local, press)[:, 2].min())
        # All lateral relocation is airborne. Before descending, even the
        # lowest finger is above ALL currently observed cloth by 2 mm.
        # Include the unchanged .5 mm position/.005 rad IK residual budget;
        # actual FK is checked again at every noncontact sample below.
        rise = max(0., float(np.asarray(cloth_q)[:, 2].max()) + .003 - minimum)
        precontact = press.copy()
        precontact[2, 3] += rise
        hover = press.copy()
        hover[2, 3] += max(.10, rise + .02)
        withdrawn = press.copy()
        withdrawn[2, 3] += max(.10, rise + .02)
        targets[side] = {'press': press, 'precontact': precontact, 'hover': hover,
                         'withdrawn': withdrawn, 'zero_hand_posture': zero,
                         'original_press': old,
                         'press_whole_hand_min_z_m': minimum,
                         'precontact_whole_cloth_clearance_m': minimum + rise - float(np.max(cloth_q[:, 2]))}
    if set(targets) != {'left', 'right'}:
        raise ValueError('Both original hands are required')
    return targets


def plan_observed_sleeve_press(task, cloth_q, actual_joints, initial, models, export,
                              template, *, control_masks, assist_enabled, collision_preflight):
    """Construct and preflight a fresh 17 s plan; never execute it.

    collision_preflight(samples) must independently check the entire newly
    generated sequence against original environment, robot and cooked hand
    geometry. It returns one indexed, passing row per sample and its provenance.
    A missing, partial or failing review cannot return an accepted plan.
    """
    if not callable(collision_preflight):
        raise ValueError('Fresh complete original-geometry preflight is mandatory')
    q = np.asarray(cloth_q, float)
    peaks, observation = observed_sleeve_peaks(task, q, control_masks, assist_enabled)
    targets = relocated_press_targets(template, peaks, models, export, q)
    park = deepcopy(initial['joint_positions_rad'])
    joints = {j.name: j for model in models.values() for chain in (model.arm, model.hand)
              for j in chain.active}
    if set(actual_joints) != set(park) or set(park) != set(joints):
        raise ValueError('Actual and original-park joints must cover the exact robot')
    seed = {n: float(v) for n, v in actual_joints.items()}
    samples, ik_rows, endpoints, floors = [], [], [], []
    elapsed, maximum_velocity_fraction = 0., 0.

    def record(label, t, contact=False):
        nonlocal maximum_velocity_fraction
        for name, joint in joints.items():
            value = seed[name]
            if not np.isfinite(value) or not joint.lower <= value <= joint.upper:
                raise ValueError('Fresh press exceeds an original joint limit')
            if samples:
                dt = t - samples[-1]['time_s']
                delta = abs(value - samples[-1]['joint_positions_rad'][name])
                if dt < 1.e-10:
                    if delta > 1.e-12:
                        raise ValueError('Discontinuous ordinary press stage boundary')
                elif delta / dt > joint.velocity + 1.e-9:
                    raise ValueError('Fresh press exceeds an original joint velocity')
                elif joint.velocity > 0:
                    maximum_velocity_fraction = max(maximum_velocity_fraction, delta / dt / joint.velocity)
        if any(m.hand_posture_prior.violation(finger, seed) for m in models.values()
               for finger in ('thumb', 'index', 'middle', 'ring', 'little')):
            raise ValueError('Fresh press exceeds an original hand posture prior')
        floor_row = {}
        for side, model in models.items():
            flange = model.arm.forward(seed, model.arm_base)[model.arm_tip] @ model.wrist_offset
            vertices = apply_transform(_hand_vertices(model, export, seed), flange)
            gap = float(vertices[:, 2].min() - q[:, 2].max())
            if not contact and gap < .002 - 1.e-12:
                raise ValueError('Airborne palm relocation does not clear the observed whole cloth')
            floor_row[side] = {'whole_hand_min_z_m': float(vertices[:, 2].min()),
                               'observed_whole_cloth_clearance_m': gap}
        floors.append({'time_s': float(t), 'label': label,
                       'intentional_cloth_compression_phase': contact, 'sides': floor_row})
        samples.append({'label': label, 'time_s': float(t), 'joint_positions_rad': dict(seed),
                        'intentional_cloth_compression_phase': contact})

    def joint_segment(end, duration, label):
        nonlocal seed, elapsed
        start = dict(seed)
        for i in range(round(duration / CONTROL_DT_S) + 1):
            u = minimum_jerk(i * CONTROL_DT_S, 0., duration)
            seed = {n: float(start[n] + u * (end[n] - start[n])) for n in start}
            record(label, elapsed + i * CONTROL_DT_S)
        elapsed += duration
        endpoints.append({'label': label, 'duration_s': duration, 'joint_positions_rad': dict(seed)})

    def solve(side, target, label, time_s, hint=None):
        model = models[side]
        guess = seed if hint is None else {**seed, **hint}
        solution = model.arm.inverse(np.linalg.inv(model.arm_base) @ target @ np.linalg.inv(model.wrist_offset),
                                    model.arm_tip, guess, iterations=180, multi_starts=0,
                                    position_tolerance=.0005, rotation_tolerance=.005)
        ik_rows.append({'side': side, 'label': label, 'time_s': time_s, **solution})
        if not solution['converged']:
            raise ValueError(f'Fresh {side} {label} is not reachable with original IK tolerances')
        return solution['joint_positions_rad']

    def cartesian_segment(key, duration, label, contact=False):
        nonlocal seed, elapsed
        start = {side: m.arm.forward(seed, m.arm_base)[m.arm_tip] @ m.wrist_offset
                 for side, m in models.items()}
        for i in range(round(duration / CONTROL_DT_S) + 1):
            u = minimum_jerk(i * CONTROL_DT_S, 0., duration)
            if i:  # The exact actual-FK boundary keeps the already issued joints.
                for side in models:
                    end = targets[side][key]
                    if np.linalg.norm(start[side][:3, :3] - end[:3, :3]) >= .006:
                        raise ValueError('Vertical pressing cannot change the reviewed palm orientation')
                    target = start[side].copy()
                    target[:3, 3] = (1-u) * start[side][:3, 3] + u * end[:3, 3]
                    seed.update(solve(side, target, label, elapsed + i * CONTROL_DT_S))
            record(label, elapsed + i * CONTROL_DT_S, contact)
        elapsed += duration
        endpoints.append({'label': label, 'duration_s': duration, 'joint_positions_rad': dict(seed)})

    joint_segment(park, *STAGES[0][::-1])
    openpark = {**park, **{n: v for t in targets.values() for n, v in t['zero_hand_posture'].items()}}
    joint_segment(openpark, *STAGES[1][::-1])
    hover = dict(openpark)
    for side in models:
        hint = template['selected_sides'][side]['upward_branch_samples'][-1]['ik']['joint_positions_rad']
        hover.update(solve(side, targets[side]['hover'], 'fresh_hover_endpoint', elapsed, hint))
    joint_segment(hover, *STAGES[2][::-1])
    for i, key in ((3, 'precontact'), (4, 'press'), (5, 'press'), (6, 'withdrawn')):
        cartesian_segment(key, *STAGES[i][::-1], contact=i >= 4)
    joint_segment(openpark, *STAGES[7][::-1])
    joint_segment(park, *STAGES[8][::-1])
    joint_segment(park, *STAGES[9][::-1])
    review = collision_preflight(deepcopy(samples))
    rows = review.get('samples', []) if isinstance(review, dict) else []
    if (not isinstance(review, dict) or review.get('all_clear') is not True or len(rows) != len(samples)
            or any(type(r.get('sample_index')) is not int or r['sample_index'] != i
                   or r.get('passed') is not True for i, r in enumerate(rows))
            or set(review.get('checked_geometry', [])) != {'environment', 'arm_self', 'dual_arm', 'whole_cooked_hand'}):
        raise ValueError('Fresh original-geometry review must cover every generated press sample')
    return {'schema': 'laundry_observed_sleeve_vertical_press_plan.v1', 'profile': PROFILE,
            'planned_duration_s': elapsed, 'control_dt_s': CONTROL_DT_S, 'peaks': peaks,
            'observation': observation, 'source_actual_joint_positions': dict(actual_joints),
            'phase_endpoints': endpoints, 'samples': samples, 'ordinary_ik': ik_rows,
            'cooked_floor_samples': floors,
            'maximum_original_joint_velocity_fraction': maximum_velocity_fraction,
            'source_clearance': review, 'error': None, 'all_passed': True,
            'attachments': False, 'cloth_state_writes': 0,
            'targets': {s: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in t.items()}
                        for s, t in targets.items()},
            'scope': 'Fresh observed-material vertical ordinary-palm plan. All lateral travel is airborne; '
                     'no loaded lateral sweep. Sampled original-geometry preflight is not continuous '
                     'collision, force, observed cloth response or task qualification.'}
