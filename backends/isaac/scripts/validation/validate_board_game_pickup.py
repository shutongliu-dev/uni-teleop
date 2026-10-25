"""Privileged single-reserve pickup diagnostic, with ordinary joint targets only.

Requires a frozen release, complete source snapshot and an explicit mesh-reviewed
grasp plan. The untouched classical bimanual controller receives robot feedback;
object truth is private planning/evaluation input. No model, object reset override,
pose playback, attachment, friction edit or game move is involved.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import traceback

import numpy as np

from bench.paths import OUTPUT_ROOT, load_json
from bench.protocol.schemas import AgentObservation, JointLimit, System0Action
from bench.control.factory import configured_policy, robot_configuration
from bench.geometry import matrix_quaternion, quaternion_matrix
from bench.isaacsim.release import dump
from bench.simulation.isaacsim_backend.contact_sample import SubstepContactExchange
from bench.simulation.isaacsim_backend.contact_precision import aggregation_metadata, summarize_normal_impulses
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.validate_board_game import load_release, table_contact_paths
from scripts.validation.validate_scenario import digest, seal_report_before_shutdown
from scripts.validation.validate_workrooms import check_window, plain


@dataclass(frozen=True)
class PickupLimits:
    initial_settle_s: float = .5
    approach_height_m: float = .10
    lift_command_m: float = .05
    cartesian_segment_m: float = .02
    minimum_bottom_clearance_m: float = .02
    airborne_hold_s: float = .30
    pinch_window_s: float = .10
    minimum_finger_contact_fraction: float = .90
    maximum_finger_contact_gap_s: float = .01
    minimum_normal_impulse_ns: float = 1.e-8
    handle_sidewall_tolerance_m: float = .003
    handle_end_margin_m: float = .004
    maximum_reported_penetration_m: float = .002
    maximum_hold_speed_m_s: float = .04
    maximum_hold_angular_speed_rad_s: float = .3
    maximum_neighbor_displacement_m: float = .005
    maximum_neighbor_lift_m: float = .003
    maximum_sim_time_s: float = 90.


def checked_plan(path, policy, side, geometry):
    plan = load_json(Path(path))
    if plan.get('schema') != 'board_game_pickup_plan.v1':
        raise ValueError('Expected board_game_pickup_plan.v1')
    if plan.get('kinematic_assets_sha256') != policy.geometry.asset_sha256:
        raise ValueError('Grasp plan must bind the actual complete bimanual kinematic asset inventory')
    for name, expected in plan['kinematic_assets_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Grasp-plan robot asset hash differs: '+name)
    row = plan['sides'][side]
    single = policy.geometry.sides[side]
    for key in ('open_hand_posture_rad', 'closed_hand_posture_rad'):
        values = row[key]
        if set(values) != set(single.hand.names):
            raise ValueError('Each hand posture must specify all 22 actual side joints')
        for joint in single.hand.active:
            value = values[joint.name]
            if type(value) not in (int, float) or not math.isfinite(value) or not joint.lower <= value <= joint.upper:
                raise ValueError('Plan posture exceeds its pinned URDF: '+joint.name)
        if any(single.hand_posture_prior.violation(finger, values) for finger in single.finger_joints):
            raise ValueError('Plan posture violates the unchanged controller posture prior')
    center = np.asarray(row['pinch_center_hand_m'], float)
    rotation = np.asarray(row['wrist_rotation_in_arm_base'], float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Finite hand-local pinch centre required')
    if (rotation.shape != (3,3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1.e-6, rtol=0)
            or abs(np.linalg.det(rotation)-1) > 1.e-6):
        raise ValueError('Proper arm-base-relative wrist rotation required')
    height = row['handle_contact_height_from_base_origin_m']
    low = geometry.piece_base_height_m/2
    if type(height) not in (int, float) or not low+.004 <= height <= low+geometry.handle_height_m-.004:
        raise ValueError('Pinch centre must lie inside the handle away from its end caps')
    review = row['mesh_review']
    if not review.get('path') or digest(review['path']) != review.get('sha256'):
        raise ValueError('Freeze an explicit source-mesh review artifact before native execution')
    return plan, row


def contact_rows(env, sample):
    """Project the same six native buffers, preserving actual contact points."""
    impulses, points, normals, separations, counts, starts = sample.for_env(env)
    counts, starts = np.asarray(counts), np.asarray(starts)
    active = np.argwhere(counts > 0)
    rows, seen = [], set()
    for sensor, filter_index in active:
        start, count = int(starts[sensor, filter_index]), int(counts[sensor, filter_index])
        if start < 0 or start+count >= env._rigid_contacts.max_contact_data_count:
            raise RuntimeError('Native contact point buffer capacity reached')
        pair = (env._contact_sensor_paths[sensor], env._contact_filter_paths[sensor][filter_index])
        key = tuple(sorted(pair))
        if key in seen:
            continue
        seen.add(key)
        selected = slice(start, start+count)
        f, p, n, d = (np.asarray(x[selected]) for x in (impulses, points, normals, separations))
        if not all(np.isfinite(x).all() for x in (f,p,n,d)):
            fields = {name: int(np.count_nonzero(~np.isfinite(value)))
                for name, value in zip(('impulses', 'points', 'normals', 'separations'), (f,p,n,d))}
            raise RuntimeError('Invalid raw native contact point evidence: '+json.dumps({
                'actors': pair, 'sensor_index': int(sensor), 'filter_index': int(filter_index),
                'start': start, 'count': count, 'nonfinite_counts': fields,
                'negative_impulse_count': int(np.count_nonzero(f < 0)),
                'minimum_finite_impulse_ns': float(f[np.isfinite(f)].min()) if np.isfinite(f).any() else None}))
        rows.append({'actor0': pair[0], 'actor1': pair[1], 'normal_impulses_ns': f.reshape(-1).tolist(),
                     'points_m': p.reshape(-1,3).tolist(), 'normals': n.reshape(-1,3).tolist(),
                     'separations_m': d.reshape(-1).tolist(),
                     **summarize_normal_impulses(f.reshape(-1), n.reshape(-1,3))})
    return rows


def handle_contacts(rows, target_path, pads, target_pose, geometry, limits, finger_parts=None):
    """Classify reported world contact points against the actual token pose.

    No collider IDs are invented: sidewall membership is a geometric check on
    reported points. The run must separately verify this frame on table support.
    """
    rotation = quaternion_matrix(np.asarray(target_pose)[[6,3,4,5]])
    finger_parts = finger_parts or {finger: {path: 'pad'} for finger, path in pads.items()}
    samples = {finger: [] for finger in pads}
    pad_samples = {finger: [] for finger in pads}
    forces = dict.fromkeys(pads, 0.)
    parts = {finger: {} for finger in pads}
    for row in rows:
        pair = (row['actor0'], row['actor1'])
        if target_path not in pair:
            continue
        other = pair[1] if pair[0] == target_path else pair[0]
        for finger, accepted in finger_parts.items():
            if other not in accepted:
                continue
            for scalar, normal, point, separation in zip(row['normal_impulses_ns'], row['normals'], row['points_m'], row['separations_m']):
                impulse = float(np.linalg.norm(float(scalar)*np.asarray(normal, dtype=np.float64)))
                # Native signed values are retained. A negative scalar still
                # counts as contact/support, but is not affirmative pressure
                # evidence for this conservative grasp qualification.
                if scalar <= 0 or impulse <= limits.minimum_normal_impulse_ns:
                    continue
                local = (np.asarray(point)-np.asarray(target_pose[:3])) @ rotation
                radial = np.linalg.norm(local[:2])
                low = geometry.piece_base_height_m/2+limits.handle_end_margin_m
                high = geometry.piece_base_height_m/2+geometry.handle_height_m-limits.handle_end_margin_m
                if (low <= local[2] <= high and abs(radial-geometry.handle_radius_m) <= limits.handle_sidewall_tolerance_m
                        and separation >= -limits.maximum_reported_penetration_m):
                    samples[finger].append(local.tolist())
                    forces[finger] += impulse
                    part = parts[finger].setdefault(other, {'component': accepted[other], 'normal_impulse_ns': 0.,
                        'local_contact_points_m': []})
                    part['normal_impulse_ns'] += impulse
                    part['local_contact_points_m'].append(local.tolist())
                    if other == pads[finger]:
                        pad_samples[finger].append(local.tolist())
    def opposed(points):
        if not all(points[finger] for finger in pads):
            return False
        a, b = (np.mean(points[finger], axis=0)[:2] for finger in ('thumb','index'))
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        return bool(denominator > 1.e-12 and float(a@b/denominator) <= -.5)
    return {'both_fingers_on_opposed_handle_sides': opposed(samples),
            'both_pads_on_opposed_handle_sides': opposed(pad_samples),
            'local_contact_points_m': samples, 'handle_normal_impulses_ns': forces,
            'actual_handle_contact_components': parts}


def token_lowest_z_m(target_pose, geometry):
    """Lowest world Z of both physical cylinders at a base-centred xyzw pose.

    An inverted or strongly tilted handle can lie below the base cylinder.
    Evaluate each entire cylinder, including both end caps, before taking the
    union's minimum; the reserve token need not remain upright during a probe.
    """
    pose = np.asarray(target_pose, float)
    cosine = float(np.clip(quaternion_matrix(pose[[6,3,4,5]])[2,2], -1., 1.))
    sine = math.sqrt(max(0., 1.-cosine*cosine))
    base_half = geometry.piece_base_height_m/2
    handle_half = geometry.handle_height_m/2
    base_lowest = pose[2]-base_half*abs(cosine)-geometry.piece_radius_m*sine
    handle_lowest = (pose[2]+(base_half+handle_half)*cosine
        -handle_half*abs(cosine)-geometry.handle_radius_m*sine)
    return float(min(base_lowest, handle_lowest))


class PickupObserver:
    """Read-only native-step hook; its exchange keeps one contact acquisition."""
    def __init__(self, env, output, piece_id, side, geometry, limits, support_spec):
        self.env, self.output, self.piece_id, self.side = env, output, piece_id, side
        self.geometry, self.limits = geometry, limits
        self.target_index = env.object_names.index(piece_id)
        self.target_path = env.scene_info['objects'][piece_id]['path']
        self.robot_paths = set(env.scene_info['robot_link_paths'].values())
        self.pads = {finger: env.scene_info['robot_link_paths'][side+'_'+finger+'_elastomer'] for finger in ('thumb','index')}
        self.finger_parts = {finger: {path: ('pad' if name.endswith('_elastomer') else 'distal_shell')
            for name, path in env.scene_info['robot_link_paths'].items()
            if name in (side+'_'+finger+'_elastomer', side+'_'+finger+'_DP')} for finger in self.pads}
        self.accepted_finger_paths = {path for parts in self.finger_parts.values() for path in parts}
        self.table_paths = set(table_contact_paths(env.scene_info))
        self.reserve_support = support_spec[piece_id]
        self.support_paths = set(self.reserve_support['paths'])
        self.surface_z = float(self.reserve_support['surface_z_m'])
        sensors = set(env._contact_sensor_paths)
        required_sensors = self.robot_paths | {row['path'] for row in env.scene_info['objects'].values()}
        required_filters = required_sensors | self.support_paths | self.table_paths
        required_filters.update(env.scene_info.get('support_paths', []))
        required_filters.update(path for fixture in env.scene_info.get('fixtures', {}).values()
            for path in fixture.get('contact_filter_paths', fixture.get('collision_paths', [])))
        actors = env.scene_info.get('collider_actor_paths', {})
        required_filters = {actors.get(path, path) for path in required_filters}
        if (not required_sensors.issubset(sensors) or any(not required_filters.issubset(filters)
                for path, filters in zip(env._contact_sensor_paths, env._contact_filter_paths)
                if path in required_sensors)):
            raise RuntimeError('Pickup evidence requires complete robot/token/environment contact filter coverage')
        self.initial = env.capture_state()
        self.original = env.world.step
        self.callback = self._step
        self.exchange = SubstepContactExchange(env, self)
        self.stream = (output/'native_frames.jsonl').open('x')
        self.samples, self.phase, self.last = 0, 'initial_settle', None
        self.window = deque()
        self.pad_window = deque()
        self.hold_started = None
        self.pad_hold_started = None
        self.maximum_airborne_hold_s = 0.
        self.maximum_both_pads_airborne_hold_s = 0.
        self.maximum_neighbor_displacement = self.maximum_neighbor_lift = 0.
        self.interferences = {}
        self.point_frame_verified_samples = 0
        self.max_penetration = 0.
        self.closed = False
        env.world.step = self.callback

    def _step(self, *args, **kwargs):
        if self.closed or args or kwargs != {'render': False}:
            raise RuntimeError('Pickup observer permits only the existing native step')
        env = self.env
        self.exchange.begin_step(env)
        result = self.original(*args, **kwargs)
        sample = self.exchange.acquire(env)
        before = None
        rejection_stage = 'native_state_capture'
        try:
            before = env.capture_state()
            if not all(np.isfinite(value).all() for value in before.values()):
                raise RuntimeError('Nonfinite native pickup state')
            rejection_stage = 'raw_contact_projection'
            raw = contact_rows(env, sample)
            rejection_stage = 'rigid_contact_projection'
            projected = env._project_rigid_contacts(sample)
            rejection_stage = 'target_pose_validation'
            target = before['object_q'][self.target_index]
            if abs(float(np.linalg.norm(target[3:]))-1.) > 1.e-3:
                raise RuntimeError('Invalid native target quaternion')
        except BaseException as exc:
            # Preserve the rejected acquisition itself. Never read the native
            # contact view a second time or reinterpret invalid data as zero.
            # The live-sample guard may itself have rejected the second
            # projection. Its already-copied arrays remain diagnostic evidence,
            # even when they can no longer be offered to the public reader.
            arrays = sample.arrays
            np.savez_compressed(self.output/'rejected_contact_sample.npz', **dict(zip(
                ('normal_impulses_ns', 'points_m', 'normals', 'separations_m', 'counts', 'starts'), arrays)))
            if before is not None:
                np.savez_compressed(self.output/'rejected_contact_state.npz', **before)
            dump(self.output/'rejected_contact_sample.json', {
                'schema': 'board_game_rejected_native_contact_sample.v1',
                'error_type': type(exc).__name__, 'error_detail': str(exc),
                'rejection_stage': rejection_stage, 'state_capture_available': before is not None,
                'phase': self.phase, 'simulation_time_s': float(env._current_step_time),
                'native_step_index': int(env.world.current_time_step_index),
                'sensor_paths': list(env._contact_sensor_paths),
                'filter_paths': [list(paths) for paths in env._contact_filter_paths],
                'maximum_contact_data_count': int(env._rigid_contacts.max_contact_data_count),
                'original_native_acquisition_retained': True, 'sample_delivered_to_public_reader': False,
                'physical_state_writes': 0, 'extra_native_contact_acquisitions': 0})
            raise
        now = float(env._current_step_time)
        pinch = handle_contacts(raw, self.target_path, self.pads, target, self.geometry, self.limits, self.finger_parts)
        support, forbidden_target = 0., []
        relevant = []
        for row in raw:
            pair = (row['actor0'], row['actor1'])
            impulse = row['impulse_magnitude_sum_ns']
            depth = max(0., -min(row['separations_m']))
            self.max_penetration = max(self.max_penetration, depth)
            if self.target_path in pair:
                other = pair[1] if pair[0] == self.target_path else pair[0]
                if other not in self.robot_paths:
                    support += impulse
                elif other not in self.accepted_finger_paths and impulse > self.limits.minimum_normal_impulse_ns:
                    forbidden_target.append(other)
                if other in self.support_paths and impulse > self.limits.minimum_normal_impulse_ns:
                    rotation = quaternion_matrix(target[[6,3,4,5]])
                    points = (np.asarray(row['points_m'])-target[:3])@rotation
                    if (np.all(np.abs(np.asarray(row['points_m'])[:,2]-self.surface_z) <= .003)
                            and np.all(np.abs(points[:,2]+self.geometry.piece_base_height_m/2) <= .003)
                            and np.all(np.linalg.norm(points[:,:2], axis=1) <= self.geometry.piece_radius_m+.003)):
                        self.point_frame_verified_samples += 1
            robots = [p for p in pair if p in self.robot_paths]
            if robots or self.target_path in pair:
                relevant.append(row)
            if robots and impulse > self.limits.minimum_normal_impulse_ns:
                other = pair[1] if pair[0] in self.robot_paths else pair[0]
                allowed = other == self.target_path and all(p in self.accepted_finger_paths for p in robots)
                if not allowed:
                    key = '|'.join(sorted(pair))
                    record = self.interferences.setdefault(key, {'actors': list(pair), 'substeps': 0,
                        'normal_impulse_ns': 0., 'maximum_penetration_m': 0.})
                    record['substeps'] += 1
                    record['normal_impulse_ns'] += impulse
                    record['maximum_penetration_m'] = max(record['maximum_penetration_m'], depth)
        other_indices = [i for i in range(len(env.object_names)) if i != self.target_index]
        drift = np.linalg.norm(before['object_q'][other_indices,:3]-self.initial['object_q'][other_indices,:3], axis=1)
        lift = before['object_q'][other_indices,2]-self.initial['object_q'][other_indices,2]
        self.maximum_neighbor_displacement = max(self.maximum_neighbor_displacement, float(drift.max(initial=0.)))
        self.maximum_neighbor_lift = max(self.maximum_neighbor_lift, float(lift.max(initial=0.)))
        bottom = token_lowest_z_m(target, self.geometry)
        velocity = before['object_qd'][self.target_index]
        for window, key in ((self.window, 'both_fingers_on_opposed_handle_sides'),
                            (self.pad_window, 'both_pads_on_opposed_handle_sides')):
            window.append((now, pinch[key]))
            while window and now-window[0][0] > self.limits.pinch_window_s+1.e-12:
                window.popleft()
        qualified_pinch = self.pinch_ready()
        airborne = (bottom-self.surface_z >= self.limits.minimum_bottom_clearance_m
            and support <= self.limits.minimum_normal_impulse_ns and qualified_pinch and not forbidden_target
            and np.linalg.norm(velocity[:3]) <= self.limits.maximum_hold_speed_m_s
            and np.linalg.norm(velocity[3:]) <= self.limits.maximum_hold_angular_speed_rad_s)
        self.hold_started = (self.hold_started if self.hold_started is not None else now) if airborne else None
        dwell = 0. if self.hold_started is None else now-self.hold_started
        self.maximum_airborne_hold_s = max(self.maximum_airborne_hold_s, dwell)
        pad_airborne = airborne and self.pinch_ready(self.pad_window)
        self.pad_hold_started = (self.pad_hold_started if self.pad_hold_started is not None else now) if pad_airborne else None
        pad_dwell = 0. if self.pad_hold_started is None else now-self.pad_hold_started
        self.maximum_both_pads_airborne_hold_s = max(self.maximum_both_pads_airborne_hold_s, pad_dwell)
        after = env.capture_state()
        if not all(np.array_equal(before[name], after[name]) for name in before):
            raise RuntimeError('Pickup evidence read changed physical state')
        record = {'native_step_index': int(env.world.current_time_step_index), 'simulation_time_s': now,
            'phase': self.phase, 'piece_id': self.piece_id, 'target_pose_xyzw': target.tolist(),
            'target_velocity_world_com': velocity.tolist(), 'all_object_poses_xyzw': before['object_q'].tolist(),
            'pinch': pinch, 'pinch_window_qualified': qualified_pinch, 'other_support_impulse_ns': support,
            'forbidden_target_robot_contacts': forbidden_target, 'token_bottom_m': float(bottom),
            'bottom_geometry': 'complete_base_and_handle_cylinders_v1',
            'airborne_stable_dwell_s': dwell, 'both_pads_airborne_stable_dwell_s': pad_dwell,
            'raw_contacts': relevant, 'state_unchanged_by_observer': True}
        self.stream.write(json.dumps(plain(record), allow_nan=False)+'\n')
        self.samples += 1
        self.last = record
        self.exchange.offer_after_readonly(env, sample, projected)
        return result

    def pinch_ready(self, window=None):
        window = self.window if window is None else window
        if not window or window[-1][0]-window[0][0]+1.e-12 < self.limits.pinch_window_s:
            return False
        fraction = sum(good for _,good in window)/len(window)
        bad, longest = 0, 0
        for _, good in window:
            bad = 0 if good else bad+1
            longest = max(longest, bad)
        return bool(fraction >= self.limits.minimum_finger_contact_fraction
                    and longest*self.env.dt <= self.limits.maximum_finger_contact_gap_s+1.e-12)

    def close(self):
        if self.closed:
            raise RuntimeError('Pickup observer already closed')
        self.closed = True
        owner = self.env.world.step == self.callback
        if owner:
            self.env.world.step = self.original
        try:
            delivery = self.exchange.close(self.env, self)
        finally:
            self.stream.close()
        # Incidental environmental contact and neighbor motion are reported,
        # separately from whether the selected token was physically picked up.
        extra_target_contacts = [r for r in self.interferences.values() if self.target_path in r['actors']]
        blocked = [r for r in self.interferences.values() if self.target_path not in r['actors']
            and not all(p in self.robot_paths for p in r['actors'])]
        checks = {'one_contact_acquisition_per_native_step': delivery['all_samples_consumed_once']
                and delivery['acquisitions'] == self.samples,
            'native_step_hook_restored': owner, 'contact_point_frame_verified_on_declared_support': self.point_frame_verified_samples > 0,
            'opposed_handle_grip_and_actual_stable_lift': bool(self.last)
                and self.last['airborne_stable_dwell_s']+1.e-12 >= self.limits.airborne_hold_s,
            'reported_penetration_within_limit': self.max_penetration <= self.limits.maximum_reported_penetration_m}
        return plain({'passed': all(checks.values()), 'checks': checks, 'samples': self.samples,
            'both_pads_preferred_pickup': bool(self.last) and self.last['both_pads_airborne_stable_dwell_s']+1.e-12 >= self.limits.airborne_hold_s,
            'both_pad_evidence_scope': 'Both pads meet opposed handle-contact checks; additional distal-shell contact is not excluded',
            'no_robot_environment_or_other_token_contact': not blocked,
            'additional_target_robot_contact_pairs': extra_target_contacts,
            'neighbors_remain_in_place': self.maximum_neighbor_displacement <= self.limits.maximum_neighbor_displacement_m
                and self.maximum_neighbor_lift <= self.limits.maximum_neighbor_lift_m,
            'maximum_airborne_hold_s': self.maximum_airborne_hold_s,
            'maximum_both_pads_airborne_hold_s': self.maximum_both_pads_airborne_hold_s,
            'maximum_neighbor_displacement_m': self.maximum_neighbor_displacement,
            'maximum_neighbor_lift_m': self.maximum_neighbor_lift, 'maximum_reported_penetration_m': self.max_penetration,
            'interference_pairs': list(self.interferences.values()), 'contact_delivery': delivery,
            'actual_reserve_support': self.reserve_support,
            'physical_steps_added': 0, 'physical_state_setters_called': 0,
            'point_evidence_scope': 'Actual native contact positions; handle membership geometrically classified, collider IDs unavailable'})


def controller_observation(env, sequence):
    raw = env.observation()
    return AgentObservation(episode_id='pickup_diagnostic', sequence=sequence, sim_time_s=env.time_s,
        instruction='Private benchmark pickup diagnostic', profile='vision_proprio',
        proprioception={'joint_positions_rad': raw['joint_positions_rad'],
                        'joint_velocities_rad_s': raw['joint_velocities_rad_s']})


def motion_phases(policy, side, row, target_pose, geometry, limits, measured):
    single = policy.geometry.sides[side]
    rotation = single.arm_base[:3,:3] @ np.asarray(row['wrist_rotation_in_arm_base'])
    if quaternion_matrix(target_pose[[6,3,4,5]])[2,2] < math.cos(math.radians(5)):
        raise ValueError('The reserve token must be upright before this bounded pickup probe')
    center = target_pose[:3]+quaternion_matrix(target_pose[[6,3,4,5]]) @ np.array(
        [0., 0., row['handle_contact_height_from_base_origin_m']])
    flange = center-rotation@np.asarray(row['pinch_center_hand_m'])
    above = flange+[0.,0.,limits.approach_height_m]
    current = single.public_geometry({name: measured[name] for name in single.names})['wrist']
    def pose(position, orientation=rotation):
        return {'frame_id': 'world', 'position_m': np.asarray(position).tolist(),
                'quaternion_wxyz': matrix_quaternion(orientation).tolist()}
    opened = row['open_hand_posture_rad']
    raised = np.asarray(current['position_m']).copy()
    raised[2] = max(raised[2], above[2]+.05)
    phases = [{'name': 'raise', 'wrist': {**current, 'position_m': raised.tolist()}},
        {'name': 'preshape', 'hand_posture_rad': opened},
        {'name': 'orient', 'wrist': pose(raised)}, {'name': 'above', 'wrist': pose(above)}]
    for i in range(1, math.ceil(limits.approach_height_m/limits.cartesian_segment_m)+1):
        fraction = i/math.ceil(limits.approach_height_m/limits.cartesian_segment_m)
        phases.append({'name': 'descend_'+str(i), 'wrist': pose(above+fraction*(flange-above))})
    phases += [{'name': 'close', 'hand_posture_rad': row['closed_hand_posture_rad']},
               {'name': 'contact_hold', 'hold_s': .2}]
    for i in range(1, math.ceil(limits.lift_command_m/limits.cartesian_segment_m)+1):
        height = limits.lift_command_m*i/math.ceil(limits.lift_command_m/limits.cartesian_segment_m)
        phases.append({'name': 'lift_'+str(i), 'wrist': pose(flange+[0.,0.,height])})
    phases.append({'name': 'airborne_hold', 'hold_s': .6})
    return phases


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Pickup artifacts belong under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    env = observer = None
    error, physical, phases_done, states = None, None, [], []
    limits = PickupLimits()
    try:
        root = args.project_root.resolve()
        if root != Path(__file__).resolve().parents[2]:
            raise ValueError('--project-root must be the actual frozen source root')
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        verified = verify_if_snapshot(root, required_modules=('scripts.validation.validate_board_game_pickup', 'bench.simulation.isaacsim_backend.env',
            'bench.control.factory', 'bench.simulation.isaacsim_backend.contact_sample'), entrypoint_file=Path(__file__))
        if verified is None:
            raise ValueError('Complete frozen source required')
        dump(output/'runtime_imports.json', verified)
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not uuid.startswith('GPU-') or ',' in uuid or os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1':
            raise ValueError('Select one physical GPU UUID in single-CUDA mode')
        scene, task, initial, geometry, settlement, release_identity = load_release(args.release)
        expected_color = 'black' if args.side == 'left' else 'white'
        if not args.piece_id.startswith(expected_color+'_') or args.piece_id not in {x['id'] for x in scene['objects']}:
            raise ValueError('Select an existing reserve token owned by this side')
        if not {'assets_root', 'base_position_m', 'assembly_version', 'arm_base_poses'} <= scene['robot'].keys():
            raise ValueError('Pickup requires explicit robot assets, assembly and opposing bases')
        policy = configured_policy(robot_configuration(scene))
        plan, selected_plan = checked_plan(args.grasp_plan, policy, args.side, geometry)
        (output/'grasp_plan.json').write_bytes(args.grasp_plan.read_bytes())
        dump(output/'plan.json', {'schema': 'board_game_pickup_run_plan.v1', 'release_identity': release_identity,
            'grasp_plan_path': str(args.grasp_plan.resolve()), 'grasp_plan_sha256': digest(args.grasp_plan),
            'piece_id': args.piece_id, 'side': args.side, 'limits': asdict(limits), 'model_calls': 0,
            'default_release_reset': True, 'reset_overrides': False, 'gpu_uuid': uuid,
            'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
            'contact_impulse_arithmetic': aggregation_metadata(),
            'grip_pressure_evidence': 'Positive native scalar only; support and interference use all signed point vector magnitudes',
            'controller_provenance': policy.provenance, 'controller_descriptor': policy.descriptor.model_dump(mode='json'),
            'scope': 'One privileged reserve accessibility/pickup diagnostic; not a game move or model score'})
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', render_width=320, render_height=240)
        env.reset(seed=args.seed, initial_state=initial)
        states.append(env.capture_state())
        dump(output/'scene_info.json', env.scene_info)
        # The shared preflight owns the complete frozen per-piece support binding.
        from scripts.validation.validate_board_game import reserve_support_spec
        from scenes.board_games.tabletop import BoardGameMatch
        supports = reserve_support_spec(env, BoardGameMatch(geometry, thresholds=settlement))
        observer = PickupObserver(env, output, args.piece_id, args.side, geometry, limits, supports)
        seq = 1
        caps = SimpleNamespace(control_period_s=.02, joint_limits={j.name: JointLimit(lower_rad=j.lower, upper_rad=j.upper) for j in policy.geometry.joints})
        policy.reset(SimpleNamespace(episode_id='pickup_diagnostic', observation_profile='vision_proprio'), caps)
        with (output/'actions.jsonl').open('x') as action_stream, (output/'phases.jsonl').open('x') as phase_stream:
            def step(command):
                nonlocal seq
                if env.time_s >= limits.maximum_sim_time_s:
                    raise RuntimeError('Bounded pickup simulation budget exhausted')
                action_stream.write(json.dumps({'sequence': seq, 'time_s': env.time_s, 'phase': observer.phase,
                                                'action': command}, allow_nan=False)+'\n')
                previous = states[-1]
                raw = env.step(command)
                states.append(env.capture_state())
                check_window(raw, states[-1], previous, env.dt)
                seq += 1
            hold = {'joint_positions_rad': initial['joint_positions_rad'], 'duration_s': .02}
            for _ in range(round(limits.initial_settle_s/.02)):
                step(hold)
            if observer.point_frame_verified_samples < round(limits.pinch_window_s/env.dt):
                raise RuntimeError('Declared reserve support did not verify contact point coordinates for at least 100ms')
            if (observer.last['other_support_impulse_ns'] <= limits.minimum_normal_impulse_ns
                    or np.linalg.norm(observer.last['target_velocity_world_com'][:3]) > limits.maximum_hold_speed_m_s):
                raise RuntimeError('Reserve token must actually rest on support before the planned approach')
            target = env.capture_state()['object_q'][env.object_names.index(args.piece_id)]
            phases = motion_phases(policy, args.side, selected_plan, target, geometry, limits,
                                    env.observation()['joint_positions_rad'])
            dump(output/'motion_plan.json', plain(phases))
            for i, phase in enumerate(phases):
                observer.phase = phase['name']
                if phase['name'] == 'lift_1' and not observer.pinch_ready():
                    raise RuntimeError('Opposed native finger contact on the handle was not established before lift')
                obs = controller_observation(env, seq)
                common = dict(action_id='pickup_'+str(i), observation_sequence=seq)
                if 'hold_s' in phase:
                    action = System0Action(**common, skill='hold', hold_duration_s=phase['hold_s'], timeout_sim_s=phase['hold_s']+1.)
                else:
                    goal = {key: value for key,value in phase.items() if key != 'name'}
                    goal.update(fingertip_frame_id=args.side+'_hand_flange', position_tolerance_m=.001,
                                orientation_tolerance_rad=.01)
                    action = System0Action(**common, skill='move_geometry', timeout_sim_s=12., hand_geometries={args.side: goal})
                policy.begin(action, obs)
                start = env.time_s
                contact_triggered_stop = False
                while True:
                    result = policy.step(controller_observation(env, seq))
                    if result.status != 'running':
                        break
                    try:
                        step(result.command.model_dump(mode='json'))
                    except BaseException:
                        policy.discard_pending_step()
                        raise
                    policy.commit_pending_step()
                    if phase['name'] == 'close' and observer.pinch_ready():
                        # Stop advancing the closing target through the existing
                        # bounded cancellation path; the next hold executes its
                        # ordinary continuous-stop commands, then preserves them.
                        policy.cancel('observed_opposed_handle_contact')
                        contact_triggered_stop = True
                        break
                row = {'phase': phase['name'], 'start_time_s': start, 'end_time_s': env.time_s,
                       'controller_result': result.model_dump(mode='json') if hasattr(result, 'model_dump') else plain(asdict(result)),
                       'pinch_window_qualified': observer.pinch_ready(),
                       'contact_triggered_stop': contact_triggered_stop}
                phase_stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
                phase_stream.flush()
                phases_done.append(row)
                print(json.dumps({'event': 'pickup_phase', 'phase': phase['name'], 'status': result.status, 'time_s': env.time_s}), flush=True)
                permitted_contact_stall = (phase['name'] == 'close' and result.status == 'failed'
                    and result.diagnostic is not None and result.diagnostic.code in ('tracking_stalled','convergence_timeout')
                    and observer.pinch_ready())
                if result.status != 'completed' and not (permitted_contact_stall or contact_triggered_stop):
                    raise RuntimeError('Pickup controller phase did not complete: '+phase['name'])
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
    finally:
        if observer is not None:
            try:
                physical = observer.close()
                dump(output/'physical_observer.json', physical)
            except BaseException as exc:
                error = error or {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
        if states:
            np.savez_compressed(output/'states.npz', **{name: np.stack([s[name] for s in states]) for name in states[0]})
        if env is not None:
            dump(output/'metadata_final.json', env.metadata())
            dump(output/'final_observation.json', env.observation())
        if 'verified' in locals() and verified is not None:
            try:
                dump(output/'runtime_imports_final.json', verify_if_snapshot(root,
                    required_modules=('scripts.validation.validate_board_game_pickup', 'bench.control.factory', 'bench.control.bimanual',
                        'bench.simulation.isaacsim_backend.contact_sample'), entrypoint_file=Path(__file__)))
            except BaseException as exc:
                error = error or {'type': type(exc).__name__, 'detail': str(exc), 'traceback': traceback.format_exc()}
    report = {'schema': 'board_game_native_pickup_diagnostic.v1', 'passed': error is None and bool(physical and physical['passed']),
        'error': error, 'physical': physical, 'phases': phases_done, 'model_calls': 0,
        'object_state_writes_after_reset': False, 'joint_targets_only_after_reset': True,
        'game_placement_qualified': False, 'full_inventory_qualified': False,
        'scope': 'Only selected side/token/default release initial scene. Actual grasp and lift require all physical checks, external exit0 and native log review.',
        'artifacts_sha256': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    return seal_report_before_shutdown(output, plain(report), env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','release','output','grasp-plan'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--piece-id', required=True)
    parser.add_argument('--side', choices=('left','right'), required=True)
    parser.add_argument('--seed', type=int, default=0)
    if not run(parser.parse_args())['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
