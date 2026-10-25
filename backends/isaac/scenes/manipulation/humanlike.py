"""Private, fail-closed SHARPA grasp heuristic from measured native evidence.

This is a versioned engineering constraint, not a human-motion-distribution
score. Whole-elastomer rigid contact is a pad proxy; contact point location,
true fingertip skin region, taxels, friction-cone closure and hardware force
calibration are not available. All physical observation fields stay private.
The release freezes this module and the explicit configuration below.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bench.paths import DATA_ROOT

VERSION = 'sharpa_humanlike_grasp_heuristic_v1'
SOURCE = 'PhysX stage-bound rigid contact tensor; normal impulses'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
PADS = {f'right_{finger}_elastomer': finger for finger in FINGERS}
EPS = 1.e-8


def default_humanlike_config(assets_root=f'{DATA_ROOT}/assets', *, object_id='cube', physics_dt_s=.0005):
    """Generate release data once; evaluation never reads live assets/code."""
    path = Path(assets_root) / 'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf'
    root = ET.parse(path).getroot()
    limits = {j.get('name'): [float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))]
              for j in root.findall('joint') if j.get('type') == 'revolute'}
    return {'version': VERSION, 'object': object_id, 'robot_id': 'sharpa_w01_right',
        'hand_urdf': {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
        'joint_limits_rad': limits, 'joint_limit_tolerance_rad': .03,
        'maximum_dip_minus_pip_rad': .60, 'minimum_contact_finger_flexion_sum_rad': .10,
        'physics_dt_s': float(physics_dt_s), 'control_period_s': .02,
        'minimum_normal_impulse_ns': 1.e-8, 'maximum_normal_direction_cosine': -.5,
        'minimum_opposed_substep_fraction': .5, 'hold_s': .12,
        'maximum_robot_penetration_m': .003,
        'wrist_approach_axis_local': [0., 0., 1.], 'wrist_approach_direction_world': [0., 0., -1.],
        'maximum_wrist_approach_angle_rad': math.radians(75.),
        'object_in_wrist_box_min_m': [-.015, -.11, .05],
        'object_in_wrist_box_max_m': [.15, .13, .30],
        'maximum_tip_direction_cosine': .5, 'minimum_tip_span_m': .015,
        'maximum_tip_span_m': .11, 'maximum_tip_object_distance_m': .09,
        'maximum_object_to_tip_segment_distance_m': .045,
        'claim': 'Engineering humanlike heuristic, not a calibrated human-motion distribution or force-closure certificate.',
        'pad_contact_semantics': 'Whole elastomer rigid-body normal contact proxy; actual skin/contact-point region is unobserved.',
        'required': ['native_finite_state', 'all_22_hand_joints_within_frozen_URDF_limits',
            'no_excessive_DIP_over_PIP_flexion', 'thumb_and_index_or_middle_flexed',
            'thumb_elastomer_and_index_or_middle_elastomer_opposing_normal_impulses_in_same_substeps',
            'opposition_substep_fraction_and_continuous_dwell', 'measured_virtual_tip_opposition_geometry',
            'measured_wrist_approach_cone_and_target_in_front_grasp_region',
            'no_robot_contact_penetration_beyond_threshold'],
        'unmeasured': ['contact_point_location', 'true_fingerpad_skin_region', 'taxels',
                       'tangential_friction', 'force_closure', 'human_motion_distribution'],
        'sampling_assumption': 'Every native physics substep is captured by the frozen native step contract; absent pair means zero solved contact.'}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + ' must be finite')
    return float(value)


def _vec(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(label + ' must be a finite vector')
    return np.asarray([_number(x, label) for x in value], dtype=float)


def _rotation(quaternion):
    q = _vec(quaternion, 4, 'wrist quaternion')
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.) > .01:
        raise ValueError('wrist quaternion must be normalized')
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _joint_names():
    names = ['right_thumb_' + x for x in ('CMC_FE', 'CMC_AA', 'MCP_FE', 'MCP_AA', 'IP')]
    names += ['right_pinky_CMC']
    names += [f'right_{f}_{j}' for f in FINGERS[1:] for j in ('MCP_FE', 'MCP_AA', 'PIP', 'DIP')]
    return set(names)


class HumanlikeGraspEvaluator:
    """Evaluate trusted native observations; never mutate simulator or input.

    ``valid`` concerns evidence integrity. ``instantaneous_pass`` is the current
    complete 20 ms window's pose and contact constraint. ``passed`` additionally
    requires continuous qualifying windows for hold_s, and is revoked on loss.
    Task evaluators must enforce it throughout their lifted/transport stage;
    this component does not infer task semantics or count table support.
    """
    def __init__(self, config):
        self.config = deepcopy(config)
        c = self.config
        if c.get('version') != VERSION or c.get('robot_id') != 'sharpa_w01_right':
            raise ValueError('Unsupported humanlike grasp contract')
        if not isinstance(c.get('object'), str) or not c['object'] or '/' in c['object']:
            raise ValueError('humanlike object must be a simple object id')
        limits = c.get('joint_limits_rad')
        if not isinstance(limits, dict) or set(limits) != _joint_names():
            raise ValueError('Freeze all and only 22 SHARPA hand joint limits')
        for name, pair in limits.items():
            low, high = _vec(pair, 2, name)
            if low >= high:
                raise ValueError('Joint limits must be ordered')
        for key in ('joint_limit_tolerance_rad', 'maximum_dip_minus_pip_rad',
                    'minimum_contact_finger_flexion_sum_rad', 'physics_dt_s', 'control_period_s',
                    'minimum_normal_impulse_ns', 'minimum_opposed_substep_fraction', 'hold_s',
                    'maximum_robot_penetration_m', 'maximum_wrist_approach_angle_rad',
                    'minimum_tip_span_m', 'maximum_tip_span_m', 'maximum_tip_object_distance_m',
                    'maximum_object_to_tip_segment_distance_m'):
            if _number(c[key], key) <= 0:
                raise ValueError(key + ' must be positive')
        if not 0 < c['minimum_opposed_substep_fraction'] <= 1:
            raise ValueError('Invalid substep fraction')
        if not -1 <= _number(c['maximum_normal_direction_cosine'], 'normal cosine') < 0:
            raise ValueError('Contact normals must oppose')
        if not -1 <= _number(c['maximum_tip_direction_cosine'], 'tip cosine') < 1:
            raise ValueError('Invalid tip opposition cosine')
        if not c['minimum_tip_span_m'] < c['maximum_tip_span_m'] or c['maximum_wrist_approach_angle_rad'] > math.pi:
            raise ValueError('Invalid pose thresholds')
        self.steps = round(c['control_period_s'] / c['physics_dt_s'])
        if self.steps < 1 or abs(self.steps * c['physics_dt_s'] - c['control_period_s']) > EPS:
            raise ValueError('Control period must contain complete physics substeps')
        for key in ('wrist_approach_axis_local', 'wrist_approach_direction_world'):
            if abs(np.linalg.norm(_vec(c[key], 3, key)) - 1.) > EPS:
                raise ValueError(key + ' must be unit length')
        lo = _vec(c['object_in_wrist_box_min_m'], 3, 'wrist box min')
        hi = _vec(c['object_in_wrist_box_max_m'], 3, 'wrist box max')
        if np.any(lo >= hi):
            raise ValueError('Wrist grasp box must be ordered')
        self.reset()

    def reset(self):
        self.previous_time = None
        self.hold_elapsed_s = 0.
        self.maximum_hold_s = 0.
        self.invalid = None
        self.last_result = None

    def _result(self, *, valid, reason, instantaneous_pass=False, grasp_attempted=False, pose_valid=False, metrics=None):
        self.last_result = {'evaluator': VERSION, 'valid': bool(valid),
            'passed': bool(valid and instantaneous_pass and self.hold_elapsed_s + EPS >= self.config['hold_s']),
            'instantaneous_pass': bool(instantaneous_pass), 'pose_valid': bool(pose_valid),
            'grasp_attempted': bool(grasp_attempted), 'reason': reason,
            'hold_elapsed_s': self.hold_elapsed_s, 'hold_required_s': self.config['hold_s'],
            'maximum_hold_s': self.maximum_hold_s, 'metrics': metrics or {},
            'semantics': 'whole_elastomer_contact_and_measured_pose_heuristic'}
        return deepcopy(self.last_result)

    def _pose(self, obs):
        c = self.config
        if obs['metrics']['state_finite'] is not True or obs['robot']['hand_id'] != c['robot_id']:
            raise ValueError('Native finite SHARPA state is missing')
        joints = obs['joint_positions_rad']
        values = {n: _number(joints[n], n) for n in c['joint_limits_rad']}
        tolerance = c['joint_limit_tolerance_rad']
        joint_ok = all(lo - tolerance <= values[n] <= hi + tolerance
                       for n, (lo, hi) in c['joint_limits_rad'].items())
        coupling_ok = all(values[f'right_{f}_DIP'] - values[f'right_{f}_PIP'] <= c['maximum_dip_minus_pip_rad'] + EPS
                          for f in FINGERS[1:])
        wrist = obs['robot']['right_hand_flange']
        origin = _vec(wrist['position_m'], 3, 'wrist position')
        rotation = _rotation(wrist['quaternion_wxyz'])
        center = _vec(obs['objects'][c['object']]['position_m'], 3, 'object position')
        local = rotation.T @ (center - origin)
        box_ok = bool(np.all(local >= np.asarray(c['object_in_wrist_box_min_m']) - EPS)
                      and np.all(local <= np.asarray(c['object_in_wrist_box_max_m']) + EPS))
        approach_cos = float((rotation @ c['wrist_approach_axis_local']) @ c['wrist_approach_direction_world'])
        approach_ok = approach_cos + EPS >= math.cos(c['maximum_wrist_approach_angle_rad'])
        positions = {f: _vec(obs['robot']['fingertips'][f'right_{f}_fingertip']['position_m'], 3, f + ' tip')
                     for f in ('thumb', 'index', 'middle')}
        thumb_flex = sum(values['right_thumb_' + j] for j in ('CMC_FE', 'MCP_FE', 'IP'))
        eligible, geometry = set(), {}
        for finger in ('index', 'middle'):
            a, b = positions['thumb'] - center, positions[finger] - center
            span = float(np.linalg.norm(b - a))
            da, db = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            cosine = float(a @ b / (da * db)) if min(da, db) > 1.e-12 else None
            alpha = float(np.clip(-a @ (b-a) / span**2, 0., 1.)) if span > 1.e-12 else 0.
            segment_distance = float(np.linalg.norm(a + alpha * (b-a)))
            flex = sum(values[f'right_{finger}_{j}'] for j in ('MCP_FE', 'PIP', 'DIP'))
            good = (cosine is not None and cosine <= c['maximum_tip_direction_cosine'] + EPS
                    and c['minimum_tip_span_m'] - EPS <= span <= c['maximum_tip_span_m'] + EPS
                    and max(da, db) <= c['maximum_tip_object_distance_m'] + EPS
                    and segment_distance <= c['maximum_object_to_tip_segment_distance_m'] + EPS
                    and min(flex, thumb_flex) + EPS >= c['minimum_contact_finger_flexion_sum_rad'])
            if good:
                eligible.add(finger)
            geometry[finger] = {'eligible': bool(good), 'tip_direction_cosine': cosine,
                'tip_span_m': span, 'maximum_tip_object_distance_m': max(da, db),
                'object_to_tip_segment_distance_m': segment_distance, 'finger_flexion_sum_rad': flex}
        return bool(joint_ok and coupling_ok and box_ok and approach_ok and eligible), eligible, {
            'joints_within_limits': joint_ok, 'finger_coupling_valid': coupling_ok,
            'object_in_wrist_grasp_box': box_ok, 'object_position_in_wrist_m': local.tolist(),
            'wrist_approach_cosine': approach_cos, 'wrist_approach_valid': approach_ok,
            'thumb_flexion_sum_rad': thumb_flex, 'finger_geometry': geometry}

    def _contacts(self, obs, time, first, eligible):
        c = self.config
        contacts = obs['contacts']
        if contacts.get('source') != SOURCE or contacts.get('errors') != [] or not isinstance(contacts.get('rigid'), list):
            raise ValueError('Complete native solver contact stream is missing')
        rows = contacts['rigid']
        duration = _number(obs['last_duration_s'], 'native window duration')
        if first and time == 0. and duration == 0. and not rows:
            return False, False, {'simultaneous_opposing_fraction': 0., 'maximum_robot_penetration_m': 0., 'pad_fingers_contacting': []}
        if abs(duration - c['control_period_s']) > EPS:
            raise ValueError('Native contact window is not one complete control period')
        start = time - duration
        by_tick, seen, contacting = {}, set(), set()
        maximum_penetration = 0.
        target = '/World/Objects/' + c['object']
        for row in rows:
            actor0, actor1 = row['actor0'], row['actor1']
            if not isinstance(actor0, str) or not isinstance(actor1, str) or actor0 == actor1:
                raise ValueError('Malformed native contact actors')
            stamp = _number(row['time_s'], 'contact timestamp')
            tick = round((stamp - start) / c['physics_dt_s'])
            if not 1 <= tick <= self.steps or abs(stamp - (start + tick * c['physics_dt_s'])) > EPS:
                raise ValueError('Stale, future or off-grid native contact sample')
            key = (tick, *sorted((actor0, actor1)))
            if key in seen:
                raise ValueError('Duplicated native contact pair at one substep')
            seen.add(key)
            amount = _number(row['impulse_magnitude_sum_ns'], 'contact impulse')
            vector = _vec(row['impulse_vector_ns'], 3, 'contact impulse vector')
            separation = _number(row['min_separation_m'], 'contact separation')
            points = row['point_count']
            nonzero = row['nonzero_impulse_normal_count']
            normal_error = _number(row['normal_length_max_abs_error'], 'normal length error')
            if (amount < 0 or type(points) is not int or points <= 0 or type(nonzero) is not int
                    or not 0 <= nonzero <= points or not 0 <= normal_error <= 32 * 2**-23
                    or row.get('impulse_components') != 'normal_only'
                    or np.linalg.norm(vector) > amount + EPS or (amount > 0 and nonzero == 0)):
                raise ValueError('Malformed native normal-impulse evidence')
            if any(a.startswith('/World/Robot/') for a in (actor0, actor1)):
                maximum_penetration = max(maximum_penetration, -separation)
            other = actor1 if actor0 == target else actor0 if actor1 == target else None
            if not other or not other.startswith('/World/Robot/'):
                continue
            finger = PADS.get(other.rsplit('/', 1)[-1])
            if finger is None or amount <= c['minimum_normal_impulse_ns']:
                continue
            contacting.add(finger)
            # Normalize pair orientation to the same target body. The overall
            # normal sign convention is immaterial to pairwise opposition.
            by_tick.setdefault(tick, {})[finger] = vector if actor0 == target else -vector
        opposed = 0
        for vectors in by_tick.values():
            thumb = vectors.get('thumb')
            if thumb is None or np.linalg.norm(thumb) <= c['minimum_normal_impulse_ns']:
                continue
            for finger in eligible:
                other = vectors.get(finger)
                if other is None or np.linalg.norm(other) <= c['minimum_normal_impulse_ns']:
                    continue
                cosine = float(thumb @ other / (np.linalg.norm(thumb) * np.linalg.norm(other)))
                if cosine <= c['maximum_normal_direction_cosine'] + EPS:
                    opposed += 1
                    break
        fraction = opposed / self.steps
        safe = maximum_penetration <= c['maximum_robot_penetration_m'] + EPS
        attempted = 'thumb' in contacting and bool({'index', 'middle'} & contacting)
        return bool(fraction + EPS >= c['minimum_opposed_substep_fraction'] and safe), attempted, {
            'simultaneous_opposing_fraction': fraction, 'opposed_substeps': opposed,
            'window_physics_substeps': self.steps, 'maximum_robot_penetration_m': maximum_penetration,
            'penetration_within_limit': safe, 'pad_fingers_contacting': sorted(contacting)}

    def update(self, observation):
        if self.invalid:
            return self._result(valid=False, reason='invalid_observation:' + self.invalid)
        try:
            time = _number(observation['time_s'], 'observation time')
            if time < 0 or (self.previous_time is not None and time <= self.previous_time):
                raise ValueError('Observation time must increase strictly from a nonnegative baseline')
            first = self.previous_time is None
            pose_ok, eligible, metrics = self._pose(observation)
            contact_ok, attempted, contact_metrics = self._contacts(observation, time, first, eligible)
            metrics.update(contact_metrics)
            gap = not first and abs(time - self.previous_time - self.config['control_period_s']) > EPS
            self.previous_time = time
            good = bool(pose_ok and contact_ok and not first and not gap)
            self.hold_elapsed_s = self.hold_elapsed_s + self.config['control_period_s'] if good else 0.
            self.maximum_hold_s = max(self.maximum_hold_s, self.hold_elapsed_s)
            reason = ('baseline_recorded' if first else 'observation_gap:dwell_reset' if gap
                      else 'pose_constraint_not_met' if not pose_ok
                      else 'contact_constraint_not_met' if not contact_ok
                      else 'passed' if self.hold_elapsed_s + EPS >= self.config['hold_s'] else 'holding')
            return self._result(valid=True, reason=reason, instantaneous_pass=good,
                                grasp_attempted=attempted, pose_valid=pose_ok, metrics=metrics)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            self.invalid = str(exc)
            self.hold_elapsed_s = 0.
            return self._result(valid=False, reason='invalid_observation:' + self.invalid)
