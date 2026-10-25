"""Privileged development baselines for the soft-body tasks (FR3 + SHARPA).

They read the simulated sponge's centroid and bounds (privileged, recorded by
the harness) and emit only ordinary geometry decisions; the executor, tasks and
evaluator are unchanged. Never certifies success.

* soft_press: a "knuckle press". The tool axis points down, the four long
  fingers bend 1.5 rad at MCP (PIP/DIP straight) so the backs of their proximal
  phalanges form a flat face 141.6 mm below the flange, centred over the
  sponge; the thumb is folded away (84 mm reach, 58 mm self-clearance). The
  face is lowered to 50 % of the nominal height, held, then raised. (0.1.0
  turned the palm face-down with a level tool axis: fr3_link7 hit the table
  at wrist z 0.83 m before any compression.) 0.2.2 turns the hand 90 deg about
  the tool axis so the face (40.5 mm along the bent fingers x 36 mm across the
  index/middle backs, measured natively at the reached posture) covers the
  32 x 40 mm top fully; 0.2.1's face missed 7 mm of one side and squeezed the
  sponge sideways (22 mm).
  0.3.0 re-centres the face on the sponge's current centroid when the 'above'
  and 'approach' phases start (0.2.2 planned on the t=0 centroid; the sponge
  had crept 2.8 mm by contact and the off-centre press squeezed it out to
  20.7 mm), and ends the press once the measured height ratio is <= 0.72
  instead of driving on to the 50 % face target (in 0.2.2 the slender column
  buckled sideways at ratio ~0.72 while the face kept descending).
  0.4.0: in v11 r2 (soft_baseline_02) the 6 mm press step crossed 0.75 -> 0.70
  inside one action, so the 0.72 stop (checked at decision boundaries) never
  fired, and the column squeezed out sideways below ~0.70 (12.5 mm in 60 ms).
  Below a measured ratio of 0.80 the press now moves 1 mm per decision (<= 0.017
  of the ratio) and stops at <= 0.735: inside the 0.70-0.75 window. The held
  compression is stable: in v11 r1 (sponge_press_r2) the knuckle joints did
  not deflect and the ratio stayed within 0.005 over the 1.4 s hold.
  0.4.1: 0.4.0 sent the 1 mm steps with the default 3 mm wrist tolerance, so each
  completed at once without moving (soft_press_04: stuck at ratio 0.78 for 45
  decisions). Fine steps now carry a 0.4 mm position tolerance.
* soft_push 0.1.0: the validated single-arm SHARPA cube push prior (fingertip
  at 40 % of the height) - tipped the sponge over (it tips when the push acts
  above ~b/mu = 20 mm / 0.9). 0.2.0: a low "scoop" - fingers slightly curled
  and tilted 40 deg forward from vertical, so only the lowest 17 mm of the hand
  can touch (its lowest point 4 mm above the table) and the thumb stays 22 mm
  behind the front and above the sponge.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from bench.geometry import matrix_rotvec, quaternion_matrix
from .privileged_policy import (PRECISE_ORIENTATION_TOLERANCE_RAD, PRECISE_POSITION_TOLERANCE_M, turned_by_mount,
                                PrivilegedManipulationPolicy, _capped, _vector)

# Knuckle-press hand prior (robot-only, from the pinned SHARPA URDF collision
# meshes): pressing face = backs of the bent proximal phalanges, centre and
# depth in the right_hand_flange frame.
KNUCKLE_POSTURE_RAD = {
    **{f'right_{finger}_{joint}': value for finger in ('index', 'middle', 'ring', 'pinky')
       for joint, value in (('MCP_FE', 1.5), ('MCP_AA', 0.), ('PIP', 0.), ('DIP', 0.))},
    'right_pinky_CMC': 0.,
    'right_thumb_CMC_FE': .663, 'right_thumb_CMC_AA': -.349, 'right_thumb_MCP_FE': -.524,
    'right_thumb_MCP_AA': -.349, 'right_thumb_IP': 0.,
}
# Pressing face measured natively (sponge_press_r3) at the posture the executor
# actually reached (MCP ~1.43 rad, PIP ~0.09 rad): the deepest 4 mm of the index
# and middle proximal-phalanx backs span flange x [10.7, 51.2] mm and y [1.3,
# 37.6] mm at depth 144.6 mm. Centre = bounding-box centre (coverage), not the
# vertex mean (0.2.1 used the mean; the face then missed 7 mm of the top).
PRESS_FACE_LOCAL_M = np.array([.03095, .01945, .1446])
# 0.2.0 aimed at 62 % (stalled at 77 %, relaxed to ~85 %); 0.2.1 at 40 % (68 %,
# but off-centre); 0.2.2 aims at 50 % with the face covering the whole top.
PRESS_HEIGHT_FRACTION = .50
# 0.3.0: the press phase also ends once the measured height ratio reaches this
# (0.72 in 0.3.x; 0.735 from 0.4.0, reached with fine steps below 0.80).
PRESS_STOP_RATIO = .735
PRESS_FINE_FROM_RATIO = .80
PRESS_FINE_STEP_M = .001
PRESS_FINE_TOLERANCE_M = .0004   # < the step, so the arm really moves (0.4.1)
RECENTRE_PHASES = ('above', 'approach')


def _soft_body(state, name, public):
    if (type(state.get('sequence')) is not int or state['sequence'] != public['sequence']
            or abs(state['sim_time_s'] - public['sim_time_s']) > 1e-8):
        raise ValueError('Privileged state must match the public observation revision')
    body = state['soft_bodies'][name]
    low = _vector(body['bounds_min_m'], 3, 'soft body bounds')
    high = _vector(body['bounds_max_m'], 3, 'soft body bounds')
    return _vector(body['centroid_m'], 3, 'soft body centroid'), low, high


class PrivilegedSoftPressPolicy:
    policy_id = 'privileged_soft_press_v1'
    VERSION = '0.4.1'   # 0.3.1: mount yaw; 0.4.0: fine press steps, stop at 0.735; 0.4.1: fine-step tolerance

    def __init__(self, scene, task):
        config = task['evaluation']
        if config['mode'] != 'soft_press' or scene['robot'].get('hand_id', 'sharpa_w01_right') != 'sharpa_w01_right':
            raise ValueError('The soft press baseline needs the soft_press task and the SHARPA hand')
        self.body = config['soft_body']
        from bench.control.fingertip import Fr3SharpaGeometry
        from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
        version = scene['robot'].get('assembly_version')
        self.geometry = Fr3SharpaGeometry(scene['robot']['assets_root'], scene['robot']['base_position_m'],
                                          assembly_version=version)
        self.ROTATION = turned_by_mount(self.ROTATION, get_hand_spec('sharpa_w01_right', version)['hand_frame_yaw_rad'])
        self.nominal_height = float(config['nominal_height_m'])
        self.table_z = float(config['table_top_z_m'])
        self.phases, self.phase_index, self.phase_attempts = [], 0, 0
        self.last_sequence = self.pending_hold = self.finished_reason = self.last_phase = None
        self.decisions, self.consecutive_timeouts = 0, 0
        self.plans, self.trace = [], []

    def _stop(self, reason):
        self.finished_reason = reason
        return {'action': 'stop', 'reason': reason}

    # Tool axis down (flange z = world -z); bent fingers (flange x) along world +y
    # so the 40.5 mm face side spans the sponge's 40 mm y side.
    ROTATION = np.column_stack([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])

    def _targets(self, centre, top):
        """Wrist positions that put the knuckle face over ``centre`` (xy) at each height."""
        offset = self.ROTATION @ PRESS_FACE_LOCAL_M
        face = lambda z: np.array([centre[0], centre[1], z]) - offset
        return {'above': face(top + .10), 'approach': face(top + .005),
                'press': face(self.table_z + PRESS_HEIGHT_FRACTION * self.nominal_height), 'release': face(top + .06)}

    def _plan(self, observation, state):
        centre, low, high = _soft_body(state, self.body, observation)
        R = self.ROTATION
        targets = self._targets(centre, high[2])
        wrist = observation['proprioception']['geometry']['wrist']
        start = np.asarray(wrist['position_m'], float)
        clearance = np.r_[start[:2], max(start[2], targets['above'][2] + .05)]
        measured = observation['proprioception']['joint_positions_rad']
        self.hand_goal = {name: value for name, value in KNUCKLE_POSTURE_RAD.items() if name in measured}
        self.phases = [
            ('raise', {'position': clearance, 'rotation': quaternion_matrix(wrist['quaternion_wxyz'])}),
            ('shape', {'position': clearance, 'rotation': quaternion_matrix(wrist['quaternion_wxyz']), 'posture': True}),
            ('orient', {'position': clearance, 'rotation': R}),
            ('above', {'position': targets['above'], 'rotation': R, 'precise': True}),
            ('approach', {'position': targets['approach'], 'rotation': R, 'step': .015, 'precise': True}),
            # Default (not precise) wrist tolerance: under the foam reaction the press
            # should complete normally so the following hold keeps its PI reference.
            ('press', {'position': targets['press'], 'rotation': R, 'step': .006, 'contact_finish': True, 'ratio_stop': True,
                       'fine_step': PRESS_FINE_STEP_M, 'fine_from_ratio': PRESS_FINE_FROM_RATIO}),
            ('press_hold', {'hold': 1.6}),
            ('release', {'position': targets['release'], 'rotation': R, 'step': .01}),
            ('settle', {'hold': 1.}),
        ]
        self.plans.append({'sequence': observation['sequence'], 'sponge_centroid_m': centre.tolist(),
                           'sponge_top_z_m': float(high[2]), 'pressed_palm_z_m': float(targets['press'][2] + (R @ PRESS_FACE_LOCAL_M)[2]),
                           'press_wrist_m': targets['press'].tolist()})

    def _recentre(self, observation, state, name):
        """Move the remaining face targets over the sponge's current centroid (creep)."""
        centre, _, high = _soft_body(state, self.body, observation)
        targets = self._targets(centre, high[2])
        for phase_name, phase in self.phases[self.phase_index:]:
            if phase_name in targets:
                phase['position'] = targets[phase_name]
        self.plans.append({'sequence': observation['sequence'], 'recentred_at_phase': name,
                           'sponge_centroid_m': centre.tolist(), 'sponge_top_z_m': float(high[2]),
                           'press_wrist_m': targets['press'].tolist()})

    def _height_ratio(self, observation, state):
        _, low, high = _soft_body(state, self.body, observation)
        return float((high[2] - low[2]) / self.nominal_height)

    def _posture_waypoint(self, observation, tip_step=.009):
        """Fingertip deltas toward FK of an interpolated hand posture (each tip <= tip_step)."""
        hand = self.geometry.hand
        measured = observation['proprioception']['joint_positions_rad']
        current = observation['proprioception']['geometry']['fingertip_positions_m']
        reference = {j.name: float(np.clip(measured[j.name], j.lower, j.upper)) for j in hand.active}
        goal = {n: float(np.clip(self.hand_goal.get(n, q), hand.active[hand.names.index(n)].lower,
                                 hand.active[hand.names.index(n)].upper)) for n, q in reference.items()}
        differences = {n: goal[n] - q for n, q in reference.items()}
        largest = max(abs(x) for x in differences.values())
        fraction = 1. if largest < 1e-9 else min(1., .12 / largest)
        for _ in range(32):
            poses = hand.forward({n: q + fraction * differences[n] for n, q in reference.items()})
            tips = {f: self.geometry._tip(poses, f) for f in self.geometry.fingertip_links}
            deltas = {f: tips[f] - np.asarray(current[f]) for f in tips}
            if max(np.linalg.norm(x) for x in deltas.values()) <= tip_step + 1e-12:
                return deltas
            fraction *= .5
        return None

    def next_decision(self, public_observation, privileged_state, previous_feedback=None):
        if self.finished_reason is not None:
            return {'action': 'stop', 'reason': self.finished_reason}
        observation = public_observation
        if self.decisions >= 128:
            return self._stop('privileged_policy_action_budget_exhausted')
        contact_finish = False
        if self.last_sequence is not None:
            status = previous_feedback.get('status')
            code = (previous_feedback.get('diagnostic') or {}).get('code')
            if self.pending_hold is not None:
                if status != 'completed':
                    return self._stop(self.pending_hold + '_hold_not_completed')
                self.trace.append({'phase': self.pending_hold, 'sequence': observation['sequence']})
                self.pending_hold = None
                self.phase_index += 1
            elif status == 'failed' and code in ('tracking_stalled', 'convergence_timeout'):
                if self.phases[self.phase_index][1].get('contact_finish'):
                    contact_finish = True
                else:
                    self.consecutive_timeouts += 1
                    if self.consecutive_timeouts >= 3:
                        return self._stop('repeated_controller_convergence_timeout')
            elif status != 'completed':
                return self._stop(f'controller_feedback_{status}_{code}')
            else:
                self.consecutive_timeouts = 0
        if not self.phases:
            self._plan(observation, privileged_state)
        wrist = observation['proprioception']['geometry']['wrist']
        while self.phase_index < len(self.phases):
            name, phase = self.phases[self.phase_index]
            self.last_phase = name
            if self.phase_attempts >= 45:
                return self._stop('phase_did_not_converge_' + name)
            if name in RECENTRE_PHASES and not phase.get('recentred'):
                self._recentre(observation, privileged_state, name)
                phase['recentred'] = True
            if 'hold' in phase:
                self.pending_hold = name
                self.last_sequence = observation['sequence']
                self.decisions += 1
                return {'action': 'hold', 'duration_s': phase['hold'], 'reason': 'privileged_soft_press_' + name}
            dp = phase['position'] - np.asarray(wrist['position_m'])
            dr = matrix_rotvec(phase['rotation'] @ quaternion_matrix(wrist['quaternion_wxyz']).T)
            if phase.get('posture'):
                deltas = self._posture_waypoint(observation)
                if deltas is None:
                    return self._stop('no_bounded_fingertip_waypoint')
                if max(np.linalg.norm(x) for x in deltas.values()) > .0015 and not contact_finish:
                    self.phase_attempts += 1
                    self.last_sequence = observation['sequence']
                    self.decisions += 1
                    return {'action': 'move_geometry', 'duration_s': 5., 'reason': 'privileged_soft_press_shape',
                            'fingertip_delta_m': {f: x.tolist() for f, x in deltas.items()}}
                self.trace.append({'phase': name, 'sequence': observation['sequence']})
                self.phase_index += 1
                self.phase_attempts = 0
                continue
            ratio = self._height_ratio(observation, privileged_state) if phase.get('ratio_stop') else None
            ratio_stop = ratio is not None and ratio <= PRESS_STOP_RATIO
            if contact_finish or ratio_stop or (np.linalg.norm(dp) <= .0025 and np.linalg.norm(dr) <= .035):
                self.trace.append({'phase': name, 'sequence': observation['sequence'], 'contact_stall_phase_end': contact_finish,
                                   'height_ratio_phase_end': ratio_stop})
                self.phase_index += 1
                self.phase_attempts = 0
                contact_finish = False
                continue
            scale = .5 if self.consecutive_timeouts else 1.
            step = phase.get('step', .045)
            fine = ratio is not None and 'fine_step' in phase and ratio <= phase['fine_from_ratio']
            if fine:
                step = phase['fine_step']
            decision = {'action': 'move_geometry', 'duration_s': 5., 'reason': 'privileged_soft_press_' + name,
                        'wrist_delta_m': _capped(dp, step * scale).tolist(),
                        'wrist_rotation_delta_rad': _capped(dr, .14 * scale).tolist()}
            if phase.get('precise'):
                decision.update(position_tolerance_m=PRECISE_POSITION_TOLERANCE_M,
                                orientation_tolerance_rad=PRECISE_ORIENTATION_TOLERANCE_RAD)
            if fine:
                decision.update(position_tolerance_m=PRESS_FINE_TOLERANCE_M,
                                orientation_tolerance_rad=PRECISE_ORIENTATION_TOLERANCE_RAD)
            self.phase_attempts += 1
            self.last_sequence = observation['sequence']
            self.decisions += 1
            return decision
        return self._stop('soft_press_trajectory_complete_independent_score_required')

    def diagnostic(self):
        return {'policy_id': self.policy_id, 'version': self.VERSION, 'ranked_agent': False, 'model_calls': 0,
                'privileged_inputs': ['soft_body_centroid_m', 'soft_body_bounds_m', 'table_height'],
                'score_inputs': False, 'object_state_writeback': False, 'phase': self.last_phase,
                'decisions': self.decisions, 'finished_reason': self.finished_reason,
                'plans': deepcopy(self.plans), 'trace': deepcopy(self.trace),
                'scope': 'Engineering baseline; not a success certificate'}


PADDLE_POSTURE_RAD = {
    **{f'right_{finger}_{joint}': value for finger in ('index', 'middle', 'ring', 'pinky')
       for joint, value in (('MCP_FE', .2), ('MCP_AA', 0.), ('PIP', .2), ('DIP', .1))},
    'right_pinky_CMC': 0.,
}
PADDLE_TILT_RAD = math.radians(40.)
# From the pinned SHARPA URDF collision meshes at PADDLE_POSTURE_RAD (thumb as in
# PUSH_HAND_POSTURE_RAD) and this tilt: flange height above the hand's lowest
# point, distance of the hand's front plane ahead of the flange, and the lateral
# offset (flange y) of the front contact zone.
PADDLE_FLANGE_ABOVE_LOWEST_M = .1583
PADDLE_FRONT_AHEAD_M = .1724
PADDLE_ZONE_LATERAL_M = .0138
PADDLE_LOWEST_HEIGHT_M = .004


class PrivilegedSoftPushPolicy(PrivilegedManipulationPolicy):
    """Push the sponge with a flat-hand paddle; the sponge's measured state is the box."""
    policy_id = 'privileged_soft_push_v1'
    VERSION = '0.2.0'

    def __init__(self, scene, task):
        config = task['evaluation']
        if config['mode'] != 'soft_push' or scene['robot'].get('hand_id', 'sharpa_w01_right') != 'sharpa_w01_right':
            raise ValueError('The soft push baseline needs the soft_push task and the SHARPA hand')
        self.operation = 'push'
        self.object_id = config['soft_body']
        self.target = _vector(config['goal_position_m'], 3, 'target')
        self.release_radius = float(config['goal_xy_radius_m']) - .006
        self.terminal_hold_s = float(config['hold_s'])
        spec = next(s for s in scene['soft_bodies'] if s['id'] == self.object_id)
        self.box_center = np.zeros(3)
        self.half_extents = np.asarray(spec['size_m'], float) / 2
        self.table_z = float(config['table_top_z_m'])
        self._init_hand(scene, task['public_goal'])
        from .privileged_policy import PUSH_HAND_POSTURE_RAD
        thumb = {n: v for n, v in PUSH_HAND_POSTURE_RAD.items() if n.startswith('right_thumb')}
        self.hand_goal = {**PUSH_HAND_POSTURE_RAD, **thumb, **PADDLE_POSTURE_RAD}
        self.tips = self._shape_tips(self.hand_goal)
        self.asset_manifest_sha256 = None
        self.phases, self.phase_index, self.phase_attempts = [], 0, 0
        self.last_phase = self.last_sequence = self.pending_hold = self.pending_recovery = None
        self.recovery_hold = False
        self.consecutive_timeouts = self.decisions = self.round_index = 0
        self.finished_reason = None
        self.plans, self.trace = [], []

    def _object(self, public, state):
        centre, low, high = _soft_body(state, self.object_id, public)
        # Measured footprint of the deformed body; axis-aligned planning box.
        self.half_extents = np.r_[(high[:2] - low[:2]) / 2, (high[2] - low[2]) / 2]
        return np.r_[centre[:2], (low[2] + high[2]) / 2], np.eye(3), float(high[2])

    def _plan_push(self, observation, center, object_rotation, top):
        difference = self.target[:2] - center[:2]
        distance = float(np.linalg.norm(difference))
        if distance < .002:
            self.finished_reason = 'object_already_at_goal_no_interaction_baseline'
            return
        direction = difference / distance
        d3 = np.r_[direction, 0.]
        half_width = float(np.abs(object_rotation.T @ d3) @ self.half_extents)
        # Fingers (flange z) down and tilted forward; palm normal (flange x) forward-up.
        up = np.array([0., 0., 1.])
        z_axis = -up * math.cos(PADDLE_TILT_RAD) + d3 * math.sin(PADDLE_TILT_RAD)
        x_axis = d3 * math.cos(PADDLE_TILT_RAD) + up * math.sin(PADDLE_TILT_RAD)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        gap = .008
        along = float(center[:2] @ direction) - half_width - gap - PADDLE_FRONT_AHEAD_M
        flange = along * d3 + (float(center[:2] @ y_axis[:2]) - PADDLE_ZONE_LATERAL_M) * y_axis
        flange[2] = self.table_z + PADDLE_LOWEST_HEIGHT_M + PADDLE_FLANGE_ABOVE_LOWEST_M
        above = flange + [0., 0., .12]
        front, lowest = PADDLE_FRONT_AHEAD_M, -PADDLE_FLANGE_ABOVE_LOWEST_M
        wrist = observation['proprioception']['geometry']['wrist']
        start = np.asarray(wrist['position_m'], float)
        orientation = quaternion_matrix(wrist['quaternion_wxyz'])
        clearance = start.copy()
        clearance[2] = max(start[2], above[2] + .05)
        destination = flange + (distance + gap) * d3
        unloaded = destination - .015 * d3
        self.phases = [
            {'name': 'raise', 'position': clearance, 'rotation': orientation},
            {'name': 'pre_shape', 'position': clearance, 'rotation': orientation, 'tips': self.tips, 'hand_goal': self.hand_goal},
            {'name': 'orient', 'position': clearance, 'rotation': rotation, 'tips': self.tips},
            {'name': 'above', 'position': above, 'rotation': rotation, 'tips': self.tips},
            {'name': 'descend', 'position': flange, 'rotation': rotation, 'tips': self.tips, 'translation_step': .01},
            {'name': 'stroke', 'position': destination, 'rotation': rotation, 'tips': self.tips, 'translation_step': .006},
            {'name': 'unload', 'position': unloaded, 'rotation': rotation, 'tips': self.tips, 'translation_step': .006},
            {'name': 'withdraw', 'position': unloaded + [0., 0., .12], 'rotation': rotation, 'tips': self.tips, 'translation_step': .015},
            {'name': 'settle', 'hold': 1.2},
        ]
        self.phase_index = 0
        self.phase_attempts = 0
        self.plans.append({'round': self.round_index, 'sequence': observation['sequence'], 'strategy': 'flat_hand_paddle',
                          'initial_object_position_m': center.tolist(), 'direction_xy': direction.tolist(),
                          'flange_world_m': flange.tolist(), 'destination_world_m': destination.tolist(),
                          'paddle_front_m': front, 'paddle_lowest_tip_m': lowest})

    def diagnostic(self):
        value = super().diagnostic()
        value.update(policy_id=self.policy_id, version=self.VERSION,
                     privileged_inputs=['soft_body_centroid_m', 'soft_body_bounds_m', 'table_height'],
                     robot_prior_source='flat-hand paddle (soft_policy.PADDLE_POSTURE_RAD; thumb from PUSH_HAND_POSTURE_RAD)')
        return value
