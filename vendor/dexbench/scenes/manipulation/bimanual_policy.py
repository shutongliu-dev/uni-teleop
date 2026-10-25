"""Privileged development baseline for the bimanual tasks (dual FR3 + SHARPA).

Reads true object poses (privileged, recorded by the harness) and emits only
ordinary bimanual geometry decisions ({'hands': {side: deltas}}); the classical
executor, frozen tasks and evaluator are unchanged. Never certifies success.

* dual_lift: each hand runs the single-arm validated SHARPA top-down pinch on
  its own cube; the left hand uses the exact mirror image (the left SHARPA is
  the right one mirrored across the flange xz-plane, verified to 1.5 um, with
  equal joint values; wrist pose T_l = M T_r, R_l = M R_r M, M = diag(1,-1,1)).
* bimanual_lift: palms face the carton's +/-y faces 5 mm off, fingers pointing
  forward along +x and tilted 35 deg down (thumbs up-forward), mid-finger at
  the carton's centre line 1 cm above its centre, so each FR3 wrist stays well
  above the table (0.1.1 held the tool axis level at the carton centre and
  fr3_left_link7 hit the table); the four long fingers then flex (MCP 0.28 rad,
  pads ~5 mm inside the faces) and stall on contact, so their position drives
  squeeze; both wrists then rise 10 cm. 0.1.3 centres the expected fingertip
  contacts over the carton's centre of mass and 3 cm above it: in 0.1.2 they
  were 47 mm forward and ~22 mm below it, so the grip acted as a hinge and the
  carton pitched 61 deg.
  0.1.5 (flange_axis_centred_v2 only): the carton grip needs each hand's palm on
  a box face, so its orientation cannot be turned by the mount yaw; reaching it
  with plain wrist steps let the IK swing both elbows inward (fr3_left_link4 hit
  fr3_right_link3 natively, mount_v2_baseline_01/box_lift). The hands now reach
  the 'above' pose along FK waypoints of a joint-space interpolation toward the
  IK solution seeded with joint 7 turned back by the mount yaw - the identity-
  mount arm configuration (arm-arm gap 161 mm instead of 1 mm, CPU walk).
  0.1.6 (releases with compliant pads, physics option robot_pad_compliance): the
  fixed 'closed' squeeze drove the pads 4-6 mm into the carton
  (pad_compliance_baseline_01/02; with compliant pads the force grows with that
  overlap). The squeeze now closes the long fingers along the palm normal by
  measured pad depth: each decision computes, from the measured joints and wrist
  and the carton pose, how deep each elastomer pad reaches past the carton face
  and steps toward SQUEEZE_DEPTH_M (steps >= 1.6 mm, because shorter fingertip
  steps complete within the 1.5 mm tolerance without moving). Rigid-pad releases
  keep the 0.1.5 squeeze.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from bench.control.bimanual import SIDES, BimanualGeometry
from bench.geometry import matrix_rotvec, quaternion_matrix, rpy_matrix
from .privileged_policy import (GRASP_HAND_POSTURES_RAD, PRECISE_ORIENTATION_TOLERANCE_RAD, turned_by_mount,
                                PRECISE_POSITION_TOLERANCE_M, _capped, _rotation, _vector)
from bench.paths import load_json

M = np.diag([1., -1., 1.])
SQUEEZE_MCP_RAD = .28
PALM_SURFACE_X_M = .0253      # right_hand_C_MC collision max along flange +x (palm side)
PALM_GAP_M = .005
MID_FINGER_M = .17            # flange to mid-finger along flange +z
FINGER_TILT_RAD = math.radians(35.)
# Mean fingertip contact relative to the mid-finger point, measured natively in
# box_lift_r2 (pads at the squeeze posture): +47 mm forward, -32 mm down.
CONTACT_FROM_MID_M = np.array([.047, 0., -.032])
CONTACT_ABOVE_COM_M = .03
LONG_FINGERS = ('index', 'middle', 'ring', 'pinky')
# 0.1.6 compliant-pad squeeze: target pad overlap past the carton face; a finger whose
# remaining gap is at most SQUEEZE_DONE_GAP_M stops (final overlap within +/- 0.8 mm).
SQUEEZE_DEPTH_M = .0006
SQUEEZE_DONE_GAP_M = .0008
SQUEEZE_MIN_STEP_M = .0016
SQUEEZE_MAX_STEP_M = .004
TIP_NAMES = {'index': 'index', 'middle': 'middle', 'ring': 'ring', 'pinky': 'little'}


def squeeze_step(gap):
    """Fingertip step (m) along the palm normal for a remaining pad-depth gap (m).

    Steps shorter than the 1.5 mm fingertip tolerance complete without moving, so the
    last step is at least SQUEEZE_MIN_STEP_M and lands on the target when it can.
    """
    if gap <= SQUEEZE_DONE_GAP_M:
        return 0.
    if gap <= SQUEEZE_MIN_STEP_M:
        return SQUEEZE_MIN_STEP_M
    if gap <= 2 * SQUEEZE_MIN_STEP_M:
        return gap
    return min(SQUEEZE_MAX_STEP_M, gap - SQUEEZE_MIN_STEP_M)


def _side_names(posture, side):
    return {name.replace('right_', side + '_', 1): value for name, value in posture.items()}


class PrivilegedBimanualPolicy:
    policy_id = 'privileged_bimanual_v1'
    VERSION = '0.1.6'   # 0.1.4: dual-lift pinch yaw; 0.1.5: v2 box-lift arm path; 0.1.6: compliant-pad squeeze by pad depth

    def __init__(self, scene: dict, task: dict):
        self.operation = task['evaluation']['mode']
        if self.operation not in ('bimanual_lift', 'dual_lift'):
            raise ValueError('PrivilegedBimanualPolicy supports the bimanual_lift and dual_lift tasks')
        robot = scene['robot']
        if robot.get('hand_id') != 'dual_fr3_sharpa_w01':
            raise ValueError('The bimanual baseline requires the dual FR3 + SHARPA robot')
        self.geometry = BimanualGeometry(robot['assets_root'], robot['base_position_m'],
                                         assembly_version=robot.get('assembly_version'))
        from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
        self.hand_frame_yaw = get_hand_spec('dual_fr3_sharpa_w01', robot.get('assembly_version'))['hand_frame_yaw_rad']
        self.table_z = float(scene['table']['position_m'][2] + scene['table']['half_extents_m'][2])
        self.bounds = {}
        for obj in scene['objects']:
            manifest = load_json(Path(obj['asset_root']) / 'manifest.json')['assets'][obj['asset_id']]
            lo, hi = np.asarray(manifest['bounds_min_m'], float), np.asarray(manifest['bounds_max_m'], float)
            self.bounds[obj['id']] = ((lo + hi) / 2, (hi - lo) / 2)
        if self.operation == 'dual_lift':
            self.assignment = {side: name for name, side in task['evaluation']['objects'].items()}
        else:
            self.box = task['evaluation']['object']
        self.terminal_hold_s = float(task['evaluation']['hold_s'])
        self.compliant_pads = bool((scene.get('isaacsim') or {}).get('robot_pad_compliance'))
        self._pad_points = None
        self.phases, self.phase_index, self.phase_attempts = [], 0, 0
        self.last_sequence = self.pending_hold = self.finished_reason = self.last_phase = None
        self.decisions, self.consecutive_timeouts = 0, 0
        self.plans, self.trace = [], []

    # --- robot-only priors ---------------------------------------------------------
    def _tips(self, side, posture):
        geometry = self.geometry.sides[side]
        q = {j.name: float(np.clip(posture.get(j.name, 0.), j.lower, j.upper)) for j in geometry.hand.active}
        poses = geometry.hand.forward(q)
        return {finger: geometry._tip(poses, finger) for finger in geometry.fingertip_links}, q

    def _object(self, name, state):
        obj = state['objects'][name]
        position = _vector(obj['position_m'], 3, 'object position')
        rotation = _rotation(obj['quaternion_wxyz'])
        centre_local, half = self.bounds[name]
        centre = position + rotation @ centre_local
        top = float(centre[2] + np.abs(rotation[2]) @ half)
        return centre, rotation, half, top

    def _plan(self, observation, state):
        geo = observation['proprioception']['hand_geometries']
        starts = {side: (np.asarray(geo[side]['wrist']['position_m'], float), quaternion_matrix(geo[side]['wrist']['quaternion_wxyz']))
                  for side in SIDES}
        hands = {side: {} for side in SIDES}
        if self.operation == 'dual_lift':
            for side in SIDES:
                centre, rotation, half, top = self._object(self.assignment[side], state)
                if rotation[2, 2] < math.cos(.35):
                    raise ValueError('Cube is outside the upright planning envelope')
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                from bench.geometry import rotvec_matrix
                R = turned_by_mount(rotvec_matrix([0., 0., yaw]) @ np.diag([1., -1., -1.]), self.hand_frame_yaw)
                pinch = np.array([.07, .05, .15])
                if side == 'left':
                    R, pinch = M @ R @ M, M @ pinch
                flange = np.r_[centre[:2], top - .01] - R @ pinch
                shaped = {shape: self._tips(side, _side_names(GRASP_HAND_POSTURES_RAD[shape], side)) for shape in ('open', 'closed')}
                hands[side] = dict(R=R, flange=flange, above=flange + [0., 0., .11],
                                   tips={k: v[0] for k, v in shaped.items()}, goals={k: v[1] for k, v in shaped.items()},
                                   object_centre=centre.tolist())
            phases = [('raise', {}), ('pre_shape', {'tips': 'open', 'goal': 'open'}),
                      ('orient', {'tips': 'open', 'rotate': True, 'precise': True}),
                      ('above', {'tips': 'open', 'at': 'above', 'precise': True}),
                      ('descend', {'tips': 'open', 'at': 'flange', 'step': .015, 'precise': True}),
                      ('close', {'tips': 'closed', 'goal': 'closed', 'at': 'flange', 'tip_step': .004, 'contact_finish': True}),
                      ('establish_contact_hold', {'hold': .4}),
                      ('lift', {'at': 'above', 'step': .015, 'no_tips': True}),
                      ('lift_hold', {'hold': self.terminal_hold_s + .6})]
        else:
            centre, rotation, half, top = self._object(self.box, state)
            if rotation[2, 2] < math.cos(.2):
                raise ValueError('Carton is outside the level planning envelope')
            half_y = float(np.abs(rotation.T @ [0., 1., 0.]) @ half)
            c, s_ = math.cos(FINGER_TILT_RAD), math.sin(FINGER_TILT_RAD)
            fingers = np.array([c, 0., -s_])
            # Right hand on the -y side: palm normal (flange x) +y, thumb (flange y) up-forward, fingers (flange z) forward-down.
            R_right = np.column_stack([[0., 1., 0.], [s_, 0., c], fingers])
            squeeze = {f'right_{finger}_MCP_FE': SQUEEZE_MCP_RAD for finger in LONG_FINGERS}
            for side in SIDES:
                sign = -1. if side == 'right' else 1.
                R = R_right if side == 'right' else M @ R_right @ M
                contact = np.array([centre[0], centre[1] + sign * (half_y + PALM_GAP_M + PALM_SURFACE_X_M),
                                    centre[2] + CONTACT_ABOVE_COM_M])
                mid = contact - CONTACT_FROM_MID_M
                flange = mid - MID_FINGER_M * fingers
                shaped = {'open': self._tips(side, {}), 'closed': self._tips(side, _side_names(squeeze, side))}
                hands[side] = dict(R=R, flange=flange, above=flange + [0., 0., .15], lifted=flange + [0., 0., .10],
                                   tips={k: v[0] for k, v in shaped.items()}, goals={k: v[1] for k, v in shaped.items()},
                                   object_centre=centre.tolist())
            phases = [('raise', {}), ('pre_shape', {'tips': 'open', 'goal': 'open'}),
                      ('orient', {'tips': 'open', 'rotate': True, 'precise': True}),
                      ('above', {'tips': 'open', 'at': 'above', 'precise': True}),
                      ('descend', {'tips': 'open', 'at': 'flange', 'step': .015, 'precise': True}),
                      ('squeeze', {'tips': 'closed', 'goal': 'closed', 'at': 'flange', 'tip_step': .004, 'contact_finish': True,
                                   'fingers': ('index', 'middle', 'ring', 'little')}),
                      ('establish_contact_hold', {'hold': .5}),
                      ('lift', {'at': 'lifted', 'step': .015, 'no_tips': True}),
                      ('lift_hold', {'hold': self.terminal_hold_s + .6})]
        clearance_z = max(max(s[0][2] for s in starts.values()), max(h['above'][2] for h in hands.values()) + .05)
        for side in SIDES:
            position, R0 = starts[side]
            hands[side]['clearance'] = np.r_[position[:2], clearance_z]
            hands[side]['start_rotation'] = R0
        if self.operation == 'bimanual_lift' and self.hand_frame_yaw:
            measured = observation['proprioception']['joint_positions_rad']
            for side in SIDES:
                hands[side]['arm_target'] = self._arm_target(side, measured, hands[side]['above'], hands[side]['R'])
            at = next(i for i, (name, _) in enumerate(phases) if name == 'orient')
            phases[at:at + 2] = [('reconfigure', {'tips': 'open', 'at': 'above', 'precise': True, 'joint_path': True})]
        self.hands = hands
        self.phases = phases
        self.phase_index = self.phase_attempts = 0
        self.plans.append({'operation': self.operation, 'sequence': observation['sequence'],
                           'hands': {side: {'flange_world_m': h['flange'].tolist(), 'object_centre_m': h['object_centre']}
                                     for side, h in hands.items()}})

    def _posture_waypoint(self, side, observation, shape, tips, tip_step, scale):
        geometry = self.geometry.sides[side]
        measured = observation['proprioception']['joint_positions_rad']
        current = observation['proprioception']['hand_geometries'][side]['fingertip_positions_m']
        reference = {j.name: float(np.clip(measured[j.name], j.lower, j.upper)) for j in geometry.hand.active}
        goal = self.hands[side]['goals'][shape]
        differences = {n: goal[n] - q for n, q in reference.items()}
        fraction = min(1., .12 * scale / max(max(abs(x) for x in differences.values()), 1e-15))
        for _ in range(32):
            posed = geometry.hand.forward({n: q + fraction * differences[n] for n, q in reference.items()})
            deltas = {f: geometry._tip(posed, f) - np.asarray(current[f]) for f in tips}
            if max(np.linalg.norm(x) for x in deltas.values()) <= tip_step + 1e-12:
                return {f: x.tolist() for f, x in deltas.items()}
            fraction *= .5
        return None

    def _arm_target(self, side, measured, position, R):
        """IK of a wrist pose seeded with joint 7 turned back by the mount yaw (mirrored left)."""
        geometry = self.geometry.sides[side]
        seed = {name: float(measured[name]) for name in geometry.names}
        seed[f'fr3_{side}_joint7'] += -self.hand_frame_yaw if side == 'right' else self.hand_frame_yaw
        seed = {name: float(np.clip(value, geometry.by_name[name].lower, geometry.by_name[name].upper)) for name, value in seed.items()}
        from bench.geometry import matrix_quaternion
        solution, _ = geometry.solve(seed, wrist={'frame_id': 'world', 'position_m': np.asarray(position).tolist(),
                                                  'quaternion_wxyz': matrix_quaternion(R).tolist()},
                                     position_tolerance_m=.0005, rotation_tolerance_rad=.005)
        return {name: solution[name] for name in geometry.arm.names}

    def _arm_waypoint(self, side, observation, step, scale):
        """Wrist pose of the largest joint-space step toward the arm target within the step caps."""
        geometry = self.geometry.sides[side]
        measured = observation['proprioception']['joint_positions_rad']
        q = {name: float(np.clip(measured[name], geometry.by_name[name].lower, geometry.by_name[name].upper)) for name in geometry.names}
        start = geometry.public_geometry(q)['wrist']
        target = self.hands[side]['arm_target']

        def displacement(fraction):
            trial = dict(q)
            trial.update({name: q[name] + fraction * (target[name] - q[name]) for name in target})
            wrist = geometry.public_geometry(trial)['wrist']
            dp = np.asarray(wrist['position_m']) - np.asarray(start['position_m'])
            dr = matrix_rotvec(quaternion_matrix(wrist['quaternion_wxyz']) @ quaternion_matrix(start['quaternion_wxyz']).T)
            return dp, dr
        within = lambda f: (lambda dp, dr: np.linalg.norm(dp) <= step * scale + 1e-12 and np.linalg.norm(dr) <= .14 * scale + 1e-12)(*displacement(f))
        low, high = 0., 1.
        if not within(high):
            for _ in range(14):
                middle = (low + high) / 2
                low, high = (middle, high) if within(middle) else (low, middle)
            high = low
        return displacement(high)

    def _pad_clouds(self):
        """Elastomer pad collision vertices of the long fingers, in their link frames."""
        if self._pad_points is None:
            import xml.etree.ElementTree as ET
            import trimesh
            self._pad_points = {}
            for side in SIDES:
                arm_path = Path(self.geometry.sides[side].arm_path)
                links = {link.get('name'): link for link in ET.parse(arm_path).getroot().findall('link')}
                for finger in LONG_FINGERS:
                    clouds = []
                    for collision in links[f'{side}_{finger}_elastomer'].findall('collision'):
                        mesh = collision.find('geometry/mesh')
                        # URDF mesh paths may be relative to the URDF file.
                        points = np.asarray(trimesh.load(arm_path.parent / mesh.get('filename'), force='mesh',
                                                         process=False).vertices, float)
                        if mesh.get('scale'):
                            points = points * np.asarray([float(x) for x in mesh.get('scale').split()])
                        origin = collision.find('origin')
                        xyz = np.asarray([float(x) for x in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()])
                        rpy = [float(x) for x in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
                        clouds.append(points @ rpy_matrix(rpy).T + xyz)
                    self._pad_points[(side, finger)] = np.vstack(clouds)
        return self._pad_points

    def _pad_depths(self, side, observation, state):
        """Measured depth (m) of each long-finger pad past the carton face it squeezes (None: not over it)."""
        centre, rotation, half, _ = self._object(self.box, state)
        geometry = self.geometry.sides[side]
        measured = observation['proprioception']['joint_positions_rad']
        poses = geometry.hand.forward({j.name: float(np.clip(measured[j.name], j.lower, j.upper)) for j in geometry.hand.active})
        wrist = observation['proprioception']['hand_geometries'][side]['wrist']
        position, R = np.asarray(wrist['position_m'], float), quaternion_matrix(wrist['quaternion_wxyz'])
        sign = 1. if side == 'right' else -1.   # the right hand squeezes the carton's -y face
        depths = {}
        for finger in LONG_FINGERS:
            T = poses[f'{side}_{finger}_elastomer']
            world = position + (self._pad_clouds()[(side, finger)] @ T[:3, :3].T + T[:3, 3]) @ R.T
            local = (world - centre) @ rotation
            over = (np.abs(local[:, 0]) <= half[0]) & (np.abs(local[:, 2]) <= half[2])
            depths[finger] = float((half[1] + sign * local[over, 1]).max()) if over.any() else None
        return depths

    def _compliant_squeeze(self, side, observation, state):
        """Fingertip deltas (hand frame) stepping each pad toward SQUEEZE_DEPTH_M along the palm normal."""
        depths = self._pad_depths(side, observation, state)
        deltas = {}
        for finger, depth in depths.items():
            step = 0. if depth is None else squeeze_step(SQUEEZE_DEPTH_M - depth)
            if step:
                deltas[TIP_NAMES[finger]] = [step, 0., 0.]   # +x of the hand frame = palm normal, toward the carton
        return deltas, depths

    def _target(self, side, name, options):
        hand = self.hands[side]
        if name in ('raise', 'pre_shape'):
            return hand['clearance'], hand['start_rotation']
        if name == 'orient':
            return hand['clearance'], hand['R']
        return hand[options['at']], hand['R']

    # --- decisions -----------------------------------------------------------------
    def _stop(self, reason):
        self.finished_reason = reason
        return {'action': 'stop', 'reason': reason}

    def _emit(self, decision, observation):
        self.last_sequence = observation['sequence']
        self.decisions += 1
        return decision

    def next_decision(self, public_observation, privileged_state, previous_feedback=None):
        if self.finished_reason is not None:
            return {'action': 'stop', 'reason': self.finished_reason}
        observation = public_observation
        if self.decisions >= 128:
            return self._stop('privileged_policy_action_budget_exhausted')
        contact_finish = False
        if self.last_sequence is not None:
            if not isinstance(previous_feedback, dict):
                raise ValueError('Every issued decision requires execution feedback')
            status = previous_feedback.get('status')
            code = (previous_feedback.get('diagnostic') or {}).get('code')
            if self.pending_hold is not None:
                if status != 'completed':
                    return self._stop(self.pending_hold + '_hold_not_completed')
                self.trace.append({'phase': self.pending_hold, 'sequence': observation['sequence'], 'sim_time_s': observation['sim_time_s']})
                self.pending_hold = None
                self.phase_index += 1
                self.phase_attempts = 0
            elif status == 'failed' and code in ('tracking_stalled', 'convergence_timeout'):
                phase = self.phases[self.phase_index][1] if self.phase_index < len(self.phases) else {}
                if phase.get('contact_finish'):
                    # Fingers blocked by the object: the executed drive keeps pressing.
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
        geo = observation['proprioception']['hand_geometries']
        while self.phase_index < len(self.phases):
            name, options = self.phases[self.phase_index]
            self.last_phase = name
            if self.phase_attempts >= 45:
                return self._stop('phase_did_not_converge_' + name)
            if 'hold' in options:
                self.pending_hold = name
                return self._emit({'action': 'hold', 'duration_s': options['hold'], 'reason': f'privileged_{self.operation}_{name}'},
                                  observation)
            hands = {}
            for side in SIDES:
                position, R = self._target(side, name, options)
                wrist = geo[side]['wrist']
                dp = position - np.asarray(wrist['position_m'])
                dr = matrix_rotvec(R @ quaternion_matrix(wrist['quaternion_wxyz']).T)
                shape = options.get('tips')
                fingers = options.get('fingers', ('thumb', 'index', 'middle', 'ring', 'little'))
                tips = {} if options.get('no_tips') or shape is None else {
                    f: self.hands[side]['tips'][shape][f] - np.asarray(geo[side]['fingertip_positions_m'][f]) for f in fingers}
                squeeze = None
                if name == 'squeeze' and self.compliant_pads:
                    squeeze, depths = self._compliant_squeeze(side, observation, privileged_state)
                    self.trace.append({'phase': 'squeeze_pad_depths', 'side': side, 'sequence': observation['sequence'],
                                       'depths_m': depths, 'steps_m': {f: d[0] for f, d in squeeze.items()}})
                    tips = {}
                reached = (np.linalg.norm(dp) <= .0025 and np.linalg.norm(dr) <= .035
                           and all(np.linalg.norm(x) <= .0015 for x in tips.values()) and not squeeze)
                if reached:
                    continue
                scale = .5 if self.consecutive_timeouts else 1.
                target = {}
                if options.get('joint_path'):
                    dp, dr = self._arm_waypoint(side, observation, options.get('step', .045), scale)
                if np.linalg.norm(dp) > 1e-6:
                    target['wrist_delta_m'] = _capped(dp, options.get('step', .045) * scale).tolist()
                if np.linalg.norm(dr) > 1e-6:
                    target['wrist_rotation_delta_rad'] = _capped(dr, .14 * scale).tolist()
                if squeeze:
                    target['fingertip_delta_m'] = squeeze
                if tips:
                    tip_step = min(.009, options.get('tip_step', .009)) * scale
                    if 'goal' in options:
                        # Waypoints are FK of an interpolated joint posture, so every
                        # intermediate fingertip target is reachable (straight-line
                        # tip interpolation between reachable tips need not be).
                        deltas = self._posture_waypoint(side, observation, options['goal'], tips, tip_step, scale)
                        if deltas is None:
                            return self._stop('no_bounded_fingertip_waypoint')
                        target['fingertip_delta_m'] = deltas
                    else:
                        target['fingertip_delta_m'] = {f: _capped(x, tip_step).tolist() for f, x in tips.items()}
                if target:
                    hands[side] = target
            if not hands or contact_finish:
                self.trace.append({'phase': name, 'sequence': observation['sequence'], 'sim_time_s': observation['sim_time_s'],
                                   'public_geometry_reached': not hands, 'contact_stall_phase_end': contact_finish})
                self.phase_index += 1
                self.phase_attempts = 0
                contact_finish = False
                continue
            decision = {'action': 'move_geometry', 'duration_s': 5., 'reason': f'privileged_{self.operation}_{name}', 'hands': hands}
            if options.get('precise'):
                decision.update(position_tolerance_m=PRECISE_POSITION_TOLERANCE_M,
                                orientation_tolerance_rad=PRECISE_ORIENTATION_TOLERANCE_RAD)
            self.phase_attempts += 1
            return self._emit(decision, observation)
        return self._stop(self.operation + '_trajectory_complete_independent_score_required')

    def diagnostic(self):
        return {'policy_id': self.policy_id, 'version': self.VERSION, 'ranked_agent': False, 'model_calls': 0,
                'privileged_inputs': ['object_position_m', 'object_quaternion_wxyz', 'asset_bounds', 'table_height'],
                'score_inputs': False, 'object_state_writeback': False, 'operation': self.operation,
                'phase': self.last_phase, 'decisions': self.decisions, 'finished_reason': self.finished_reason,
                'plans': deepcopy(self.plans), 'trace': deepcopy(self.trace),
                'robot_prior': 'single-arm SHARPA grasp prior (mirrored for the left hand) / palm-side finger squeeze',
                'scope': 'Engineering baseline; geometry planning is not collision or success certification'}
