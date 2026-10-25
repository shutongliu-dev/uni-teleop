"""Privileged development policy emitting ordinary fingertip model decisions.

Object poses and the frozen asset bounds are read-only planning inputs. The
policy never receives a simulator, service, scorer, or object-state setter.
Its output must pass the usual model parser and classical controller.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from bench.geometry import matrix_rotvec, quaternion_matrix, rotvec_matrix
from bench.control.kinematics import URDFKinematics


# Robot-only posture/contact geometry from the independently checked 64-action
# push_alignment_r3_controller14_01 baseline. These are planning priors, not
# collision certificates. The original controller and scorer are not imported.
PUSH_HAND_POSTURE_RAD = {
    'right_thumb_CMC_FE': .4, 'right_thumb_CMC_AA': .1,
    'right_thumb_MCP_FE': .7, 'right_thumb_MCP_AA': -.2, 'right_thumb_IP': .8,
    'right_index_MCP_FE': .15, 'right_index_MCP_AA': -.1,
    'right_index_PIP': .35, 'right_index_DIP': .25,
    **{f'right_{finger}_{joint}': value
       for finger in ('middle', 'ring', 'pinky')
       for joint, value in (('MCP_FE', .6), ('MCP_AA', .3491), ('PIP', 1.5), ('DIP', 1.2))},
    'right_pinky_CMC': 0.,
}
HAND_URDF_SHA256 = 'ff9ffc2af5069c09378d4ffc7870d53c507a50941849be79b7dfe7c5f9dcb6cc'
ROBOT_PRIOR_SHA256 = '230e4c4043bcf49fa0fd89135e38d564455888667d31cc9a6a14f9b0e11d2ede'
INDEX_FORWARD_SUPPORT_M = .001650642417810059
NONINDEX_RECESS_M = .043852968138901216
FINGERS = ('thumb', 'index', 'middle', 'ring', 'little')
TASK_OPERATIONS = {'01_cube_push': 'push', '02_cube_lift_hold': 'lift_hold',
                   '03_cube_pick_place': 'pick_place', '04_cube_grasp_hold': 'grasp_hold'}
# The two endpoint postures below are robot-only FK priors recorded in the
# successful grasp/lift/place v4_controller14 engineering trajectories. Their
# provenance is retained as data; archived executors are never imported.
GRASP_PRIOR_SUMMARY_SHA256 = 'd989657cd093da050a51b2c252b9b28cf0595eec7a739ecf1a84c0f9cdc55456'
GRASP_PRIOR_SOURCE = ('development/bench_capability_repair_20261012T054520Z/'
                      'native/grasp_v4_controller14_01/summary.json')
GRASP_HAND_POSTURES_RAD = {
    'open': dict(PUSH_HAND_POSTURE_RAD, right_thumb_CMC_FE=.4685350207873381,
        right_thumb_CMC_AA=.05492726365273376, right_thumb_MCP_FE=.27201938620618454,
        right_thumb_MCP_AA=-.19032918312335542, right_thumb_IP=.2602125965682699,
        right_index_MCP_FE=.5178952782032245, right_index_MCP_AA=.21792102714412587,
        right_index_PIP=.865789003358936, right_index_DIP=1.1220944065416567),
    'closed': dict(PUSH_HAND_POSTURE_RAD, right_thumb_CMC_FE=.42185103682126357,
        right_thumb_CMC_AA=.05489909167401758, right_thumb_MCP_FE=.4525274017118162,
        right_thumb_MCP_AA=-.23880449665410342, right_thumb_IP=.4085457784267067,
        right_index_MCP_FE=.5133815595527851, right_index_MCP_AA=-.012511166019883982,
        right_index_PIP=.8967780963630108, right_index_DIP=1.1252416194243304),
}


# Public opt-in wrist completion tolerances for the free-space grasp approach.
PRECISE_POSITION_TOLERANCE_M = .001
PRECISE_ORIENTATION_TOLERANCE_RAD = .01


def _vector(value, count, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != count:
        raise ValueError(f'{label} must have {count} finite coordinates')
    if any(isinstance(x, (bool, np.bool_)) for x in value):
        raise ValueError(f'{label} cannot contain booleans')
    result = np.asarray(value, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f'{label} must have {count} finite coordinates')
    return result


def _rotation(value):
    q = _vector(value, 4, 'quaternion')
    if abs(np.linalg.norm(q) - 1.) > 1e-5:
        raise ValueError('quaternion must have unit length')
    return quaternion_matrix(q)


def _capped(vector, maximum):
    norm = float(np.linalg.norm(vector))
    return vector * min(1., maximum / max(norm, 1e-15))


def _thumb_index_contact(observation):
    """Current identity-free pad contact proxy, never a grasp certificate."""
    tactile = observation.get('tactile') or {}
    time = tactile.get('sim_time_s')
    if (tactile.get('status') != 'available' or isinstance(time, bool)
            or not isinstance(time, (int, float)) or not math.isfinite(time)
            or abs(time - observation['sim_time_s']) > 1e-8):
        return False
    fingers = tactile.get('fingers') or {}
    return all(fingers.get(f, {}).get('status') == 'valid'
               and fingers[f].get('rigid_contact_any') is True for f in ('thumb', 'index'))


def turned_by_mount(rotation, hand_frame_yaw):
    """Turn a hand-frame target about its own z axis by the mount yaw.

    The hand's pose relative to the object is that of the identity-mount plan
    turned 180 deg about the vertical axis through the pinch centre: the same
    grasp for an object symmetric under that turn (the cube), now in the
    natural posture. fr3_link8 keeps the identity-mount plan's orientation, so
    joint 7 stays mid-range. Identity mounts return the input unchanged.
    """
    if not hand_frame_yaw:
        return rotation
    c, s = math.cos(hand_frame_yaw), math.sin(hand_frame_yaw)
    return rotation @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class PrivilegedManipulationPolicy:
    """Shared bounded phase runner for the four fixed cube development tasks.

    Historical grasp evidence predates the current contact scorer: executing a
    complete plan is not a success claim. Only the native independent evaluator
    can accept this revision, which never reads evaluator state.
    """

    policy_id = 'privileged_cube_manipulation_v1'
    # 1.1.6: top-down grasps are turned by the hand-frame (mount) yaw, so with
    # flange_axis_centred_v2 the hand takes the natural posture; identical
    # decisions for v1 (identity) mounts. 1.1.7: a contact-finished close is not
    # stopped by its third controller stall (unchanged before that point).
    VERSION = '1.1.7'

    def __init__(self, scene: dict, task: dict):
        self.operation = TASK_OPERATIONS.get(task.get('id'))
        if self.operation is None:
            raise ValueError('PrivilegedManipulationPolicy supports only the four fixed cube tasks')
        self._require_embodiment(scene)
        goal = task['public_goal']
        if goal.get('operation') != self.operation or goal.get('frame_id') != 'world':
            raise ValueError('Expected a matching world-frame public manipulation goal')
        self.object_id = goal['object_id']
        self.target = _vector(goal['target_position_m'], 3, 'target')
        # Public terminal geometry guides release; success is decided elsewhere.
        radius = goal['success_contract']['terminal']['goal_xy_radius_m']
        if isinstance(radius, bool) or not math.isfinite(radius) or radius <= .01:
            raise ValueError('Invalid public goal radius')
        self.release_radius = float(radius) - .006
        # The public terminal hold sets how long a tabletop grasp must be kept.
        self.terminal_hold_s = float(goal['success_contract']['terminal']['hold_s'])
        obj = next(o for o in scene['objects'] if o['id'] == self.object_id)
        if obj['asset_id'] != 'collab_cube_small':
            raise ValueError('The manipulation baseline requires the fixed cube asset')
        asset_manifest = Path(obj['asset_root']) / 'manifest.json'
        asset_bytes = asset_manifest.read_bytes()
        asset = json.loads(asset_bytes)['assets'][obj['asset_id']]
        lo = _vector(asset['bounds_min_m'], 3, 'asset minimum')
        hi = _vector(asset['bounds_max_m'], 3, 'asset maximum')
        if np.any(hi <= lo):
            raise ValueError('Invalid asset bounds')
        self.box_center = (lo + hi) / 2
        self.half_extents = (hi - lo) / 2
        self.table_z = float(scene['table']['position_m'][2] + scene['table']['half_extents_m'][2])
        self._init_hand(scene, goal)
        self.asset_manifest_sha256 = hashlib.sha256(asset_bytes).hexdigest()
        self.phases = []
        self.phase_index = 0
        self.phase_attempts = 0
        self.last_phase = None
        self.last_sequence = None
        self.pending_hold = None
        self.pending_recovery = None
        self.recovery_hold = False
        self.consecutive_timeouts = 0
        self.decisions = 0
        self.round_index = 0
        self.finished_reason = None
        self.plans = []
        self.trace = []

    def _require_embodiment(self, scene):
        if scene['robot'].get('hand_id', 'sharpa_w01_right') != 'sharpa_w01_right':
            raise ValueError('The manipulation prior requires the SHARPA W01 hand')

    def _init_hand(self, scene, goal):
        from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
        self.hand_frame_yaw = get_hand_spec('sharpa_w01_right', scene['robot'].get('assembly_version'))['hand_frame_yaw_rad']
        hand_path = Path(scene['robot']['assets_root']) / 'robots/sharpa/wave_01/derived/right_sharpa_wave_with_flange.urdf'
        if hashlib.sha256(hand_path.read_bytes()).hexdigest() != HAND_URDF_SHA256:
            raise ValueError('Hand asset differs from the static manipulation prior')
        self.hand = URDFKinematics(hand_path)
        self.hand_goal = dict(PUSH_HAND_POSTURE_RAD)
        self.tips = self._shape_tips(self.hand_goal)
        self.hand_goals = deepcopy(GRASP_HAND_POSTURES_RAD)
        self.grasp_tips = {shape: self._shape_tips(q) for shape, q in self.hand_goals.items()}
        allowed_coupling = goal.get('grasp_constraints', {}).get('criteria', {}).get('maximum_dip_minus_pip_rad', .6)
        if self.operation != 'push' and any(q['right_index_DIP'] - q['right_index_PIP'] > allowed_coupling
                                           for q in self.hand_goals.values()):
            raise ValueError('Hand prior violates the public DIP/PIP posture bound')

    def _shape_tips(self, posture):
        poses = self.hand.forward(posture)
        return {finger: poses[f'right_{"pinky" if finger == "little" else finger}_fingertip'][:3, 3]
                for finger in FINGERS}

    def _object(self, public, state):
        if (type(state.get('sequence')) is not int or state['sequence'] != public['sequence']
                or not math.isfinite(state['sim_time_s'])
                or abs(state['sim_time_s'] - public['sim_time_s']) > 1e-8):
            raise ValueError('Privileged state must match the public observation revision')
        obj = state['objects'][self.object_id]
        position = _vector(obj['position_m'], 3, 'object position')
        rotation = _rotation(obj['quaternion_wxyz'])
        center = position + rotation @ self.box_center
        top = float(center[2] + np.abs(rotation[2]) @ self.half_extents)
        return center, rotation, top

    def _plan(self, observation, center, object_rotation, top):
        # The upright prior defines initial contact geometry. Once planned,
        # carry phases follow fixed wrist targets and retain the accepted hand
        # drive; ordinary load-induced tilt must not discard that plan. The
        # unchanged native evaluator still checks the actual task constraints.
        if object_rotation[2, 2] < math.cos(.35):
            raise ValueError('Object is outside the upright manipulation planning envelope')
        if self.operation == 'push':
            self._plan_push(observation, center, object_rotation, top)
        else:
            self._plan_grasp(observation, center, object_rotation, top)

    def _plan_push(self, observation, center, object_rotation, top):
        difference = self.target[:2] - center[:2]
        distance = float(np.linalg.norm(difference))
        if distance < .002:
            self.finished_reason = 'object_already_at_goal_no_interaction_baseline'
            return
        direction = difference / distance
        world_direction = np.r_[direction, 0.]
        half_width = float(np.abs(object_rotation.T @ world_direction) @ self.half_extents)
        boundary = center[:2] - half_width * direction
        yaw = math.atan2(direction[1], direction[0]) - math.pi / 12
        rotation = rotvec_matrix([0., 0., yaw]) @ np.diag([1., -1., -1.])
        # Matches the successful pad-leading geometry with 6.5 mm clearance for
        # folded fingers above the box; initial 1 mm reset drop is left physical.
        index_z = max(self.table_z + .012, top + .0065 - NONINDEX_RECESS_M)
        gap = .008
        tip = np.r_[boundary - (gap + INDEX_FORWARD_SUPPORT_M) * direction, index_z]
        flange = tip - rotation @ self.tips['index']
        above = flange + [0., 0., .12]
        wrist = observation['proprioception']['geometry']['wrist']
        start = _vector(wrist['position_m'], 3, 'wrist position')
        orientation = _rotation(wrist['quaternion_wxyz'])
        clearance = start.copy()
        clearance[2] = max(start[2], above[2] + .05)
        destination = flange + np.r_[(distance + gap) * direction, 0.]
        unloaded = destination - .015 * world_direction
        self.phases = [
            {'name': 'raise', 'position': clearance, 'rotation': orientation},
            {'name': 'pre_shape', 'position': clearance, 'rotation': orientation, 'tips': self.tips, 'hand_goal': self.hand_goal},
            {'name': 'orient', 'position': clearance, 'rotation': rotation, 'tips': self.tips},
            {'name': 'above', 'position': above, 'rotation': rotation, 'tips': self.tips},
            {'name': 'descend', 'position': flange, 'rotation': rotation, 'tips': self.tips, 'translation_step': .01},
            {'name': 'stroke', 'position': destination, 'rotation': rotation, 'tips': self.tips, 'translation_step': .008},
            {'name': 'unload', 'position': unloaded, 'rotation': rotation, 'tips': self.tips, 'translation_step': .008},
            {'name': 'withdraw', 'position': unloaded + [0., 0., .12], 'rotation': rotation, 'tips': self.tips, 'translation_step': .015},
            {'name': 'settle', 'hold': 1.},
        ]
        self.phase_index = 0
        self.phase_attempts = 0
        self.plans.append({'round': self.round_index, 'sequence': observation['sequence'],
                          'initial_object_position_m': center.tolist(), 'direction_xy': direction.tolist(),
                          'index_world_z_m': index_z, 'flange_world_m': flange.tolist(),
                          'destination_world_m': destination.tolist(), 'object_back_plane_xy_m': boundary.tolist()})

    def _grasp_frame(self, center, object_rotation, top):
        """Wrist rotation and flange position of the top-down pinch (hand-specific)."""
        # Align the pinch across the object's local Y faces. At the fixed
        # initial orientation this is exactly the successful historical prior.
        yaw = math.atan2(object_rotation[1, 0], object_rotation[0, 0])
        rotation = turned_by_mount(rotvec_matrix([0., 0., yaw]) @ np.diag([1., -1., -1.]), self.hand_frame_yaw)
        pinch_center = np.array([.07, .05, .15])
        flange = np.r_[center[:2], top - .01] - rotation @ pinch_center
        return rotation, flange, yaw, pinch_center

    def _plan_grasp(self, observation, center, object_rotation, top):
        rotation, flange, yaw, pinch_center = self._grasp_frame(center, object_rotation, top)
        above = flange + [0., 0., .11]
        wrist = observation['proprioception']['geometry']['wrist']
        start = _vector(wrist['position_m'], 3, 'wrist position')
        orientation = _rotation(wrist['quaternion_wxyz'])
        clearance = start.copy()
        clearance[2] = max(start[2], above[2] + .05)
        opened, closed = self.grasp_tips['open'], self.grasp_tips['closed']
        self.phases = [
            {'name': 'raise', 'position': clearance, 'rotation': orientation},
            {'name': 'pre_shape', 'position': clearance, 'rotation': orientation, 'tips': opened, 'hand_goal': self.hand_goals['open']},
            # The fingers pass close to the object's faces: request tight wrist
            # completion so tolerance-bounded per-move errors cannot accumulate.
            {'name': 'orient', 'position': clearance, 'rotation': rotation, 'tips': opened, 'precise': True},
            {'name': 'above', 'position': above, 'rotation': rotation, 'tips': opened, 'precise': True},
            {'name': 'descend', 'position': flange, 'rotation': rotation, 'tips': opened, 'translation_step': .015,
             'precise': True},
            {'name': 'close', 'position': flange, 'rotation': rotation, 'tips': closed,
             'hand_goal': self.hand_goals['closed'], 'tip_step': .004, 'allow_contact_finish': True},
            {'name': 'establish_contact_hold', 'hold': .4},
            # A tabletop grasp must outlast the public terminal hold; the episode
            # ends at success, so a longer hold never runs past it.
            {'name': 'maintain_contact_hold',
             'hold': self.terminal_hold_s + .6 if self.operation == 'grasp_hold' else .6},
        ]
        if self.operation in ('lift_hold', 'pick_place'):
            # Deliberately omit fingertips: the real controller retains its
            # accepted hand targets/recovery drive instead of squeezing again.
            self.phases += [
                {'name': 'lift', 'position': above, 'rotation': rotation,
                 'translation_step': .015, 'require_contact_before_start': True},
                {'name': 'lift_hold', 'hold': 1.},
            ]
        if self.operation == 'pick_place':
            destination = flange + np.r_[self.target[:2] - center[:2], 0.]
            self.phases += [
                {'name': 'transfer', 'position': destination + [0., 0., .11], 'rotation': rotation, 'translation_step': .015},
                {'name': 'lower', 'position': destination, 'rotation': rotation, 'translation_step': .015},
                {'name': 'release', 'position': destination, 'rotation': rotation,
                 'tips': opened, 'hand_goal': self.hand_goals['open'], 'tip_step': .004},
                {'name': 'withdraw', 'position': destination + [0., 0., .11], 'rotation': rotation, 'tips': opened},
                {'name': 'settle', 'hold': 1.},
            ]
        self.phase_index = 0
        self.phase_attempts = 0
        self.plans.append({'round': self.round_index, 'sequence': observation['sequence'],
                          'operation': self.operation, 'initial_object_position_m': center.tolist(),
                          'flange_world_m': flange.tolist(), 'object_yaw_rad': yaw,
                          'pinch_center_local_m': pinch_center.tolist(),
                          'carry_hand_control': 'retain_accepted_targets_no_fingertip_recommand'})

    def _stop(self, reason):
        self.finished_reason = reason
        return {'action': 'stop', 'reason': reason}

    def _emit(self, decision, observation):
        self.last_sequence = observation['sequence']
        self.decisions += 1
        return decision

    def _recovery_completed(self, observation, feedback):
        """A recovery ACK belongs only to recovery, never the interrupted phase."""
        pending = self.pending_recovery
        action_id = feedback.get('action_id')
        public_time = observation.get('sim_time_s')
        if (feedback.get('status') != 'completed'
                or observation.get('last_action_status') != 'completed'
                or feedback.get('termination_reason') != 'time_hold_completed'
                or not isinstance(action_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:/-]{1,160}', action_id)
                or action_id != observation.get('last_action_id')
                or action_id == pending['failed_action_id']
                or type(observation.get('sequence')) is not int
                or observation['sequence'] <= pending['sequence']
                or type(public_time) not in (int, float) or not math.isfinite(public_time)):
            return False
        elapsed, end = feedback.get('elapsed_sim_s'), feedback.get('sim_time_s')
        if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or type(end) not in (int, float) or not math.isfinite(end)
                or elapsed < pending['duration_s'] - 1e-8
                or abs(end - public_time) > 1e-8
                or abs(end - pending['start_sim_s'] - elapsed) > 1e-8):
            return False
        return True

    def next_decision(self, public_observation, privileged_state, previous_feedback=None):
        if self.finished_reason is not None:
            return {'action': 'stop', 'reason': self.finished_reason}
        observation = public_observation
        center, object_rotation, top = self._object(observation, privileged_state)
        if self.decisions >= 128:
            return self._stop('privileged_policy_action_budget_exhausted')
        if self.last_sequence is not None:
            if not isinstance(previous_feedback, dict):
                raise ValueError('Every issued decision requires execution feedback')
            status = previous_feedback.get('status')
            diagnostic = previous_feedback.get('diagnostic') or {}
            code = diagnostic.get('code')
            if self.pending_recovery is not None:
                if not self._recovery_completed(observation, previous_feedback):
                    return self._stop('controller_recovery_hold_not_completed')
                self.trace.append({'phase': 'controller_limit_recovery',
                                   'interrupted_phase': self.pending_recovery['phase'],
                                   'sequence': observation['sequence'], 'sim_time_s': observation['sim_time_s'],
                                   'acknowledged_hold_s': previous_feedback['elapsed_sim_s']})
                self.pending_recovery = None
                self.consecutive_timeouts = 0
                # The phase index is unchanged. Its failed hold, if any, is
                # reissued below with a new start time after recovery completes.
            elif status == 'failed' and code in ('convergence_timeout', 'tracking_stalled'):
                # Controller 1.7 ends a blocked objective early as tracking_stalled.
                self.consecutive_timeouts += 1
                # 1.1.7: a closing pinch blocked by the object it has just reached is
                # finished by the contact proxy below, not stopped (1.1.6 stopped on
                # the third stall although thumb and index were already in contact).
                closing = (self.phase_index < len(self.phases)
                           and self.phases[self.phase_index].get('allow_contact_finish')
                           and self.phase_attempts > 0 and _thumb_index_contact(observation))
                if self.consecutive_timeouts >= 3 and not closing:
                    return self._stop('repeated_controller_convergence_timeout')
            elif status != 'completed':
                # One explicit hold exercises the ordinary controller recovery;
                # it is not counted as a successful motion or physical recovery.
                if status == 'failed' and code == 'invalid_robot_state' and not self.recovery_hold:
                    self.recovery_hold = True
                    self.pending_recovery = {'phase': self.last_phase, 'duration_s': .5,
                                             'start_sim_s': observation['sim_time_s'],
                                             'sequence': observation['sequence'],
                                             'failed_action_id': previous_feedback.get('action_id')}
                    self.pending_hold = None
                    return self._emit({'action': 'hold', 'duration_s': .5, 'reason': 'controller_limit_recovery'}, observation)
                return self._stop('controller_feedback_' + str(status) + '_' + str(code))
            else:
                self.consecutive_timeouts = 0
            if observation['sequence'] <= self.last_sequence:
                raise ValueError('Execution did not produce a fresh observation')
            if self.pending_hold is not None:
                hold = self.pending_hold
                elapsed = previous_feedback.get('elapsed_sim_s')
                if (status != 'completed' or isinstance(elapsed, bool)
                        or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed)
                        or elapsed < hold['duration_s'] - 1e-8
                        or observation['sim_time_s'] - hold['start_sim_s'] < hold['duration_s'] - 1e-8):
                    return self._stop(hold['phase'] + '_hold_not_completed')
                self.trace.append({'phase': hold['phase'], 'sequence': observation['sequence'],
                                   'sim_time_s': observation['sim_time_s'], 'acknowledged_hold_s': elapsed})
                self.pending_hold = None
                self.phase_index += 1
                self.phase_attempts = 0
        if not self.phases:
            self._plan(observation, center, object_rotation, top)
            if self.finished_reason:
                return self._stop(self.finished_reason)
        geometry = observation['proprioception']['geometry']
        if geometry['wrist']['frame_id'] != 'world' or geometry['fingertip_frame_id'] != 'right_hand_flange':
            raise ValueError('Unsupported public robot geometry frame')
        while self.phase_index < len(self.phases):
            phase = self.phases[self.phase_index]
            self.last_phase = phase['name']
            if phase['name'] == 'stroke' and np.linalg.norm(center[:2] - self.target[:2]) <= self.release_radius:
                # Move away horizontally before lifting; keep the measured wrist
                # reference, avoiding a forward jump to an unreached endpoint.
                now = _vector(geometry['wrist']['position_m'], 3, 'wrist position')
                direction = np.r_[self.plans[-1]['direction_xy'], 0.]
                unloaded = now - .015 * direction
                self.phases[self.phase_index + 1]['position'] = unloaded
                self.phases[self.phase_index + 2]['position'] = unloaded + [0., 0., .12]
                self.trace.append({'phase': 'object_near_goal_begin_release', 'sequence': observation['sequence'],
                                   'xy_error_m': float(np.linalg.norm(center[:2] - self.target[:2]))})
                self.phase_index += 1
                self.phase_attempts = 0
                continue
            if self.phase_attempts >= 45:
                return self._stop('phase_did_not_converge_' + phase['name'])
            if phase.get('require_contact_before_start') and self.phase_attempts == 0 and not _thumb_index_contact(observation):
                return self._stop('thumb_index_contact_proxy_missing_before_lift')
            if 'hold' in phase:
                self.pending_hold = {'phase': phase['name'], 'duration_s': phase['hold'],
                                     'start_sim_s': observation['sim_time_s']}
                return self._emit({'action': 'hold', 'duration_s': phase['hold'],
                                   'reason': 'privileged_' + self.operation + '_' + phase['name']}, observation)
            dp = phase['position'] - geometry['wrist']['position_m']
            dr = matrix_rotvec(phase['rotation'] @ _rotation(geometry['wrist']['quaternion_wxyz']).T)
            tips = {f: point - np.asarray(geometry['fingertip_positions_m'][f]) for f, point in phase.get('tips', {}).items()}
            reached = (np.linalg.norm(dp) <= .0025 and np.linalg.norm(dr) <= .035
                       and all(np.linalg.norm(x) <= .0015 for x in tips.values()))
            contact_finish = bool(phase.get('allow_contact_finish') and self.phase_attempts > 0
                                  and _thumb_index_contact(observation))
            if reached or contact_finish:
                self.trace.append({'phase': phase['name'], 'sequence': observation['sequence'],
                                   'sim_time_s': observation['sim_time_s'], 'public_geometry_reached': bool(reached),
                                   'contact_proxy_phase_end': contact_finish,
                                   'last_primitive_status': None if previous_feedback is None else previous_feedback.get('status')})
                self.phase_index += 1
                self.phase_attempts = 0
                continue
            scale = .5 if self.consecutive_timeouts else 1.
            decision = {'action': 'move_geometry', 'duration_s': 5., 'reason': 'privileged_' + self.operation + '_' + phase['name'],
                        'wrist_delta_m': _capped(dp, phase.get('translation_step', .045) * scale).tolist(),
                        'wrist_rotation_delta_rad': _capped(dr, .14 * scale).tolist()}
            if phase.get('precise'):
                decision.update(position_tolerance_m=PRECISE_POSITION_TOLERANCE_M,
                                orientation_tolerance_rad=PRECISE_ORIENTATION_TOLERANCE_RAD)
            tip_step = min(.009, phase.get('tip_step', .009)) * scale
            if 'hand_goal' in phase and not self._direct_hand_waypoint(phase, tips):
                measured = observation['proprioception']['joint_positions_rad']
                reference = {j.name: float(np.clip(measured[j.name], j.lower, j.upper)) for j in self.hand.active}
                differences = {n: phase['hand_goal'][n] - q for n, q in reference.items()}
                fraction = min(1., .12 * scale / max(max(abs(x) for x in differences.values()), 1e-15))
                for _ in range(32):
                    posed = self._shape_tips({n: q + fraction * differences[n] for n, q in reference.items()})
                    deltas = {f: posed[f] - geometry['fingertip_positions_m'][f] for f in tips}
                    if max(np.linalg.norm(x) for x in deltas.values()) <= tip_step + 1e-12:
                        break
                    fraction *= .5
                else:
                    return self._stop('no_bounded_fingertip_waypoint')
                decision['fingertip_delta_m'] = {f: x.tolist() for f, x in deltas.items()}
            elif tips:
                decision['fingertip_delta_m'] = {f: _capped(x, tip_step).tolist() for f, x in tips.items()}
            self.phase_attempts += 1
            return self._emit(decision, observation)
        if self.operation != 'push':
            return self._stop(self.operation + '_trajectory_complete_independent_score_required')
        if np.linalg.norm(center[:2] - self.target[:2]) <= self.release_radius:
            return self._stop('released_trajectory_complete_independent_score_required')
        if self.round_index >= 1:
            return self._stop('released_object_outside_target_after_two_rounds')
        self.round_index += 1
        self._plan(observation, center, object_rotation, top)
        # Do not process the previous ACK twice while transitioning rounds.
        self.last_sequence = None
        return self.next_decision(observation, privileged_state)

    def _direct_hand_waypoint(self, phase, tips):
        """Whether a posture-shaped hand phase steps straight toward its tip targets
        instead of along the interpolated posture (never for this SHARPA policy)."""
        return False

    def diagnostic(self):
        return {'policy_id': self.policy_id, 'version': self.VERSION, 'ranked_agent': False, 'model_calls': 0,
                'privileged_inputs': ['object_position_m', 'object_quaternion_wxyz', 'asset_bounds', 'table_height'],
                'score_inputs': False, 'object_state_writeback': False,
                'operation': self.operation,
                'robot_prior_sha256': ROBOT_PRIOR_SHA256 if self.operation == 'push' else GRASP_PRIOR_SUMMARY_SHA256,
                'robot_prior_source': None if self.operation == 'push' else GRASP_PRIOR_SOURCE,
                'hand_urdf_sha256': HAND_URDF_SHA256,
                'asset_manifest_sha256': self.asset_manifest_sha256,
                'phase': self.last_phase, 'decisions': self.decisions,
                'round': self.round_index, 'finished_reason': self.finished_reason,
                'plans': deepcopy(self.plans), 'trace': deepcopy(self.trace),
                'contact_scope': 'Identity-free uncalibrated pad proxy; self-contact possible; not a grasp-success certificate',
                'scope': 'Engineering baseline; geometry planning is not collision or success certification'}
