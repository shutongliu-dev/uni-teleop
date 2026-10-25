"""Privileged development baseline for FR3 + other registered hands.

The SHARPA baseline's phase runner, feedback/recovery handling and ordinary
decisions are reused unchanged; only the robot-only hand priors differ:

* push: a flat-hand paddle. The palm normal (flange +x) faces the push
  direction, fingers (flange +z) point down with their tips 15 mm above the
  table; the hand is centred on the cube's push line. Fingers are slightly
  flexed (0.15 rad) to stay off the straight-finger singularity.
* grasp tasks: per-hand pinch priors (thumb against index), designed on CPU
  from the controller's own FK/IK; see PINCH_PRIORS provenance.

Like the SHARPA baseline it reads true object poses (privileged, recorded),
emits only ordinary geometry decisions, and never certifies success.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from bench.control.fingertip import Fr3SharpaGeometry
from .privileged_policy import PrivilegedManipulationPolicy, turned_by_mount

PADDLE_PAD_THICKNESS_M = .012
PADDLE_TIP_HEIGHT_M = .015
PADDLE_FLEX_RAD = .15
# Per-hand flexion joints of the non-thumb fingers (abduction joints stay 0).
PADDLE_FLEXION = {
    'allegro_v5_right': [f'allegro_joint_{n}_0' for n in (1, 2, 3, 5, 6, 7, 9, 10, 11)],
    'leap_v1_right': [f'leap_{n}' for n in (1, 2, 3, 5, 6, 7, 9, 10, 11)],
    'shadow_hand_e_right': [f'rh_{f}J{n}' for f in ('FF', 'MF', 'RF', 'LF') for n in (3, 2)],
}
PINCH_PRIORS = {
    # Designed on CPU by probes/design_pinch_priors.py attempt 5 (pinch across the
    # cube's 32 mm local-x faces; open/closed tip-centre gaps 69/49 mm; margins:
    # open column 8.5 mm, closed non-pad 10.2 mm,
    # finger self-clearance 5.8 mm). Robot-only prior, not a grasp certificate.
    # 0.1.3: middle/ring PIP/DIP 1.6 -> 1.0 rad (MCP stays 1.6). The design checked
    # finger groups against each other but not against the palm: at 1.6/1.6/1.6
    # their fingertips rest on the palm (0.07 mm source-mesh gap), which stalled the
    # pre-shape natively (mount_v2_baseline_01/leap_grasp). Now 24-28 mm; the
    # thumb/index pinch and every design margin are unchanged (pinch_priors/NOTE.md).
    # 0.1.4: staged pre-shape ('preshape_stages': thumb, then middle/ring fold, then the
    # index in the final pre-shape). Moving all fingers at once swept the abducting index
    # fingertip through the folding middle finger (1.2 mm on the joint path; jammed natively
    # in mount_v2_baseline_01/leap_grasp_r2); folding first left the thumb 2.7 mm from the
    # middle finger. This order keeps every finger/finger and finger/palm gap >= 7.8 mm on
    # the source meshes (pinch_priors/leap_preshape_path_check.json).
    'leap_v1_right': {
        'preshape_stages': [['leap_12', 'leap_13', 'leap_14', 'leap_15'],
                            ['leap_5', 'leap_6', 'leap_7', 'leap_9', 'leap_10', 'leap_11']],
        'pinch_center_local_m': [0.03954, 0.08045, 0.11198],
        'pinch_axis_local': [-0.6409, 0.76762, 0.0],
        'pinch_faces': 'cube local x (32 mm)', 'below_top_m': 0.012,
        'open': {'leap_0': 0.777, 'leap_1': 0.0373, 'leap_10': 1.0, 'leap_11': 1.0, 'leap_12': 0.0571, 'leap_13': 0.0115, 'leap_14': 0.6463, 'leap_15': 0.575, 'leap_2': 1.7816, 'leap_3': 0.8718, 'leap_4': 0.0, 'leap_5': 1.6, 'leap_6': 1.0, 'leap_7': 1.0, 'leap_8': 0.0, 'leap_9': 1.6},
        'closed': {'leap_0': 0.9069, 'leap_1': 0.0353, 'leap_10': 1.0, 'leap_11': 1.0, 'leap_12': 0.1182, 'leap_13': 0.0361, 'leap_14': 0.6049, 'leap_15': 0.7554, 'leap_2': 1.885, 'leap_3': 0.8934, 'leap_4': 0.0, 'leap_5': 1.6, 'leap_6': 1.0, 'leap_7': 1.0, 'leap_8': 0.0, 'leap_9': 1.6},
    },
}


class PrivilegedEmbodimentPolicy(PrivilegedManipulationPolicy):
    policy_id = 'privileged_cube_manipulation_embodiment_v1'
    # 0.1.1: the push paddle keeps the thumb at the hand's home posture and does
    # not command it (LEAP's zero thumb self-collides and stalled 0.1.0).
    VERSION = '0.1.4'   # 0.1.2: mount-yaw-turned pinch; 0.1.3: LEAP palm clearance; 0.1.4: staged pre-shape

    def _require_embodiment(self, scene):
        hand = scene['robot'].get('hand_id', 'sharpa_w01_right')
        if hand not in PADDLE_FLEXION:
            raise ValueError('The embodiment baseline supports the registered non-SHARPA hands only')
        self.hand_id = hand

    def _init_hand(self, scene, goal):
        robot = scene['robot']
        self.geometry = Fr3SharpaGeometry(robot['assets_root'], robot['base_position_m'], hand_id=self.hand_id,
                                          assembly_version=robot.get('assembly_version'))
        self.hand = self.geometry.hand
        self.fingers = tuple(self.geometry.fingertip_links)
        from bench.simulation.isaacsim_backend.robot_registry import get_hand_spec
        spec = get_hand_spec(self.hand_id, robot.get('assembly_version'))
        self.hand_frame_yaw = spec['hand_frame_yaw_rad']
        home = spec.get('home_joint_positions_rad', {})
        self.hand_goal = self._posture({**home, **{name: PADDLE_FLEX_RAD for name in PADDLE_FLEXION[self.hand_id]}})
        self.tips = {f: p for f, p in self._shape_tips(self.hand_goal).items() if f != 'thumb'}
        prior = PINCH_PRIORS.get(self.hand_id)
        if self.operation != 'push' and prior is None:
            raise ValueError(f'No pinch prior is designed for {self.hand_id}')
        self.pinch = deepcopy(prior)
        self.hand_goals = {} if prior is None else {shape: self._posture(prior[shape]) for shape in ('open', 'closed')}
        self.grasp_tips = {shape: self._shape_tips(q) for shape, q in self.hand_goals.items()}

    def _grasp_frame(self, center, object_rotation, top):
        """Top-down pinch frame from the hand's designed prior (pinch axis -> the
        prior's cube faces, fingers down), choosing the yaw nearest zero."""
        from bench.geometry import rotvec_matrix
        prior = self.pinch
        axis_local = np.asarray(prior['pinch_axis_local'], dtype=float)
        center_local = np.asarray(prior['pinch_center_local_m'], dtype=float)
        face_axis = object_rotation[:, 0 if prior['pinch_faces'].startswith('cube local x') else 1].copy()
        face_axis[2] = 0.
        face_axis /= np.linalg.norm(face_axis)
        base = np.diag([1., -1., -1.])                     # flange z -> world -z
        axis_world = base @ axis_local
        yaw = math.atan2(face_axis[1], face_axis[0]) - math.atan2(axis_world[1], axis_world[0])
        yaw = min((math.remainder(yaw + k * math.pi, 2 * math.pi) for k in (0, 1)), key=abs)
        rotation = turned_by_mount(rotvec_matrix([0., 0., yaw]) @ base, self.hand_frame_yaw)
        flange = np.r_[center[:2], top - prior['below_top_m']] - rotation @ center_local
        return rotation, flange, yaw, center_local

    def _plan_grasp(self, observation, center, object_rotation, top):
        """The base top-down pinch plan; a prior may stage its pre-shape: each listed
        joint group moves to the open posture in turn (the others wait), and the final
        pre_shape moves whatever remains."""
        super()._plan_grasp(observation, center, object_rotation, top)
        stages = self.pinch.get('preshape_stages') if self.pinch else None
        if not stages:
            return
        index = next(i for i, phase in enumerate(self.phases) if phase['name'] == 'pre_shape')
        measured = observation['proprioception']['joint_positions_rad']
        posture = self._posture({j.name: measured[j.name] for j in self.hand.active})
        staged = []
        for number, joints in enumerate(stages, 1):
            posture = self._posture({**posture, **{name: self.hand_goals['open'][name] for name in joints}})
            staged.append(dict(self.phases[index], name=f'pre_shape_stage_{number}', hand_goal=dict(posture),
                               tips=self._shape_tips(posture)))
        self.phases[index:index] = staged

    def _direct_hand_waypoint(self, phase, tips):
        """Within the prior's 'preshape_direct_window_m' of its tip targets a pre-shape
        steps straight toward them. The controller resolves a redundant finger's IK
        from the measured posture, so the posture-interpolated waypoints alone can
        cycle a few mm short of the target (LEAP thumb, mount_v2_baseline_01/leap_grasp_r3)."""
        window = (self.pinch or {}).get('preshape_direct_window_m')
        return (window is not None and bool(tips) and phase['name'].startswith('pre_shape')
                and max(float(np.linalg.norm(x)) for x in tips.values()) <= window)

    def _posture(self, values):
        """A complete public hand posture: listed joints, all others at zero (clipped)."""
        return {j.name: float(np.clip(values.get(j.name, 0.), j.lower, j.upper)) for j in self.hand.active}

    def _shape_tips(self, posture):
        poses = self.hand.forward(posture)
        return {finger: self.geometry._tip(poses, finger) for finger in self.fingers}

    def _plan_push(self, observation, center, object_rotation, top):
        difference = self.target[:2] - center[:2]
        distance = float(np.linalg.norm(difference))
        if distance < .002:
            self.finished_reason = 'object_already_at_goal_no_interaction_baseline'
            return
        direction = difference / distance
        d3 = np.r_[direction, 0.]
        normal = np.array([-direction[1], direction[0], 0.])
        half_width = float(np.abs(object_rotation.T @ d3) @ self.half_extents)
        # Palm normal along the push, fingers down: flange x = d, z = -world z.
        rotation = np.column_stack([d3, np.cross([0., 0., -1.], d3), [0., 0., -1.]])
        paddle = [rotation @ self.tips[f] for f in self.tips]
        lowest = min(p[2] for p in paddle)
        front = max(p @ d3 for p in paddle) + PADDLE_PAD_THICKNESS_M
        lateral = float(np.mean([p @ normal for p in paddle]))
        gap = .008
        along = float(center[:2] @ direction) - half_width - gap - front
        flange = along * d3 + (float(center[:2] @ normal[:2]) - lateral) * normal
        flange[2] = self.table_z + PADDLE_TIP_HEIGHT_M - lowest
        above = flange + [0., 0., .12]
        wrist = observation['proprioception']['geometry']['wrist']
        start = np.asarray(wrist['position_m'], dtype=float)
        orientation = self._wrist_rotation(wrist)
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
            {'name': 'stroke', 'position': destination, 'rotation': rotation, 'tips': self.tips, 'translation_step': .008},
            {'name': 'unload', 'position': unloaded, 'rotation': rotation, 'tips': self.tips, 'translation_step': .008},
            {'name': 'withdraw', 'position': unloaded + [0., 0., .12], 'rotation': rotation, 'tips': self.tips, 'translation_step': .015},
            {'name': 'settle', 'hold': 1.},
        ]
        self.phase_index = 0
        self.phase_attempts = 0
        self.plans.append({'round': self.round_index, 'sequence': observation['sequence'], 'hand_id': self.hand_id,
                          'strategy': 'flat_hand_paddle', 'initial_object_position_m': center.tolist(),
                          'direction_xy': direction.tolist(), 'flange_world_m': flange.tolist(),
                          'destination_world_m': destination.tolist(), 'paddle_front_m': front,
                          'paddle_lowest_tip_m': lowest})

    @staticmethod
    def _wrist_rotation(wrist):
        from bench.geometry import quaternion_matrix
        return quaternion_matrix(np.asarray(wrist['quaternion_wxyz'], dtype=float))

    def diagnostic(self):
        value = super().diagnostic()
        value.update(policy_id=self.policy_id, version=self.VERSION, hand_id=self.hand_id,
                     hand_urdf_sha256=None, robot_prior_sha256=None,
                     robot_prior_source='embodiment_policy: paddle push / PINCH_PRIORS',
                     controller_assets_sha256=self.geometry.asset_sha256)
        return value
