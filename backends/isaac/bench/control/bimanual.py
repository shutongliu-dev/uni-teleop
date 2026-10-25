"""Bimanual classical executor: one unchanged ClassicalFingertipPolicy per arm.

The dual FR3 + SHARPA robot is controlled as two independent classical
executors (left/right), each bound to its own arm + hand joints, flange frame
and base offset. A bimanual action carries one geometry objective per moved
hand (``hand_geometries``); an omitted hand maintains its reference with a
timed hold that ends when every moved hand has completed. A failure of either
side fails the action and stops the other side. No coordination, force
control or grasp certification is implied.
"""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from bench.protocol.policies import PolicyStep
from bench.protocol.schemas import (AgentObservation, PolicyDescriptor, RobotExecutionState, RobotJointCommand,
                                 RobotPostureProfile, System0Action)
from bench.control.embodiment import BIMANUAL_ROBOT, load_bimanual_side
from bench.control.fingertip import CONTROL_DT, ClassicalFingertipPolicy, ExecutorLimits, Fr3SharpaGeometry

SIDES = ('left', 'right')
VERSION = '1.2.0'


class BimanualGeometry:
    """Public robot-only FK of both arms (one Fr3SharpaGeometry per side)."""
    robot_id = BIMANUAL_ROBOT

    def __init__(self, assets_root, base_position_m, base_quaternion_wxyz=(1., 0., 0., 0.), *, hand_id=BIMANUAL_ROBOT,
                 assembly_version=None, arm_base_poses=None, instances=None):
        from bench.simulation.isaacsim_backend.robot_instances import ROBOT_ID, checked_instances
        if hand_id == ROBOT_ID:
            from .instance_embodiment import load_instance
            configuration = {'hand_id': hand_id, 'assembly_version': assembly_version, 'instances': instances}
            if arm_base_poses is not None:
                configuration['arm_base_poses'] = arm_base_poses
            self.instances = checked_instances(configuration)
            self.robot_id = hand_id
            self.sides = {name: Fr3SharpaGeometry(assets_root, base_position_m, base_quaternion_wxyz,
                embodiment=load_instance(assets_root, name, item)) for name, item in self.instances.items()}
        elif hand_id != BIMANUAL_ROBOT or instances is not None:
            raise ValueError('BimanualGeometry supports the dual FR3 + SHARPA robot only')
        else:
            self.sides = {side: Fr3SharpaGeometry(assets_root, base_position_m, base_quaternion_wxyz,
                                              embodiment=load_bimanual_side(assets_root, side, assembly_version,
                                                                            arm_base_poses=arm_base_poses))
                      for side in SIDES}
        self.base = next(iter(self.sides.values())).base
        self.names = [name for side in self.sides for name in self.sides[side].names]
        self.joints = [joint for side in self.sides for joint in self.sides[side].joints]
        self.by_name = {joint.name: joint for joint in self.joints}
        self.frame_ids = [self.sides[side].frame_id for side in self.sides]
        self.finger_joints = {f'{side}_{finger}': chain for side in self.sides
                              for finger, chain in self.sides[side].finger_joints.items()}
        self.asset_sha256 = {path: sha for side in self.sides for path, sha in self.sides[side].asset_sha256.items()}
        self.embodiment = SimpleNamespace(expected_mount={side: self.sides[side].embodiment.expected_mount for side in self.sides})
        self.arm_names = [name for side in self.sides for name in self.sides[side].arm.names]

    def split(self, joint_map):
        return {side: {name: joint_map[name] for name in self.sides[side].names} for side in self.sides}

    def public_hand_geometries(self, joint_positions_rad):
        parts = self.split(joint_positions_rad)
        return {side: self.sides[side].public_finger_geometry(parts[side]) for side in self.sides}


class BimanualFingertipPolicy:
    """System0 plugin for the dual robot; the per-side executors are unchanged 1.9.0 code."""

    def __init__(self, assets_root, base_position_m, base_quaternion_wxyz=(1., 0., 0., 0.), *, limits=None,
                 hand_id=BIMANUAL_ROBOT, assembly_version=None, arm_base_poses=None, instances=None):
        self.geometry = BimanualGeometry(assets_root, base_position_m, base_quaternion_wxyz, hand_id=hand_id,
                                         assembly_version=assembly_version, arm_base_poses=arm_base_poses, instances=instances)
        self.limits = limits or ExecutorLimits()
        self.sides = {side: ClassicalFingertipPolicy(assets_root, base_position_m, base_quaternion_wxyz,
                                                     limits=self.limits, embodiment=self.geometry.sides[side].embodiment)
                      for side in self.geometry.sides}
        for side in self.sides:
            # Share the already-loaded per-side geometry (identical embodiment object).
            self.sides[side].geometry = self.geometry.sides[side]
        representative = self.sides['right'] if instances is None else next(iter(self.sides.values()))
        single = representative.descriptor
        self.provenance = {'sides': {side: self.sides[side].provenance for side in self.sides},
                           'composition': 'independent per-arm classical executors; omitted hand holds until moved hands complete; '
                                          'either side failing fails the action and stops the other'}
        if arm_base_poses is not None or instances is not None:
            self.provenance['world_from_arm_base'] = {
                side: self.geometry.sides[side].arm_base.tolist() for side in self.sides}
        if instances is not None:
            from pathlib import Path
            self.provenance['instances'] = self.geometry.instances
            self.provenance['composition_sources'] = {str(path.relative_to(Path(__file__).parents[2])):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in (
                    Path(__file__), Path(__file__).with_name('instance_embodiment.py'),
                    Path(__file__).parents[1] / 'simulation/isaacsim_backend/robot_instances.py',
                    Path(__file__).parents[1] / 'simulation/isaacsim_backend/instance_asset.py')}
        digest = hashlib.sha256(json.dumps(self.provenance, sort_keys=True, separators=(',', ':'),
                                           allow_nan=False, default=str).encode()).hexdigest()
        pairs = [pair for side in self.sides for pair in self.sides[side].geometry.hand_posture_prior.describe()['joint_pairs']]
        profile = {key: value for key, value in representative.geometry.hand_posture_prior.describe().items()
                   if key not in ('notes', 'joint_pairs', 'profile_id')}
        self.descriptor = PolicyDescriptor(policy_id='classical_fingertip_bimanual' if instances is None else 'classical_fingertip_instances', version=VERSION,
            implementation_sha256=digest, trained=False, observation_profile='vision_proprio',
            skills=['move_geometry', 'hold'], hold_modes=single.hold_modes, control_period_s=CONTROL_DT,
            geometry_inputs=['wrist_pose', 'fingertip_positions', 'hand_posture'], force_objectives=False,
            robot_posture_profile=RobotPostureProfile(profile_id='sharpa_dip_pip_bimanual_v1', joint_pairs=pairs, **profile),
            hand_goal_semantics=single.hand_goal_semantics, geometry_tolerance_policy=single.geometry_tolerance_policy,
            camera_observation_mode='decision_only', supported_robot_ids=[self.geometry.robot_id])
        self._active = None
        self._is_hold = False

    # --- per-side projections -------------------------------------------------
    def _side_observation(self, observation, side):
        names = set(self.geometry.sides[side].names)
        proprio = observation.proprioception
        keep = lambda values: None if values is None else {n: v for n, v in values.items() if n in names}
        side_proprio = proprio.model_copy(update={
            'joint_positions_rad': keep(proprio.joint_positions_rad),
            'joint_velocities_rad_s': keep(proprio.joint_velocities_rad_s),
            'commanded_joint_positions_rad': keep(proprio.commanded_joint_positions_rad),
            'geometry': None, 'hand_geometries': None})
        return observation.model_copy(update={'proprioception': side_proprio})

    def _side_capabilities(self, capabilities, side):
        geometry = self.geometry.sides[side]
        return SimpleNamespace(control_period_s=capabilities.control_period_s, arm_joints=list(geometry.arm.names),
                               hand_joints=list(geometry.hand.names),
                               joint_limits={n: capabilities.joint_limits[n] for n in geometry.names})

    @staticmethod
    def _state(policy):
        return {key: copy.deepcopy(value) for key, value in vars(policy).items()
                if key not in ('geometry', 'limits', 'provenance', 'descriptor')}

    @staticmethod
    def _restore_state(policy, saved):
        # A first begin installs plan fields that did not exist before it.
        # Remove those too when a later instance rejects the composite plan.
        for key in tuple(vars(policy)):
            if key not in ('geometry', 'limits', 'provenance', 'descriptor') and key not in saved:
                delattr(policy, key)
        vars(policy).update(saved)

    # --- System0 protocol -----------------------------------------------------
    def reset(self, spec, capabilities):
        for side in self.sides:
            self.sides[side].reset(spec, self._side_capabilities(capabilities, side))
        self._active = None

    def begin(self, action, observation):
        if self._active is not None:
            raise ValueError('Cancel or finish the active action before beginning another')
        if action.skill == 'hold':
            plan = {side: System0Action(action_id=action.action_id, observation_sequence=action.observation_sequence,
                                        skill='hold', timeout_sim_s=action.timeout_sim_s,
                                        hold_duration_s=action.hold_duration_s,
                                        hold_mode=action.hold_mode) for side in self.sides}
            moving = set()
        else:
            if action.geometry is not None or not action.hand_geometries:
                raise ValueError('Bimanual move_geometry needs hand_geometries (one objective per moved hand)')
            moving = set(action.hand_geometries)
            if not moving <= set(self.sides):
                raise ValueError('Geometry selects an undeclared robot instance')
            plan = {side: (System0Action(action_id=action.action_id, observation_sequence=action.observation_sequence,
                                         skill='move_geometry', timeout_sim_s=action.timeout_sim_s,
                                         geometry=action.hand_geometries[side]) if side in moving else
                           System0Action(action_id=action.action_id, observation_sequence=action.observation_sequence,
                                         skill='hold', timeout_sim_s=action.timeout_sim_s,
                                         hold_duration_s=action.timeout_sim_s)) for side in self.sides}
        # Transactional across both sides: a rejected side restores the other.
        saved = {side: self._state(self.sides[side]) for side in self.sides}
        try:
            for side in self.sides:
                self.sides[side].begin(plan[side], self._side_observation(observation, side))
        except BaseException:
            for side in self.sides:
                self._restore_state(self.sides[side], saved[side])
            raise
        self._active = {side: True for side in self.sides}
        self._moving = moving
        self._is_hold = action.skill == 'hold'

    def _merged_state(self, observation, states):
        objectives = [states[side].objective_status for side in self.sides if side in self._moving]
        objective = ('not_checked' if not objectives or 'not_checked' in objectives
                     else 'reached' if all(x == 'reached' for x in objectives) else 'not_reached')
        stationary = [states[side].reference_stationary for side in self.sides]
        settled = [states[side].wrist_settled for side in self.sides if side in self._moving]
        return RobotExecutionState(protocol='robot_execution_state_v1', sim_time_s=float(observation.sim_time_s),
            completion_scope='timed_reference_maintenance' if self._is_hold else 'requested_geometry',
            objective_status=objective,
            reference_stationary=None if None in stationary else all(stationary),
            wrist_settled=None if not settled or None in settled else all(settled),
            posture_objective_joint_names=[n for side in self.sides for n in states[side].posture_objective_joint_names],
            recovery_fixed_drive_joint_names=[n for side in self.sides for n in states[side].recovery_fixed_drive_joint_names],
            maintained_joint_error_rad={k: v for side in self.sides for k, v in states[side].maintained_joint_error_rad.items()})

    def step(self, observation):
        if self._active is None:
            raise ValueError('No active bimanual action')
        results, states = {}, {}
        for side in self.sides:
            side_observation = self._side_observation(observation, side)
            if self._active[side]:
                results[side] = self.sides[side].step(side_observation)
                states[side] = results[side].robot_execution_state
            else:
                states[side] = self.sides[side].execution_state(side_observation)
        failed = [side for side, result in results.items() if result.status == 'failed']
        if failed:
            for side in self.sides:
                if side not in failed:
                    self.sides[side].discard_pending_step()
                    self.sides[side].cancel('peer_hand_failed')
            self._active = None
            first = results[failed[0]]
            return PolicyStep(status='failed', detail=f'{failed[0]} hand: {first.detail}', diagnostic=first.diagnostic,
                              robot_execution_state=self._merged_state(observation, states))
        for side, result in results.items():
            if result.status == 'completed':
                self._active[side] = False
        moved_done = all(not self._active[side] for side in self._moving)
        if self._moving and moved_done:
            # Moved hands are done: end the omitted hands' maintenance holds.
            for side in self.sides:
                if self._active[side]:
                    self.sides[side].discard_pending_step()
                    self.sides[side].finish_hold_early()
                    self._active[side] = False
        if not any(self._active.values()):
            self._active = None
            label = 'Both hands' if self.geometry.robot_id == BIMANUAL_ROBOT else 'All instances'
            return PolicyStep(status='completed', detail=label + ' completed their requested geometry or maintenance; '
                              'no grasp/contact/task success is inferred', robot_execution_state=self._merged_state(observation, states))
        command = {}
        for side in self.sides:
            result = results.get(side)
            if result is not None and result.status == 'running':
                command.update(result.command.joint_positions_rad)
            else:
                # A completed side keeps its last executed drive reference.
                command.update(dict(zip(self.geometry.sides[side].names, self.sides[side].previous.tolist())))
        return PolicyStep(status='running', command=RobotJointCommand(duration_s=CONTROL_DT, joint_positions_rad=command),
                          robot_execution_state=self._merged_state(observation, states))

    def commit_pending_step(self):
        for side in self.sides:
            self.sides[side].commit_pending_step()

    def discard_pending_step(self):
        for side in self.sides:
            self.sides[side].discard_pending_step()

    def cancel(self, reason):
        for side in self.sides:
            self.sides[side].cancel(reason)
        self._active = None

    def execution_state(self, observation):
        states = {side: self.sides[side].execution_state(self._side_observation(observation, side)) for side in self.sides}
        return self._merged_state(observation, states)
