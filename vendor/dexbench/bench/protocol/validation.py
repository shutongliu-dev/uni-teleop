"""Server-side capability negotiation and command validation."""
from __future__ import annotations

import math

from .schemas import (DirectJointAction, EpisodeSpec, RobotCapabilities,
                      RobotJointCommand, System0Action)


def negotiate(spec: EpisodeSpec, capabilities: RobotCapabilities) -> None:
    if (spec.robot_id, spec.robot_asset_sha256) != (capabilities.robot_id, capabilities.robot_asset_sha256):
        raise ValueError("robot identity/asset pin does not match server capabilities")
    if spec.track not in capabilities.tracks or spec.observation_profile not in capabilities.observation_profiles:
        raise ValueError("requested track/observation profile is unavailable")
    if spec.track in ('system0', 'free_tool_agent') and spec.system0 != capabilities.system0:
        raise ValueError("System0 descriptor must exactly match registered server policy")
    if (spec.system0 and spec.system0.supported_robot_ids
            and spec.robot_id not in spec.system0.supported_robot_ids):
        raise ValueError('The pinned controller does not support this robot')


def _period(duration: float, control_period: float) -> None:
    count = duration / control_period
    if count < 1 - 1e-8 or abs(count - round(count)) > 1e-7:
        raise ValueError("duration must be an integer number of native control periods")


def validate_joint_command(command: RobotJointCommand, capabilities: RobotCapabilities, *, single_step: bool = True) -> None:
    if set(command.joint_positions_rad) != set(capabilities.joint_limits):
        raise ValueError("complete named arm and hand joint targets are required")
    _period(command.duration_s, capabilities.control_period_s)
    if single_step and abs(command.duration_s - capabilities.control_period_s) > 1e-9:
        raise ValueError("plugin output must cover exactly one native control period")
    for name, position in command.joint_positions_rad.items():
        limit = capabilities.joint_limits[name]
        if not limit.lower_rad <= position <= limit.upper_rad:
            raise ValueError(f"joint target outside physical limits: {name}")


def validate_action(action: DirectJointAction | System0Action,
                    capabilities: RobotCapabilities, spec: EpisodeSpec,
                    *, observation_sequence: int | None = None) -> None:
    negotiate(spec, capabilities)
    if spec.track != 'free_tool_agent' and action.kind != spec.track:
        raise ValueError("action belongs to a different control track")
    if observation_sequence is not None and action.observation_sequence != observation_sequence:
        raise ValueError("action must reference the current observation sequence")
    if isinstance(action, DirectJointAction):
        if set(action.arm_joint_positions_rad) != set(capabilities.arm_joints):
            raise ValueError("arm targets must exactly match advertised arm joints")
        if set(action.hand_joint_positions_rad) != set(capabilities.hand_joints):
            raise ValueError("hand targets must exactly match advertised hand joints")
        if action.duration_s > spec.budgets.max_action_sim_s:
            raise ValueError("action duration exceeds per-action budget")
        validate_joint_command(RobotJointCommand(
            joint_positions_rad=action.arm_joint_positions_rad | action.hand_joint_positions_rad,
            duration_s=action.duration_s), capabilities, single_step=False)
        return
    if action.timeout_sim_s > spec.budgets.max_action_sim_s:
        raise ValueError("skill timeout exceeds per-action budget")
    _period(action.timeout_sim_s, capabilities.control_period_s)
    if action.skill == 'hold':
        if action.hold_mode not in capabilities.system0.hold_modes:
            raise ValueError('Hold mode is not advertised by this executor')
        _period(action.hold_duration_s, capabilities.control_period_s)
    if action.skill not in capabilities.system0.skills:
        raise ValueError("skill is not advertised by the pinned System0 policy")
    if action.hand_geometries is not None:
        for side, objective in action.hand_geometries.items():
            if f'{side}_hand_flange' not in capabilities.frame_ids:
                raise ValueError("hand objective addresses a hand this robot does not advertise")
            _validate_geometry(action, objective, capabilities)
        return
    if action.geometry is not None:
        _validate_geometry(action, action.geometry, capabilities)


def _validate_geometry(action, geometry, capabilities):
    required = set()
    if geometry.wrist:
        required.add("wrist_pose")
    if geometry.contacts:
        required.add("contacts")
    if geometry.wrist_trajectory:
        required.add("wrist_trajectory")
    if geometry.hand_posture_rad:
        required.add("hand_posture")
    if geometry.fingertip_positions_m:
        required.add('fingertip_positions')
    if not required <= set(capabilities.system0.geometry_inputs):
        raise ValueError("policy does not advertise the requested geometry inputs")
    if any(c.desired_normal_force_n is not None for c in geometry.contacts) and not capabilities.system0.force_objectives:
        raise ValueError("policy does not support force objectives")
    if not {c.finger for c in geometry.contacts} <= set(capabilities.finger_names):
        raise ValueError("contact objective requests an unadvertised finger")
    if not set(geometry.fingertip_positions_m) <= set(capabilities.finger_names):
        raise ValueError('fingertip objective requests an unadvertised finger')
    frames = [x.frame_id for x in geometry.contacts]
    if geometry.fingertip_positions_m:
        frames.append(geometry.fingertip_frame_id)
    if geometry.wrist:
        frames.append(geometry.wrist.frame_id)
    frames.extend(x.wrist.frame_id for x in geometry.wrist_trajectory)
    if not set(frames) <= set(capabilities.frame_ids):
        raise ValueError("geometry uses an unadvertised coordinate frame")
    if geometry.wrist_trajectory and geometry.wrist_trajectory[-1].time_s > action.timeout_sim_s:
        raise ValueError("trajectory extends beyond action timeout")
    if geometry.hand_posture_rad:
        if not set(geometry.hand_posture_rad) <= set(capabilities.hand_joints):
            raise ValueError("hand posture contains unknown/non-hand joints")
        for name, value in geometry.hand_posture_rad.items():
            limit = capabilities.joint_limits[name]
            if not limit.lower_rad <= value <= limit.upper_rad:
                raise ValueError(f"hand posture outside limits: {name}")
