"""Human motion → pinned retarget callback → agent bench joint command.

This module constructs commands only. The caller owns HTTP acknowledgement,
control rate, retries, episode logs, and the simulator owner thread.
"""
from __future__ import annotations
from bench.protocol.schemas import DirectJointAction, RobotCapabilities, RobotJointCommand
from uni_teleop.mapping.manus.bridge import RetargetRegistry


def build_command(frame: dict, registry: RetargetRegistry, capabilities: RobotCapabilities, *,
                  previous_targets: dict[str, float], now_ns: int, now_clock_id: str, duration_s: float | None = None):
    """Use the actual robot morphology and preserve arm targets when no tracked wrist exists."""
    if frame['robot_profile_id'] != capabilities.robot_id:
        raise ValueError('capture retarget profile and selected robot differ')
    dt = capabilities.control_period_s if duration_s is None else duration_s
    if abs(dt-capabilities.control_period_s) > 1e-9:
        raise ValueError('teleop command must span one configured control period')
    limits = {name: (limit.lower_rad, limit.upper_rad) for name, limit in capabilities.joint_limits.items()}
    targets, metadata = registry.map(frame, previous_targets=previous_targets, joint_limits=limits,
                                    arm_joint_names=capabilities.arm_joints, now_ns=now_ns, now_clock_id=now_clock_id)
    for name, value in targets.items():
        speed = capabilities.joint_limits[name].max_velocity_rad_s
        if speed is not None and abs(value-previous_targets[name]) > speed*dt+1e-9:
            raise ValueError(f'retarget command exceeds declared joint speed: {name}')
    command = RobotJointCommand(joint_positions_rad={name: float(value) for name, value in targets.items()}, duration_s=float(dt))
    metadata.update(source_stream_id=frame['stream_id'], source_sequence=frame['sequence'],
                    calibration_id=frame['calibration_id'], robot_profile_id=frame['robot_profile_id'])
    return command, metadata


def direct_action(command: RobotJointCommand, capabilities: RobotCapabilities, *,
                  action_id: str, observation_sequence: int) -> DirectJointAction:
    """HTTP POST /episodes/{id}/actions; identical retries reuse this exact payload/action_id."""
    if set(command.joint_positions_rad) != set(capabilities.joint_limits):
        raise ValueError('command joint names differ from selected robot')
    return DirectJointAction(action_id=action_id, observation_sequence=observation_sequence,
        duration_s=command.duration_s,
        arm_joint_positions_rad={name: command.joint_positions_rad[name] for name in capabilities.arm_joints},
        hand_joint_positions_rad={name: command.joint_positions_rad[name] for name in capabilities.hand_joints})
