"""Public robot description and read-only IK contracts, independent of executors."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field, StrictBool, model_validator

from .hand_presets import HandPresetCatalog
from .schemas import (ActionDiagnostic, Contract, FingerName, GeometryObjective, Hash,
                      JointLimit, JointMap, HandNamespace, Name, Nonnegative, PolicyDescriptor, Positive,
                      RobotGeometry, RobotPostureJointPair, RobotPostureProfile)


def _literal_false(value):
    if value is not False:
        raise ValueError('Expected the literal boolean false')
    return value


FalseClaim = Annotated[Literal[False], BeforeValidator(_literal_false)]
Text = Annotated[str, Field(min_length=1, max_length=4096)]
Side = HandNamespace


class RobotContract(Contract):
    model_config = ConfigDict(strict=True, revalidate_instances='always')


class IKQuery(RobotContract):
    observation_sequence: int = Field(strict=True, ge=0)
    geometry: GeometryObjective | None = None
    hand_geometries: dict[Side, GeometryObjective] | None = None

    @model_validator(mode='after')
    def one_target_form(self):
        if (self.geometry is not None) == bool(self.hand_geometries):
            raise ValueError('Provide geometry or nonempty hand_geometries')
        for target in ([self.geometry] if self.geometry else self.hand_geometries.values()):
            if target.contacts or target.wrist_trajectory:
                raise ValueError('IK queries support wrist, fingertips and hand posture only')
        return self


class RobotUnits(RobotContract):
    position: Literal['m']
    angle: Literal['rad']
    time: Literal['simulation_seconds']


class FingerGuide(RobotContract):
    joint_names: list[Name] = Field(min_length=1, max_length=32)
    tip_link: Name


class CoupledPostureConstraint(RobotPostureProfile):
    notes: Text


class UnconstrainedPostureConstraint(RobotContract):
    profile_id: Literal['unconstrained_v1']
    joint_pairs: list[RobotPostureJointPair] = Field(max_length=0)
    scope: Literal['requested_finger_goals_only']
    notes: Text


class HandGuide(RobotContract):
    frame_id: Name
    wrist_pose_frame: Literal['world']
    fingertip_position_frame: Name
    mount_convention: Text
    tip_semantics: Text
    fingers: dict[FingerName, FingerGuide] = Field(min_length=1, max_length=5)
    posture_constraint: CoupledPostureConstraint | UnconstrainedPostureConstraint

    @model_validator(mode='after')
    def matching_frames(self):
        if self.frame_id != self.fingertip_position_frame:
            raise ValueError('Finger positions must use the advertised hand frame')
        return self


class IKQuerySupport(RobotContract):
    available: StrictBool
    endpoint: Literal['POST /episodes/{episode_id}/robot/ik']
    input_source: Literal['published_joint_proprioception']
    advances_physics: FalseClaim
    mutates_controller: FalseClaim
    collision_planning: FalseClaim
    duration_preflight: FalseClaim


class HandPresetSupport(RobotContract):
    """Read compatibility for guides published before versioned preset catalogs."""
    available: FalseClaim
    reason: Text


class ContactFeedbackSupport(RobotContract):
    tactile_available: StrictBool
    mode: Literal['unavailable', 'simulated_rigid_contact_proxy']
    grasp_detected_available: FalseClaim
    interpretation: Text

    @model_validator(mode='after')
    def matching_mode(self):
        if self.tactile_available != (self.mode != 'unavailable'):
            raise ValueError('Tactile mode and availability must agree')
        return self


class RobotGuide(RobotContract):
    protocol: Literal['bench_robot_guide_v1']
    robot_id: Name
    robot_asset_sha256: Hash
    units: RobotUnits
    quaternion_order: Literal['wxyz']
    camera_frame: Literal['optical_x_right_y_down_z_forward']
    frame_ids: list[Name] = Field(max_length=256)
    arm_joints: list[Name] = Field(min_length=1, max_length=56)
    hand_joints: list[Name] = Field(min_length=1, max_length=224)
    joint_limits: dict[Name, JointLimit]
    hands: dict[Side, HandGuide] = Field(max_length=8)
    control_period_s: Positive
    direct_joint_execution: Literal['step_target_v1', 'min_jerk_from_acknowledged_target_v1',
                                    'min_jerk_velocity_preflight_v2']
    executor: PolicyDescriptor | None
    ik_query: IKQuerySupport
    hand_presets: HandPresetCatalog | HandPresetSupport
    contact_feedback: ContactFeedbackSupport

    @model_validator(mode='after')
    def inventories(self):
        names = self.arm_joints + self.hand_joints
        if len(names) != len(set(names)) or set(names) != set(self.joint_limits):
            raise ValueError('Robot guide joint names must uniquely match limits')
        if self.ik_query.available != bool(self.hands):
            raise ValueError('IK availability must agree with the public hand geometry')
        if isinstance(self.hand_presets, HandPresetCatalog) and not set(self.hand_presets.sides) <= set(self.hands):
            raise ValueError('Preset sides must have advertised public hand geometry')
        for side, hand in self.hands.items():
            if hand.frame_id not in self.frame_ids or hand.frame_id != f'{side}_hand_flange':
                raise ValueError('Public hand frame must match its side and advertised frames')
            for finger in hand.fingers.values():
                if len(finger.joint_names) != len(set(finger.joint_names)) or not set(finger.joint_names) <= set(self.hand_joints):
                    raise ValueError('Finger chain must use unique advertised hand joints')
            for pair in hand.posture_constraint.joint_pairs:
                if not {pair.proximal_joint, pair.distal_joint} <= set(self.hand_joints):
                    raise ValueError('Posture constraint must use advertised hand joints')
        if self.executor and abs(self.executor.control_period_s - self.control_period_s) > 1e-9:
            raise ValueError('Robot and executor control periods must match')
        return self


class IKResiduals(RobotContract):
    fingertip_errors_m: dict[FingerName, Nonnegative]
    wrist_position_error_m: Nonnegative
    wrist_rotation_error_rad: Nonnegative


class IKSuccess(RobotContract):
    solution_found: Literal[True]
    joint_positions_rad: JointMap
    verified_geometry: RobotGeometry
    residuals: IKResiduals

    @model_validator(mode='before')
    @classmethod
    def exact_true(cls, value):
        if isinstance(value, dict) and value.get('solution_found') is not True:
            raise ValueError('Successful IK requires literal true')
        return value


class IKFailure(RobotContract):
    solution_found: FalseClaim
    diagnostic: ActionDiagnostic


class IKResult(RobotContract):
    """Solver result before the runtime attaches its observation timestamp."""
    protocol: Literal['bench_ik_query_v1']
    observation_sequence: int = Field(strict=True, ge=0)
    solution_found: StrictBool
    hands: dict[Side, IKSuccess | IKFailure] = Field(min_length=1, max_length=8)
    advances_physics: FalseClaim
    mutates_controller: FalseClaim
    collision_checked: FalseClaim
    trajectory_checked: FalseClaim
    scope: Text

    @model_validator(mode='after')
    def coherent_solutions(self):
        if self.solution_found != all(hand.solution_found for hand in self.hands.values()):
            raise ValueError('Overall IK result must agree with all selected hands')
        for side, hand in self.hands.items():
            if isinstance(hand, IKSuccess) and (hand.verified_geometry.wrist.frame_id != 'world'
                    or hand.verified_geometry.fingertip_frame_id != f'{side}_hand_flange'):
                raise ValueError('Verified IK geometry must use the selected hand and world frames')
        return self


class IKQueryResult(IKResult):
    """Public HTTP result tied to the sampled observation time."""
    sim_time_s: Nonnegative
