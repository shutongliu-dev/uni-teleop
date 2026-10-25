"""Agent benchmark v1 contracts. No simulator dependencies or state-write actions."""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_serializer, model_validator
from bench.sensors.contact_schema import ContactHistoryValue
from .task_scoring import TerminalTaskScoring, SCORABLE_OUTCOMES

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Positive = Annotated[Finite, Field(gt=0)]
Nonnegative = Annotated[Finite, Field(ge=0)]
Name = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.:/-]+$")]
# Action identifiers share one HTTP path segment with status/cancel routes and
# the public contact-history contract. Other names may still contain slashes.
ActionName = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.:-]{1,160}$")]
HandNamespace = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
HandFrame = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}_hand_flange$")]
Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Vec3 = Annotated[list[Finite], Field(min_length=3, max_length=3)]
Quaternion = Annotated[list[Finite], Field(min_length=4, max_length=4)]
JointMap = Annotated[dict[Name, Finite], Field(min_length=1, max_length=256)]
Track = Literal["direct_joints", "system0", "free_tool_agent"]
ObservationProfile = Literal["vision_proprio", "privileged_state"]
Outcome = Literal["success", "task_failure", "sim_timeout", "wall_timeout", "action_budget", "model_budget", "invalid_action", "policy_error", "safety_stop", "cancelled", "infrastructure_error", "evaluator_invalid"]
# Contact can push a measured joint slightly past its URDF bound. Beyond this
# band the service ends the episode as a physics-invalid safety stop; inside it,
# controllers clamp only their internal references and raw feedback is unchanged.
MEASURED_JOINT_LIMIT_SAFETY_BAND_RAD = .03


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class Budgets(Contract):
    # Explicit null disables that episode-wide horizon. Action durations and
    # action/model-count limits retain their independent meanings.
    sim_time_s: Positive | None = 120.0
    wall_time_s: Positive | None = 1800.0
    max_actions: Annotated[int, Field(strict=True, ge=1)] = 6000
    max_model_calls: Annotated[int, Field(strict=True, ge=1)] = 256
    max_action_sim_s: Positive = 5.0


class RobotPostureJointPair(Contract):
    proximal_joint: Name
    distal_joint: Name

    @model_validator(mode='after')
    def distinct(self):
        if self.proximal_joint == self.distal_joint:
            raise ValueError('posture coupling requires two different joints')
        return self


class RobotPostureProfile(Contract):
    """Static robot-reference coupling; it does not certify a humanlike grasp."""
    profile_id: Name
    max_distal_minus_proximal_rad: Nonnegative
    joint_pairs: list[RobotPostureJointPair] = Field(min_length=1,max_length=32)
    scope: Literal['requested_finger_goals_only'] = 'requested_finger_goals_only'
    measured_state_projected: Literal[False] = False
    unselected_fingers: Literal['preserve'] = 'preserve'
    grasp_success_guarantee: Literal[False] = False

    @model_validator(mode='before')
    @classmethod
    def explicit_false_claims(cls,value):
        if isinstance(value,dict):
            for key in ('measured_state_projected','grasp_success_guarantee'):
                if key in value and value[key] is not False:
                    raise ValueError(key+' must be the literal boolean false')
        return value

    @model_validator(mode='after')
    def unique_pairs(self):
        pairs=[(row.proximal_joint,row.distal_joint) for row in self.joint_pairs]
        if len(set(pairs)) != len(pairs):
            raise ValueError('posture joint pairs must be unique')
        return self


class HandGoalSemantics(Contract):
    """Declared hand-reference behavior, not a grasp or force guarantee."""
    protocol: Literal['accepted_hand_goal_retention_v1'] = 'accepted_hand_goal_retention_v1'
    scope: Literal['move_geometry_unselected_finger_chains'] = 'move_geometry_unselected_finger_chains'
    selected_fingers: Literal['explicit_tip_keys_union_hand_posture_joint_chains'] = 'explicit_tip_keys_union_hand_posture_joint_chains'
    normal_omission: Literal['previous_policy_begin_accepted_goal'] = 'previous_policy_begin_accepted_goal'
    initial_omission: Literal['measured_reference'] = 'measured_reference'
    recovery_omission: Literal['committed_reference_continuous_stop'] = 'committed_reference_continuous_stop'
    selected_partial_posture_omission: Literal['measured_reference'] = 'measured_reference'
    wrist_omission: Literal['current_measured_pose'] = 'current_measured_pose'
    selected_tip_ik: Literal['verified_position_may_change_redundant_joint_solution'] = 'verified_position_may_change_redundant_joint_solution'


ExecutionJointNames = Annotated[list[Name], Field(max_length=256)]
ExecutionJointErrors = Annotated[dict[Name, Nonnegative], Field(max_length=256)]
MAX_ROBOT_EXECUTION_STATE_BYTES = 32768


class RequestedCompletionScope(Contract):
    wrist: Literal['requested_or_measured_maintained_pose']
    fingertips: Literal['explicit_tip_fk_at_advertised_tolerance']
    joint_posture: Literal['explicit_entries_on_chains_without_simultaneous_tip']
    unselected_hand: Literal['maintenance_only_not_joint_completion']
    reference: Literal['all_acknowledged_command_velocities_stationary']


class SimulationReferenceBand(Contract):
    epsilon_rad: Positive
    scope: Literal['simulation_internal_reference_eligibility_not_hardware_safety']
    raw_feedback_projected: Literal[False]
    emitted_targets: Literal['strict_urdf_bounds']

    @model_validator(mode='before')
    @classmethod
    def exact_false(cls, value):
        if isinstance(value, dict) and value.get('raw_feedback_projected') is not False:
            raise ValueError('raw_feedback_projected must be literal false')
        return value


class HandGoalSemanticsV2(Contract):
    # Required, no default: the absence of protocol cannot silently opt into v2.
    protocol: Literal['accepted_hand_goal_retention_v2']
    scope: Literal['move_geometry_and_hold']
    selected_fingers: Literal['explicit_tip_keys_union_hand_posture_joint_chains']
    normal_omission: Literal['previous_accepted_goal_or_persistent_fixed_drive']
    initial_omission: Literal['measured_reference']
    recovery_omission: Literal['committed_continuous_stop_fixed_drive_without_pi']
    selected_partial_posture_omission: Literal['measured_reference']
    wrist_omission: Literal['current_measured_pose']
    selected_tip_ik: Literal['verified_position_may_change_redundant_joint_solution']
    simultaneous_tip_posture: Literal['same_chain_posture_is_ik_reference_not_joint_objective']
    recovery_hold: Literal['all_controlled_joints_fixed_drive']
    geometry_arm: Literal['resume_tracking_requested_or_measured_wrist']
    fixed_hand_exit: Literal['successful_explicit_chain_reselection_only']
    rejected_begin: Literal['all_goal_mode_anchor_and_pending_state_unchanged']
    completion: RequestedCompletionScope
    measured_reference: SimulationReferenceBand
    feedback_protocol: Literal['robot_execution_state_v1']


TerminationReason = Literal[
    'controller_converged', 'time_hold_completed', 'direct_duration_completed',
    'task_terminated', 'begin_rejected', 'controller_failed', 'cancelled',
    'episode_terminated', 'contact_stalled', 'board_turn_changed', 'turn_changed',
]


class RobotExecutionState(Contract):
    """Robot-only, bounded-shape state; no arbitrary policy text or task values.

    Service MUST also validate names against the actual capability inventory,
    scalar error envelopes against robot joint limits, and sample time against
    the latest acknowledged observation. Shape checking alone is insufficient.
    """
    model_config = ConfigDict(revalidate_instances="always")
    protocol: Literal['robot_execution_state_v1']
    sim_time_s: Nonnegative
    completion_scope: Literal['requested_geometry', 'timed_reference_maintenance', 'direct_command_duration']
    objective_status: Literal['reached', 'not_reached', 'not_checked']
    reference_stationary: Annotated[bool, Field(strict=True)] | None
    posture_objective_joint_names: ExecutionJointNames = Field(default_factory=list)
    recovery_fixed_drive_joint_names: ExecutionJointNames = Field(default_factory=list)
    maintained_joint_error_rad: ExecutionJointErrors = Field(default_factory=dict)
    # False when a geometry objective completed at its deadline while the
    # measured wrist was still moving. Absent when not evaluated, so earlier
    # records and frozen clients see an unchanged payload.
    wrist_settled: Annotated[bool, Field(strict=True)] | None = None

    @model_serializer(mode='wrap')
    def omit_unevaluated_settle(self, handler):
        value = handler(self)
        if isinstance(value, dict) and value.get('wrist_settled') is None:
            value.pop('wrist_settled', None)
        return value

    @model_validator(mode='after')
    def unique_names(self):
        for values in (self.posture_objective_joint_names, self.recovery_fixed_drive_joint_names):
            if len(values) != len(set(values)):
                raise ValueError('Joint names must be unique')
        return self



class GeometryTolerancePolicy(Contract):
    """Existing geometry acceptance policy; not a force/grasp guarantee."""
    protocol: Literal['geometry_tolerance_policy_v1', 'geometry_tolerance_policy_v2']
    wrist_position_completion_cap_m: Positive
    wrist_orientation_completion_cap_rad: Positive
    fingertip_completion_m: Positive
    joint_posture_completion_rad: Positive
    ik_verification_scale: Annotated[Positive, Field(le=1)]
    wrist_rule: Literal['minimum_of_request_and_controller_cap',
                        'minimum_of_request_cap_and_displacement_relative_default']
    # v2: a fingertip target whose closest reachable point lies within this
    # distance executes that point and reports ik_target_at_reach_boundary.
    fingertip_ik_acceptance_m: Positive | None = None
    wrist_settle_speed_m_s: Positive | None = None
    stall_window_s: Positive | None = None


class EffectiveGeometryTolerances(Contract):
    """Only objectives checked by the diagnostic's actual execution stage."""
    stage: Literal['ik_verification', 'completion']
    wrist_position_m: Positive | None = None
    wrist_orientation_rad: Positive | None = None
    fingertips_m: dict[Literal['thumb', 'index', 'middle', 'ring', 'little'], Positive] = Field(default_factory=dict, max_length=5)
    joint_posture_rad: Positive | None = None

    @model_validator(mode='after')
    def checked_objectives(self):
        if (self.wrist_position_m is None) != (self.wrist_orientation_rad is None):
            raise ValueError('Wrist position and orientation thresholds must be paired')
        if self.stage == 'ik_verification' and self.joint_posture_rad is not None:
            raise ValueError('Posture joint thresholds apply only to completion')
        if self.wrist_position_m is None and not self.fingertips_m and self.joint_posture_rad is None:
            raise ValueError('Effective geometry tolerances need a checked objective')
        return self


class DirectControlHandoff(Contract):
    protocol: Literal['acknowledged_direct_handoff_v1'] = 'acknowledged_direct_handoff_v1'
    direct_execution: Literal['native_joint_target_bypass'] = 'native_joint_target_bypass'
    direct_qva_smoothing: Literal[False] = False
    reentry_reference: Literal['last_acknowledged_drive_and_discrete_velocity'] = 'last_acknowledged_drive_and_discrete_velocity'
    unknown_velocity: Literal['reject_classical_until_two_consecutive_command_acks'] = 'reject_classical_until_two_consecutive_command_acks'
    stale_goals: Literal['replace_with_direct_drive_fixed_reference'] = 'replace_with_direct_drive_fixed_reference'
    integral: Literal['freeze_until_continuous_stop_then_clear'] = 'freeze_until_continuous_stop_then_clear'
    reentry_limits: Literal['original_classical_qva_and_braking_viability'] = 'original_classical_qva_and_braking_viability'

    @model_validator(mode='before')
    @classmethod
    def strict_false(cls, value):
        if isinstance(value, dict) and 'direct_qva_smoothing' in value and value['direct_qva_smoothing'] is not False:
            raise ValueError('direct_qva_smoothing must be literal false')
        return value


class PolicyDescriptor(Contract):
    policy_id: Name
    version: Name
    implementation_sha256: Hash
    weights_sha256: Hash | None = None
    trained: bool
    observation_profile: ObservationProfile = "vision_proprio"
    skills: list[Name] = Field(default_factory=list, max_length=256)
    control_period_s: Positive = 0.02
    geometry_inputs: list[Literal["wrist_pose", "contacts", "wrist_trajectory", "hand_posture", "fingertip_positions"]] = Field(default_factory=list)
    force_objectives: bool = False
    camera_observation_mode: Literal['every_control_step', 'decision_only'] = 'every_control_step'
    supported_robot_ids: list[Name] = Field(default_factory=list)
    robot_posture_profile: RobotPostureProfile | None = None
    hand_goal_semantics: HandGoalSemantics | HandGoalSemanticsV2 | None = None
    geometry_tolerance_policy: GeometryTolerancePolicy | None = None
    direct_control_handoff: DirectControlHandoff | None = None
    hold_modes: list[Literal['reference', 'acknowledged_drive']] = Field(default_factory=lambda: ['reference'])

    @model_serializer(mode='wrap')
    def omit_legacy_hold_modes(self, handler):
        value = handler(self)
        if value.get('hold_modes') == ['reference']:
            value.pop('hold_modes')
        return value

    @model_validator(mode="after")
    def trained_weights(self):
        if self.trained and self.weights_sha256 is None:
            raise ValueError("a trained policy requires a frozen weights/checkpoint digest")
        if len(self.skills) != len(set(self.skills)):
            raise ValueError("advertised skill names must be unique")
        if (not self.hold_modes or len(self.hold_modes) != len(set(self.hold_modes))
                or ('acknowledged_drive' in self.hold_modes and 'hold' not in self.skills)):
            raise ValueError('Hold modes must be unique and fixed-drive maintenance requires hold')
        if self.hand_goal_semantics is not None and ('move_geometry' not in self.skills or not self.geometry_inputs):
            raise ValueError('hand goal semantics require move_geometry and declared geometry inputs')
        if self.geometry_tolerance_policy is not None and ('move_geometry' not in self.skills or not self.geometry_inputs):
            raise ValueError('Geometry tolerances require move_geometry and declared geometry inputs')
        if self.direct_control_handoff is not None and (self.policy_id != 'classical_fingertip'
                or not {'move_geometry', 'hold'} <= set(self.skills) or self.trained):
            raise ValueError('Direct handoff requires the pinned classical geometry executor')
        return self


class EpisodeSpec(Contract):
    protocol: Literal["agent_bench_v1"] = "agent_bench_v1"
    episode_id: Name
    agent_id: Name
    agent_version: Name
    agent_config_sha256: Hash
    task_id: Name
    task_release_sha256: Hash
    evaluator_sha256: Hash
    robot_id: Name
    robot_asset_sha256: Hash
    physics_profile_sha256: Hash
    scene_variant: Name = 'original_v1'
    scene_sha256: Hash | None = None
    runtime_code_sha256: Hash | None = None
    sensor_config_sha256: Hash | None = None
    execution_environment_sha256: Hash | None = None
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
    track: Track
    observation_profile: ObservationProfile = "vision_proprio"
    execution_mode: Literal["pause_physics_during_agent", "continuous_physics"] = "pause_physics_during_agent"
    instruction: str = Field(min_length=1, max_length=4096)
    budgets: Budgets = Field(default_factory=Budgets)
    system0: PolicyDescriptor | None = None
    split: Literal["development", "held_out"] = "development"

    @model_validator(mode="after")
    def policy_track(self):
        if (self.track in ('system0', 'free_tool_agent')) != (self.system0 is not None):
            raise ValueError('Policy tracks require a pinned policy; direct track forbids it')
        if self.track == 'free_tool_agent' and (self.system0.policy_id != 'classical_fingertip'
                or self.system0.direct_control_handoff is None):
            raise ValueError('Free tool track requires the classical acknowledged handoff contract')
        return self


class EvaluationPlan(Contract):
    """Organizer-pinned task/seed/policy matrix; never chosen after seeing scores."""
    protocol: Literal["agent_bench_plan_v1"] = "agent_bench_plan_v1"
    plan_id: Name
    description: str = Field(default="", max_length=4096)
    episodes: list[EpisodeSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [x.episode_id for x in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("planned episode IDs must be unique")
        return self


class JointLimit(Contract):
    lower_rad: Finite
    upper_rad: Finite
    max_velocity_rad_s: Positive | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.lower_rad >= self.upper_rad:
            raise ValueError("joint limits must be ordered")
        return self


class RobotCapabilities(Contract):
    protocol: Literal["agent_bench_v1"] = "agent_bench_v1"
    robot_id: Name
    robot_asset_sha256: Hash
    arm_joints: list[Name] = Field(min_length=1, max_length=56)
    hand_joints: list[Name] = Field(min_length=1, max_length=224)
    finger_names: list[Literal["thumb", "index", "middle", "ring", "little"]] = Field(default_factory=list)
    joint_limits: dict[Name, JointLimit]
    control_period_s: Positive = 0.02
    tracks: list[Track] = Field(default_factory=lambda: ["direct_joints"])
    observation_profiles: list[ObservationProfile] = Field(default_factory=lambda: ["vision_proprio"])
    frame_ids: list[Name] = Field(default_factory=lambda: ["world", "robot_base"])
    camera_ids: list[Name] = Field(default_factory=list)
    depth_available: bool = False
    system0: PolicyDescriptor | None = None
    object_state_writeback: Literal[False] = False
    cloth_state_writeback: Literal[False] = False
    tactile_available: bool = False
    tactile_mode: Literal['unavailable', 'simulated_rigid_contact_proxy'] = 'unavailable'
    tactile_cloth_available: Literal[False] = False
    # How a direct joint action's target is executed over its duration.
    direct_joint_execution: Literal['step_target_v1', 'min_jerk_from_acknowledged_target_v1',
                                    'min_jerk_velocity_preflight_v2'] = 'step_target_v1'
    # Static robot facts: segment lengths and uncalibrated drive gains.
    hand_geometry: dict[Name, Positive] | None = None

    @model_validator(mode="after")
    def inventory(self):
        names = self.arm_joints + self.hand_joints
        if len(names) != len(set(names)) or set(names) != set(self.joint_limits):
            raise ValueError("arm/hand names must be unique and exactly match limits")
        if len(self.finger_names) != len(set(self.finger_names)):
            raise ValueError("finger names must be unique")
        if ("system0" in self.tracks) != (self.system0 is not None):
            raise ValueError("only advertise System0 when a pinned policy is installed")
        if 'free_tool_agent' in self.tracks and (self.system0 is None
                or self.system0.policy_id != 'classical_fingertip'
                or self.system0.direct_control_handoff is None):
            raise ValueError('Free tool track requires an installed classical handoff policy')
        if len(self.tracks) != len(set(self.tracks)):
            raise ValueError('Control tracks must be unique')
        if self.tactile_available != (self.tactile_mode != 'unavailable'):
            raise ValueError('tactile mode and availability must agree')
        if self.system0 and abs(self.system0.control_period_s - self.control_period_s) > 1e-9:
            raise ValueError("System0 and robot control periods must match")
        if self.system0 and self.system0.robot_posture_profile:
            for pair in self.system0.robot_posture_profile.joint_pairs:
                if not {pair.proximal_joint,pair.distal_joint} <= set(self.hand_joints):
                    raise ValueError('posture coupling must reference advertised hand joints')
        return self


class Pose(Contract):
    frame_id: Name
    position_m: Vec3
    quaternion_wxyz: Quaternion

    @model_validator(mode="after")
    def rotation(self):
        if abs(sum(x * x for x in self.quaternion_wxyz) - 1) > 1e-5:
            raise ValueError("quaternion must be unit length in wxyz order")
        return self


class TargetReference(Contract):
    """Agent-generated semantic name/part; never an oracle pose lookup request."""
    object: Name
    part: Name | None = None


class ContactObjective(Contract):
    finger: Literal["thumb", "index", "middle", "ring", "little"]
    role: Literal["opposition", "primary", "support", "avoid"]
    contact: Literal["required", "preferred", "forbidden"]
    frame_id: Name
    position_m: Vec3
    normal_xyz: Vec3 | None = None
    position_tolerance_m: Positive = 0.01
    desired_normal_force_n: Nonnegative | None = None

    @model_validator(mode="after")
    def normal(self):
        if self.normal_xyz is not None and abs(sum(x*x for x in self.normal_xyz) - 1) > 1e-5:
            raise ValueError("contact normal must be unit length")
        if self.contact == "forbidden" and self.desired_normal_force_n not in (None, 0):
            raise ValueError("forbidden contact cannot request positive force")
        return self


class Waypoint(Contract):
    time_s: Nonnegative
    wrist: Pose


class GeometryObjective(Contract):
    wrist: Pose | None = None
    contacts: list[ContactObjective] = Field(default_factory=list, max_length=64)
    wrist_trajectory: list[Waypoint] = Field(default_factory=list, max_length=256)
    hand_posture_rad: JointMap | None = None
    fingertip_positions_m: dict[Literal['thumb', 'index', 'middle', 'ring', 'little'], Vec3] = Field(default_factory=dict)
    # A bimanual robot addresses its left hand in left_hand_flange.
    fingertip_frame_id: HandFrame = 'right_hand_flange'
    position_tolerance_m: Positive = 0.01
    orientation_tolerance_rad: Positive = 0.1

    @model_validator(mode="after")
    def populated(self):
        if not (self.wrist or self.contacts or self.wrist_trajectory or self.hand_posture_rad or self.fingertip_positions_m):
            raise ValueError("dense objective requires pose, contacts, trajectory or hand posture")
        times = [x.time_s for x in self.wrist_trajectory]
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("trajectory time must strictly increase")
        return self


class DirectJointAction(Contract):
    kind: Literal["direct_joints"] = "direct_joints"
    action_id: ActionName
    observation_sequence: Annotated[int, Field(strict=True, ge=0)]
    duration_s: Positive
    arm_joint_positions_rad: JointMap
    hand_joint_positions_rad: JointMap


class System0Action(Contract):
    kind: Literal["system0"] = "system0"
    action_id: ActionName
    observation_sequence: Annotated[int, Field(strict=True, ge=0)]
    skill: Name
    target: TargetReference | None = None
    geometry: GeometryObjective | None = None
    object_effect: Literal["hold", "move", "release", "deform", "none"] = "none"
    timeout_sim_s: Positive
    hold_duration_s: Positive | None = None
    hold_mode: Literal['reference', 'acknowledged_drive'] = 'reference'
    # Bimanual robots: one objective per moved hand, in that hand's flange frame;
    # an omitted hand keeps its reference. Exclusive with ``geometry``.
    hand_geometries: dict[HandNamespace, GeometryObjective] | None = None

    @model_serializer(mode='wrap')
    def omit_single_arm_hands(self, handler):
        value = handler(self)
        if isinstance(value, dict) and value.get('hand_geometries') is None:
            value.pop('hand_geometries', None)
        if isinstance(value, dict) and value.get('hold_mode') == 'reference':
            value.pop('hold_mode', None)
        return value

    @model_validator(mode="after")
    def timed_hold(self):
        if self.skill == 'hold':
            if self.hold_duration_s is None or self.timeout_sim_s < self.hold_duration_s:
                raise ValueError('hold requires a positive duration within its timeout')
            if (self.geometry is not None or self.hand_geometries is not None or self.target is not None
                    or self.object_effect != 'none'):
                raise ValueError('hold retains the robot reference; geometry, target and object effects are forbidden')
        elif self.hold_duration_s is not None or self.hold_mode != 'reference':
            raise ValueError('hold_duration_s and hold_mode are only valid for the hold skill')
        if self.hand_geometries is not None:
            if self.geometry is not None or not self.hand_geometries:
                raise ValueError('hand_geometries replaces geometry and needs at least one hand')
            if any(objective.fingertip_frame_id != f'{side}_hand_flange' for side, objective in self.hand_geometries.items()):
                raise ValueError('each hand objective uses its own hand flange frame')
        return self


AgentAction = Annotated[DirectJointAction | System0Action, Field(discriminator="kind")]
ACTION_ADAPTER = TypeAdapter(AgentAction)


class RobotJointCommand(Contract):
    """The sole plugin→physics command; no object mutation or arbitrary code."""
    joint_positions_rad: JointMap
    duration_s: Positive


class CameraObservation(Contract):
    camera_id: Name
    sim_time_s: Nonnegative
    rgb_uri: str = Field(min_length=1, max_length=2048)
    rgb_sha256: Hash
    width: Annotated[int, Field(strict=True, gt=0)]
    height: Annotated[int, Field(strict=True, gt=0)]
    depth_uri: str | None = Field(default=None, min_length=1, max_length=2048)
    depth_sha256: Hash | None = None
    intrinsics_3x3: Annotated[list[Finite], Field(min_length=9, max_length=9)] | None = None
    world_from_camera: Pose | None = None
    depth_unit: Literal["m"] = "m"
    depth_encoding: Literal["float32_npy"] | None = None
    depth_semantics: Literal["camera_z"] | None = None
    depth_invalid_value: Finite | None = None
    depth_sim_time_s: Nonnegative | None = None
    depth_rgb_aligned: Literal[True] | None = None

    @model_validator(mode="after")
    def depth_pair(self):
        if (self.depth_uri is None) != (self.depth_sha256 is None):
            raise ValueError("depth requires URI and digest together")
        details = (self.depth_encoding, self.depth_semantics, self.depth_invalid_value,
                   self.depth_sim_time_s, self.depth_rgb_aligned)
        if self.depth_uri is None:
            if any(value is not None for value in details):
                raise ValueError("depth metadata requires a depth image")
        else:
            if any(value is None for value in details):
                raise ValueError("depth requires encoding, semantics, invalid value, timestamp and RGB alignment")
            if self.depth_invalid_value != 0.0:
                raise ValueError("depth invalid pixels must be encoded as zero metres")
            if abs(self.depth_sim_time_s - self.sim_time_s) > 1e-9:
                raise ValueError("RGB and depth timestamps must match")
            if self.intrinsics_3x3 is None or self.world_from_camera is None:
                raise ValueError("aligned depth requires shared camera intrinsics and extrinsics")
        return self


FingerName = Literal['thumb', 'index', 'middle', 'ring', 'little']


class FingerJointCenters(Contract):
    """URDF joint origins along one finger, proximal to distal, in the flange frame."""
    base_m: Vec3            # MCP flexion (thumb: CMC flexion) joint origin
    proximal_m: Vec3        # MCP abduction/PP origin (thumb: MCP abduction)
    middle_m: Vec3 | None = None  # PIP joint origin; the thumb has no middle phalanx
    distal_m: Vec3          # DIP (thumb: IP) joint origin


class RobotGeometry(Contract):
    """Robot-only FK of measured joints; no privileged object state."""
    wrist: Pose
    fingertip_frame_id: HandFrame = 'right_hand_flange'
    fingertip_positions_m: dict[FingerName, Vec3]
    # robot_geometry_v2: tip-frame orientation (axis z points distally along the
    # distal phalanx, +x toward the palmar pad) and URDF joint origins per finger.
    fingertip_quaternions_wxyz: dict[FingerName, Quaternion] | None = None
    finger_joint_centers_m: dict[FingerName, FingerJointCenters] | None = None


class Proprioception(Contract):
    joint_positions_rad: JointMap
    joint_velocities_rad_s: JointMap
    # Last acknowledged position-drive targets (a real arm's commanded q_d).
    # They differ from measured positions under gravity, load or contact.
    commanded_joint_positions_rad: JointMap | None = None
    geometry: RobotGeometry | None = None
    # Bimanual robots publish one FK per hand instead of ``geometry``.
    hand_geometries: dict[HandNamespace, RobotGeometry] | None = None

    @model_serializer(mode='wrap')
    def omit_single_arm_hands(self, handler):
        value = handler(self)
        if isinstance(value, dict) and value.get('hand_geometries') is None:
            value.pop('hand_geometries', None)
        return value

    @model_validator(mode="after")
    def matching_joints(self):
        if set(self.joint_positions_rad) != set(self.joint_velocities_rad_s):
            raise ValueError("proprioception position and velocity joint inventories must match")
        if (self.commanded_joint_positions_rad is not None
                and set(self.commanded_joint_positions_rad) != set(self.joint_positions_rad)):
            raise ValueError("commanded and measured joint inventories must match")
        return self


class FingerTactile(Contract):
    status: Literal['valid', 'unavailable', 'not_sampled', 'invalid']
    rigid_contact_any: Annotated[bool, Field(strict=True)] | None = None
    normal_impulse_sum_ns: Nonnegative | None = None
    mean_normal_load_n: Nonnegative | None = None

    @model_validator(mode='after')
    def validity(self):
        values = (self.rigid_contact_any, self.normal_impulse_sum_ns, self.mean_normal_load_n)
        if self.status == 'valid' and any(v is None for v in values):
            raise ValueError('valid tactile readings require all measurements')
        if self.status != 'valid' and any(v is not None for v in values):
            raise ValueError('missing or invalid tactile is not a zero-force reading')
        if self.status == 'valid':
            if self.rigid_contact_any != (self.normal_impulse_sum_ns > 0.):
                raise ValueError('tactile contact flag must match positive solved normal impulse')
            if (self.normal_impulse_sum_ns == 0.) != (self.mean_normal_load_n == 0.):
                raise ValueError('zero tactile impulse and zero mean load must agree')
        return self


class TactileCoverage(Contract):
    rigid: Annotated[bool, Field(strict=True)]
    cloth: Literal[False] = False
    self_contacts: Literal[True] = True
    # SHARPA: elastomer pad bodies; other hands: their whole distal rigid links.
    region: Literal['whole_elastomer_collision_body', 'whole_distal_link_collision_body'] = 'whole_elastomer_collision_body'
    taxels: Literal[False] = False
    shear: Literal[False] = False
    torque: Literal[False] = False

    @model_validator(mode='before')
    @classmethod
    def literal_coverage_flags(cls, value):
        if isinstance(value, dict):
            for key, expected in (('cloth', False), ('self_contacts', True),
                                  ('taxels', False), ('shear', False), ('torque', False)):
                if key in value and value[key] is not expected:
                    raise ValueError('tactile coverage flags must be literal booleans')
        return value


class TactileObservation(Contract):
    protocol: Literal['rigid_fingertip_contact_proxy_v1'] = 'rigid_fingertip_contact_proxy_v1'
    mode: Literal['simulated_rigid_contact_proxy'] = 'simulated_rigid_contact_proxy'
    status: Literal['available', 'unavailable', 'not_sampled', 'invalid']
    sim_time_s: Nonnegative
    window_start_sim_s: Nonnegative | None = None
    window_duration_s: Positive | None = None
    window_completeness: Literal['assumed_from_native_step_contract', 'not_available']
    force_calibrated: Literal[False] = False
    frame_id: Literal['scalar_invariant'] = 'scalar_invariant'
    coverage: TactileCoverage
    fingers: dict[Literal['thumb', 'index', 'middle', 'ring', 'little'], FingerTactile]

    @model_validator(mode='before')
    @classmethod
    def literal_calibration_flag(cls, value):
        if isinstance(value, dict) and 'force_calibrated' in value and value['force_calibrated'] is not False:
            raise ValueError('tactile force_calibrated must be the literal boolean false')
        return value

    @model_validator(mode='after')
    def consistent_projection(self):
        five = {'thumb', 'index', 'middle', 'ring', 'little'}
        # SHARPA elastomer pads are five channels; a four-finger hand's distal
        # links omit little. No other subset is a complete projection.
        if set(self.fingers) != five and not (self.coverage.region == 'whole_distal_link_collision_body'
                                              and set(self.fingers) == five - {'little'}):
            raise ValueError('tactile requires exactly all five finger channels (four-finger hands: all four)')
        expected_status = 'valid' if self.status == 'available' else self.status
        if any(finger.status != expected_status for finger in self.fingers.values()):
            raise ValueError('tactile finger statuses must match the observation status')
        # Coverage describes configured native support, even at reset or when
        # a supported stream is invalid. Neither state is a measured zero.
        if self.coverage.rigid != (self.status != 'unavailable'):
            raise ValueError('tactile status and configured rigid coverage disagree')
        if self.status != 'available':
            if (self.window_start_sim_s is not None or self.window_duration_s is not None
                    or self.window_completeness != 'not_available'):
                raise ValueError('unmeasured tactile must not claim a sampled window')
            if self.status == 'not_sampled' and self.sim_time_s != 0.:
                raise ValueError('not_sampled tactile is the reset observation only')
            return self
        if (self.window_start_sim_s is None or self.window_duration_s is None
                or self.window_completeness != 'assumed_from_native_step_contract'):
            raise ValueError('available tactile requires a complete declared window')
        # Match the projector's absolute native-clock tolerance, including its
        # near-reset max(0, time-period); never scale tolerance with episode age.
        if (self.sim_time_s <= 0. or self.window_start_sim_s > self.sim_time_s
                or not math.isclose(self.window_start_sim_s + self.window_duration_s,
                                    self.sim_time_s, rel_tol=0., abs_tol=1e-8)):
            raise ValueError('tactile window must end at its observation time')
        for finger in self.fingers.values():
            expected_load = finger.normal_impulse_sum_ns / self.window_duration_s
            if not math.isfinite(expected_load) or not math.isclose(
                    finger.mean_normal_load_n, expected_load, rel_tol=1e-9, abs_tol=0.):
                raise ValueError('tactile mean load must equal normal impulse divided by window duration')
        return self


class AgentObservation(Contract):
    protocol: Literal["agent_bench_v1"] = "agent_bench_v1"
    episode_id: Name
    sequence: Annotated[int, Field(strict=True, ge=0)]
    sim_time_s: Nonnegative
    instruction: str = Field(min_length=1, max_length=4096)
    profile: ObservationProfile = "vision_proprio"
    proprioception: Proprioception
    cameras: list[CameraObservation] = Field(default_factory=list)
    tactile: TactileObservation | None = None
    # Bimanual robots: one tactile projection per hand instead of ``tactile``.
    hand_tactile: dict[HandNamespace, TactileObservation] | None = None
    last_action_id: ActionName | None = None
    last_action_status: Literal["idle", "accepted", "running", "completed", "cancelled", "failed"] = "idle"
    # Explicit typed privileged fields, separate leaderboard. Evaluator internals
    # (rewards, goal poses, phases, success checks) are absent even in this profile.
    privileged_object_poses: dict[Name, Pose] | None = None
    privileged_cloth_vertices_m: list[Vec3] | None = None

    @model_validator(mode="after")
    def visibility(self):
        if self.profile == "vision_proprio" and (self.privileged_object_poses is not None or self.privileged_cloth_vertices_m is not None):
            raise ValueError("privileged state is forbidden in vision_proprio")
        if len({camera.camera_id for camera in self.cameras}) != len(self.cameras):
            raise ValueError("camera IDs must be unique within an observation")
        if any(abs(camera.sim_time_s - self.sim_time_s) > 1e-9 for camera in self.cameras):
            raise ValueError("camera and observation timestamps must match")
        if self.tactile is not None and abs(self.tactile.sim_time_s - self.sim_time_s) > 1e-9:
            raise ValueError('tactile and observation timestamps must match')
        if self.hand_tactile is not None and (self.tactile is not None or any(
                abs(t.sim_time_s - self.sim_time_s) > 1e-9 for t in self.hand_tactile.values())):
            raise ValueError('per-hand tactile replaces tactile and must match the observation time')
        return self

    @model_serializer(mode='wrap')
    def omit_single_arm_hand_tactile(self, handler):
        value = handler(self)
        if isinstance(value, dict) and value.get('hand_tactile') is None:
            value.pop('hand_tactile', None)
        return value


class DurationPreflight(Contract):
    """Robot-reference timing eligibility, not a convergence guarantee."""
    protocol: Literal['duration_preflight_v1'] = 'duration_preflight_v1'
    skill: Literal['move_geometry', 'hold', 'direct_joints']
    requested_duration_s: Positive
    minimum_duration_s: Positive
    control_period_s: Positive
    minimum_steps: Annotated[int, Field(strict=True, ge=1)]
    braking_steps: Annotated[int, Field(strict=True, ge=0)]
    trajectory_steps: Annotated[int, Field(strict=True, ge=0)]
    settling_steps: Annotated[int, Field(strict=True, ge=0)]
    semantics: Literal['reference_duration_lower_bound_not_convergence_guarantee'] = 'reference_duration_lower_bound_not_convergence_guarantee'

    @model_validator(mode='after')
    def timing_identity(self):
        if self.minimum_steps != max(1, self.braking_steps + self.trajectory_steps + self.settling_steps):
            raise ValueError('Timing steps do not match the preflight minimum')
        try:
            expected = self.minimum_steps * self.control_period_s
        except OverflowError as error:
            raise ValueError('Timing steps overflow the finite duration domain') from error
        if not math.isfinite(expected) or self.minimum_duration_s != expected:
            raise ValueError('Minimum duration must equal minimum steps times control period')
        if self.skill == 'hold' and (self.trajectory_steps or self.settling_steps):
            raise ValueError('Hold uses braking only, without a geometric trajectory or settling margin')
        if self.skill == 'move_geometry' and self.trajectory_steps < 1:
            raise ValueError('Geometry motion requires at least one trajectory step')
        ratio = self.requested_duration_s / self.control_period_s
        if not math.isfinite(ratio):
            raise ValueError('Requested duration/control period ratio must be finite')
        steps = round(ratio)
        try:
            aligned = abs(steps * self.control_period_s - self.requested_duration_s) <= 1e-8
        except OverflowError as error:
            raise ValueError('Requested steps overflow the finite duration domain') from error
        if aligned and steps >= self.minimum_steps:
            raise ValueError('A valid grid-aligned duration is not an insufficient-duration rejection')
        return self


class ActionDiagnostic(Contract):
    """Allowlisted robot-execution information; arbitrary policy text stays private."""
    code: Literal['ik_solution_unverified', 'trajectory_limit_guard', 'convergence_timeout',
                  'invalid_robot_state', 'insufficient_duration', 'posture_constraint_violation', 'cancelled',
                  'control_handoff_unverified', 'measured_limit_recovery',
                  'ik_target_at_reach_boundary', 'tracking_stalled', 'measured_limit_projection']
    joint_name: Name | None = None
    joint_error_rad: Nonnegative | None = None
    wrist_position_error_m: Nonnegative | None = None
    wrist_orientation_error_rad: Nonnegative | None = None
    fingertip_errors_m: dict[Literal['thumb', 'index', 'middle', 'ring', 'little'], Nonnegative] = Field(default_factory=dict)
    # Joints at a URDF bound in the closest IK solution: the reachable-set edge.
    limiting_joint_names: list[Name] = Field(default_factory=list, max_length=32)
    # Closest tip position the bounded IK found (flange frame); not proof of global reach.
    best_found_fingertips_m: dict[Literal['thumb', 'index', 'middle', 'ring', 'little'], Vec3] = Field(default_factory=dict)
    elapsed_sim_s: Nonnegative | None = None
    geometry_tolerances: EffectiveGeometryTolerances | None = None
    duration_preflight: DurationPreflight | None = None

    @model_validator(mode='after')
    def matching_geometry_stage(self):
        if self.duration_preflight is not None and self.code != 'insufficient_duration':
            raise ValueError('Duration preflight requires an insufficient_duration diagnostic')
        value = self.geometry_tolerances
        if value is None:
            return self  # Old saved payloads remain readable without invented thresholds.
        expected = {'ik_solution_unverified': 'ik_verification', 'ik_target_at_reach_boundary': 'ik_verification',
                    'convergence_timeout': 'completion', 'tracking_stalled': 'completion'}
        if expected.get(self.code) != value.stage:
            raise ValueError('Geometry tolerance stage does not match diagnostic code')
        if ((self.wrist_position_error_m is None) != (value.wrist_position_m is None)
                or (self.wrist_orientation_error_rad is None) != (value.wrist_orientation_rad is None)
                or set(self.fingertip_errors_m) != set(value.fingertips_m)
                or (self.joint_name is None) != (value.joint_posture_rad is None)
                or (self.joint_error_rad is None) != (value.joint_posture_rad is None)):
            raise ValueError('Residual and tolerance objectives must match exactly')
        return self


class ActionEvent(Contract):
    action_id: ActionName
    status: Literal["accepted", "running", "completed", "cancelled", "failed"]
    sim_time_s: Nonnegative
    wall_time_s: Nonnegative
    detail: str = Field(default="", max_length=1024)
    diagnostic: ActionDiagnostic | None = None
    termination_reason: TerminationReason | None = None
    robot_execution_state: RobotExecutionState | None = None
    contact_history: ContactHistoryValue | None = None


class EpisodeResult(Contract):
    spec: EpisodeSpec
    outcome: Outcome
    elapsed_sim_s: Nonnegative
    elapsed_wall_s: Nonnegative
    model_latency_s: Nonnegative = 0.0
    model_calls: Annotated[int, Field(strict=True, ge=0)] = 0
    model_accounting: Literal["unavailable", "server_observed", "client_reported"] = "unavailable"
    action_attempts: Annotated[int, Field(strict=True, ge=0)] = 0
    completion_sim_s: Nonnegative | None = None
    completion_wall_s: Nonnegative | None = None
    physics_valid: bool = True
    evaluator_evidence_sha256: Hash | None = None
    failure_detail: str = Field(default="", max_length=2048)
    scoring: TerminalTaskScoring | None = None

    @model_serializer(mode="wrap")
    def omit_legacy_scoring(self, handler):
        result = handler(self)
        if self.scoring is None:
            result.pop("scoring", None)
        return result

    @model_validator(mode="after")
    def completion(self):
        if self.scoring is not None and self.scoring.score_valid and (
                not self.physics_valid or self.evaluator_evidence_sha256 is None
                or self.outcome not in SCORABLE_OUTCOMES):
            raise ValueError("valid terminal scoring requires valid physics and evaluator evidence")
        if self.model_latency_s > self.elapsed_wall_s + 1e-6:
            raise ValueError("model latency is a subset of wall time")
        if self.outcome == "success":
            if not self.physics_valid or self.evaluator_evidence_sha256 is None:
                raise ValueError("success requires valid physics and evaluator evidence")
            if self.completion_sim_s is None or self.completion_wall_s is None:
                raise ValueError("success requires first verified completion times")
            if self.completion_sim_s > self.elapsed_sim_s + 1e-6 or self.completion_wall_s > self.elapsed_wall_s + 1e-6:
                raise ValueError("completion cannot follow elapsed time")
            if ((self.spec.budgets.sim_time_s is not None and self.completion_sim_s > self.spec.budgets.sim_time_s)
                    or (self.spec.budgets.wall_time_s is not None and self.completion_wall_s > self.spec.budgets.wall_time_s)):
                raise ValueError("success cannot exceed episode budget")
            if self.action_attempts > self.spec.budgets.max_actions or self.model_calls > self.spec.budgets.max_model_calls:
                raise ValueError("success cannot exceed action/model call budgets")
        elif self.completion_sim_s is not None or self.completion_wall_s is not None:
            raise ValueError("unsuccessful episodes do not have completion times")
        return self


def parse_action(payload: dict) -> DirectJointAction | System0Action:
    return ACTION_ADAPTER.validate_python(payload)


def schemas() -> dict[str, dict]:
    # Import after the base contracts are defined; functions compose these
    # contracts and expose the same schema to Python hosts and HTTP clients.
    from .functions import function_schemas
    return {**function_schemas(), "action": ACTION_ADAPTER.json_schema(), **{cls.__name__: cls.model_json_schema() for cls in
            (EpisodeSpec, EvaluationPlan, RobotCapabilities, AgentObservation, RobotJointCommand, ActionDiagnostic, ActionEvent, EpisodeResult)}}
