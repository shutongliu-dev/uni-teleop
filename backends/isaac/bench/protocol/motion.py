"""Read-only controller planning and explicitly limited robot geometry advice."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, model_validator

from .robot import FalseClaim, IKQuery, RobotContract, Text
from .schemas import ActionDiagnostic, Hash, JointMap, Name, Nonnegative, Positive


class MotionQuery(IKQuery):
    """Preview the current executor without stepping it or spending an action.

    Self-collision advice samples only original FR3 link0..7 nonadjacent convex
    shapes along the nominal joint path. It excludes hands, other arms, objects,
    environment, braking/feedback deviations, native contact offsets and motion
    between samples. No detected intersection is NOT a safety guarantee.
    """
    requested_duration_s: Positive | None = Field(default=None, description=
        'Optional proposed timeout, compared with the original controller minimum and episode cap; never executes or extends an action.')


class ArmIntersection(RobotContract):
    links: list[Name] = Field(min_length=2, max_length=2)
    maximum_intersection_radius_m: Nonnegative
    first_sample_fraction: float = Field(ge=0., le=1.)
    at_goal: StrictBool


class MotionCollision(RobotContract):
    status: Literal['sampled', 'unsupported']
    reason: Literal['source_import_unavailable', 'source_import_unverified', 'dependency_unavailable'] | None = None
    detected: StrictBool | None = None
    goal_detected: StrictBool | None = None
    sample_count: int = Field(default=0, ge=0, le=65)
    colliding_sample_count: int = Field(default=0, ge=0, le=65)
    maximum_sample_joint_step_rad: Nonnegative | None = None
    intersection_radius_threshold_m: Positive = .001
    pairs: list[ArmIntersection] = Field(default_factory=list, max_length=21)
    robot_source_sha256: Hash | None = None
    scope: Literal['sampled_nominal_joint_path_nonadjacent_fr3_link0_through_link7'] = 'sampled_nominal_joint_path_nonadjacent_fr3_link0_through_link7'
    exclusions: list[str] = Field(default_factory=lambda: [
        'hand_and_hand_vs_arm', 'other_arms', 'objects_and_environment',
        'braking_and_feedback_deviations', 'native_cooking_and_contact_offsets',
        'continuous_motion_between_samples'])
    collision_free_guarantee: FalseClaim = False

    @model_validator(mode='after')
    def coherent(self):
        if self.status == 'unsupported':
            if self.reason is None or self.detected is not None or self.goal_detected is not None or self.sample_count or self.pairs:
                raise ValueError('Unsupported geometry cannot claim an intersection check')
        elif (self.reason is not None or self.detected is None or self.goal_detected is None
              or self.sample_count < 2 or self.robot_source_sha256 is None
              or self.detected != bool(self.pairs) or self.colliding_sample_count > self.sample_count):
            raise ValueError('Incomplete sampled collision result')
        return self


class MotionHandPlan(RobotContract):
    initial_joint_positions_rad: JointMap
    goal_joint_positions_rad: JointMap
    trajectory_duration_s: Positive
    braking_steps: int = Field(ge=0)
    settling_steps: int = Field(ge=0)
    minimum_duration_s: Positive
    collision: MotionCollision


class MotionQueryResult(RobotContract):
    protocol: Literal['bench_motion_query_v1'] = 'bench_motion_query_v1'
    observation_sequence: int = Field(ge=0)
    sim_time_s: Nonnegative
    turn_index: int | None = Field(default=None, ge=0)
    planning_succeeded: StrictBool
    minimum_duration_s: Positive | None = None
    max_action_sim_s: Positive
    fits_action_budget: StrictBool | None = None
    requested_duration_s: Positive | None = None
    requested_duration_fits: StrictBool | None = None
    diagnostic: ActionDiagnostic | None = None
    hands: dict[Name, MotionHandPlan] = Field(default_factory=dict, max_length=8)
    advances_physics: FalseClaim = False
    mutates_controller: FalseClaim = False
    consumes_action_attempt: FalseClaim = False
    convergence_guaranteed: FalseClaim = False
    scope: Text = ('Original executor begin on an isolated copy of current robot references. '
        'Timing is a reference lower bound, not convergence. Collision advice is sampled nominal '
        'FR3 link0..7 nonadjacent arm convex geometry only; it excludes hands, other arms, objects, '
        'environment, braking/feedback deviations, native contact offsets and continuous motion '
        'between samples. No detected intersection does not establish a safe path.')

    @model_validator(mode='after')
    def coherent(self):
        if self.planning_succeeded:
            if not self.hands or self.minimum_duration_s is None or self.fits_action_budget is None:
                raise ValueError('Successful planning requires bounded duration and hand plans')
        elif self.hands or self.minimum_duration_s is not None or self.fits_action_budget is not None or self.diagnostic is None:
            raise ValueError('Failed planning requires the original diagnostic without a fictitious plan')
        return self
