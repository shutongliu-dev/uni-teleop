"""Static, versioned task contracts. Never project live evaluator evidence."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class PublicContract(BaseModel):
    # Unknown keys are intentionally discarded at every nested public boundary.
    model_config = ConfigDict(extra='ignore', strict=True, frozen=True)


class Terminal(PublicContract):
    hold_s: Positive
    maximum_linear_speed_m_s: Positive
    maximum_angular_speed_rad_s: Positive
    maximum_tilt_rad: Positive
    tilt_definition: Literal['angle_between_object_local_positive_z_and_world_positive_z'] | None = None
    goal_geometry_required: bool
    goal_xy_radius_m: Positive
    goal_z_tolerance_m: Positive
    table_support_required: bool
    table_support_absent_required: bool
    hand_contact_required: bool
    all_robot_contact_absent_required: bool
    humanlike_grasp_required: bool
    # v3 grasp_hold: thumb pad vs index+middle pads, mean normal force over the
    # hold; every 20 ms window keeps at least half of it.
    minimum_pinch_normal_force_n: Positive | None = None


class Interaction(PublicContract):
    allowed_target_robot_contacts: Literal['hand_only']
    non_hand_robot_contact_is_irreversible: Literal[True]
    minimum_hand_contact_duration_s: Positive
    minimum_hand_contact_intervals: Literal[2]
    minimum_displacement_from_initial_m: Nonnegative
    contact_definition: Literal['solver_positive_normal_impulse']
    minimum_pair_normal_impulse_ns: Positive
    duration_semantics: Literal['unique_positive_hand_contact_substeps_times_physics_dt'] | None = None
    physics_dt_s: Positive | None = None
    contact_sampling: Literal['complete_consecutive_native_substeps_required'] | None = None

    @field_validator('non_hand_robot_contact_is_irreversible', 'minimum_hand_contact_intervals', mode='before')
    @classmethod
    def exact_literal_type(cls, value, info):
        expected = bool if info.field_name == 'non_hand_robot_contact_is_irreversible' else int
        if type(value) is not expected:
            raise ValueError('Literal has the wrong JSON type')
        return value


class Stages(PublicContract):
    minimum_lift_from_initial_m: Nonnegative
    lift_stage_hold_s: Nonnegative
    in_hand_transport_required: bool
    transport_minimum_lift_from_initial_m: Nonnegative
    continuous_grasp_from_lift_m: Nonnegative
    continuous_grasp_until: Literal['table_support_returns', 'not_applicable']
    humanlike_violation_is_irreversible: bool


class EpisodeConstraints(PublicContract):
    maximum_displacement_from_initial_m: Positive | None
    maximum_lift_from_initial_m: Positive | None
    maximum_motion_speed_m_s: Positive
    scope: Literal['entire_episode; returning_to_goal_does_not_erase_a_violation']


class PhysicalValidity(PublicContract):
    profile_id: Literal['agent_bench_physics_v1']
    maximum_reported_rigid_penetration_m: Positive
    scope: Literal['all_reported_rigid_pairs; not_a_full_collision_mesh_certificate']


class SuccessContract(PublicContract):
    protocol: Literal['simple_manipulation_success_v1', 'simple_manipulation_success_v2',
                      'simple_manipulation_success_v3']
    operation: Literal['push', 'lift_hold', 'pick_place', 'grasp_hold']
    terminal: Terminal
    interaction: Interaction
    stages: Stages
    episode_constraints: EpisodeConstraints
    physical_validity: PhysicalValidity
    humanlike_dwell_semantics: Literal['grasp_dwell_precedes_and_does_not_replace_terminal_hold']

    @model_validator(mode='after')
    def require_substep_contract(self):
        if self.protocol != 'simple_manipulation_success_v1' and any(
                getattr(self.interaction, key) is None
                for key in ('duration_semantics', 'physics_dt_s', 'contact_sampling')):
            raise ValueError('v2+ success contract requires explicit substep contact semantics')
        if self.protocol != 'simple_manipulation_success_v1' and self.terminal.tilt_definition is None:
            raise ValueError('v2+ success contract requires an explicit tilt definition')
        if (self.terminal.minimum_pinch_normal_force_n is not None) != (
                self.protocol == 'simple_manipulation_success_v3' and self.operation == 'grasp_hold'):
            raise ValueError('Only v3 grasp_hold contracts publish a minimum pinch force')
        return self


Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector3 = Annotated[list[Finite], Field(min_length=3, max_length=3)]


class GraspCriteria(PublicContract):
    hold_s: Positive
    joint_limit_tolerance_rad: Positive
    maximum_dip_minus_pip_rad: Positive
    minimum_contact_finger_flexion_sum_rad: Positive
    maximum_normal_direction_cosine: Annotated[Finite, Field(ge=-1, le=1)]
    minimum_opposed_substep_fraction: Annotated[Positive, Field(le=1)]
    maximum_robot_penetration_m: Positive
    wrist_approach_axis_local: Vector3
    wrist_approach_direction_world: Vector3
    maximum_wrist_approach_angle_rad: Positive
    object_in_wrist_box_min_m: Vector3
    object_in_wrist_box_max_m: Vector3
    maximum_tip_direction_cosine: Annotated[Finite, Field(ge=-1, le=1)]
    minimum_tip_span_m: Positive
    maximum_tip_span_m: Positive
    maximum_tip_object_distance_m: Positive
    maximum_object_to_tip_segment_distance_m: Positive


class GraspContract(PublicContract):
    humanlike_grasp_required: bool
    description: str
    scope: str
    push_grasp_scoring: str | None = None  # v8+ publishes it only on push tasks
    stable_grasp_before_lift_s: Positive | None = None
    continuous_grasp_from_lift_m: Positive | None = None
    continuous_grasp_until: Literal['table_support_returns'] | None = None
    violation_is_irreversible: bool | None = None
    criteria: GraspCriteria


def project_grasp_contract(value: dict) -> dict:
    return GraspContract.model_validate(value).model_dump(mode='json', exclude_unset=True)


def project_success_contract(value: dict) -> dict:
    """Allowlist only the declared static schema, including all nested objects."""
    result = SuccessContract.model_validate(value).model_dump(mode='json')
    for key in ('duration_semantics', 'physics_dt_s', 'contact_sampling'):
        if result['interaction'][key] is None:
            del result['interaction'][key]
    if result['terminal']['tilt_definition'] is None:
        del result['terminal']['tilt_definition']
    if result['terminal']['minimum_pinch_normal_force_n'] is None:
        del result['terminal']['minimum_pinch_normal_force_n']
    return result


def make_success_contract(task: dict, maximum_reported_rigid_penetration_m: float) -> dict:
    """Generate from the same frozen evaluation constants, never observations.

    The target's initial/current pose, contact identities, URDF paths, current
    score and privileged object trajectories are deliberately absent.
    """
    c = task['evaluation']
    mode = c['mode']
    carry = mode in ('lift_hold', 'pick_place')
    pinch = c.get('minimum_pinch_normal_force_n')
    return project_success_contract({
        'protocol': 'simple_manipulation_success_v3' if pinch is not None else 'simple_manipulation_success_v2',
        'operation': mode,
        'terminal': {
            'hold_s': c['hold_s'], 'maximum_linear_speed_m_s': c['maximum_speed_m_s'],
            'maximum_angular_speed_rad_s': c['maximum_angular_speed_rad_s'],
            'maximum_tilt_rad': c['maximum_tilt_rad'],
            'tilt_definition': 'angle_between_object_local_positive_z_and_world_positive_z',
            'goal_geometry_required': mode != 'lift_hold',
            'goal_xy_radius_m': c['goal_xy_radius_m'], 'goal_z_tolerance_m': c['goal_z_tolerance_m'],
            'table_support_required': mode != 'lift_hold',
            'table_support_absent_required': mode == 'lift_hold',
            'hand_contact_required': mode in ('lift_hold', 'grasp_hold'),
            'all_robot_contact_absent_required': mode in ('push', 'pick_place'),
            # Hands without the SHARPA humanlike contract are scored by contacts/pinch only.
            'humanlike_grasp_required': mode in ('lift_hold', 'grasp_hold') and 'humanlike_grasp' in c,
            'minimum_pinch_normal_force_n': pinch,
        },
        'interaction': {
            'allowed_target_robot_contacts': 'hand_only', 'non_hand_robot_contact_is_irreversible': True,
            'minimum_hand_contact_duration_s': c['minimum_contact_duration_s'],
            'minimum_hand_contact_intervals': 2,
            'minimum_displacement_from_initial_m': 0. if mode == 'grasp_hold' else c['minimum_displacement_m'],
            'contact_definition': 'solver_positive_normal_impulse', 'minimum_pair_normal_impulse_ns': 1.e-8,
            'duration_semantics': 'unique_positive_hand_contact_substeps_times_physics_dt',
            'physics_dt_s': c['physics_dt_s'],
            'contact_sampling': 'complete_consecutive_native_substeps_required',
        },
        'stages': {
            'minimum_lift_from_initial_m': c['minimum_lift_m'] if carry else 0.,
            'lift_stage_hold_s': c['lift_stage_hold_s'] if mode == 'pick_place' else 0.,
            'in_hand_transport_required': mode == 'pick_place',
            'transport_minimum_lift_from_initial_m': c['minimum_lift_m'] / 2. if mode == 'pick_place' else 0.,
            'continuous_grasp_from_lift_m': c.get('humanlike_lift_start_m', 0.) if carry else 0.,
            'continuous_grasp_until': 'table_support_returns' if carry else 'not_applicable',
            'humanlike_violation_is_irreversible': 'humanlike_grasp' in c and mode != 'grasp_hold',
        },
        'episode_constraints': {
            'maximum_displacement_from_initial_m': c['goal_xy_radius_m'] if mode == 'grasp_hold' else None,
            'maximum_lift_from_initial_m': (c['minimum_lift_m'] / 2. if mode == 'push'
                                          else c['goal_z_tolerance_m'] if mode == 'grasp_hold' else None),
            'maximum_motion_speed_m_s': c['maximum_motion_speed_m_s'],
            'scope': 'entire_episode; returning_to_goal_does_not_erase_a_violation',
        },
        'physical_validity': {
            'profile_id': 'agent_bench_physics_v1',
            'maximum_reported_rigid_penetration_m': maximum_reported_rigid_penetration_m,
            'scope': 'all_reported_rigid_pairs; not_a_full_collision_mesh_certificate',
        },
        'humanlike_dwell_semantics': 'grasp_dwell_precedes_and_does_not_replace_terminal_hold',
    })


class BimanualSuccessContract(PublicContract):
    """Public success conditions of a two-hand task (dual FR3 + SHARPA)."""
    protocol: Literal['bimanual_manipulation_success_v1']
    operation: Literal['bimanual_lift', 'dual_lift']
    holding_hands: dict[str, list[Literal['left', 'right']]]
    exclusive_holding: bool
    minimum_lift_from_initial_m: Positive
    hold_s: Positive
    maximum_linear_speed_m_s: Positive
    maximum_angular_speed_rad_s: Positive
    maximum_tilt_rad: Positive
    tilt_definition: Literal['angle_between_object_local_positive_z_and_world_positive_z']
    table_support_absent_required: Literal[True]
    allowed_target_robot_contacts: Literal['hand_only']
    non_hand_robot_contact_is_irreversible: Literal[True]
    minimum_contact_duration_per_required_hand_s: Positive
    contact_definition: Literal['solver_positive_normal_impulse']
    maximum_motion_speed_m_s: Positive
    physical_validity: PhysicalValidity


def make_bimanual_success_contract(task: dict, maximum_reported_rigid_penetration_m: float) -> dict:
    c = task['evaluation']
    mode = c['mode']
    holders = ({c['object']: ['left', 'right']} if mode == 'bimanual_lift'
               else {name: [side] for name, side in c['objects'].items()})
    return BimanualSuccessContract.model_validate({
        'protocol': 'bimanual_manipulation_success_v1', 'operation': mode, 'holding_hands': holders,
        'exclusive_holding': mode == 'dual_lift', 'minimum_lift_from_initial_m': c['minimum_lift_m'], 'hold_s': c['hold_s'],
        'maximum_linear_speed_m_s': c['maximum_speed_m_s'], 'maximum_angular_speed_rad_s': c['maximum_angular_speed_rad_s'],
        'maximum_tilt_rad': c['maximum_tilt_rad'],
        'tilt_definition': 'angle_between_object_local_positive_z_and_world_positive_z',
        'table_support_absent_required': True, 'allowed_target_robot_contacts': 'hand_only',
        'non_hand_robot_contact_is_irreversible': True,
        'minimum_contact_duration_per_required_hand_s': c['minimum_contact_duration_s'],
        'contact_definition': 'solver_positive_normal_impulse', 'maximum_motion_speed_m_s': c['maximum_motion_speed_m_s'],
        'physical_validity': {'profile_id': 'agent_bench_physics_v1',
                              'maximum_reported_rigid_penetration_m': maximum_reported_rigid_penetration_m,
                              'scope': 'all_reported_rigid_pairs; not_a_full_collision_mesh_certificate'},
    }).model_dump(mode='json')
