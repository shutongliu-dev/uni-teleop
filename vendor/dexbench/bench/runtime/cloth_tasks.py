"""Only static, explicit task requirements may be published to the Agent."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

Number = Annotated[StrictFloat, Field(allow_inf_nan=False)]
Positive = Annotated[Number, Field(gt=0)]
Fraction = Annotated[Number, Field(gt=0, le=1)]
Vec2 = tuple[Number, Number]
Vec3 = tuple[Number, Number, Number]


class PublicClothContract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    version: Literal['cloth_table_drag_cpu_candidate_v1']
    frame: Literal['world_m_z_up']
    task: Literal['translate_entire_cloth_along_positive_y']
    nominal_cloth_size_xy_m: tuple[Positive, Positive]
    table_top_z_m: Number
    table_xy_bounds_m: tuple[Vec2, Vec2]
    all_vertices_minimum_y_translation_m: Positive
    centroid_goal_xy_m: Vec2
    centroid_goal_radius_m: Positive
    terminal_vertex_bounds_m: tuple[Vec3, Vec3]
    spread_metric: Literal['fraction_of_fixed_target_cell_centres_inside_projected_mesh_union']
    spread_rectangle_xy_m: tuple[Vec2, Vec2]
    spread_grid_cells_xy: tuple[Annotated[StrictInt, Field(ge=3, le=50)],
                               Annotated[StrictInt, Field(ge=3, le=50)]]
    minimum_projected_coverage_fraction: Fraction
    maximum_vertex_speed_m_s: Positive
    maximum_sampled_vertex_motion_speed_m_s: Positive
    continuous_hold_s: Positive
    whole_episode_vertex_bounds_m: tuple[Vec3, Vec3]
    sample_period_s: Positive
    sample_period_tolerance_s: Positive
    reset_time_tolerance_s: Positive
    reset_vertex_tolerance_m: Positive
    reset_maximum_vertex_speed_m_s: Positive
    numeric_comparison_tolerance_m: Annotated[Positive, Field(le=1e-6)]
    reset_rule: Literal['first_sample_at_zero_must_match_frozen_initial_mesh_then_no_rebaseline']
    sampling_rule: Literal['every_control_tick_required_invalid_or_missing_sample_irreversibly_fails']
    path_rule: Literal['all_sampled_vertices_inside_table_envelope_for_entire_episode']
    physical_acceptance: Literal['independent_full_substep_physics_acceptance_required']
    force_semantics: Literal['no_cloth_force_or_grasp_contact_claim']
    public_observation_policy: Literal['no_live_vertices_velocities_topology_or_contact_identities']

    @model_validator(mode='after')
    def coherent(self):
        for bounds in (self.table_xy_bounds_m, self.terminal_vertex_bounds_m,
                       self.spread_rectangle_xy_m, self.whole_episode_vertex_bounds_m):
            if any(lo >= hi for lo, hi in zip(*bounds)):
                raise ValueError('bounds must have positive extent')
        terminal, whole = self.terminal_vertex_bounds_m, self.whole_episode_vertex_bounds_m
        if any(terminal[0][i] < whole[0][i] or terminal[1][i] > whole[1][i] for i in range(3)):
            raise ValueError('terminal bounds must be within whole-episode bounds')
        if self.sample_period_tolerance_s >= self.sample_period_s / 10:
            raise ValueError('sample tolerance cannot hide a missing or duplicate tick')
        if self.reset_time_tolerance_s >= self.sample_period_s / 10:
            raise ValueError('reset tolerance cannot hide a skipped first tick')
        if self.continuous_hold_s < self.sample_period_s:
            raise ValueError('hold requires at least two observations')
        for axis in range(2):
            if (whole[0][axis] < self.table_xy_bounds_m[0][axis]
                    or whole[1][axis] > self.table_xy_bounds_m[1][axis]):
                raise ValueError('whole-episode envelope must be over the table')
            if (self.spread_rectangle_xy_m[0][axis] < terminal[0][axis]
                    or self.spread_rectangle_xy_m[1][axis] > terminal[1][axis]):
                raise ValueError('coverage rectangle must be in terminal region')
        return self


# This module contains static release metadata only. It never receives a
# runtime observation, scene object, evaluator instance or simulator handle.
from collections.abc import Mapping
from pydantic import StrictStr

PublicText = Annotated[StrictStr, Field(min_length=1, max_length=4096)]
PublicIdentifier = Annotated[StrictStr, Field(min_length=1, max_length=128)]


class ClothGoal(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    object_id: Literal['cloth']
    operation: Literal['cloth_table_drag']
    frame_id: Literal['world_m_z_up']
    description: PublicText
    success_contract: PublicClothContract


class ClothStep(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    id: PublicIdentifier
    instruction_zh: PublicText


class ClothTaskMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    task_id: PublicIdentifier
    title_zh: PublicText
    steps: Annotated[list[ClothStep], Field(min_length=1, max_length=16)]
    goal: ClothGoal

    @model_validator(mode='after')
    def unique_steps(self):
        names = [step.id for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError('public step identifiers must be unique')
        return self


def declared(value, model):
    """Strip all unknown fields before validating known static fields strictly."""
    if not isinstance(value, Mapping):
        raise ValueError('static public metadata must be an object')
    return {key: value[key] for key in model.model_fields if key in value}


def project_cloth_task(task_id, task):
    """Explicit allowlist at every level; no evaluation/initial geometry leaks.

    Malformed declared fields fail closed through the server's generic 503
    handler; error details remain operator-private. Never fall back to private
    evaluation values to fill a missing public requirement.
    """
    if not isinstance(task, Mapping) or task.get('id') != task_id:
        raise ValueError('manifest/task identity mismatch')
    raw_goal = task.get('public_goal')
    goal = declared(raw_goal, ClothGoal)
    goal['success_contract'] = declared(raw_goal.get('success_contract'), PublicClothContract)
    raw_steps = task.get('steps')
    if not isinstance(raw_steps, list):
        raise ValueError('static public steps must be a list')
    value = ClothTaskMetadata.model_validate({
        'task_id': task_id, 'title_zh': task.get('title_zh'),
        'steps': [declared(step, ClothStep) for step in raw_steps], 'goal': goal,
    })
    row = value.model_dump(mode='json')
    row['instruction'] = value.title_zh + '。' + '；'.join(step.instruction_zh for step in value.steps)
    return row
