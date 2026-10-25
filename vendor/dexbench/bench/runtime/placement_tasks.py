"""Allowlisted placement intent, independent of observations and evaluator state."""
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator

Number = Annotated[StrictFloat, Field(allow_inf_nan=False)]
CellIndex = Annotated[StrictInt, Field(ge=0, le=2)]
Cell = tuple[CellIndex, CellIndex]
Text = Annotated[StrictStr, Field(min_length=1, max_length=4096)]


class PublicPlacementGoal(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    operation: Literal['place_three_pieces']
    frame_id: Literal['world', 'world_m_z_up']
    description: Text
    piece_ids: tuple[Literal['black_00'], Literal['black_01'], Literal['black_02']]
    candidate_cells: Annotated[list[Cell], Field(min_length=9, max_length=9)]
    target_cells: Annotated[list[Cell], Field(min_length=3, max_length=3)]
    target_positions_m: Annotated[list[tuple[Number, Number, Number]], Field(min_length=3, max_length=3)]

    @model_validator(mode='after')
    def distinct_cells(self):
        if len(set(self.candidate_cells)) != 9 or len(set(self.target_cells)) != 3:
            raise ValueError('Placement requires all nine candidates and three distinct goals')
        return self


class DeclaredRules(BaseModel):
    """Every nested rule model strips undeclared private fields independently."""
    model_config = ConfigDict(extra='ignore', frozen=True)


class PlacementThresholds(DeclaredRules):
    physics_dt_s: Number
    control_period_s: Number
    dwell_s: Number
    maximum_linear_speed_m_s: Number
    maximum_angular_speed_rad_s: Number
    maximum_tilt_rad: Number
    bottom_z_tolerance_m: Number
    edge_clearance_m: Number
    minimum_support_impulse_ns: Number
    maximum_robot_contact_impulse_ns: Number
    maximum_penetration_m: Number
    maximum_stable_position_range_m: Number
    maximum_stable_orientation_range_rad: Number
    initial_position_tolerance_m: Number
    initial_orientation_tolerance_rad: Number


class PlacementSuccess(DeclaredRules):
    piece_ids: tuple[Literal['black_00'], Literal['black_01'], Literal['black_02']]
    required_count: Literal[3]
    piece_to_target_assignment: Text
    thresholds: PlacementThresholds
    cell_acceptance: Text
    support: Text
    release: Text
    stability: Text
    reset_credit: Literal[False]


class PlacementRecovery(DeclaredRules):
    wrong_cell_or_disturbed_piece: Text
    duplicate_target_occupants: Text
    placement_intent_required: Literal[False]
    automatic_forfeit: Literal[False]
    api_or_evidence_fault: Text
    budget_exhaustion: Text


class PlacementSelection(DeclaredRules):
    version: Literal['sha256_ranked_cells_v1']
    candidate_cells: Annotated[list[Cell], Field(min_length=9, max_length=9)]
    count: Literal[3]
    replacement: Literal[False]
    algorithm: Text
    seed_range: tuple[Literal[0], Literal[4294967295]]
    scope: Text


class PublicPlacementRules(DeclaredRules):
    schema_version: Literal['card_room_placement_rules_v1'] = Field(alias='schema')
    status: Text
    instruction_zh: Text
    target_cells: Annotated[list[Cell], Field(min_length=3, max_length=3)]
    target_selection: PlacementSelection
    success: PlacementSuccess
    recovery: PlacementRecovery
    physical_actions: Text
    evidence_scope: Text
    qualification: Text


def project_placement_task(task, *, episode=True):
    """Read declared public fields only; no actor paths or live evidence escape."""
    if not isinstance(task, Mapping) or task.get('id') != 'task0':
        raise ValueError('Placement task identity differs')
    public = task.get('public_goal')
    if not isinstance(public, Mapping):
        raise ValueError('Placement public goal is missing')
    goal = PublicPlacementGoal.model_validate({key: public[key]
        for key in PublicPlacementGoal.model_fields if key in public}).model_dump(mode='json')
    rules = PublicPlacementRules.model_validate(task.get('public_rules')).model_dump(mode='json', by_alias=True)
    if rules['target_cells'] != goal['target_cells']:
        raise ValueError('Placement public rules and goal differ')
    seed = task.get('episode_seed')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Placement episode seed must be an unsigned 32-bit integer')
    title = task.get('title_zh')
    steps = task.get('steps')
    if not isinstance(title, str) or not isinstance(steps, list) or not steps or any(
            not isinstance(row, Mapping) or not isinstance(row.get('instruction_zh'), str) for row in steps):
        raise ValueError('Placement public instructions are missing')
    row = {'task_id': 'task0', 'instruction': title+'。'+'；'.join(s['instruction_zh'] for s in steps),
           'task_kind': 'manipulation', 'create_endpoint': 'POST /episodes', 'goal': goal,
           'public_rules': rules,
           'target_selection': 'Omit seed for a fresh random episode; specify seed to reproduce three distinct target cells. '
                               'The actual seed is returned in the episode. Any piece may fill any target.'}
    if episode:
        row['episode_seed'] = seed
    else:
        # The release's seed-zero template is not the next episode's goal.
        goal.pop('target_cells')
        goal.pop('target_positions_m')
        rules.pop('target_cells')
    return row
