"""Public finite practice selections and goals; no private native evidence."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

Code = Annotated[StrictStr, Field(pattern=r'^[a-z][a-z0-9_]*$', max_length=100)]
Digest = Annotated[StrictStr, Field(pattern=r'^[0-9a-f]{64}$')]
ObjectId = Annotated[StrictStr, Field(pattern=r'^[a-z][A-Za-z0-9_]*$', max_length=100)]
Number = Annotated[float, Field(allow_inf_nan=False)]


class PracticeSelection(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    active_side: Literal['left', 'right']
    case_id: Code
    split: Literal['train', 'validation']


class PracticeTarget(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    object_id: ObjectId
    position_m: Annotated[list[Number], Field(min_length=3, max_length=3)] | None = None
    quaternion_wxyz: Annotated[list[Number], Field(min_length=4, max_length=4)] | None = None
    cells: list[Annotated[list[StrictInt], Field(min_length=2, max_length=2)]] | None = None
    orientation: Literal['exact', 'upright', 'face_up', 'face_down', 'flat'] = 'exact'
    region_bounds_xy_m: Annotated[list[list[Number]], Field(min_length=2, max_length=2)] | None = None


class PracticeStage(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    operation: Literal['place', 'flip', 'restore', 'collect', 'shoot']
    targets: Annotated[list[PracticeTarget], Field(min_length=1, max_length=12)]
    minimum_points: Annotated[StrictInt, Field(ge=0, le=3)] = 0
    obstacle_ids: list[ObjectId] = Field(default_factory=list)
    require_obstacle_contact: bool = False
    require_line_crossing: bool = True


class PracticeGoal(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    frame_id: Literal['world_m_z_up'] = 'world_m_z_up'
    stages: Annotated[list[PracticeStage], Field(min_length=1, max_length=8)]
    description: Annotated[StrictStr, Field(min_length=1, max_length=4096)]


class PracticeBox(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    position_m: Annotated[list[Number], Field(min_length=3, max_length=3)]
    dimensions_m: Annotated[list[Number], Field(min_length=3, max_length=3)]


class PracticePieceGeometry(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    object_id: ObjectId
    kind: Code
    dimensions_m: Annotated[list[Number], Field(min_length=3, max_length=3)]
    mass_kg: Annotated[Number, Field(gt=0)]
    collision_boxes: list[PracticeBox] = Field(default_factory=list)


class PracticeApparatus(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    board_top_z_m: Number
    deck_bounds_xy_m: Annotated[list[list[Number]], Field(min_length=2, max_length=2)]
    pieces: list[PracticePieceGeometry]
    launch_region_bounds_xy_m: Annotated[list[list[Number]], Field(min_length=2, max_length=2)] | None = None
    launch_line_y_m: Number | None = None
    launch_direction_y: Literal[-1, 1] | None = None
    scoring_center_xy_m: Annotated[list[Number], Field(min_length=2, max_length=2)] | None = None
    scoring_radii_m: list[Number] = Field(default_factory=list)
    scoring_points: list[StrictInt] = Field(default_factory=list)
    geometry_scope: Literal['static_original_apparatus_geometry_not_live_state_or_grasp_strategy'] = 'static_original_apparatus_geometry_not_live_state_or_grasp_strategy'


class PublicPracticeCase(PracticeSelection):
    model_config = ConfigDict(extra='forbid', frozen=True, serialize_by_alias=True)
    schema_version: Literal['card_rsi_practice_case.v1'] = Field(default='card_rsi_practice_case.v1', alias='schema')
    gym_id: Code
    parent_task_id: Code
    stratum: Code
    goal: PracticeGoal
    apparatus: PracticeApparatus
    parent_release_sha256: Digest
    public_rules_sha256: Digest
    case_sha256: Digest


class PracticeProgress(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    stage_index: Annotated[StrictInt, Field(ge=0, le=8)]
    stage_count: Annotated[StrictInt, Field(ge=1, le=8)]
    confirmed_stages: Annotated[StrictInt, Field(ge=0, le=8)]

