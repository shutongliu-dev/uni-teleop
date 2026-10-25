"""Versioned callable bench contracts shared by all client transports."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, TypeAdapter, create_model, model_serializer, model_validator

from .schemas import (ActionDiagnostic, ActionName, AgentObservation, Budgets, CameraObservation,
    Contract, EpisodeResult, GeometryObjective, JointMap, HandNamespace, Name, Nonnegative, Pose,
    Positive, RobotCapabilities, RobotExecutionState, System0Action, TerminationReason, Vec3)
from .robot import IKQuery, IKQueryResult, RobotGuide
from .hand_presets import HandPresetRequest, HandPresetResult, HandPresetCatalog
from .task_referee import PublicTaskRequirements
from bench.sensors.contact_schema import ContactHistoryValue

PROTOCOL = 'bench_functions_v1'
VERSION = '1.0.0'


class NoArguments(Contract):
    pass


class SnapshotArguments(Contract):
    observation_sequence: int = Field(strict=True, ge=0)


class CameraArguments(SnapshotArguments):
    camera_id: Name


class PixelArguments(CameraArguments):
    u: int = Field(strict=True, ge=0)
    v: int = Field(strict=True, ge=0)


class TransformArguments(SnapshotArguments):
    source_frame: Name = Field(description='world, robot_base, a published hand flange, or camera:<camera_id>; no object frames.')
    target_frame: Name = Field(description='Destination uses the same public frame inventory as source_frame.')
    points_m: list[Vec3] = Field(min_length=1, max_length=1024)


class ActionArguments(SnapshotArguments):
    action_id: ActionName


class MoveGeometryArguments(ActionArguments):
    timeout_sim_s: Positive
    geometry: GeometryObjective | None = None
    hand_geometries: dict[HandNamespace, GeometryObjective] | None = None

    @model_validator(mode='after')
    def one_target(self):
        if (self.geometry is not None) == bool(self.hand_geometries):
            raise ValueError('Provide geometry or nonempty hand_geometries')
        System0Action(**self.model_dump(), skill='move_geometry')
        return self


class MoveWristArguments(ActionArguments):
    hand: HandNamespace
    wrist: Pose
    timeout_sim_s: Positive
    position_tolerance_m: Positive = .01
    orientation_tolerance_rad: Positive = .1

    @model_validator(mode='after')
    def world_pose(self):
        if self.wrist.frame_id != 'world':
            raise ValueError('Wrist targets use the world frame')
        return self


class HandTargetsArguments(ActionArguments):
    hand: HandNamespace
    hand_posture_rad: JointMap | None = None
    fingertip_positions_m: dict[Literal['thumb', 'index', 'middle', 'ring', 'little'], Vec3] = Field(default_factory=dict)
    timeout_sim_s: Positive

    @model_validator(mode='after')
    def target_required(self):
        if not (self.hand_posture_rad or self.fingertip_positions_m):
            raise ValueError('A hand joint or fingertip target is required')
        return self


class MoveJointsArguments(ActionArguments):
    duration_s: Positive
    arm_joint_positions_rad: JointMap
    hand_joint_positions_rad: JointMap


class HoldArguments(ActionArguments):
    duration_s: Positive
    mode: Literal['reference', 'acknowledged_drive'] = Field(default='reference', description=
        'reference retains the accepted geometric reference and tracking behavior; acknowledged_drive '
        'requires an advertised executor and verified stationary drive ACK, holds those native targets '
        'despite measured drift, and resumes tracking only on explicit geometric reselection.')

    @model_serializer(mode='wrap')
    def omit_default_mode(self, handler):
        value = handler(self)
        if value.get('mode') == 'reference':
            value.pop('mode')
        return value


class ActionIdArguments(Contract):
    action_id: ActionName


class ActionStatus(Contract):
    action_id: ActionName
    status: Literal['accepted', 'running', 'completed', 'cancelled', 'failed']
    sim_time_s: Nonnegative
    episode_finished: bool
    elapsed_sim_s: Nonnegative
    diagnostic: ActionDiagnostic | None = None
    termination_reason: TerminationReason | None = None
    robot_execution_state: RobotExecutionState | None = None
    contact_history: ContactHistoryValue | None = None


class CancelStatus(ActionStatus):
    cancellation_requested: bool


class CameraResult(Contract):
    observation_sequence: int = Field(strict=True, ge=0)
    camera: CameraObservation


class PixelResult(SnapshotArguments):
    sim_time_s: Nonnegative
    camera_id: Name
    pixel_uv: tuple[int, int]
    depth_m: Positive
    frame_id: Literal['world'] = 'world'
    point_m: Vec3
    source: Literal['public_rgbd_camera_z'] = 'public_rgbd_camera_z'


class TransformResult(TransformArguments):
    sim_time_s: Nonnegative
    source: Literal['published_robot_and_camera_frames'] = 'published_robot_and_camera_frames'


class FunctionError(Contract):
    code: Literal['invalid_arguments', 'unsupported_function', 'unsupported_capability',
        'stale_observation', 'episode_busy', 'episode_finished', 'action_conflict',
        'not_found', 'invalid_depth', 'budget_exhausted', 'result_pending',
        'permission_denied', 'infrastructure_error']
    message: str = Field(max_length=512)
    http_status: int = Field(ge=400, le=599)


class BudgetState(Contract):
    limits: Budgets
    action_attempts: int = Field(strict=True, ge=0)
    elapsed_sim_s: Nonnegative
    elapsed_wall_s: Nonnegative
    remaining_wall_s: Nonnegative | None
    model_accounting: Literal['unavailable'] = 'unavailable'
    client_tool_budget_enforced_here: Literal[False] = False


class CallContext(Contract):
    episode_id: Name
    observation_sequence: int = Field(strict=True, ge=0)
    observation_sim_time_s: Nonnegative
    episode_finished: bool
    budget: BudgetState
    request_recorded: bool


# Names are shared by dispatch, discovery and JSON Schema. No client has to
# reverse-engineer action kind/track conversions or maintain its own inventory.
FUNCTION_INPUTS = {
    'get_robot': NoArguments,
    'get_task': NoArguments,
    'observe': NoArguments,
    'read_camera': CameraArguments,
    'project_pixel': PixelArguments,
    'transform_points': TransformArguments,
    'query_ik': IKQuery,
    'get_hand_presets': NoArguments,
    'hand_preset': HandPresetRequest,
    'move_geometry': MoveGeometryArguments,
    'move_wrist': MoveWristArguments,
    'set_hand_targets': HandTargetsArguments,
    'move_joints': MoveJointsArguments,
    'hold': HoldArguments,
    'get_action': ActionIdArguments,
    'cancel_action': ActionIdArguments,
    'get_result': NoArguments,
    'finish_episode': NoArguments,
}
FUNCTION_OUTPUTS = {
    'get_robot': RobotGuide, 'observe': AgentObservation,
    'get_task': PublicTaskRequirements,
    'read_camera': CameraResult, 'project_pixel': PixelResult,
    'transform_points': TransformResult, 'query_ik': IKQueryResult,
    'get_hand_presets': HandPresetCatalog, 'hand_preset': HandPresetResult,
    **{name: ActionStatus for name in ('move_geometry', 'move_wrist', 'set_hand_targets',
                                      'move_joints', 'hold', 'get_action')},
    'cancel_action': CancelStatus, 'get_result': EpisodeResult, 'finish_episode': EpisodeResult,
}
ACTION_FUNCTIONS = frozenset(('move_geometry', 'move_wrist', 'set_hand_targets', 'move_joints', 'hold'))
FunctionName = Literal[tuple(FUNCTION_INPUTS)]


class CallBase(Contract):
    protocol: Literal['bench_functions_v1'] = PROTOCOL
    request_id: Name


CALL_MODELS = {
    name: create_model('Call' + ''.join(word.title() for word in name.split('_')),
        __base__=CallBase, function=(Literal[name], name), arguments=(arguments, ...))
    for name, arguments in FUNCTION_INPUTS.items()
}
FunctionCall = Annotated[Union[tuple(CALL_MODELS.values())], Field(discriminator='function')]
CALL_ADAPTER = TypeAdapter(FunctionCall)


class ReplyBase(Contract):
    protocol: Literal['bench_functions_v1'] = PROTOCOL
    capability_version: Literal['1.0.0'] = VERSION
    request_id: Name | None
    context: CallContext | None


class FunctionFailure(ReplyBase):
    ok: Literal[False] = False
    function: FunctionName | None
    error: FunctionError


class FunctionSuccess(ReplyBase):
    ok: Literal[True] = Field(default=True, description='The function returned a valid result; this does not mean an action or task succeeded. Inspect result.status or result.outcome.')


REPLY_MODELS = {
    name: create_model('Result' + ''.join(word.title() for word in name.split('_')),
        __base__=FunctionSuccess, function=(Literal[name], name), result=(result, ...))
    for name, result in FUNCTION_OUTPUTS.items()
}
FunctionSuccessValue = Annotated[Union[tuple(REPLY_MODELS.values())], Field(discriminator='function')]
FunctionResponse = Annotated[FunctionSuccessValue | FunctionFailure, Field(discriminator='ok')]
RESPONSE_ADAPTER = TypeAdapter(FunctionResponse)


class FunctionCapability(Contract):
    name: FunctionName
    version: Literal['1.0.0'] = VERSION
    available: bool
    unavailable_reason: str | None
    input_schema: str
    output_schema: str
    advances_physics: bool
    consumes_action_attempt: bool
    asynchronous_action: bool


class FunctionCatalog(Contract):
    protocol: Literal['bench_functions_v1'] = PROTOCOL
    capability_version: Literal['1.0.0'] = VERSION
    robot_capabilities: RobotCapabilities
    control_track: str
    functions: list[FunctionCapability]
    schema_endpoint: Literal['/functions/schema'] = '/functions/schema'
    call_endpoint: Literal['POST /episodes/{episode_id}/call',
                           'POST /board-games/{episode_id}/players/{player_id}/call',
                           'POST /turn-games/{episode_id}/players/{player_id}/call'] = 'POST /episodes/{episode_id}/call'
    reference: None = None


def function_schemas():
    models = {*FUNCTION_INPUTS.values(), *FUNCTION_OUTPUTS.values(), FunctionError,
              CallContext, FunctionCatalog, FunctionCapability, ActionStatus, CancelStatus}
    return {'FunctionCall': CALL_ADAPTER.json_schema(),
            'FunctionResponse': RESPONSE_ADAPTER.json_schema(),
            **{model.__name__: model.model_json_schema() for model in sorted(models, key=lambda cls: cls.__name__)}}
