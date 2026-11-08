"""Static reset geometry extension; existing referee contracts stay unchanged."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_serializer, model_validator

from .task_referee import Code, Number, PublicTaskRequirements as RefereeTaskRequirements

Vector2 = Annotated[list[Number], Field(min_length=2, max_length=2)]
Vector3 = Annotated[list[Number], Field(min_length=3, max_length=3)]
Bounds3 = Annotated[list[Vector3], Field(min_length=2, max_length=2)]


class StaticModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, serialize_by_alias=True)


class StaticBodyGeometry(StaticModel):
    object_id: Code
    local_bounds_m: Bounds3
    nominal_mass_kg: Annotated[Number, Field(gt=0)]

    @model_validator(mode='after')
    def positive_extent(self):
        if any(lo >= hi for lo, hi in zip(*self.local_bounds_m)):
            raise ValueError('Static body bounds must have positive extent')
        return self


class RemoteDimensions(StaticModel):
    remote: Vector3
    holder_opening: Vector2


class RemoteGoal(StaticModel):
    operation: Literal['remote_return_to_holder']


class WaterDimensions(StaticModel):
    kettle_inner_profile: Annotated[list[Vector3], Field(min_length=2, max_length=100)]
    cup_inner_profile: Annotated[list[Vector3], Field(min_length=2, max_length=100)]


class HandleGeometry(StaticModel):
    capsule_radius_m: Annotated[Number, Field(gt=0)]


class WaterGoal(StaticModel):
    operation: Literal['pour_and_return']
    return_position_m: Vector3
    target_interval_ml: Vector2


class StorageDimensions(StaticModel):
    item: Vector3


class StorageGoal(StaticModel):
    operation: Literal['collect_and_close']
    tray_inside_xy_m: Annotated[list[Vector2], Field(min_length=2, max_length=2)]
    tray_floor_z_m: Number


class StaticGeometry(StaticModel):
    bodies: Annotated[list[StaticBodyGeometry], Field(min_length=1, max_length=100)]

    @model_validator(mode='after')
    def unique_bodies(self):
        ids = [body.object_id for body in self.bodies]
        if len(set(ids)) != len(ids):
            raise ValueError('Static body identities must be unique')
        return self


class RemoteGeometry(StaticGeometry):
    dimensions_m: RemoteDimensions
    goal: RemoteGoal


class WaterGeometry(StaticGeometry):
    dimensions_m: WaterDimensions
    goal: WaterGoal
    handle_geometry: HandleGeometry


class StorageGeometry(StaticGeometry):
    dimensions_m: StorageDimensions
    goal: StorageGoal


class PublicResetCase(StaticModel):
    schema_version: Literal['living_room_public_reset_case.v1'] = Field(alias='schema')
    seed: Annotated[StrictInt, Field(ge=0, le=9)]
    configuration_id: Literal['single_arm', 'dual_same_side', 'dual_opposed']
    case_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    static_geometry: RemoteGeometry | WaterGeometry | StorageGeometry
    scope: Literal['Static authored dimensions and goals for this episode; live object poses and native evidence remain private.']


class PublicTaskRequirements(RefereeTaskRequirements):
    """The optional extension is absent from every legacy task response."""
    reset_case: PublicResetCase | None = None

    @model_serializer(mode='wrap')
    def omit_undeclared_reset_case(self, handler):
        result = super().omit_undeclared_scoring(handler)
        if result.get('reset_case') is None:
            result.pop('reset_case', None)
        return result

    @model_validator(mode='after')
    def matching_reset_task(self):
        if self.reset_case is not None:
            expected = {'remote_operation': RemoteGeometry, 'water_pouring': WaterGeometry,
                        'storage_box': StorageGeometry}.get(self.task_id)
            if expected is None or not isinstance(self.reset_case.static_geometry, expected):
                raise ValueError('Static reset geometry differs from the task')
        return self
