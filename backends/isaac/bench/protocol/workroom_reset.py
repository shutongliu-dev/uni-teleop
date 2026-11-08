"""Finite ordinary reset identity and nominal geometry, never live truth."""
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_serializer, model_validator

from .living_reset import (PublicResetCase, RemoteGeometry, WaterGeometry, StorageGeometry,
                          StaticModel, StaticGeometry)
from .task_referee import PublicTaskRequirements as RefereeTaskRequirements
from .laundry_reset import PublicLaundryResetCase

TASK_FAMILIES = {
    'stationery_sorting': 'study', 'book_stacking': 'study', 'book_shelving': 'study',
    'produce_transfer': 'kitchen', 'utensil_sorting': 'kitchen',
    'pantry_stocking': 'kitchen', 'table_clearing': 'kitchen',
    'sample_sorting': 'laboratory', 'vial_racking': 'laboratory',
    'parts_sorting': 'mechanical_assembly',
    'peg_insertion': 'mechanical_assembly', 'shaft_spacer_assembly': 'mechanical_assembly',
    'fastener_assembly': 'mechanical_assembly', 'gear_train_assembly': 'mechanical_assembly',
    'nist_connector_insertion': 'mechanical_assembly', 'nist_belt_drive_assembly': 'mechanical_assembly',
    'nist_cable_routing': 'mechanical_assembly', 'nist_wire_harness': 'mechanical_assembly',
    'cooperative_insertion': 'mechanical_assembly',
    'book_retrieval': 'study', 'drawer_retrieval': 'study', 'reading_stand': 'study',
    'breakfast_serving': 'kitchen', 'pot_lid_placement': 'kitchen', 'surface_wiping': 'kitchen',
    'cap_placement': 'laboratory', 'solid_weighing': 'laboratory',
    'pipette_transfer': 'laboratory', 'tube_clamp_transfer': 'laboratory',
}
SCOPE = 'Static authored dimensions for this episode; live object poses and native evidence remain private.'


class PublicWorkroomResetCase(StaticModel):
    schema_version: Literal['workroom_public_reset_case.v1'] = Field(alias='schema')
    family: Literal['study', 'kitchen', 'laboratory', 'mechanical_assembly']
    task_id: Literal['stationery_sorting', 'book_stacking', 'book_shelving',
                     'produce_transfer', 'utensil_sorting', 'pantry_stocking', 'table_clearing',
                     'sample_sorting', 'vial_racking', 'parts_sorting',
                     'peg_insertion', 'shaft_spacer_assembly', 'fastener_assembly',
                     'gear_train_assembly', 'nist_connector_insertion', 'nist_belt_drive_assembly',
                     'nist_cable_routing', 'nist_wire_harness', 'cooperative_insertion',
                     'book_retrieval', 'drawer_retrieval', 'reading_stand',
                     'breakfast_serving', 'pot_lid_placement', 'surface_wiping',
                     'cap_placement', 'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer']
    seed: Annotated[StrictInt, Field(ge=0, le=9)]
    configuration_id: Literal['single_arm', 'dual_same_side', 'dual_opposed']
    case_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    static_geometry: StaticGeometry
    scope: Literal['Static authored dimensions for this episode; live object poses and native evidence remain private.']

    @model_validator(mode='after')
    def correct_family(self):
        if TASK_FAMILIES[self.task_id] != self.family:
            raise ValueError('Reset family differs from the task')
        if self.family == 'kitchen' and self.configuration_id != 'single_arm':
            raise ValueError('Kitchen retains its original single-arm installation')
        if self.task_id == 'cooperative_insertion' and self.configuration_id != 'dual_same_side':
            raise ValueError('Cooperative insertion retains its one original same-side dual installation')
        return self


class PublicTaskRequirements(RefereeTaskRequirements):
    reset_case: PublicResetCase | PublicWorkroomResetCase | PublicLaundryResetCase | None = None

    @model_serializer(mode='wrap')
    def omit_undeclared_reset_case(self, handler):
        result = super().omit_undeclared_scoring(handler)
        if result.get('reset_case') is None:
            result.pop('reset_case', None)
        return result

    @model_validator(mode='after')
    def matching_reset_task(self):
        if isinstance(self.reset_case, (PublicWorkroomResetCase, PublicLaundryResetCase)):
            if self.reset_case.task_id != self.task_id:
                raise ValueError('Reset case differs from the public task')
        elif self.reset_case is not None:
            expected = {'remote_operation': RemoteGeometry, 'water_pouring': WaterGeometry,
                        'storage_box': StorageGeometry}.get(self.task_id)
            if expected is None or not isinstance(self.reset_case.static_geometry, expected):
                raise ValueError('Living reset geometry differs from the task')
        return self
