"""Public finite cloth reset identity, without cloth nodes or evaluator geometry."""
from typing import Annotated, Literal

from pydantic import Field, StrictInt

from .living_reset import StaticModel

TASK_IDS = ('towel_flattening', 'towel_folding', 'towel_stacking',
            'shirt_folding', 'trousers_folding')
VARIATION = 'xy_translation_and_visual_texture_only'
PARAMETERIZED_VARIATION = 'xy_yaw_appearance_and_local_material_parameters'
SCOPE = ('Frozen cloth placement and appearance identity only; live cloth nodes, '
         'reset coordinates and evaluator geometry remain private.')
PARAMETERIZED_SCOPE = ('Frozen cloth placement, appearance and local material parameter identity only; '
                       'live cloth nodes, reset coordinates and evaluator geometry remain private.')


class PublicLaundryResetCase(StaticModel):
    schema_version: Literal['laundry_public_reset_case.v1'] = Field(alias='schema')
    family: Literal['laundry_room']
    task_id: Literal['towel_flattening', 'towel_folding', 'towel_stacking',
                     'shirt_folding', 'trousers_folding']
    seed: Annotated[StrictInt, Field(ge=0, le=9)]
    configuration_id: Literal['single_arm', 'dual_same_side', 'dual_opposed']
    case_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    case_fingerprint: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    parent_release_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    variation: Literal['xy_translation_and_visual_texture_only',
                       'xy_yaw_appearance_and_local_material_parameters']
    scope: Literal['Frozen cloth placement and appearance identity only; live cloth nodes, reset coordinates and evaluator geometry remain private.',
                   'Frozen cloth placement, appearance and local material parameter identity only; live cloth nodes, reset coordinates and evaluator geometry remain private.']
