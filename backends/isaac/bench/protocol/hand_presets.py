"""Robot-only hand reference contracts; no automatic grasp or execution."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .schemas import Contract, Finite, Hash, JointMap, Name, Pose, Positive

HandSide = Literal['left', 'right']
HandPresetName = Literal['open', 'pinch_prepose']
Amount = Annotated[Finite, Field(ge=0., le=1.)]
Finger = Literal['thumb', 'index', 'middle', 'ring', 'little']


class HandPresetRequest(Contract):
    preset: HandPresetName = Field(description='A robot-only reference posture, not a grasp or execution command.')
    side: HandSide = Field(description='Must select a SHARPA hand actually present in this episode.')
    # q(amount) = amount * anchor; amount=0 is the URDF zero/open pose.
    # The open preset's anchor is itself zero, so every amount gives open.
    amount: Amount = Field(default=1., description=(
        'Joint interpolation fraction: 0 is URDF zero/open, 1 is the frozen prepose. '
        'Not an object aperture or closure force. For open, every amount returns zero.'))


class HandPresetSource(Contract):
    upstream_repository: str
    upstream_commit: Annotated[str, Field(pattern=r'^[a-f0-9]{40}$')]
    hand_urdf_sha256: Hash
    hand_kinematics_sha256: Hash
    anchor_sha256: Hash
    anchor_origin: Name
    anchor_evidence_sha256: Hash
    implementation_sha256: dict[Name, Hash]


class HandPresetValidation(Contract):
    scope: Literal['robot_kinematic_reference_only'] = 'robot_kinematic_reference_only'
    joint_limits: Literal['entire_amount_interval_by_convexity'] = 'entire_amount_interval_by_convexity'
    posture_coupling: Literal['entire_amount_interval_by_convexity'] = 'entire_amount_interval_by_convexity'
    fk: Literal['actual_side_urdf_at_requested_amount'] = 'actual_side_urdf_at_requested_amount'
    # Certified on the frozen robot-only family using a Lipschitz interval bound.
    # Virtual frame origins being separated says nothing about mesh clearance.
    all_amount_virtual_tip_separation_lower_bound_m: Positive = .0196
    collision_checked: Literal[False] = False
    trajectory_checked: Literal[False] = False
    native_grasp_qualified: Literal[False] = False
    advances_physics: Literal[False] = False
    mutates_controller: Literal[False] = False

    @model_validator(mode='before')
    @classmethod
    def false_claims_are_booleans(cls, value):
        if isinstance(value, dict):
            for key in ('collision_checked', 'trajectory_checked', 'native_grasp_qualified',
                        'advances_physics', 'mutates_controller'):
                if key in value and value[key] is not False:
                    raise ValueError(key + ' must be the literal boolean false')
        return value


class HandPresetResult(Contract):
    protocol: Literal['bench_hand_preset_v1'] = 'bench_hand_preset_v1'
    library_version: Literal['sharpa_w01_preposes_v1'] = 'sharpa_w01_preposes_v1'
    preset: HandPresetName
    side: HandSide
    amount: Amount
    frame_id: Name
    hand_posture_rad: JointMap = Field(description=(
        'All 22 joints of the selected hand, in radians; usable as an ordinary '
        'move_geometry hand_posture_rad target with separate motion/duration checks.'))
    fingertip_poses: dict[Finger, Pose]
    thumb_index_distance_m: Positive
    source: HandPresetSource
    validation: HandPresetValidation

    @model_validator(mode='after')
    def complete_hand_only(self):
        suffixes = {'thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA',
                    'thumb_IP', 'pinky_CMC'} | {
            f'{finger}_{joint}' for finger in ('index', 'middle', 'ring', 'pinky')
            for joint in ('MCP_FE', 'MCP_AA', 'PIP', 'DIP')}
        if set(self.hand_posture_rad) != {self.side + '_' + suffix for suffix in suffixes}:
            raise ValueError('A preset must contain exactly one side\'s 22 hand joints')
        if self.frame_id != self.side + '_hand_flange':
            raise ValueError('Preset frame must identify the requested hand')
        if set(self.fingertip_poses) != {'thumb', 'index', 'middle', 'ring', 'little'}:
            raise ValueError('All five fingertip poses are required')
        if any(pose.frame_id != self.frame_id for pose in self.fingertip_poses.values()):
            raise ValueError('Fingertip poses must use the requested hand frame')
        return self


class HandPresetDefinition(Contract):
    preset: HandPresetName
    meaning: str


class HandPresetAmountRange(Contract):
    minimum: Literal[0.] = 0.
    maximum: Literal[1.] = 1.
    default: Literal[1.] = 1.
    zero: Literal['open URDF zero configuration'] = 'open URDF zero configuration'
    unit: Literal['joint_space_interpolation_fraction'] = 'joint_space_interpolation_fraction'


class HandPresetCatalog(Contract):
    protocol: Literal['bench_hand_preset_catalog_v1'] = 'bench_hand_preset_catalog_v1'
    library_version: Literal['sharpa_w01_preposes_v1'] = 'sharpa_w01_preposes_v1'
    available: Annotated[bool, Field(strict=True)]
    reason: str | None = None
    sides: list[HandSide]
    presets: list[HandPresetDefinition]
    amount: HandPresetAmountRange = Field(default_factory=HandPresetAmountRange)
    unavailable_sides: dict[HandSide, str]
    unavailable_presets: dict[Name, str]
    validation: HandPresetValidation
    execution: str

    @model_validator(mode='after')
    def availability_is_explicit(self):
        if self.available != bool(self.sides) or self.available != bool(self.presets):
            raise ValueError('Preset availability must match the actual sides and entries')
        if not self.available and not self.reason:
            raise ValueError('Unavailable preset catalogs must give a reason')
        if len(set(self.sides)) != len(self.sides):
            raise ValueError('Preset catalog sides must be unique')
        if set(self.sides) & self.unavailable_sides.keys():
            raise ValueError('A side cannot be both available and unavailable')
        return self
