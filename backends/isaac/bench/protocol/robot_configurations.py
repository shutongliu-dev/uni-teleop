"""Public, immutable robot installation choices made before episode creation."""
from copy import deepcopy
from typing import Literal

from pydantic import Field, FiniteFloat, model_serializer, model_validator

from .schemas import Contract, Hash, Name

CONFIGURATION_IDS = ('single_arm', 'dual_same_side', 'dual_opposed')
ConfigurationId = Literal['single_arm', 'dual_same_side', 'dual_opposed']
ROBOT_REQUIREMENT_KEYS = ('hand_id', 'assembly_version', 'articulations', 'active_sides')
PLACEMENT_KEYS = ('base_position_m', 'base_quaternion_wxyz', 'base_yaw_rad', 'hand_chirality')
LAYOUT_KEYS = ('robot_layout_id', 'reference_surface', 'frame', 'units', 'control_scope')


class PublicRobotPlacement(Contract):
    base_position_m: list[FiniteFloat] = Field(min_length=3, max_length=3)
    base_quaternion_wxyz: list[FiniteFloat] = Field(min_length=4, max_length=4)
    base_yaw_rad: FiniteFloat
    hand_chirality: Literal['left', 'right']

    @model_validator(mode='after')
    def normalized_orientation(self):
        if abs(sum(v*v for v in self.base_quaternion_wxyz)-1.) > 1.e-5:
            raise ValueError('Published robot base orientation must be normalized')
        return self


class PublicRobotLayout(Contract):
    robot_layout_id: Name
    placements: dict[Literal['left', 'right'], PublicRobotPlacement]
    reference_surface: str = Field(min_length=1, max_length=2048)
    frame: Literal['world']
    units: Literal['metres']
    control_scope: Literal['one_agent_29_dof', 'one_agent_58_dof']

    @model_validator(mode='after')
    def consistent_hands(self):
        if not self.placements or any(side != row.hand_chirality for side, row in self.placements.items()):
            raise ValueError('Published placement must preserve official hand chirality')
        return self


class PublicRobotRequirements(Contract):
    hand_id: Name
    assembly_version: Name
    articulations: int = Field(strict=True, ge=1, le=2)
    active_sides: list[Literal['left', 'right']] | None = None

    @model_serializer(mode='wrap')
    def omit_unset_sides(self, serializer):
        value = serializer(self)
        if self.active_sides is None:
            value.pop('active_sides', None)
        return value


class PublicRobotConfiguration(Contract):
    configuration_id: ConfigurationId
    label_zh: str = Field(min_length=1, max_length=256)
    description_zh: str = Field(min_length=1, max_length=4096)
    task_release_sha256: Hash
    robot_requirements: PublicRobotRequirements
    layout: PublicRobotLayout | None = None

    @model_serializer(mode='wrap')
    def omit_absent_layout(self, serializer):
        value = serializer(self)
        if self.layout is None:
            value.pop('layout', None)
        return value

    @model_validator(mode='after')
    def consistent_installation(self):
        robot = self.robot_requirements
        if ((self.configuration_id == 'single_arm') != (robot.articulations == 1)
                or (self.configuration_id != 'single_arm' and
                    (robot.hand_id != 'dual_fr3_sharpa_w01' or robot.active_sides != ['left', 'right']))):
            raise ValueError('Configuration robot identity disagrees with its installation')
        if self.layout is not None:
            dual = self.configuration_id != 'single_arm'
            if (set(self.layout.placements) != ({'left', 'right'} if dual else {'right'})
                    or self.layout.control_scope != ('one_agent_58_dof' if dual else 'one_agent_29_dof')):
                raise ValueError('Public layout disagrees with the configured hand inventory')
        return self


def project_robot_configurations(options):
    """Project trusted catalog configuration metadata, never paths or geometry."""
    if not isinstance(options, dict) or not options:
        raise ValueError('Robot configurations require a nonempty mapping')
    result = {}
    for key, option in options.items():
        value = {
            'configuration_id': key,
            'label_zh': option['label_zh'], 'description_zh': option['description_zh'],
            'task_release_sha256': option['source_release']['manifest_sha256'],
            'robot_requirements': {name: deepcopy(option['robot_requirements'][name])
                for name in ROBOT_REQUIREMENT_KEYS if name in option['robot_requirements']},
        }
        if option.get('layout') is not None:
            layout = option['layout']
            value['layout'] = {name: deepcopy(layout[name]) for name in LAYOUT_KEYS}
            value['layout']['placements'] = {side: {name: deepcopy(row[name]) for name in PLACEMENT_KEYS}
                                              for side, row in layout['placements'].items()}
        row = PublicRobotConfiguration.model_validate(value)
        result[key] = row.model_dump(mode='json')
    return result


def checked_public_configurations(options):
    if not isinstance(options, dict) or not options:
        raise ValueError('Missing public robot configurations')
    rows = {key: PublicRobotConfiguration.model_validate(value).model_dump(mode='json')
            for key, value in options.items()}
    if any(row['configuration_id'] != key for key, row in rows.items()):
        raise ValueError('Robot configuration ID differs from its mapping key')
    return rows
