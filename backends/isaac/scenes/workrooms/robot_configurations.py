"""Explicit pre-episode installations for the reviewed single-agent task set.

The original task/release remains the default. A configuration is a separately
frozen physical installation, never a runtime robot swap inside an episode.
"""
from copy import deepcopy

SCHEMA = 'workroom_robot_configuration_release.v1'
VERSION = 'isaacsim-workroom-robot-configuration-release-v1'
CONFIGURATION_IDS = ('single_arm', 'dual_same_side', 'dual_opposed')
ELIGIBLE_TASKS = {
    'mechanical_assembly': ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
        'fastener_assembly', 'gear_train_assembly', 'nist_connector_insertion',
        'nist_belt_drive_assembly', 'nist_cable_routing', 'nist_wire_harness'),
    'laboratory': ('sample_sorting', 'vial_racking', 'cap_placement',
        'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer'),
    'study': ('stationery_sorting', 'book_stacking', 'book_retrieval',
        'book_shelving', 'drawer_retrieval', 'reading_stand'),
    'living_room': ('remote_operation', 'water_pouring', 'tower_building', 'storage_box'),
    'card_room': ('task0',),
}
LABELS = {'single_arm': '单臂', 'dual_same_side': '同侧双臂', 'dual_opposed': '对侧双臂'}
DESCRIPTIONS = {
    'single_arm': '原单臂与右灵巧手，保持原任务安装。',
    'dual_same_side': '一名 Agent 同时控制官方左、右灵巧手；两台机械臂位于工作桌同侧，可并排协作。',
    'dual_opposed': '一名 Agent 同时控制官方左、右灵巧手；两台机械臂位于工作桌相对两侧，可相向协作。',
}
SCENE_DESCRIPTIONS = {
    'mechanical_assembly': '机械装配室工作桌，任务零件、工装与固定桌边工具分区放置。',
    'laboratory': '实验室含主操作岛和后侧工作台；部分任务需要在两个台面间搬运器具。',
    'study': '书房书桌与后侧书架，部分任务包括取放书籍、开抽屉取物或展开阅读架。',
    'living_room': '住宅客厅茶几，任务物品在茶几上；包括遥控器、倒水器具、积木和带扣收纳盒。',
    'card_room': '棋牌室桌面上的三乘三棋盘与供棋托盘，单 Agent 将三枚棋子放入指定格。',
    'laundry_room': '洗衣房折衣台，任务为毛巾摊平、对折、叠放及单件T恤或长裤折叠；机器人安装在回合前选择。',
}


def eligible(family, task_id):
    return task_id in ELIGIBLE_TASKS.get(family, ())


def attach_robot_configurations(definition):
    """Attach only published pins bound to this exact original task release."""
    from .robot_configuration_catalog import RELEASES
    family, task_id = definition['family'], definition['task_id']
    if family == 'living_room' and definition.get('release_schema') == 'living_room_randomized_release.v1':
        from .living_room.randomization.catalog import RELEASES as living_releases
        if task_id not in living_releases:
            raise ValueError('Randomized living task has no published installation choices')
        row = living_releases[task_id]
    elif family == 'laundry_room':
        from .laundry_room.robot_configuration_catalog import RELEASES as laundry_releases
        if task_id not in laundry_releases:
            return definition
        from .laundry_room.rules import CURRENT_TASK_IDS
        if task_id not in CURRENT_TASK_IDS:
            raise ValueError('Historical laundry tasks do not gain current installation options')
        row = laundry_releases[task_id]
    else:
        if not eligible(family, task_id) or (family, task_id) not in RELEASES:
            return definition
        row = RELEASES[family, task_id]
    if row['base_release'] != definition.get('source_release'):
        raise ValueError('Robot configuration options belong to a different base release')
    result = deepcopy(definition)
    options = {'single_arm': {
        'source_release': deepcopy(definition['source_release']),
        'robot_requirements': deepcopy(row.get('single_arm_robot_requirements',
                                               definition['robot_requirements'])),
        'release_schema': definition['release_schema'],
        'release_schema_version': definition.get('release_schema_version'),
        'qualification': deepcopy(definition.get('qualification', {})),
        'layout': deepcopy(row.get('single_arm_layout', {})),
    }}
    for configuration_id in CONFIGURATION_IDS[1:]:
        options[configuration_id] = deepcopy(row['configurations'][configuration_id])
    default = row.get('default_configuration_id', 'single_arm')
    if family == 'laundry_room':
        if set(row['configurations']) != set(CONFIGURATION_IDS):
            raise ValueError('Laundry choices require three independently frozen installations')
        options = {key: deepcopy(row['configurations'][key]) for key in CONFIGURATION_IDS}
        if default not in options or options[default]['source_release'] != definition['source_release']:
            raise ValueError('Laundry default must match its exact published task release')
    for configuration_id, option in options.items():
        option.update(label_zh=LABELS[configuration_id],
                      description_zh=DESCRIPTIONS[configuration_id])
        if family == 'laundry_room' and configuration_id == 'single_arm':
            option['description_zh'] = '一名Agent控制官方右SHARPA与一台FR3，共29个关节；可单手重抓并借助台面整理衣物。'
    result.update(default_configuration_id=default, robot_configurations=options,
                  scene_description_zh=SCENE_DESCRIPTIONS[family])
    if family == 'laundry_room':
        for key in ('rules', 'robot_requirements', 'qualification', 'objects'):
            if key in options[default]:
                result[key] = deepcopy(options[default][key])
        result['layout'].update(deepcopy(options[default]['layout']))
        result['layout']['robot_installation'] = options[default]['layout']['robot_layout_id']
    return result


def configured_definition(definition, configuration_id=None):
    """Return a selected immutable installation without changing task identity."""
    if configuration_id is None:
        return deepcopy(definition)
    options = definition.get('robot_configurations', {})
    if configuration_id not in options:
        raise ValueError('Task does not publish the requested robot configuration')
    result, selected = deepcopy(definition), options[configuration_id]
    for key in ('source_release', 'robot_requirements', 'release_schema',
                'release_schema_version', 'qualification', 'rules', 'objects',
                'robot_profile', 'physical_instance_pair'):
        if key in selected:
            result[key] = deepcopy(selected[key])
    result['configuration_id'] = configuration_id
    result['layout'] = {**result['layout'], **deepcopy(selected.get('layout', {})),
                        'configuration_id': configuration_id}
    if result['family'] == 'laundry_room':
        result['layout']['robot_installation'] = result['layout']['robot_layout_id']
    return result
