"""Published reset inputs, independent of development validation results.

Every row binds three immutable installations, each with seeds 0 through 9.
The tower tasks deliberately retain their existing catalog releases.
"""
from copy import deepcopy

from bench.paths import DATA_ROOT


SCHEMA = 'living_room_randomized_release.v1'
VERSION = 'isaacsim-living-room-randomized-release-v1'
TASK_IDS = ('remote_operation', 'water_pouring', 'storage_box')
CONFIGURATIONS = ('single_arm', 'dual_same_side', 'dual_opposed')

# Filled only with reviewed, immutable input manifests. Never add test reports.
RELEASES = {'remote_operation': {'family': 'living_room',
                      'task_id': 'remote_operation',
                      'default_configuration_id': 'single_arm',
                      'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/initial_randomization_v1/single_arm',
                                       'manifest_sha256': '2b043e747e73fa709665f672e62e5b4bbb904f1d55d04b1b7bbcb7a19f82c361'},
                      'single_arm_robot_requirements': {'arm': 'FR3',
                                                        'hand_id': 'sharpa_w01_right',
                                                        'assembly_version': 'isaacsim_6_1_v2',
                                                        'articulations': 1,
                                                        'active_sides': ['right'],
                                                        'configuration_module': 'scenes.workrooms.living_room.dual_robot_layout',
                                                        'configuration_factory': 'robot_configurations',
                                                        'base_positions_m': {'front': [0.65,
                                                                                       -0.76,
                                                                                       0.78]},
                                                        'base_constraint': 'native_fixed_base_v1',
                                                        'geometry_scale': [1.0, 1.0, 1.0],
                                                        'ownership': 'One agent per active arm; '
                                                                     'shared bench controls and '
                                                                     'original source joints.',
                                                        'qualification': 'Existing '
                                                                         'ready-pose/endpoint '
                                                                         'checks are not '
                                                                         'task-motion '
                                                                         'qualification.'},
                      'single_arm_layout': {'robot_layout_id': 'living_room_remote_operation_original_single_arm',
                                            'placements': {'right': {'base_position_m': [0.65,
                                                                                         -0.76,
                                                                                         0.78],
                                                                     'base_quaternion_wxyz': [0.7071067811865476,
                                                                                              0.0,
                                                                                              0.0,
                                                                                              0.7071067811865476],
                                                                     'base_yaw_rad': 1.5707963267948968,
                                                                     'hand_chirality': 'right'}},
                                            'reference_surface': 'original fixed task installation',
                                            'frame': 'world',
                                            'units': 'metres',
                                            'control_scope': 'one_agent_29_dof'},
                      'configurations': {'single_arm': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/initial_randomization_v1/single_arm',
                                                                           'manifest_sha256': '2b043e747e73fa709665f672e62e5b4bbb904f1d55d04b1b7bbcb7a19f82c361'},
                                                        'release_schema': 'living_room_randomized_release.v1',
                                                        'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                        'robot_requirements': {'arm': 'FR3',
                                                                               'hand_id': 'sharpa_w01_right',
                                                                               'assembly_version': 'isaacsim_6_1_v2',
                                                                               'articulations': 1,
                                                                               'active_sides': ['right'],
                                                                               'configuration_module': 'scenes.workrooms.living_room.dual_robot_layout',
                                                                               'configuration_factory': 'robot_configurations',
                                                                               'base_positions_m': {'front': [0.65,
                                                                                                              -0.76,
                                                                                                              0.78]},
                                                                               'base_constraint': 'native_fixed_base_v1',
                                                                               'geometry_scale': [1.0,
                                                                                                  1.0,
                                                                                                  1.0],
                                                                               'ownership': 'One '
                                                                                            'agent '
                                                                                            'per '
                                                                                            'active '
                                                                                            'arm; '
                                                                                            'shared '
                                                                                            'bench '
                                                                                            'controls '
                                                                                            'and '
                                                                                            'original '
                                                                                            'source '
                                                                                            'joints.',
                                                                               'qualification': 'Existing '
                                                                                                'ready-pose/endpoint '
                                                                                                'checks '
                                                                                                'are '
                                                                                                'not '
                                                                                                'task-motion '
                                                                                                'qualification.'},
                                                        'layout': {'robot_layout_id': 'living_room_remote_operation_original_single_arm',
                                                                   'placements': {'right': {'base_position_m': [0.65,
                                                                                                                -0.76,
                                                                                                                0.78],
                                                                                            'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                     0.0,
                                                                                                                     0.0,
                                                                                                                     0.7071067811865476],
                                                                                            'base_yaw_rad': 1.5707963267948968,
                                                                                            'hand_chirality': 'right'}},
                                                                   'reference_surface': 'original '
                                                                                        'fixed '
                                                                                        'task '
                                                                                        'installation',
                                                                   'frame': 'world',
                                                                   'units': 'metres',
                                                                   'control_scope': 'one_agent_29_dof'},
                                                        'objects': {'intended_roles': ['free_remote',
                                                                                       'two_passive_buttons',
                                                                                       'remote_holder',
                                                                                       'television'],
                                                                    'asset_bindings': 'Exact '
                                                                                      'per-seed '
                                                                                      'USD, '
                                                                                      'texture, '
                                                                                      'collider '
                                                                                      'and mass '
                                                                                      'inputs are '
                                                                                      'pinned in '
                                                                                      'the '
                                                                                      'release.'},
                                                        'rules': {'intent': '拿起正面朝上的遥控器开启电视，显示10秒倒计时后自动关机，再归架松手稳定。',
                                                                  'implementation': 'scenes.workrooms.living_room.tasks.remote_operation.rules',
                                                                  'contract': {'asset_freeze_required': ['Remote/button '
                                                                                                         'collision '
                                                                                                         'geometry '
                                                                                                         'and '
                                                                                                         'masses',
                                                                                                         'Button '
                                                                                                         'travel, '
                                                                                                         'spring '
                                                                                                         'and '
                                                                                                         'damping',
                                                                                                         'Press/release '
                                                                                                         'hysteresis',
                                                                                                         'Holder '
                                                                                                         'fit',
                                                                                                         'Stable '
                                                                                                         'speed/contact/penetration '
                                                                                                         'tolerances'],
                                                                               'budget_exhaustion': 'incomplete',
                                                                               'buttons': {'ids': ['power',
                                                                                                   'confirm'],
                                                                                           'mechanism': 'Independent '
                                                                                                        'passive '
                                                                                                        'spring-return '
                                                                                                        'prismatic '
                                                                                                        'buttons.',
                                                                                           'no_credit': ['Color '
                                                                                                         'change '
                                                                                                         'without '
                                                                                                         'contact',
                                                                                                         'One '
                                                                                                         'long '
                                                                                                         'hold '
                                                                                                         'counted '
                                                                                                         'repeatedly',
                                                                                                         'Remote '
                                                                                                         'supported '
                                                                                                         'by '
                                                                                                         'table '
                                                                                                         'or '
                                                                                                         'holder '
                                                                                                         'during '
                                                                                                         'credited '
                                                                                                         'press',
                                                                                                         'Agent '
                                                                                                         'claims '
                                                                                                         'that '
                                                                                                         'a '
                                                                                                         'button '
                                                                                                         'was '
                                                                                                         'pressed'],
                                                                                           'press_evidence': ['Physical '
                                                                                                              'finger/button '
                                                                                                              'contact.',
                                                                                                              'Actual '
                                                                                                              'button '
                                                                                                              'joint '
                                                                                                              'crosses '
                                                                                                              'the '
                                                                                                              'asset-bound '
                                                                                                              'press '
                                                                                                              'displacement.',
                                                                                                              'Remote '
                                                                                                              'is '
                                                                                                              'off '
                                                                                                              'the '
                                                                                                              'table/holder '
                                                                                                              'and '
                                                                                                              'genuinely '
                                                                                                              'held '
                                                                                                              'by '
                                                                                                              'the '
                                                                                                              'same '
                                                                                                              'hand.',
                                                                                                              'Button '
                                                                                                              'physically '
                                                                                                              'returns '
                                                                                                              'to '
                                                                                                              'or '
                                                                                                              'above '
                                                                                                              'its '
                                                                                                              'release '
                                                                                                              'position '
                                                                                                              'before '
                                                                                                              'a '
                                                                                                              'new '
                                                                                                              'press.']},
                                                                               'competitive': False,
                                                                               'evidence_fault': 'invalid_no_score',
                                                                               'evidence_scope': 'Private '
                                                                                                 'native '
                                                                                                 'body/joint/contact '
                                                                                                 'samples '
                                                                                                 'only. '
                                                                                                 'CPU '
                                                                                                 'replay '
                                                                                                 'verifies '
                                                                                                 'logic; '
                                                                                                 'independent '
                                                                                                 'mechanical '
                                                                                                 'probes '
                                                                                                 'do '
                                                                                                 'not '
                                                                                                 'establish '
                                                                                                 'same-hand '
                                                                                                 'robot '
                                                                                                 'success.',
                                                                               'failure': ['Remote '
                                                                                           'leaves '
                                                                                           'the '
                                                                                           'permitted '
                                                                                           'workspace '
                                                                                           'or '
                                                                                           'falls '
                                                                                           'to the '
                                                                                           'room '
                                                                                           'floor.'],
                                                                               'instruction_zh': '拿起正面朝上的遥控器，用同一只手悬空握持遥控器，完整按下、松开红色电源键开启电视；电视显示10秒倒计时，到时自动关机。最后将遥控器按正确朝向放回收纳架，松手稳定至少0.300秒。',
                                                                               'qualification': 'Candidate '
                                                                                                'physical '
                                                                                                'buttons '
                                                                                                'have '
                                                                                                'independent '
                                                                                                'native '
                                                                                                'mechanism '
                                                                                                'evidence. '
                                                                                                'The '
                                                                                                'read-only '
                                                                                                'observer/scorer '
                                                                                                'is '
                                                                                                'implemented; '
                                                                                                'native '
                                                                                                'same-hand '
                                                                                                'robot '
                                                                                                'task '
                                                                                                'execution '
                                                                                                'is '
                                                                                                'not '
                                                                                                'qualified.',
                                                                               'randomization': ['遥控器正面朝上，物理初态固定；10秒倒计时和归架目标不随seed改变。'],
                                                                               'recovery': {'early_power_off': 'No '
                                                                                                               'success; '
                                                                                                               'turn '
                                                                                                               'on '
                                                                                                               'again '
                                                                                                               'and '
                                                                                                               'restart '
                                                                                                               'the '
                                                                                                               'continuous '
                                                                                                               '10-second '
                                                                                                               'interval.',
                                                                                            'regrasp': True,
                                                                                            'table_assisted_flip': True,
                                                                                            'table_drop': 'Recover '
                                                                                                          'by '
                                                                                                          'actual '
                                                                                                          'regrasp.'},
                                                                               'referee': {'instruction_zh': '拿起正面朝上的遥控器，用同一只手悬空握持遥控器，完整按下、松开红色电源键开启电视；电视显示10秒倒计时，到时自动关机。最后将遥控器按正确朝向放回收纳架，松手稳定至少0.300秒。',
                                                                                           'observation_scope': '公开静态规则与粗粒度裁判阶段；逐原生子步位姿、接触和连续计时由私有裁判核验。',
                                                                                           'outcomes': {'budget_exhaustion': '运行器结束回合，未完成不制造成功。',
                                                                                                        'cancelled': '取消或运行故障独立记录，不制造任务成功或物理失败。',
                                                                                                        'evaluator_invalid': '缺失、矛盾或不连续原生证据，无有效分数。',
                                                                                                        'ongoing': '尚未完成，可在预算内继续。',
                                                                                                        'safety_stop': '物理安全中止单独记录。',
                                                                                                        'success': '有效物理按键开启电视、完成10秒倒计时自动关机及归架松手稳定。',
                                                                                                        'task_failure': '有效证据确认遥控器落地或离开允许操作区域。'},
                                                                                           'phases': {'await_power_on': '电视已关闭，等待一次有效的电源键完整按压和释放。',
                                                                                                      'await_return': '倒计时已完成并自动关机，等待遥控器归架、松手稳定。',
                                                                                                      'budget_exhausted': '预算结束，未取得本次成功判定。',
                                                                                                      'completed': '已完成开机、连续开机、关机和遥控器归架松手稳定。',
                                                                                                      'evidence_invalid': '原生物理证据无效。',
                                                                                                      'holding_power_on': '电视显示10秒倒计时，结束自动关机。',
                                                                                                      'task_failure': '遥控器落地或离开允许操作区域。'},
                                                                                           'physical_actions': '仅通过机器人真实接触操作遥控器；长按不能重复计数，桌面或收纳架支撑中的按压不计。',
                                                                                           'reason_descriptions': {'await_power_on': '电视已关闭，等待一次有效的电源键完整按压和释放。',
                                                                                                                   'await_return': '倒计时已完成并自动关机，等待遥控器归架、松手稳定。',
                                                                                                                   'budget_exhausted': '预算结束，未取得本次成功判定。',
                                                                                                                   'completed': '已完成开机、连续开机、关机和遥控器归架松手稳定。',
                                                                                                                   'evidence_invalid': '原生物理证据无效。',
                                                                                                                   'holding_power_on': '电视显示10秒倒计时，结束自动关机。',
                                                                                                                   'task_failure': '遥控器落地或离开允许操作区域。'},
                                                                                           'recovery': ['允许翻正、重新抓取和重试。',
                                                                                                        '提前关机不成功；下次开机重新计算连续10秒，多次短开机时间不累计。'],
                                                                                           'schema': 'physical_task_referee_contract.v1',
                                                                                           'success_conditions': ['电视初始关闭；每次有效电源键完整按压并真实释放后只切换一次电视电源。',
                                                                                                                  '用同一只手悬空握持遥控器，以真实手指接触按压实体电源键，并让按键松开回弹。',
                                                                                                                  '电视连续开机10秒，屏幕倒计时结束后自动关机；提前手动关机可重试。',
                                                                                                                  '遥控器整体按正确朝向放入收纳架，由架子真实支撑；手与遥控器、按键和架子均无接触。',
                                                                                                                  '遥控器及按键全部释放并连续稳定至少0.300秒，才取得成功。',
                                                                                                                  '自动关机保留本次完成周期资格，仍需归架松手稳定；确认键不影响电视。'],
                                                                                           'task_id': 'remote_operation',
                                                                                           'thresholds': {'held_before_press_s': 0.02,
                                                                                                          'max_actions': 6000,
                                                                                                          'max_time_s': 120.0,
                                                                                                          'press_position_m': -0.0014,
                                                                                                          'release_position_m': -0.0003,
                                                                                                          'required_on_s': 10.0,
                                                                                                          'stable_s': 0.3}},
                                                                               'schema': 'living_room_remote_operation_rules_v4',
                                                                               'stages': ['power_on',
                                                                                          'countdown',
                                                                                          'automatic_power_off',
                                                                                          'return_to_holder'],
                                                                               'status': 'implemented_reference',
                                                                               'success': {'all_of': ['A '
                                                                                                      'valid '
                                                                                                      'handheld '
                                                                                                      'physical '
                                                                                                      'power '
                                                                                                      'press-and-release '
                                                                                                      'cycle '
                                                                                                      'turns '
                                                                                                      'the '
                                                                                                      'initially '
                                                                                                      'off '
                                                                                                      'television '
                                                                                                      'on.',
                                                                                                      'The '
                                                                                                      'television '
                                                                                                      'remains '
                                                                                                      'continuously '
                                                                                                      'on '
                                                                                                      'for '
                                                                                                      'at '
                                                                                                      'least '
                                                                                                      '10 '
                                                                                                      'native '
                                                                                                      'simulation '
                                                                                                      'seconds.',
                                                                                                      'The '
                                                                                                      'native '
                                                                                                      '10-second '
                                                                                                      'countdown '
                                                                                                      'automatically '
                                                                                                      'turns '
                                                                                                      'the '
                                                                                                      'television '
                                                                                                      'off; '
                                                                                                      'keep '
                                                                                                      'this '
                                                                                                      'completed-cycle '
                                                                                                      'credit '
                                                                                                      'while '
                                                                                                      'returning '
                                                                                                      'the '
                                                                                                      'remote.',
                                                                                                      'Whole '
                                                                                                      'remote '
                                                                                                      'is '
                                                                                                      'within '
                                                                                                      'the '
                                                                                                      'correctly '
                                                                                                      'oriented '
                                                                                                      'holder '
                                                                                                      'acceptance '
                                                                                                      'volume.',
                                                                                                      'Holder '
                                                                                                      'physically '
                                                                                                      'supports '
                                                                                                      'the '
                                                                                                      'remote; '
                                                                                                      'all '
                                                                                                      'hand '
                                                                                                      'contacts '
                                                                                                      'have '
                                                                                                      'ended.',
                                                                                                      'Remote '
                                                                                                      'and '
                                                                                                      'all '
                                                                                                      'buttons '
                                                                                                      'retain '
                                                                                                      'released '
                                                                                                      'supported '
                                                                                                      'stability '
                                                                                                      'for '
                                                                                                      '0.300 '
                                                                                                      'seconds.'],
                                                                                           'automatic_power_off': True,
                                                                                           'continuous_on_resets_when_off': True,
                                                                                           'holder_return_required': True,
                                                                                           'stable_duration_s': 0.3},
                                                                               'television': {'automatic_power_off': True,
                                                                                              'confirm_button': 'No '
                                                                                                                'effect '
                                                                                                                'on '
                                                                                                                'television '
                                                                                                                'power '
                                                                                                                'or '
                                                                                                                'task '
                                                                                                                'progress.',
                                                                                              'countdown': 'Visible '
                                                                                                           'native-simulation-time '
                                                                                                           'countdown; '
                                                                                                           'automatically '
                                                                                                           'off '
                                                                                                           'after '
                                                                                                           'one '
                                                                                                           'uninterrupted '
                                                                                                           '10-second '
                                                                                                           'interval.',
                                                                                              'initial_power_on': False,
                                                                                              'initial_remote_face_up': True,
                                                                                              'required_on_s': 10.0,
                                                                                              'time_basis': 'Consecutive '
                                                                                                            'native '
                                                                                                            'simulation '
                                                                                                            'time; '
                                                                                                            'wall '
                                                                                                            'time '
                                                                                                            'and '
                                                                                                            'separate '
                                                                                                            'short '
                                                                                                            'on '
                                                                                                            'periods '
                                                                                                            'do '
                                                                                                            'not '
                                                                                                            'count.',
                                                                                              'toggle_button': 'power',
                                                                                              'toggle_on': 'One '
                                                                                                           'valid '
                                                                                                           'physical '
                                                                                                           'power '
                                                                                                           'press-and-release '
                                                                                                           'cycle, '
                                                                                                           'at '
                                                                                                           'its '
                                                                                                           'native '
                                                                                                           'release '
                                                                                                           'time.'},
                                                                               'thresholds': {'held_before_press_s': 0.02,
                                                                                              'holder_bottom_tolerance_m': 0.002,
                                                                                              'holder_orientation_tolerance_rad': 0.15,
                                                                                              'holder_xy_tolerance_m': 0.001,
                                                                                              'joint_lower_m': -0.002,
                                                                                              'joint_tolerance_m': 0.0002,
                                                                                              'joint_upper_m': 0.0,
                                                                                              'max_actions': 6000,
                                                                                              'max_time_s': 120.0,
                                                                                              'maximum_angular_speed_rad_s': 0.15,
                                                                                              'maximum_linear_speed_m_s': 0.02,
                                                                                              'maximum_penetration_m': 0.002,
                                                                                              'maximum_stable_orientation_range_rad': 0.025,
                                                                                              'maximum_stable_position_range_m': 0.001,
                                                                                              'minimum_contact_impulse_ns': 1e-10,
                                                                                              'minimum_handheld_clearance_m': 0.005,
                                                                                              'press_position_m': -0.0014,
                                                                                              'release_position_m': -0.0003,
                                                                                              'stable_s': 0.3}}},
                                                        'qualification': {'status': 'prepared_unvalidated',
                                                                          'native_preflight': False,
                                                                          'manipulation': False,
                                                                          'physics_accepted': False,
                                                                          'public_http_smoke': False,
                                                                          'public_functions_smoke': False,
                                                                          'scope': 'Frozen '
                                                                                   'randomized '
                                                                                   'inputs; '
                                                                                   'separately '
                                                                                   'recorded '
                                                                                   'development '
                                                                                   'checks are not '
                                                                                   'runtime '
                                                                                   'admission '
                                                                                   'data.'}},
                                         'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/initial_randomization_v1/dual_same_side',
                                                                               'manifest_sha256': '65c96895af421b3852f6c8640c272f78b4ab5ce7173fe4d07470ec3825a6324e'},
                                                            'release_schema': 'living_room_randomized_release.v1',
                                                            'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                            'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                   'assembly_version': 'isaacsim_6_1_v2',
                                                                                   'articulations': 2,
                                                                                   'active_sides': ['left',
                                                                                                    'right']},
                                                            'layout': {'robot_layout_id': 'living_room_dual_same_side_left_right_sharpa_v1',
                                                                       'placements': {'left': {'base_position_m': [0.35,
                                                                                                                   -0.68,
                                                                                                                   0.78],
                                                                                               'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                        0.0,
                                                                                                                        0.0,
                                                                                                                        0.7071067811865475],
                                                                                               'base_yaw_rad': 1.5707963267948966,
                                                                                               'hand_chirality': 'left'},
                                                                                      'right': {'base_position_m': [0.95,
                                                                                                                    -0.68,
                                                                                                                    0.78],
                                                                                                'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                         0.0,
                                                                                                                         0.0,
                                                                                                                         0.7071067811865475],
                                                                                                'base_yaw_rad': 1.5707963267948966,
                                                                                                'hand_chirality': 'right'}},
                                                                       'reference_surface': 'fixed '
                                                                                            'room '
                                                                                            'task '
                                                                                            'table',
                                                                       'frame': 'world',
                                                                       'units': 'metres',
                                                                       'control_scope': 'one_agent_58_dof'},
                                                            'objects': {'intended_roles': ['free_remote',
                                                                                           'two_passive_buttons',
                                                                                           'remote_holder',
                                                                                           'television'],
                                                                        'asset_bindings': 'Exact '
                                                                                          'per-seed '
                                                                                          'USD, '
                                                                                          'texture, '
                                                                                          'collider '
                                                                                          'and '
                                                                                          'mass '
                                                                                          'inputs '
                                                                                          'are '
                                                                                          'pinned '
                                                                                          'in the '
                                                                                          'release.'},
                                                            'rules': {'intent': '拿起正面朝上的遥控器开启电视，显示10秒倒计时后自动关机，再归架松手稳定。',
                                                                      'implementation': 'scenes.workrooms.living_room.tasks.remote_operation.rules',
                                                                      'contract': {'asset_freeze_required': ['Remote/button '
                                                                                                             'collision '
                                                                                                             'geometry '
                                                                                                             'and '
                                                                                                             'masses',
                                                                                                             'Button '
                                                                                                             'travel, '
                                                                                                             'spring '
                                                                                                             'and '
                                                                                                             'damping',
                                                                                                             'Press/release '
                                                                                                             'hysteresis',
                                                                                                             'Holder '
                                                                                                             'fit',
                                                                                                             'Stable '
                                                                                                             'speed/contact/penetration '
                                                                                                             'tolerances'],
                                                                                   'budget_exhaustion': 'incomplete',
                                                                                   'buttons': {'ids': ['power',
                                                                                                       'confirm'],
                                                                                               'mechanism': 'Independent '
                                                                                                            'passive '
                                                                                                            'spring-return '
                                                                                                            'prismatic '
                                                                                                            'buttons.',
                                                                                               'no_credit': ['Color '
                                                                                                             'change '
                                                                                                             'without '
                                                                                                             'contact',
                                                                                                             'One '
                                                                                                             'long '
                                                                                                             'hold '
                                                                                                             'counted '
                                                                                                             'repeatedly',
                                                                                                             'Remote '
                                                                                                             'supported '
                                                                                                             'by '
                                                                                                             'table '
                                                                                                             'or '
                                                                                                             'holder '
                                                                                                             'during '
                                                                                                             'credited '
                                                                                                             'press',
                                                                                                             'Agent '
                                                                                                             'claims '
                                                                                                             'that '
                                                                                                             'a '
                                                                                                             'button '
                                                                                                             'was '
                                                                                                             'pressed'],
                                                                                               'press_evidence': ['Physical '
                                                                                                                  'finger/button '
                                                                                                                  'contact.',
                                                                                                                  'Actual '
                                                                                                                  'button '
                                                                                                                  'joint '
                                                                                                                  'crosses '
                                                                                                                  'the '
                                                                                                                  'asset-bound '
                                                                                                                  'press '
                                                                                                                  'displacement.',
                                                                                                                  'Remote '
                                                                                                                  'is '
                                                                                                                  'off '
                                                                                                                  'the '
                                                                                                                  'table/holder '
                                                                                                                  'and '
                                                                                                                  'genuinely '
                                                                                                                  'held '
                                                                                                                  'by '
                                                                                                                  'one '
                                                                                                                  'or '
                                                                                                                  'both '
                                                                                                                  'declared '
                                                                                                                  'hands.',
                                                                                                                  'Button '
                                                                                                                  'physically '
                                                                                                                  'returns '
                                                                                                                  'to '
                                                                                                                  'or '
                                                                                                                  'above '
                                                                                                                  'its '
                                                                                                                  'release '
                                                                                                                  'position '
                                                                                                                  'before '
                                                                                                                  'a '
                                                                                                                  'new '
                                                                                                                  'press.']},
                                                                                   'competitive': False,
                                                                                   'evidence_fault': 'invalid_no_score',
                                                                                   'evidence_scope': 'Private '
                                                                                                     'native '
                                                                                                     'body/joint/contact '
                                                                                                     'samples '
                                                                                                     'only. '
                                                                                                     'CPU '
                                                                                                     'replay '
                                                                                                     'verifies '
                                                                                                     'logic; '
                                                                                                     'independent '
                                                                                                     'mechanical '
                                                                                                     'probes '
                                                                                                     'do '
                                                                                                     'not '
                                                                                                     'establish '
                                                                                                     'bimanual '
                                                                                                     'robot '
                                                                                                     'success.',
                                                                                   'failure': ['Remote '
                                                                                               'leaves '
                                                                                               'the '
                                                                                               'permitted '
                                                                                               'workspace '
                                                                                               'or '
                                                                                               'falls '
                                                                                               'to '
                                                                                               'the '
                                                                                               'room '
                                                                                               'floor.'],
                                                                                   'instruction_zh': '拿起正面朝上的遥控器，可用左右手协作悬空握持遥控器，完整按下、松开红色电源键开启电视；电视显示10秒倒计时，到时自动关机。最后将遥控器按正确朝向放回收纳架，松手稳定至少0.300秒。',
                                                                                   'qualification': 'Candidate '
                                                                                                    'physical '
                                                                                                    'buttons '
                                                                                                    'have '
                                                                                                    'independent '
                                                                                                    'native '
                                                                                                    'mechanism '
                                                                                                    'evidence. '
                                                                                                    'The '
                                                                                                    'read-only '
                                                                                                    'observer/scorer '
                                                                                                    'is '
                                                                                                    'implemented; '
                                                                                                    'native '
                                                                                                    'bimanual '
                                                                                                    'robot '
                                                                                                    'task '
                                                                                                    'execution '
                                                                                                    'is '
                                                                                                    'not '
                                                                                                    'qualified.',
                                                                                   'randomization': ['遥控器正面朝上，物理初态固定；10秒倒计时和归架目标不随seed改变。'],
                                                                                   'recovery': {'early_power_off': 'No '
                                                                                                                   'success; '
                                                                                                                   'turn '
                                                                                                                   'on '
                                                                                                                   'again '
                                                                                                                   'and '
                                                                                                                   'restart '
                                                                                                                   'the '
                                                                                                                   'continuous '
                                                                                                                   '10-second '
                                                                                                                   'interval.',
                                                                                                'regrasp': True,
                                                                                                'table_assisted_flip': True,
                                                                                                'table_drop': 'Recover '
                                                                                                              'by '
                                                                                                              'actual '
                                                                                                              'regrasp.'},
                                                                                   'referee': {'instruction_zh': '拿起正面朝上的遥控器，可用左右手协作悬空握持遥控器，完整按下、松开红色电源键开启电视；电视显示10秒倒计时，到时自动关机。最后将遥控器按正确朝向放回收纳架，松手稳定至少0.300秒。',
                                                                                               'observation_scope': '公开静态规则与粗粒度裁判阶段；逐原生子步位姿、接触和连续计时由私有裁判核验。',
                                                                                               'outcomes': {'budget_exhaustion': '运行器结束回合，未完成不制造成功。',
                                                                                                            'cancelled': '取消或运行故障独立记录，不制造任务成功或物理失败。',
                                                                                                            'evaluator_invalid': '缺失、矛盾或不连续原生证据，无有效分数。',
                                                                                                            'ongoing': '尚未完成，可在预算内继续。',
                                                                                                            'safety_stop': '物理安全中止单独记录。',
                                                                                                            'success': '有效物理按键开启电视、完成10秒倒计时自动关机及归架松手稳定。',
                                                                                                            'task_failure': '有效证据确认遥控器落地或离开允许操作区域。'},
                                                                                               'phases': {'await_power_on': '电视已关闭，等待一次有效的电源键完整按压和释放。',
                                                                                                          'await_return': '倒计时已完成并自动关机，等待遥控器归架、松手稳定。',
                                                                                                          'budget_exhausted': '预算结束，未取得本次成功判定。',
                                                                                                          'completed': '已完成开机、连续开机、关机和遥控器归架松手稳定。',
                                                                                                          'evidence_invalid': '原生物理证据无效。',
                                                                                                          'holding_power_on': '电视显示10秒倒计时，结束自动关机。',
                                                                                                          'task_failure': '遥控器落地或离开允许操作区域。'},
                                                                                               'physical_actions': '仅通过机器人真实接触操作遥控器；长按不能重复计数，桌面或收纳架支撑中的按压不计。',
                                                                                               'reason_descriptions': {'await_power_on': '电视已关闭，等待一次有效的电源键完整按压和释放。',
                                                                                                                       'await_return': '倒计时已完成并自动关机，等待遥控器归架、松手稳定。',
                                                                                                                       'budget_exhausted': '预算结束，未取得本次成功判定。',
                                                                                                                       'completed': '已完成开机、连续开机、关机和遥控器归架松手稳定。',
                                                                                                                       'evidence_invalid': '原生物理证据无效。',
                                                                                                                       'holding_power_on': '电视显示10秒倒计时，结束自动关机。',
                                                                                                                       'task_failure': '遥控器落地或离开允许操作区域。'},
                                                                                               'recovery': ['允许翻正、重新抓取和重试。',
                                                                                                            '提前关机不成功；下次开机重新计算连续10秒，多次短开机时间不累计。'],
                                                                                               'schema': 'physical_task_referee_contract.v1',
                                                                                               'success_conditions': ['电视初始关闭；每次有效电源键完整按压并真实释放后只切换一次电视电源。',
                                                                                                                      '可用左右手协作悬空握持遥控器，以真实手指接触按压实体电源键，并让按键松开回弹。',
                                                                                                                      '电视连续开机10秒，屏幕倒计时结束后自动关机；提前手动关机可重试。',
                                                                                                                      '遥控器整体按正确朝向放入收纳架，由架子真实支撑；手与遥控器、按键和架子均无接触。',
                                                                                                                      '遥控器及按键全部释放并连续稳定至少0.300秒，才取得成功。',
                                                                                                                      '自动关机保留本次完成周期资格，仍需归架松手稳定；确认键不影响电视。'],
                                                                                               'task_id': 'remote_operation',
                                                                                               'thresholds': {'held_before_press_s': 0.02,
                                                                                                              'max_actions': 6000,
                                                                                                              'max_time_s': 120.0,
                                                                                                              'press_position_m': -0.0014,
                                                                                                              'release_position_m': -0.0003,
                                                                                                              'required_on_s': 10.0,
                                                                                                              'stable_s': 0.3}},
                                                                                   'schema': 'living_room_remote_operation_rules_v4',
                                                                                   'stages': ['power_on',
                                                                                              'countdown',
                                                                                              'automatic_power_off',
                                                                                              'return_to_holder'],
                                                                                   'status': 'implemented_reference',
                                                                                   'success': {'all_of': ['A '
                                                                                                          'valid '
                                                                                                          'handheld '
                                                                                                          'physical '
                                                                                                          'power '
                                                                                                          'press-and-release '
                                                                                                          'cycle '
                                                                                                          'turns '
                                                                                                          'the '
                                                                                                          'initially '
                                                                                                          'off '
                                                                                                          'television '
                                                                                                          'on.',
                                                                                                          'The '
                                                                                                          'television '
                                                                                                          'remains '
                                                                                                          'continuously '
                                                                                                          'on '
                                                                                                          'for '
                                                                                                          'at '
                                                                                                          'least '
                                                                                                          '10 '
                                                                                                          'native '
                                                                                                          'simulation '
                                                                                                          'seconds.',
                                                                                                          'The '
                                                                                                          'native '
                                                                                                          '10-second '
                                                                                                          'countdown '
                                                                                                          'automatically '
                                                                                                          'turns '
                                                                                                          'the '
                                                                                                          'television '
                                                                                                          'off; '
                                                                                                          'keep '
                                                                                                          'this '
                                                                                                          'completed-cycle '
                                                                                                          'credit '
                                                                                                          'while '
                                                                                                          'returning '
                                                                                                          'the '
                                                                                                          'remote.',
                                                                                                          'Whole '
                                                                                                          'remote '
                                                                                                          'is '
                                                                                                          'within '
                                                                                                          'the '
                                                                                                          'correctly '
                                                                                                          'oriented '
                                                                                                          'holder '
                                                                                                          'acceptance '
                                                                                                          'volume.',
                                                                                                          'Holder '
                                                                                                          'physically '
                                                                                                          'supports '
                                                                                                          'the '
                                                                                                          'remote; '
                                                                                                          'all '
                                                                                                          'hand '
                                                                                                          'contacts '
                                                                                                          'have '
                                                                                                          'ended.',
                                                                                                          'Remote '
                                                                                                          'and '
                                                                                                          'all '
                                                                                                          'buttons '
                                                                                                          'retain '
                                                                                                          'released '
                                                                                                          'supported '
                                                                                                          'stability '
                                                                                                          'for '
                                                                                                          '0.300 '
                                                                                                          'seconds.'],
                                                                                               'automatic_power_off': True,
                                                                                               'continuous_on_resets_when_off': True,
                                                                                               'holder_return_required': True,
                                                                                               'stable_duration_s': 0.3},
                                                                                   'television': {'automatic_power_off': True,
                                                                                                  'confirm_button': 'No '
                                                                                                                    'effect '
                                                                                                                    'on '
                                                                                                                    'television '
                                                                                                                    'power '
                                                                                                                    'or '
                                                                                                                    'task '
                                                                                                                    'progress.',
                                                                                                  'countdown': 'Visible '
                                                                                                               'native-simulation-time '
                                                                                                               'countdown; '
                                                                                                               'automatically '
                                                                                                               'off '
                                                                                                               'after '
                                                                                                               'one '
                                                                                                               'uninterrupted '
                                                                                                               '10-second '
                                                                                                               'interval.',
                                                                                                  'initial_power_on': False,
                                                                                                  'initial_remote_face_up': True,
                                                                                                  'required_on_s': 10.0,
                                                                                                  'time_basis': 'Consecutive '
                                                                                                                'native '
                                                                                                                'simulation '
                                                                                                                'time; '
                                                                                                                'wall '
                                                                                                                'time '
                                                                                                                'and '
                                                                                                                'separate '
                                                                                                                'short '
                                                                                                                'on '
                                                                                                                'periods '
                                                                                                                'do '
                                                                                                                'not '
                                                                                                                'count.',
                                                                                                  'toggle_button': 'power',
                                                                                                  'toggle_on': 'One '
                                                                                                               'valid '
                                                                                                               'physical '
                                                                                                               'power '
                                                                                                               'press-and-release '
                                                                                                               'cycle, '
                                                                                                               'at '
                                                                                                               'its '
                                                                                                               'native '
                                                                                                               'release '
                                                                                                               'time.'},
                                                                                   'thresholds': {'held_before_press_s': 0.02,
                                                                                                  'holder_bottom_tolerance_m': 0.002,
                                                                                                  'holder_orientation_tolerance_rad': 0.15,
                                                                                                  'holder_xy_tolerance_m': 0.001,
                                                                                                  'joint_lower_m': -0.002,
                                                                                                  'joint_tolerance_m': 0.0002,
                                                                                                  'joint_upper_m': 0.0,
                                                                                                  'max_actions': 6000,
                                                                                                  'max_time_s': 120.0,
                                                                                                  'maximum_angular_speed_rad_s': 0.15,
                                                                                                  'maximum_linear_speed_m_s': 0.02,
                                                                                                  'maximum_penetration_m': 0.002,
                                                                                                  'maximum_stable_orientation_range_rad': 0.025,
                                                                                                  'maximum_stable_position_range_m': 0.001,
                                                                                                  'minimum_contact_impulse_ns': 1e-10,
                                                                                                  'minimum_handheld_clearance_m': 0.005,
                                                                                                  'press_position_m': -0.0014,
                                                                                                  'release_position_m': -0.0003,
                                                                                                  'stable_s': 0.3}}},
                                                            'qualification': {'status': 'prepared_unvalidated',
                                                                              'native_preflight': False,
                                                                              'manipulation': False,
                                                                              'physics_accepted': False,
                                                                              'public_http_smoke': False,
                                                                              'public_functions_smoke': False,
                                                                              'scope': 'Frozen '
                                                                                       'randomized '
                                                                                       'inputs; '
                                                                                       'separately '
                                                                                       'recorded '
                                                                                       'development '
                                                                                       'checks are '
                                                                                       'not '
                                                                                       'runtime '
                                                                                       'admission '
                                                                                       'data.'}},
                                         'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/initial_randomization_v1/dual_opposed',
                                                                             'manifest_sha256': '91fdb5c82ffa1033714a17bd83332c40b0b30ca1a1e5c871bd083edb328fed7c'},
                                                          'release_schema': 'living_room_randomized_release.v1',
                                                          'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                          'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                 'assembly_version': 'isaacsim_6_1_v2',
                                                                                 'articulations': 2,
                                                                                 'active_sides': ['left',
                                                                                                  'right']},
                                                          'layout': {'robot_layout_id': 'living_room_dual_opposed_left_right_sharpa_v1',
                                                                     'placements': {'left': {'base_position_m': [0.65,
                                                                                                                 0.76,
                                                                                                                 0.78],
                                                                                             'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                      0.0,
                                                                                                                      0.0,
                                                                                                                      -0.7071067811865475],
                                                                                             'base_yaw_rad': -1.5707963267948966,
                                                                                             'hand_chirality': 'left'},
                                                                                    'right': {'base_position_m': [0.65,
                                                                                                                  -0.76,
                                                                                                                  0.78],
                                                                                              'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                       0.0,
                                                                                                                       0.0,
                                                                                                                       0.7071067811865475],
                                                                                              'base_yaw_rad': 1.5707963267948966,
                                                                                              'hand_chirality': 'right'}},
                                                                     'reference_surface': 'fixed '
                                                                                          'room '
                                                                                          'task '
                                                                                          'table',
                                                                     'frame': 'world',
                                                                     'units': 'metres',
                                                                     'control_scope': 'one_agent_58_dof'},
                                                          'objects': {'intended_roles': ['free_remote',
                                                                                         'two_passive_buttons',
                                                                                         'remote_holder',
                                                                                         'television'],
                                                                      'asset_bindings': 'Exact '
                                                                                        'per-seed '
                                                                                        'USD, '
                                                                                        'texture, '
                                                                                        'collider '
                                                                                        'and mass '
                                                                                        'inputs '
                                                                                        'are '
                                                                                        'pinned in '
                                                                                        'the '
                                                                                        'release.'},
                                                          'rules': {'intent': '拿起正面朝上的遥控器开启电视，显示10秒倒计时后自动关机，再归架松手稳定。',
                                                                    'implementation': 'scenes.workrooms.living_room.tasks.remote_operation.rules',
                                                                    'contract': {'asset_freeze_required': ['Remote/button '
                                                                                                           'collision '
                                                                                                           'geometry '
                                                                                                           'and '
                                                                                                           'masses',
                                                                                                           'Button '
                                                                                                           'travel, '
                                                                                                           'spring '
                                                                                                           'and '
                                                                                                           'damping',
                                                                                                           'Press/release '
                                                                                                           'hysteresis',
                                                                                                           'Holder '
                                                                                                           'fit',
                                                                                                           'Stable '
                                                                                                           'speed/contact/penetration '
                                                                                                           'tolerances'],
                                                                                 'budget_exhaustion': 'incomplete',
                                                                                 'buttons': {'ids': ['power',
                                                                                                     'confirm'],
                                                                                             'mechanism': 'Independent '
                                                                                                          'passive '
                                                                                                          'spring-return '
                                                                                                          'prismatic '
                                                                                                          'buttons.',
                                                                                             'no_credit': ['Color '
                                                                                                           'change '
                                                                                                           'without '
                                                                                                           'contact',
                                                                                                           'One '
                                                                                                           'long '
                                                                                                           'hold '
                                                                                                           'counted '
                                                                                                           'repeatedly',
                                                                                                           'Remote '
                                                                                                           'supported '
                                                                                                           'by '
                                                                                                           'table '
                                                                                                           'or '
                                                                                                           'holder '
                                                                                                           'during '
                                                                                                           'credited '
                                                                                                           'press',
                                                                                                           'Agent '
                                                                                                           'claims '
                                                                                                           'that '
                                                                                                           'a '
                                                                                                           'button '
                                                                                                           'was '
                                                                                                           'pressed'],
                                                                                             'press_evidence': ['Physical '
                                                                                                                'finger/button '
                                                                                                                'contact.',
                                                                                                                'Actual '
                                                                                                                'button '
                                                                                                                'joint '
                                                                                                                'crosses '
                                                                                                                'the '
                                                                                                                'asset-bound '
                                                                                                                'press '
                                                                                                                'displacement.',
                                                                                                                'Remote '
                                                                                                                'is '
                                                                                                                'off '
                                                                                                                'the '
                                                                                                                'table/holder '
                                                                                                                'and '
                                                                                                                'genuinely '
                                                                                                                'held '
                                                                                                                'by '
                                                                                                                'one '
                                                                                                                'or '
                                                                                                                'both '
                                                                                                                'declared '
                                                                                                                'hands.',
                                                                                                                'Button '
                                                                                                                'physically '
                                                                                                                'returns '
                                                                                                                'to '
                                                                                                                'or '
                                                                                                                'above '
                                                                                                                'its '
                                                                                                                'release '
                                                                                                                'position '
                                                                                                                'before '
                                                                                                                'a '
                                                                                                                'new '
                                                                                                                'press.']},
                                                                                 'competitive': False,
                                                                                 'evidence_fault': 'invalid_no_score',
                                                                                 'evidence_scope': 'Private '
                                                                                                   'native '
                                                                                                   'body/joint/contact '
                                                                                                   'samples '
                                                                                                   'only. '
                                                                                                   'CPU '
                                                                                                   'replay '
                                                                                                   'verifies '
                                                                                                   'logic; '
                                                                                                   'independent '
                                                                                                   'mechanical '
                                                                                                   'probes '
                                                                                                   'do '
                                                                                                   'not '
                                                                                                   'establish '
                                                                                                   'bimanual '
                                                                                                   'robot '
                                                                                                   'success.',
                                                                                 'failure': ['Remote '
                                                                                             'leaves '
                                                                                             'the '
                                                                                             'permitted '
                                                                                             'workspace '
                                                                                             'or '
                                                                                             'falls '
                                                                                             'to '
                                                                                             'the '
                                                                                             'room '
                                                                                             'floor.'],
                                                                                 'instruction_zh': '拿起正面朝上的遥控器，可用左右手协作悬空握持遥控器，完整按下、松开红色电源键开启电视；电视显示10秒倒计时，到时自动关机。最后将遥控器按正确朝向放回收纳架，松手稳定至少0.300秒。',
                                                                                 'qualification': 'Candidate '
                                                                                                  'physical '
                                                                                                  'buttons '
                                                                                                  'have '
                                                                                                  'independent '
                                                                                                  'native '
                                                                                                  'mechanism '
                                                                                                  'evidence. '
                                                                                                  'The '
                                                                                                  'read-only '
                                                                                                  'observer/scorer '
                                                                                                  'is '
                                                                                                  'implemented; '
                                                                                                  'native '
                                                                                                  'bimanual '
                                                                                                  'robot '
                                                                                                  'task '
                                                                                                  'execution '
                                                                                                  'is '
                                                                                                  'not '
                                                                                                  'qualified.',
                                                                                 'randomization': ['遥控器正面朝上，物理初态固定；10秒倒计时和归架目标不随seed改变。'],
                                                                                 'recovery': {'early_power_off': 'No '
                                                                                                                 'success; '
                                                                                                                 'turn '
                                                                                                                 'on '
                                                                                                                 'again '
                                                                                                                 'and '
                                                                                                                 'restart '
                                                                                                                 'the '
                                                                                                                 'continuous '
                                                                                                                 '10-second '
                                                                                                                 'interval.',
                                                                                              'regrasp': True,
                                                                                              'table_assisted_flip': True,
                                                                                              'table_drop': 'Recover '
                                                                                                            'by '
                                                                                                            'actual '
                                                                                                            'regrasp.'},
                                                                                 'referee': {'instruction_zh': '拿起正面朝上的遥控器，可用左右手协作悬空握持遥控器，完整按下、松开红色电源键开启电视；电视显示10秒倒计时，到时自动关机。最后将遥控器按正确朝向放回收纳架，松手稳定至少0.300秒。',
                                                                                             'observation_scope': '公开静态规则与粗粒度裁判阶段；逐原生子步位姿、接触和连续计时由私有裁判核验。',
                                                                                             'outcomes': {'budget_exhaustion': '运行器结束回合，未完成不制造成功。',
                                                                                                          'cancelled': '取消或运行故障独立记录，不制造任务成功或物理失败。',
                                                                                                          'evaluator_invalid': '缺失、矛盾或不连续原生证据，无有效分数。',
                                                                                                          'ongoing': '尚未完成，可在预算内继续。',
                                                                                                          'safety_stop': '物理安全中止单独记录。',
                                                                                                          'success': '有效物理按键开启电视、完成10秒倒计时自动关机及归架松手稳定。',
                                                                                                          'task_failure': '有效证据确认遥控器落地或离开允许操作区域。'},
                                                                                             'phases': {'await_power_on': '电视已关闭，等待一次有效的电源键完整按压和释放。',
                                                                                                        'await_return': '倒计时已完成并自动关机，等待遥控器归架、松手稳定。',
                                                                                                        'budget_exhausted': '预算结束，未取得本次成功判定。',
                                                                                                        'completed': '已完成开机、连续开机、关机和遥控器归架松手稳定。',
                                                                                                        'evidence_invalid': '原生物理证据无效。',
                                                                                                        'holding_power_on': '电视显示10秒倒计时，结束自动关机。',
                                                                                                        'task_failure': '遥控器落地或离开允许操作区域。'},
                                                                                             'physical_actions': '仅通过机器人真实接触操作遥控器；长按不能重复计数，桌面或收纳架支撑中的按压不计。',
                                                                                             'reason_descriptions': {'await_power_on': '电视已关闭，等待一次有效的电源键完整按压和释放。',
                                                                                                                     'await_return': '倒计时已完成并自动关机，等待遥控器归架、松手稳定。',
                                                                                                                     'budget_exhausted': '预算结束，未取得本次成功判定。',
                                                                                                                     'completed': '已完成开机、连续开机、关机和遥控器归架松手稳定。',
                                                                                                                     'evidence_invalid': '原生物理证据无效。',
                                                                                                                     'holding_power_on': '电视显示10秒倒计时，结束自动关机。',
                                                                                                                     'task_failure': '遥控器落地或离开允许操作区域。'},
                                                                                             'recovery': ['允许翻正、重新抓取和重试。',
                                                                                                          '提前关机不成功；下次开机重新计算连续10秒，多次短开机时间不累计。'],
                                                                                             'schema': 'physical_task_referee_contract.v1',
                                                                                             'success_conditions': ['电视初始关闭；每次有效电源键完整按压并真实释放后只切换一次电视电源。',
                                                                                                                    '可用左右手协作悬空握持遥控器，以真实手指接触按压实体电源键，并让按键松开回弹。',
                                                                                                                    '电视连续开机10秒，屏幕倒计时结束后自动关机；提前手动关机可重试。',
                                                                                                                    '遥控器整体按正确朝向放入收纳架，由架子真实支撑；手与遥控器、按键和架子均无接触。',
                                                                                                                    '遥控器及按键全部释放并连续稳定至少0.300秒，才取得成功。',
                                                                                                                    '自动关机保留本次完成周期资格，仍需归架松手稳定；确认键不影响电视。'],
                                                                                             'task_id': 'remote_operation',
                                                                                             'thresholds': {'held_before_press_s': 0.02,
                                                                                                            'max_actions': 6000,
                                                                                                            'max_time_s': 120.0,
                                                                                                            'press_position_m': -0.0014,
                                                                                                            'release_position_m': -0.0003,
                                                                                                            'required_on_s': 10.0,
                                                                                                            'stable_s': 0.3}},
                                                                                 'schema': 'living_room_remote_operation_rules_v4',
                                                                                 'stages': ['power_on',
                                                                                            'countdown',
                                                                                            'automatic_power_off',
                                                                                            'return_to_holder'],
                                                                                 'status': 'implemented_reference',
                                                                                 'success': {'all_of': ['A '
                                                                                                        'valid '
                                                                                                        'handheld '
                                                                                                        'physical '
                                                                                                        'power '
                                                                                                        'press-and-release '
                                                                                                        'cycle '
                                                                                                        'turns '
                                                                                                        'the '
                                                                                                        'initially '
                                                                                                        'off '
                                                                                                        'television '
                                                                                                        'on.',
                                                                                                        'The '
                                                                                                        'television '
                                                                                                        'remains '
                                                                                                        'continuously '
                                                                                                        'on '
                                                                                                        'for '
                                                                                                        'at '
                                                                                                        'least '
                                                                                                        '10 '
                                                                                                        'native '
                                                                                                        'simulation '
                                                                                                        'seconds.',
                                                                                                        'The '
                                                                                                        'native '
                                                                                                        '10-second '
                                                                                                        'countdown '
                                                                                                        'automatically '
                                                                                                        'turns '
                                                                                                        'the '
                                                                                                        'television '
                                                                                                        'off; '
                                                                                                        'keep '
                                                                                                        'this '
                                                                                                        'completed-cycle '
                                                                                                        'credit '
                                                                                                        'while '
                                                                                                        'returning '
                                                                                                        'the '
                                                                                                        'remote.',
                                                                                                        'Whole '
                                                                                                        'remote '
                                                                                                        'is '
                                                                                                        'within '
                                                                                                        'the '
                                                                                                        'correctly '
                                                                                                        'oriented '
                                                                                                        'holder '
                                                                                                        'acceptance '
                                                                                                        'volume.',
                                                                                                        'Holder '
                                                                                                        'physically '
                                                                                                        'supports '
                                                                                                        'the '
                                                                                                        'remote; '
                                                                                                        'all '
                                                                                                        'hand '
                                                                                                        'contacts '
                                                                                                        'have '
                                                                                                        'ended.',
                                                                                                        'Remote '
                                                                                                        'and '
                                                                                                        'all '
                                                                                                        'buttons '
                                                                                                        'retain '
                                                                                                        'released '
                                                                                                        'supported '
                                                                                                        'stability '
                                                                                                        'for '
                                                                                                        '0.300 '
                                                                                                        'seconds.'],
                                                                                             'automatic_power_off': True,
                                                                                             'continuous_on_resets_when_off': True,
                                                                                             'holder_return_required': True,
                                                                                             'stable_duration_s': 0.3},
                                                                                 'television': {'automatic_power_off': True,
                                                                                                'confirm_button': 'No '
                                                                                                                  'effect '
                                                                                                                  'on '
                                                                                                                  'television '
                                                                                                                  'power '
                                                                                                                  'or '
                                                                                                                  'task '
                                                                                                                  'progress.',
                                                                                                'countdown': 'Visible '
                                                                                                             'native-simulation-time '
                                                                                                             'countdown; '
                                                                                                             'automatically '
                                                                                                             'off '
                                                                                                             'after '
                                                                                                             'one '
                                                                                                             'uninterrupted '
                                                                                                             '10-second '
                                                                                                             'interval.',
                                                                                                'initial_power_on': False,
                                                                                                'initial_remote_face_up': True,
                                                                                                'required_on_s': 10.0,
                                                                                                'time_basis': 'Consecutive '
                                                                                                              'native '
                                                                                                              'simulation '
                                                                                                              'time; '
                                                                                                              'wall '
                                                                                                              'time '
                                                                                                              'and '
                                                                                                              'separate '
                                                                                                              'short '
                                                                                                              'on '
                                                                                                              'periods '
                                                                                                              'do '
                                                                                                              'not '
                                                                                                              'count.',
                                                                                                'toggle_button': 'power',
                                                                                                'toggle_on': 'One '
                                                                                                             'valid '
                                                                                                             'physical '
                                                                                                             'power '
                                                                                                             'press-and-release '
                                                                                                             'cycle, '
                                                                                                             'at '
                                                                                                             'its '
                                                                                                             'native '
                                                                                                             'release '
                                                                                                             'time.'},
                                                                                 'thresholds': {'held_before_press_s': 0.02,
                                                                                                'holder_bottom_tolerance_m': 0.002,
                                                                                                'holder_orientation_tolerance_rad': 0.15,
                                                                                                'holder_xy_tolerance_m': 0.001,
                                                                                                'joint_lower_m': -0.002,
                                                                                                'joint_tolerance_m': 0.0002,
                                                                                                'joint_upper_m': 0.0,
                                                                                                'max_actions': 6000,
                                                                                                'max_time_s': 120.0,
                                                                                                'maximum_angular_speed_rad_s': 0.15,
                                                                                                'maximum_linear_speed_m_s': 0.02,
                                                                                                'maximum_penetration_m': 0.002,
                                                                                                'maximum_stable_orientation_range_rad': 0.025,
                                                                                                'maximum_stable_position_range_m': 0.001,
                                                                                                'minimum_contact_impulse_ns': 1e-10,
                                                                                                'minimum_handheld_clearance_m': 0.005,
                                                                                                'press_position_m': -0.0014,
                                                                                                'release_position_m': -0.0003,
                                                                                                'stable_s': 0.3}}},
                                                          'qualification': {'status': 'prepared_unvalidated',
                                                                            'native_preflight': False,
                                                                            'manipulation': False,
                                                                            'physics_accepted': False,
                                                                            'public_http_smoke': False,
                                                                            'public_functions_smoke': False,
                                                                            'scope': 'Frozen '
                                                                                     'randomized '
                                                                                     'inputs; '
                                                                                     'separately '
                                                                                     'recorded '
                                                                                     'development '
                                                                                     'checks are '
                                                                                     'not runtime '
                                                                                     'admission '
                                                                                     'data.'}}}},
 'storage_box': {'family': 'living_room',
                 'task_id': 'storage_box',
                 'default_configuration_id': 'single_arm',
                 'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/storage_box/initial_randomization_v1/single_arm',
                                  'manifest_sha256': 'c0319eea4c9f849ab6f2a57e95854c0ce5dff58b1c0334cf1639e22c1922e0da'},
                 'single_arm_robot_requirements': {'arm': 'FR3',
                                                   'hand_id': 'sharpa_w01_right',
                                                   'assembly_version': 'isaacsim_6_1_v2',
                                                   'articulations': 1,
                                                   'active_sides': ['right'],
                                                   'configuration_module': 'scenes.workrooms.living_room.dual_robot_layout',
                                                   'configuration_factory': 'robot_configurations',
                                                   'base_positions_m': {'front': [0.65,
                                                                                  -0.76,
                                                                                  0.78]},
                                                   'base_constraint': 'native_fixed_base_v1',
                                                   'geometry_scale': [1.0, 1.0, 1.0],
                                                   'ownership': 'One agent per active arm; shared '
                                                                'bench controls and original '
                                                                'source joints.',
                                                   'qualification': 'Existing ready-pose/endpoint '
                                                                    'checks are not task-motion '
                                                                    'qualification.'},
                 'single_arm_layout': {'robot_layout_id': 'living_room_storage_box_original_single_arm',
                                       'placements': {'right': {'base_position_m': [0.65,
                                                                                    -0.76,
                                                                                    0.78],
                                                                'base_quaternion_wxyz': [0.7071067811865476,
                                                                                         0.0,
                                                                                         0.0,
                                                                                         0.7071067811865476],
                                                                'base_yaw_rad': 1.5707963267948968,
                                                                'hand_chirality': 'right'}},
                                       'reference_surface': 'original fixed task installation',
                                       'frame': 'world',
                                       'units': 'metres',
                                       'control_scope': 'one_agent_29_dof'},
                 'configurations': {'single_arm': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/storage_box/initial_randomization_v1/single_arm',
                                                                      'manifest_sha256': 'c0319eea4c9f849ab6f2a57e95854c0ce5dff58b1c0334cf1639e22c1922e0da'},
                                                   'release_schema': 'living_room_randomized_release.v1',
                                                   'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                   'robot_requirements': {'arm': 'FR3',
                                                                          'hand_id': 'sharpa_w01_right',
                                                                          'assembly_version': 'isaacsim_6_1_v2',
                                                                          'articulations': 1,
                                                                          'active_sides': ['right'],
                                                                          'configuration_module': 'scenes.workrooms.living_room.dual_robot_layout',
                                                                          'configuration_factory': 'robot_configurations',
                                                                          'base_positions_m': {'front': [0.65,
                                                                                                         -0.76,
                                                                                                         0.78]},
                                                                          'base_constraint': 'native_fixed_base_v1',
                                                                          'geometry_scale': [1.0,
                                                                                             1.0,
                                                                                             1.0],
                                                                          'ownership': 'One agent '
                                                                                       'per active '
                                                                                       'arm; '
                                                                                       'shared '
                                                                                       'bench '
                                                                                       'controls '
                                                                                       'and '
                                                                                       'original '
                                                                                       'source '
                                                                                       'joints.',
                                                                          'qualification': 'Existing '
                                                                                           'ready-pose/endpoint '
                                                                                           'checks '
                                                                                           'are '
                                                                                           'not '
                                                                                           'task-motion '
                                                                                           'qualification.'},
                                                   'layout': {'robot_layout_id': 'living_room_storage_box_original_single_arm',
                                                              'placements': {'right': {'base_position_m': [0.65,
                                                                                                           -0.76,
                                                                                                           0.78],
                                                                                       'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                0.0,
                                                                                                                0.0,
                                                                                                                0.7071067811865476],
                                                                                       'base_yaw_rad': 1.5707963267948968,
                                                                                       'hand_chirality': 'right'}},
                                                              'reference_surface': 'original fixed '
                                                                                   'task '
                                                                                   'installation',
                                                              'frame': 'world',
                                                              'units': 'metres',
                                                              'control_scope': 'one_agent_29_dof'},
                                                   'objects': {'intended_roles': ['dynamic_box',
                                                                                  'passive_hinged_lid',
                                                                                  'two_physical_rotary_catches',
                                                                                  'blue_item',
                                                                                  'collection_tray'],
                                                               'asset_bindings': 'Exact per-seed '
                                                                                 'USD, texture, '
                                                                                 'collider and '
                                                                                 'mass inputs are '
                                                                                 'pinned in the '
                                                                                 'release.'},
                                                   'rules': {'intent': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                             'implementation': 'scenes.workrooms.living_room.tasks.storage_box.rules',
                                                             'contract': {'competitive': False,
                                                                          'instruction_zh': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                                          'referee': {'instruction_zh': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                                                      'object_properties': [{'basis': '项目几何与未标定仿真先验。',
                                                                                                             'id': 'box',
                                                                                                             'label': '收纳盒盒体',
                                                                                                             'nominal_mass_kg': 0.8},
                                                                                                            {'basis': '项目几何与未标定仿真先验。',
                                                                                                             'id': 'lid',
                                                                                                             'label': '盒盖',
                                                                                                             'nominal_mass_kg': 0.12},
                                                                                                            {'basis': '项目几何与未标定仿真先验。',
                                                                                                             'id': 'left_catch',
                                                                                                             'label': '左旋转卡扣',
                                                                                                             'nominal_mass_kg': 0.025},
                                                                                                            {'basis': '项目几何与未标定仿真先验。',
                                                                                                             'id': 'right_catch',
                                                                                                             'label': '右旋转卡扣',
                                                                                                             'nominal_mass_kg': 0.025},
                                                                                                            {'basis': '项目几何与未标定仿真先验。',
                                                                                                             'id': 'item',
                                                                                                             'label': '蓝色小盒',
                                                                                                             'nominal_mass_kg': 0.08}],
                                                                                      'observation_scope': '机构、目标和标称物性公开；物体真值、接触与连续稳定证据为裁判私有。',
                                                                                      'outcomes': {'budget_exhaustion': '未完成。',
                                                                                                   'cancelled': '取消。',
                                                                                                   'evaluator_invalid': '证据不完整，不评分。',
                                                                                                   'ongoing': '可以继续。',
                                                                                                   'safety_stop': '物理异常停止。',
                                                                                                   'success': '已完成。',
                                                                                                   'task_failure': '物件跌落。'},
                                                                                      'phases': {'closing': '关盖并旋回两个卡扣。',
                                                                                                 'completed': '物件已收集，盒盖与卡扣关闭，松手稳定。',
                                                                                                 'evidence_invalid': '原生物理证据不完整。',
                                                                                                 'failed': '物件跌落或出现严重穿透。',
                                                                                                 'incomplete': '预算结束，任务尚未完成。',
                                                                                                 'opening': '旋开卡扣并打开盒盖。',
                                                                                                 'settling': '松手并等待稳定。',
                                                                                                 'transfer': '取出蓝色小盒并放入右侧收集盘。'},
                                                                                      'physical_actions': '通过普通机器人关节动作操作真实物体；可单手借助桌面，也可左右手配合。不得直接写物体位姿或卡扣状态。',
                                                                                      'reason_descriptions': {'closing': '关盖并旋回两个卡扣。',
                                                                                                              'completed': '物件已收集，盒盖与卡扣关闭，松手稳定。',
                                                                                                              'evidence_invalid': '原生物理证据不完整。',
                                                                                                              'failed': '物件跌落或出现严重穿透。',
                                                                                                              'incomplete': '预算结束，任务尚未完成。',
                                                                                                              'incomplete_budget': '预算结束，任务未完成。',
                                                                                                              'native_evidence_unknown': '原生证据缺失或不连续。',
                                                                                                              'object_fell': '任务物件跌到桌面以下。',
                                                                                                              'opening': '旋开卡扣并打开盒盖。',
                                                                                                              'physics_safety_stop': '任务物件出现严重穿透。',
                                                                                                              'reset_baseline_not_scored': '初态不计分。',
                                                                                                              'robot_operation_required': '需要机器人实际参与操作。',
                                                                                                              'settling': '松手并等待稳定。',
                                                                                                              'transfer': '取出蓝色小盒并放入右侧收集盘。'},
                                                                                      'recovery': ['允许重抓、调整、重新开盖或重新放置。',
                                                                                                   '两个卡扣先后及使用哪只手不限，可借助桌面支撑。'],
                                                                                      'schema': 'physical_task_referee_contract.v1',
                                                                                      'success_conditions': ['两个卡扣实际旋开且盒盖实际打开，不能穿过仍关闭的卡扣。',
                                                                                                             '蓝色小盒整体放在收集盘内，由真实盘底接触支撑。',
                                                                                                             '盒盖关闭到5度以内，两个卡扣各回到8度以内；不能只把物件移出而留盖或卡扣打开。',
                                                                                                             '机器人实际参与操作；所有任务物体脱离机器人接触并稳定至少0.3秒。'],
                                                                                      'task_id': 'storage_box',
                                                                                      'thresholds': {'catch_closed_deg': 8.0,
                                                                                                     'catch_open_deg': 85.0,
                                                                                                     'lid_closed_deg': 5.0,
                                                                                                     'lid_open_deg': 45.0,
                                                                                                     'max_actions': 9000,
                                                                                                     'max_time_s': 180.0,
                                                                                                     'maximum_angular_speed_rad_s': 0.15,
                                                                                                     'maximum_linear_speed_m_s': 0.02,
                                                                                                     'maximum_penetration_m': 0.003,
                                                                                                     'stable_orientation_range_rad': 0.04,
                                                                                                     'stable_position_range_m': 0.003,
                                                                                                     'stable_s': 0.3,
                                                                                                     'tray_bottom_tolerance_m': 0.004}},
                                                                          'reset_credit': False,
                                                                          'schema': 'living_storage_box_rules.v1'}},
                                                   'qualification': {'status': 'prepared_unvalidated',
                                                                     'native_preflight': False,
                                                                     'manipulation': False,
                                                                     'physics_accepted': False,
                                                                     'public_http_smoke': False,
                                                                     'public_functions_smoke': False,
                                                                     'scope': 'Frozen randomized '
                                                                              'inputs; separately '
                                                                              'recorded '
                                                                              'development checks '
                                                                              'are not runtime '
                                                                              'admission data.'}},
                                    'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/storage_box/initial_randomization_v1/dual_same_side',
                                                                          'manifest_sha256': '0f9160ee6141d9bef7ab524f024fba38f684fef091b2275d2d89f12c544554ca'},
                                                       'release_schema': 'living_room_randomized_release.v1',
                                                       'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                                              'articulations': 2,
                                                                              'active_sides': ['left',
                                                                                               'right']},
                                                       'layout': {'robot_layout_id': 'living_room_dual_same_side_left_right_sharpa_v1',
                                                                  'placements': {'left': {'base_position_m': [0.35,
                                                                                                              -0.68,
                                                                                                              0.78],
                                                                                          'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                   0.0,
                                                                                                                   0.0,
                                                                                                                   0.7071067811865475],
                                                                                          'base_yaw_rad': 1.5707963267948966,
                                                                                          'hand_chirality': 'left'},
                                                                                 'right': {'base_position_m': [0.95,
                                                                                                               -0.68,
                                                                                                               0.78],
                                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                    0.0,
                                                                                                                    0.0,
                                                                                                                    0.7071067811865475],
                                                                                           'base_yaw_rad': 1.5707963267948966,
                                                                                           'hand_chirality': 'right'}},
                                                                  'reference_surface': 'fixed room '
                                                                                       'task table',
                                                                  'frame': 'world',
                                                                  'units': 'metres',
                                                                  'control_scope': 'one_agent_58_dof'},
                                                       'objects': {'intended_roles': ['dynamic_box',
                                                                                      'passive_hinged_lid',
                                                                                      'two_physical_rotary_catches',
                                                                                      'blue_item',
                                                                                      'collection_tray'],
                                                                   'asset_bindings': 'Exact '
                                                                                     'per-seed '
                                                                                     'USD, '
                                                                                     'texture, '
                                                                                     'collider and '
                                                                                     'mass inputs '
                                                                                     'are pinned '
                                                                                     'in the '
                                                                                     'release.'},
                                                       'rules': {'intent': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                                 'implementation': 'scenes.workrooms.living_room.tasks.storage_box.rules',
                                                                 'contract': {'competitive': False,
                                                                              'instruction_zh': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                                              'referee': {'instruction_zh': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                                                          'object_properties': [{'basis': '项目几何与未标定仿真先验。',
                                                                                                                 'id': 'box',
                                                                                                                 'label': '收纳盒盒体',
                                                                                                                 'nominal_mass_kg': 0.8},
                                                                                                                {'basis': '项目几何与未标定仿真先验。',
                                                                                                                 'id': 'lid',
                                                                                                                 'label': '盒盖',
                                                                                                                 'nominal_mass_kg': 0.12},
                                                                                                                {'basis': '项目几何与未标定仿真先验。',
                                                                                                                 'id': 'left_catch',
                                                                                                                 'label': '左旋转卡扣',
                                                                                                                 'nominal_mass_kg': 0.025},
                                                                                                                {'basis': '项目几何与未标定仿真先验。',
                                                                                                                 'id': 'right_catch',
                                                                                                                 'label': '右旋转卡扣',
                                                                                                                 'nominal_mass_kg': 0.025},
                                                                                                                {'basis': '项目几何与未标定仿真先验。',
                                                                                                                 'id': 'item',
                                                                                                                 'label': '蓝色小盒',
                                                                                                                 'nominal_mass_kg': 0.08}],
                                                                                          'observation_scope': '机构、目标和标称物性公开；物体真值、接触与连续稳定证据为裁判私有。',
                                                                                          'outcomes': {'budget_exhaustion': '未完成。',
                                                                                                       'cancelled': '取消。',
                                                                                                       'evaluator_invalid': '证据不完整，不评分。',
                                                                                                       'ongoing': '可以继续。',
                                                                                                       'safety_stop': '物理异常停止。',
                                                                                                       'success': '已完成。',
                                                                                                       'task_failure': '物件跌落。'},
                                                                                          'phases': {'closing': '关盖并旋回两个卡扣。',
                                                                                                     'completed': '物件已收集，盒盖与卡扣关闭，松手稳定。',
                                                                                                     'evidence_invalid': '原生物理证据不完整。',
                                                                                                     'failed': '物件跌落或出现严重穿透。',
                                                                                                     'incomplete': '预算结束，任务尚未完成。',
                                                                                                     'opening': '旋开卡扣并打开盒盖。',
                                                                                                     'settling': '松手并等待稳定。',
                                                                                                     'transfer': '取出蓝色小盒并放入右侧收集盘。'},
                                                                                          'physical_actions': '通过普通机器人关节动作操作真实物体；可单手借助桌面，也可左右手配合。不得直接写物体位姿或卡扣状态。',
                                                                                          'reason_descriptions': {'closing': '关盖并旋回两个卡扣。',
                                                                                                                  'completed': '物件已收集，盒盖与卡扣关闭，松手稳定。',
                                                                                                                  'evidence_invalid': '原生物理证据不完整。',
                                                                                                                  'failed': '物件跌落或出现严重穿透。',
                                                                                                                  'incomplete': '预算结束，任务尚未完成。',
                                                                                                                  'incomplete_budget': '预算结束，任务未完成。',
                                                                                                                  'native_evidence_unknown': '原生证据缺失或不连续。',
                                                                                                                  'object_fell': '任务物件跌到桌面以下。',
                                                                                                                  'opening': '旋开卡扣并打开盒盖。',
                                                                                                                  'physics_safety_stop': '任务物件出现严重穿透。',
                                                                                                                  'reset_baseline_not_scored': '初态不计分。',
                                                                                                                  'robot_operation_required': '需要机器人实际参与操作。',
                                                                                                                  'settling': '松手并等待稳定。',
                                                                                                                  'transfer': '取出蓝色小盒并放入右侧收集盘。'},
                                                                                          'recovery': ['允许重抓、调整、重新开盖或重新放置。',
                                                                                                       '两个卡扣先后及使用哪只手不限，可借助桌面支撑。'],
                                                                                          'schema': 'physical_task_referee_contract.v1',
                                                                                          'success_conditions': ['两个卡扣实际旋开且盒盖实际打开，不能穿过仍关闭的卡扣。',
                                                                                                                 '蓝色小盒整体放在收集盘内，由真实盘底接触支撑。',
                                                                                                                 '盒盖关闭到5度以内，两个卡扣各回到8度以内；不能只把物件移出而留盖或卡扣打开。',
                                                                                                                 '机器人实际参与操作；所有任务物体脱离机器人接触并稳定至少0.3秒。'],
                                                                                          'task_id': 'storage_box',
                                                                                          'thresholds': {'catch_closed_deg': 8.0,
                                                                                                         'catch_open_deg': 85.0,
                                                                                                         'lid_closed_deg': 5.0,
                                                                                                         'lid_open_deg': 45.0,
                                                                                                         'max_actions': 9000,
                                                                                                         'max_time_s': 180.0,
                                                                                                         'maximum_angular_speed_rad_s': 0.15,
                                                                                                         'maximum_linear_speed_m_s': 0.02,
                                                                                                         'maximum_penetration_m': 0.003,
                                                                                                         'stable_orientation_range_rad': 0.04,
                                                                                                         'stable_position_range_m': 0.003,
                                                                                                         'stable_s': 0.3,
                                                                                                         'tray_bottom_tolerance_m': 0.004}},
                                                                              'reset_credit': False,
                                                                              'schema': 'living_storage_box_rules.v1'}},
                                                       'qualification': {'status': 'prepared_unvalidated',
                                                                         'native_preflight': False,
                                                                         'manipulation': False,
                                                                         'physics_accepted': False,
                                                                         'public_http_smoke': False,
                                                                         'public_functions_smoke': False,
                                                                         'scope': 'Frozen '
                                                                                  'randomized '
                                                                                  'inputs; '
                                                                                  'separately '
                                                                                  'recorded '
                                                                                  'development '
                                                                                  'checks are not '
                                                                                  'runtime '
                                                                                  'admission '
                                                                                  'data.'}},
                                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/storage_box/initial_randomization_v1/dual_opposed',
                                                                        'manifest_sha256': '19ef03d97e59ebc15f7724b65c7d1f6965c6f6870ae8c0483f817e73aee98123'},
                                                     'release_schema': 'living_room_randomized_release.v1',
                                                     'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                                            'articulations': 2,
                                                                            'active_sides': ['left',
                                                                                             'right']},
                                                     'layout': {'robot_layout_id': 'living_room_dual_opposed_left_right_sharpa_v1',
                                                                'placements': {'left': {'base_position_m': [0.65,
                                                                                                            0.76,
                                                                                                            0.78],
                                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                 0.0,
                                                                                                                 0.0,
                                                                                                                 -0.7071067811865475],
                                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                                        'hand_chirality': 'left'},
                                                                               'right': {'base_position_m': [0.65,
                                                                                                             -0.76,
                                                                                                             0.78],
                                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                  0.0,
                                                                                                                  0.0,
                                                                                                                  0.7071067811865475],
                                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                                         'hand_chirality': 'right'}},
                                                                'reference_surface': 'fixed room '
                                                                                     'task table',
                                                                'frame': 'world',
                                                                'units': 'metres',
                                                                'control_scope': 'one_agent_58_dof'},
                                                     'objects': {'intended_roles': ['dynamic_box',
                                                                                    'passive_hinged_lid',
                                                                                    'two_physical_rotary_catches',
                                                                                    'blue_item',
                                                                                    'collection_tray'],
                                                                 'asset_bindings': 'Exact per-seed '
                                                                                   'USD, texture, '
                                                                                   'collider and '
                                                                                   'mass inputs '
                                                                                   'are pinned in '
                                                                                   'the release.'},
                                                     'rules': {'intent': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                               'implementation': 'scenes.workrooms.living_room.tasks.storage_box.rules',
                                                               'contract': {'competitive': False,
                                                                            'instruction_zh': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                                            'referee': {'instruction_zh': '旋开收纳盒前侧的两个金属卡扣，打开盒盖，把盒内蓝色小盒取出放入右侧收集盘，再关盖并把两个卡扣旋回；全部松手放稳。',
                                                                                        'object_properties': [{'basis': '项目几何与未标定仿真先验。',
                                                                                                               'id': 'box',
                                                                                                               'label': '收纳盒盒体',
                                                                                                               'nominal_mass_kg': 0.8},
                                                                                                              {'basis': '项目几何与未标定仿真先验。',
                                                                                                               'id': 'lid',
                                                                                                               'label': '盒盖',
                                                                                                               'nominal_mass_kg': 0.12},
                                                                                                              {'basis': '项目几何与未标定仿真先验。',
                                                                                                               'id': 'left_catch',
                                                                                                               'label': '左旋转卡扣',
                                                                                                               'nominal_mass_kg': 0.025},
                                                                                                              {'basis': '项目几何与未标定仿真先验。',
                                                                                                               'id': 'right_catch',
                                                                                                               'label': '右旋转卡扣',
                                                                                                               'nominal_mass_kg': 0.025},
                                                                                                              {'basis': '项目几何与未标定仿真先验。',
                                                                                                               'id': 'item',
                                                                                                               'label': '蓝色小盒',
                                                                                                               'nominal_mass_kg': 0.08}],
                                                                                        'observation_scope': '机构、目标和标称物性公开；物体真值、接触与连续稳定证据为裁判私有。',
                                                                                        'outcomes': {'budget_exhaustion': '未完成。',
                                                                                                     'cancelled': '取消。',
                                                                                                     'evaluator_invalid': '证据不完整，不评分。',
                                                                                                     'ongoing': '可以继续。',
                                                                                                     'safety_stop': '物理异常停止。',
                                                                                                     'success': '已完成。',
                                                                                                     'task_failure': '物件跌落。'},
                                                                                        'phases': {'closing': '关盖并旋回两个卡扣。',
                                                                                                   'completed': '物件已收集，盒盖与卡扣关闭，松手稳定。',
                                                                                                   'evidence_invalid': '原生物理证据不完整。',
                                                                                                   'failed': '物件跌落或出现严重穿透。',
                                                                                                   'incomplete': '预算结束，任务尚未完成。',
                                                                                                   'opening': '旋开卡扣并打开盒盖。',
                                                                                                   'settling': '松手并等待稳定。',
                                                                                                   'transfer': '取出蓝色小盒并放入右侧收集盘。'},
                                                                                        'physical_actions': '通过普通机器人关节动作操作真实物体；可单手借助桌面，也可左右手配合。不得直接写物体位姿或卡扣状态。',
                                                                                        'reason_descriptions': {'closing': '关盖并旋回两个卡扣。',
                                                                                                                'completed': '物件已收集，盒盖与卡扣关闭，松手稳定。',
                                                                                                                'evidence_invalid': '原生物理证据不完整。',
                                                                                                                'failed': '物件跌落或出现严重穿透。',
                                                                                                                'incomplete': '预算结束，任务尚未完成。',
                                                                                                                'incomplete_budget': '预算结束，任务未完成。',
                                                                                                                'native_evidence_unknown': '原生证据缺失或不连续。',
                                                                                                                'object_fell': '任务物件跌到桌面以下。',
                                                                                                                'opening': '旋开卡扣并打开盒盖。',
                                                                                                                'physics_safety_stop': '任务物件出现严重穿透。',
                                                                                                                'reset_baseline_not_scored': '初态不计分。',
                                                                                                                'robot_operation_required': '需要机器人实际参与操作。',
                                                                                                                'settling': '松手并等待稳定。',
                                                                                                                'transfer': '取出蓝色小盒并放入右侧收集盘。'},
                                                                                        'recovery': ['允许重抓、调整、重新开盖或重新放置。',
                                                                                                     '两个卡扣先后及使用哪只手不限，可借助桌面支撑。'],
                                                                                        'schema': 'physical_task_referee_contract.v1',
                                                                                        'success_conditions': ['两个卡扣实际旋开且盒盖实际打开，不能穿过仍关闭的卡扣。',
                                                                                                               '蓝色小盒整体放在收集盘内，由真实盘底接触支撑。',
                                                                                                               '盒盖关闭到5度以内，两个卡扣各回到8度以内；不能只把物件移出而留盖或卡扣打开。',
                                                                                                               '机器人实际参与操作；所有任务物体脱离机器人接触并稳定至少0.3秒。'],
                                                                                        'task_id': 'storage_box',
                                                                                        'thresholds': {'catch_closed_deg': 8.0,
                                                                                                       'catch_open_deg': 85.0,
                                                                                                       'lid_closed_deg': 5.0,
                                                                                                       'lid_open_deg': 45.0,
                                                                                                       'max_actions': 9000,
                                                                                                       'max_time_s': 180.0,
                                                                                                       'maximum_angular_speed_rad_s': 0.15,
                                                                                                       'maximum_linear_speed_m_s': 0.02,
                                                                                                       'maximum_penetration_m': 0.003,
                                                                                                       'stable_orientation_range_rad': 0.04,
                                                                                                       'stable_position_range_m': 0.003,
                                                                                                       'stable_s': 0.3,
                                                                                                       'tray_bottom_tolerance_m': 0.004}},
                                                                            'reset_credit': False,
                                                                            'schema': 'living_storage_box_rules.v1'}},
                                                     'qualification': {'status': 'prepared_unvalidated',
                                                                       'native_preflight': False,
                                                                       'manipulation': False,
                                                                       'physics_accepted': False,
                                                                       'public_http_smoke': False,
                                                                       'public_functions_smoke': False,
                                                                       'scope': 'Frozen randomized '
                                                                                'inputs; '
                                                                                'separately '
                                                                                'recorded '
                                                                                'development '
                                                                                'checks are not '
                                                                                'runtime admission '
                                                                                'data.'}}}},
 'water_pouring': {'family': 'living_room',
                   'task_id': 'water_pouring',
                   'default_configuration_id': 'single_arm',
                   'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/initial_randomization_v1/single_arm',
                                    'manifest_sha256': '6b06274c7e40ff3de2384a233fc1b1b7cdd7ffd006a2a940057cb332c3fed4af'},
                   'single_arm_robot_requirements': {'arm': 'FR3',
                                                     'hand_id': 'sharpa_w01_right',
                                                     'assembly_version': 'isaacsim_6_1_v2',
                                                     'articulations': 1,
                                                     'active_sides': ['right'],
                                                     'configuration_module': 'scenes.workrooms.living_room.dual_robot_layout',
                                                     'configuration_factory': 'robot_configurations',
                                                     'base_positions_m': {'front': [0.65,
                                                                                    -0.76,
                                                                                    0.78]},
                                                     'base_constraint': 'native_fixed_base_v1',
                                                     'geometry_scale': [1.0, 1.0, 1.0],
                                                     'ownership': 'One agent per active arm; '
                                                                  'shared bench controls and '
                                                                  'original source joints.',
                                                     'qualification': 'Existing '
                                                                      'ready-pose/endpoint checks '
                                                                      'are not task-motion '
                                                                      'qualification.'},
                   'single_arm_layout': {'robot_layout_id': 'living_room_water_pouring_original_single_arm',
                                         'placements': {'right': {'base_position_m': [0.65,
                                                                                      -0.76,
                                                                                      0.78],
                                                                  'base_quaternion_wxyz': [0.7071067811865476,
                                                                                           0.0,
                                                                                           0.0,
                                                                                           0.7071067811865476],
                                                                  'base_yaw_rad': 1.5707963267948968,
                                                                  'hand_chirality': 'right'}},
                                         'reference_surface': 'original fixed task installation',
                                         'frame': 'world',
                                         'units': 'metres',
                                         'control_scope': 'one_agent_29_dof'},
                   'configurations': {'single_arm': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/initial_randomization_v1/single_arm',
                                                                        'manifest_sha256': '6b06274c7e40ff3de2384a233fc1b1b7cdd7ffd006a2a940057cb332c3fed4af'},
                                                     'release_schema': 'living_room_randomized_release.v1',
                                                     'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                     'robot_requirements': {'arm': 'FR3',
                                                                            'hand_id': 'sharpa_w01_right',
                                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                                            'articulations': 1,
                                                                            'active_sides': ['right'],
                                                                            'configuration_module': 'scenes.workrooms.living_room.dual_robot_layout',
                                                                            'configuration_factory': 'robot_configurations',
                                                                            'base_positions_m': {'front': [0.65,
                                                                                                           -0.76,
                                                                                                           0.78]},
                                                                            'base_constraint': 'native_fixed_base_v1',
                                                                            'geometry_scale': [1.0,
                                                                                               1.0,
                                                                                               1.0],
                                                                            'ownership': 'One '
                                                                                         'agent '
                                                                                         'per '
                                                                                         'active '
                                                                                         'arm; '
                                                                                         'shared '
                                                                                         'bench '
                                                                                         'controls '
                                                                                         'and '
                                                                                         'original '
                                                                                         'source '
                                                                                         'joints.',
                                                                            'qualification': 'Existing '
                                                                                             'ready-pose/endpoint '
                                                                                             'checks '
                                                                                             'are '
                                                                                             'not '
                                                                                             'task-motion '
                                                                                             'qualification.'},
                                                     'layout': {'robot_layout_id': 'living_room_water_pouring_original_single_arm',
                                                                'placements': {'right': {'base_position_m': [0.65,
                                                                                                             -0.76,
                                                                                                             0.78],
                                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                  0.0,
                                                                                                                  0.0,
                                                                                                                  0.7071067811865476],
                                                                                         'base_yaw_rad': 1.5707963267948968,
                                                                                         'hand_chirality': 'right'}},
                                                                'reference_surface': 'original '
                                                                                     'fixed task '
                                                                                     'installation',
                                                                'frame': 'world',
                                                                'units': 'metres',
                                                                'control_scope': 'one_agent_29_dof'},
                                                     'objects': {'intended_roles': ['spouted_water_kettle',
                                                                                    'free_cup',
                                                                                    'water_particles',
                                                                                    'spill_tray'],
                                                                 'asset_bindings': 'Exact per-seed '
                                                                                   'USD, texture, '
                                                                                   'collider and '
                                                                                   'mass inputs '
                                                                                   'are pinned in '
                                                                                   'the release.'},
                                                     'rules': {'intent': '抓起带把手水壶，向杯中倒入目标水量，控制洒漏，再将壶放回并松手。',
                                                               'implementation': 'scenes.workrooms.living_room.tasks.water_pouring.rules',
                                                               'contract': {'asset_freeze_required': ['Vessel '
                                                                                                      'mesh/cavity/spout/handle '
                                                                                                      'dimensions, '
                                                                                                      'dry '
                                                                                                      'masses '
                                                                                                      'and '
                                                                                                      'inertia',
                                                                                                      'Water '
                                                                                                      'density, '
                                                                                                      'particle '
                                                                                                      'masses/spacing, '
                                                                                                      'fluid '
                                                                                                      'material '
                                                                                                      'and '
                                                                                                      'solver '
                                                                                                      'settings',
                                                                                                      'Collision '
                                                                                                      'thickness/resolution, '
                                                                                                      'finite '
                                                                                                      'leak '
                                                                                                      'and '
                                                                                                      'mass-accounting '
                                                                                                      'tolerances',
                                                                                                      'Cup '
                                                                                                      'occupancy, '
                                                                                                      'flight/settling, '
                                                                                                      'spill-region '
                                                                                                      'classification, '
                                                                                                      'interval '
                                                                                                      'coverage, '
                                                                                                      'upright '
                                                                                                      'and '
                                                                                                      'return-zone '
                                                                                                      'thresholds',
                                                                                                      'Readback '
                                                                                                      'provenance; '
                                                                                                      'known '
                                                                                                      'quantitative '
                                                                                                      'water-load '
                                                                                                      'limitation '
                                                                                                      'recorded '
                                                                                                      'separately '
                                                                                                      'from '
                                                                                                      'task '
                                                                                                      'scoring'],
                                                                            'budget_exhaustion': 'incomplete',
                                                                            'competitive': False,
                                                                            'evidence_fault': 'invalid_no_score',
                                                                            'failure': ['Cumulative '
                                                                                        'spill '
                                                                                        'exceeds '
                                                                                        'the '
                                                                                        'published '
                                                                                        'limit.',
                                                                                        'The cup '
                                                                                        'tips '
                                                                                        'beyond '
                                                                                        'its '
                                                                                        'published '
                                                                                        'failure '
                                                                                        'limit, or '
                                                                                        'either '
                                                                                        'vessel '
                                                                                        'falls '
                                                                                        'below the '
                                                                                        'table.'],
                                                                            'fluid': {'candidate': 'PhysX '
                                                                                                   'GPU '
                                                                                                   'PBD '
                                                                                                   'fluid',
                                                                                      'forbidden_shortcuts': ['Animated '
                                                                                                              'water '
                                                                                                              'stream',
                                                                                                              'Rigid '
                                                                                                              'blue '
                                                                                                              'water '
                                                                                                              'block',
                                                                                                              'Changing '
                                                                                                              'a '
                                                                                                              'water-level '
                                                                                                              'scalar '
                                                                                                              'from '
                                                                                                              'kettle '
                                                                                                              'angle',
                                                                                                              'Spawning '
                                                                                                              'fluid '
                                                                                                              'at '
                                                                                                              'the '
                                                                                                              'spout',
                                                                                                              'Deleting '
                                                                                                              'spilled '
                                                                                                              'particles',
                                                                                                              'Kinematic '
                                                                                                              'kettle '
                                                                                                              'during '
                                                                                                              'robot '
                                                                                                              'pouring'],
                                                                                      'native_adapter': 'living_water_observation_v1',
                                                                                      'quantitative_water_load_calibration_required_for_scene_use': False,
                                                                                      'rendering': 'Particle '
                                                                                                   'surface '
                                                                                                   'reconstruction '
                                                                                                   'is '
                                                                                                   'visual '
                                                                                                   'only; '
                                                                                                   'score '
                                                                                                   'from '
                                                                                                   'unsmoothed '
                                                                                                   'simulation '
                                                                                                   'state.'},
                                                                            'goal': {'cup_usable_capacity_ml': 200.0,
                                                                                     'initial_water_ml': 250.0,
                                                                                     'kettle_usable_capacity_ml': 500.0,
                                                                                     'max_spill_ml': 5.0,
                                                                                     'status': 'published_nominal_quantities',
                                                                                     'target_interval_ml': [90.0,
                                                                                                            110.0],
                                                                                     'target_tolerance_ml': 10.0,
                                                                                     'target_water_ml': 100.0},
                                                                            'instruction_zh': '单臂抓起水壶，向杯中倒入公布的目标水量，减少洒漏，再把壶正放回归位区并松手。',
                                                                            'measurement': {'conservation': 'Initial '
                                                                                                            'water '
                                                                                                            'minus '
                                                                                                            'current '
                                                                                                            'kettle '
                                                                                                            'and '
                                                                                                            'cup '
                                                                                                            'water '
                                                                                                            'is '
                                                                                                            'the '
                                                                                                            'outside-water '
                                                                                                            'difference; '
                                                                                                            'subtract '
                                                                                                            'in-flight/unresolved '
                                                                                                            'water '
                                                                                                            'to '
                                                                                                            'obtain '
                                                                                                            'current '
                                                                                                            'spill. '
                                                                                                            'Every '
                                                                                                            'original '
                                                                                                            'particle '
                                                                                                            'is '
                                                                                                            'assigned '
                                                                                                            'once. '
                                                                                                            'Cumulative '
                                                                                                            'spill '
                                                                                                            'is '
                                                                                                            'a '
                                                                                                            'separate '
                                                                                                            'history, '
                                                                                                            'never '
                                                                                                            'added '
                                                                                                            'to '
                                                                                                            'the '
                                                                                                            'current '
                                                                                                            'mass '
                                                                                                            'partition '
                                                                                                            'or '
                                                                                                            'reduced '
                                                                                                            'by '
                                                                                                            'recovery.',
                                                                                            'cup_water': 'Current '
                                                                                                         'original-water '
                                                                                                         'inventory '
                                                                                                         'contained '
                                                                                                         'inside '
                                                                                                         'the '
                                                                                                         'actual '
                                                                                                         'cup '
                                                                                                         'cavity '
                                                                                                         'below '
                                                                                                         'its '
                                                                                                         'rim, '
                                                                                                         'excluding '
                                                                                                         'exterior/airborne/escaped '
                                                                                                         'particles.',
                                                                                            'missing_inventory': 'Missing '
                                                                                                                 'particle '
                                                                                                                 'IDs, '
                                                                                                                 'lost '
                                                                                                                 'mass, '
                                                                                                                 'stale '
                                                                                                                 'readback, '
                                                                                                                 'reset '
                                                                                                                 'changes '
                                                                                                                 'or '
                                                                                                                 'incomplete '
                                                                                                                 'intervals '
                                                                                                                 'invalidate '
                                                                                                                 'the '
                                                                                                                 'episode, '
                                                                                                                 'never '
                                                                                                                 'count '
                                                                                                                 'as '
                                                                                                                 'low '
                                                                                                                 'spill.',
                                                                                            'public_feedback': 'Public '
                                                                                                               'task '
                                                                                                               'rules '
                                                                                                               'declare '
                                                                                                               'water '
                                                                                                               'targets '
                                                                                                               'and '
                                                                                                               'spill '
                                                                                                               'budget. '
                                                                                                               'The '
                                                                                                               'saved '
                                                                                                               'referee '
                                                                                                               'result '
                                                                                                               'reports '
                                                                                                               'initial, '
                                                                                                               'kettle '
                                                                                                               'and '
                                                                                                               'cup '
                                                                                                               'mL, '
                                                                                                               'water '
                                                                                                               'difference, '
                                                                                                               'unresolved '
                                                                                                               'mL, '
                                                                                                               'current '
                                                                                                               'and '
                                                                                                               'cumulative '
                                                                                                               'spill, '
                                                                                                               'and '
                                                                                                               'remaining '
                                                                                                               'spill '
                                                                                                               'budget; '
                                                                                                               'raw '
                                                                                                               'particle '
                                                                                                               'truth '
                                                                                                               'stays '
                                                                                                               'service-owned.',
                                                                                            'quantity_basis': 'Sum '
                                                                                                              'frozen '
                                                                                                              'per-particle '
                                                                                                              'physical '
                                                                                                              'mass, '
                                                                                                              'divide '
                                                                                                              'by '
                                                                                                              'frozen '
                                                                                                              'reference '
                                                                                                              'water '
                                                                                                              'density; '
                                                                                                              'report '
                                                                                                              'equivalent '
                                                                                                              'mL.',
                                                                                            'spill': 'Cumulative '
                                                                                                     'mass '
                                                                                                     'of '
                                                                                                     'unique '
                                                                                                     'original '
                                                                                                     'particles '
                                                                                                     'entering '
                                                                                                     'frozen '
                                                                                                     'spill '
                                                                                                     'regions '
                                                                                                     'on '
                                                                                                     'the '
                                                                                                     'tray/table/floor/exterior '
                                                                                                     'vessel '
                                                                                                     'surfaces, '
                                                                                                     'or '
                                                                                                     'escaping '
                                                                                                     'the '
                                                                                                     'task '
                                                                                                     'domain. '
                                                                                                     'Classify '
                                                                                                     'from '
                                                                                                     'raw '
                                                                                                     'particle '
                                                                                                     'trajectories '
                                                                                                     'and '
                                                                                                     'geometry; '
                                                                                                     'in-flight '
                                                                                                     'water '
                                                                                                     'remains '
                                                                                                     'pending '
                                                                                                     'until '
                                                                                                     'resolved.',
                                                                                            'spill_recovery': 'Recovering '
                                                                                                              'spilled '
                                                                                                              'water '
                                                                                                              'never '
                                                                                                              'reduces '
                                                                                                              'cumulative '
                                                                                                              'spill.',
                                                                                            'support_and_release': 'Native '
                                                                                                                   'rigid '
                                                                                                                   'contact '
                                                                                                                   'impulses; '
                                                                                                                   'sleeping '
                                                                                                                   'support '
                                                                                                                   'may '
                                                                                                                   'retain '
                                                                                                                   'its '
                                                                                                                   'last '
                                                                                                                   'observed '
                                                                                                                   'state '
                                                                                                                   'only '
                                                                                                                   'while '
                                                                                                                   'the '
                                                                                                                   'vessel '
                                                                                                                   'pose '
                                                                                                                   'is '
                                                                                                                   'unchanged '
                                                                                                                   'and '
                                                                                                                   'motion '
                                                                                                                   'is '
                                                                                                                   'below '
                                                                                                                   'the '
                                                                                                                   'published '
                                                                                                                   'thresholds. '
                                                                                                                   'Release '
                                                                                                                   'means '
                                                                                                                   'no '
                                                                                                                   'active '
                                                                                                                   'robot '
                                                                                                                   'contact '
                                                                                                                   'during '
                                                                                                                   'the '
                                                                                                                   'sampled '
                                                                                                                   'window.',
                                                                                            'unresolved_allowance_ml': 1.0,
                                                                                            'water_difference': 'Initial '
                                                                                                                'water '
                                                                                                                'minus '
                                                                                                                'current '
                                                                                                                'kettle '
                                                                                                                'and '
                                                                                                                'cup '
                                                                                                                'water '
                                                                                                                'equals '
                                                                                                                'current '
                                                                                                                'outside '
                                                                                                                'water, '
                                                                                                                'including '
                                                                                                                'in-flight/unresolved '
                                                                                                                'water. '
                                                                                                                'Subtract '
                                                                                                                'in-flight/unresolved '
                                                                                                                'water '
                                                                                                                'to '
                                                                                                                'obtain '
                                                                                                                'current '
                                                                                                                'spill; '
                                                                                                                'settled '
                                                                                                                'water '
                                                                                                                'difference '
                                                                                                                'equals '
                                                                                                                'current '
                                                                                                                'spill '
                                                                                                                'when '
                                                                                                                'no '
                                                                                                                'water '
                                                                                                                'remains '
                                                                                                                'unresolved.'},
                                                                            'objects': {'cup': 'Empty '
                                                                                               'free '
                                                                                               'dynamic '
                                                                                               'cup '
                                                                                               'with '
                                                                                               'a '
                                                                                               'real '
                                                                                               'open '
                                                                                               'cavity; '
                                                                                               'never '
                                                                                               'fixed '
                                                                                               'to '
                                                                                               'the '
                                                                                               'table.',
                                                                                        'kettle': 'Free '
                                                                                                  'dynamic '
                                                                                                  'rigid '
                                                                                                  'body '
                                                                                                  'with '
                                                                                                  'an '
                                                                                                  'open '
                                                                                                  'spout, '
                                                                                                  'real '
                                                                                                  'inner '
                                                                                                  'cavity, '
                                                                                                  'handle '
                                                                                                  'clearance '
                                                                                                  'and '
                                                                                                  'a '
                                                                                                  'physically '
                                                                                                  'represented '
                                                                                                  'secured '
                                                                                                  'lid '
                                                                                                  'if '
                                                                                                  'present.',
                                                                                        'spill_tray': 'Task-owned '
                                                                                                      'physical '
                                                                                                      'catch '
                                                                                                      'tray; '
                                                                                                      'all '
                                                                                                      'water '
                                                                                                      'landing '
                                                                                                      'outside '
                                                                                                      'the '
                                                                                                      'vessel '
                                                                                                      'interiors '
                                                                                                      'still '
                                                                                                      'counts '
                                                                                                      'as '
                                                                                                      'spill.',
                                                                                        'water': 'Fixed '
                                                                                                 'initial '
                                                                                                 'inventory '
                                                                                                 'of '
                                                                                                 'mass-bearing '
                                                                                                 'fluid '
                                                                                                 'particles; '
                                                                                                 'no '
                                                                                                 'spawning, '
                                                                                                 'deletion '
                                                                                                 'or '
                                                                                                 'teleportation '
                                                                                                 'after '
                                                                                                 'reset.'},
                                                                            'overfill': 'Outside '
                                                                                        'the '
                                                                                        'target '
                                                                                        'interval '
                                                                                        'is not '
                                                                                        'success; '
                                                                                        'draining/deleting '
                                                                                        'water or '
                                                                                        'automatic '
                                                                                        'reset is '
                                                                                        'forbidden.',
                                                                            'qualification': 'Practical '
                                                                                             'read-only '
                                                                                             'task '
                                                                                             'scoring. '
                                                                                             'Native '
                                                                                             'initialization '
                                                                                             'and '
                                                                                             'robot-policy '
                                                                                             'success '
                                                                                             'remain '
                                                                                             'separate '
                                                                                             'evidence; '
                                                                                             'quantitative '
                                                                                             'water-weight '
                                                                                             'calibration '
                                                                                             'is a '
                                                                                             'later '
                                                                                             'improvement.',
                                                                            'randomization': ['Initial '
                                                                                              'release '
                                                                                              'uses '
                                                                                              'the '
                                                                                              'frozen '
                                                                                              'vessel '
                                                                                              'poses; '
                                                                                              'episode '
                                                                                              'seed '
                                                                                              'and '
                                                                                              'nominal '
                                                                                              'water '
                                                                                              'goal '
                                                                                              'are '
                                                                                              'published.'],
                                                                            'recovery': ['Put down '
                                                                                         'and '
                                                                                         'regrasp '
                                                                                         'the '
                                                                                         'kettle.',
                                                                                         'Adjust '
                                                                                         'spout '
                                                                                         'alignment, '
                                                                                         'pour '
                                                                                         'rate and '
                                                                                         'wrist '
                                                                                         'angle.',
                                                                                         'Stop and '
                                                                                         'top up '
                                                                                         'an '
                                                                                         'underfilled '
                                                                                         'cup '
                                                                                         'through '
                                                                                         'actual '
                                                                                         'robot '
                                                                                         'actions.'],
                                                                            'schema': 'living_room_water_pouring_rules_v2',
                                                                            'stages': ['grasp_handle',
                                                                                       'lift_kettle',
                                                                                       'align_spout',
                                                                                       'tilt_and_pour',
                                                                                       'stop_flow',
                                                                                       'return_kettle',
                                                                                       'release_and_settle'],
                                                                            'status': 'prepared_practical_task',
                                                                            'success': {'all_of': ['Actual '
                                                                                                   'water '
                                                                                                   'transfer '
                                                                                                   'from '
                                                                                                   'the '
                                                                                                   'lifted '
                                                                                                   'kettle '
                                                                                                   'through '
                                                                                                   'its '
                                                                                                   'spout '
                                                                                                   'is '
                                                                                                   'observed; '
                                                                                                   'reset '
                                                                                                   'and '
                                                                                                   'visual '
                                                                                                   'appearance '
                                                                                                   'alone '
                                                                                                   'earn '
                                                                                                   'no '
                                                                                                   'credit.',
                                                                                                   'Retained '
                                                                                                   'cup '
                                                                                                   'water '
                                                                                                   'lies '
                                                                                                   'within '
                                                                                                   'the '
                                                                                                   'published '
                                                                                                   'target '
                                                                                                   'interval, '
                                                                                                   'and '
                                                                                                   'cumulative '
                                                                                                   'spill '
                                                                                                   'does '
                                                                                                   'not '
                                                                                                   'exceed '
                                                                                                   'its '
                                                                                                   'budget.',
                                                                                                   'Cup '
                                                                                                   'remains '
                                                                                                   'upright '
                                                                                                   'and '
                                                                                                   'supported; '
                                                                                                   'kettle '
                                                                                                   'is '
                                                                                                   'upright '
                                                                                                   'and '
                                                                                                   'supported '
                                                                                                   'inside '
                                                                                                   'its '
                                                                                                   'designated '
                                                                                                   'return '
                                                                                                   'zone.',
                                                                                                   'Hand '
                                                                                                   'contacts '
                                                                                                   'with '
                                                                                                   'both '
                                                                                                   'vessels '
                                                                                                   'have '
                                                                                                   'ended; '
                                                                                                   'no '
                                                                                                   'vessel '
                                                                                                   'is '
                                                                                                   'supported '
                                                                                                   'by '
                                                                                                   'the '
                                                                                                   'robot.',
                                                                                                   'Cup '
                                                                                                   'volume '
                                                                                                   'and '
                                                                                                   'released '
                                                                                                   'vessel '
                                                                                                   'poses '
                                                                                                   'meet '
                                                                                                   'the '
                                                                                                   'published '
                                                                                                   'sampled '
                                                                                                   'settling '
                                                                                                   'thresholds '
                                                                                                   'throughout '
                                                                                                   'a '
                                                                                                   'complete '
                                                                                                   '0.3-second '
                                                                                                   'window; '
                                                                                                   'at '
                                                                                                   'most '
                                                                                                   '1 '
                                                                                                   'mL '
                                                                                                   'may '
                                                                                                   'remain '
                                                                                                   'unresolved.'],
                                                                                        'reset_credit': False,
                                                                                        'stable_duration_s': 0.3,
                                                                                        'thresholds': {'dwell_s': 0.3,
                                                                                                       'failure_tilt_rad': 1.7453292519943295,
                                                                                                       'maximum_angular_speed_rad_s': 0.2,
                                                                                                       'maximum_cup_volume_range_ml': 2.0,
                                                                                                       'maximum_linear_speed_m_s': 0.03,
                                                                                                       'maximum_observation_gap_s': 0.1,
                                                                                                       'maximum_robot_contact_force_n': 0.01,
                                                                                                       'maximum_stable_orientation_range_rad': 0.1,
                                                                                                       'maximum_stable_position_range_m': 0.01,
                                                                                                       'maximum_tilt_rad': 0.2617993877991494,
                                                                                                       'maximum_unresolved_ml': 1.0,
                                                                                                       'minimum_lift_m': 0.02,
                                                                                                       'minimum_spout_exit_ml': 1.0,
                                                                                                       'minimum_support_force_n': 0.01,
                                                                                                       'return_xy_radius_m': 0.06,
                                                                                                       'return_z_tolerance_m': 0.015,
                                                                                                       'sleep_support_orientation_tolerance_rad': 0.03,
                                                                                                       'sleep_support_position_tolerance_m': 0.002}}}},
                                                     'qualification': {'status': 'prepared_unvalidated',
                                                                       'native_preflight': False,
                                                                       'manipulation': False,
                                                                       'physics_accepted': False,
                                                                       'public_http_smoke': False,
                                                                       'public_functions_smoke': False,
                                                                       'scope': 'Frozen randomized '
                                                                                'inputs; '
                                                                                'separately '
                                                                                'recorded '
                                                                                'development '
                                                                                'checks are not '
                                                                                'runtime admission '
                                                                                'data.'}},
                                      'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/initial_randomization_v1/dual_same_side',
                                                                            'manifest_sha256': '5d644701fbc63edd6b65300f063d56a33904bd1afd6739f1308e1e1f6f0fc353'},
                                                         'release_schema': 'living_room_randomized_release.v1',
                                                         'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                         'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                'assembly_version': 'isaacsim_6_1_v2',
                                                                                'articulations': 2,
                                                                                'active_sides': ['left',
                                                                                                 'right']},
                                                         'layout': {'robot_layout_id': 'living_room_dual_same_side_left_right_sharpa_v1',
                                                                    'placements': {'left': {'base_position_m': [0.35,
                                                                                                                -0.68,
                                                                                                                0.78],
                                                                                            'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                     0.0,
                                                                                                                     0.0,
                                                                                                                     0.7071067811865475],
                                                                                            'base_yaw_rad': 1.5707963267948966,
                                                                                            'hand_chirality': 'left'},
                                                                                   'right': {'base_position_m': [0.95,
                                                                                                                 -0.68,
                                                                                                                 0.78],
                                                                                             'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                      0.0,
                                                                                                                      0.0,
                                                                                                                      0.7071067811865475],
                                                                                             'base_yaw_rad': 1.5707963267948966,
                                                                                             'hand_chirality': 'right'}},
                                                                    'reference_surface': 'fixed '
                                                                                         'room '
                                                                                         'task '
                                                                                         'table',
                                                                    'frame': 'world',
                                                                    'units': 'metres',
                                                                    'control_scope': 'one_agent_58_dof'},
                                                         'objects': {'intended_roles': ['spouted_water_kettle',
                                                                                        'free_cup',
                                                                                        'water_particles',
                                                                                        'spill_tray'],
                                                                     'asset_bindings': 'Exact '
                                                                                       'per-seed '
                                                                                       'USD, '
                                                                                       'texture, '
                                                                                       'collider '
                                                                                       'and mass '
                                                                                       'inputs are '
                                                                                       'pinned in '
                                                                                       'the '
                                                                                       'release.'},
                                                         'rules': {'intent': '抓起带把手水壶，向杯中倒入目标水量，控制洒漏，再将壶放回并松手。',
                                                                   'implementation': 'scenes.workrooms.living_room.tasks.water_pouring.rules',
                                                                   'contract': {'asset_freeze_required': ['Vessel '
                                                                                                          'mesh/cavity/spout/handle '
                                                                                                          'dimensions, '
                                                                                                          'dry '
                                                                                                          'masses '
                                                                                                          'and '
                                                                                                          'inertia',
                                                                                                          'Water '
                                                                                                          'density, '
                                                                                                          'particle '
                                                                                                          'masses/spacing, '
                                                                                                          'fluid '
                                                                                                          'material '
                                                                                                          'and '
                                                                                                          'solver '
                                                                                                          'settings',
                                                                                                          'Collision '
                                                                                                          'thickness/resolution, '
                                                                                                          'finite '
                                                                                                          'leak '
                                                                                                          'and '
                                                                                                          'mass-accounting '
                                                                                                          'tolerances',
                                                                                                          'Cup '
                                                                                                          'occupancy, '
                                                                                                          'flight/settling, '
                                                                                                          'spill-region '
                                                                                                          'classification, '
                                                                                                          'interval '
                                                                                                          'coverage, '
                                                                                                          'upright '
                                                                                                          'and '
                                                                                                          'return-zone '
                                                                                                          'thresholds',
                                                                                                          'Readback '
                                                                                                          'provenance; '
                                                                                                          'known '
                                                                                                          'quantitative '
                                                                                                          'water-load '
                                                                                                          'limitation '
                                                                                                          'recorded '
                                                                                                          'separately '
                                                                                                          'from '
                                                                                                          'task '
                                                                                                          'scoring'],
                                                                                'budget_exhaustion': 'incomplete',
                                                                                'competitive': False,
                                                                                'evidence_fault': 'invalid_no_score',
                                                                                'failure': ['Cumulative '
                                                                                            'spill '
                                                                                            'exceeds '
                                                                                            'the '
                                                                                            'published '
                                                                                            'limit.',
                                                                                            'The '
                                                                                            'cup '
                                                                                            'tips '
                                                                                            'beyond '
                                                                                            'its '
                                                                                            'published '
                                                                                            'failure '
                                                                                            'limit, '
                                                                                            'or '
                                                                                            'either '
                                                                                            'vessel '
                                                                                            'falls '
                                                                                            'below '
                                                                                            'the '
                                                                                            'table.'],
                                                                                'fluid': {'candidate': 'PhysX '
                                                                                                       'GPU '
                                                                                                       'PBD '
                                                                                                       'fluid',
                                                                                          'forbidden_shortcuts': ['Animated '
                                                                                                                  'water '
                                                                                                                  'stream',
                                                                                                                  'Rigid '
                                                                                                                  'blue '
                                                                                                                  'water '
                                                                                                                  'block',
                                                                                                                  'Changing '
                                                                                                                  'a '
                                                                                                                  'water-level '
                                                                                                                  'scalar '
                                                                                                                  'from '
                                                                                                                  'kettle '
                                                                                                                  'angle',
                                                                                                                  'Spawning '
                                                                                                                  'fluid '
                                                                                                                  'at '
                                                                                                                  'the '
                                                                                                                  'spout',
                                                                                                                  'Deleting '
                                                                                                                  'spilled '
                                                                                                                  'particles',
                                                                                                                  'Kinematic '
                                                                                                                  'kettle '
                                                                                                                  'during '
                                                                                                                  'robot '
                                                                                                                  'pouring'],
                                                                                          'native_adapter': 'living_water_observation_v1',
                                                                                          'quantitative_water_load_calibration_required_for_scene_use': False,
                                                                                          'rendering': 'Particle '
                                                                                                       'surface '
                                                                                                       'reconstruction '
                                                                                                       'is '
                                                                                                       'visual '
                                                                                                       'only; '
                                                                                                       'score '
                                                                                                       'from '
                                                                                                       'unsmoothed '
                                                                                                       'simulation '
                                                                                                       'state.'},
                                                                                'goal': {'cup_usable_capacity_ml': 200.0,
                                                                                         'initial_water_ml': 250.0,
                                                                                         'kettle_usable_capacity_ml': 500.0,
                                                                                         'max_spill_ml': 5.0,
                                                                                         'status': 'published_nominal_quantities',
                                                                                         'target_interval_ml': [90.0,
                                                                                                                110.0],
                                                                                         'target_tolerance_ml': 10.0,
                                                                                         'target_water_ml': 100.0},
                                                                                'instruction_zh': '用机器人抓起水壶，向杯中倒入公布的目标水量，减少洒漏，再把壶正放回归位区并松手。',
                                                                                'measurement': {'conservation': 'Initial '
                                                                                                                'water '
                                                                                                                'minus '
                                                                                                                'current '
                                                                                                                'kettle '
                                                                                                                'and '
                                                                                                                'cup '
                                                                                                                'water '
                                                                                                                'is '
                                                                                                                'the '
                                                                                                                'outside-water '
                                                                                                                'difference; '
                                                                                                                'subtract '
                                                                                                                'in-flight/unresolved '
                                                                                                                'water '
                                                                                                                'to '
                                                                                                                'obtain '
                                                                                                                'current '
                                                                                                                'spill. '
                                                                                                                'Every '
                                                                                                                'original '
                                                                                                                'particle '
                                                                                                                'is '
                                                                                                                'assigned '
                                                                                                                'once. '
                                                                                                                'Cumulative '
                                                                                                                'spill '
                                                                                                                'is '
                                                                                                                'a '
                                                                                                                'separate '
                                                                                                                'history, '
                                                                                                                'never '
                                                                                                                'added '
                                                                                                                'to '
                                                                                                                'the '
                                                                                                                'current '
                                                                                                                'mass '
                                                                                                                'partition '
                                                                                                                'or '
                                                                                                                'reduced '
                                                                                                                'by '
                                                                                                                'recovery.',
                                                                                                'cup_water': 'Current '
                                                                                                             'original-water '
                                                                                                             'inventory '
                                                                                                             'contained '
                                                                                                             'inside '
                                                                                                             'the '
                                                                                                             'actual '
                                                                                                             'cup '
                                                                                                             'cavity '
                                                                                                             'below '
                                                                                                             'its '
                                                                                                             'rim, '
                                                                                                             'excluding '
                                                                                                             'exterior/airborne/escaped '
                                                                                                             'particles.',
                                                                                                'missing_inventory': 'Missing '
                                                                                                                     'particle '
                                                                                                                     'IDs, '
                                                                                                                     'lost '
                                                                                                                     'mass, '
                                                                                                                     'stale '
                                                                                                                     'readback, '
                                                                                                                     'reset '
                                                                                                                     'changes '
                                                                                                                     'or '
                                                                                                                     'incomplete '
                                                                                                                     'intervals '
                                                                                                                     'invalidate '
                                                                                                                     'the '
                                                                                                                     'episode, '
                                                                                                                     'never '
                                                                                                                     'count '
                                                                                                                     'as '
                                                                                                                     'low '
                                                                                                                     'spill.',
                                                                                                'public_feedback': 'Public '
                                                                                                                   'task '
                                                                                                                   'rules '
                                                                                                                   'declare '
                                                                                                                   'water '
                                                                                                                   'targets '
                                                                                                                   'and '
                                                                                                                   'spill '
                                                                                                                   'budget. '
                                                                                                                   'The '
                                                                                                                   'saved '
                                                                                                                   'referee '
                                                                                                                   'result '
                                                                                                                   'reports '
                                                                                                                   'initial, '
                                                                                                                   'kettle '
                                                                                                                   'and '
                                                                                                                   'cup '
                                                                                                                   'mL, '
                                                                                                                   'water '
                                                                                                                   'difference, '
                                                                                                                   'unresolved '
                                                                                                                   'mL, '
                                                                                                                   'current '
                                                                                                                   'and '
                                                                                                                   'cumulative '
                                                                                                                   'spill, '
                                                                                                                   'and '
                                                                                                                   'remaining '
                                                                                                                   'spill '
                                                                                                                   'budget; '
                                                                                                                   'raw '
                                                                                                                   'particle '
                                                                                                                   'truth '
                                                                                                                   'stays '
                                                                                                                   'service-owned.',
                                                                                                'quantity_basis': 'Sum '
                                                                                                                  'frozen '
                                                                                                                  'per-particle '
                                                                                                                  'physical '
                                                                                                                  'mass, '
                                                                                                                  'divide '
                                                                                                                  'by '
                                                                                                                  'frozen '
                                                                                                                  'reference '
                                                                                                                  'water '
                                                                                                                  'density; '
                                                                                                                  'report '
                                                                                                                  'equivalent '
                                                                                                                  'mL.',
                                                                                                'spill': 'Cumulative '
                                                                                                         'mass '
                                                                                                         'of '
                                                                                                         'unique '
                                                                                                         'original '
                                                                                                         'particles '
                                                                                                         'entering '
                                                                                                         'frozen '
                                                                                                         'spill '
                                                                                                         'regions '
                                                                                                         'on '
                                                                                                         'the '
                                                                                                         'tray/table/floor/exterior '
                                                                                                         'vessel '
                                                                                                         'surfaces, '
                                                                                                         'or '
                                                                                                         'escaping '
                                                                                                         'the '
                                                                                                         'task '
                                                                                                         'domain. '
                                                                                                         'Classify '
                                                                                                         'from '
                                                                                                         'raw '
                                                                                                         'particle '
                                                                                                         'trajectories '
                                                                                                         'and '
                                                                                                         'geometry; '
                                                                                                         'in-flight '
                                                                                                         'water '
                                                                                                         'remains '
                                                                                                         'pending '
                                                                                                         'until '
                                                                                                         'resolved.',
                                                                                                'spill_recovery': 'Recovering '
                                                                                                                  'spilled '
                                                                                                                  'water '
                                                                                                                  'never '
                                                                                                                  'reduces '
                                                                                                                  'cumulative '
                                                                                                                  'spill.',
                                                                                                'support_and_release': 'Native '
                                                                                                                       'rigid '
                                                                                                                       'contact '
                                                                                                                       'impulses; '
                                                                                                                       'sleeping '
                                                                                                                       'support '
                                                                                                                       'may '
                                                                                                                       'retain '
                                                                                                                       'its '
                                                                                                                       'last '
                                                                                                                       'observed '
                                                                                                                       'state '
                                                                                                                       'only '
                                                                                                                       'while '
                                                                                                                       'the '
                                                                                                                       'vessel '
                                                                                                                       'pose '
                                                                                                                       'is '
                                                                                                                       'unchanged '
                                                                                                                       'and '
                                                                                                                       'motion '
                                                                                                                       'is '
                                                                                                                       'below '
                                                                                                                       'the '
                                                                                                                       'published '
                                                                                                                       'thresholds. '
                                                                                                                       'Release '
                                                                                                                       'means '
                                                                                                                       'no '
                                                                                                                       'active '
                                                                                                                       'robot '
                                                                                                                       'contact '
                                                                                                                       'during '
                                                                                                                       'the '
                                                                                                                       'sampled '
                                                                                                                       'window.',
                                                                                                'unresolved_allowance_ml': 1.0,
                                                                                                'water_difference': 'Initial '
                                                                                                                    'water '
                                                                                                                    'minus '
                                                                                                                    'current '
                                                                                                                    'kettle '
                                                                                                                    'and '
                                                                                                                    'cup '
                                                                                                                    'water '
                                                                                                                    'equals '
                                                                                                                    'current '
                                                                                                                    'outside '
                                                                                                                    'water, '
                                                                                                                    'including '
                                                                                                                    'in-flight/unresolved '
                                                                                                                    'water. '
                                                                                                                    'Subtract '
                                                                                                                    'in-flight/unresolved '
                                                                                                                    'water '
                                                                                                                    'to '
                                                                                                                    'obtain '
                                                                                                                    'current '
                                                                                                                    'spill; '
                                                                                                                    'settled '
                                                                                                                    'water '
                                                                                                                    'difference '
                                                                                                                    'equals '
                                                                                                                    'current '
                                                                                                                    'spill '
                                                                                                                    'when '
                                                                                                                    'no '
                                                                                                                    'water '
                                                                                                                    'remains '
                                                                                                                    'unresolved.'},
                                                                                'objects': {'cup': 'Empty '
                                                                                                   'free '
                                                                                                   'dynamic '
                                                                                                   'cup '
                                                                                                   'with '
                                                                                                   'a '
                                                                                                   'real '
                                                                                                   'open '
                                                                                                   'cavity; '
                                                                                                   'never '
                                                                                                   'fixed '
                                                                                                   'to '
                                                                                                   'the '
                                                                                                   'table.',
                                                                                            'kettle': 'Free '
                                                                                                      'dynamic '
                                                                                                      'rigid '
                                                                                                      'body '
                                                                                                      'with '
                                                                                                      'an '
                                                                                                      'open '
                                                                                                      'spout, '
                                                                                                      'real '
                                                                                                      'inner '
                                                                                                      'cavity, '
                                                                                                      'handle '
                                                                                                      'clearance '
                                                                                                      'and '
                                                                                                      'a '
                                                                                                      'physically '
                                                                                                      'represented '
                                                                                                      'secured '
                                                                                                      'lid '
                                                                                                      'if '
                                                                                                      'present.',
                                                                                            'spill_tray': 'Task-owned '
                                                                                                          'physical '
                                                                                                          'catch '
                                                                                                          'tray; '
                                                                                                          'all '
                                                                                                          'water '
                                                                                                          'landing '
                                                                                                          'outside '
                                                                                                          'the '
                                                                                                          'vessel '
                                                                                                          'interiors '
                                                                                                          'still '
                                                                                                          'counts '
                                                                                                          'as '
                                                                                                          'spill.',
                                                                                            'water': 'Fixed '
                                                                                                     'initial '
                                                                                                     'inventory '
                                                                                                     'of '
                                                                                                     'mass-bearing '
                                                                                                     'fluid '
                                                                                                     'particles; '
                                                                                                     'no '
                                                                                                     'spawning, '
                                                                                                     'deletion '
                                                                                                     'or '
                                                                                                     'teleportation '
                                                                                                     'after '
                                                                                                     'reset.'},
                                                                                'overfill': 'Outside '
                                                                                            'the '
                                                                                            'target '
                                                                                            'interval '
                                                                                            'is '
                                                                                            'not '
                                                                                            'success; '
                                                                                            'draining/deleting '
                                                                                            'water '
                                                                                            'or '
                                                                                            'automatic '
                                                                                            'reset '
                                                                                            'is '
                                                                                            'forbidden.',
                                                                                'qualification': 'Practical '
                                                                                                 'read-only '
                                                                                                 'task '
                                                                                                 'scoring. '
                                                                                                 'Native '
                                                                                                 'initialization '
                                                                                                 'and '
                                                                                                 'robot-policy '
                                                                                                 'success '
                                                                                                 'remain '
                                                                                                 'separate '
                                                                                                 'evidence; '
                                                                                                 'quantitative '
                                                                                                 'water-weight '
                                                                                                 'calibration '
                                                                                                 'is '
                                                                                                 'a '
                                                                                                 'later '
                                                                                                 'improvement.',
                                                                                'randomization': ['Initial '
                                                                                                  'release '
                                                                                                  'uses '
                                                                                                  'the '
                                                                                                  'frozen '
                                                                                                  'vessel '
                                                                                                  'poses; '
                                                                                                  'episode '
                                                                                                  'seed '
                                                                                                  'and '
                                                                                                  'nominal '
                                                                                                  'water '
                                                                                                  'goal '
                                                                                                  'are '
                                                                                                  'published.'],
                                                                                'recovery': ['Put '
                                                                                             'down '
                                                                                             'and '
                                                                                             'regrasp '
                                                                                             'the '
                                                                                             'kettle.',
                                                                                             'Adjust '
                                                                                             'spout '
                                                                                             'alignment, '
                                                                                             'pour '
                                                                                             'rate '
                                                                                             'and '
                                                                                             'wrist '
                                                                                             'angle.',
                                                                                             'Stop '
                                                                                             'and '
                                                                                             'top '
                                                                                             'up '
                                                                                             'an '
                                                                                             'underfilled '
                                                                                             'cup '
                                                                                             'through '
                                                                                             'actual '
                                                                                             'robot '
                                                                                             'actions.'],
                                                                                'schema': 'living_room_water_pouring_rules_v2',
                                                                                'stages': ['grasp_handle',
                                                                                           'lift_kettle',
                                                                                           'align_spout',
                                                                                           'tilt_and_pour',
                                                                                           'stop_flow',
                                                                                           'return_kettle',
                                                                                           'release_and_settle'],
                                                                                'status': 'prepared_practical_task',
                                                                                'success': {'all_of': ['Actual '
                                                                                                       'water '
                                                                                                       'transfer '
                                                                                                       'from '
                                                                                                       'the '
                                                                                                       'lifted '
                                                                                                       'kettle '
                                                                                                       'through '
                                                                                                       'its '
                                                                                                       'spout '
                                                                                                       'is '
                                                                                                       'observed; '
                                                                                                       'reset '
                                                                                                       'and '
                                                                                                       'visual '
                                                                                                       'appearance '
                                                                                                       'alone '
                                                                                                       'earn '
                                                                                                       'no '
                                                                                                       'credit.',
                                                                                                       'Retained '
                                                                                                       'cup '
                                                                                                       'water '
                                                                                                       'lies '
                                                                                                       'within '
                                                                                                       'the '
                                                                                                       'published '
                                                                                                       'target '
                                                                                                       'interval, '
                                                                                                       'and '
                                                                                                       'cumulative '
                                                                                                       'spill '
                                                                                                       'does '
                                                                                                       'not '
                                                                                                       'exceed '
                                                                                                       'its '
                                                                                                       'budget.',
                                                                                                       'Cup '
                                                                                                       'remains '
                                                                                                       'upright '
                                                                                                       'and '
                                                                                                       'supported; '
                                                                                                       'kettle '
                                                                                                       'is '
                                                                                                       'upright '
                                                                                                       'and '
                                                                                                       'supported '
                                                                                                       'inside '
                                                                                                       'its '
                                                                                                       'designated '
                                                                                                       'return '
                                                                                                       'zone.',
                                                                                                       'Hand '
                                                                                                       'contacts '
                                                                                                       'with '
                                                                                                       'both '
                                                                                                       'vessels '
                                                                                                       'have '
                                                                                                       'ended; '
                                                                                                       'no '
                                                                                                       'vessel '
                                                                                                       'is '
                                                                                                       'supported '
                                                                                                       'by '
                                                                                                       'the '
                                                                                                       'robot.',
                                                                                                       'Cup '
                                                                                                       'volume '
                                                                                                       'and '
                                                                                                       'released '
                                                                                                       'vessel '
                                                                                                       'poses '
                                                                                                       'meet '
                                                                                                       'the '
                                                                                                       'published '
                                                                                                       'sampled '
                                                                                                       'settling '
                                                                                                       'thresholds '
                                                                                                       'throughout '
                                                                                                       'a '
                                                                                                       'complete '
                                                                                                       '0.3-second '
                                                                                                       'window; '
                                                                                                       'at '
                                                                                                       'most '
                                                                                                       '1 '
                                                                                                       'mL '
                                                                                                       'may '
                                                                                                       'remain '
                                                                                                       'unresolved.'],
                                                                                            'reset_credit': False,
                                                                                            'stable_duration_s': 0.3,
                                                                                            'thresholds': {'dwell_s': 0.3,
                                                                                                           'failure_tilt_rad': 1.7453292519943295,
                                                                                                           'maximum_angular_speed_rad_s': 0.2,
                                                                                                           'maximum_cup_volume_range_ml': 2.0,
                                                                                                           'maximum_linear_speed_m_s': 0.03,
                                                                                                           'maximum_observation_gap_s': 0.1,
                                                                                                           'maximum_robot_contact_force_n': 0.01,
                                                                                                           'maximum_stable_orientation_range_rad': 0.1,
                                                                                                           'maximum_stable_position_range_m': 0.01,
                                                                                                           'maximum_tilt_rad': 0.2617993877991494,
                                                                                                           'maximum_unresolved_ml': 1.0,
                                                                                                           'minimum_lift_m': 0.02,
                                                                                                           'minimum_spout_exit_ml': 1.0,
                                                                                                           'minimum_support_force_n': 0.01,
                                                                                                           'return_xy_radius_m': 0.06,
                                                                                                           'return_z_tolerance_m': 0.015,
                                                                                                           'sleep_support_orientation_tolerance_rad': 0.03,
                                                                                                           'sleep_support_position_tolerance_m': 0.002}}}},
                                                         'qualification': {'status': 'prepared_unvalidated',
                                                                           'native_preflight': False,
                                                                           'manipulation': False,
                                                                           'physics_accepted': False,
                                                                           'public_http_smoke': False,
                                                                           'public_functions_smoke': False,
                                                                           'scope': 'Frozen '
                                                                                    'randomized '
                                                                                    'inputs; '
                                                                                    'separately '
                                                                                    'recorded '
                                                                                    'development '
                                                                                    'checks are '
                                                                                    'not runtime '
                                                                                    'admission '
                                                                                    'data.'}},
                                      'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/initial_randomization_v1/dual_opposed',
                                                                          'manifest_sha256': '32adf89e6d9eef7cfce02c6ef306472a2e477ac9f214f4550241386081bb409b'},
                                                       'release_schema': 'living_room_randomized_release.v1',
                                                       'release_schema_version': 'isaacsim-living-room-randomized-release-v1',
                                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                                              'articulations': 2,
                                                                              'active_sides': ['left',
                                                                                               'right']},
                                                       'layout': {'robot_layout_id': 'living_room_dual_opposed_left_right_sharpa_v1',
                                                                  'placements': {'left': {'base_position_m': [0.65,
                                                                                                              0.76,
                                                                                                              0.78],
                                                                                          'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                   0.0,
                                                                                                                   0.0,
                                                                                                                   -0.7071067811865475],
                                                                                          'base_yaw_rad': -1.5707963267948966,
                                                                                          'hand_chirality': 'left'},
                                                                                 'right': {'base_position_m': [0.65,
                                                                                                               -0.76,
                                                                                                               0.78],
                                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                    0.0,
                                                                                                                    0.0,
                                                                                                                    0.7071067811865475],
                                                                                           'base_yaw_rad': 1.5707963267948966,
                                                                                           'hand_chirality': 'right'}},
                                                                  'reference_surface': 'fixed room '
                                                                                       'task table',
                                                                  'frame': 'world',
                                                                  'units': 'metres',
                                                                  'control_scope': 'one_agent_58_dof'},
                                                       'objects': {'intended_roles': ['spouted_water_kettle',
                                                                                      'free_cup',
                                                                                      'water_particles',
                                                                                      'spill_tray'],
                                                                   'asset_bindings': 'Exact '
                                                                                     'per-seed '
                                                                                     'USD, '
                                                                                     'texture, '
                                                                                     'collider and '
                                                                                     'mass inputs '
                                                                                     'are pinned '
                                                                                     'in the '
                                                                                     'release.'},
                                                       'rules': {'intent': '抓起带把手水壶，向杯中倒入目标水量，控制洒漏，再将壶放回并松手。',
                                                                 'implementation': 'scenes.workrooms.living_room.tasks.water_pouring.rules',
                                                                 'contract': {'asset_freeze_required': ['Vessel '
                                                                                                        'mesh/cavity/spout/handle '
                                                                                                        'dimensions, '
                                                                                                        'dry '
                                                                                                        'masses '
                                                                                                        'and '
                                                                                                        'inertia',
                                                                                                        'Water '
                                                                                                        'density, '
                                                                                                        'particle '
                                                                                                        'masses/spacing, '
                                                                                                        'fluid '
                                                                                                        'material '
                                                                                                        'and '
                                                                                                        'solver '
                                                                                                        'settings',
                                                                                                        'Collision '
                                                                                                        'thickness/resolution, '
                                                                                                        'finite '
                                                                                                        'leak '
                                                                                                        'and '
                                                                                                        'mass-accounting '
                                                                                                        'tolerances',
                                                                                                        'Cup '
                                                                                                        'occupancy, '
                                                                                                        'flight/settling, '
                                                                                                        'spill-region '
                                                                                                        'classification, '
                                                                                                        'interval '
                                                                                                        'coverage, '
                                                                                                        'upright '
                                                                                                        'and '
                                                                                                        'return-zone '
                                                                                                        'thresholds',
                                                                                                        'Readback '
                                                                                                        'provenance; '
                                                                                                        'known '
                                                                                                        'quantitative '
                                                                                                        'water-load '
                                                                                                        'limitation '
                                                                                                        'recorded '
                                                                                                        'separately '
                                                                                                        'from '
                                                                                                        'task '
                                                                                                        'scoring'],
                                                                              'budget_exhaustion': 'incomplete',
                                                                              'competitive': False,
                                                                              'evidence_fault': 'invalid_no_score',
                                                                              'failure': ['Cumulative '
                                                                                          'spill '
                                                                                          'exceeds '
                                                                                          'the '
                                                                                          'published '
                                                                                          'limit.',
                                                                                          'The cup '
                                                                                          'tips '
                                                                                          'beyond '
                                                                                          'its '
                                                                                          'published '
                                                                                          'failure '
                                                                                          'limit, '
                                                                                          'or '
                                                                                          'either '
                                                                                          'vessel '
                                                                                          'falls '
                                                                                          'below '
                                                                                          'the '
                                                                                          'table.'],
                                                                              'fluid': {'candidate': 'PhysX '
                                                                                                     'GPU '
                                                                                                     'PBD '
                                                                                                     'fluid',
                                                                                        'forbidden_shortcuts': ['Animated '
                                                                                                                'water '
                                                                                                                'stream',
                                                                                                                'Rigid '
                                                                                                                'blue '
                                                                                                                'water '
                                                                                                                'block',
                                                                                                                'Changing '
                                                                                                                'a '
                                                                                                                'water-level '
                                                                                                                'scalar '
                                                                                                                'from '
                                                                                                                'kettle '
                                                                                                                'angle',
                                                                                                                'Spawning '
                                                                                                                'fluid '
                                                                                                                'at '
                                                                                                                'the '
                                                                                                                'spout',
                                                                                                                'Deleting '
                                                                                                                'spilled '
                                                                                                                'particles',
                                                                                                                'Kinematic '
                                                                                                                'kettle '
                                                                                                                'during '
                                                                                                                'robot '
                                                                                                                'pouring'],
                                                                                        'native_adapter': 'living_water_observation_v1',
                                                                                        'quantitative_water_load_calibration_required_for_scene_use': False,
                                                                                        'rendering': 'Particle '
                                                                                                     'surface '
                                                                                                     'reconstruction '
                                                                                                     'is '
                                                                                                     'visual '
                                                                                                     'only; '
                                                                                                     'score '
                                                                                                     'from '
                                                                                                     'unsmoothed '
                                                                                                     'simulation '
                                                                                                     'state.'},
                                                                              'goal': {'cup_usable_capacity_ml': 200.0,
                                                                                       'initial_water_ml': 250.0,
                                                                                       'kettle_usable_capacity_ml': 500.0,
                                                                                       'max_spill_ml': 5.0,
                                                                                       'status': 'published_nominal_quantities',
                                                                                       'target_interval_ml': [90.0,
                                                                                                              110.0],
                                                                                       'target_tolerance_ml': 10.0,
                                                                                       'target_water_ml': 100.0},
                                                                              'instruction_zh': '用机器人抓起水壶，向杯中倒入公布的目标水量，减少洒漏，再把壶正放回归位区并松手。',
                                                                              'measurement': {'conservation': 'Initial '
                                                                                                              'water '
                                                                                                              'minus '
                                                                                                              'current '
                                                                                                              'kettle '
                                                                                                              'and '
                                                                                                              'cup '
                                                                                                              'water '
                                                                                                              'is '
                                                                                                              'the '
                                                                                                              'outside-water '
                                                                                                              'difference; '
                                                                                                              'subtract '
                                                                                                              'in-flight/unresolved '
                                                                                                              'water '
                                                                                                              'to '
                                                                                                              'obtain '
                                                                                                              'current '
                                                                                                              'spill. '
                                                                                                              'Every '
                                                                                                              'original '
                                                                                                              'particle '
                                                                                                              'is '
                                                                                                              'assigned '
                                                                                                              'once. '
                                                                                                              'Cumulative '
                                                                                                              'spill '
                                                                                                              'is '
                                                                                                              'a '
                                                                                                              'separate '
                                                                                                              'history, '
                                                                                                              'never '
                                                                                                              'added '
                                                                                                              'to '
                                                                                                              'the '
                                                                                                              'current '
                                                                                                              'mass '
                                                                                                              'partition '
                                                                                                              'or '
                                                                                                              'reduced '
                                                                                                              'by '
                                                                                                              'recovery.',
                                                                                              'cup_water': 'Current '
                                                                                                           'original-water '
                                                                                                           'inventory '
                                                                                                           'contained '
                                                                                                           'inside '
                                                                                                           'the '
                                                                                                           'actual '
                                                                                                           'cup '
                                                                                                           'cavity '
                                                                                                           'below '
                                                                                                           'its '
                                                                                                           'rim, '
                                                                                                           'excluding '
                                                                                                           'exterior/airborne/escaped '
                                                                                                           'particles.',
                                                                                              'missing_inventory': 'Missing '
                                                                                                                   'particle '
                                                                                                                   'IDs, '
                                                                                                                   'lost '
                                                                                                                   'mass, '
                                                                                                                   'stale '
                                                                                                                   'readback, '
                                                                                                                   'reset '
                                                                                                                   'changes '
                                                                                                                   'or '
                                                                                                                   'incomplete '
                                                                                                                   'intervals '
                                                                                                                   'invalidate '
                                                                                                                   'the '
                                                                                                                   'episode, '
                                                                                                                   'never '
                                                                                                                   'count '
                                                                                                                   'as '
                                                                                                                   'low '
                                                                                                                   'spill.',
                                                                                              'public_feedback': 'Public '
                                                                                                                 'task '
                                                                                                                 'rules '
                                                                                                                 'declare '
                                                                                                                 'water '
                                                                                                                 'targets '
                                                                                                                 'and '
                                                                                                                 'spill '
                                                                                                                 'budget. '
                                                                                                                 'The '
                                                                                                                 'saved '
                                                                                                                 'referee '
                                                                                                                 'result '
                                                                                                                 'reports '
                                                                                                                 'initial, '
                                                                                                                 'kettle '
                                                                                                                 'and '
                                                                                                                 'cup '
                                                                                                                 'mL, '
                                                                                                                 'water '
                                                                                                                 'difference, '
                                                                                                                 'unresolved '
                                                                                                                 'mL, '
                                                                                                                 'current '
                                                                                                                 'and '
                                                                                                                 'cumulative '
                                                                                                                 'spill, '
                                                                                                                 'and '
                                                                                                                 'remaining '
                                                                                                                 'spill '
                                                                                                                 'budget; '
                                                                                                                 'raw '
                                                                                                                 'particle '
                                                                                                                 'truth '
                                                                                                                 'stays '
                                                                                                                 'service-owned.',
                                                                                              'quantity_basis': 'Sum '
                                                                                                                'frozen '
                                                                                                                'per-particle '
                                                                                                                'physical '
                                                                                                                'mass, '
                                                                                                                'divide '
                                                                                                                'by '
                                                                                                                'frozen '
                                                                                                                'reference '
                                                                                                                'water '
                                                                                                                'density; '
                                                                                                                'report '
                                                                                                                'equivalent '
                                                                                                                'mL.',
                                                                                              'spill': 'Cumulative '
                                                                                                       'mass '
                                                                                                       'of '
                                                                                                       'unique '
                                                                                                       'original '
                                                                                                       'particles '
                                                                                                       'entering '
                                                                                                       'frozen '
                                                                                                       'spill '
                                                                                                       'regions '
                                                                                                       'on '
                                                                                                       'the '
                                                                                                       'tray/table/floor/exterior '
                                                                                                       'vessel '
                                                                                                       'surfaces, '
                                                                                                       'or '
                                                                                                       'escaping '
                                                                                                       'the '
                                                                                                       'task '
                                                                                                       'domain. '
                                                                                                       'Classify '
                                                                                                       'from '
                                                                                                       'raw '
                                                                                                       'particle '
                                                                                                       'trajectories '
                                                                                                       'and '
                                                                                                       'geometry; '
                                                                                                       'in-flight '
                                                                                                       'water '
                                                                                                       'remains '
                                                                                                       'pending '
                                                                                                       'until '
                                                                                                       'resolved.',
                                                                                              'spill_recovery': 'Recovering '
                                                                                                                'spilled '
                                                                                                                'water '
                                                                                                                'never '
                                                                                                                'reduces '
                                                                                                                'cumulative '
                                                                                                                'spill.',
                                                                                              'support_and_release': 'Native '
                                                                                                                     'rigid '
                                                                                                                     'contact '
                                                                                                                     'impulses; '
                                                                                                                     'sleeping '
                                                                                                                     'support '
                                                                                                                     'may '
                                                                                                                     'retain '
                                                                                                                     'its '
                                                                                                                     'last '
                                                                                                                     'observed '
                                                                                                                     'state '
                                                                                                                     'only '
                                                                                                                     'while '
                                                                                                                     'the '
                                                                                                                     'vessel '
                                                                                                                     'pose '
                                                                                                                     'is '
                                                                                                                     'unchanged '
                                                                                                                     'and '
                                                                                                                     'motion '
                                                                                                                     'is '
                                                                                                                     'below '
                                                                                                                     'the '
                                                                                                                     'published '
                                                                                                                     'thresholds. '
                                                                                                                     'Release '
                                                                                                                     'means '
                                                                                                                     'no '
                                                                                                                     'active '
                                                                                                                     'robot '
                                                                                                                     'contact '
                                                                                                                     'during '
                                                                                                                     'the '
                                                                                                                     'sampled '
                                                                                                                     'window.',
                                                                                              'unresolved_allowance_ml': 1.0,
                                                                                              'water_difference': 'Initial '
                                                                                                                  'water '
                                                                                                                  'minus '
                                                                                                                  'current '
                                                                                                                  'kettle '
                                                                                                                  'and '
                                                                                                                  'cup '
                                                                                                                  'water '
                                                                                                                  'equals '
                                                                                                                  'current '
                                                                                                                  'outside '
                                                                                                                  'water, '
                                                                                                                  'including '
                                                                                                                  'in-flight/unresolved '
                                                                                                                  'water. '
                                                                                                                  'Subtract '
                                                                                                                  'in-flight/unresolved '
                                                                                                                  'water '
                                                                                                                  'to '
                                                                                                                  'obtain '
                                                                                                                  'current '
                                                                                                                  'spill; '
                                                                                                                  'settled '
                                                                                                                  'water '
                                                                                                                  'difference '
                                                                                                                  'equals '
                                                                                                                  'current '
                                                                                                                  'spill '
                                                                                                                  'when '
                                                                                                                  'no '
                                                                                                                  'water '
                                                                                                                  'remains '
                                                                                                                  'unresolved.'},
                                                                              'objects': {'cup': 'Empty '
                                                                                                 'free '
                                                                                                 'dynamic '
                                                                                                 'cup '
                                                                                                 'with '
                                                                                                 'a '
                                                                                                 'real '
                                                                                                 'open '
                                                                                                 'cavity; '
                                                                                                 'never '
                                                                                                 'fixed '
                                                                                                 'to '
                                                                                                 'the '
                                                                                                 'table.',
                                                                                          'kettle': 'Free '
                                                                                                    'dynamic '
                                                                                                    'rigid '
                                                                                                    'body '
                                                                                                    'with '
                                                                                                    'an '
                                                                                                    'open '
                                                                                                    'spout, '
                                                                                                    'real '
                                                                                                    'inner '
                                                                                                    'cavity, '
                                                                                                    'handle '
                                                                                                    'clearance '
                                                                                                    'and '
                                                                                                    'a '
                                                                                                    'physically '
                                                                                                    'represented '
                                                                                                    'secured '
                                                                                                    'lid '
                                                                                                    'if '
                                                                                                    'present.',
                                                                                          'spill_tray': 'Task-owned '
                                                                                                        'physical '
                                                                                                        'catch '
                                                                                                        'tray; '
                                                                                                        'all '
                                                                                                        'water '
                                                                                                        'landing '
                                                                                                        'outside '
                                                                                                        'the '
                                                                                                        'vessel '
                                                                                                        'interiors '
                                                                                                        'still '
                                                                                                        'counts '
                                                                                                        'as '
                                                                                                        'spill.',
                                                                                          'water': 'Fixed '
                                                                                                   'initial '
                                                                                                   'inventory '
                                                                                                   'of '
                                                                                                   'mass-bearing '
                                                                                                   'fluid '
                                                                                                   'particles; '
                                                                                                   'no '
                                                                                                   'spawning, '
                                                                                                   'deletion '
                                                                                                   'or '
                                                                                                   'teleportation '
                                                                                                   'after '
                                                                                                   'reset.'},
                                                                              'overfill': 'Outside '
                                                                                          'the '
                                                                                          'target '
                                                                                          'interval '
                                                                                          'is not '
                                                                                          'success; '
                                                                                          'draining/deleting '
                                                                                          'water '
                                                                                          'or '
                                                                                          'automatic '
                                                                                          'reset '
                                                                                          'is '
                                                                                          'forbidden.',
                                                                              'qualification': 'Practical '
                                                                                               'read-only '
                                                                                               'task '
                                                                                               'scoring. '
                                                                                               'Native '
                                                                                               'initialization '
                                                                                               'and '
                                                                                               'robot-policy '
                                                                                               'success '
                                                                                               'remain '
                                                                                               'separate '
                                                                                               'evidence; '
                                                                                               'quantitative '
                                                                                               'water-weight '
                                                                                               'calibration '
                                                                                               'is '
                                                                                               'a '
                                                                                               'later '
                                                                                               'improvement.',
                                                                              'randomization': ['Initial '
                                                                                                'release '
                                                                                                'uses '
                                                                                                'the '
                                                                                                'frozen '
                                                                                                'vessel '
                                                                                                'poses; '
                                                                                                'episode '
                                                                                                'seed '
                                                                                                'and '
                                                                                                'nominal '
                                                                                                'water '
                                                                                                'goal '
                                                                                                'are '
                                                                                                'published.'],
                                                                              'recovery': ['Put '
                                                                                           'down '
                                                                                           'and '
                                                                                           'regrasp '
                                                                                           'the '
                                                                                           'kettle.',
                                                                                           'Adjust '
                                                                                           'spout '
                                                                                           'alignment, '
                                                                                           'pour '
                                                                                           'rate '
                                                                                           'and '
                                                                                           'wrist '
                                                                                           'angle.',
                                                                                           'Stop '
                                                                                           'and '
                                                                                           'top up '
                                                                                           'an '
                                                                                           'underfilled '
                                                                                           'cup '
                                                                                           'through '
                                                                                           'actual '
                                                                                           'robot '
                                                                                           'actions.'],
                                                                              'schema': 'living_room_water_pouring_rules_v2',
                                                                              'stages': ['grasp_handle',
                                                                                         'lift_kettle',
                                                                                         'align_spout',
                                                                                         'tilt_and_pour',
                                                                                         'stop_flow',
                                                                                         'return_kettle',
                                                                                         'release_and_settle'],
                                                                              'status': 'prepared_practical_task',
                                                                              'success': {'all_of': ['Actual '
                                                                                                     'water '
                                                                                                     'transfer '
                                                                                                     'from '
                                                                                                     'the '
                                                                                                     'lifted '
                                                                                                     'kettle '
                                                                                                     'through '
                                                                                                     'its '
                                                                                                     'spout '
                                                                                                     'is '
                                                                                                     'observed; '
                                                                                                     'reset '
                                                                                                     'and '
                                                                                                     'visual '
                                                                                                     'appearance '
                                                                                                     'alone '
                                                                                                     'earn '
                                                                                                     'no '
                                                                                                     'credit.',
                                                                                                     'Retained '
                                                                                                     'cup '
                                                                                                     'water '
                                                                                                     'lies '
                                                                                                     'within '
                                                                                                     'the '
                                                                                                     'published '
                                                                                                     'target '
                                                                                                     'interval, '
                                                                                                     'and '
                                                                                                     'cumulative '
                                                                                                     'spill '
                                                                                                     'does '
                                                                                                     'not '
                                                                                                     'exceed '
                                                                                                     'its '
                                                                                                     'budget.',
                                                                                                     'Cup '
                                                                                                     'remains '
                                                                                                     'upright '
                                                                                                     'and '
                                                                                                     'supported; '
                                                                                                     'kettle '
                                                                                                     'is '
                                                                                                     'upright '
                                                                                                     'and '
                                                                                                     'supported '
                                                                                                     'inside '
                                                                                                     'its '
                                                                                                     'designated '
                                                                                                     'return '
                                                                                                     'zone.',
                                                                                                     'Hand '
                                                                                                     'contacts '
                                                                                                     'with '
                                                                                                     'both '
                                                                                                     'vessels '
                                                                                                     'have '
                                                                                                     'ended; '
                                                                                                     'no '
                                                                                                     'vessel '
                                                                                                     'is '
                                                                                                     'supported '
                                                                                                     'by '
                                                                                                     'the '
                                                                                                     'robot.',
                                                                                                     'Cup '
                                                                                                     'volume '
                                                                                                     'and '
                                                                                                     'released '
                                                                                                     'vessel '
                                                                                                     'poses '
                                                                                                     'meet '
                                                                                                     'the '
                                                                                                     'published '
                                                                                                     'sampled '
                                                                                                     'settling '
                                                                                                     'thresholds '
                                                                                                     'throughout '
                                                                                                     'a '
                                                                                                     'complete '
                                                                                                     '0.3-second '
                                                                                                     'window; '
                                                                                                     'at '
                                                                                                     'most '
                                                                                                     '1 '
                                                                                                     'mL '
                                                                                                     'may '
                                                                                                     'remain '
                                                                                                     'unresolved.'],
                                                                                          'reset_credit': False,
                                                                                          'stable_duration_s': 0.3,
                                                                                          'thresholds': {'dwell_s': 0.3,
                                                                                                         'failure_tilt_rad': 1.7453292519943295,
                                                                                                         'maximum_angular_speed_rad_s': 0.2,
                                                                                                         'maximum_cup_volume_range_ml': 2.0,
                                                                                                         'maximum_linear_speed_m_s': 0.03,
                                                                                                         'maximum_observation_gap_s': 0.1,
                                                                                                         'maximum_robot_contact_force_n': 0.01,
                                                                                                         'maximum_stable_orientation_range_rad': 0.1,
                                                                                                         'maximum_stable_position_range_m': 0.01,
                                                                                                         'maximum_tilt_rad': 0.2617993877991494,
                                                                                                         'maximum_unresolved_ml': 1.0,
                                                                                                         'minimum_lift_m': 0.02,
                                                                                                         'minimum_spout_exit_ml': 1.0,
                                                                                                         'minimum_support_force_n': 0.01,
                                                                                                         'return_xy_radius_m': 0.06,
                                                                                                         'return_z_tolerance_m': 0.015,
                                                                                                         'sleep_support_orientation_tolerance_rad': 0.03,
                                                                                                         'sleep_support_position_tolerance_m': 0.002}}}},
                                                       'qualification': {'status': 'prepared_unvalidated',
                                                                         'native_preflight': False,
                                                                         'manipulation': False,
                                                                         'physics_accepted': False,
                                                                         'public_http_smoke': False,
                                                                         'public_functions_smoke': False,
                                                                         'scope': 'Frozen '
                                                                                  'randomized '
                                                                                  'inputs; '
                                                                                  'separately '
                                                                                  'recorded '
                                                                                  'development '
                                                                                  'checks are not '
                                                                                  'runtime '
                                                                                  'admission '
                                                                                  'data.'}}}}}


def bind_definition(definition):
    """Select new inputs without transferring the parent's qualifications."""
    task_id = definition['task_id']
    if task_id not in RELEASES:
        return definition
    if definition['family'] != 'living_room' or task_id not in TASK_IDS:
        raise ValueError('Randomized catalog only supports the three ordinary living tasks')
    row = RELEASES[task_id]
    if set(row['configurations']) != set(CONFIGURATIONS):
        raise ValueError('Randomized task requires three frozen installation choices')
    selected = row['configurations']['single_arm']
    if selected['source_release'] != row['base_release']:
        raise ValueError('Randomized default differs from its single-arm release')
    result = deepcopy(definition)
    for key in ('source_release', 'release_schema', 'release_schema_version',
                'robot_requirements', 'qualification', 'objects', 'rules'):
        result[key] = deepcopy(selected[key])
    result['layout'].update(deepcopy(selected['layout']))
    result['layout']['initialization'] = (
        'Choose a frozen seed 0..9 before episode creation; seed 0 preserves the '
        'canonical scene, while 1..9 vary task-object poses, textures and real geometry.')
    return result


def publication_row(definition, releases):
    """Build input-only catalog data from verified new releases for one task."""
    from pathlib import Path
    from .common import file_hash, read_json
    from bench.isaacsim.release import validate

    task_id = definition['task_id']
    if task_id not in TASK_IDS or set(releases) != set(CONFIGURATIONS):
        raise ValueError('Publication needs three installations of a supported ordinary task')
    options = {}
    for configuration_id in CONFIGURATIONS:
        root = Path(releases[configuration_id]).resolve()
        manifest = validate(root)
        if (manifest.get('schema') != SCHEMA or manifest.get('schema_version') != VERSION
                or manifest.get('seed_values') != list(range(10))):
            raise ValueError('Publication requires the exact ten-case randomized release schema')
        if (manifest['task_id'], manifest['configuration_id']) != (task_id, configuration_id):
            raise ValueError('Publication task or installation differs from the frozen release')
        task = read_json(root/'task.json')
        original = definition['robot_configurations'][configuration_id]
        objects = deepcopy(original.get('objects', definition['objects']))
        objects.pop('asset_root', None)
        objects['asset_bindings'] = 'Exact per-seed USD, texture, collider and mass inputs are pinned in the release.'
        options[configuration_id] = {
            'source_release': {'path': str(root), 'manifest_sha256': file_hash(root/'release_manifest.json')},
            'release_schema': SCHEMA, 'release_schema_version': VERSION,
            'robot_requirements': {**(deepcopy(definition['robot_requirements'])
                                      if configuration_id == 'single_arm' else {}),
                                   **deepcopy(original['robot_requirements']),
                                   **({'active_sides': ['right']} if configuration_id == 'single_arm' else {})},
            'layout': deepcopy(original.get('layout', {})),
            'objects': objects,
            'rules': {**deepcopy(original.get('rules', definition['rules'])),
                      'contract': deepcopy(task['public_rules'])},
            'qualification': {'status': 'prepared_unvalidated', 'native_preflight': False,
                'manipulation': False, 'physics_accepted': False, 'public_http_smoke': False,
                'public_functions_smoke': False,
                'scope': 'Frozen randomized inputs; separately recorded development checks are not runtime admission data.'},
        }
    return {'family': 'living_room', 'task_id': task_id,
        'default_configuration_id': 'single_arm',
        'base_release': deepcopy(options['single_arm']['source_release']),
        'single_arm_robot_requirements': deepcopy(options['single_arm']['robot_requirements']),
        'single_arm_layout': deepcopy(options['single_arm']['layout']),
        'configurations': options}
