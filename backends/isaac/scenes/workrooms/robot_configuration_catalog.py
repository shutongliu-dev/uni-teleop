"""Immutable pins for reviewed tasks and their optional installations.

Generated from source-bound preparation indexes; evidence is attached separately
from immutable release manifests. Do not rebind a row to a newer base implicitly.
"""

from bench.paths import DATA_ROOT
RELEASES = {('mechanical_assembly', 'parts_sorting'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/parts_sorting/desktop_v1',
                                                             'manifest_sha256': '8dbdadd5d0a092f42ac37d9ec6eba55cd48ba726e867bdb3df92db95a784a08f'},
                                            'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                                  'placements': {'right': {'base_position_m': [-0.23,
                                                                                                               0.0,
                                                                                                               0.78],
                                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                                    0.0,
                                                                                                                    0.0,
                                                                                                                    0.0],
                                                                                           'base_yaw_rad': 0.0,
                                                                                           'hand_chirality': 'right'}},
                                                                  'reference_surface': 'original fixed task '
                                                                                       'installation',
                                                                  'frame': 'world',
                                                                  'units': 'metres',
                                                                  'control_scope': 'one_agent_29_dof'},
                                            'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/mechanical_assembly/parts_sorting/dual_same_side/v1',
                                                                                                     'manifest_sha256': 'b39c2647319f6943dae43819f7e1b919298fbf5a2aa215f15d8e04dd60cc5ee4'},
                                                                                  'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                  'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                  'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                                                         'articulations': 2,
                                                                                                         'active_sides': ['left',
                                                                                                                          'right']},
                                                                                  'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                                                             'placements': {'left': {'base_position_m': [-0.1,
                                                                                                                                         0.3,
                                                                                                                                         0.78],
                                                                                                                     'base_quaternion_wxyz': [1.0,
                                                                                                                                              0.0,
                                                                                                                                              0.0,
                                                                                                                                              0.0],
                                                                                                                     'base_yaw_rad': 0.0,
                                                                                                                     'hand_chirality': 'left'},
                                                                                                            'right': {'base_position_m': [-0.1,
                                                                                                                                          -0.3,
                                                                                                                                          0.78],
                                                                                                                      'base_quaternion_wxyz': [1.0,
                                                                                                                                               0.0,
                                                                                                                                               0.0,
                                                                                                                                               0.0],
                                                                                                                      'base_yaw_rad': 0.0,
                                                                                                                      'hand_chirality': 'right'}},
                                                                                             'reference_surface': 'fixed '
                                                                                                                  'room '
                                                                                                                  'task '
                                                                                                                  'table',
                                                                                             'frame': 'world',
                                                                                             'units': 'metres',
                                                                                             'control_scope': 'one_agent_58_dof'},
                                                                                  'qualification': {'native_preflight': False,
                                                                                                    'agent_bench_interface': False,
                                                                                                    'robot_manipulation': False,
                                                                                                    'physics_accepted': False,
                                                                                                    'workspace_ik': True,
                                                                                                    'workspace_clearance': True,
                                                                                                    'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/mechanical_assembly_parts_sorting_dual_same_side.json',
                                                                                                                           'sha256': '81707fb9e835bcf11a31fc089b4b3437f54caf3ccc8c19a9d53b5faf58413e01'},
                                                                                                    'qualification_scope': 'Discrete '
                                                                                                                           'fingertip '
                                                                                                                           'approach '
                                                                                                                           'IK '
                                                                                                                           'and '
                                                                                                                           'original '
                                                                                                                           'shape '
                                                                                                                           'AABB '
                                                                                                                           'clearance; '
                                                                                                                           'continuous '
                                                                                                                           'motion '
                                                                                                                           'and '
                                                                                                                           'manipulation '
                                                                                                                           'remain '
                                                                                                                           'unqualified.',
                                                                                                    'nonadjacent_arm_self_collision': True,
                                                                                                    'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                                    'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                               'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/mechanical_assembly/parts_sorting/dual_opposed/v1',
                                                                                                   'manifest_sha256': '843aca6fb2becaabc3fee0e97a86b4843c6b84d87f2bf2b78035c790d6786b78'},
                                                                                'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                       'assembly_version': 'isaacsim_6_1_v2',
                                                                                                       'articulations': 2,
                                                                                                       'active_sides': ['left',
                                                                                                                        'right']},
                                                                                'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                                                           'placements': {'left': {'base_position_m': [0.42,
                                                                                                                                       0.76,
                                                                                                                                       0.78],
                                                                                                                   'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                            0.0,
                                                                                                                                            0.0,
                                                                                                                                            -0.7071067811865475],
                                                                                                                   'base_yaw_rad': -1.5707963267948966,
                                                                                                                   'hand_chirality': 'left'},
                                                                                                          'right': {'base_position_m': [0.42,
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
                                                                                'qualification': {'native_preflight': False,
                                                                                                  'agent_bench_interface': False,
                                                                                                  'robot_manipulation': False,
                                                                                                  'physics_accepted': False,
                                                                                                  'workspace_ik': True,
                                                                                                  'workspace_clearance': True,
                                                                                                  'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/mechanical_assembly_parts_sorting_dual_opposed.json',
                                                                                                                         'sha256': '889a41bc8fe8f21c51ecb33984a78b562ae5c215a9ea30d97cedaa391ae0afc2'},
                                                                                                  'qualification_scope': 'Discrete '
                                                                                                                         'fingertip '
                                                                                                                         'approach '
                                                                                                                         'IK '
                                                                                                                         'and '
                                                                                                                         'original '
                                                                                                                         'shape '
                                                                                                                         'AABB '
                                                                                                                         'clearance; '
                                                                                                                         'continuous '
                                                                                                                         'motion '
                                                                                                                         'and '
                                                                                                                         'manipulation '
                                                                                                                         'remain '
                                                                                                                         'unqualified.',
                                                                                                  'nonadjacent_arm_self_collision': True,
                                                                                                  'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                                  'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                                            'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                                              'articulations': 1}},
 ('mechanical_assembly', 'peg_insertion'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/peg_insertion/desktop_v1',
                                                             'manifest_sha256': '589cbd190d1e15bad940d1553cdae025f79d57148cfe587ad89e7b686a1f3a3f'},
                                            'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                                  'placements': {'right': {'base_position_m': [-0.23,
                                                                                                               0.0,
                                                                                                               0.78],
                                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                                    0.0,
                                                                                                                    0.0,
                                                                                                                    0.0],
                                                                                           'base_yaw_rad': 0.0,
                                                                                           'hand_chirality': 'right'}},
                                                                  'reference_surface': 'original fixed task '
                                                                                       'installation',
                                                                  'frame': 'world',
                                                                  'units': 'metres',
                                                                  'control_scope': 'one_agent_29_dof'},
                                            'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/mechanical_assembly/peg_insertion/dual_same_side/v1',
                                                                                                     'manifest_sha256': 'd07198c2477cd1f4eb7eff3362a9b71c2b0123e42409e9566bf94f16c070195f'},
                                                                                  'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                  'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                  'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                                                         'articulations': 2,
                                                                                                         'active_sides': ['left',
                                                                                                                          'right']},
                                                                                  'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                                                             'placements': {'left': {'base_position_m': [-0.1,
                                                                                                                                         0.3,
                                                                                                                                         0.78],
                                                                                                                     'base_quaternion_wxyz': [1.0,
                                                                                                                                              0.0,
                                                                                                                                              0.0,
                                                                                                                                              0.0],
                                                                                                                     'base_yaw_rad': 0.0,
                                                                                                                     'hand_chirality': 'left'},
                                                                                                            'right': {'base_position_m': [-0.1,
                                                                                                                                          -0.3,
                                                                                                                                          0.78],
                                                                                                                      'base_quaternion_wxyz': [1.0,
                                                                                                                                               0.0,
                                                                                                                                               0.0,
                                                                                                                                               0.0],
                                                                                                                      'base_yaw_rad': 0.0,
                                                                                                                      'hand_chirality': 'right'}},
                                                                                             'reference_surface': 'fixed '
                                                                                                                  'room '
                                                                                                                  'task '
                                                                                                                  'table',
                                                                                             'frame': 'world',
                                                                                             'units': 'metres',
                                                                                             'control_scope': 'one_agent_58_dof'},
                                                                                  'qualification': {'native_preflight': False,
                                                                                                    'agent_bench_interface': False,
                                                                                                    'robot_manipulation': False,
                                                                                                    'physics_accepted': False,
                                                                                                    'workspace_ik': True,
                                                                                                    'workspace_clearance': True,
                                                                                                    'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/mechanical_assembly_peg_insertion_dual_same_side.json',
                                                                                                                           'sha256': 'd526630cd6eff78db0860eced6301b00ee321cf081ebb1ac97c93e2de267dc5c'},
                                                                                                    'qualification_scope': 'Discrete '
                                                                                                                           'fingertip '
                                                                                                                           'approach '
                                                                                                                           'IK '
                                                                                                                           'and '
                                                                                                                           'original '
                                                                                                                           'shape '
                                                                                                                           'AABB '
                                                                                                                           'clearance; '
                                                                                                                           'continuous '
                                                                                                                           'motion '
                                                                                                                           'and '
                                                                                                                           'manipulation '
                                                                                                                           'remain '
                                                                                                                           'unqualified.',
                                                                                                    'nonadjacent_arm_self_collision': True,
                                                                                                    'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                                    'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                               'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/mechanical_assembly/peg_insertion/dual_opposed/v1',
                                                                                                   'manifest_sha256': '067b365c7a0ed066cde18477ef077806452766b57b84d7f9208c436756bd2bb6'},
                                                                                'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                       'assembly_version': 'isaacsim_6_1_v2',
                                                                                                       'articulations': 2,
                                                                                                       'active_sides': ['left',
                                                                                                                        'right']},
                                                                                'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                                                           'placements': {'left': {'base_position_m': [0.42,
                                                                                                                                       0.76,
                                                                                                                                       0.78],
                                                                                                                   'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                            0.0,
                                                                                                                                            0.0,
                                                                                                                                            -0.7071067811865475],
                                                                                                                   'base_yaw_rad': -1.5707963267948966,
                                                                                                                   'hand_chirality': 'left'},
                                                                                                          'right': {'base_position_m': [0.42,
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
                                                                                'qualification': {'native_preflight': False,
                                                                                                  'agent_bench_interface': False,
                                                                                                  'robot_manipulation': False,
                                                                                                  'physics_accepted': False,
                                                                                                  'workspace_ik': True,
                                                                                                  'workspace_clearance': True,
                                                                                                  'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/mechanical_assembly_peg_insertion_dual_opposed.json',
                                                                                                                         'sha256': '0f5bbd2fd4d3ec799dcae6df737ffdfc8fa06d11f5c2668a3f9ac1e1a7f60709'},
                                                                                                  'qualification_scope': 'Discrete '
                                                                                                                         'fingertip '
                                                                                                                         'approach '
                                                                                                                         'IK '
                                                                                                                         'and '
                                                                                                                         'original '
                                                                                                                         'shape '
                                                                                                                         'AABB '
                                                                                                                         'clearance; '
                                                                                                                         'continuous '
                                                                                                                         'motion '
                                                                                                                         'and '
                                                                                                                         'manipulation '
                                                                                                                         'remain '
                                                                                                                         'unqualified.',
                                                                                                  'nonadjacent_arm_self_collision': True,
                                                                                                  'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                                  'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                                            'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                                              'articulations': 1}},
 ('mechanical_assembly', 'shaft_spacer_assembly'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/shaft_spacer_assembly/desktop_v1',
                                                                     'manifest_sha256': '4bda201af19b2701394c1e503455ab62de54584ab75d7829d1999cdc5a8c2532'},
                                                    'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                                          'placements': {'right': {'base_position_m': [-0.23,
                                                                                                                       0.0,
                                                                                                                       0.78],
                                                                                                   'base_quaternion_wxyz': [1.0,
                                                                                                                            0.0,
                                                                                                                            0.0,
                                                                                                                            0.0],
                                                                                                   'base_yaw_rad': 0.0,
                                                                                                   'hand_chirality': 'right'}},
                                                                          'reference_surface': 'original '
                                                                                               'fixed task '
                                                                                               'installation',
                                                                          'frame': 'world',
                                                                          'units': 'metres',
                                                                          'control_scope': 'one_agent_29_dof'},
                                                    'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/mechanical_assembly/shaft_spacer_assembly/dual_same_side/v1',
                                                                                                             'manifest_sha256': 'c1f5aa6e6940c2ddeb67541759643f24216b1ffd0e3ec603048f9c465e0ecc43'},
                                                                                          'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                          'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                          'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                                 'assembly_version': 'isaacsim_6_1_v2',
                                                                                                                 'articulations': 2,
                                                                                                                 'active_sides': ['left',
                                                                                                                                  'right']},
                                                                                          'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                                                                     'placements': {'left': {'base_position_m': [-0.1,
                                                                                                                                                 0.3,
                                                                                                                                                 0.78],
                                                                                                                             'base_quaternion_wxyz': [1.0,
                                                                                                                                                      0.0,
                                                                                                                                                      0.0,
                                                                                                                                                      0.0],
                                                                                                                             'base_yaw_rad': 0.0,
                                                                                                                             'hand_chirality': 'left'},
                                                                                                                    'right': {'base_position_m': [-0.1,
                                                                                                                                                  -0.3,
                                                                                                                                                  0.78],
                                                                                                                              'base_quaternion_wxyz': [1.0,
                                                                                                                                                       0.0,
                                                                                                                                                       0.0,
                                                                                                                                                       0.0],
                                                                                                                              'base_yaw_rad': 0.0,
                                                                                                                              'hand_chirality': 'right'}},
                                                                                                     'reference_surface': 'fixed '
                                                                                                                          'room '
                                                                                                                          'task '
                                                                                                                          'table',
                                                                                                     'frame': 'world',
                                                                                                     'units': 'metres',
                                                                                                     'control_scope': 'one_agent_58_dof'},
                                                                                          'qualification': {'native_preflight': False,
                                                                                                            'agent_bench_interface': False,
                                                                                                            'robot_manipulation': False,
                                                                                                            'physics_accepted': False,
                                                                                                            'workspace_ik': True,
                                                                                                            'workspace_clearance': True,
                                                                                                            'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/mechanical_assembly_shaft_spacer_assembly_dual_same_side.json',
                                                                                                                                   'sha256': '2d822e2b415547f2b4e26800f8601e6ec0087f3acc69e5c23f3ac94d6cd73e3b'},
                                                                                                            'qualification_scope': 'Discrete '
                                                                                                                                   'fingertip '
                                                                                                                                   'approach '
                                                                                                                                   'IK '
                                                                                                                                   'and '
                                                                                                                                   'original '
                                                                                                                                   'shape '
                                                                                                                                   'AABB '
                                                                                                                                   'clearance; '
                                                                                                                                   'continuous '
                                                                                                                                   'motion '
                                                                                                                                   'and '
                                                                                                                                   'manipulation '
                                                                                                                                   'remain '
                                                                                                                                   'unqualified.',
                                                                                                            'nonadjacent_arm_self_collision': True,
                                                                                                            'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                                            'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                                       'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/mechanical_assembly/shaft_spacer_assembly/dual_opposed/v1',
                                                                                                           'manifest_sha256': '958f0dae478d605ae9c77d570d4db006141b409853d0f51e1be25040959e036a'},
                                                                                        'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                        'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                        'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                               'assembly_version': 'isaacsim_6_1_v2',
                                                                                                               'articulations': 2,
                                                                                                               'active_sides': ['left',
                                                                                                                                'right']},
                                                                                        'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                                                                   'placements': {'left': {'base_position_m': [0.42,
                                                                                                                                               0.76,
                                                                                                                                               0.78],
                                                                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                    0.0,
                                                                                                                                                    0.0,
                                                                                                                                                    -0.7071067811865475],
                                                                                                                           'base_yaw_rad': -1.5707963267948966,
                                                                                                                           'hand_chirality': 'left'},
                                                                                                                  'right': {'base_position_m': [0.42,
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
                                                                                        'qualification': {'native_preflight': False,
                                                                                                          'agent_bench_interface': False,
                                                                                                          'robot_manipulation': False,
                                                                                                          'physics_accepted': False,
                                                                                                          'workspace_ik': True,
                                                                                                          'workspace_clearance': True,
                                                                                                          'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/mechanical_assembly_shaft_spacer_assembly_dual_opposed.json',
                                                                                                                                 'sha256': '9de8e8e3560af2ed272676f719e88e6a021d9603f2b339c55e72a9ac3e8054a0'},
                                                                                                          'qualification_scope': 'Discrete '
                                                                                                                                 'fingertip '
                                                                                                                                 'approach '
                                                                                                                                 'IK '
                                                                                                                                 'and '
                                                                                                                                 'original '
                                                                                                                                 'shape '
                                                                                                                                 'AABB '
                                                                                                                                 'clearance; '
                                                                                                                                 'continuous '
                                                                                                                                 'motion '
                                                                                                                                 'and '
                                                                                                                                 'manipulation '
                                                                                                                                 'remain '
                                                                                                                                 'unqualified.',
                                                                                                          'nonadjacent_arm_self_collision': True,
                                                                                                          'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                                          'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                                                    'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                                      'assembly_version': 'isaacsim_6_1_v2',
                                                                                      'articulations': 1}},
 ('mechanical_assembly', 'fastener_assembly'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/fastener_assembly/runtime_inputs_v1',
                  'manifest_sha256': '498d5946278144e17df8af60ba93d65771fab176335f86af68e89d5b52888f42'},
 'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                       'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/fastener_assembly/dual_same_side_runtime_inputs_v1',
                                                          'manifest_sha256': 'a672ce594525bc15f2577276d74ff858d2656e93aa7b7b0944effd0c02a38a55'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [-0.1,
                                                                                              0.3,
                                                                                              0.78],
                                                                          'base_quaternion_wxyz': [1.0,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   0.0],
                                                                          'base_yaw_rad': 0.0,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [-0.1,
                                                                                               -0.3,
                                                                                               0.78],
                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    0.0],
                                                                           'base_yaw_rad': 0.0,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'fixed room task table',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False,
                                                         'qualification_scope': 'Immutable physical input '
                                                                                'rebinding; development '
                                                                                'outcomes are not task '
                                                                                'selection or admission '
                                                                                'inputs.'}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/fastener_assembly/dual_opposed_runtime_inputs_v1',
                                                        'manifest_sha256': '9a8c88511a898d27c9d6245f1ba056f8442abc58bec52e85298b68ec7277521e'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.42, 0.76, 0.78],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.42,
                                                                                             -0.76,
                                                                                             0.78],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'fixed room task table',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False,
                                                       'qualification_scope': 'Immutable physical input '
                                                                              'rebinding; development '
                                                                              'outcomes are not task '
                                                                              'selection or admission '
                                                                              'inputs.'}}},
 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 1}},
 ('mechanical_assembly', 'gear_train_assembly'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/gear_train_assembly/desktop_v1',
                                                                   'manifest_sha256': '4edad1f71dd67688679613802c025857bf3075764f0f86c315bf7ec01cd3d9e5'},
                                                  'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                                        'placements': {'right': {'base_position_m': [-0.23,
                                                                                                                     0.0,
                                                                                                                     0.78],
                                                                                                 'base_quaternion_wxyz': [1.0,
                                                                                                                          0.0,
                                                                                                                          0.0,
                                                                                                                          0.0],
                                                                                                 'base_yaw_rad': 0.0,
                                                                                                 'hand_chirality': 'right'}},
                                                                        'reference_surface': 'original fixed '
                                                                                             'task '
                                                                                             'installation',
                                                                        'frame': 'world',
                                                                        'units': 'metres',
                                                                        'control_scope': 'one_agent_29_dof'},
                                                  'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/mechanical_assembly/gear_train_assembly/dual_same_side/v1',
                                                                                                           'manifest_sha256': 'be71c9cd7c2aaf09e5964fbcbf4e082421f62d9dfd3f60eabb8127b83f17a10a'},
                                                                                        'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                        'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                        'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                               'assembly_version': 'isaacsim_6_1_v2',
                                                                                                               'articulations': 2,
                                                                                                               'active_sides': ['left',
                                                                                                                                'right']},
                                                                                        'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                                                                   'placements': {'left': {'base_position_m': [-0.1,
                                                                                                                                               0.3,
                                                                                                                                               0.78],
                                                                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                                                                    0.0,
                                                                                                                                                    0.0,
                                                                                                                                                    0.0],
                                                                                                                           'base_yaw_rad': 0.0,
                                                                                                                           'hand_chirality': 'left'},
                                                                                                                  'right': {'base_position_m': [-0.1,
                                                                                                                                                -0.3,
                                                                                                                                                0.78],
                                                                                                                            'base_quaternion_wxyz': [1.0,
                                                                                                                                                     0.0,
                                                                                                                                                     0.0,
                                                                                                                                                     0.0],
                                                                                                                            'base_yaw_rad': 0.0,
                                                                                                                            'hand_chirality': 'right'}},
                                                                                                   'reference_surface': 'fixed '
                                                                                                                        'room '
                                                                                                                        'task '
                                                                                                                        'table',
                                                                                                   'frame': 'world',
                                                                                                   'units': 'metres',
                                                                                                   'control_scope': 'one_agent_58_dof'},
                                                                                        'qualification': {'native_preflight': True,
                                                                                                          'agent_bench_interface': True,
                                                                                                          'robot_manipulation': False,
                                                                                                          'physics_accepted': False,
                                                                                                          'workspace_ik': True,
                                                                                                          'workspace_clearance': True,
                                                                                                          'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/mechanical_assembly_gear_train_assembly_dual_same_side.json',
                                                                                                                                 'sha256': '8b1d877296c8c8a0240ce082a1419b712fdf45c59bc1880ba0a8ce1297062923'},
                                                                                                          'qualification_scope': 'Discrete '
                                                                                                                                 'fingertip '
                                                                                                                                 'approach '
                                                                                                                                 'IK '
                                                                                                                                 'and '
                                                                                                                                 'original '
                                                                                                                                 'shape '
                                                                                                                                 'AABB '
                                                                                                                                 'clearance; '
                                                                                                                                 'continuous '
                                                                                                                                 'motion '
                                                                                                                                 'and '
                                                                                                                                 'manipulation '
                                                                                                                                 'remain '
                                                                                                                                 'unqualified.',
                                                                                                          'nonadjacent_arm_self_collision': True,
                                                                                                          'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                                          'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'},
                                                                                                          'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/http_mechanical_same01/qualification.json',
                                                                                                                                        'sha256': 'fed114a1a8d255177effcd072d4b582974f5cfabdcef1f5f665bdbbbd8e3b7cc'},
                                                                                                          'native_scope': 'This '
                                                                                                                          'exact '
                                                                                                                          'release '
                                                                                                                          'only: '
                                                                                                                          'configuration '
                                                                                                                          'selection, '
                                                                                                                          'one '
                                                                                                                          'Agent/58 '
                                                                                                                          'DOF, '
                                                                                                                          'five '
                                                                                                                          'RGB-D '
                                                                                                                          'streams, '
                                                                                                                          'read-only '
                                                                                                                          'IK, '
                                                                                                                          'simultaneous '
                                                                                                                          'wrist '
                                                                                                                          'motion/return/hold. '
                                                                                                                          'No '
                                                                                                                          'manipulation '
                                                                                                                          'qualification.'}},
                                                                     'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/mechanical_assembly/gear_train_assembly/dual_opposed/v1',
                                                                                                         'manifest_sha256': '0b2759ca7caf7bd9b1b2f5c601f5064319239cebc63f22cd6c11d257449ba75b'},
                                                                                      'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                      'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                      'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                             'assembly_version': 'isaacsim_6_1_v2',
                                                                                                             'articulations': 2,
                                                                                                             'active_sides': ['left',
                                                                                                                              'right']},
                                                                                      'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                                                                 'placements': {'left': {'base_position_m': [0.42,
                                                                                                                                             0.76,
                                                                                                                                             0.78],
                                                                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                  0.0,
                                                                                                                                                  0.0,
                                                                                                                                                  -0.7071067811865475],
                                                                                                                         'base_yaw_rad': -1.5707963267948966,
                                                                                                                         'hand_chirality': 'left'},
                                                                                                                'right': {'base_position_m': [0.42,
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
                                                                                      'qualification': {'native_preflight': False,
                                                                                                        'agent_bench_interface': False,
                                                                                                        'robot_manipulation': False,
                                                                                                        'physics_accepted': False,
                                                                                                        'workspace_ik': True,
                                                                                                        'workspace_clearance': True,
                                                                                                        'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/mechanical_assembly_gear_train_assembly_dual_opposed.json',
                                                                                                                               'sha256': 'f36b24c0bc9431d4ea00b4af7df098312a0a857cf7b0d09f8b8778a979755154'},
                                                                                                        'qualification_scope': 'Discrete '
                                                                                                                               'fingertip '
                                                                                                                               'approach '
                                                                                                                               'IK '
                                                                                                                               'and '
                                                                                                                               'original '
                                                                                                                               'shape '
                                                                                                                               'AABB '
                                                                                                                               'clearance; '
                                                                                                                               'continuous '
                                                                                                                               'motion '
                                                                                                                               'and '
                                                                                                                               'manipulation '
                                                                                                                               'remain '
                                                                                                                               'unqualified.',
                                                                                                        'nonadjacent_arm_self_collision': True,
                                                                                                        'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                                        'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                                                  'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                                    'assembly_version': 'isaacsim_6_1_v2',
                                                                                    'articulations': 1}},
 ('mechanical_assembly', 'nist_connector_insertion'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch01/nist_connector_insertion/desktop_v1',
                                                                        'manifest_sha256': '76886985c22a43daeac20cbebcab525efc3c340ab1668e8f9867270c3840ba6f'},
                                                       'single_arm_layout': {'robot_layout_id': 'mechanical_assembly_nist_connector_insertion_original_single_arm',
                                                                             'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                                                                      'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                                                                      'base_yaw_rad': 0.0,
                                                                                                      'hand_chirality': 'right'}},
                                                                             'reference_surface': 'original fixed task installation',
                                                                             'frame': 'world',
                                                                             'units': 'metres',
                                                                             'control_scope': 'one_agent_29_dof'},
                                                       'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/dual_batch01/mechanical_assembly/nist_connector_insertion/dual_same_side/v1',
                                                                                                                'manifest_sha256': '7ad125dcde37691ea8d28d6f85f8964657c735e3794813aff26dfb36ffcde850'},
                                                                                             'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                             'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                             'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                                    'assembly_version': 'isaacsim_6_1_v2',
                                                                                                                    'articulations': 2,
                                                                                                                    'active_sides': ['left', 'right']},
                                                                                             'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                                                                        'placements': {'left': {'base_position_m': [-0.1,
                                                                                                                                                    0.3,
                                                                                                                                                    0.78],
                                                                                                                                'base_quaternion_wxyz': [1.0,
                                                                                                                                                         0.0,
                                                                                                                                                         0.0,
                                                                                                                                                         0.0],
                                                                                                                                'base_yaw_rad': 0.0,
                                                                                                                                'hand_chirality': 'left'},
                                                                                                                       'right': {'base_position_m': [-0.1,
                                                                                                                                                     -0.3,
                                                                                                                                                     0.78],
                                                                                                                                 'base_quaternion_wxyz': [1.0,
                                                                                                                                                          0.0,
                                                                                                                                                          0.0,
                                                                                                                                                          0.0],
                                                                                                                                 'base_yaw_rad': 0.0,
                                                                                                                                 'hand_chirality': 'right'}},
                                                                                                        'reference_surface': 'fixed room task table',
                                                                                                        'frame': 'world',
                                                                                                        'units': 'metres',
                                                                                                        'control_scope': 'one_agent_58_dof'},
                                                                                             'qualification': {'native_preflight': False,
                                                                                                               'agent_bench_interface': False,
                                                                                                               'robot_manipulation': False,
                                                                                                               'physics_accepted': False,
                                                                                                               'workspace_discrete': True,
                                                                                                               'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace01/nist_connector_insertion_dual_same_side.json',
                                                                                                                                           'sha256': '29979109011087ec8982f386ebf837576016b51f3a800cae9197a68508ff9ca8',
                                                                                                                                           'passed': True,
                                                                                                                                           'scope': 'Discrete '
                                                                                                                                                    'actual '
                                                                                                                                                    'fingertip '
                                                                                                                                                    'approach and '
                                                                                                                                                    'original USD '
                                                                                                                                                    'collision '
                                                                                                                                                    'bounds; no '
                                                                                                                                                    'continuous '
                                                                                                                                                    'path or '
                                                                                                                                                    'manipulation '
                                                                                                                                                    'qualification.'},
                                                                                                               'object_interactions': True,
                                                                                                               'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task6_goal01/qualification.json',
                                                                                                                                           'sha256': '0678af29ff6a10e29b4937de3e8826fe76650c541d146c16522bab9cbabff0e6',
                                                                                                                                           'passed': True,
                                                                                                                                           'scope': 'Asset-specific '
                                                                                                                                                    'passive goal '
                                                                                                                                                    'fit/support '
                                                                                                                                                    'for '
                                                                                                                                                    'Task6/8/9; '
                                                                                                                                                    '10mm '
                                                                                                                                                    'already-guided '
                                                                                                                                                    'idler '
                                                                                                                                                    'insertion for '
                                                                                                                                                    'Task7. No '
                                                                                                                                                    'robot '
                                                                                                                                                    'operation '
                                                                                                                                                    'credit.',
                                                                                                                                           'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_connector_insertion/usability_v1/manifest.json',
                                                                                                                                                              'sha256': '066a3c93be4bae05c3765b8054e60688ed3a22226cfd1a8e5f2f989e9283e3ee'},
                                                                                                                                           'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/qualification02.json',
                                                                                                                                                                   'sha256': 'a9ce0d252506a14e28cba028091ad977bab2a77a46b7bb672296fd873f49e771'},
                                                                                                                                           'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/root-independent-identity-review02.json',
                                                                                                                                                                     'sha256': '0f371c44ac87663055d2332be0c276fef810a9af83fff104c4f1b343c286d2bb'}}],
                                                                                                               'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                                                                                                                                             'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'}}},
                                                                          'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/dual_batch01/mechanical_assembly/nist_connector_insertion/dual_opposed/v1',
                                                                                                              'manifest_sha256': '96db1bcc42a4af49019a0db72c7d373523dadde797a7f5698230b97c33a30f3c'},
                                                                                           'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                           'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                           'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                                  'assembly_version': 'isaacsim_6_1_v2',
                                                                                                                  'articulations': 2,
                                                                                                                  'active_sides': ['left', 'right']},
                                                                                           'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                                                                      'placements': {'left': {'base_position_m': [0.42, 0.76, 0.78],
                                                                                                                              'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                       0.0,
                                                                                                                                                       0.0,
                                                                                                                                                       -0.7071067811865475],
                                                                                                                              'base_yaw_rad': -1.5707963267948966,
                                                                                                                              'hand_chirality': 'left'},
                                                                                                                     'right': {'base_position_m': [0.42,
                                                                                                                                                   -0.76,
                                                                                                                                                   0.78],
                                                                                                                               'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                        0.0,
                                                                                                                                                        0.0,
                                                                                                                                                        0.7071067811865475],
                                                                                                                               'base_yaw_rad': 1.5707963267948966,
                                                                                                                               'hand_chirality': 'right'}},
                                                                                                      'reference_surface': 'fixed room task table',
                                                                                                      'frame': 'world',
                                                                                                      'units': 'metres',
                                                                                                      'control_scope': 'one_agent_58_dof'},
                                                                                           'qualification': {'native_preflight': False,
                                                                                                             'agent_bench_interface': False,
                                                                                                             'robot_manipulation': False,
                                                                                                             'physics_accepted': False,
                                                                                                             'workspace_discrete': True,
                                                                                                             'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace01/nist_connector_insertion_dual_opposed.json',
                                                                                                                                         'sha256': '445d1ca84dfaab7561e7765a3a21df7c515103d5c5005d9da288f16373dee21d',
                                                                                                                                         'passed': True,
                                                                                                                                         'scope': 'Discrete actual '
                                                                                                                                                  'fingertip '
                                                                                                                                                  'approach and '
                                                                                                                                                  'original USD '
                                                                                                                                                  'collision '
                                                                                                                                                  'bounds; no '
                                                                                                                                                  'continuous path '
                                                                                                                                                  'or manipulation '
                                                                                                                                                  'qualification.'},
                                                                                                             'object_interactions': True,
                                                                                                             'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task6_goal01/qualification.json',
                                                                                                                                         'sha256': '0678af29ff6a10e29b4937de3e8826fe76650c541d146c16522bab9cbabff0e6',
                                                                                                                                         'passed': True,
                                                                                                                                         'scope': 'Asset-specific '
                                                                                                                                                  'passive goal '
                                                                                                                                                  'fit/support for '
                                                                                                                                                  'Task6/8/9; 10mm '
                                                                                                                                                  'already-guided '
                                                                                                                                                  'idler insertion '
                                                                                                                                                  'for Task7. No '
                                                                                                                                                  'robot operation '
                                                                                                                                                  'credit.',
                                                                                                                                         'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_connector_insertion/usability_v1/manifest.json',
                                                                                                                                                            'sha256': '066a3c93be4bae05c3765b8054e60688ed3a22226cfd1a8e5f2f989e9283e3ee'},
                                                                                                                                         'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/qualification02.json',
                                                                                                                                                                 'sha256': 'a9ce0d252506a14e28cba028091ad977bab2a77a46b7bb672296fd873f49e771'},
                                                                                                                                         'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/root-independent-identity-review02.json',
                                                                                                                                                                   'sha256': '0f371c44ac87663055d2332be0c276fef810a9af83fff104c4f1b343c286d2bb'}}],
                                                                                                             'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                                                                                                                                           'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'}}}},
                                                       'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                                         'articulations': 1}},
 ('mechanical_assembly', 'nist_belt_drive_assembly'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch02/nist_belt_drive_assembly/desktop_v1',
                                                                        'manifest_sha256': '4e1b0371aee0483a9a11a11f3bdb78d040aa45e7293233a5a52d6cc465b53f62'},
                                                       'single_arm_layout': {'robot_layout_id': 'mechanical_assembly_nist_belt_drive_assembly_original_single_arm',
                                                                             'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                                                                      'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                                                                      'base_yaw_rad': 0.0,
                                                                                                      'hand_chirality': 'right'}},
                                                                             'reference_surface': 'original fixed task installation',
                                                                             'frame': 'world',
                                                                             'units': 'metres',
                                                                             'control_scope': 'one_agent_29_dof'},
                                                       'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/dual_batch02/mechanical_assembly/nist_belt_drive_assembly/dual_same_side/v1',
                                                                                                                'manifest_sha256': '554478709a3e3503767c5c5b8052c853794298bd0f24b7eeac5f45d12a987954'},
                                                                                             'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                             'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                             'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                                    'assembly_version': 'isaacsim_6_1_v2',
                                                                                                                    'articulations': 2,
                                                                                                                    'active_sides': ['left', 'right']},
                                                                                             'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                                                                        'placements': {'left': {'base_position_m': [-0.1,
                                                                                                                                                    0.3,
                                                                                                                                                    0.78],
                                                                                                                                'base_quaternion_wxyz': [1.0,
                                                                                                                                                         0.0,
                                                                                                                                                         0.0,
                                                                                                                                                         0.0],
                                                                                                                                'base_yaw_rad': 0.0,
                                                                                                                                'hand_chirality': 'left'},
                                                                                                                       'right': {'base_position_m': [-0.1,
                                                                                                                                                     -0.3,
                                                                                                                                                     0.78],
                                                                                                                                 'base_quaternion_wxyz': [1.0,
                                                                                                                                                          0.0,
                                                                                                                                                          0.0,
                                                                                                                                                          0.0],
                                                                                                                                 'base_yaw_rad': 0.0,
                                                                                                                                 'hand_chirality': 'right'}},
                                                                                                        'reference_surface': 'fixed room task table',
                                                                                                        'frame': 'world',
                                                                                                        'units': 'metres',
                                                                                                        'control_scope': 'one_agent_58_dof'},
                                                                                             'qualification': {'native_preflight': False,
                                                                                                               'agent_bench_interface': False,
                                                                                                               'robot_manipulation': False,
                                                                                                               'physics_accepted': False,
                                                                                                               'workspace_discrete': True,
                                                                                                               'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace02/nist_belt_drive_assembly_dual_same_side.json',
                                                                                                                                           'sha256': 'aa6d98813694edbcee0925e7693bf6fad919cf181bd075bf3e2e3144f3fd6baa',
                                                                                                                                           'passed': True,
                                                                                                                                           'scope': 'Discrete '
                                                                                                                                                    'actual '
                                                                                                                                                    'fingertip '
                                                                                                                                                    'approach and '
                                                                                                                                                    'original USD '
                                                                                                                                                    'collision '
                                                                                                                                                    'bounds; no '
                                                                                                                                                    'continuous '
                                                                                                                                                    'path or '
                                                                                                                                                    'manipulation '
                                                                                                                                                    'qualification.'},
                                                                                                               'object_interactions': True,
                                                                                                               'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/guided01/qualification.json',
                                                                                                                                           'sha256': 'd6a76fd3267185c88dc7da4af7fc5d2b8ee3d3fb750dd79458a41c49926db69f',
                                                                                                                                           'passed': True,
                                                                                                                                           'scope': 'Asset-specific '
                                                                                                                                                    'passive goal '
                                                                                                                                                    'fit/support '
                                                                                                                                                    'for '
                                                                                                                                                    'Task6/8/9; '
                                                                                                                                                    '10mm '
                                                                                                                                                    'already-guided '
                                                                                                                                                    'idler '
                                                                                                                                                    'insertion for '
                                                                                                                                                    'Task7. No '
                                                                                                                                                    'robot '
                                                                                                                                                    'operation '
                                                                                                                                                    'credit.',
                                                                                                                                           'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_belt_drive_assembly/usability_v3/manifest.json',
                                                                                                                                                              'sha256': '3e30ffadf4ba290c936336774c6bfd4409ef070570119b2fd55c9bd8a2b3a3dd'},
                                                                                                                                           'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/qualification.json',
                                                                                                                                                                   'sha256': 'c92c088b20d45cc4e82fa3e9c7c6866aa602e41e9d81f4cace031eea3ee028a5'},
                                                                                                                                           'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/object_identity_bridge01.json',
                                                                                                                                                                     'sha256': 'f1d61163062a8b954e9d5e15fa32cef63ca54539b66cbf36395cce07791ef992'}}],
                                                                                                               'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                                                                                                                                             'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'}}},
                                                                          'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/dual_batch02/mechanical_assembly/nist_belt_drive_assembly/dual_opposed/v1',
                                                                                                              'manifest_sha256': '5ae9ca6b336f6358d68f05175316a8f6354d96d7cdcc6e84960f584312222aa2'},
                                                                                           'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                           'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                           'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                                  'assembly_version': 'isaacsim_6_1_v2',
                                                                                                                  'articulations': 2,
                                                                                                                  'active_sides': ['left', 'right']},
                                                                                           'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                                                                      'placements': {'left': {'base_position_m': [0.42, 0.76, 0.78],
                                                                                                                              'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                       0.0,
                                                                                                                                                       0.0,
                                                                                                                                                       -0.7071067811865475],
                                                                                                                              'base_yaw_rad': -1.5707963267948966,
                                                                                                                              'hand_chirality': 'left'},
                                                                                                                     'right': {'base_position_m': [0.42,
                                                                                                                                                   -0.76,
                                                                                                                                                   0.78],
                                                                                                                               'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                        0.0,
                                                                                                                                                        0.0,
                                                                                                                                                        0.7071067811865475],
                                                                                                                               'base_yaw_rad': 1.5707963267948966,
                                                                                                                               'hand_chirality': 'right'}},
                                                                                                      'reference_surface': 'fixed room task table',
                                                                                                      'frame': 'world',
                                                                                                      'units': 'metres',
                                                                                                      'control_scope': 'one_agent_58_dof'},
                                                                                           'qualification': {'native_preflight': True,
                                                                                                             'agent_bench_interface': True,
                                                                                                             'robot_manipulation': False,
                                                                                                             'physics_accepted': False,
                                                                                                             'workspace_discrete': True,
                                                                                                             'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace02/nist_belt_drive_assembly_dual_opposed.json',
                                                                                                                                         'sha256': 'e9c97d20f6880b62a4d0b7e9c9293f3c28269a63ca8b97d42e9002567a5604db',
                                                                                                                                         'passed': True,
                                                                                                                                         'scope': 'Discrete actual '
                                                                                                                                                  'fingertip '
                                                                                                                                                  'approach and '
                                                                                                                                                  'original USD '
                                                                                                                                                  'collision '
                                                                                                                                                  'bounds; no '
                                                                                                                                                  'continuous path '
                                                                                                                                                  'or manipulation '
                                                                                                                                                  'qualification.'},
                                                                                                             'object_interactions': True,
                                                                                                             'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/guided01/qualification.json',
                                                                                                                                         'sha256': 'd6a76fd3267185c88dc7da4af7fc5d2b8ee3d3fb750dd79458a41c49926db69f',
                                                                                                                                         'passed': True,
                                                                                                                                         'scope': 'Asset-specific '
                                                                                                                                                  'passive goal '
                                                                                                                                                  'fit/support for '
                                                                                                                                                  'Task6/8/9; 10mm '
                                                                                                                                                  'already-guided '
                                                                                                                                                  'idler insertion '
                                                                                                                                                  'for Task7. No '
                                                                                                                                                  'robot operation '
                                                                                                                                                  'credit.',
                                                                                                                                         'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_belt_drive_assembly/usability_v3/manifest.json',
                                                                                                                                                            'sha256': '3e30ffadf4ba290c936336774c6bfd4409ef070570119b2fd55c9bd8a2b3a3dd'},
                                                                                                                                         'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/qualification.json',
                                                                                                                                                                 'sha256': 'c92c088b20d45cc4e82fa3e9c7c6866aa602e41e9d81f4cace031eea3ee028a5'},
                                                                                                                                         'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/task7/object_identity_bridge01.json',
                                                                                                                                                                   'sha256': 'f1d61163062a8b954e9d5e15fa32cef63ca54539b66cbf36395cce07791ef992'}}],
                                                                                                             'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                                                                                                                                           'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'},
                                                                                                             'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/http_task7_opposed01/qualification.json',
                                                                                                                                           'sha256': 'bf7c099a532cbf49e55af3de84231fc6e2d68a63f5d7d0a098cb4e70c76b8b2b',
                                                                                                                                           'passed': True,
                                                                                                                                           'configuration_id': 'dual_opposed',
                                                                                                                                           'release_manifest_sha256': '5ae9ca6b336f6358d68f05175316a8f6354d96d7cdcc6e84960f584312222aa2',
                                                                                                                                           'scope': 'Exact '
                                                                                                                                                    'frozen-source '
                                                                                                                                                    'public '
                                                                                                                                                    'HTTP/Agent '
                                                                                                                                                    'interface, '
                                                                                                                                                    'RGB-D, '
                                                                                                                                                    'read-only IK, '
                                                                                                                                                    'wrist '
                                                                                                                                                    'micro-motion/return/hold; '
                                                                                                                                                    'no '
                                                                                                                                                    'manipulation '
                                                                                                                                                    'qualification.'}}}},
                                                       'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                                         'articulations': 1}},
 ('mechanical_assembly', 'nist_cable_routing'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch02/nist_cable_routing/desktop_v1',
                                                                  'manifest_sha256': '4106fa53fbadf5d096c11522ada8dc85e6588588305b2141cd3590eddfc404c7'},
                                                 'single_arm_layout': {'robot_layout_id': 'mechanical_assembly_nist_cable_routing_original_single_arm',
                                                                       'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                                                                'base_yaw_rad': 0.0,
                                                                                                'hand_chirality': 'right'}},
                                                                       'reference_surface': 'original fixed task installation',
                                                                       'frame': 'world',
                                                                       'units': 'metres',
                                                                       'control_scope': 'one_agent_29_dof'},
                                                 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/dual_batch02/mechanical_assembly/nist_cable_routing/dual_same_side/v1',
                                                                                                          'manifest_sha256': 'a8a20cb58ab58fc4d790551b6a543ee5e46da61155124ac757a5e68b237fd481'},
                                                                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                                                                              'articulations': 2,
                                                                                                              'active_sides': ['left', 'right']},
                                                                                       'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                                                                  'placements': {'left': {'base_position_m': [-0.1,
                                                                                                                                              0.3,
                                                                                                                                              0.78],
                                                                                                                          'base_quaternion_wxyz': [1.0,
                                                                                                                                                   0.0,
                                                                                                                                                   0.0,
                                                                                                                                                   0.0],
                                                                                                                          'base_yaw_rad': 0.0,
                                                                                                                          'hand_chirality': 'left'},
                                                                                                                 'right': {'base_position_m': [-0.1,
                                                                                                                                               -0.3,
                                                                                                                                               0.78],
                                                                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                                                                    0.0,
                                                                                                                                                    0.0,
                                                                                                                                                    0.0],
                                                                                                                           'base_yaw_rad': 0.0,
                                                                                                                           'hand_chirality': 'right'}},
                                                                                                  'reference_surface': 'fixed room task table',
                                                                                                  'frame': 'world',
                                                                                                  'units': 'metres',
                                                                                                  'control_scope': 'one_agent_58_dof'},
                                                                                       'qualification': {'native_preflight': True,
                                                                                                         'agent_bench_interface': True,
                                                                                                         'robot_manipulation': False,
                                                                                                         'physics_accepted': False,
                                                                                                         'workspace_discrete': True,
                                                                                                         'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace02/nist_cable_routing_dual_same_side.json',
                                                                                                                                     'sha256': '05516a18c0511fca8fced9e27b9b5ea57b70152186245601b2db4cc0251071d9',
                                                                                                                                     'passed': True,
                                                                                                                                     'scope': 'Discrete '
                                                                                                                                              'actual '
                                                                                                                                              'fingertip '
                                                                                                                                              'approach and '
                                                                                                                                              'original USD '
                                                                                                                                              'collision '
                                                                                                                                              'bounds; no '
                                                                                                                                              'continuous '
                                                                                                                                              'path or '
                                                                                                                                              'manipulation '
                                                                                                                                              'qualification.'},
                                                                                                         'object_interactions': True,
                                                                                                         'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task8_goal02/qualification.json',
                                                                                                                                     'sha256': '356c576686245f102ff1e56e3e1e66d649463670e6d23b0fad2e436b18ebcc45',
                                                                                                                                     'passed': True,
                                                                                                                                     'scope': 'Asset-specific '
                                                                                                                                              'passive goal '
                                                                                                                                              'fit/support '
                                                                                                                                              'for '
                                                                                                                                              'Task6/8/9; '
                                                                                                                                              '10mm '
                                                                                                                                              'already-guided '
                                                                                                                                              'idler '
                                                                                                                                              'insertion for '
                                                                                                                                              'Task7. No '
                                                                                                                                              'robot '
                                                                                                                                              'operation '
                                                                                                                                              'credit.',
                                                                                                                                     'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_cable_routing/usability_v2/manifest.json',
                                                                                                                                                        'sha256': 'cd2f5f48258306ae8f0a3b4ff79c598c4c7e652ae371c9d6f3a637b527e29be5'},
                                                                                                                                     'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/qualification02.json',
                                                                                                                                                             'sha256': 'a9ce0d252506a14e28cba028091ad977bab2a77a46b7bb672296fd873f49e771'},
                                                                                                                                     'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/root-independent-identity-review02.json',
                                                                                                                                                               'sha256': '0f371c44ac87663055d2332be0c276fef810a9af83fff104c4f1b343c286d2bb'}}],
                                                                                                         'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                                                                                                                                       'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'},
                                                                                                         'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/http_task8_same01/qualification.json',
                                                                                                                                       'sha256': 'abd5c47c2cc9ec6736af617231cc2f85e994f043cfaf7f613aafe7a69e903af3',
                                                                                                                                       'passed': True,
                                                                                                                                       'configuration_id': 'dual_same_side',
                                                                                                                                       'release_manifest_sha256': 'a8a20cb58ab58fc4d790551b6a543ee5e46da61155124ac757a5e68b237fd481',
                                                                                                                                       'scope': 'Exact '
                                                                                                                                                'frozen-source '
                                                                                                                                                'public '
                                                                                                                                                'HTTP/Agent '
                                                                                                                                                'interface, '
                                                                                                                                                'RGB-D, '
                                                                                                                                                'read-only '
                                                                                                                                                'IK, wrist '
                                                                                                                                                'micro-motion/return/hold; '
                                                                                                                                                'no '
                                                                                                                                                'manipulation '
                                                                                                                                                'qualification.'}}},
                                                                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/dual_batch02/mechanical_assembly/nist_cable_routing/dual_opposed/v1',
                                                                                                        'manifest_sha256': '51657ab88cdf803cbddb9b19751cee26c18a1a682e65607a825b998071d6ed6e'},
                                                                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                                                                            'articulations': 2,
                                                                                                            'active_sides': ['left', 'right']},
                                                                                     'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                                                                'placements': {'left': {'base_position_m': [0.42, 0.76, 0.78],
                                                                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                 0.0,
                                                                                                                                                 0.0,
                                                                                                                                                 -0.7071067811865475],
                                                                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                                                                        'hand_chirality': 'left'},
                                                                                                               'right': {'base_position_m': [0.42,
                                                                                                                                             -0.76,
                                                                                                                                             0.78],
                                                                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                  0.0,
                                                                                                                                                  0.0,
                                                                                                                                                  0.7071067811865475],
                                                                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                                                                         'hand_chirality': 'right'}},
                                                                                                'reference_surface': 'fixed room task table',
                                                                                                'frame': 'world',
                                                                                                'units': 'metres',
                                                                                                'control_scope': 'one_agent_58_dof'},
                                                                                     'qualification': {'native_preflight': False,
                                                                                                       'agent_bench_interface': False,
                                                                                                       'robot_manipulation': False,
                                                                                                       'physics_accepted': False,
                                                                                                       'workspace_discrete': True,
                                                                                                       'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace02/nist_cable_routing_dual_opposed.json',
                                                                                                                                   'sha256': '672ecd07c9c28376fb0ecff93ce7844056c54d93cc385b07bfe3715e524b0937',
                                                                                                                                   'passed': True,
                                                                                                                                   'scope': 'Discrete actual '
                                                                                                                                            'fingertip '
                                                                                                                                            'approach and '
                                                                                                                                            'original USD '
                                                                                                                                            'collision '
                                                                                                                                            'bounds; no '
                                                                                                                                            'continuous path '
                                                                                                                                            'or manipulation '
                                                                                                                                            'qualification.'},
                                                                                                       'object_interactions': True,
                                                                                                       'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task8_goal02/qualification.json',
                                                                                                                                   'sha256': '356c576686245f102ff1e56e3e1e66d649463670e6d23b0fad2e436b18ebcc45',
                                                                                                                                   'passed': True,
                                                                                                                                   'scope': 'Asset-specific '
                                                                                                                                            'passive goal '
                                                                                                                                            'fit/support for '
                                                                                                                                            'Task6/8/9; 10mm '
                                                                                                                                            'already-guided '
                                                                                                                                            'idler insertion '
                                                                                                                                            'for Task7. No '
                                                                                                                                            'robot operation '
                                                                                                                                            'credit.',
                                                                                                                                   'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_cable_routing/usability_v2/manifest.json',
                                                                                                                                                      'sha256': 'cd2f5f48258306ae8f0a3b4ff79c598c4c7e652ae371c9d6f3a637b527e29be5'},
                                                                                                                                   'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/qualification02.json',
                                                                                                                                                           'sha256': 'a9ce0d252506a14e28cba028091ad977bab2a77a46b7bb672296fd873f49e771'},
                                                                                                                                   'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/root-independent-identity-review02.json',
                                                                                                                                                             'sha256': '0f371c44ac87663055d2332be0c276fef810a9af83fff104c4f1b343c286d2bb'}}],
                                                                                                       'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                                                                                                                                     'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'}}}},
                                                 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                                   'assembly_version': 'isaacsim_6_1_v2',
                                                                                   'articulations': 1}},
 ('mechanical_assembly', 'nist_wire_harness'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/batch03/nist_wire_harness/desktop_v1',
                                                                 'manifest_sha256': 'd17ab5072ad1d49245cb715a6bf4a3f1233787483912aaa0ead8d4b06e6f3ba3'},
                                                'single_arm_layout': {'robot_layout_id': 'mechanical_assembly_nist_wire_harness_original_single_arm',
                                                                      'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                                                               'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                                                               'base_yaw_rad': 0.0,
                                                                                               'hand_chirality': 'right'}},
                                                                      'reference_surface': 'original fixed task installation',
                                                                      'frame': 'world',
                                                                      'units': 'metres',
                                                                      'control_scope': 'one_agent_29_dof'},
                                                'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/dual_batch03/mechanical_assembly/nist_wire_harness/dual_same_side/v1',
                                                                                                         'manifest_sha256': '67652c54ce0c4b01c0cfdad95e02a8374f6f1a142d5205461378667e948cb2c4'},
                                                                                      'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                      'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                      'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                             'assembly_version': 'isaacsim_6_1_v2',
                                                                                                             'articulations': 2,
                                                                                                             'active_sides': ['left', 'right']},
                                                                                      'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                                                                 'placements': {'left': {'base_position_m': [-0.1,
                                                                                                                                             0.3,
                                                                                                                                             0.78],
                                                                                                                         'base_quaternion_wxyz': [1.0,
                                                                                                                                                  0.0,
                                                                                                                                                  0.0,
                                                                                                                                                  0.0],
                                                                                                                         'base_yaw_rad': 0.0,
                                                                                                                         'hand_chirality': 'left'},
                                                                                                                'right': {'base_position_m': [-0.1,
                                                                                                                                              -0.3,
                                                                                                                                              0.78],
                                                                                                                          'base_quaternion_wxyz': [1.0,
                                                                                                                                                   0.0,
                                                                                                                                                   0.0,
                                                                                                                                                   0.0],
                                                                                                                          'base_yaw_rad': 0.0,
                                                                                                                          'hand_chirality': 'right'}},
                                                                                                 'reference_surface': 'fixed room task table',
                                                                                                 'frame': 'world',
                                                                                                 'units': 'metres',
                                                                                                 'control_scope': 'one_agent_58_dof'},
                                                                                      'qualification': {'native_preflight': False,
                                                                                                        'agent_bench_interface': False,
                                                                                                        'robot_manipulation': False,
                                                                                                        'physics_accepted': False,
                                                                                                        'workspace_discrete': True,
                                                                                                        'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace03/nist_wire_harness_dual_same_side.json',
                                                                                                                                    'sha256': '96c974dd5cb852b0933f99b0ea45dc53bf781d33c36ff404a6ab24aac4cb1e8a',
                                                                                                                                    'passed': True,
                                                                                                                                    'scope': 'Discrete '
                                                                                                                                             'actual '
                                                                                                                                             'fingertip '
                                                                                                                                             'approach and '
                                                                                                                                             'original USD '
                                                                                                                                             'collision '
                                                                                                                                             'bounds; no '
                                                                                                                                             'continuous '
                                                                                                                                             'path or '
                                                                                                                                             'manipulation '
                                                                                                                                             'qualification.'},
                                                                                                        'object_interactions': True,
                                                                                                        'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task9_goal03/qualification.json',
                                                                                                                                    'sha256': 'eb24f50ceb60d2d122db5befb9c67bf9f81a26944fda4bb39eeafd2c9bbdbaa2',
                                                                                                                                    'passed': True,
                                                                                                                                    'scope': 'Asset-specific '
                                                                                                                                             'passive goal '
                                                                                                                                             'fit/support '
                                                                                                                                             'for '
                                                                                                                                             'Task6/8/9; '
                                                                                                                                             '10mm '
                                                                                                                                             'already-guided '
                                                                                                                                             'idler '
                                                                                                                                             'insertion for '
                                                                                                                                             'Task7. No '
                                                                                                                                             'robot '
                                                                                                                                             'operation '
                                                                                                                                             'credit.',
                                                                                                                                    'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_wire_harness/usability_v2/manifest.json',
                                                                                                                                                       'sha256': '9443ce0e43ed40b114cef9668cc0364f85ff9658eb87aac0c464ec2d57d3f6ed'},
                                                                                                                                    'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/qualification02.json',
                                                                                                                                                            'sha256': 'a9ce0d252506a14e28cba028091ad977bab2a77a46b7bb672296fd873f49e771'},
                                                                                                                                    'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/root-independent-identity-review02.json',
                                                                                                                                                              'sha256': '0f371c44ac87663055d2332be0c276fef810a9af83fff104c4f1b343c286d2bb'}}],
                                                                                                        'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                                                                                                                                      'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'}}},
                                                                   'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261023/dual_batch03/mechanical_assembly/nist_wire_harness/dual_opposed/v1',
                                                                                                       'manifest_sha256': '3d96120c63f4e59475d97874b527db624ca062db074e6b7c5469fe02ccca7d54'},
                                                                                    'release_schema': 'workroom_robot_configuration_release.v1',
                                                                                    'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                                    'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                           'assembly_version': 'isaacsim_6_1_v2',
                                                                                                           'articulations': 2,
                                                                                                           'active_sides': ['left', 'right']},
                                                                                    'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                                                               'placements': {'left': {'base_position_m': [0.42, 0.76, 0.78],
                                                                                                                       'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                0.0,
                                                                                                                                                0.0,
                                                                                                                                                -0.7071067811865475],
                                                                                                                       'base_yaw_rad': -1.5707963267948966,
                                                                                                                       'hand_chirality': 'left'},
                                                                                                              'right': {'base_position_m': [0.42,
                                                                                                                                            -0.76,
                                                                                                                                            0.78],
                                                                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                                                 0.0,
                                                                                                                                                 0.0,
                                                                                                                                                 0.7071067811865475],
                                                                                                                        'base_yaw_rad': 1.5707963267948966,
                                                                                                                        'hand_chirality': 'right'}},
                                                                                               'reference_surface': 'fixed room task table',
                                                                                               'frame': 'world',
                                                                                               'units': 'metres',
                                                                                               'control_scope': 'one_agent_58_dof'},
                                                                                    'qualification': {'native_preflight': False,
                                                                                                      'agent_bench_interface': False,
                                                                                                      'robot_manipulation': False,
                                                                                                      'physics_accepted': False,
                                                                                                      'workspace_discrete': True,
                                                                                                      'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace03/nist_wire_harness_dual_opposed.json',
                                                                                                                                  'sha256': 'af6f1932bab40b09e3d3fdd551edc688099fd2a89f5639980b74e6aa4d1d6df9',
                                                                                                                                  'passed': True,
                                                                                                                                  'scope': 'Discrete actual '
                                                                                                                                           'fingertip '
                                                                                                                                           'approach and '
                                                                                                                                           'original USD '
                                                                                                                                           'collision '
                                                                                                                                           'bounds; no '
                                                                                                                                           'continuous path '
                                                                                                                                           'or manipulation '
                                                                                                                                           'qualification.'},
                                                                                                      'object_interactions': True,
                                                                                                      'scene_contact_evidence': [{'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/native_task9_goal03/qualification.json',
                                                                                                                                  'sha256': 'eb24f50ceb60d2d122db5befb9c67bf9f81a26944fda4bb39eeafd2c9bbdbaa2',
                                                                                                                                  'passed': True,
                                                                                                                                  'scope': 'Asset-specific '
                                                                                                                                           'passive goal '
                                                                                                                                           'fit/support for '
                                                                                                                                           'Task6/8/9; 10mm '
                                                                                                                                           'already-guided '
                                                                                                                                           'idler insertion '
                                                                                                                                           'for Task7. No '
                                                                                                                                           'robot operation '
                                                                                                                                           'credit.',
                                                                                                                                  'asset_manifest': {'path': f'{DATA_ROOT}/assets/task_parts/mechanical_assembly/nist_wire_harness/usability_v2/manifest.json',
                                                                                                                                                     'sha256': '9443ce0e43ed40b114cef9668cc0364f85ff9658eb87aac0c464ec2d57d3f6ed'},
                                                                                                                                  'qualification_index': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/qualification02.json',
                                                                                                                                                          'sha256': 'a9ce0d252506a14e28cba028091ad977bab2a77a46b7bb672296fd873f49e771'},
                                                                                                                                  'asset_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/scale_audit/root-independent-identity-review02.json',
                                                                                                                                                            'sha256': '0f371c44ac87663055d2332be0c276fef810a9af83fff104c4f1b343c286d2bb'}}],
                                                                                                      'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/nist_usability_20261023/root/workspace_binding03.json',
                                                                                                                                    'sha256': 'b3c6755b71c8dcd1bf6a7ef3e6c082e0aafd9c2629b3731bb01a82d66a9b286c'}}}},
                                                'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                                  'assembly_version': 'isaacsim_6_1_v2',
                                                                                  'articulations': 1}},
 ('study', 'stationery_sorting'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/study/stationery_sorting/v5',
                                                    'manifest_sha256': '29b22f2ceb94f0c114a75ab51a7b37bcfdd4ba39b3538ea9db73be390515ac4f'},
                                   'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                         'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                                                  'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                                                  'base_yaw_rad': 0.0,
                                                                                  'hand_chirality': 'right'}},
                                                         'reference_surface': 'original fixed task installation',
                                                         'frame': 'world',
                                                         'units': 'metres',
                                                         'control_scope': 'one_agent_29_dof'},
                                   'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261025/study_pencil_grip01/study/stationery_sorting/dual_same_side/v1',
                                                                                            'manifest_sha256': '71d9481fcec6aa809b54168a86cff5f07035806199df9d020412d69dce20395c'},
                                                                         'release_schema': 'workroom_robot_configuration_release.v1',
                                                                         'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                         'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                'assembly_version': 'isaacsim_6_1_v2',
                                                                                                'articulations': 2,
                                                                                                'active_sides': ['left', 'right']},
                                                                         'layout': {'robot_layout_id': 'study_dual_same_side_left_right_sharpa_v1',
                                                                                    'placements': {'left': {'base_position_m': [-0.23,
                                                                                                                                0.3,
                                                                                                                                0.78],
                                                                                                            'base_quaternion_wxyz': [1.0,
                                                                                                                                     0.0,
                                                                                                                                     0.0,
                                                                                                                                     0.0],
                                                                                                            'base_yaw_rad': 0.0,
                                                                                                            'hand_chirality': 'left'},
                                                                                                   'right': {'base_position_m': [-0.23,
                                                                                                                                 -0.3,
                                                                                                                                 0.78],
                                                                                                             'base_quaternion_wxyz': [1.0,
                                                                                                                                      0.0,
                                                                                                                                      0.0,
                                                                                                                                      0.0],
                                                                                                             'base_yaw_rad': 0.0,
                                                                                                             'hand_chirality': 'right'}},
                                                                                    'reference_surface': 'fixed room task table',
                                                                                    'frame': 'world',
                                                                                    'units': 'metres',
                                                                                    'control_scope': 'one_agent_58_dof'},
                                                                         'qualification': {'native_preflight': False,
                                                                                           'agent_bench_interface': False,
                                                                                           'robot_manipulation': False,
                                                                                           'physics_accepted': False}},
                                                      'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261025/study_pencil_grip01/study/stationery_sorting/dual_opposed/v1',
                                                                                          'manifest_sha256': 'dbee8b58560078f8a184d5f22e9852585a40b3ac38dba84c7e78457b7e298e59'},
                                                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                                                              'articulations': 2,
                                                                                              'active_sides': ['left', 'right']},
                                                                       'layout': {'robot_layout_id': 'study_dual_opposed_left_right_sharpa_v1',
                                                                                  'placements': {'left': {'base_position_m': [1.35, 0.0, 0.78],
                                                                                                          'base_quaternion_wxyz': [6.123233995736766e-17,
                                                                                                                                   0.0,
                                                                                                                                   0.0,
                                                                                                                                   1.0],
                                                                                                          'base_yaw_rad': 3.141592653589793,
                                                                                                          'hand_chirality': 'left'},
                                                                                                 'right': {'base_position_m': [-0.23,
                                                                                                                               0.0,
                                                                                                                               0.78],
                                                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                                                    0.0,
                                                                                                                                    0.0,
                                                                                                                                    0.0],
                                                                                                           'base_yaw_rad': 0.0,
                                                                                                           'hand_chirality': 'right'}},
                                                                                  'reference_surface': 'fixed room task table',
                                                                                  'frame': 'world',
                                                                                  'units': 'metres',
                                                                                  'control_scope': 'one_agent_58_dof'},
                                                                       'qualification': {'native_preflight': False,
                                                                                         'agent_bench_interface': False,
                                                                                         'robot_manipulation': False,
                                                                                         'physics_accepted': False}}},
                                   'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                     'assembly_version': 'isaacsim_6_1_v2',
                                                                     'articulations': 1}},
 ('study', 'book_stacking'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/study/book_stacking/v3',
                                               'manifest_sha256': '842db0ee31458aaacfb7b75fc70e5343b4eb4a3b8a4423cd7e8e2ac77dc66136'},
                              'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                    'placements': {'right': {'base_position_m': [-0.23,
                                                                                                 0.0,
                                                                                                 0.78],
                                                                             'base_quaternion_wxyz': [1.0,
                                                                                                      0.0,
                                                                                                      0.0,
                                                                                                      0.0],
                                                                             'base_yaw_rad': 0.0,
                                                                             'hand_chirality': 'right'}},
                                                    'reference_surface': 'original fixed task installation',
                                                    'frame': 'world',
                                                    'units': 'metres',
                                                    'control_scope': 'one_agent_29_dof'},
                              'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/study/book_stacking/dual_same_side/v1',
                                                                                       'manifest_sha256': 'ce40716bf6e36086239d22442c9e3fdc5c5d8208bc8af9ae3bd4301484b301d1'},
                                                                    'release_schema': 'workroom_robot_configuration_release.v1',
                                                                    'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                    'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                           'assembly_version': 'isaacsim_6_1_v2',
                                                                                           'articulations': 2,
                                                                                           'active_sides': ['left',
                                                                                                            'right']},
                                                                    'layout': {'robot_layout_id': 'study_dual_same_side_left_right_sharpa_v1',
                                                                               'placements': {'left': {'base_position_m': [-0.23,
                                                                                                                           0.3,
                                                                                                                           0.78],
                                                                                                       'base_quaternion_wxyz': [1.0,
                                                                                                                                0.0,
                                                                                                                                0.0,
                                                                                                                                0.0],
                                                                                                       'base_yaw_rad': 0.0,
                                                                                                       'hand_chirality': 'left'},
                                                                                              'right': {'base_position_m': [-0.23,
                                                                                                                            -0.3,
                                                                                                                            0.78],
                                                                                                        'base_quaternion_wxyz': [1.0,
                                                                                                                                 0.0,
                                                                                                                                 0.0,
                                                                                                                                 0.0],
                                                                                                        'base_yaw_rad': 0.0,
                                                                                                        'hand_chirality': 'right'}},
                                                                               'reference_surface': 'fixed '
                                                                                                    'room '
                                                                                                    'task '
                                                                                                    'table',
                                                                               'frame': 'world',
                                                                               'units': 'metres',
                                                                               'control_scope': 'one_agent_58_dof'},
                                                                    'qualification': {'native_preflight': False,
                                                                                      'agent_bench_interface': False,
                                                                                      'robot_manipulation': False,
                                                                                      'physics_accepted': False,
                                                                                      'workspace_ik': True,
                                                                                      'workspace_clearance': True,
                                                                                      'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/study_book_stacking_dual_same_side.json',
                                                                                                             'sha256': '42b3af97598cfce90ec79b8d40ccac60fad7aedaba1c9cff5f9286376c4609ce'},
                                                                                      'qualification_scope': 'Discrete '
                                                                                                             'fingertip '
                                                                                                             'approach '
                                                                                                             'IK '
                                                                                                             'and '
                                                                                                             'original '
                                                                                                             'shape '
                                                                                                             'AABB '
                                                                                                             'clearance; '
                                                                                                             'continuous '
                                                                                                             'motion '
                                                                                                             'and '
                                                                                                             'manipulation '
                                                                                                             'remain '
                                                                                                             'unqualified.',
                                                                                      'nonadjacent_arm_self_collision': True,
                                                                                      'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                      'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                 'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/study/book_stacking/dual_opposed/v1',
                                                                                     'manifest_sha256': 'c0391c9e86bba49ba55e0e364753472f5ef65a1c7dbce435803c3ffe18bbcbea'},
                                                                  'release_schema': 'workroom_robot_configuration_release.v1',
                                                                  'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                  'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                                         'articulations': 2,
                                                                                         'active_sides': ['left',
                                                                                                          'right']},
                                                                  'layout': {'robot_layout_id': 'study_dual_opposed_left_right_sharpa_v1',
                                                                             'placements': {'left': {'base_position_m': [1.35,
                                                                                                                         0.0,
                                                                                                                         0.78],
                                                                                                     'base_quaternion_wxyz': [6.123233995736766e-17,
                                                                                                                              0.0,
                                                                                                                              0.0,
                                                                                                                              1.0],
                                                                                                     'base_yaw_rad': 3.141592653589793,
                                                                                                     'hand_chirality': 'left'},
                                                                                            'right': {'base_position_m': [-0.23,
                                                                                                                          0.0,
                                                                                                                          0.78],
                                                                                                      'base_quaternion_wxyz': [1.0,
                                                                                                                               0.0,
                                                                                                                               0.0,
                                                                                                                               0.0],
                                                                                                      'base_yaw_rad': 0.0,
                                                                                                      'hand_chirality': 'right'}},
                                                                             'reference_surface': 'fixed '
                                                                                                  'room task '
                                                                                                  'table',
                                                                             'frame': 'world',
                                                                             'units': 'metres',
                                                                             'control_scope': 'one_agent_58_dof'},
                                                                  'qualification': {'native_preflight': False,
                                                                                    'agent_bench_interface': False,
                                                                                    'robot_manipulation': False,
                                                                                    'physics_accepted': False,
                                                                                    'workspace_ik': True,
                                                                                    'workspace_clearance': True,
                                                                                    'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/study_book_stacking_dual_opposed.json',
                                                                                                           'sha256': '2374082edd6a49e2dc59702034ef6f02d9de6c8816f143a068684dff3f4f27ce'},
                                                                                    'qualification_scope': 'Discrete '
                                                                                                           'fingertip '
                                                                                                           'approach '
                                                                                                           'IK '
                                                                                                           'and '
                                                                                                           'original '
                                                                                                           'shape '
                                                                                                           'AABB '
                                                                                                           'clearance; '
                                                                                                           'continuous '
                                                                                                           'motion '
                                                                                                           'and '
                                                                                                           'manipulation '
                                                                                                           'remain '
                                                                                                           'unqualified.',
                                                                                    'nonadjacent_arm_self_collision': True,
                                                                                    'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                    'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                              'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                'assembly_version': 'isaacsim_6_1_v2',
                                                                'articulations': 1}},
 ('study', 'book_retrieval'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/study/book_retrieval/v3',
                                                'manifest_sha256': 'ed0a9dc880c2e9633f61276116ed03fe1e24a1cb9655fd97a455addd5f7158e1'},
                               'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                     'placements': {'right': {'base_position_m': [-0.23,
                                                                                                  0.0,
                                                                                                  0.78],
                                                                              'base_quaternion_wxyz': [1.0,
                                                                                                       0.0,
                                                                                                       0.0,
                                                                                                       0.0],
                                                                              'base_yaw_rad': 0.0,
                                                                              'hand_chirality': 'right'}},
                                                     'reference_surface': 'original fixed task installation',
                                                     'frame': 'world',
                                                     'units': 'metres',
                                                     'control_scope': 'one_agent_29_dof'},
                               'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/study/book_retrieval/dual_same_side/v1',
                                                                                        'manifest_sha256': '5fe3f8a4c545a02c109fe50e2d476032d1deebe2485aeb7ef01e9ae03433dfbf'},
                                                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                                                            'articulations': 2,
                                                                                            'active_sides': ['left',
                                                                                                             'right']},
                                                                     'layout': {'robot_layout_id': 'study_dual_same_side_left_right_sharpa_v1',
                                                                                'placements': {'left': {'base_position_m': [-0.23,
                                                                                                                            0.3,
                                                                                                                            0.78],
                                                                                                        'base_quaternion_wxyz': [1.0,
                                                                                                                                 0.0,
                                                                                                                                 0.0,
                                                                                                                                 0.0],
                                                                                                        'base_yaw_rad': 0.0,
                                                                                                        'hand_chirality': 'left'},
                                                                                               'right': {'base_position_m': [-0.23,
                                                                                                                             -0.3,
                                                                                                                             0.78],
                                                                                                         'base_quaternion_wxyz': [1.0,
                                                                                                                                  0.0,
                                                                                                                                  0.0,
                                                                                                                                  0.0],
                                                                                                         'base_yaw_rad': 0.0,
                                                                                                         'hand_chirality': 'right'}},
                                                                                'reference_surface': 'fixed '
                                                                                                     'room '
                                                                                                     'task '
                                                                                                     'table',
                                                                                'frame': 'world',
                                                                                'units': 'metres',
                                                                                'control_scope': 'one_agent_58_dof'},
                                                                     'qualification': {'native_preflight': False,
                                                                                       'agent_bench_interface': False,
                                                                                       'robot_manipulation': False,
                                                                                       'physics_accepted': False,
                                                                                       'workspace_ik': True,
                                                                                       'workspace_clearance': True,
                                                                                       'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/study_book_retrieval_dual_same_side.json',
                                                                                                              'sha256': 'f5225a30b9f0321572e3edf9a8cf5deac6d2f51af5043e29d094beeea41abad1'},
                                                                                       'qualification_scope': 'Discrete '
                                                                                                              'fingertip '
                                                                                                              'approach '
                                                                                                              'IK '
                                                                                                              'and '
                                                                                                              'original '
                                                                                                              'shape '
                                                                                                              'AABB '
                                                                                                              'clearance; '
                                                                                                              'continuous '
                                                                                                              'motion '
                                                                                                              'and '
                                                                                                              'manipulation '
                                                                                                              'remain '
                                                                                                              'unqualified.',
                                                                                       'nonadjacent_arm_self_collision': True,
                                                                                       'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                       'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                  'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/study/book_retrieval/dual_opposed/v1',
                                                                                      'manifest_sha256': '7a2b60fd79233e7880486dcd649a16cfdac5f5d82782e4fbd20b5c486364d999'},
                                                                   'release_schema': 'workroom_robot_configuration_release.v1',
                                                                   'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                   'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                          'assembly_version': 'isaacsim_6_1_v2',
                                                                                          'articulations': 2,
                                                                                          'active_sides': ['left',
                                                                                                           'right']},
                                                                   'layout': {'robot_layout_id': 'study_dual_opposed_left_right_sharpa_v1',
                                                                              'placements': {'left': {'base_position_m': [1.35,
                                                                                                                          0.0,
                                                                                                                          0.78],
                                                                                                      'base_quaternion_wxyz': [6.123233995736766e-17,
                                                                                                                               0.0,
                                                                                                                               0.0,
                                                                                                                               1.0],
                                                                                                      'base_yaw_rad': 3.141592653589793,
                                                                                                      'hand_chirality': 'left'},
                                                                                             'right': {'base_position_m': [-0.23,
                                                                                                                           0.0,
                                                                                                                           0.78],
                                                                                                       'base_quaternion_wxyz': [1.0,
                                                                                                                                0.0,
                                                                                                                                0.0,
                                                                                                                                0.0],
                                                                                                       'base_yaw_rad': 0.0,
                                                                                                       'hand_chirality': 'right'}},
                                                                              'reference_surface': 'fixed '
                                                                                                   'room '
                                                                                                   'task '
                                                                                                   'table',
                                                                              'frame': 'world',
                                                                              'units': 'metres',
                                                                              'control_scope': 'one_agent_58_dof'},
                                                                   'qualification': {'native_preflight': False,
                                                                                     'agent_bench_interface': False,
                                                                                     'robot_manipulation': False,
                                                                                     'physics_accepted': False,
                                                                                     'workspace_ik': True,
                                                                                     'workspace_clearance': True,
                                                                                     'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/study_book_retrieval_dual_opposed.json',
                                                                                                            'sha256': '0eed130848c12416e5a04704cca05d4aa5be2cdec4dd91ed1dd3bca7d2a14e59'},
                                                                                     'qualification_scope': 'Discrete '
                                                                                                            'fingertip '
                                                                                                            'approach '
                                                                                                            'IK '
                                                                                                            'and '
                                                                                                            'original '
                                                                                                            'shape '
                                                                                                            'AABB '
                                                                                                            'clearance; '
                                                                                                            'continuous '
                                                                                                            'motion '
                                                                                                            'and '
                                                                                                            'manipulation '
                                                                                                            'remain '
                                                                                                            'unqualified.',
                                                                                     'nonadjacent_arm_self_collision': True,
                                                                                     'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                     'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                               'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                 'assembly_version': 'isaacsim_6_1_v2',
                                                                 'articulations': 1}},
 ('study', 'book_shelving'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/study/book_shelving/v3',
                                               'manifest_sha256': '04ecff976951a3d4257caf90523b0cd6f39df4a0ceea78bdabf0b87a56586b15'},
                              'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                    'placements': {'right': {'base_position_m': [-0.23,
                                                                                                 0.0,
                                                                                                 0.78],
                                                                             'base_quaternion_wxyz': [1.0,
                                                                                                      0.0,
                                                                                                      0.0,
                                                                                                      0.0],
                                                                             'base_yaw_rad': 0.0,
                                                                             'hand_chirality': 'right'}},
                                                    'reference_surface': 'original fixed task installation',
                                                    'frame': 'world',
                                                    'units': 'metres',
                                                    'control_scope': 'one_agent_29_dof'},
                              'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/study/book_shelving/dual_same_side/v1',
                                                                                       'manifest_sha256': 'ecd2044614d2bb88dbfbacc4b4553493373d0597113b37667fddada56c81f7b1'},
                                                                    'release_schema': 'workroom_robot_configuration_release.v1',
                                                                    'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                    'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                           'assembly_version': 'isaacsim_6_1_v2',
                                                                                           'articulations': 2,
                                                                                           'active_sides': ['left',
                                                                                                            'right']},
                                                                    'layout': {'robot_layout_id': 'study_dual_same_side_left_right_sharpa_v1',
                                                                               'placements': {'left': {'base_position_m': [-0.23,
                                                                                                                           0.3,
                                                                                                                           0.78],
                                                                                                       'base_quaternion_wxyz': [1.0,
                                                                                                                                0.0,
                                                                                                                                0.0,
                                                                                                                                0.0],
                                                                                                       'base_yaw_rad': 0.0,
                                                                                                       'hand_chirality': 'left'},
                                                                                              'right': {'base_position_m': [-0.23,
                                                                                                                            -0.3,
                                                                                                                            0.78],
                                                                                                        'base_quaternion_wxyz': [1.0,
                                                                                                                                 0.0,
                                                                                                                                 0.0,
                                                                                                                                 0.0],
                                                                                                        'base_yaw_rad': 0.0,
                                                                                                        'hand_chirality': 'right'}},
                                                                               'reference_surface': 'fixed '
                                                                                                    'room '
                                                                                                    'task '
                                                                                                    'table',
                                                                               'frame': 'world',
                                                                               'units': 'metres',
                                                                               'control_scope': 'one_agent_58_dof'},
                                                                    'qualification': {'native_preflight': False,
                                                                                      'agent_bench_interface': False,
                                                                                      'robot_manipulation': False,
                                                                                      'physics_accepted': False,
                                                                                      'workspace_ik': True,
                                                                                      'workspace_clearance': True,
                                                                                      'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/study_book_shelving_dual_same_side.json',
                                                                                                             'sha256': '2edf1439d2f468e7d5edd4ecb62788a5b8ac3e279f42087949b0808e756f5ade'},
                                                                                      'qualification_scope': 'Discrete '
                                                                                                             'fingertip '
                                                                                                             'approach '
                                                                                                             'IK '
                                                                                                             'and '
                                                                                                             'original '
                                                                                                             'shape '
                                                                                                             'AABB '
                                                                                                             'clearance; '
                                                                                                             'continuous '
                                                                                                             'motion '
                                                                                                             'and '
                                                                                                             'manipulation '
                                                                                                             'remain '
                                                                                                             'unqualified.',
                                                                                      'nonadjacent_arm_self_collision': True,
                                                                                      'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                      'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                 'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/study/book_shelving/dual_opposed/v1',
                                                                                     'manifest_sha256': '092571bbac57cb7b8427ca3674dd7333ded4bc349a02acd70b39e1a5489ac129'},
                                                                  'release_schema': 'workroom_robot_configuration_release.v1',
                                                                  'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                  'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                                         'articulations': 2,
                                                                                         'active_sides': ['left',
                                                                                                          'right']},
                                                                  'layout': {'robot_layout_id': 'study_dual_opposed_left_right_sharpa_v1',
                                                                             'placements': {'left': {'base_position_m': [1.35,
                                                                                                                         0.0,
                                                                                                                         0.78],
                                                                                                     'base_quaternion_wxyz': [6.123233995736766e-17,
                                                                                                                              0.0,
                                                                                                                              0.0,
                                                                                                                              1.0],
                                                                                                     'base_yaw_rad': 3.141592653589793,
                                                                                                     'hand_chirality': 'left'},
                                                                                            'right': {'base_position_m': [-0.23,
                                                                                                                          0.0,
                                                                                                                          0.78],
                                                                                                      'base_quaternion_wxyz': [1.0,
                                                                                                                               0.0,
                                                                                                                               0.0,
                                                                                                                               0.0],
                                                                                                      'base_yaw_rad': 0.0,
                                                                                                      'hand_chirality': 'right'}},
                                                                             'reference_surface': 'fixed '
                                                                                                  'room task '
                                                                                                  'table',
                                                                             'frame': 'world',
                                                                             'units': 'metres',
                                                                             'control_scope': 'one_agent_58_dof'},
                                                                  'qualification': {'native_preflight': False,
                                                                                    'agent_bench_interface': False,
                                                                                    'robot_manipulation': False,
                                                                                    'physics_accepted': False,
                                                                                    'workspace_ik': True,
                                                                                    'workspace_clearance': True,
                                                                                    'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/study_book_shelving_dual_opposed.json',
                                                                                                           'sha256': '0e13ea02a3b18c4e39fa8c951d05ee821e9cc86c4e8d7c6a65953b31499f6f31'},
                                                                                    'qualification_scope': 'Discrete '
                                                                                                           'fingertip '
                                                                                                           'approach '
                                                                                                           'IK '
                                                                                                           'and '
                                                                                                           'original '
                                                                                                           'shape '
                                                                                                           'AABB '
                                                                                                           'clearance; '
                                                                                                           'continuous '
                                                                                                           'motion '
                                                                                                           'and '
                                                                                                           'manipulation '
                                                                                                           'remain '
                                                                                                           'unqualified.',
                                                                                    'nonadjacent_arm_self_collision': True,
                                                                                    'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                    'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                              'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                'assembly_version': 'isaacsim_6_1_v2',
                                                                'articulations': 1}},
 ('study', 'drawer_retrieval'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/study/drawer_retrieval/v4',
                                                  'manifest_sha256': '08e4a953d8f31f49619d0b8a10086a287c8a3b7a1b9b04d147bbd7d0ad4913f0'},
                                 'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                       'placements': {'right': {'base_position_m': [-0.23,
                                                                                                    0.0,
                                                                                                    0.78],
                                                                                'base_quaternion_wxyz': [1.0,
                                                                                                         0.0,
                                                                                                         0.0,
                                                                                                         0.0],
                                                                                'base_yaw_rad': 0.0,
                                                                                'hand_chirality': 'right'}},
                                                       'reference_surface': 'original fixed task '
                                                                            'installation',
                                                       'frame': 'world',
                                                       'units': 'metres',
                                                       'control_scope': 'one_agent_29_dof'},
                                 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/study/drawer_retrieval/dual_same_side/v1',
                                                                                          'manifest_sha256': 'ea716a84466b56413768ab442b299f9baf3fb3c378407380d7e93b2f039f9dd9'},
                                                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                                                              'articulations': 2,
                                                                                              'active_sides': ['left',
                                                                                                               'right']},
                                                                       'layout': {'robot_layout_id': 'study_dual_same_side_left_right_sharpa_v1',
                                                                                  'placements': {'left': {'base_position_m': [-0.23,
                                                                                                                              0.3,
                                                                                                                              0.78],
                                                                                                          'base_quaternion_wxyz': [1.0,
                                                                                                                                   0.0,
                                                                                                                                   0.0,
                                                                                                                                   0.0],
                                                                                                          'base_yaw_rad': 0.0,
                                                                                                          'hand_chirality': 'left'},
                                                                                                 'right': {'base_position_m': [-0.23,
                                                                                                                               -0.3,
                                                                                                                               0.78],
                                                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                                                    0.0,
                                                                                                                                    0.0,
                                                                                                                                    0.0],
                                                                                                           'base_yaw_rad': 0.0,
                                                                                                           'hand_chirality': 'right'}},
                                                                                  'reference_surface': 'fixed '
                                                                                                       'room '
                                                                                                       'task '
                                                                                                       'table',
                                                                                  'frame': 'world',
                                                                                  'units': 'metres',
                                                                                  'control_scope': 'one_agent_58_dof'},
                                                                       'qualification': {'native_preflight': False,
                                                                                         'agent_bench_interface': False,
                                                                                         'robot_manipulation': False,
                                                                                         'physics_accepted': False,
                                                                                         'workspace_ik': True,
                                                                                         'workspace_clearance': True,
                                                                                         'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/study_drawer_retrieval_dual_same_side.json',
                                                                                                                'sha256': 'f1d107026843403c03c18e565717f5495a0bdb75c8c00a6c4a73aed0d60f8bb7'},
                                                                                         'qualification_scope': 'Discrete '
                                                                                                                'fingertip '
                                                                                                                'approach '
                                                                                                                'IK '
                                                                                                                'and '
                                                                                                                'original '
                                                                                                                'shape '
                                                                                                                'AABB '
                                                                                                                'clearance; '
                                                                                                                'continuous '
                                                                                                                'motion '
                                                                                                                'and '
                                                                                                                'manipulation '
                                                                                                                'remain '
                                                                                                                'unqualified.',
                                                                                         'nonadjacent_arm_self_collision': True,
                                                                                         'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                         'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/study/drawer_retrieval/dual_opposed/v1',
                                                                                        'manifest_sha256': 'b8d974b1aa897a49aa044f90f7ebad06f0e11454a9f6c5bedbc2be72e96a226c'},
                                                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                                                            'articulations': 2,
                                                                                            'active_sides': ['left',
                                                                                                             'right']},
                                                                     'layout': {'robot_layout_id': 'study_dual_opposed_left_right_sharpa_v1',
                                                                                'placements': {'left': {'base_position_m': [1.35,
                                                                                                                            0.0,
                                                                                                                            0.78],
                                                                                                        'base_quaternion_wxyz': [6.123233995736766e-17,
                                                                                                                                 0.0,
                                                                                                                                 0.0,
                                                                                                                                 1.0],
                                                                                                        'base_yaw_rad': 3.141592653589793,
                                                                                                        'hand_chirality': 'left'},
                                                                                               'right': {'base_position_m': [-0.23,
                                                                                                                             0.0,
                                                                                                                             0.78],
                                                                                                         'base_quaternion_wxyz': [1.0,
                                                                                                                                  0.0,
                                                                                                                                  0.0,
                                                                                                                                  0.0],
                                                                                                         'base_yaw_rad': 0.0,
                                                                                                         'hand_chirality': 'right'}},
                                                                                'reference_surface': 'fixed '
                                                                                                     'room '
                                                                                                     'task '
                                                                                                     'table',
                                                                                'frame': 'world',
                                                                                'units': 'metres',
                                                                                'control_scope': 'one_agent_58_dof'},
                                                                     'qualification': {'native_preflight': False,
                                                                                       'agent_bench_interface': False,
                                                                                       'robot_manipulation': False,
                                                                                       'physics_accepted': False,
                                                                                       'workspace_ik': True,
                                                                                       'workspace_clearance': True,
                                                                                       'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/study_drawer_retrieval_dual_opposed.json',
                                                                                                              'sha256': '418c7634a2704e5ae4c20f3d808205cf6205113705e90c0e7074a41fd6fe29b2'},
                                                                                       'qualification_scope': 'Discrete '
                                                                                                              'fingertip '
                                                                                                              'approach '
                                                                                                              'IK '
                                                                                                              'and '
                                                                                                              'original '
                                                                                                              'shape '
                                                                                                              'AABB '
                                                                                                              'clearance; '
                                                                                                              'continuous '
                                                                                                              'motion '
                                                                                                              'and '
                                                                                                              'manipulation '
                                                                                                              'remain '
                                                                                                              'unqualified.',
                                                                                       'nonadjacent_arm_self_collision': True,
                                                                                       'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                       'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                                 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                   'assembly_version': 'isaacsim_6_1_v2',
                                                                   'articulations': 1}},
 ('living_room', 'remote_operation'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/v7',
                                                        'manifest_sha256': '13b29b28b9fcc300126fee5b642938546367e71448afa71276ebe799e37d5349'},
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
                                                             'reference_surface': 'original fixed task '
                                                                                  'installation',
                                                             'frame': 'world',
                                                             'units': 'metres',
                                                             'control_scope': 'one_agent_29_dof'},
                                       'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/living_room/remote_operation/dual_same_side/v1',
                                                                                                'manifest_sha256': 'febd1eb1a98bef7c518672d285ecef3d11e103869f5414db99c9942825ccb836'},
                                                                             'release_schema': 'workroom_robot_configuration_release.v1',
                                                                             'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
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
                                                                             'qualification': {'native_preflight': False,
                                                                                               'agent_bench_interface': False,
                                                                                               'robot_manipulation': False,
                                                                                               'physics_accepted': False,
                                                                                               'workspace_ik': True,
                                                                                               'workspace_clearance': True,
                                                                                               'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/living_room_remote_operation_dual_same_side.json',
                                                                                                                      'sha256': 'a6cf6a5ed5add0ce122408944a0a660484137453ccaf6b26586c858b8bd9f7a7'},
                                                                                               'qualification_scope': 'Discrete '
                                                                                                                      'fingertip '
                                                                                                                      'approach '
                                                                                                                      'IK '
                                                                                                                      'and '
                                                                                                                      'original '
                                                                                                                      'shape '
                                                                                                                      'AABB '
                                                                                                                      'clearance; '
                                                                                                                      'continuous '
                                                                                                                      'motion '
                                                                                                                      'and '
                                                                                                                      'manipulation '
                                                                                                                      'remain '
                                                                                                                      'unqualified.',
                                                                                               'nonadjacent_arm_self_collision': True,
                                                                                               'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                               'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                          'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/living_room/remote_operation/dual_opposed/v1',
                                                                                              'manifest_sha256': 'd3f2e980e9be208734d1b188a28eb614df8b9df842d8b14150ce7529da384d86'},
                                                                           'release_schema': 'workroom_robot_configuration_release.v1',
                                                                           'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
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
                                                                           'qualification': {'native_preflight': False,
                                                                                             'agent_bench_interface': False,
                                                                                             'robot_manipulation': False,
                                                                                             'physics_accepted': False,
                                                                                             'workspace_ik': True,
                                                                                             'workspace_clearance': True,
                                                                                             'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/living_room_remote_operation_dual_opposed.json',
                                                                                                                    'sha256': 'dde9de6d22a64e77b585ddbfcf8adf080430136d92f1bfec9584b98f4af76c5c'},
                                                                                             'qualification_scope': 'Discrete '
                                                                                                                    'fingertip '
                                                                                                                    'approach '
                                                                                                                    'IK '
                                                                                                                    'and '
                                                                                                                    'original '
                                                                                                                    'shape '
                                                                                                                    'AABB '
                                                                                                                    'clearance; '
                                                                                                                    'continuous '
                                                                                                                    'motion '
                                                                                                                    'and '
                                                                                                                    'manipulation '
                                                                                                                    'remain '
                                                                                                                    'unqualified.',
                                                                                             'nonadjacent_arm_self_collision': True,
                                                                                             'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                             'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                                       'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                         'articulations': 1}},
 ('living_room', 'water_pouring'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/water_pouring/v6',
                                                     'manifest_sha256': 'c31bd1582381ad850b3363570e701cd2623e720c20ed6b797d72461d412d28b5'},
                                    'single_arm_layout': {'robot_layout_id': 'living_room_water_pouring_original_single_arm',
                                                          'placements': {'right': {'base_position_m': [0.65, -0.76, 0.78],
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
                                    'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261024/batch02/living_room/water_pouring/dual_same_side/v1',
                                                                                             'manifest_sha256': 'cf562e039e1157e66fe4655320280934c51afd1be74efeea0d0fae61c5c14904'},
                                                                          'release_schema': 'workroom_robot_configuration_release.v1',
                                                                          'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                          'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                                 'assembly_version': 'isaacsim_6_1_v2',
                                                                                                 'articulations': 2,
                                                                                                 'active_sides': ['left', 'right']},
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
                                                                                     'reference_surface': 'fixed room task table',
                                                                                     'frame': 'world',
                                                                                     'units': 'metres',
                                                                                     'control_scope': 'one_agent_58_dof'},
                                                                          'qualification': {'native_preflight': True,
                                                                                            'agent_bench_interface': True,
                                                                                            'robot_manipulation': False,
                                                                                            'physics_accepted': False,
                                                                                            'workspace_ik': True,
                                                                                            'workspace_clearance': True,
                                                                                            'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/living_room_water_pouring_dual_same_side.json',
                                                                                                                   'sha256': '584688593d32cc799bfb5c8a032b34453f7d65626d47316602fb73ea95c1bb1a'},
                                                                                            'qualification_scope': 'Current exact '
                                                                                                                   'release/source_runtime05 '
                                                                                                                   'HTTP and '
                                                                                                                   'one-Agent/58-DOF '
                                                                                                                   'checks passed. Prior '
                                                                                                                   'discrete IK/clearance '
                                                                                                                   'evidence remains bound '
                                                                                                                   'through unchanged '
                                                                                                                   'physical scene/robot '
                                                                                                                   'identity; no new '
                                                                                                                   'trajectory, '
                                                                                                                   'manipulation or force '
                                                                                                                   'calibration claim.',
                                                                                            'nonadjacent_arm_self_collision': True,
                                                                                            'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                            'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'},
                                                                                            'native_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/root/http_dual_same_side02/qualification.json',
                                                                                                                'sha256': '16d2603c83c0bcc0e10238b6f3aa82ff4451bcf1f8bbe90d856738cc90d52030'},
                                                                                            'object_interaction_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/task2/qualification.json',
                                                                                                                            'sha256': '8ba8c0229257c59411af544b11c173ac65eb7c06df9aeac9852a478d0a9fb7b3',
                                                                                                                            'scope': 'Native '
                                                                                                                                     'object-only '
                                                                                                                                     'retention, '
                                                                                                                                     'aimed '
                                                                                                                                     'and '
                                                                                                                                     'missed '
                                                                                                                                     'pouring; '
                                                                                                                                     'not '
                                                                                                                                     'robot '
                                                                                                                                     'manipulation.'},
                                                                                            'physical_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/root/independent_release_review02.json',
                                                                                                                         'sha256': '9f3ef39571953342557957ef57f60bfa7b4b3ef0718fa1f6b9b90118fd612a14'}}},
                                                       'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261024/batch02/living_room/water_pouring/dual_opposed/v1',
                                                                                           'manifest_sha256': 'f15592e5476a8d17e352cf846da118c421bc57470ac5feb61ea97c7ce70446c3'},
                                                                        'release_schema': 'workroom_robot_configuration_release.v1',
                                                                        'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                        'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                               'assembly_version': 'isaacsim_6_1_v2',
                                                                                               'articulations': 2,
                                                                                               'active_sides': ['left', 'right']},
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
                                                                                   'reference_surface': 'fixed room task table',
                                                                                   'frame': 'world',
                                                                                   'units': 'metres',
                                                                                   'control_scope': 'one_agent_58_dof'},
                                                                        'qualification': {'native_preflight': True,
                                                                                          'agent_bench_interface': True,
                                                                                          'robot_manipulation': False,
                                                                                          'physics_accepted': False,
                                                                                          'workspace_ik': True,
                                                                                          'workspace_clearance': True,
                                                                                          'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/living_room_water_pouring_dual_opposed.json',
                                                                                                                 'sha256': 'c0940d6d4f6e17aa6def998e7d47e19e371d7738fff6d8d4b3bf486a97e3a878'},
                                                                                          'qualification_scope': 'Current exact '
                                                                                                                 'release/source_runtime05 '
                                                                                                                 'HTTP and '
                                                                                                                 'one-Agent/58-DOF checks '
                                                                                                                 'passed. Prior discrete '
                                                                                                                 'IK/clearance evidence '
                                                                                                                 'remains bound through '
                                                                                                                 'unchanged physical '
                                                                                                                 'scene/robot identity; no '
                                                                                                                 'new trajectory, '
                                                                                                                 'manipulation or force '
                                                                                                                 'calibration claim.',
                                                                                          'nonadjacent_arm_self_collision': True,
                                                                                          'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                          'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'},
                                                                                          'native_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/root/http_dual_opposed02/qualification.json',
                                                                                                              'sha256': 'cc512eeecc2253e9aceb0dc1726e9925d7fc4de52c6d6adb937a07a5d2686493'},
                                                                                          'object_interaction_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/task2/qualification.json',
                                                                                                                          'sha256': '8ba8c0229257c59411af544b11c173ac65eb7c06df9aeac9852a478d0a9fb7b3',
                                                                                                                          'scope': 'Native '
                                                                                                                                   'object-only '
                                                                                                                                   'retention, '
                                                                                                                                   'aimed '
                                                                                                                                   'and '
                                                                                                                                   'missed '
                                                                                                                                   'pouring; '
                                                                                                                                   'not '
                                                                                                                                   'robot '
                                                                                                                                   'manipulation.'},
                                                                                          'physical_identity_bridge': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/root/independent_release_review02.json',
                                                                                                                       'sha256': '9f3ef39571953342557957ef57f60bfa7b4b3ef0718fa1f6b9b90118fd612a14'}}}},
                                    'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                      'assembly_version': 'isaacsim_6_1_v2',
                                                                      'articulations': 1}},
 ('living_room', 'tower_building'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/tower_building/v5',
                                                      'manifest_sha256': '59c32f726e715b3b19edff4ca4aeed7d156091eb1460e86044fb9ba4b880b374'},
                                     'single_arm_layout': {'robot_layout_id': 'living_room_tower_building_original_single_arm',
                                                           'placements': {'right': {'base_position_m': [0.65,
                                                                                                        -0.88,
                                                                                                        0.78],
                                                                                    'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                             0.0,
                                                                                                             0.0,
                                                                                                             0.7071067811865476],
                                                                                    'base_yaw_rad': 1.5707963267948968,
                                                                                    'hand_chirality': 'right'}},
                                                           'reference_surface': 'original fixed task '
                                                                                'installation',
                                                           'frame': 'world',
                                                           'units': 'metres',
                                                           'control_scope': 'one_agent_29_dof'},
                                     'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/living_room/tower_building/dual_same_side/v1',
                                                                                              'manifest_sha256': '02a5d0a9a272ff75b4e1dcff39d6bb8fff771b03e18e0a87d6eda540251ae4b8'},
                                                                           'release_schema': 'workroom_robot_configuration_release.v1',
                                                                           'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
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
                                                                           'qualification': {'native_preflight': False,
                                                                                             'agent_bench_interface': False,
                                                                                             'robot_manipulation': False,
                                                                                             'physics_accepted': False,
                                                                                             'workspace_ik': True,
                                                                                             'workspace_clearance': True,
                                                                                             'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/living_room_tower_building_dual_same_side.json',
                                                                                                                    'sha256': '7c5d78fd6ccf205b0616c338ce109bc587b7da94d6d130be79d0d938d476450b'},
                                                                                             'qualification_scope': 'Discrete '
                                                                                                                    'fingertip '
                                                                                                                    'approach '
                                                                                                                    'IK '
                                                                                                                    'and '
                                                                                                                    'original '
                                                                                                                    'shape '
                                                                                                                    'AABB '
                                                                                                                    'clearance; '
                                                                                                                    'continuous '
                                                                                                                    'motion '
                                                                                                                    'and '
                                                                                                                    'manipulation '
                                                                                                                    'remain '
                                                                                                                    'unqualified.',
                                                                                             'nonadjacent_arm_self_collision': True,
                                                                                             'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                             'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                                        'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/living_room/tower_building/dual_opposed/v1',
                                                                                            'manifest_sha256': 'fc9802b2920263bb07b7018770bc5c7887706881ddf8486935f486dd7d3b1cb8'},
                                                                         'release_schema': 'workroom_robot_configuration_release.v1',
                                                                         'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
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
                                                                         'qualification': {'native_preflight': False,
                                                                                           'agent_bench_interface': False,
                                                                                           'robot_manipulation': False,
                                                                                           'physics_accepted': False,
                                                                                           'workspace_ik': True,
                                                                                           'workspace_clearance': True,
                                                                                           'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/living_room_tower_building_dual_opposed.json',
                                                                                                                  'sha256': '03377b0b348417e9caca335244a36d50aa4e5b14667b8412c646681397b28597'},
                                                                                           'qualification_scope': 'Discrete '
                                                                                                                  'fingertip '
                                                                                                                  'approach '
                                                                                                                  'IK '
                                                                                                                  'and '
                                                                                                                  'original '
                                                                                                                  'shape '
                                                                                                                  'AABB '
                                                                                                                  'clearance; '
                                                                                                                  'continuous '
                                                                                                                  'motion '
                                                                                                                  'and '
                                                                                                                  'manipulation '
                                                                                                                  'remain '
                                                                                                                  'unqualified.',
                                                                                           'nonadjacent_arm_self_collision': True,
                                                                                           'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                           'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                                     'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                       'assembly_version': 'isaacsim_6_1_v2',
                                                                       'articulations': 1}},
 ('card_room', 'task0'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/task0/v5',
                                           'manifest_sha256': '3035adc3f58924cd460f6881495e1ed283925998f7d71693585614a80768af71'},
                          'single_arm_layout': {'robot_layout_id': 'card_room_task0_original_single_arm',
                                                'placements': {'right': {'base_position_m': [0.65, -0.6, 0.9],
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
                          'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/card_room/task0/dual_same_side/v1',
                                                                                   'manifest_sha256': 'e2a74c8653124001cca5c531ef2a36a8c455780823c33684183f56866fbc7183'},
                                                                'release_schema': 'workroom_robot_configuration_release.v1',
                                                                'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                       'assembly_version': 'isaacsim_6_1_v2',
                                                                                       'articulations': 2,
                                                                                       'active_sides': ['left',
                                                                                                        'right']},
                                                                'layout': {'robot_layout_id': 'card_room_dual_same_side_left_right_sharpa_v1',
                                                                           'placements': {'left': {'base_position_m': [0.35,
                                                                                                                       -0.66,
                                                                                                                       0.9],
                                                                                                   'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                            0.0,
                                                                                                                            0.0,
                                                                                                                            0.7071067811865475],
                                                                                                   'base_yaw_rad': 1.5707963267948966,
                                                                                                   'hand_chirality': 'left'},
                                                                                          'right': {'base_position_m': [0.95,
                                                                                                                        -0.66,
                                                                                                                        0.9],
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
                                                                'qualification': {'native_preflight': False,
                                                                                  'agent_bench_interface': False,
                                                                                  'robot_manipulation': False,
                                                                                  'physics_accepted': False,
                                                                                  'workspace_ik': True,
                                                                                  'workspace_clearance': True,
                                                                                  'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/card_room_task0_dual_same_side.json',
                                                                                                         'sha256': 'e6aa35f2330a219ce80001cc3b04b6c316a8d761f1714b43b497b2c5100b30bd'},
                                                                                  'qualification_scope': 'Discrete '
                                                                                                         'fingertip '
                                                                                                         'approach '
                                                                                                         'IK '
                                                                                                         'and '
                                                                                                         'original '
                                                                                                         'shape '
                                                                                                         'AABB '
                                                                                                         'clearance; '
                                                                                                         'continuous '
                                                                                                         'motion '
                                                                                                         'and '
                                                                                                         'manipulation '
                                                                                                         'remain '
                                                                                                         'unqualified.',
                                                                                  'nonadjacent_arm_self_collision': True,
                                                                                  'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                  'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}},
                                             'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261023/batch01/card_room/task0/dual_opposed/v1',
                                                                                 'manifest_sha256': '70d6d4ccecb428152a9080d18960561174cb0d5bea7a4897dae6758972bdcf42'},
                                                              'release_schema': 'workroom_robot_configuration_release.v1',
                                                              'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                              'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                     'assembly_version': 'isaacsim_6_1_v2',
                                                                                     'articulations': 2,
                                                                                     'active_sides': ['left',
                                                                                                      'right']},
                                                              'layout': {'robot_layout_id': 'card_room_dual_opposed_left_right_sharpa_v1',
                                                                         'placements': {'left': {'base_position_m': [0.65,
                                                                                                                     0.72,
                                                                                                                     0.9],
                                                                                                 'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                                          0.0,
                                                                                                                          0.0,
                                                                                                                          -0.7071067811865475],
                                                                                                 'base_yaw_rad': -1.5707963267948966,
                                                                                                 'hand_chirality': 'left'},
                                                                                        'right': {'base_position_m': [0.65,
                                                                                                                      -0.72,
                                                                                                                      0.9],
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
                                                              'qualification': {'native_preflight': False,
                                                                                'agent_bench_interface': False,
                                                                                'robot_manipulation': False,
                                                                                'physics_accepted': False,
                                                                                'workspace_ik': True,
                                                                                'workspace_clearance': True,
                                                                                'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_final03/card_room_task0_dual_opposed.json',
                                                                                                       'sha256': 'b974b6854edf4c5896ca189097e8d755c6d0a5a52cb24671bee01012a29b148c'},
                                                                                'qualification_scope': 'Discrete '
                                                                                                       'fingertip '
                                                                                                       'approach '
                                                                                                       'IK '
                                                                                                       'and '
                                                                                                       'original '
                                                                                                       'shape '
                                                                                                       'AABB '
                                                                                                       'clearance; '
                                                                                                       'continuous '
                                                                                                       'motion '
                                                                                                       'and '
                                                                                                       'manipulation '
                                                                                                       'remain '
                                                                                                       'unqualified.',
                                                                                'nonadjacent_arm_self_collision': True,
                                                                                'arm_self_collision_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/bimanual_options_20261023/geometry_self_collision01.json',
                                                                                                                'sha256': '56114f0207f608bacf36993ef3ef1b44c830d1faa2c7bb8ec56d2704b7212816'}}}},
                          'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 1}},
 ('laboratory', 'sample_sorting'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/sample_sorting/budget_v1',
                  'manifest_sha256': 'e808bd77c1c58f7d8da374c3fbe1a636ec63ca54ad98f3e84a1fb57c8e2982c1'},
 'single_arm_layout': {'robot_layout_id': 'laboratory_sample_sorting_original_single_arm',
                       'placements': {'right': {'base_position_m': [0.7, 1.2875, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/sample_sorting/dual_same_side/v1',
                                                          'manifest_sha256': '2815d6f3f2c2581272f447718d79dc78c5be389cdcb0cef2ec103cfe4ed1e60d'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'laboratory_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [1.0, 1.1, 0.9],
                                                                          'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   -0.7071067811865475],
                                                                          'base_yaw_rad': -1.5707963267948966,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [0.4, 1.1, 0.9],
                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    -0.7071067811865475],
                                                                           'base_yaw_rad': -1.5707963267948966,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'main laboratory island; rear counter '
                                                                       'remains a separate target surface',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False,
                                                         'qualification_scope': 'Frozen budget termination '
                                                                                'repair; development '
                                                                                'validation outcomes remain '
                                                                                'outside task inputs and '
                                                                                'admission.'}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/sample_sorting/dual_opposed/v1',
                                                        'manifest_sha256': '13ab30ff7c4a3e60a6cabb042743fa16a1dc6277f5a57df30ecd7e43ffd7877d'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'laboratory_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.7, 1.1, 0.9],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.7,
                                                                                             -0.12,
                                                                                             0.925],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'Main island: north floor pedestal, '
                                                                     'south tabletop bolted plate on the '
                                                                     '0.90 m worktop; rear counter is a '
                                                                     'separate target surface.',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False,
                                                       'qualification_scope': 'Frozen budget termination '
                                                                              'repair; development '
                                                                              'validation outcomes remain '
                                                                              'outside task inputs and '
                                                                              'admission.'}}},
 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 1}},
 ('laboratory', 'vial_racking'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/vial_racking/budget_v1',
                  'manifest_sha256': '54c2b9fb037ea120880019d280e50472fd90f8730c07442bc2e0cce50fa963e4'},
 'single_arm_layout': {'robot_layout_id': 'laboratory_vial_racking_original_single_arm',
                       'placements': {'right': {'base_position_m': [0.7, 1.2875, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/vial_racking/dual_same_side/v1',
                                                          'manifest_sha256': 'f7467c570f64fc9d0d3cfbd0c57fbaaa4c28351d5ffcf2a4631c6285fece6e43'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'laboratory_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [1.0, 1.1, 0.9],
                                                                          'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   -0.7071067811865475],
                                                                          'base_yaw_rad': -1.5707963267948966,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [0.4, 1.1, 0.9],
                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    -0.7071067811865475],
                                                                           'base_yaw_rad': -1.5707963267948966,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'main laboratory island; rear counter '
                                                                       'remains a separate target surface',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False,
                                                         'qualification_scope': 'Frozen budget termination '
                                                                                'repair; development '
                                                                                'validation outcomes remain '
                                                                                'outside task inputs and '
                                                                                'admission.'}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/vial_racking/dual_opposed/v1',
                                                        'manifest_sha256': '684868591390ef9d4096e6ec5597895ae08c0b0edd3322c7dd3d05f2fa220b01'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'laboratory_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.7, 1.1, 0.9],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.7,
                                                                                             -0.12,
                                                                                             0.925],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'main laboratory island; rear counter '
                                                                     'remains a separate target surface',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False,
                                                       'qualification_scope': 'Frozen budget termination '
                                                                              'repair; development '
                                                                              'validation outcomes remain '
                                                                              'outside task inputs and '
                                                                              'admission.'}}}},
 ('laboratory', 'cap_placement'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/cap_placement/budget_v1',
                  'manifest_sha256': 'e39534d6de813b520ad59d6d8eee01749fa3c695ba637fb1341554c70f0d57d0'},
 'single_arm_layout': {'robot_layout_id': 'laboratory_cap_placement_original_single_arm',
                       'placements': {'right': {'base_position_m': [0.7, 1.2875, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/cap_placement/dual_same_side/v1',
                                                          'manifest_sha256': '2bf9ee48afd6773fe715a88a6530bc247d9c3232b754eff33cb38888ca8fbcc3'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'laboratory_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [1.0, 1.1, 0.9],
                                                                          'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   -0.7071067811865475],
                                                                          'base_yaw_rad': -1.5707963267948966,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [0.4, 1.1, 0.9],
                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    -0.7071067811865475],
                                                                           'base_yaw_rad': -1.5707963267948966,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'main laboratory island; rear counter '
                                                                       'remains a separate target surface',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False,
                                                         'qualification_scope': 'Frozen budget termination '
                                                                                'repair; development '
                                                                                'validation outcomes remain '
                                                                                'outside task inputs and '
                                                                                'admission.'}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/cap_placement/dual_opposed/v1',
                                                        'manifest_sha256': 'e890fd66aa191501fe81647b694a97976e9cfbb6cfad3d64cbaef784b9671616'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'laboratory_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.7, 1.1, 0.9],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.7,
                                                                                             -0.12,
                                                                                             0.925],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'Main island: north floor pedestal, '
                                                                     'south tabletop bolted plate on the '
                                                                     '0.90 m worktop; rear counter is a '
                                                                     'separate target surface.',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False,
                                                       'qualification_scope': 'Frozen budget termination '
                                                                              'repair; development '
                                                                              'validation outcomes remain '
                                                                              'outside task inputs and '
                                                                              'admission.'}}},
 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 1}},
 ('laboratory', 'solid_weighing'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/solid_weighing/budget_v1',
                  'manifest_sha256': '4f65c9bbeb5db831c6a174d0ed958dadfeac702928ae5c12e8dd5a5d96e27eae'},
 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 1},
 'single_arm_layout': {'robot_layout_id': 'laboratory_solid_weighing_original_single_arm',
                       'placements': {'right': {'base_position_m': [0.7, 1.2875, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/solid_weighing/dual_same_side/v1',
                                                          'manifest_sha256': 'f702fde005df2def32131a547f74b0b3c5916b9ab23750cf3131986a347b8f57'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'laboratory_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [1.0, 1.1, 0.9],
                                                                          'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   -0.7071067811865475],
                                                                          'base_yaw_rad': -1.5707963267948966,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [0.4, 1.1, 0.9],
                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    -0.7071067811865475],
                                                                           'base_yaw_rad': -1.5707963267948966,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'main laboratory island; rear counter '
                                                                       'remains a separate target surface',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False,
                                                         'qualification_scope': 'Frozen budget termination '
                                                                                'repair; development '
                                                                                'validation outcomes remain '
                                                                                'outside task inputs and '
                                                                                'admission.'}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/solid_weighing/dual_opposed/v1',
                                                        'manifest_sha256': '2a11cbeca201179a64235d34066a1524aa71ac6981ae74b35734f535840d59ce'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'laboratory_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.7, 1.1, 0.9],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.7,
                                                                                             -0.12,
                                                                                             0.925],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'main laboratory island; rear counter '
                                                                     'remains a separate target surface',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False,
                                                       'qualification_scope': 'Frozen budget termination '
                                                                              'repair; development '
                                                                              'validation outcomes remain '
                                                                              'outside task inputs and '
                                                                              'admission.'}}}},
 ('laboratory', 'pipette_transfer'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/pipette_transfer/budget_v1',
                  'manifest_sha256': 'fbdb937c8a1b6a7771e266644e571eac2492c9c8bd3135ffbdc0b8c5dd614091'},
 'single_arm_layout': {'robot_layout_id': 'laboratory_pipette_transfer_original_single_arm',
                       'placements': {'right': {'base_position_m': [0.7, 1.2875, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/pipette_transfer/dual_same_side/v1',
                                                          'manifest_sha256': '52c1b07880a780014d2eb03609c222a708526a3a4fada5a79b18e4c827068b7f'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'laboratory_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [1.0, 1.1, 0.9],
                                                                          'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   -0.7071067811865475],
                                                                          'base_yaw_rad': -1.5707963267948966,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [0.4, 1.1, 0.9],
                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    -0.7071067811865475],
                                                                           'base_yaw_rad': -1.5707963267948966,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'main laboratory island; rear counter '
                                                                       'remains a separate target surface',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False,
                                                         'qualification_scope': 'Frozen budget termination '
                                                                                'repair; development '
                                                                                'validation outcomes remain '
                                                                                'outside task inputs and '
                                                                                'admission.'}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/pipette_transfer/dual_opposed/v1',
                                                        'manifest_sha256': '8670f1c650b2282a360bc01bbe0e1fc11fbbb7dd35784e11836c03bbb63e2eb5'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'laboratory_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.7, 1.1, 0.9],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.7,
                                                                                             -0.12,
                                                                                             0.925],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'main laboratory island; rear counter '
                                                                     'remains a separate target surface',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False,
                                                       'qualification_scope': 'Frozen budget termination '
                                                                              'repair; development '
                                                                              'validation outcomes remain '
                                                                              'outside task inputs and '
                                                                              'admission.'}}}}}


# Interaction expansion: immutable installations; evidence is attached separately.
RELEASES.update({('study', 'reading_stand'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/study/reading_stand/v1',
                                               'manifest_sha256': '7179879f41144ecff96e54a75d5f15e7f3c0bce68b91c29fa2e47e351d4a4cd8'},
                              'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                'assembly_version': 'isaacsim_6_1_v2',
                                                                'articulations': 1},
                              'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                                                    'placements': {'right': {'base_position_m': [-0.23,
                                                                                                 0.0,
                                                                                                 0.78],
                                                                             'base_quaternion_wxyz': [1.0,
                                                                                                      0.0,
                                                                                                      0.0,
                                                                                                      0.0],
                                                                             'base_yaw_rad': 0.0,
                                                                             'hand_chirality': 'right'}},
                                                    'reference_surface': 'original fixed task installation',
                                                    'frame': 'world',
                                                    'units': 'metres',
                                                    'control_scope': 'one_agent_29_dof'},
                              'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261025/interaction_expansion01/study/reading_stand/dual_same_side/v1',
                                                                                       'manifest_sha256': '5c6f1bea0358e5fd0eb77477541d8e5a3c0bb30abd0fd4fb9d4848154523848a'},
                                                                    'release_schema': 'workroom_robot_configuration_release.v1',
                                                                    'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                    'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                           'assembly_version': 'isaacsim_6_1_v2',
                                                                                           'articulations': 2,
                                                                                           'active_sides': ['left',
                                                                                                            'right']},
                                                                    'layout': {'robot_layout_id': 'study_dual_same_side_left_right_sharpa_v1',
                                                                               'placements': {'left': {'base_position_m': [-0.23,
                                                                                                                           0.3,
                                                                                                                           0.78],
                                                                                                       'base_quaternion_wxyz': [1.0,
                                                                                                                                0.0,
                                                                                                                                0.0,
                                                                                                                                0.0],
                                                                                                       'base_yaw_rad': 0.0,
                                                                                                       'hand_chirality': 'left'},
                                                                                              'right': {'base_position_m': [-0.23,
                                                                                                                            -0.3,
                                                                                                                            0.78],
                                                                                                        'base_quaternion_wxyz': [1.0,
                                                                                                                                 0.0,
                                                                                                                                 0.0,
                                                                                                                                 0.0],
                                                                                                        'base_yaw_rad': 0.0,
                                                                                                        'hand_chirality': 'right'}},
                                                                               'reference_surface': 'fixed '
                                                                                                    'room task '
                                                                                                    'table',
                                                                               'frame': 'world',
                                                                               'units': 'metres',
                                                                               'control_scope': 'one_agent_58_dof'},
                                                                    'qualification': {'native_preflight': True,
                                                                                      'agent_bench_interface': True,
                                                                                      'robot_manipulation': False,
                                                                                      'physics_accepted': False,
                                                                                      'discrete_workspace': True,
                                                                                      'nonadjacent_arm_self_collision': True,
                                                                                      'object_interactions': True,
                                                                                      'configuration_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/configuration_qualification.json',
                                                                                                                 'sha256': '670335af5e31afb4ee76ffe4988b0ee397c695b6a59e2abfe9d77df40901d08e'},
                                                                                      'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/workspace_reading_stand_dual_same_sidefinal02.json',
                                                                                                             'sha256': 'e3f8f59aa660ba665a44a4d96f831685faee2e783c6fd256c85b83cd9fb09cce'},
                                                                                      'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/http_reading_stand_dual_same_side01/qualification.json',
                                                                                                                    'sha256': '850f4f373fd09104d99dcb35f0b3634c1ebe2448eed85428f2bec54894aa24b2'},
                                                                                      'scope': 'Exact selected '
                                                                                               'release: real '
                                                                                               'HTTP/Agent and '
                                                                                               'native-recorded '
                                                                                               'discrete shape '
                                                                                               'checks. Object '
                                                                                               'mechanics are '
                                                                                               'separately '
                                                                                               'bound; no '
                                                                                               'complete robot '
                                                                                               'manipulation '
                                                                                               'or continuous '
                                                                                               'path '
                                                                                               'qualification.'}},
                                                 'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261025/interaction_expansion01/study/reading_stand/dual_opposed/v1',
                                                                                     'manifest_sha256': 'f9cc0f1c9853f63970c6c9c0a10e28566723b5b3c006c9775a029b5e326a33de'},
                                                                  'release_schema': 'workroom_robot_configuration_release.v1',
                                                                  'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                                                  'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                                         'articulations': 2,
                                                                                         'active_sides': ['left',
                                                                                                          'right']},
                                                                  'layout': {'robot_layout_id': 'study_dual_opposed_left_right_sharpa_v1',
                                                                             'placements': {'left': {'base_position_m': [1.35,
                                                                                                                         0.0,
                                                                                                                         0.78],
                                                                                                     'base_quaternion_wxyz': [6.123233995736766e-17,
                                                                                                                              0.0,
                                                                                                                              0.0,
                                                                                                                              1.0],
                                                                                                     'base_yaw_rad': 3.141592653589793,
                                                                                                     'hand_chirality': 'left'},
                                                                                            'right': {'base_position_m': [-0.23,
                                                                                                                          0.0,
                                                                                                                          0.78],
                                                                                                      'base_quaternion_wxyz': [1.0,
                                                                                                                               0.0,
                                                                                                                               0.0,
                                                                                                                               0.0],
                                                                                                      'base_yaw_rad': 0.0,
                                                                                                      'hand_chirality': 'right'}},
                                                                             'reference_surface': 'fixed room '
                                                                                                  'task table',
                                                                             'frame': 'world',
                                                                             'units': 'metres',
                                                                             'control_scope': 'one_agent_58_dof'},
                                                                  'qualification': {'native_preflight': True,
                                                                                    'agent_bench_interface': True,
                                                                                    'robot_manipulation': False,
                                                                                    'physics_accepted': False,
                                                                                    'discrete_workspace': True,
                                                                                    'nonadjacent_arm_self_collision': True,
                                                                                    'object_interactions': True,
                                                                                    'configuration_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/configuration_qualification.json',
                                                                                                               'sha256': '670335af5e31afb4ee76ffe4988b0ee397c695b6a59e2abfe9d77df40901d08e'},
                                                                                    'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/workspace_reading_stand_dual_opposedfinal02.json',
                                                                                                           'sha256': '61d959759602a0e87908e3d200fefa736589ed001efe3e7a4ea922e8f7348d0a'},
                                                                                    'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/http_reading_stand_dual_opposed01/qualification.json',
                                                                                                                  'sha256': 'eefc5001104c1a18905a6189f35f7e94b1e050cd7c122a6d12e259980a8a12a0'},
                                                                                    'scope': 'Exact selected '
                                                                                             'release: real '
                                                                                             'HTTP/Agent and '
                                                                                             'native-recorded '
                                                                                             'discrete shape '
                                                                                             'checks. Object '
                                                                                             'mechanics are '
                                                                                             'separately '
                                                                                             'bound; no '
                                                                                             'complete robot '
                                                                                             'manipulation or '
                                                                                             'continuous path '
                                                                                             'qualification.'}}}},
 ('living_room', 'storage_box'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/storage_box/v3',
                                                   'manifest_sha256': 'c1c5c032c9d5dd34b9ef78506f7ca7fd534bfca73cc6ab88285a4c49fe39c163'},
                                  'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                    'assembly_version': 'isaacsim_6_1_v2',
                                                                    'articulations': 1},
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
                                  'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261025/interaction_expansion02/living_room/storage_box/dual_same_side/v1',
                                                                                           'manifest_sha256': '91493b3fc775360870dba483a2c92b96b34f7cd04dcab0dec4ef6f162b92d10e'},
                                                                        'release_schema': 'workroom_robot_configuration_release.v1',
                                                                        'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
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
                                                                        'qualification': {'native_preflight': True,
                                                                                          'agent_bench_interface': True,
                                                                                          'robot_manipulation': False,
                                                                                          'physics_accepted': False,
                                                                                          'discrete_workspace': True,
                                                                                          'nonadjacent_arm_self_collision': True,
                                                                                          'object_interactions': True,
                                                                                          'configuration_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/configuration_qualification.json',
                                                                                                                     'sha256': '670335af5e31afb4ee76ffe4988b0ee397c695b6a59e2abfe9d77df40901d08e'},
                                                                                          'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/workspace_storage_box_dual_same_sidefinal02.json',
                                                                                                                 'sha256': 'b5384723d71ec8d74675203a7406299b8f364a6e71616ddce32891ae71e78ac4'},
                                                                                          'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/http_storage_box_dual_same_side02/qualification.json',
                                                                                                                        'sha256': '1d0ae31263a292a3296942476c2628598ddbcc75e1cd6d4e5576e10e45269c17'},
                                                                                          'scope': 'Exact '
                                                                                                   'selected '
                                                                                                   'release: '
                                                                                                   'real '
                                                                                                   'HTTP/Agent '
                                                                                                   'and '
                                                                                                   'native-recorded '
                                                                                                   'discrete '
                                                                                                   'shape '
                                                                                                   'checks. '
                                                                                                   'Object '
                                                                                                   'mechanics '
                                                                                                   'are '
                                                                                                   'separately '
                                                                                                   'bound; no '
                                                                                                   'complete '
                                                                                                   'robot '
                                                                                                   'manipulation '
                                                                                                   'or '
                                                                                                   'continuous '
                                                                                                   'path '
                                                                                                   'qualification.'}},
                                                     'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261025/interaction_expansion02/living_room/storage_box/dual_opposed/v1',
                                                                                         'manifest_sha256': '176d9023bafab0f729dff18da83877c2660dbce7426495565409f90508c1da44'},
                                                                      'release_schema': 'workroom_robot_configuration_release.v1',
                                                                      'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
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
                                                                      'qualification': {'native_preflight': True,
                                                                                        'agent_bench_interface': True,
                                                                                        'robot_manipulation': False,
                                                                                        'physics_accepted': False,
                                                                                        'discrete_workspace': True,
                                                                                        'nonadjacent_arm_self_collision': True,
                                                                                        'object_interactions': True,
                                                                                        'configuration_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/configuration_qualification.json',
                                                                                                                   'sha256': '670335af5e31afb4ee76ffe4988b0ee397c695b6a59e2abfe9d77df40901d08e'},
                                                                                        'workspace_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/workspace_storage_box_dual_opposedfinal02.json',
                                                                                                               'sha256': '404da647e0b89a3b33b309807347f825df62956fd5926ff3d468bd5cbdc8c095'},
                                                                                        'native_interface_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/http_storage_box_dual_opposed02/qualification.json',
                                                                                                                      'sha256': '7e49b5a8cecf43538aa050f59c27f74f63ef0d7b73184c895746c8a5ef6ade75'},
                                                                                        'scope': 'Exact '
                                                                                                 'selected '
                                                                                                 'release: '
                                                                                                 'real '
                                                                                                 'HTTP/Agent '
                                                                                                 'and '
                                                                                                 'native-recorded '
                                                                                                 'discrete '
                                                                                                 'shape '
                                                                                                 'checks. '
                                                                                                 'Object '
                                                                                                 'mechanics '
                                                                                                 'are '
                                                                                                 'separately '
                                                                                                 'bound; no '
                                                                                                 'complete '
                                                                                                 'robot '
                                                                                                 'manipulation '
                                                                                                 'or '
                                                                                                 'continuous '
                                                                                                 'path '
                                                                                                 'qualification.'}}}},
 ('laboratory', 'tube_clamp_transfer'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/laboratory/tube_clamp_transfer/budget_v1',
                  'manifest_sha256': '8f5fbf3adbea2c48db1e47e8cfd8078159160f630b164faa187ceead77f83293'},
 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 1},
 'single_arm_layout': {'robot_layout_id': 'laboratory_tube_clamp_transfer_original_single_arm',
                       'placements': {'right': {'base_position_m': [0.7, 1.2875, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/tube_clamp_transfer/dual_same_side/v1',
                                                          'manifest_sha256': 'dd71e9af88aebafcd3a8bf7b8876c0ed9f45adc114cf2c26a54ea9d73c26df53'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'laboratory_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [1.0, 1.1, 0.9],
                                                                          'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   -0.7071067811865475],
                                                                          'base_yaw_rad': -1.5707963267948966,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [0.4, 1.1, 0.9],
                                                                           'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    -0.7071067811865475],
                                                                           'base_yaw_rad': -1.5707963267948966,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'main laboratory island; rear counter '
                                                                       'remains a separate target surface',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False,
                                                         'qualification_scope': 'Frozen budget termination '
                                                                                'repair; development '
                                                                                'validation outcomes remain '
                                                                                'outside task inputs and '
                                                                                'admission.'}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/laboratory_budget_v1/tube_clamp_transfer/dual_opposed/v1',
                                                        'manifest_sha256': '71b5397aa471cb58218bf303783f16e87e9d5379f5e546b0ece3d4478c9491b7'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'laboratory_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.7, 1.1, 0.9],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.7,
                                                                                             -0.12,
                                                                                             0.925],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'main laboratory island; rear counter '
                                                                     'remains a separate target surface',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False,
                                                       'qualification_scope': 'Frozen budget termination '
                                                                              'repair; development '
                                                                              'validation outcomes remain '
                                                                              'outside task inputs and '
                                                                              'admission.'}}}}})


# Larger idler grip inputs; independent development results are not catalog gates.
RELEASES[('mechanical_assembly', 'gear_train_assembly')] = {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/gear_train_assembly/idler_grip_v2',
                  'manifest_sha256': '1445f61c09816c139cf700ff87dae82c6b1ba3745e6a33c6eeb2733770caace8'},
 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 1},
 'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                       'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/gear_grip02/mechanical_assembly/gear_train_assembly/dual_same_side/v1',
                                                          'manifest_sha256': '731c66bd2137225b7d358e86820105a924e5e0a47c3b73c22669ea66d123f3b3'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [-0.1,
                                                                                              0.3,
                                                                                              0.78],
                                                                          'base_quaternion_wxyz': [1.0,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   0.0],
                                                                          'base_yaw_rad': 0.0,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [-0.1,
                                                                                               -0.3,
                                                                                               0.78],
                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    0.0],
                                                                           'base_yaw_rad': 0.0,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'fixed room task table',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/gear_grip02/mechanical_assembly/gear_train_assembly/dual_opposed/v1',
                                                        'manifest_sha256': 'a2f60d954716b0e18a2e43d77144d7f65ca56ad14fe2d85672b4db3e3ea3cd8f'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.42,
                                                                                            0.76,
                                                                                            0.78],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.42,
                                                                                             -0.76,
                                                                                             0.78],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'fixed room task table',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False}}}}


# Visible remote buttons and native ten-second automatic television countdown.
RELEASES.update({('living_room', 'remote_operation'): {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/remote_operation/v8',
                                                        'manifest_sha256': 'f523e34e292aa63ff0cd0539cdd8a1f0469a8a362278bd890bc9b5a82590481b'},
                                       'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                                                         'assembly_version': 'isaacsim_6_1_v2',
                                                                         'articulations': 1},
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
                                                             'reference_surface': 'original fixed task '
                                                                                  'installation',
                                                             'frame': 'world',
                                                             'units': 'metres',
                                                             'control_scope': 'one_agent_29_dof'},
                                       'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/living_remote_countdown01/living_room/remote_operation/dual_same_side/v1',
                                                                                                'manifest_sha256': '22f3c9f0a63a3819b7ca74c5dedb5ba2c25eef25d66aab5fb18a87ffac4c7add'},
                                                                             'release_schema': 'workroom_robot_configuration_release.v1',
                                                                             'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
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
                                                                             'qualification': {'native_preflight': False,
                                                                                               'agent_bench_interface': False,
                                                                                               'robot_manipulation': False,
                                                                                               'physics_accepted': False}},
                                                          'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261026/living_remote_countdown01/living_room/remote_operation/dual_opposed/v1',
                                                                                              'manifest_sha256': '4b53548d5b2aa47b0495413b106a45ec9c2878e182f2a4e7de3c36ad50d23292'},
                                                                           'release_schema': 'workroom_robot_configuration_release.v1',
                                                                           'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
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
                                                                           'qualification': {'native_preflight': False,
                                                                                             'agent_bench_interface': False,
                                                                                             'robot_manipulation': False,
                                                                                             'physics_accepted': False}}}}})


# Belt supply loop 1 mm along -y, clear of the idler support shoulder (usability_v5);
# the 20261023 belt row above is the superseded record. Native evidence is not inherited.
RELEASES[('mechanical_assembly', 'nist_belt_drive_assembly')] = {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261107b/batch01/nist_belt_drive_assembly/desktop_v1',
                  'manifest_sha256': 'ec43adbeec31682e74923e0c5220e00c33bfaae8a7411abadea8359ab34b0fde'},
 'single_arm_layout': {'robot_layout_id': 'mechanical_assembly_nist_belt_drive_assembly_original_single_arm',
                       'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261107b/dual_batch01/mechanical_assembly/nist_belt_drive_assembly/dual_same_side/v1',
                                                          'manifest_sha256': 'eb4a9f2ce2f920b6be8aff622f54884b55f27cea17bfde0a160b7f7e8cb3a3e2'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [-0.1,
                                                                                              0.3,
                                                                                              0.78],
                                                                          'base_quaternion_wxyz': [1.0,
                                                                                                   0.0,
                                                                                                   0.0,
                                                                                                   0.0],
                                                                          'base_yaw_rad': 0.0,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [-0.1,
                                                                                               -0.3,
                                                                                               0.78],
                                                                           'base_quaternion_wxyz': [1.0,
                                                                                                    0.0,
                                                                                                    0.0,
                                                                                                    0.0],
                                                                           'base_yaw_rad': 0.0,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'fixed room task table',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False,
                                                         'workspace_discrete': True,
                                                         'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/beltbase45_evidence02/workspace01/nist_belt_drive_assembly_dual_same_side.json',
                                                                                     'sha256': 'e675cd57c010698c381f5c5704f08789191f6efd1a1e61737101f48cc60d4a56',
                                                                                     'passed': True,
                                                                                     'scope': 'Discrete '
                                                                                              'actual '
                                                                                              'fingertip '
                                                                                              'approach '
                                                                                              'and '
                                                                                              'original '
                                                                                              'USD '
                                                                                              'collision '
                                                                                              'bounds; '
                                                                                              'no '
                                                                                              'continuous '
                                                                                              'path '
                                                                                              'or '
                                                                                              'manipulation '
                                                                                              'qualification.'},
                                                         'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/beltbase45_evidence02/workspace_binding01.json',
                                                                                       'sha256': 'e3928b9b3369c37cc0022eca242da9b09bfa2a7e8787d1d6ad5afa119493285f'}}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/nist_usability_20261107b/dual_batch01/mechanical_assembly/nist_belt_drive_assembly/dual_opposed/v1',
                                                        'manifest_sha256': '264988b622e9c480250be511c53bcf88f1fb55f8e47c251f31b17b5ba5b4ca5a'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.42,
                                                                                            0.76,
                                                                                            0.78],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.42,
                                                                                             -0.76,
                                                                                             0.78],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'fixed room task table',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False,
                                                       'workspace_discrete': True,
                                                       'scene_geometry_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/beltbase45_evidence02/workspace01/nist_belt_drive_assembly_dual_opposed.json',
                                                                                   'sha256': '07a3b8b85f3d3317cda4414e7c809f83559921afef1c2fcf10276fb0b8a99849',
                                                                                   'passed': True,
                                                                                   'scope': 'Discrete '
                                                                                            'actual '
                                                                                            'fingertip '
                                                                                            'approach '
                                                                                            'and '
                                                                                            'original '
                                                                                            'USD '
                                                                                            'collision '
                                                                                            'bounds; '
                                                                                            'no '
                                                                                            'continuous '
                                                                                            'path '
                                                                                            'or '
                                                                                            'manipulation '
                                                                                            'qualification.'},
                                                       'workspace_release_binding': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/parameterized_seeds_20261105/beltbase45_evidence02/workspace_binding01.json',
                                                                                     'sha256': 'e3928b9b3369c37cc0022eca242da9b09bfa2a7e8787d1d6ad5afa119493285f'}}}},
 'single_arm_robot_requirements': {'arm': 'fr3',
                                   'hand_id': 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 1,
                                   'layout_id': 'mechanical_single_arm_v1',
                                   'base_position_m': [-0.23, 0, 0.78],
                                   'active_sides': ['right']}}


# Open-end wrench 50 mm closer to the robot (wrench_reach_v1); the idler grip row above is the
# superseded record. Native and manipulation evidence is not inherited.
RELEASES[('mechanical_assembly', 'gear_train_assembly')] = {'base_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/gear_train_assembly/wrench_reach_v1',
                  'manifest_sha256': '1768eea9144936eae13f408bead4fb881939f6299589b87e50d4f5870e2c35a7'},
 'single_arm_robot_requirements': {'hand_id': 'sharpa_w01_right',
                                   'assembly_version': 'isaacsim_6_1_v2',
                                   'articulations': 1},
 'single_arm_layout': {'robot_layout_id': 'mechanical_single_arm_v1',
                       'placements': {'right': {'base_position_m': [-0.23, 0.0, 0.78],
                                                'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                'base_yaw_rad': 0.0,
                                                'hand_chirality': 'right'}},
                       'reference_surface': 'original fixed task installation',
                       'frame': 'world',
                       'units': 'metres',
                       'control_scope': 'one_agent_29_dof'},
 'configurations': {'dual_same_side': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261107/gear_wrench01/mechanical_assembly/gear_train_assembly/dual_same_side/v1',
                                                          'manifest_sha256': '853ec4e3b878eaa33e990a52401b417997ce5b2a64752f561d20dd6a0897cf1a'},
                                       'release_schema': 'workroom_robot_configuration_release.v1',
                                       'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                       'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                              'assembly_version': 'isaacsim_6_1_v2',
                                                              'articulations': 2,
                                                              'active_sides': ['left', 'right']},
                                       'layout': {'robot_layout_id': 'mechanical_assembly_dual_same_side_left_right_sharpa_v1',
                                                  'placements': {'left': {'base_position_m': [-0.1, 0.3, 0.78],
                                                                          'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                                          'base_yaw_rad': 0.0,
                                                                          'hand_chirality': 'left'},
                                                                 'right': {'base_position_m': [-0.1, -0.3, 0.78],
                                                                           'base_quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                                                                           'base_yaw_rad': 0.0,
                                                                           'hand_chirality': 'right'}},
                                                  'reference_surface': 'fixed room task table',
                                                  'frame': 'world',
                                                  'units': 'metres',
                                                  'control_scope': 'one_agent_58_dof'},
                                       'qualification': {'native_preflight': False,
                                                         'agent_bench_interface': False,
                                                         'robot_manipulation': False,
                                                         'physics_accepted': False}},
                    'dual_opposed': {'source_release': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/robot_configurations_20261107/gear_wrench01/mechanical_assembly/gear_train_assembly/dual_opposed/v1',
                                                        'manifest_sha256': '7cc132b608abd452981c0e13caff973b1fbc143580d5f5e33bf4447b346a7c9d'},
                                     'release_schema': 'workroom_robot_configuration_release.v1',
                                     'release_schema_version': 'isaacsim-workroom-robot-configuration-release-v1',
                                     'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01',
                                                            'assembly_version': 'isaacsim_6_1_v2',
                                                            'articulations': 2,
                                                            'active_sides': ['left', 'right']},
                                     'layout': {'robot_layout_id': 'mechanical_assembly_dual_opposed_left_right_sharpa_v1',
                                                'placements': {'left': {'base_position_m': [0.42, 0.76, 0.78],
                                                                        'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                 0.0,
                                                                                                 0.0,
                                                                                                 -0.7071067811865475],
                                                                        'base_yaw_rad': -1.5707963267948966,
                                                                        'hand_chirality': 'left'},
                                                               'right': {'base_position_m': [0.42, -0.76, 0.78],
                                                                         'base_quaternion_wxyz': [0.7071067811865476,
                                                                                                  0.0,
                                                                                                  0.0,
                                                                                                  0.7071067811865475],
                                                                         'base_yaw_rad': 1.5707963267948966,
                                                                         'hand_chirality': 'right'}},
                                                'reference_surface': 'fixed room task table',
                                                'frame': 'world',
                                                'units': 'metres',
                                                'control_scope': 'one_agent_58_dof'},
                                     'qualification': {'native_preflight': False,
                                                       'agent_bench_interface': False,
                                                       'robot_manipulation': False,
                                                       'physics_accepted': False}}}}
