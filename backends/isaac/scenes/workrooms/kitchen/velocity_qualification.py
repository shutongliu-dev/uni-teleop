"""Static evidence for the two repaired kitchen releases; catalog reads no files.

Keep physics source14, public HTTP source15 and read-only geometry source17
separate. These metadata do not change immutable preparation/acceptance flags.
"""
from copy import deepcopy

from bench.paths import DATA_ROOT

INDEX = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/qualification.json',
 'sha256': 'f34c065ed9d1c06270cc2b1f408e1f41606c9c4f2ca7516a6cd93c8b156fe335'}
SOURCES = {'object_dynamics': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/source14',
                     'manifest_sha256': '2f2b2c2034ea997eea0a53533e5f96bdde3ac99a77d9438bb96849c3c34d11e7',
                     'commit': '2223b89863a1f9d0a157b842e408dc918fc37298',
                     'seal': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/source14_seal.json',
                              'sha256': '65d6f6a04120bbf97d7dfa85ba19ec6ab313a97b77fd74a51fbc3a4ff9330ef0'}},
 'agent_bench_http': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/source15',
                      'manifest_sha256': '1e2a3f48765555d14b47e043f0e52ba74d3a7cf1c21a1de725ed8aa9ecd5c151',
                      'commit': 'c5af3de3c9f9018a87068e1bffea70203d8202b8',
                      'seal': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/source15_seal.json',
                               'sha256': '2840f72a6dca8ef70a7dcdbf5c6d81b32330f4d7e2b5a74afa2d9e589d3dffe5'}},
 'native_geometry': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/source17',
                     'manifest_sha256': '2b07a2b03979a91aa8c90b7bb1efbaada33f538f13e6759881822ca99582b745',
                     'commit': '4a712cbc1b038e92a10e65d22990cd67aeea5e7a',
                     'seal': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/source17_seal.json',
                              'sha256': '25f013dd2af2dce5c0ac51d6bca15f1d6f45931a1ed7377d15ee5b3a8414817e'}}}
REPORTS = {'utensil_sorting': {'canonical': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/canonical_hybrid01/utensil_sorting/qualification.json',
                                   'sha256': 'fe9967c59bc892db0f623d82ccb7eeff5f0bb108607cefc40773e5a0e7817ae2'},
                     'goal': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/goal_hybrid01/utensil_sorting/qualification.json',
                              'sha256': '13a3cf3a1fb0b78c09075f63461d6c97f65ad0cb66a6ce5ee5b3775866db6756'},
                     'source_review': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/canonical_hybrid01_review.json',
                                       'sha256': '8a7e3a1cb91c8e8d6ec4d6458ed9384abc5c21a965ad4e0c77b3d8c0b06b2d55'},
                     'goal_review': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/goal_hybrid01_review.json',
                                     'sha256': 'ff4dca0e951e6db4675d7db560e3b894e1a6ef98f9aff365f22df36f5bba1c80'},
                     'http': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/http_hybrid01/utensil_sorting/qualification.json',
                              'sha256': '2413807f1c4ed473acfee8568e756035fab0b1085bda0eeb28d840f3709c56a8'},
                     'http_review': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/http_hybrid01/utensil_sorting_review.json',
                                     'sha256': '56593e354ff7f44920edf8d729347ee5c4484adb8023a504b5016631733e9677'},
                     'geometry': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/research/velocity_independent01/hybrid_geometry_report.json',
                                  'sha256': 'e17af08523125b3e92bef5a2513f7f8981aa61d14cfaadad18df9617ce95c30a'},
                     'geometry_native_attempt': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/hybrid_geometry02/utensil_sorting/qualification.json',
                                                 'sha256': 'ce7f39e39a528facfe51d2c9ce3da7a911be0553b1712b28b84f007ae74763b7'},
                     'workspace': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/root/workspace_hybrid01/report.json',
                                   'sha256': 'f4520d1fbee01a181162a198d8d9ee8c837ec2f7099efc31ac912c5889625aa3'},
                     'workspace_counts': {'sample_count': 24,
                                          'ik_solved_count': 24,
                                          'fixed_ready_hand_clearance_passed_count': 24,
                                          'all_clearances_passed': True}},
 'table_clearing': {'canonical': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/canonical_hybrid02/table_clearing/qualification.json',
                                  'sha256': '4d5cafde671a2942c100eb0edfb2333fb1db969d842edbdbbf92f65ce7bce285'},
                    'goal': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/goal_hybrid01/table_clearing/qualification.json',
                             'sha256': 'f511703c9f0a6f5f000275a778574c4aaa7c17427c0ecc575e0bf3ebb3bc94d7'},
                    'source_review': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/canonical_hybrid02_review.json',
                                      'sha256': 'c9febd0ced59df0cf08ac51e5309178c444399dfca0349f109b91360be8b4469'},
                    'goal_review': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/goal_hybrid01_review.json',
                                    'sha256': 'ff4dca0e951e6db4675d7db560e3b894e1a6ef98f9aff365f22df36f5bba1c80'},
                    'http': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/http_hybrid02/table_clearing/qualification.json',
                             'sha256': '8d7c8ea5fac566e51f8e66740e90e1bc5740b163656c6a166c5c10ed7638f2b8'},
                    'http_review': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/http_hybrid02/table_clearing_review.json',
                                    'sha256': '3df292a089b93fd1c70bc45b5c6a1fffc6757b1460ed073490044d799693bb4f'},
                    'geometry': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/research/velocity_independent01/hybrid_geometry_report.json',
                                 'sha256': 'e17af08523125b3e92bef5a2513f7f8981aa61d14cfaadad18df9617ce95c30a'},
                    'geometry_native_attempt': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/velocity/hybrid_geometry03/table_clearing/qualification.json',
                                                'sha256': 'ebc8976f8b07b3da800c614d4008f21e598dde0a76bbfc2ddb4bfceaffea200e'},
                    'workspace': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_flexible_velocity_fix_20261024/root/workspace_hybrid01/report.json',
                                  'sha256': 'f4520d1fbee01a181162a198d8d9ee8c837ec2f7099efc31ac912c5889625aa3'},
                    'workspace_counts': {'sample_count': 32,
                                         'ik_solved_count': 32,
                                         'fixed_ready_hand_clearance_passed_count': 31,
                                         'all_clearances_passed': False}}}

def velocity_scene_qualification(task_id):
    """Return source-bound metadata only for the exact repaired candidates."""
    if task_id not in REPORTS:
        raise ValueError('No repaired kitchen evidence: ' + str(task_id))
    report = deepcopy(REPORTS[task_id])
    return {
        'status': 'scene_support_and_agent_interface_checked',
        'native_preflight': True, 'scene_support': True,
        'agent_bench_interface': True, 'native_geometry_checked': True,
        'manipulation': False, 'physics_accepted': False,
        'native_source': deepcopy(SOURCES['agent_bench_http']),
        'native_sources': deepcopy(SOURCES),
        'validation_index': deepcopy(INDEX),
        'scene_contact_evidence': [report['canonical'], report['goal']],
        'canonical_support_evidence': report['canonical'],
        'goal_support_evidence': report['goal'],
        'canonical_support_independent_review': report['source_review'],
        'goal_support_independent_review': report['goal_review'],
        'native_interface_evidence': report['http'],
        'native_interface_support_review': report['http_review'],
        'native_geometry_evidence': report['geometry'],
        'native_geometry_original_attempt': report['geometry_native_attempt'],
        'raw_terminal_native_velocity_limits_passed': True,
        'discrete_workspace_evidence': report['workspace'],
        'discrete_workspace_counts': report['workspace_counts'],
        'scope': 'Exact immutable household_contacthybrid_v1 release. Full source14 '
                 'source/goal native support, penetration, release and every raw terminal '
                 'velocity sample pass unchanged thresholds. Native mass/COM/full inertia '
                 'and material observations match the original definitions. Source15 '
                 'actual public HTTP/Agent wrist motion, return and hold pass. Source17 '
                 'read-only native five-shape spoon inventory, cooked handle geometry and '
                 'material/cavity SDF queries are independently audited offline. The original '
                 'all-space two-pitch SDF gate remains failed in sparse far exterior samples; '
                 'the original bound passes for sampled zero-boundary bracketing. The general '
                 'convex-only collector remains unchecked for the expected SDF bowl. '
                 'Normal-Z/weight and unmodified contact API evidence are separately '
                 'reported, not complete friction/torque calibration. Discrete workspace '
                 'counts retain table fork target fixed-ready-hand clearance failure. '
                 'Historical failed candidates remain indexed. No grasp, swept path or '
                 'full robot task is qualified; prepared flags and shared controllers '
                 'remain unchanged. Later source trees inherit no whole-tree native claim.',
    }
