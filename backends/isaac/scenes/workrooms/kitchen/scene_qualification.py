"""Static, source-bound evidence pointers for the six rigid kitchen candidates.

Catalog discovery never opens the reports. Physics and public contracts live in
the immutable releases; these pointers do not promote their preparation flags.
"""
from __future__ import annotations

from bench.paths import DATA_ROOT

ROOT = f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/kitchen_dynamics_repair_20261023'
SOURCE_MANIFEST_SHA256 = 'a70953328e02eec52868c1162ac2326f3f89aa47c8b5dab5b585af0a53f16afc'
INDEX_SHA256 = '8b49995ac77bbaab8078992bf9bb929ecb451e5de7490e76a83af4b40ce9324c'
GOAL_REVIEW_SHA256 = '7c2dfdc26b145d6ffcdd05f0cedc5587cd0baeba1ab7f492f2e6a0f3fa339f47'
# HTTP, independent HTTP review, goal, and separate raw-velocity diagnostic.
REPORTS = {'breakfast_serving': ('c225fbb663f912736633d73c4546a88de86b057407ce71bfdd52d8ebbc8f1ee3',
                       '42df7994bacf991046382db40cef829ff76f8b063c453cc299aed59b58011f37',
                       'ff56719a0f9acec883c349a1a42eae124cbe553637cae1cf686e5a23a7ba6bc3',
                       True),
 'produce_transfer': ('dd833dc5a0e89cac605f313fccdfac415894ebf7ce646b606055920471eaba1b',
                      'aae6cf6b2e44e673f70cce2a0b8bed1056b34d4e6974c2c88ff7b2c8dcc028de',
                      '4605c217af76e6b3b3422dfd00a4af3c0799393519cb09d446e6d2d30ba2be64',
                      True),
 'utensil_sorting': ('329569c129da4717304b7b5b231c60f4ca05d42366d862fc2ec99bfb02dc6cf3',
                     '5ae5f34d7210db78354bd931320eacf4e122674b8fc01234b7bde929a1a77a15',
                     '6a0e4d4741a54ec0ceab9359b56a868b902c5674d4e63b1b994c8af77164d8f9',
                     False),
 'pot_lid_placement': ('6e95292506ca5c13d8d0f214aa98f3a065ef7ebbada55a06a7e3cdc748a23aa7',
                       '8e3b6d2858a39481ec22935660f45843f73891373df246ec5ad4b08b2074643d',
                       '3b56e0c8f837b2bd65805f6c134e7c5ab6d249275d08d9cdc4a16e00c97a8e86',
                       True),
 'pantry_stocking': ('e99854c69c70b7de186d31529114c0e1d82a28acfc2d3fc233162252ffd2167e',
                     '0b769ba91512a9a4932de672a97e0231a2e3ba0d69f4a42b108aa2bdead53425',
                     'e2c6f02092c10630f34db467509161c46b1c0da93fc9384abfd7daf3bda57af8',
                     True),
 'table_clearing': ('e185a42285e4791199b426247621f40f356b1dd27fd644fe23684ac6a0e96eda',
                    '76f2c1f99db7d369a057556afdecb1ebdd1e5961acc453cfba7ade23dc323c71',
                    '197ef395fc25ddae485fa256b93305b03c6c99816875fe2b73f79be8dd5964c1',
                    False)}


def rigid_scene_qualification(task_id):
    """Return fresh metadata only for the exact six completed rigid runs."""
    if task_id not in REPORTS:
        raise ValueError('No current source-bound rigid evidence: ' + str(task_id))
    http_sha, review_sha, goal_sha, raw_velocity = REPORTS[task_id]
    goal = {'path': f'{ROOT}/current_kitchen_goal01/{task_id}/qualification.json', 'sha256': goal_sha}
    return {
        'status': 'scene_support_and_agent_interface_checked',
        'native_preflight': True, 'scene_support': True, 'agent_bench_interface': True,
        'manipulation': False, 'physics_accepted': False,
        'native_source': {'path': f'{ROOT}/source15', 'manifest_sha256': SOURCE_MANIFEST_SHA256},
        'validation_index': {'path': f'{ROOT}/current_kitchen_matrix01/qualification.json', 'sha256': INDEX_SHA256},
        'scene_contact_evidence': [dict(goal)], 'goal_support_evidence': dict(goal),
        'native_interface_evidence': {
            'path': f'{ROOT}/current_kitchen_http02/{task_id}/qualification.json', 'sha256': http_sha},
        'native_interface_support_review': {
            'path': f'{ROOT}/current_kitchen_http02/{task_id}_review.json', 'sha256': review_sha},
        'goal_support_independent_review': {
            'path': f'{ROOT}/current_kitchen_matrix01/goal_review_final01.json', 'sha256': GOAL_REVIEW_SHA256},
        'raw_terminal_native_velocity_limits_passed': raw_velocity,
        'scope': 'Exact frozen source15 native source/goal support and actual Agent/Bench HTTP only. '
                 'All native mass/inertia/COM comparisons pass; normal-only impulses are reported separately '
                 'from complete force balance. Utensil and table-clearing raw angular-velocity excursions '
                 'remain explicit despite passing the unchanged rolling-pose stability rule. Workspace '
                 'references retain their own source identity. No grasp, swept-path, full task or calibrated '
                 'physics qualification; prepared flags and shared controllers remain unchanged. '
                 'Later main/source trees do not inherit this native qualification.',
    }
