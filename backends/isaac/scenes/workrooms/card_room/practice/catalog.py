"""Companion practice entries; formal game numbers and releases are unchanged."""
from .cases import GYM_IDS, STRATA, parent_id, rules_for
from bench.paths import DATA_ROOT

# Filled only after complete immutable preparation. No report/outcome is an
# admission dependency. Canonical definitions remain planned before publication.
RELEASES = {'polyomino_territory_practice': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/polyomino_territory_practice/operation_credit_v2',
                                  'manifest_sha256': '05d71148ea988eee5fb6fb62bb5bbc2df526fbf1525fcd6aba5c465e71cf5627'},
 'barrier_race_practice': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/barrier_race_practice/operation_credit_v2',
                           'manifest_sha256': 'a9e13bc0ba059285e72cc3f99ec7d19d5121a3de80777b585ef576d9f1efd50d'},
 'memory_pairs_practice': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/memory_pairs_practice/operation_credit_v2',
                           'manifest_sha256': '344f15e2c22cc09264eefd6451677ad915d68343f2f76bd70499c4e623ec29d2'},
 'disk_curling_practice': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/disk_curling_practice/operation_credit_v2',
                           'manifest_sha256': '423ecbc1e8d2b7ce845a7825f34a59917a56e84279233f71fbe31448d4a8cb40'}}


def task_definition(gym_id):
    task_id = parent_id(gym_id)
    release = RELEASES.get(gym_id)
    return {'schema': 'workroom_task_definition.v1', 'family': 'card_room',
        'task_id': gym_id, 'title': task_id+' RSI 练习',
        'status': 'implemented' if release else 'planned', 'difficulty': 20+GYM_IDS.index(gym_id),
        'release_schema': 'card_rsi_practice_release.v1',
        'entrypoint': 'scenes.workrooms.card_room.practice.release',
        'objects': {'parent_task_id': task_id, 'scope': 'Original scene-v2 chess apparatus, materials and collision geometry.'},
        'layout': {'reset': 'Finite frozen reset-only transform layers; same original room and robot bases.'},
        'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01', 'articulations': 2, 'dof_per_articulation': 29,
            'ownership': 'One learner selects left or right per episode; the other original robot keeps its ACK drive.'},
        'rules': {'contract': rules_for(gym_id), 'learning': 'task_known_rsi_practice_then_frozen_formal_evaluation'},
        'evaluator': {'implementation': 'scenes.workrooms.card_room.practice.referee.PracticeReferee' if release else None},
        'source_release': release,
        'cannot_run_reason': None if release else 'Immutable practice release has not yet been published.',
        'practice': {'create_endpoint': 'POST /episodes', 'selection': ['active_side', 'case_id', 'split'],
            'strata': list(STRATA[task_id]), 'formal_task_id': task_id, 'formal_game_result': False},
        'qualification': {'native_manipulation': False, 'model_success': False, 'physics_accepted': False,
            'scope': 'Practice scores apply only to their physical goals; no automatic formal gameplay qualification.'}}
