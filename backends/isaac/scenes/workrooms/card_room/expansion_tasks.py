"""Published physical card-game tasks; inspection remains simulation-free."""
from __future__ import annotations

import json

from .game_designs import TASK_IDS, design_for
from .expansion_referee import public_rule_contract
from .scene import environment_reference
from bench.paths import DATA_ROOT


# Immutable input pins only; development outcomes are not admission dependencies.
RELEASES = {'polyomino_territory': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/polyomino_territory/operation_credit_v2',
                         'manifest_sha256': '166ce6f00094146bb8211d0ebbf8d9e1a36a749159ee3f082f4ce31cc35f025b'},
 'barrier_race': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/barrier_race/operation_credit_v2',
                  'manifest_sha256': 'eea3c0e6ca338e23367c5e195a4823483664e7b05d439ae7d30995a082f2e829'},
 'memory_pairs': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/memory_pairs/operation_credit_v2',
                  'manifest_sha256': 'd812a3368f4b7b301f0c7d4442a424ef77cd9b93f4821463a205f25ce94e8ce1'},
 'disk_curling': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/disk_curling/operation_credit_v2',
                  'manifest_sha256': '4cd8303c384a8265000a16f7d20508fa83fc352895218ddbd8592faca6fe5ff5'}}


def task_definition(task_id):
    design = design_for(task_id)
    return {
        'schema': 'workroom_task_definition.v1', 'family': 'card_room',
        'task_id': task_id, 'title': design['title'], 'status': 'implemented',
        'release_schema': 'card_game_expansion_release.v1',
        'difficulty': 4+TASK_IDS.index(task_id), 'runnable': True,
        'entrypoint': f'scenes.workrooms.card_room.tasks.{task_id}.task',
        'objects': {'design_implementation': 'scenes.workrooms.card_room.game_designs.design_for',
                    'asset_root': f'{DATA_ROOT}/assets/task_parts/board_games/{task_id}/scene_v2',
                    'inventory_count': len(design['pieces']),
                    'scope': 'Task-owned tabletop and objects; original parent room is unchanged.'},
        'layout': {'frame': 'world', 'board': design['board'],
                   'initialization': 'Reset-only from immutable, source-bound scene candidate assets.'},
        'robot_requirements': {'hand_id': 'dual_fr3_sharpa_w01', 'articulations': 2,
                               'dof_per_articulation': 29,
                               'ownership': 'Two independent players; original opposed card-room robot layout. Each player owns only its original 29 DOF; black/left and white/right.'},
        'rules': {'implementation': 'scenes.workrooms.card_room.expansion_referee.public_rule_contract',
                  'contract': public_rule_contract(task_id), 'scope': 'Public game rules and native settlement thresholds; intent only reserves an operation.'},
        'evaluator': {'implementation': 'scenes.workrooms.card_room.expansion_referee.ExpansionMatch',
                      'scope': 'Complete native substeps, actual hand contact, support and released stability; missing evidence cannot earn a move.'},
        'source_release': dict(RELEASES[task_id]),
        'scene_candidate': {'design_schema': design['schema'],
                            'parent_environment': environment_reference(),
                            'build_entrypoint': 'scripts.assets.build_card_game_candidates.build_candidate',
                            'author_implementation': 'scenes.workrooms.card_room.game_expansion_assets.author_task',
                            'public_game_session_implemented': True,
                            'game_protocol': 'bench_card_game_state_v1',
                            'inspect_module': f'scenes.workrooms.card_room.tasks.{task_id}.task'},
        'qualification': {'native_manipulation': False,
                          'model_success': False, 'physics_accepted': False,
                          'scope': 'Scene construction/object engineering evidence is separate from full robot play; development reports never serve as runtime admission inputs.'},
    }


def candidate_specification(task_id):
    return {'schema': 'card_room_scene_candidate_spec.v1', 'task_id': task_id,
            'environment': environment_reference(), 'design': design_for(task_id),
            'public_rules': public_rule_contract(task_id), 'runnable': True,
            'scope': 'Static scene design inspection. No physics, robot execution or model call.'}


def main(task_id):
    print(json.dumps(candidate_specification(task_id), ensure_ascii=False, indent=2))
