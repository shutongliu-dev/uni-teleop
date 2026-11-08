from bench.paths import DATA_ROOT
from scenes.workrooms.living_room.task_design import planned_task, bind_prepared_release
from scenes.workrooms.living_room.tower_geometry import geometry_contract

TASK = planned_task(
    task_id='tower_extraction', number=4, title='Task4 · 双臂轮流抽塔对抗', articulations=2, confirmed=True,
    intent='两方轮流从桌面高塔抽一根放入己方实体收集托盘，不放回塔顶；造成倒塔的一方输。',
    objects={'intended_roles': ['independent_wooden_blocks', 'front_collection_tray', 'rear_collection_tray'], 'asset_bindings': None},
    rule_contract={'geometry': geometry_contract(), 'competitive': True, 'players': ['front', 'rear'],
                   'operation': 'extract_one_and_collect', 'restack_on_top': False,
                   'protected_bottom_layers': 1, 'protect_highest_complete_layer_and_above': True,
                   'handoff_stability_s': 1., 'runtime_parameters_required': ['block_size_m'],
                   'initial_layers': 12, 'block_size_m': [.20, .045, .030],
                   'in_layer_gap_m': .010, 'collection_support': 'own_tray_floor'},
    reference_evaluator='scenes.workrooms.living_room.tasks.tower_extraction.evaluator.TowerExtractionGame',
    design={'skills': ['contact_based_block_selection', 'controlled_push_pull', 'regrasp', 'stability_assessment'],
            'reset': '12-layer, 36-block tower stands directly on tabletop; front/rear physical collection trays start empty.',
            'players': 'Independent identities and action ownership; only current player may move.',
            'fairness': 'Seed-balanced starting side, symmetric collection capacity and equal turn allowance.',
            'failure_attribution': 'Current player retains responsibility until released stable handoff.',
            'non_game_outcomes': 'Missing evidence, invalid reset or inactive-player physical interference abort without winner; timeout/budget incomplete.',
            'dimensions': '200×45×30 mm blocks, 10 mm adjacent gaps, 22.5 mm end overhang; native grasp/extraction still requires separate qualification.'})


TASK = bind_prepared_release(
    TASK, path=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/tower_extraction/v9',
    manifest_sha256='59b55b59de1d802e40e7b8f9b6df7aa28a45e897f14a04c4b3be57e4d432e8e1',
)

TASK['qualification']['object_interaction_evidence'] = {
    'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/living_interactions_20261024/towers/qualification.json',
    'sha256': '1c7ae15762120a8ea2d24fd2f5894505da4db5d8591a5550147df07240fed9f5',
    'scope': 'Object-only v8 reset, collection support and bounded-force extraction with complete released '
             'terminal review. Original extraction report export failures retained; separate raw-trace '
             'recovery passed. No robot extraction, legal game turn or match qualification.',
}
