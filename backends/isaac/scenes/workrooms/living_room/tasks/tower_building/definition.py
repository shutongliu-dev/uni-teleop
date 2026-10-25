from bench.paths import DATA_ROOT
from scenes.workrooms.living_room.task_design import planned_task, bind_prepared_release

TASK = planned_task(
    task_id='tower_building', number=3, title='Task3 · 单臂搭积木塔', articulations=1, confirmed=True,
    intent='从两侧托盘取木块自由搭塔，不限制塔形；按本局松手后达到的最高稳定高度登记分数。',
    objects={'intended_roles': ['independent_wooden_blocks', 'physical_supply_trays'], 'asset_bindings': None},
    rule_contract={'goal': 'Record the highest released stable tower height reached during the episode.',
                   'scoring_mode': 'freeform_height', 'score_unit': 'm', 'retain_best_after_collapse': True,
                   'inventory_block_count': 36, 'block_size_m': [.15, .045, .025],
                   'stable_duration_s': 1., 'tower_form_restricted': False},
    reference_evaluator='scenes.workrooms.living_room.tasks.tower_building.evaluator.TowerBuildingEvaluator',
    design={'skills': ['slender_block_grasp', 'freeform_construction', 'precise_release', 'multi_body_balance', 'physical_recovery'],
            'reset': '36 blocks in two physical side trays, nine two-block stacks per tray; tabletop build area empty.',
            'score': 'Highest actual block top above tabletop after one second released stability; real support chain to table, no tray or external support.',
            'success': 'Continuous height score; no binary target height, required count, layer pattern or orientation.',
            'recovery': 'Regrasp and rebuild freely; retain the highest confirmed stable height after later collapse.',
            'dimensions': '150×45×25 mm wooden blocks, explicit versioned scene candidate.'})


TASK = bind_prepared_release(
    TASK, path=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/living_room/tower_building/v5',
    manifest_sha256='78118f6ae52e523c532e2e57b98f6129baf3fd1fee39a1a56432640b2c0b4e28',
)
