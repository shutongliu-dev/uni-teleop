"""Single-player placement prerequisite, separate from competitive game levels."""

from bench.paths import DATA_ROOT
TASK = {
    'schema': 'workroom_task_definition.v1', 'family': 'card_room', 'task_id': 'task0',
    'title': 'Task0 · 单臂随机三子摆放', 'status': 'implemented', 'difficulty': 0,
    'entrypoint': 'scenes.workrooms.card_room.tasks.task0.task',
    'release_schema': 'card_room_placement_task_release.v1',
    'release_schema_version': 'isaacsim-card-room-placement-release-v1',
    'objects': {
        'asset_root': f'{DATA_ROOT}/assets/task_parts/board_games/tic_tac_toe_3x3/v3',
        'board': 'board', 'inventory_counts': {'black': 3}, 'colors': ['black'],
        'stands': ['stand_left'], 'reserve_stations': ['reserve_black_west'],
        'collision_assets': {
            'tokens': 'token_black_union_sdf1250_sparse32_subgrid3_surface05mm',
            'board': 'board_triangle_grid10mm',
            'binding': 'Pinned asset roots and dependency hashes in the frozen release scene and manifest.'}},
    'layout': {'frame': 'world', 'board_center_xy_m': [.65, 0.], 'table_top_z_m': .78,
        'reserve_surface_z_m': .80, 'pitch_m': .14, 'layout_id': 'card_room_single_arm_v1',
        'initialization': 'Frozen three-piece supply and robot reset; the episode seed only selects targets.'},
    'robot_requirements': {'hand_id': 'sharpa_w01_right', 'articulations': 1,
        'dof_per_articulation': 29, 'base_constraint': 'native_fixed_base_v1',
        'base_position_m': [.65, -.60, .90], 'ownership': 'One agent controls one arm and right hand.'},
    'rules': {'implementation': 'scenes.workrooms.card_room.tasks.task0.rules',
        'board_size': 3, 'target_count': 3, 'target_selection': 'episode_seed_without_replacement',
        'intent': 'Place all three pieces in three randomly assigned distinct cells, release and stabilize.',
        'competitive': False, 'requires_three_in_a_row': False},
    'evaluator': {'implementation': 'scenes.workrooms.card_room.tasks.task0.evaluator',
        'frozen_path': 'evaluator/sequence_evaluator.py',
        'scope': 'Actual three-piece placement; 20ms poses and complete native contact windows, 300ms release stability.'},
    'source_release': {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/card_room/task0/v5',
        'manifest_sha256': '9a7171a6e979ca7bbbe914df3e07a43349e74e4559a3cdf0d96b7075a30c3fbe'},
    'qualification': {'result_source': 'external_evidence_report', 'manipulation': False,
        'native_preflight': True, 'physics_accepted': False,
        'contact_repair_regression': True, 'public_http_smoke': True,
        'evidence': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/task0_contact_closeout_20261018/qualification.json',
        'evidence_sha256': '2c53cdc4d8223bcc232fb2f78b3b6e7d7ac0b1bdaa33b5d891a0d9724e876775',
        'scope': 'Known large contact ejections did not recur in the complete original-command regression; public HTTP passed. Old corrective commands are not a strategy-success gate after physics changes. No new model or general manipulation qualification; frozen preparation status remains unchanged.'},
}
