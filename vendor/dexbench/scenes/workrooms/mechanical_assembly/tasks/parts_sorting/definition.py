"""L1: frozen sorting scene with separated lanes and the source-preserving SDF ring."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.task_design import planned_task, bind_prepared_release


TASK = planned_task(
    task_id='parts_sorting', title='L1 · 大零件分类', difficulty=1,
    intent='将三个零件分别放入与其形状图标相同的浅盘，全部松手后保持稳定。',
    objects={'intended_roles': ['large_block', 'large_cylinder', 'thick_ring', 'three_sorting_bins'],
             'inventory_counts': {'block': 1, 'cylinder': 1, 'ring': 1}, 'asset_bindings': None},
    design={
        'skills': ['recognize_shape', 'top_down_grasp', 'transfer', 'release'],
        'candidate_geometry_m': {
            'block_xyz': [.060, .060, .045], 'cylinder_diameter_height': [.060, .045],
            'ring_outer_inner_height': [.075, .035, .035],
            'bin_inner_xy': [.130, .130], 'bin_wall_height': .018,
            'minimum_initial_part_center_spacing': .140,
        },
        'reset_design': 'Three separate upright parts with no clutter or stacking; shape icons supplement color.',
        'sequence': ['Pick one part.', 'Move above its bin.', 'Lower and release.', 'Repeat for the remaining two.'],
        'success_design': {
            'all_of': ['All three complete collision geometries lie inside the matching bin interiors.',
                       'Each part is supported by its bin, released by the hand and stable.'],
            'candidate_stable_seconds': .5,
        },
        'failure_design': ['Unrecoverable drop outside reachable workspace.', 'Budget exhausted before all bins are correct.'],
        'recovery_design': 'A wrong-bin placement is recoverable; do not terminate before the agent can correct it.',
        'difficulty_control': 'No tight pockets, orientation target or pile picking; six pieces are a later separately released extension.',
        'next_validation': ['Native pickup/release for each shape.', 'Whole-geometry bin containment and support evidence.'],
    },
)

TASK = bind_prepared_release(
    TASK,
    path=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases/parts_sorting',
    manifest_sha256='27a0ca3c69388cf4aed1b9ca34ad87bcaebbeb8dfdf1f351c2a9aba16a724252',
    evidence=[{
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/sorting/release_binding01/report.json',
        'sha256': 'ece977658e4b341853e765acf02539b293e8012a4897252cba6bd6571d72bba0'}, {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/sorting/native_sdf1024_01_independent_review.json',
        'sha256': 'ba3f9eac66c47dd00dc7c0413a11d999704ef8a3148bf3b3367aab9e569929e8'}],
    scope='The corrected separated supply/tray lanes and explicit source-preserving SDF ring match the retained four-case native gravity evidence. Current whole-geometry scoring is separately frozen and CPU checked; no robot transport or canonical episode success is claimed.',
)
