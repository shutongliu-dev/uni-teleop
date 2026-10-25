"""L3: three broad, unthreaded rings on one fixed guided shaft."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.task_design import planned_task, bind_prepared_release


TASK = planned_task(
    task_id='shaft_spacer_assembly', title='L3 · 轴上隔套顺序装配', difficulty=3,
    intent='按示意图从下到上依次装入隔套、宽环和顶环，三件都落座后松手。',
    objects={'intended_roles': ['spacer', 'wide_ring', 'top_ring', 'fixed_vertical_shaft', 'three_supply_nests'],
             'inventory_counts': {'spacer': 1, 'wide_ring': 1, 'top_ring': 1}, 'asset_bindings': None},
    design={
        'skills': ['part_sequence', 'annular_part_grasp', 'guided_lowering', 'stack_settlement'],
        'candidate_geometry_m': {
            'shaft_diameter': .024, 'shaft_exposed_height': .100, 'shaft_tip_diameter': .010,
            'shaft_taper_height': .024, 'part_bore_diameter': .032,
            'part_lower_entry_diameter': .046, 'part_entry_depth': .008,
            'spacer_outer_diameter_height': [.070, .030],
            'wide_ring_outer_diameter_height': [.085, .020],
            'top_ring_outer_diameter_height': [.070, .025],
        },
        'reset_design': 'Parts lie flat and apart with holes visible; the shaft and bottom shoulder are fixed to the workplate.',
        'sequence': ['Place the spacer over the shaft and release.', 'Place the wide ring onto the spacer.',
                     'Place the top ring onto the wide ring and release.'],
        'success_design': {
            'all_of': ['Correct bottom-to-top order and all three bores surround the shaft.',
                       'Support chain is shoulder → spacer → wide ring → top ring.',
                       'No hand holds the stack and no object is merely hovering or penetrating another.'],
            'candidate_stable_seconds': .5,
        },
        'failure_design': ['Unrecoverable drop.', 'Budget exhausted before a supported stack is complete.'],
        'recovery_design': 'The agent may lift an incorrectly placed top part off and rebuild the stack.',
        'difficulty_control': 'Wide annular parts, visible order, no keys, bearings, circlips or threads; no in-hand reorientation.',
        'next_validation': ['Hand clearance around the shaft tip for all three release heights.',
                            'Native support contacts and recovery from a wrong ordering.'],
    },
)

TASK = bind_prepared_release(
    TASK,
    path=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261019/shaft_spacer_candidate_v2',
    manifest_sha256='ca7a0b5453a8b234001004404f96b410da9e4fff596586623237bb55a2d7b8a6',
    evidence=[{
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/peg_shaft_delivery02/binding_report.json',
        'sha256': '04f534600bf3616eab68d5315b9a51f72bd7068adf3a699a2e4437fb33a5e371'}],
    scope='The updated task-owned layout and zero-model privileged contract are frozen with the source-preserving dense SDF rings and explicit PGS profile. Native robot manipulation remains unqualified.',
)
