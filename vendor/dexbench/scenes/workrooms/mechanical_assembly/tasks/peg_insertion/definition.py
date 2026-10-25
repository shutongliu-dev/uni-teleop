"""L2: guided, shallow placements without threading or press fit."""
from bench.paths import DATA_ROOT
from scenes.workrooms.mechanical_assembly.task_design import planned_task, bind_prepared_release


TASK = planned_task(
    task_id='peg_insertion', title='L2 · 粗销浅孔放置', difficulty=2,
    intent='将两根粗销放入各自标记的孔位，放到底并松手。',
    objects={'intended_roles': ['two_large_pegs', 'fixed_two_socket_workplate', 'upright_supply_cradles'],
             'inventory_counts': {'peg': 2}, 'asset_bindings': None},
    design={
        'skills': ['upright_grasp', 'coarse_xy_alignment', 'guided_lowering', 'release'],
        'candidate_geometry_m': {
            'peg_diameter': .032, 'peg_length': .070, 'peg_grip_diameter': .048,
            'socket_bore_diameter': .038, 'socket_entry_diameter': .060,
            'socket_guide_depth': .012, 'straight_seating_depth': .015,
            'socket_center_spacing': .160,
        },
        'reset_design': 'Pegs start vertical in shallow open cradles; fixed sockets face upward.',
        'sequence': ['Grasp an exposed wide peg head.', 'Center it over the matching broad entry.',
                     'Lower slowly through the passive guide.', 'Release after the lower seat supports the peg.'],
        'success_design': {
            'all_of': ['Both correct pegs rest on their socket bottom seats.',
                       'Pegs are upright, released and stable; hovering above an entry is insufficient.'],
            'candidate_axis_tilt_degrees': 10., 'candidate_stable_seconds': .5,
        },
        'failure_design': ['Unrecoverable drop.', 'Completion budget exhausted; no failure merely for first contact with the funnel.'],
        'difficulty_control': 'Circular parts remove yaw alignment; broad entries and short bores reduce required precision. No interference fit.',
        'next_validation': ['Actual contact-offset and bore-clearance audit.',
                            'Native insertion from several entry offsets without snapping or geometry penetration.'],
    },
)

TASK = bind_prepared_release(
    TASK,
    path=f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_referee_20261018/rules_v1/releases/peg_insertion',
    manifest_sha256='5c116c6c99ea52734b2f2cea829931975ec81af0f1b59840288f757fb1a88828',
    evidence=[{
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/peg/release_binding02/report.json',
        'sha256': '2a8cad465b0218e29d103c1184983ca69ed98f7e3285e11e2efa63cb34ec13e4'}, {
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_task_validation_20261018/peg_shaft_delivery02/binding_report.json',
        'sha256': '04f534600bf3616eab68d5315b9a51f72bd7068adf3a699a2e4437fb33a5e371'}],
    scope='The original supply reset uses the explicit PGS scene covered by supply, centered, guided and outside-socket native probes. Half-step settling failures and the longer observation remain separate evidence, without a transient convergence or robot manipulation claim.',
)
