"""Task6: unfold a physical reading panel, insert its support pin and load a book."""
from bench.paths import DATA_ROOT
from scenes.workrooms.study.task_design import prepared_task

TASK = prepared_task(
    task_id='reading_stand', number=6, title='Task6 · 展开锁定阅读支架并放书',
    objects={'intended_roles': ['hinged_panel', 'sliding_support_pin', 'hardback_book', 'fixed_base_and_guide'],
             'inventory_counts': {'panel': 1, 'support_pin': 1, 'book': 1},
             'mechanism': 'Passive native revolute hinge and 280 mm transverse slide; actual pin-panel contact carries the load.'},
    design={'skills': ['constrained_rotation', 'sliding_pin_insertion', 'inclined_book_placement',
                       'mechanism_support_and_release'],
        'reset': 'Folded reading panel, withdrawn support pin and independent green book on the desk.',
        'success': 'Unfold panel to 50–60 degrees; insert the support pin; place the whole book on the reading face; release with real contacts and settle 400 ms.',
        'recovery': 'Regrasp, adjust the angle, withdraw/reinsert the pin and stage the book on the desk; action order and hand assignment are free.',
        'fixed_room_unchanged': True,
        'difficulty_progression': 'Task6 combines a constrained mechanism with independent object placement; not a measured success-rate ranking.'},
    release_version='v1',
    manifest_sha256='7179879f41144ecff96e54a75d5f15e7f3c0bce68b91c29fa2e47e351d4a4cd8',
)
TASK['entrypoint'] = 'scenes.workrooms.study.tasks.reading_stand.task.build_task_inputs'
TASK['evaluator']['implementation'] = 'scenes.workrooms.study.reading_stand_evaluation.SequenceEvaluator'
TASK['recommended_budgets'] = {'sim_time_s': 240., 'max_actions': None}

TASK['qualification'].update(
    object_interactions=True,
    object_interaction_evidence={
        'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/study/qualification.json',
        'sha256': 'b8bb7a0a6a27a4a95bc770512017be68e22a07ab572e789cff310b9d5bf458dd',
        'passed': True,
        'scope': 'Seven real object probes, 33.4 s/66807 frames, exact original-interpreter full readback and 400 ms released support. Includes sequential one-object control, unlocked/withdrawn/unseated negatives and zero-progress reset. Explicit engineering forces are rejected by the ordinary referee; no robot manipulation claim.',
    },
)

# Post-run metadata only; the frozen release and its preparation flags are unchanged.
TASK['qualification'].update({'native_preflight': True,
 'agent_bench_interface': True,
 'public_http_smoke': True,
 'public_functions_smoke': True,
 'object_interactions': True,
 'discrete_workspace': True,
 'nonadjacent_arm_self_collision': True,
 'configuration_evidence': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/interaction_expansion_20261025/root/configuration_qualification.json',
                            'sha256': '670335af5e31afb4ee76ffe4988b0ee397c695b6a59e2abfe9d77df40901d08e'},
 'manipulation': False,
 'physics_accepted': False,
 'scope': 'Native object mechanisms, exact selected Bench/Agent interfaces and recorded-stage discrete '
          'approaches verified separately. Complete robot manipulation and continuous paths remain '
          'unqualified.'})
