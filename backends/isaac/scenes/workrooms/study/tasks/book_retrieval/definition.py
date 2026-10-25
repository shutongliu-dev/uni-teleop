"""Frozen study engineering candidate; native qualification remains separate."""
from scenes.workrooms.study.task_design import prepared_task


TASK = prepared_task(
    task_id='book_retrieval', number=3, title='Task3 · 指定书取出',
    objects={'intended_roles': ['one_retrieval_book', 'two_protected_books', 'desk_destination_slot'],
             'inventory_counts': {'robotics': 1, 'astronomy': 1, 'botany': 1},
             'retrieval_target_id': 'robotics',
             'protected_ids': ['astronomy', 'botany']},
    design={'skills': ['shelf_approach',
                       'selective_grasp',
                       'neighbor_preservation',
                       'shelf_to_desk_transfer'],
            'reset': 'All three books stand on the shelf; the target desk region is empty.',
            'success': 'Only robotics ends in its desk slot; astronomy and botany remain upright in '
                       'their own complete original shelf slots.',
            'recovery': 'A disturbed protected book can be restored through robot actions before '
                        'the full common stability window.'},
    release_version='v3',
    manifest_sha256='8a385118e42561684c80dfb78ae9f0eb2333daf6035d6e119254137956e4e91d',
)
