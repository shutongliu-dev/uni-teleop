"""Frozen study engineering candidate; native qualification remains separate."""
from scenes.workrooms.study.task_design import prepared_task


TASK = prepared_task(
    task_id='book_stacking', number=2, title='Task2 · 按序叠书',
    objects={'intended_roles': ['three_distinct_books', 'ordered_stack_slots'],
             'inventory_counts': {'robotics': 1, 'astronomy': 1, 'botany': 1},
             'stack_order': ['robotics', 'astronomy', 'botany']},
    design={'skills': ['book_grasp', 'directed_orientation', 'ordered_stacking', 'support_chain'],
            'reset': 'Three independent books start apart with different headings on the desk; '
                     'the required ordered stack has not been formed.',
            'success': 'Bottom to top: robotics, astronomy, botany; front covers face upward, '
                       'spines share a direction, and each layer rests on its immediate lower body.'},
    release_version='v3',
    manifest_sha256='842db0ee31458aaacfb7b75fc70e5343b4eb4a3b8a4423cd7e8e2ac77dc66136',
)
