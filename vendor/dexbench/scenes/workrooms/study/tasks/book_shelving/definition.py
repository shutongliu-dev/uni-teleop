"""Frozen study engineering candidate; native qualification remains separate."""
from scenes.workrooms.study.task_design import prepared_task


TASK = prepared_task(
    task_id='book_shelving', number=4, title='Task4 · 按类别上架',
    objects={'intended_roles': ['category_labeled_books', 'separate_shelf_slots'],
             'inventory_counts': {'robotics': 1, 'botany': 1, 'architecture': 1},
             'categories': {'robotics': 'science', 'botany': 'science', 'architecture': 'arts'}},
    design={'skills': ['category_recognition',
                       'desk_to_shelf_transfer',
                       'narrow_slot_alignment',
                       'upright_release'],
            'reset': 'Three books start apart on the desk; their category shelf slots are empty.',
            'success': 'All books occupy separate identity slots in their category bay, upright '
                       'with the +X spine facing the +X opening; same-category books cannot '
                       'exchange slots.'},
    release_version='v3',
    manifest_sha256='358624a445ac09a75c4dfa2088bb1bafac80657cdc758db5d0913db56b081bdf',
)
