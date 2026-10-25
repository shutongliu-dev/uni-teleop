"""Frozen study engineering candidate; native qualification remains separate."""
from scenes.workrooms.study.task_design import prepared_task


TASK = prepared_task(
    task_id='stationery_sorting', number=1, title='Task1 · 文具分类',
    objects={'intended_roles': ['thick_pencil', 'marker', 'eraser', 'three_separate_identity_slots'],
             'inventory_counts': {'pencil': 1, 'marker': 1, 'eraser': 1}},
    design={'skills': ['identity_recognition', 'tabletop_grasp', 'transfer', 'release'],
            'reset': 'Three stationery bodies start separately on the desk, outside their own '
                     'target slots.',
            'success': 'Each whole body occupies its own category and identity slot, supported by '
                       'the desk; all three released and stable together.'},
    release_version='v4',
    manifest_sha256='69acaf448f031fdbbeb2811093dc5f8b60ea7b592972732ccfccd40477583d1e',
)
