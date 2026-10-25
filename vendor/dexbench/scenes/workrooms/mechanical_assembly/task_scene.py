"""Empty mechanical task composition; no task objects or controller strategy."""
from copy import deepcopy
from pathlib import Path

from bench.isaacsim.release import task_inputs
from bench.paths import DATA_ROOT, load_json
from bench.simulation.isaacsim_backend.workroom import checked
from scenes.workrooms.registry import BASE_RELEASE, PHYSICS_OVERRIDES
from scenes.workrooms.observations import camera_specs
from .scene import environment_reference
from .robot_layout import robot_configuration, ready_configuration


DEFAULT_STAND_ROOT = Path(f'{DATA_ROOT}/assets/task_parts/mechanical_workshop_supports/v1')
STAND_MANIFEST_SHA256 = '640a7aed72a2fbe8aad779f7aab454880b72b1da2f4e5b417a252b54af6c168f'


def empty_task_scene(stand_root=DEFAULT_STAND_ROOT, *, solver_type=None):
    """Return ``scene, initial_state, ready_design`` with the fixed shared layout.

    This CPU-only constructor preserves the existing robot, room and native
    physics options unless the task explicitly selects TGS or PGS. Omission
    preserves the existing scene. The selection is frozen in the scene, without
    claiming native qualification or changing any material or controller.
    """
    if solver_type is not None and (not isinstance(solver_type, str) or solver_type not in ('TGS', 'PGS')):
        raise ValueError('solver_type must be None, TGS or PGS')
    stand_root = Path(stand_root).resolve()
    checked(stand_root / 'manifest.json', STAND_MANIFEST_SHA256)
    _, scene, _, _ = task_inputs(BASE_RELEASE, '01_cube_push')
    for key in ('room', 'scene_variant', 'static_props', 'surfaces', 'robot_pedestal', 'soft_bodies'):
        scene.pop(key, None)
    reference = environment_reference()
    checked(reference['usd_path'], reference['usd_sha256'])
    checked(reference['dependency_manifest_path'], reference['dependency_manifest_sha256'])
    description = load_json(Path(reference['dependency_manifest_path']))
    scene.update(robot=robot_configuration(), objects=[], fixtures=[], goal_markers=[],
                 cloth={'enabled': False}, workbench={'enabled': False},
                 workcell_details={'enabled': False}, workroom=reference,
                 cameras=camera_specs(description),
                 description='Mechanical task composition; native task dynamics not yet qualified.')
    scene['isaacsim'].update(deepcopy(PHYSICS_OVERRIDES))
    initial, ready_design = ready_configuration(scene['robot'])
    scene['fixtures'].append({'id': 'robot_stand', 'asset_id': 'mechanical_robot_stand',
                              'asset_root': str(stand_root), 'position_m': [0., 0., 0.]})
    scene['mechanical_task_provenance'] = {
        'shared_layout_id': 'mechanical_single_arm_v1',
        'stand_manifest_sha256': STAND_MANIFEST_SHA256,
        'task_objects_added_by_caller': True,
        'physics_accepted_at_preparation': False,
        'scope': 'Same fixed room, stand and ready robot; no qualification transfer to task contacts.'}
    if solver_type is not None:
        scene['isaacsim']['solver_type'] = solver_type
        scene['mechanical_task_provenance']['solver_selection'] = {
            'requested': solver_type,
            'configuration_path': 'scene.isaacsim.solver_type',
            'scope': 'Explicit task release option; requires evidence for this frozen configuration.',
            'native_validation_claimed': False}
    return scene, initial, ready_design
