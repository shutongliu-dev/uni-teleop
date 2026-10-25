"""The new empty task base preserves the accepted diagnostic composition."""
from copy import deepcopy
import importlib

import pytest

from scenes.workrooms.mechanical_assembly.dynamics import diagnostic_scene
from scenes.workrooms.mechanical_assembly.task_scene import DEFAULT_STAND_ROOT, empty_task_scene


def test_empty_task_base_preserves_room_robot_and_physics_without_threaded_assets():
    scene, initial, ready = empty_task_scene()
    old_scene, old_initial, _, old_ready = diagnostic_scene(DEFAULT_STAND_ROOT)
    for field in ('robot', 'workroom', 'cameras', 'isaacsim', 'physics', 'fixtures'):
        assert scene[field] == old_scene[field]
    assert initial == old_initial
    assert ready == old_ready
    assert scene['objects'] == []
    assert [row['id'] for row in scene['fixtures']] == ['robot_stand']
    assert 'mechanical_diagnostic_provenance' not in scene
    assert 'threaded' not in str(scene['mechanical_task_provenance'])
    snapshot = deepcopy(scene)
    other, _, _ = empty_task_scene()
    other['robot']['base_position_m'][0] = 99.
    assert scene == snapshot


@pytest.mark.parametrize('value', [False, True, 0, 'pgs', 'CPU', [], {}])
def test_explicit_solver_rejects_invalid_selection_before_loading_assets(value):
    with pytest.raises(ValueError, match='solver_type'):
        empty_task_scene('/nonexistent', solver_type=value)


@pytest.mark.parametrize('task_name', [
    'parts_sorting', 'peg_insertion', 'shaft_spacer_assembly', 'gear_train_assembly'])
@pytest.mark.parametrize('solver', ['PGS', 'TGS'])
def test_task_solver_option_preserves_geometry_reset_material_and_thresholds(task_name, solver):
    module = importlib.import_module('scenes.workrooms.mechanical_assembly.tasks.'+task_name+'.task')
    original = module.build_task_inputs()
    assert module.build_task_inputs(solver_type=None) == original
    selected = module.build_task_inputs(solver_type=solver)
    assert selected['scene']['isaacsim'].pop('solver_type') == solver
    provenance = selected['scene']['mechanical_task_provenance'].pop('solver_selection')
    assert provenance['requested'] == solver
    assert provenance['native_validation_claimed'] is False
    assert selected == original


@pytest.mark.parametrize('task_name,method', [
    ('parts_sorting', 'contact_probe_inputs'),
    ('peg_insertion', 'build_probe_inputs'),
    ('shaft_spacer_assembly', 'build_probe_inputs')])
def test_reset_only_probe_carries_the_same_explicit_solver_option(task_name, method):
    module = importlib.import_module('scenes.workrooms.mechanical_assembly.tasks.'+task_name+'.task')
    original = getattr(module, method)()
    selected = getattr(module, method)(solver_type='PGS')
    assert selected['scene']['isaacsim'].pop('solver_type') == 'PGS'
    selected['scene']['mechanical_task_provenance'].pop('solver_selection')
    assert selected == original
