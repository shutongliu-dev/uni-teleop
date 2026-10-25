"""A CPU solver is explicit and cannot replace GPU-only scene capabilities."""
from types import SimpleNamespace

import pytest

from bench.simulation.isaacsim_backend.readback_profile import (
    CPU_RIGID, CUDA_TENSORS, select_profile, tensor_pipeline, validate_gpu_solver_configuration,
)


def test_cpu_profile_is_explicit_and_does_not_change_default():
    assert select_profile({}) == CUDA_TENSORS
    assert select_profile({'native_readback_profile': CPU_RIGID}) == CPU_RIGID


@pytest.mark.parametrize('capability', [
    {'cloth': {'enabled': True}}, {'soft_bodies': [{'id': 'soft'}]},
    {'task_layers': [{'particle_sets': {'fluid': {}}}]},
    {'native_task_bundle': {'schema': 'native-usd-task-bundle-v1', 'particle_sets': {'fluid': {}}}},
])
def test_cpu_profile_rejects_gpu_only_capabilities(capability):
    with pytest.raises(ValueError, match='CPU rigid profile cannot simulate'):
        select_profile({'native_readback_profile': CPU_RIGID, **capability})


def test_cpu_profile_checks_native_buffers_and_actual_solver():
    view = SimpleNamespace(device='cpu', device_ordinal=-1, param_device_ordinal=-1, cuda_context=None)
    assert tensor_pipeline(view, 'cuda:0', CPU_RIGID)['state_command_device'] == 'cpu'
    values = {'gpu_dynamics_enabled': False, 'broadphase_type': 'MBP',
              'native_gpu_override': -1, 'actual_readback_suppressed': False}
    assert validate_gpu_solver_configuration(values, CPU_RIGID) == values
    for key, value in [('gpu_dynamics_enabled', True), ('broadphase_type', 'GPU'),
                       ('native_gpu_override', 1), ('actual_readback_suppressed', True)]:
        with pytest.raises(RuntimeError, match='CPU rigid profile requires'):
            validate_gpu_solver_configuration({**values, key: value}, CPU_RIGID)
    view.device, view.device_ordinal = 'cuda:0', 0
    with pytest.raises(RuntimeError, match='CPU state tensors'):
        tensor_pipeline(view, 'cuda:0', CPU_RIGID)


def test_sandbox_declares_cpu_but_released_task_does_not():
    from scenes.teleop_sandbox import inputs
    from scenes.workrooms.catalog import load_task
    assert select_profile(inputs()[0]) == CPU_RIGID
    assert select_profile(load_task('kitchen', 'breakfast_serving')['scene']) != CPU_RIGID


def test_operator_cpu_override_changes_only_runtime_profile():
    from scripts.teleop.pico_isaac import parser, load_inputs
    options = parser()
    original, original_initial, release = load_inputs(options.parse_args([]))
    selected, initial, selected_release = load_inputs(options.parse_args(['--physics-profile', CPU_RIGID]))
    assert selected.pop('native_readback_profile') == CPU_RIGID
    assert selected == original
    assert initial == original_initial and release == selected_release
    assert select_profile(load_inputs(options.parse_args([]))[0]) == CUDA_TENSORS


def test_operator_override_rejects_particles_without_mutating_scene(monkeypatch):
    from scripts.teleop.pico_isaac import parser, load_inputs
    from scenes import teleop_sandbox
    scene = {'native_task_bundle': {'particle_sets': {'fluid': {}}}}
    monkeypatch.setattr(teleop_sandbox, 'inputs', lambda: (scene, {}, None))
    with pytest.raises(ValueError, match='CPU rigid profile cannot simulate'):
        load_inputs(parser().parse_args(['--sandbox', '--physics-profile', CPU_RIGID]))
    assert 'native_readback_profile' not in scene


def test_unscored_practice_preserves_scene_and_explicitly_disables_observer():
    from scripts.teleop.pico_isaac import parser, load_inputs
    from bench.simulation.isaacsim_backend.native_task_bundle import task_observer_enabled
    original, _, _ = load_inputs(parser().parse_args([]))
    practice, _, _ = load_inputs(parser().parse_args(['--no-task-scoring']))
    assert task_observer_enabled(original)
    assert not task_observer_enabled(practice)
    assert practice.pop('native_task_observer_enabled') is False
    assert practice == original


@pytest.mark.parametrize('capability', ['particle_sets', 'reduced_dynamics', 'visual_state_bindings', 'volume_visual_bindings'])
def test_unscored_practice_cannot_remove_simulation_dependencies(capability):
    from bench.simulation.isaacsim_backend.native_task_bundle import task_observer_enabled
    scene = {'native_task_observer_enabled': False, 'native_task_bundle': {capability: {'model': 'required'}}}
    with pytest.raises(ValueError, match='Task observation is required'):
        task_observer_enabled(scene)
