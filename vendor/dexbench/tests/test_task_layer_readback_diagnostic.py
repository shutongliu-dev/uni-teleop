from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from bench.paths import load_json
from scripts.validation.task_layer_readback import read_without_step, review_trials
from bench.simulation.isaacsim_backend.env import gpu_tensor_pipeline
from bench.simulation.isaacsim_backend.task_readback import configure_particle_export


def trajectory():
    q = np.zeros((4, 2, 3))
    q[:, :, 2] = 1 - np.arange(4)[:, None]**2 * 1.e-5
    qd = np.zeros_like(q)
    qd[:, :, 2] = -np.arange(4)[:, None] * .005
    return {'native_physics_step_index': np.arange(2, 6),
        'native_world_time_s': np.arange(2, 6)*.0005,
        'task_particle_fluid_q': q, 'task_particle_fluid_qd': qd,
        'task_particle_fluid_ids': np.tile([11, 79], (4, 1)),
        'object_q': np.zeros((4, 1, 7))}


def compare(a, b):
    return review_trials(a, b, {'fluid': {'count': 2}}, dt=.0005,
        reset_tolerance_m=.0001, trajectory_tolerance_m=.001, velocity_tolerance_m_s=.02)


def test_numerical_variation_is_reported_without_requiring_bitwise_replay():
    a, b = trajectory(), trajectory()
    b['task_particle_fluid_q'][1:, :, 2] += 2.e-6
    b['task_particle_fluid_qd'][1:, :, 2] += .0002
    review = compare(a, b)
    assert review['reset_distribution_reproduced']
    assert review['trajectory_within_declared_tolerances']
    item = review['particle_sets']['fluid']
    assert item['both_trials_show_position_and_velocity_changes']
    assert item['trajectory_max_distance_m'] == pytest.approx(2.e-6)


@pytest.mark.parametrize('fault', ['previous_spill', 'reindexed', 'stale', 'skipped_clock'])
def test_reset_inventory_and_stale_readback_failures_are_visible(fault):
    a, b = trajectory(), trajectory()
    if fault == 'previous_spill':
        b['task_particle_fluid_q'][0, 0, 2] -= .1
    elif fault == 'reindexed':
        b['task_particle_fluid_ids'][2] = [79, 11]
    elif fault == 'stale':
        b['task_particle_fluid_q'][:] = b['task_particle_fluid_q'][0]
        b['task_particle_fluid_qd'][:] = b['task_particle_fluid_qd'][0]
    else:
        b['native_physics_step_index'][2:] += 1
    review = compare(a, b)
    if fault in ('previous_spill', 'reindexed'):
        assert not review['reset_distribution_reproduced']
    elif fault == 'stale':
        assert not review['particle_sets']['fluid']['both_trials_show_position_and_velocity_changes']
    else:
        assert not review['native_clocks_consecutive']


def test_readback_checks_actual_clocks_and_repeated_values():
    env = SimpleNamespace(world=SimpleNamespace(current_time_step_index=4, current_time=.002), time_s=.001)
    env.capture_state = lambda: {'q': np.array([.2, .3])}
    assert np.array_equal(read_without_step(env)['q'], [.2, .3])

    def secretly_step():
        env.world.current_time_step_index += 1
        return {'q': np.array([.2, .3])}

    env.capture_state = secretly_step
    with pytest.raises(RuntimeError, match='advanced'):
        read_without_step(env)

    samples = iter([{'q': np.array([.2])}, {'q': np.array([.3])}])
    env.capture_state = lambda: deepcopy(next(samples))
    with pytest.raises(RuntimeError, match='within one native tick'):
        read_without_step(env)


def test_particle_export_uses_explicit_host_transport_without_disabling_gpu_physics():
    values = {'/physics/suppressReadback': True}
    configure_particle_export(SimpleNamespace(set_bool=lambda key, value: values.__setitem__(key, value)))
    assert values['/physics/suppressReadback'] is False
    assert values['/physics/updateParticlesToUsd'] is True
    assert all('gpuDynamics' not in key and 'broadphase' not in key for key in values)
    view = SimpleNamespace(device='cuda:0', device_ordinal=0, param_device_ordinal=-1, cuda_context=123)
    assert gpu_tensor_pipeline(view, 'cuda:0')['native_parameter_device_ordinal'] == -1
    view.device, view.device_ordinal, view.cuda_context = 'cpu', -1, None
    with pytest.raises(RuntimeError, match='requested CUDA device'):
        gpu_tensor_pipeline(view, 'cuda:0')


def test_native_tick_cache_reads_host_once_and_refreshes_after_actual_step():
    from bench.simulation.isaacsim_backend.task_readback import TaskLayerReadback
    values = {'points': np.array([[1., 2., 3.]]),
              'velocities': np.array([[.1, .2, .3]]), 'ids': np.array([19])}
    readback = TaskLayerReadback.__new__(TaskLayerReadback)
    readback.body_paths, readback.joints = {}, []
    readback.particles = {'fluid': {'path': '/World/Fluid', 'count': 1}}
    readback.initial_particles = {'fluid': (None, None, values['ids'].copy())}
    readback._particle_cache = None
    readback._usd_particle_arrays = lambda item: tuple(values[key].copy() for key in ('points','velocities','ids'))
    syncs = []
    readback._synchronize_particles = lambda world: syncs.append(world.current_time_step_index)
    world = SimpleNamespace(current_time_step_index=2, current_time=.001)
    first = readback.capture_arrays(world, [], [], [])
    values['points'][0, 2] -= .001
    second = readback.capture_arrays(world, [], [], [])
    assert np.array_equal(first['task_particle_fluid_q'], second['task_particle_fluid_q'])
    world.current_time_step_index, world.current_time = 3, .0015
    third = readback.capture_arrays(world, [], [], [])
    assert third['task_particle_fluid_q'][0, 2] == pytest.approx(2.999)
    assert syncs == [2, 3]
    assert (world.current_time_step_index, world.current_time) == (3, .0015)



def test_host_profile_requires_explicit_particle_capability_and_no_silent_fallback():
    from bench.simulation.isaacsim_backend.readback_profile import select_profile
    assert select_profile({}) == 'cuda_tensors_v1'
    particle = {'task_layers': [{'particle_sets': {'fluid': {}}}]}
    with pytest.raises(ValueError, match='explicit gpu_solver_usd_host_v1'):
        select_profile(particle)
    particle['native_readback_profile'] = 'gpu_solver_usd_host_v1'
    assert select_profile(particle) == 'gpu_solver_usd_host_v1'
    with pytest.raises(ValueError, match='requires declared particle'):
        select_profile({'native_readback_profile': 'gpu_solver_usd_host_v1'})
    with pytest.raises(ValueError, match='Unknown'):
        select_profile({'native_readback_profile': 'auto'})


def test_native_tensor_transport_and_gpu_solver_are_separate_explicit_guards():
    from bench.simulation.isaacsim_backend.readback_profile import tensor_pipeline, validate_gpu_solver_configuration
    cpu = SimpleNamespace(device='cpu', device_ordinal=-1, param_device_ordinal=-1, cuda_context=None)
    assert tensor_pipeline(cpu, 'cuda:0', 'gpu_solver_usd_host_v1')['state_command_device'] == 'cpu'
    with pytest.raises(RuntimeError, match='requested CUDA device'):
        tensor_pipeline(cpu, 'cuda:0')
    cuda = SimpleNamespace(device='cuda:0', device_ordinal=0, param_device_ordinal=-1, cuda_context=123)
    with pytest.raises(RuntimeError, match='CPU state tensors'):
        tensor_pipeline(cuda, 'cuda:0', 'gpu_solver_usd_host_v1')
    values = {'gpu_dynamics_enabled': True, 'broadphase_type': 'GPU',
              'native_gpu_override': -1, 'actual_readback_suppressed': False}
    assert validate_gpu_solver_configuration(values, 'gpu_solver_usd_host_v1') == values
    for key, value in [('gpu_dynamics_enabled', False), ('broadphase_type', 'MBP'),
                       ('native_gpu_override', 0), ('actual_readback_suppressed', True)]:
        with pytest.raises(RuntimeError):
            validate_gpu_solver_configuration({**values, key: value}, 'gpu_solver_usd_host_v1')


def test_state_command_tensor_device_follows_actual_transport_not_solver_device():
    pytest.importorskip('torch')
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    env = FR3DexterousIsaacEnv.__new__(FR3DexterousIsaacEnv)
    env.device, env.tensor_device = 'cuda:0', 'cpu'
    value = env._tensor(np.array([1., 2.], dtype=np.float32))
    assert str(value.device) == 'cpu' and value.is_contiguous()
    assert value.numpy().dtype == np.float32
    indices = env._tensor(np.array([0], dtype=np.int32))
    assert indices.numpy().dtype == np.int32


def test_bundle_profile_projection_is_versioned_without_mutating_frozen_input():
    from bench.simulation.isaacsim_backend.readback_profile import resolve_profile
    scene = {'native_task_bundle': {'schema': 'native-usd-task-bundle-v1', 'particle_sets': {'water': {}}}}
    original = deepcopy(scene)
    result = resolve_profile(scene)
    assert result['profile'] == 'gpu_solver_usd_host_v1'
    assert result['resolution'] == 'native_bundle_host_readback_projection_v1'
    assert result['declared_profile'] is None and scene == original
    for schema, explicit in [('unrecognized-v2', None), ('native-usd-task-bundle-v1', 'cuda_tensors_v1')]:
        invalid = deepcopy(scene)
        invalid['native_task_bundle']['schema'] = schema
        if explicit is not None:
            invalid['native_readback_profile'] = explicit
        with pytest.raises(ValueError, match='explicit gpu_solver_usd_host_v1'):
            resolve_profile(invalid)


def test_rigid_canary_rejects_stale_fabric_pose_and_invalid_orientation():
    from bench.simulation.isaacsim_backend.task_readback import validate_rigid_synchronization
    native = [1., 2., 3., 0., 0., 0., 1.]
    validate_rigid_synchronization('/World/Body', native, native[:3], [0, 0, 0, -1])
    for position, quat in [([1, 2, 2.999], [0, 0, 0, 1]), ([1, 2, 3], [0, 0, .001, 1]),
                           ([1, 2, 3], [0, 0, 0, 0])]:
        with pytest.raises(RuntimeError, match='synchronization'):
            validate_rigid_synchronization('/World/Body', native, position, quat)


def test_bundle_hold_samples_after_native_observer_refresh():
    from scripts.validation.task_layer_readback import hold_trial
    env = SimpleNamespace(world=SimpleNamespace(current_time_step_index=0, current_time=0.), time_s=0.,
                          dt=.0005, _targets=np.array([[.2]]), joint_names=['joint'], actuated_joint_names=['joint'],
                          _task_reader=object())
    observed = {'value': 0}
    env._sample_task_state = lambda contacts: observed.update(value=env.world.current_time_step_index)
    env.capture_state = lambda: {'native_physics_step_index': np.array(observed['value']),
        'native_world_time_s': np.array(env.world.current_time)}
    def step(action):
        for _ in range(40):
            env.world.current_time_step_index += 1
            env.world.current_time += env.dt
            env.time_s += env.dt
            env._sample_task_state([])
    env.step = step
    states, action = hold_trial(env, 1)
    assert np.array_equal(states['native_physics_step_index'], np.arange(41))
    assert action == {'joint_positions_rad': {'joint': .2}, 'duration_s': .02}


@pytest.mark.parametrize('bad_time', [False, True])
def test_hold_archives_actual_clock_and_uses_environment_float_tolerance(tmp_path, bad_time):
    from scripts.validation.task_layer_readback import hold_trial
    import json
    env = SimpleNamespace(world=SimpleNamespace(current_time_step_index=0, current_time=0.), time_s=0.,
                          dt=.0005, _targets=np.array([[.2]]), joint_names=['joint'], actuated_joint_names=['joint'])
    env._read_rigid_contacts = lambda: None
    env.capture_state = lambda: {'native_physics_step_index': np.array(env.world.current_time_step_index),
                                'native_world_time_s': np.array(env.world.current_time)}
    def step(action):
        for _ in range(40):
            env.world.current_time_step_index += 1
            env.world.current_time += float(np.float32(env.dt)) + (1.e-6 if bad_time else 0.)
            env.time_s += env.dt
            env._read_rigid_contacts()
    env.step = step
    archive = tmp_path/'trial.npz'
    if bad_time:
        with pytest.raises(RuntimeError, match='observed='):
            hold_trial(env, 3, archive_path=archive)
    else:
        state, _ = hold_trial(env, 3, archive_path=archive)
        assert len(state['native_physics_step_index']) == 121
    assert archive.is_file()
    data = load_json(archive.with_suffix('.clock.json'))
    assert data['elapsed_tolerance_s'] == pytest.approx(3.e-8)
    assert data['observed_samples'] == 121
    if not bad_time:
        assert data['elapsed_error_s'] > 1.e-9
