"""A practical observation must not turn disjoint contacts into a drive chain."""
from types import SimpleNamespace

import numpy as np

from scripts.validation.gear_practical_observer import PracticalGoalObserver
from scripts.validation.gear_transmission_checks import probe_contract, positive_contact


def test_native_steps_and_disjoint_contact_times_are_preserved(tmp_path, monkeypatch):
    import scripts.validation.gear_practical_observer as module
    monkeypatch.setattr(module, 'declared_mount_pairs', lambda env: [])
    names = ['input_gear', 'idler_gear', 'output_gear', 'socket_crank']
    paths = {n: {'path': '/'+n} for n in names}
    force_calls, captures = [], []
    world = SimpleNamespace(current_time_step_index=0)
    world.step = lambda **kw: setattr(world, 'current_time_step_index', world.current_time_step_index+1)
    view = SimpleNamespace(get_velocities=lambda: np.zeros((1, 6)),
        apply_forces_and_torques_at_position=lambda *a: force_calls.append(a))
    env = SimpleNamespace(world=world, dt=.0005, time_s=0.,
        scene_info={'objects': paths, 'robot_link_paths': {}}, object_names=names,
        _object_views={'socket_crank': view}, _tensor=np.asarray, _one=np.array([0]))
    def state():
        captures.append(True)
        return {'object_q': np.tile([0., 0., .802, 0., 0., 0., 1.], (4, 1)),
                'object_qd': np.zeros((4, 6))}
    env.capture_state = state
    observer = PracticalGoalObserver(env, tmp_path, probe_contract(), loaded_case=True, goal_contract={})
    for i in range(40):
        env._current_step_time = (i+1)*env.dt
        env.world.step(render=False)
    env.time_s = .02
    env._contact_sampling = {'complete': True, 'sampled_substeps': 40}
    env._contact_reports = [dict(actor0='/'+a, actor1='/'+b, time_s=(i+1)*env.dt,
        impulse_magnitude_sum_ns=.01, raw_scalar_abs_impulse_sum_ns=.01,
        impulse_vector_ns=[0., 0., .01], min_separation_m=-1.e-6)
        for i, (a, b) in enumerate((('socket_crank', 'input_gear'),
            ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear')))]
    observer.after_control()
    row = observer.rows[-1]
    assert world.current_time_step_index == 40 and len(captures) == 1 and not force_calls
    assert all(positive_contact(row, '/'+a, '/'+b) for a, b in (
        ('socket_crank', 'input_gear'), ('input_gear', 'idler_gear'), ('idler_gear', 'output_gear')))
    assert row['simultaneous_drive_chain_samples'] == 0
    # A stronger engineering load still wakes and drives only the real tool;
    # release must discard the integral and make no new force or wake call.
    wakes = []
    view.count, view.wake_up = 1, lambda: wakes.append(True)
    observer.contract.update(wake_loaded_tool=True, maximum_torque_nm=2.8,
        integral_load_coefficient_nm_per_rad=8.)
    observer.goal_contract = {'target_rad': np.deg2rad(30.), 'cutoff_tolerance_rad': np.deg2rad(3.)}
    observer.phase_definition = {'id': 'goal_drive', 'loaded': True, 'target_speed_rad_s': np.deg2rad(15.)}
    env._current_step_time += env.dt
    env.world.step(render=False)
    assert len(wakes) == len(force_calls) == 1
    assert abs(force_calls[-1][1][0, 2]) <= 2.8 and observer.integral_tool_torque_nm > 0.
    observer.phase_definition = {'id': 'release', 'loaded': False, 'target_speed_rad_s': 0.}
    env._current_step_time += env.dt
    env.world.step(render=False)
    assert len(wakes) == len(force_calls) == 1 and observer.integral_tool_torque_nm == 0.
    assert observer.close()['sampled_native_steps'] == 40
