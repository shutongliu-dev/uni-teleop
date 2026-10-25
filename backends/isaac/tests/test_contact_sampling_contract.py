"""CPU contract checks for acquisition accounting, not native physics acceptance."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv


def fake_env():
    env = object.__new__(FR3DexterousIsaacEnv)
    env.dt, env.time_s, env.last_duration_s = .0005, 0., 0.
    env.terminated = False
    env.world = SimpleNamespace(current_time_step_index=3, current_time=.0015,
                                is_playing=lambda: True)
    env._contact_reports, env._contact_errors = [], []
    env._robot_view = SimpleNamespace(set_dof_position_targets=lambda *args: None)
    env._one, env.joint_names = None, []
    env._validate_action = lambda action: None
    env._expand_mimic_targets = lambda positions: positions
    env._tensor = lambda array: array
    env._project_rigid_contacts = lambda: []
    env.observation = lambda: deepcopy(env._contact_sampling)
    def step(render=False):
        env.world.current_time_step_index += 1
        env.world.current_time += env.dt
    env.world.step = step
    return env


def test_empty_contacts_still_account_for_each_actual_acquisition():
    env = fake_env()
    env._begin_contact_sampling(baseline=True)
    assert env._contact_sampling['sampled_substeps'] == 0
    result = env.step({'joint_positions_rad': {}})
    assert result['complete'] is True
    assert result['sampled_substeps'] == 40
    assert result['native_start_step_index'] == 3
    assert result['native_end_step_index'] == 43
    assert result['native_end_time_s'] - result['native_start_time_s'] == pytest.approx(.02)
    assert result['end_sim_time_s'] == pytest.approx(.02)
    assert env._contact_reports == []


def test_missing_acquisition_cannot_claim_expected_substep_count():
    env = fake_env()
    env._read_rigid_contacts = lambda: None
    with pytest.raises(RuntimeError, match='sampling missed'):
        env.step({'joint_positions_rad': {}})
    assert env._contact_sampling['sampled_substeps'] == 0
    assert env._contact_sampling['complete'] is False


def test_failed_acquisition_never_increments_counter_or_completes_window():
    env = fake_env()
    def failure():
        raise RuntimeError('contact tensor failed')
    env._project_rigid_contacts = failure
    with pytest.raises(RuntimeError, match='contact tensor failed'):
        env.step({'joint_positions_rad': {}})
    assert env._contact_sampling['sampled_substeps'] == 0
    assert env._contact_sampling['complete'] is False


def test_clock_must_remain_unchanged_during_the_actual_acquisition():
    env = fake_env()
    def advancing_read():
        env.world.step()
        return []
    env._project_rigid_contacts = advancing_read
    with pytest.raises(RuntimeError, match='clock changed while reading'):
        env.step({'joint_positions_rad': {}})
    assert env._contact_sampling['sampled_substeps'] == 0
    assert env._contact_sampling['complete'] is False


@pytest.mark.parametrize('damage', ['duplicate', 'native_skip', 'native_clock', 'public_clock'])
def test_sample_requires_one_matching_native_and_public_substep(damage):
    env = fake_env()
    env._begin_contact_sampling()
    env.world.step()
    env.time_s = env._current_step_time = .0005
    env._read_rigid_contacts()
    if damage == 'native_skip':
        env.world.current_time_step_index += 2
        env.world.current_time += env.dt
        env.time_s = env._current_step_time = .001
    elif damage == 'native_clock':
        env.world.step()
        env.world.current_time += .0001
        env.time_s = env._current_step_time = .001
    elif damage == 'public_clock':
        env.world.step()
        env.time_s = .001
    with pytest.raises(RuntimeError, match='acquisition clock'):
        env._read_rigid_contacts()
    assert env._contact_sampling['sampled_substeps'] == 1
    assert env._contact_sampling['complete'] is False


@pytest.mark.parametrize('value', [float('nan'), float('inf'), .000500002])
def test_scalar_clock_check_still_rejects_nonfinite_or_outside_tolerance(value):
    env = fake_env()
    env._begin_contact_sampling()
    env.world.step()
    env.time_s = env._current_step_time = value
    with pytest.raises(RuntimeError, match='acquisition clock'):
        env._read_rigid_contacts()
    assert env._contact_sampling['sampled_substeps'] == 0
