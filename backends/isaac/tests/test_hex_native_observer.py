"""CPU adapter tests with explicit native-view doubles; not physics evidence."""
from types import SimpleNamespace

import numpy as np
import pytest

from scenes.board_games.hex import HexMatch, HexNativeObserver

BOARD = '/World/Fixtures/hex_board/collision_000'
ROBOT = '/World/Robot/left/left_thumb'


class BatchView:
    def __init__(self, match):
        # Reverse the IDs so accidental order-based association cannot pass.
        self.names = sorted(match.inventory, reverse=True)
        self.prim_paths = ['/World/Objects/'+name for name in self.names]
        self.count = len(self.names)
        self.transforms = np.array([[*match.geometry.reserve_position(name.split('_')[0], int(name[-2:])),
                                     0., 0., 0., 1.] for name in self.names])
        self.velocities = np.zeros((50, 6))
        self.coms = np.tile([0., 0., .01, 0., 0., 0., 1.], (50, 1))
        self.transform_reads = self.velocity_reads = self.com_reads = 0

    def get_coms(self):
        self.com_reads += 1
        return self.coms

    def get_transforms(self):
        self.transform_reads += 1
        return self.transforms

    def get_velocities(self):
        self.velocity_reads += 1
        return self.velocities


class FakeEnv:
    def __init__(self, match):
        self._reset_count = 1
        self.dt = .01
        self.time_s = 0.
        self.world = SimpleNamespace(current_time_step_index=10, current_time=1.)
        self.batch = BatchView(match)
        self.created_views = []
        self._sim_view = SimpleNamespace(create_rigid_body_view=self.create_view)
        self.scene_info = {
            'objects': {name: {'path': '/World/Objects/'+name} for name in match.inventory},
            'fixtures': {'hex_board': {'collision_paths': [BOARD]}},
            'robot_link_paths': {'left_thumb': ROBOT, 'right_thumb': '/World/Robot/right/right_thumb'}}
        self._contact_sensor_paths = list(self.batch.prim_paths)
        self._contact_filter_paths = [[BOARD, ROBOT, '/World/Robot/right/right_thumb']
                                      for _ in self.batch.prim_paths]
        self._rigid_contacts = object()
        self._contact_reports = []
        self._contact_errors = []
        self._contact_sampling = {'sampled_substeps': 0}
        self.acquisitions = 0
        self.next_rows = []

    def create_view(self, pattern):
        self.created_views.append(pattern)
        return self.batch

    def _read_rigid_contacts(self):
        self.acquisitions += 1
        self._contact_reports.extend([{**row, 'time_s': self.time_s} for row in self.next_rows])
        self._contact_sampling.update(sampled_substeps=self._contact_sampling['sampled_substeps']+1,
                                      native_end_step_index=self.world.current_time_step_index,
                                      end_sim_time_s=self.time_s)

    def advance(self):
        self.time_s += self.dt
        self.world.current_time += self.dt
        self.world.current_time_step_index += 1
        self._read_rigid_contacts()

    def place(self, name, position):
        self.batch.transforms[self.batch.names.index(name), :3] = position


def contact(name='red_00', other=BOARD, **changes):
    return {'actor0': '/World/Objects/'+name, 'actor1': other,
            'impulse_magnitude_sum_ns': .01, 'min_separation_m': -.0001, **changes}


def test_adapter_batches_once_per_substep_reuses_original_contact_and_commits():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    env = FakeEnv(match)
    env.place('red_00', match.geometry.cell_center(3, 3))
    env.next_rows = [contact()]
    observer = HexNativeObserver(env, match).attach()
    assert env.created_views == ['/World/Objects/*']
    for _ in range(31):
        env.advance()
    assert observer.last_result.committed
    assert observer.commits == 1
    assert env.acquisitions == 31
    assert env.batch.transform_reads == env.batch.velocity_reads == 31
    assert env.batch.com_reads == 1
    assert observer.last_frame.pieces['red_00'].position_m == pytest.approx(match.geometry.cell_center(3, 3))
    assert observer.last_frame.pieces['red_00'].maximum_penetration_m == .0001
    summary = observer.close()
    assert not summary['failed']
    assert summary['closed']
    assert summary['final_context_verified'] is True
    assert all(summary['closure_checks'].values())
    assert summary['additional_contact_acquisitions'] == 0
    assert '_read_rigid_contacts' not in env.__dict__
    env.advance()
    assert observer.samples == 31
    assert env.acquisitions == 32


def test_adapter_missing_separation_is_not_zero_and_cannot_commit():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    env = FakeEnv(match)
    env.place('red_00', match.geometry.cell_center(3, 3))
    row = contact()
    del row['min_separation_m']
    env.next_rows = [row]
    observer = HexNativeObserver(env, match).attach()
    env.advance()
    assert observer.last_frame.contacts_complete is False
    assert observer.last_frame.pieces['red_00'].maximum_penetration_m is None
    assert observer.unsupported_samples == 1
    assert not observer.last_result.committed
    assert not match.rules.board
    observer.close()


def test_adapter_no_contact_is_a_complete_sample_but_not_support():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    env = FakeEnv(match)
    env.place('red_00', match.geometry.cell_center(3, 3))
    observer = HexNativeObserver(env, match).attach()
    env.advance()
    assert observer.last_frame.contacts_complete
    assert observer.last_frame.pieces['red_00'].maximum_penetration_m == 0.
    assert observer.last_result.reason == 'native_board_support_required'
    observer.close()


def test_adapter_robot_contact_both_actor_directions_and_com_velocity():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    env = FakeEnv(match)
    env.place('red_00', match.geometry.cell_center(3, 3))
    env.next_rows = [contact(), contact(actor0=ROBOT, actor1='/World/Objects/red_00')]
    index = env.batch.names.index('red_00')
    # Omega cross COM = (0,-.01,0); a stationary origin therefore has this COM velocity.
    env.batch.velocities[index] = [0., -.01, 0., 1., 0., 0.]
    observer = HexNativeObserver(env, match).attach()
    env.advance()
    evidence = observer.last_frame.pieces['red_00']
    assert evidence.linear_velocity_m_s == pytest.approx((0., 0., 0.))
    assert evidence.all_robot_contact_impulse_ns == .01
    assert observer.last_result.reason == 'robot_must_release_piece'
    observer.close()


@pytest.mark.parametrize('mutation', ['reset', 'step_gap', 'mapping', 'dt'])
def test_adapter_stale_native_context_fails_closed(mutation):
    match = HexMatch()
    env = FakeEnv(match)
    observer = HexNativeObserver(env, match).attach()
    if mutation == 'reset':
        env._reset_count += 1
    elif mutation == 'step_gap':
        env.world.current_time_step_index += 1
    elif mutation == 'mapping':
        env._contact_filter_paths[0].append('/other')
    else:
        env.dt = .005
    with pytest.raises(RuntimeError, match='stale, skipped or mismatched'):
        env.advance()
    assert observer.summary()['failed']
    assert observer.samples == 0
    observer.close()


def test_adapter_checks_board_filter_coverage_before_attach():
    match = HexMatch()
    env = FakeEnv(match)
    env._contact_filter_paths[0].remove(BOARD)
    with pytest.raises(RuntimeError, match='do not identify'):
        HexNativeObserver(env, match)
    assert '_read_rigid_contacts' not in env.__dict__


def test_adapter_slices_only_new_rows_when_control_window_reports_accumulate():
    match = HexMatch()
    match.begin_move('player_0', 'red_00', 3, 3)
    env = FakeEnv(match)
    env.place('red_00', match.geometry.cell_center(3, 3))
    env.next_rows = [contact()]
    observer = HexNativeObserver(env, match).attach()
    env.advance()
    env.advance()
    assert len(env._contact_reports) == 2
    assert observer.last_frame.pieces['red_00'].board_support_impulse_ns == .01
    assert observer.summary()['reported_contact_pairs'] == 2
    observer.close()


@pytest.mark.parametrize('mutation,check', [
    ('unobserved_step', 'no_unobserved_native_step'),
    ('public_time', 'same_public_time'),
    ('native_time', 'same_native_time'),
    ('reset', 'same_reset_epoch'),
    ('mapping', 'same_contact_mapping'),
    ('dt', 'same_physics_dt'),
    ('contact_view', 'same_native_contact_view'),
    ('token_mapping', 'same_token_prim_mapping'),
    ('sampling', 'acquisition_ends_at_last_native_step'),
])
def test_final_context_seal_rejects_changes_after_last_observed_sample(mutation, check):
    match = HexMatch()
    env = FakeEnv(match)
    observer = HexNativeObserver(env, match).attach()
    env.advance()
    if mutation == 'unobserved_step':
        env.world.current_time_step_index += 1
        env.world.current_time += env.dt
        env.time_s += env.dt
    elif mutation == 'public_time':
        env.time_s += env.dt
    elif mutation == 'native_time':
        env.world.current_time += env.dt
    elif mutation == 'reset':
        env._reset_count += 1
    elif mutation == 'mapping':
        env._contact_filter_paths[0].append('/other')
    elif mutation == 'dt':
        env.dt = .005
    elif mutation == 'contact_view':
        env._rigid_contacts = object()
    elif mutation == 'token_mapping':
        env.batch.prim_paths.reverse()
    else:
        env._contact_sampling['native_end_step_index'] += 1
    acquisitions = env.acquisitions
    summary = observer.close()
    assert summary['failed'] is True
    assert summary['final_context_verified'] is False
    assert summary['closure_checks'][check] is False
    assert summary['closed'] is True
    assert '_read_rigid_contacts' not in env.__dict__
    assert not hasattr(env, '_hex_observer')
    assert env.acquisitions == acquisitions
    assert observer.close() == summary
