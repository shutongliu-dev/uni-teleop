"""Private Board-game observer attached after each existing native contact acquisition.

No simulator imports, additional physics steps, contact acquisitions or setters.
One all-token RigidBodyView supplies poses/velocities; its COM parameters are read
once on attachment so native COM velocities can be translated to base origins.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .settlement import BoardGameMatch, NativeFrame, PieceEvidence
from bench.paths import OUTPUT_ROOT

DEFAULT_BOARD_CONTACT_PATH = '/World/Fixtures/board/collision_000'


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value).copy()


def _rotate_xyzw(quaternion, vector):
    x, y, z, w = quaternion
    q = np.asarray((x, y, z))
    return vector+2*np.cross(q, np.cross(q, vector)+w*vector)


class BoardGameNativeObserver:
    """Construct after reset, call attach once, close before the next reset.

    The original `_read_rigid_contacts` remains the sole contact acquisition.
    Incomplete/unsupported evidence is recorded and cannot commit a move.
    Exceptions fail closed and are propagated to the native caller.
    """

    def __init__(self, env, match: BoardGameMatch, *, board_fixture_id='board', board_contact_paths=None,
                 frame_jsonl: str | Path | None = None):
        self.env, self.match = env, match
        self.token_count = len(match.inventory)
        if not isinstance(board_fixture_id, str) or not board_fixture_id or "/" in board_fixture_id:
            raise ValueError("Board fixture ID must be a single nonempty prim name")
        self.board_fixture_id = board_fixture_id
        self.epoch = env._reset_count
        self.dt = float(env.dt)
        self.last_step = int(env.world.current_time_step_index)
        self.last_time = float(env.time_s)
        self.last_native_time = float(env.world.current_time)
        self._view = env._sim_view.create_rigid_body_view('/World/Objects/*')
        self._paths = tuple(self._view.prim_paths)
        objects = env.scene_info['objects']
        self._piece_path = {name: str(objects[name]['path']) for name in match.inventory}
        self._path_piece = {path: name for name, path in self._piece_path.items()}
        if (self._view.count != self.token_count or len(self._paths) != self.token_count
                or len(set(self._paths)) != self.token_count
                or set(self._paths) != set(self._path_piece)):
            raise RuntimeError('Board-game batch rigid view must map exactly the authored token inventory')
        self._coms = _array(self._view.get_coms())
        if self._coms.shape != (self.token_count, 7) or not np.isfinite(self._coms).all():
            raise RuntimeError('Missing finite native token centres of mass')
        fixture = env.scene_info.get('fixtures', {}).get(self.board_fixture_id, {})
        if board_contact_paths is None:
            board_contact_paths = fixture.get('contact_filter_paths', fixture.get('collision_paths',
                                                                               ['/World/Fixtures/'+self.board_fixture_id+'/collision_000']))
        self.board_contact_paths = tuple(board_contact_paths)
        board_root = '/World/Fixtures/'+self.board_fixture_id
        if not self.board_contact_paths or any(path != board_root and not str(path).startswith(board_root+'/')
                                              for path in self.board_contact_paths):
            raise RuntimeError('Explicit Board-game board collider/actor contact identities are required')
        self._mapping = self._contact_mapping()
        sensors, filters = self._mapping
        for path in self._paths:
            if path not in sensors:
                raise RuntimeError('Native contact sensor missing token: '+path)
            if not set(self.board_contact_paths).issubset(filters[sensors.index(path)]):
                raise RuntimeError('Native token contact filters do not identify the Board-game board')
        self._robot_paths = frozenset(env.scene_info['robot_link_paths'].values())
        if not self._robot_paths or any(not path.startswith('/World/Robot/') for path in self._robot_paths):
            raise RuntimeError('Native robot contact paths are missing or not under /World/Robot')
        # Either sensor direction may contribute a pair. Require coverage from
        # each token sensor to both complete native robot link inventories.
        for path in self._paths:
            if not self._robot_paths.issubset(filters[sensors.index(path)]):
                raise RuntimeError('Native token filters do not cover both robots')
        self._contact_view = env._rigid_contacts
        self._original = env._read_rigid_contacts
        self._previous_instance_reader = env.__dict__.get('_read_rigid_contacts')
        self._wrapper = None
        self._attached = self._closed = self._failed = False
        self._closure_checks = None
        self.samples = self.commits = self.unsupported_samples = 0
        self.reported_contact_pairs = self.reported_separations = 0
        self.minimum_reported_separation_m = None
        self.last_frame: NativeFrame | None = None
        self.last_result = None
        self.reasons = Counter()
        self._stream = None
        self.frame_jsonl = None
        if frame_jsonl is not None:
            path = Path(frame_jsonl).resolve()
            if not path.is_relative_to(OUTPUT_ROOT):
                raise ValueError('Private Board-game frames belong under shared datasets')
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open('x')
            self.frame_jsonl = str(path)

    def _contact_mapping(self):
        return (tuple(self.env._contact_sensor_paths),
                tuple(tuple(paths) for paths in self.env._contact_filter_paths))

    def attach(self):
        if self._attached or self._closed or getattr(self.env, '_board_game_observer', None) is not None:
            raise RuntimeError('Board-game native observer already attached or closed')

        def after_original_acquisition():
            if self._failed:
                raise RuntimeError('Board-game native observer has failed')
            try:
                start = len(self.env._contact_reports)
                sampled_before = self.env._contact_sampling['sampled_substeps']
                value = self._original()
                self._observe(self.env._contact_reports[start:], sampled_before)
                return value
            except BaseException:
                self._failed = True
                raise

        self._wrapper = after_original_acquisition
        self.env._board_game_observer = self
        self.env._read_rigid_contacts = self._wrapper
        self._attached = True
        return self

    def _observe(self, rows, sampled_before):
        env = self.env
        step, now, native_now = (int(env.world.current_time_step_index),
                                 float(env.time_s), float(env.world.current_time))
        sampling = env._contact_sampling
        if (env._reset_count != self.epoch or env._rigid_contacts is not self._contact_view
                or self._contact_mapping() != self._mapping or float(env.dt) != self.dt
                or tuple(self._view.prim_paths) != self._paths
                or step != self.last_step+1 or not math.isclose(now-self.last_time, self.dt, abs_tol=1.e-9)
                or not math.isclose(native_now-self.last_native_time, self.dt, abs_tol=1.e-9)
                or sampling['sampled_substeps'] != sampled_before+1
                or sampling['native_end_step_index'] != step
                or not math.isclose(float(sampling['end_sim_time_s']), now, abs_tol=1.e-9)):
            raise RuntimeError('Board-game observer received stale, skipped or mismatched native acquisition')
        transforms = _array(self._view.get_transforms())
        velocities = _array(self._view.get_velocities())
        if (transforms.shape != (self.token_count, 7) or velocities.shape != (self.token_count, 6)
                or not np.isfinite(transforms).all() or not np.isfinite(velocities).all()
                or np.any(np.abs(np.sum(transforms[:, 3:]**2, axis=1)-1) > 1.e-3)):
            raise RuntimeError('Invalid batched native token state')
        if (int(env.world.current_time_step_index) != step or float(env.time_s) != now
                or float(env.world.current_time) != native_now):
            raise RuntimeError('Native clock changed during Board-game state read')
        support = {name: 0. for name in self.match.inventory}
        robot = dict(support)
        # Zero means no negative separation among this complete set of reported
        # contacts. It never means a certificate of no geometric intersection.
        penetration = dict(support)
        unsupported = set()
        seen = set()
        for row in rows:
            pair = (row.get('actor0'), row.get('actor1'))
            if (not all(isinstance(path, str) for path in pair) or pair[0] == pair[1]
                    or tuple(sorted(pair)) in seen or row.get('time_s') != now):
                raise RuntimeError('Invalid, duplicated or stale native contact row')
            seen.add(tuple(sorted(pair)))
            names = [self._path_piece[path] for path in pair if path in self._path_piece]
            self.reported_contact_pairs += 1
            separation = row.get('min_separation_m')
            if not isinstance(separation, (int, float)) or not math.isfinite(separation):
                unsupported.add('reported_contact_separation_unavailable')
                for name in names:
                    penetration[name] = None
            else:
                self.reported_separations += 1
                self.minimum_reported_separation_m = (float(separation) if self.minimum_reported_separation_m is None
                    else min(self.minimum_reported_separation_m, float(separation)))
                for name in names:
                    if penetration[name] is not None:
                        penetration[name] = max(penetration[name], -float(separation))
            impulse = row.get('impulse_magnitude_sum_ns')
            if not isinstance(impulse, (int, float)) or not math.isfinite(impulse) or impulse < 0:
                unsupported.add('reported_contact_impulse_unavailable')
                continue
            for index, path in enumerate(pair):
                name = self._path_piece.get(path)
                if name is None:
                    continue
                other = pair[1-index]
                if other in self.board_contact_paths:
                    support[name] += float(impulse)
                if other in self._robot_paths:
                    robot[name] += float(impulse)
                elif other.startswith('/World/Robot/'):
                    unsupported.add('robot_contact_actor_not_in_native_inventory')
        if env._contact_errors:
            unsupported.add('native_contact_acquisition_errors')
        pieces = {}
        for index, path in enumerate(self._paths):
            name = self._path_piece[path]
            transform, velocity = transforms[index], velocities[index]
            quaternion = transform[3:]/np.linalg.norm(transform[3:])
            offset_world = _rotate_xyzw(quaternion, self._coms[index, :3])
            origin_velocity = velocity[:3]-np.cross(velocity[3:], offset_world)
            pieces[name] = PieceEvidence(
                tuple(float(v) for v in transform[:3]),
                tuple(float(v) for v in quaternion[[3, 0, 1, 2]]),
                tuple(float(v) for v in origin_velocity), tuple(float(v) for v in velocity[3:]),
                support[name], robot[name], penetration[name])
        frame = NativeFrame(step, now, self.dt, pieces, not unsupported)
        result = self.match.observe(frame)
        self.samples += 1
        self.commits += int(result.committed)
        self.unsupported_samples += int(bool(unsupported))
        self.reasons[result.reason] += 1
        self.last_frame, self.last_result = frame, result
        self.last_step, self.last_time, self.last_native_time = step, now, native_now
        if self._stream is not None:
            self._stream.write(json.dumps({'frame': asdict(frame), 'result': asdict(result),
                                           'unsupported': sorted(unsupported)}, allow_nan=False)+'\n')

    def summary(self) -> dict:
        return {'schema': 'board_game_native_observer.v1', 'attached': self._attached,
                'closed': self._closed, 'failed': self._failed, 'reset_epoch': self.epoch,
                'samples': self.samples, 'commits': self.commits,
                'unsupported_samples': self.unsupported_samples,
                'reported_contact_pairs': self.reported_contact_pairs,
                'reported_separations': self.reported_separations,
                'minimum_reported_separation_m': self.minimum_reported_separation_m,
                'penetration_scope': 'reported_native_contact_separations_not_full_mesh_certificate',
                'board_contact_paths': list(self.board_contact_paths),
                'batch_view_pattern': '/World/Objects/*', 'batch_view_token_count': self.token_count,
                'game_id': self.match.geometry.game_id, 'board_fixture_id': self.board_fixture_id,
                'velocity_frame': 'world_base_origin_corrected_from_native_COM_velocity',
                'physical_steps_added': 0, 'additional_contact_acquisitions': 0,
                'physical_state_setters_called': 0, 'reasons': dict(self.reasons),
                'last_result': asdict(self.last_result) if self.last_result else None,
                'final_context_verified': (None if self._closure_checks is None
                                           else all(self._closure_checks.values())),
                'closure_checks': self._closure_checks,
                'frame_jsonl': self.frame_jsonl}

    def _seal_final_context(self):
        """Detect a final skipped acquisition or context change before detaching.

        A next-sample guard alone cannot observe a reset or direct native advance
        that occurs after the last delivered sample. This seal reads clocks and
        identities only; it acquires no physics/contact state and advances none.
        """
        env = self.env
        self._closure_checks = {
            'same_reset_epoch': env._reset_count == self.epoch,
            'same_native_contact_view': env._rigid_contacts is self._contact_view,
            'same_contact_mapping': self._contact_mapping() == self._mapping,
            'same_physics_dt': float(env.dt) == self.dt,
            'same_token_prim_mapping': tuple(self._view.prim_paths) == self._paths,
            'no_unobserved_native_step': int(env.world.current_time_step_index) == self.last_step,
            'same_public_time': float(env.time_s) == self.last_time,
            'same_native_time': float(env.world.current_time) == self.last_native_time,
        }
        if self.samples:
            sampling = env._contact_sampling
            self._closure_checks.update(
                acquisition_ends_at_last_native_step=sampling.get('native_end_step_index') == self.last_step,
                acquisition_ends_at_last_public_time=sampling.get('end_sim_time_s') == self.last_time)
        if not all(self._closure_checks.values()):
            self._failed = True

    def close(self) -> dict:
        if self._closed:
            return self.summary()
        if self._attached:
            if (getattr(self.env, '_board_game_observer', None) is not self
                    or self.env._read_rigid_contacts is not self._wrapper):
                self._failed = True
                raise RuntimeError('Board-game native observer ownership changed before close')
        self._seal_final_context()
        if self._attached:
            if self._previous_instance_reader is None:
                del self.env.__dict__['_read_rigid_contacts']
            else:
                self.env._read_rigid_contacts = self._previous_instance_reader
            del self.env._board_game_observer
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
        self._attached, self._closed = False, True
        return self.summary()
