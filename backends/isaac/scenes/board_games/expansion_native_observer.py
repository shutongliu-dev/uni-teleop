"""Read-only native-substep evidence for the four card-room expansions.

Hooks the existing contact acquisition once. No simulator state setters,
additional contact acquisitions or physics steps occur here. All records are
private; the referee alone produces allowlisted public events and game state.
"""
from collections import Counter
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path

import numpy as np

from bench.paths import OUTPUT_ROOT


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return np.asarray(value).copy()


def rotate_xyzw(q, vector):
    xyz, w = q[:3], q[3]
    return vector + 2*np.cross(xyz, np.cross(xyz, vector) + w*vector)


def json_value(value):
    return asdict(value) if is_dataclass(value) else value


def robot_ownership(link_paths):
    """Map the actual imported link inventory, retaining arm/hand distinction."""
    result = {}
    for name, path in link_paths.items():
        choices = [(player, name.startswith(side+'_')) for player, side in
                   (('player_0', 'left'), ('player_1', 'right'))
                   if name.startswith((side+'_', 'fr3_'+side+'_'))]
        if len(choices) != 1 or not path.startswith('/World/Robot/') or path in result:
            raise ValueError('Robot contact inventory lacks unambiguous frozen ownership: '+name)
        player, hand = choices[0]
        result[path] = {'player_id': player, 'hand': hand, 'link_name': name}
    if {v['player_id'] for v in result.values()} != {'player_0', 'player_1'}:
        raise ValueError('Both native robot contact inventories are required')
    return result


def aggregate_contacts(rows, *, now, paths, deck_paths, owners, static_paths=()):
    """Preserve signed action/reaction; never count floor contacts as support."""
    support = {name: 0. for name in paths.values()}
    robot = {name: {'player_0': 0., 'player_1': 0.} for name in support}
    nonhand = {name: dict(value) for name, value in robot.items()}
    object_support = {name: {} for name in support}
    static_support = dict(support)
    penetration = dict(support)
    seen = set()
    for row in rows:
        pair = (row.get('actor0'), row.get('actor1'))
        if (not all(isinstance(p, str) for p in pair) or pair[0] == pair[1]
                or tuple(sorted(pair)) in seen or row.get('time_s') != now):
            raise RuntimeError('Invalid, duplicated or stale native contact row')
        seen.add(tuple(sorted(pair)))
        vector = np.asarray(row.get('impulse_vector_ns'), dtype=float)
        magnitude, separation = row.get('impulse_magnitude_sum_ns'), row.get('min_separation_m')
        if (vector.shape != (3,) or not np.isfinite(vector).all()
                or type(magnitude) not in (float, int) or not math.isfinite(magnitude) or magnitude < 0.
                or type(separation) not in (float, int) or not math.isfinite(separation)):
            raise RuntimeError('Missing finite signed native contact impulse/separation')
        for index, path in enumerate(pair):
            name = paths.get(path)
            if name is None:
                continue
            other = pair[1-index]
            upward = max(0., float(vector[2])*(1 if index == 0 else -1))
            penetration[name] = max(penetration[name], -float(separation))
            if other in deck_paths:
                support[name] += upward
            if other in paths:
                object_support[name][paths[other]] = upward
            elif other in static_paths:
                static_support[name] += upward
            if other in owners:
                owner = owners[other]
                robot[name][owner['player_id']] += float(magnitude)
                if not owner['hand']:
                    nonhand[name][owner['player_id']] += float(magnitude)
            elif other.startswith('/World/Robot/'):
                raise RuntimeError('Contact with undeclared robot body')
    return support, robot, nonhand, object_support, penetration, static_support


class ExpansionNativeObserver:
    def __init__(self, env, match, metadata, *, frame_jsonl=None):
        self.env, self.match = env, match
        self.metadata = metadata
        self.epoch, self.dt = env._reset_count, float(env.dt)
        self.last_step = int(env.world.current_time_step_index)
        self.last_time, self.last_native_time = float(env.time_s), float(env.world.current_time)
        self._view = env._sim_view.create_rigid_body_view('/World/CardGame/Actors/*')
        self._paths = tuple(self._view.prim_paths)
        self._path_names = {path: name for name, path in metadata['actors'].items()}
        if (len(self._paths) != len(self._path_names) or self._view.count != len(self._path_names)
                or set(self._paths) != set(self._path_names)
                or any(env.scene_info['objects'].get(name, {}).get('path') != path
                       for path, name in self._path_names.items())):
            raise RuntimeError('Native expansion body inventory differs from the frozen asset')
        self._coms = array(self._view.get_coms())
        if self._coms.shape != (len(self._paths), 7) or not np.isfinite(self._coms).all():
            raise RuntimeError('Missing finite native centers of mass')
        self.owners = robot_ownership(env.scene_info['robot_link_paths'])
        collider_map = env.scene_info.get('collider_actor_paths', {})
        declared = metadata['fixture_collision_paths']
        if declared != ['/World/CardGame/Fixtures/Deck']:
            raise RuntimeError('Only the frozen tabletop Deck is a direct support')
        self.deck_paths = tuple(collider_map.get(path, path) for path in declared)
        self.static_paths = frozenset(env.scene_info.get('support_paths', ())) | frozenset(
            path for info in env.scene_info.get('fixtures', {}).values()
            for path in info.get('contact_filter_paths', info.get('collision_paths', ())))
        self.static_paths -= set(self.deck_paths) | set(self._paths) | set(self.owners)
        self._mapping = self._contact_mapping()
        sensors, filters = self._mapping
        required = set(self.deck_paths) | set(self.owners) | set(self._paths) | set(self.static_paths)
        for path in self._paths:
            if path not in sensors or not required <= set(filters[sensors.index(path)]):
                raise RuntimeError('Incomplete native object/Deck/two-robot/known-static contact coverage')
        self._contact_view = env._rigid_contacts
        self._original = env._read_rigid_contacts
        self._previous = env.__dict__.get('_read_rigid_contacts')
        self._wrapper = None
        self._attached = self._closed = self._failed = False
        self._closure_checks = None
        self.samples = self.unsupported_samples = 0
        self.reasons = Counter()
        self.last_result = None
        self._stream = None
        self.frame_jsonl = None
        masses = array(self._view.get_masses()).reshape(-1)
        inertias = array(self._view.get_inertias()).reshape(-1, 3, 3)
        checks = {}
        for index, path in enumerate(self._paths):
            source = metadata['mass_properties'][self._path_names[path]]
            checks[self._path_names[path]] = bool(
                np.isclose(masses[index], source['mass_kg'], rtol=1e-5, atol=1e-8)
                and np.allclose(self._coms[index, :3], source['center_of_mass_m'], rtol=0., atol=2e-7)
                and np.allclose(inertias[index], source['inertia_tensor_kg_m2'], rtol=2e-4, atol=1e-10))
        native = {'body_paths': list(self._paths), 'coms': self._coms.tolist(),
                  'masses_kg': masses.tolist(), 'inertias_kg_m2': inertias.tolist(),
                  'mass_properties_verified': checks,
                  'expected_mass_properties': metadata['mass_properties'],
                  'runtime_contact_sampling': env.scene_info['native_task_bundle'].get('runtime_contact_sampling'),
                  'robot_contact_ownership': self.owners, 'deck_contact_paths': list(self.deck_paths),
                  'known_static_contact_paths': sorted(self.static_paths),
                  'reset_epoch': self.epoch, 'native_step': self.last_step, 'physics_dt_s': self.dt,
                  'physical_state_setters_called': 0, 'contact_projection': 'normal_only_signed_pair_impulses'}
        if frame_jsonl is not None:
            path = Path(frame_jsonl).resolve()
            if not path.is_relative_to(OUTPUT_ROOT):
                raise ValueError('Private game evidence must remain under shared datasets')
            path.parent.mkdir(parents=True, exist_ok=True)
            (path.parent/'observer_inputs.json').write_text(json.dumps(native, indent=2, allow_nan=False)+'\n')
            if not all(checks.values()):
                raise RuntimeError('Native task mass, COM or inertia differs from immutable scene-v2 input')
            self._stream = path.open('x')
            self.frame_jsonl = str(path)
        elif not all(checks.values()):
            raise RuntimeError('Native task mass, COM or inertia differs from immutable scene-v2 input')

    def _contact_mapping(self):
        return (tuple(self.env._contact_sensor_paths), tuple(tuple(row) for row in self.env._contact_filter_paths))

    def attach(self):
        if self._attached or self._closed or getattr(self.env, '_board_game_observer', None) is not None:
            raise RuntimeError('A native game observer already owns this contact exchange')

        def after_acquisition():
            if self._failed:
                raise RuntimeError('Native game observer has failed')
            try:
                start = len(self.env._contact_reports)
                sampled = self.env._contact_sampling['sampled_substeps']
                result = self._original()
                self._observe(self.env._contact_reports[start:], sampled)
                return result
            except BaseException:
                self._failed = True
                raise

        self._wrapper = after_acquisition
        self.env._board_game_observer = self
        self.env._read_rigid_contacts = after_acquisition
        self._attached = True
        return self

    def _observe(self, rows, sampled_before):
        from scenes.workrooms.card_room.expansion_referee import ExpansionNativeFrame, ExpansionObjectEvidence
        env = self.env
        step, now, native_now = int(env.world.current_time_step_index), float(env.time_s), float(env.world.current_time)
        sampling = env._contact_sampling
        if (env._reset_count != self.epoch or env._rigid_contacts is not self._contact_view
                or self._contact_mapping() != self._mapping or float(env.dt) != self.dt
                or tuple(self._view.prim_paths) != self._paths or step != self.last_step+1
                or not math.isclose(now-self.last_time, self.dt, rel_tol=0., abs_tol=1.e-9)
                or not math.isclose(native_now-self.last_native_time, self.dt, rel_tol=0., abs_tol=1.e-9)
                or sampling['sampled_substeps'] != sampled_before+1
                or sampling['native_end_step_index'] != step or sampling['end_sim_time_s'] != now):
            raise RuntimeError('Stale, skipped or mismatched expansion native acquisition')
        poses, velocities = array(self._view.get_transforms()), array(self._view.get_velocities())
        count = len(self._paths)
        if (poses.shape != (count, 7) or velocities.shape != (count, 6)
                or not np.isfinite(poses).all() or not np.isfinite(velocities).all()
                or np.any(np.abs(np.sum(poses[:, 3:]**2, axis=1)-1) > 1.e-3)):
            raise RuntimeError('Invalid batched native expansion body state')
        if (int(env.world.current_time_step_index) != step or float(env.time_s) != now
                or float(env.world.current_time) != native_now):
            raise RuntimeError('Native clock changed during expansion state read')
        support, robot, nonhand, object_support, penetration, static_support = aggregate_contacts(
            rows, now=now, paths=self._path_names, deck_paths=self.deck_paths, owners=self.owners,
            static_paths=self.static_paths)
        objects = {}
        for index, path in enumerate(self._paths):
            name = self._path_names[path]
            pose, velocity = poses[index], velocities[index]
            quat = pose[3:]/np.linalg.norm(pose[3:])
            origin_velocity = velocity[:3]-np.cross(velocity[3:], rotate_xyzw(quat, self._coms[index, :3]))
            objects[name] = ExpansionObjectEvidence(
                position_m=tuple(float(v) for v in pose[:3]),
                quaternion_wxyz=tuple(float(v) for v in quat[[3, 0, 1, 2]]),
                linear_velocity_m_s=tuple(float(v) for v in origin_velocity),
                angular_velocity_rad_s=tuple(float(v) for v in velocity[3:]),
                deck_support_impulse_ns=support[name], robot_contact_impulses_ns=robot[name],
                robot_nonhand_contact_impulses_ns=nonhand[name], object_support_impulses_ns=object_support[name],
                other_static_support_impulse_ns=static_support[name],
                maximum_penetration_m=penetration[name])
        complete = not bool(env._contact_errors)
        frame = ExpansionNativeFrame(step, now, self.dt, objects, complete)
        result = self.match.observe(frame)
        self.samples += 1
        self.unsupported_samples += int(not complete)
        self.last_result = json_value(result)
        if isinstance(self.last_result, dict):
            self.reasons[self.last_result.get('reason', 'observed')] += 1
        self.last_step, self.last_time, self.last_native_time = step, now, native_now
        if self._stream is not None:
            self._stream.write(json.dumps({'frame': asdict(frame), 'raw_contacts': rows,
                'result': self.last_result}, allow_nan=False)+'\n')

    def summary(self):
        return {'schema': 'card_game_expansion_native_observer.v1', 'samples': self.samples,
                'unsupported_samples': self.unsupported_samples, 'failed': self._failed,
                'last_native_step': self.last_step, 'reset_epoch': self.epoch,
                'deck_contact_paths': list(self.deck_paths), 'robot_contact_ownership': self.owners,
                'physical_steps_added': 0, 'additional_contact_acquisitions': 0,
                'physical_state_setters_called': 0, 'velocity_frame': 'world_body_origin_corrected_from_COM',
                'contact_scope': 'signed_normal_impulses_and_reported_separations; no_tangential_impulse_claim',
                'final_context_verified': None if self._closure_checks is None else all(self._closure_checks.values()),
                'closure_checks': self._closure_checks, 'frame_jsonl': self.frame_jsonl}

    def close(self):
        if self._closed:
            return self.summary()
        env = self.env
        if self._attached and (getattr(env, '_board_game_observer', None) is not self
                               or env._read_rigid_contacts is not self._wrapper):
            self._failed = True
            raise RuntimeError('Native game observer ownership changed before close')
        self._closure_checks = {
            'same_reset_epoch': env._reset_count == self.epoch,
            'same_contact_view': env._rigid_contacts is self._contact_view,
            'same_contact_mapping': self._contact_mapping() == self._mapping,
            'same_body_mapping': tuple(self._view.prim_paths) == self._paths,
            'same_physics_dt': float(env.dt) == self.dt,
            'no_unobserved_native_step': int(env.world.current_time_step_index) == self.last_step,
            'same_public_time': float(env.time_s) == self.last_time,
            'same_native_time': float(env.world.current_time) == self.last_native_time,
        }
        if self.samples:
            self._closure_checks['acquisition_ends_at_last_step'] = env._contact_sampling['native_end_step_index'] == self.last_step
        self._failed |= not all(self._closure_checks.values())
        if self._attached:
            if self._previous is None:
                del env.__dict__['_read_rigid_contacts']
            else:
                env._read_rigid_contacts = self._previous
            del env._board_game_observer
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
        self._attached, self._closed = False, True
        return self.summary()
