"""Read-only conversion of native task-layer tensors and signed contacts."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np


def configuration_sha256(configuration):
    payload = {key: value for key, value in configuration.items() if key != 'configuration_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _under(path, root):
    return path == root or path.startswith(root + '/')


def raw_contacts(arrays, sensors, filters):
    """Retain every native point and its signed impulse on canonical actor A.

    Repeated and reversed native views must agree exactly. A full contact buffer
    is rejected because its last valid entry cannot prove absence of truncation.
    """
    if len(arrays) != 6 or len(filters) != len(sensors) or not sensors:
        raise ValueError('Incomplete native contact mapping')
    impulse, point, normal, separation, count, start = [np.asarray(v) for v in arrays]
    if (impulse.ndim not in (1, 2) or not len(impulse)
            or (impulse.ndim == 2 and impulse.shape[1] != 1)
            or separation.shape not in ((len(impulse),), (len(impulse), 1))):
        raise ValueError('Invalid native scalar contact arrays')
    width = len(filters[0])
    if (not width or any(len(row) != width for row in filters)
            or count.shape != (len(sensors), width) or start.shape != count.shape
            or not np.issubdtype(count.dtype, np.integer) or not np.issubdtype(start.dtype, np.integer)
            or (count < 0).any() or (start < 0).any()
            or point.shape != (len(impulse), 3) or normal.shape != point.shape
            or len(separation) != len(impulse)):
        raise ValueError('Invalid native contact arrays')
    result, seen = [], {}
    for i, j in zip(*np.nonzero(count)):
        first, n = int(start[i, j]), int(count[i, j])
        if first + n >= len(impulse):
            raise ValueError('Native contact buffer full or overflowed')
        a, b = sensors[i], filters[i][j]
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            raise ValueError('Invalid native contact actor pair')
        pair, sign = tuple(sorted((a, b))), 1 if a < b else -1
        rows = []
        for k in range(first, first + n):
            vector = float(np.asarray(impulse[k]).reshape(-1)[0]) * normal[k] * sign
            row = (point[k], vector, float(np.asarray(separation[k]).reshape(-1)[0]))
            if not np.isfinite(np.r_[row[0], row[1], row[2]]).all():
                raise ValueError('Nonfinite native contact')
            rows.append(row)
        fingerprint = sorted(tuple(np.r_[p, v, s]) for p, v, s in rows)
        if pair in seen:
            if fingerprint != seen[pair]:
                raise ValueError('Mirrored native contact views disagree')
            continue
        seen[pair] = fingerprint
        result.extend({'a': pair[0], 'b': pair[1], 'point_m': p.tolist(),
            'impulse_on_a_n_s': v.tolist(), 'separation_m': s} for p, v, s in rows)
    return result


class NativeObserver:
    """Observe declared rigid bodies; never derive support from proximity."""
    def __init__(self, env, task):
        self.env = env
        evaluation = task['evaluation']
        self.body_paths = deepcopy(evaluation['body_paths'])
        self.support_paths = deepcopy(evaluation['support_paths'])
        self.dt = float(evaluation['physics_dt_s'])
        self.configuration_sha256 = configuration_sha256(evaluation['configuration'])
        if evaluation['configuration'].get('configuration_sha256') != self.configuration_sha256:
            raise ValueError('Study configuration hash differs from frozen rules')
        if self.dt <= 0 or not math.isfinite(self.dt) or self.dt != float(env.dt):
            raise ValueError('Study native integration period differs from frozen rules')
        paths = [*self.body_paths.values(), *self.support_paths.values()]
        if (not self.body_paths or set(self.body_paths) & set(self.support_paths) or len(set(paths)) != len(paths)
                or any(not isinstance(p, str) or not p.startswith('/World/') for p in paths)):
            raise ValueError('Study observation requires distinct explicit actor paths')
        actual = {name: row['path'] for name, row in env.scene_info['objects'].items()}
        if actual != self.body_paths:
            raise ValueError('Study rigid inventory differs from frozen body paths')
        self.robot_paths = set(env.scene_info['robot_link_paths'].values())
        self.sensors = tuple(env._contact_sensor_paths)
        self.filters = tuple(tuple(row) for row in env._contact_filter_paths)
        if (len(self.filters) != len(self.sensors) or len(set(self.sensors)) != len(self.sensors)
                or not self.robot_paths or not set(self.body_paths.values()) <= set(self.sensors)):
            raise ValueError('Study native contact sensors omit task bodies')
        required = set(paths) | self.robot_paths
        for name, path in self.body_paths.items():
            index = self.sensors.index(path)
            if not required <= set(self.filters[index]):
                raise ValueError('Study native contact filters omit required support or robot: ' + name)

    def actor(self, path):
        for name, root in sorted(self.body_paths.items(), key=lambda row: -len(row[1])):
            if _under(path, root):
                return name
        for name, root in self.support_paths.items():
            if _under(path, root):
                return name
        for root in self.robot_paths:
            if _under(path, root):
                return 'robot/' + root.rsplit('/', 1)[-1]
        # Unknown furniture stays an external actor, never a guessed support.
        return 'external/' + path.strip('/').replace('/', '_')

    def contacts(self, rows):
        result = []
        for row in rows:
            if not any(_under(row[k], path) for k in ('a', 'b') for path in self.body_paths.values()):
                continue
            result.append({**deepcopy(row), 'a': self.actor(row['a']), 'b': self.actor(row['b'])})
        return result

    def frame(self, state, rows, *, episode_id, step_index, time_s, actions):
        if any(not np.isfinite(value).all() for value in state.values()):
            raise ValueError('Nonfinite native study physics state')
        native = self.env.task_runtime_state(state)
        if (native.get('schema') != 'bench_task_runtime_v1'
                or native.get('sampling', {}).get('complete') is not True
                or native.get('sampling', {}).get('physics_steps_added') != 0
                or int(native['native_physics_step_index']) != int(state['native_physics_step_index'])
                or float(native['native_world_time_s']) != float(state['native_world_time_s'])):
            raise ValueError('Incomplete or mixed-tick study native readback')
        if set(native['objects']) != set(self.body_paths) or native['joints'] or native['particle_sets']:
            raise ValueError('Study native body inventory changed')
        objects = {}
        for name, row in native['objects'].items():
            if row.get('path') != self.body_paths[name]:
                raise ValueError('Study native body path changed')
            objects[name] = {}
            for key, width in (('position_m', 3), ('quaternion_wxyz', 4),
                               ('linear_velocity_m_s', 3), ('angular_velocity_rad_s', 3)):
                value = np.asarray(row[key], dtype=float)
                if value.shape != (width,) or not np.isfinite(value).all():
                    raise ValueError('Invalid native study body state')
                if key == 'quaternion_wxyz' and abs(np.linalg.norm(value) - 1.) > 1.e-4:
                    raise ValueError('Invalid native study quaternion')
                objects[name][key] = value.tolist()
        return {'schema': 'study_native_observation_v1',
            'configuration_sha256': self.configuration_sha256,
            'episode_id': episode_id, 'step_index': step_index, 'time_s': float(time_s),
            'physics_dt_s': self.dt, 'sampling_complete': True, 'contacts_complete': True,
            'physics_valid': True, 'robot_only_actions': True, 'object_pose_writes': False,
            'budget_exhausted': False, 'robot_action_count': actions,
            'objects': objects, 'contacts': self.contacts(rows),
            'native_physics_step_index': int(native['native_physics_step_index']),
            'native_world_time_s': float(native['native_world_time_s'])}
