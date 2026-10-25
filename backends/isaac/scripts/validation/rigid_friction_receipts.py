"""Optional raw friction evidence for private scene diagnostics.

Friction anchors have their own counts and indices. They must never be zipped
with normal contacts. This collector does not calibrate vector sign, frame,
units or temporal meaning and does not infer a total contact wrench.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.simulation.isaacsim_backend.contact_sample import copy_array


class FrictionReceipts:
    def __init__(self, env, output):
        self.output = Path(output)
        self.view = env._rigid_contacts
        self.sensors = tuple(env._contact_sensor_paths)
        self.filters = tuple(tuple(row) for row in env._contact_filter_paths)
        self.capacity = int(self.view.max_contact_data_count)
        self.shape = (int(self.view.sensor_count), int(self.view.filter_count))
        if (self.capacity <= 0 or len(self.sensors) != self.shape[0]
                or len(self.filters) != self.shape[0]
                or any(len(row) != self.shape[1] for row in self.filters)):
            raise ValueError('Friction view and path mapping dimensions disagree')
        metadata = {
            'schema': 'native_directed_friction_receipts.v1',
            'sensor_paths': self.sensors, 'filter_paths_per_sensor': self.filters,
            'capacity': self.capacity, 'count_matrix_shape': self.shape,
            'api': 'RigidContactView.get_friction_data', 'api_dt_argument': 1.,
            'independent_of_normal_contact_indices': True,
            'directed_pairs_preserved_without_deduplication': True,
            'omitted_pairs_have_zero_count': True,
            'inactive_start_indices_and_unused_buffer_values_have_no_contact_meaning': True,
            'native_frame_sign_units_and_time_semantics_calibrated': False,
            'quantitative_force_or_torque_qualified': False,
        }
        content = json.dumps(metadata, indent=2)+'\n'
        with (self.output/'friction_mapping.json').open('x') as stream:
            stream.write(content)
        self.mapping_sha256 = hashlib.sha256(content.encode()).hexdigest()
        self.pending_arrays = None

    def _mapping_unchanged(self, env):
        return (env._rigid_contacts is self.view
            and tuple(env._contact_sensor_paths) == self.sensors
            and tuple(tuple(row) for row in env._contact_filter_paths) == self.filters
            and int(self.view.max_contact_data_count) == self.capacity
            and (int(self.view.sensor_count), int(self.view.filter_count)) == self.shape)

    def capture(self, env, normal_sample):
        """Copy all four buffers before any further native tensor query."""
        arrays, copy_failures = {}, []
        try:
            if self.pending_arrays is not None:
                raise ValueError('Previous friction capture lacks readonly verification')
            # The exchange has already detached all SIX normal buffers; this
            # API reuses its count/start buffers and would corrupt live views.
            normal_sample.for_env(env)
            if not self._mapping_unchanged(env):
                raise ValueError('Friction view/path mapping changed')
            for index, value in enumerate(self.view.get_friction_data(dt=1.)):
                try:
                    arrays[index] = copy_array(value)
                except Exception as exc:
                    copy_failures.append({'index': index, 'type': type(exc).__name__, 'detail': str(exc)})
            if copy_failures:
                raise ValueError('Failed to copy one or more native friction arrays')
            normal_sample.for_env(env)
            if not self._mapping_unchanged(env):
                raise ValueError('Friction view/path mapping changed during capture')
            receipt = self._decode(tuple(arrays.values()))
            self.pending_arrays = arrays
            return receipt
        except Exception as exc:
            self._save_failure(env, arrays, exc, 'capture', copy_failures=copy_failures)
            raise

    def _save_failure(self, env, arrays, exc, phase, *, copy_failures=(), states=None):
        # Retain every successfully copied array, including those after a bad
        # buffer. A readonly rejection is invalid evidence, never a valid row.
        payload = {f'raw_{i}': a for i, a in arrays.items()}
        for label, state in (states or {}).items():
            if state is not None:
                payload.update({f'{label}_{key}': value for key, value in state.items()})
        if payload:
            with (self.output/'friction_capture_failure.npz').open('xb') as stream:
                np.savez_compressed(stream, **payload)
        with (self.output/'friction_capture_failure.json').open('x') as stream:
            json.dump({'type': type(exc).__name__, 'detail': str(exc), 'phase': phase,
                'native_step': int(env.world.current_time_step_index),
                'mapping_sha256': self.mapping_sha256, 'raw_array_count': len(arrays),
                'copied_array_indices': sorted(arrays), 'copy_failures': list(copy_failures),
                'valid_frame': False}, stream)

    def accept_readonly(self):
        if self.pending_arrays is None:
            raise ValueError('No friction capture to verify')
        self.pending_arrays = None

    def reject_readonly(self, env, before, after, exc):
        self._save_failure(env, self.pending_arrays or {}, exc, 'readonly',
            states={'before': before, 'after': after})
        self.pending_arrays = None

    def _decode(self, arrays):
        if len(arrays) != 4:
            raise ValueError('Friction API must return all four independent arrays')
        vectors, points, counts, starts = arrays
        if (vectors.shape != (self.capacity, 3) or points.shape != (self.capacity, 3)
                or counts.shape != self.shape or starts.shape != self.shape):
            raise ValueError('Friction tensor dimensions disagree with declared view')
        if not all(np.issubdtype(a.dtype, np.integer) for a in (counts, starts)):
            raise ValueError('Friction counts and starts must be integer tensors')
        if np.any(counts < 0):
            raise ValueError('Negative friction count')
        pairs = []
        for i, j in zip(*np.nonzero(counts), strict=True):
            start, count = int(starts[i, j]), int(counts[i, j])
            end = start+count
            # Reaching capacity cannot establish absence of truncated data.
            if start < 0 or end >= self.capacity:
                raise ValueError('Friction active range invalid or capacity exhausted')
            active_vectors, active_points = vectors[start:end], points[start:end]
            if not np.isfinite(active_vectors).all() or not np.isfinite(active_points).all():
                raise ValueError('Nonfinite active friction vectors or anchors')
            pairs.append({'sensor_index': int(i), 'filter_index': int(j),
                'start': start, 'count': count,
                'raw_friction_vectors_dt1': active_vectors.tolist(),
                'raw_anchor_points': active_points.tolist()})
        return {'mapping_sha256': self.mapping_sha256, 'pairs': pairs,
            'directed_anchor_count': sum(p['count'] for p in pairs)}
