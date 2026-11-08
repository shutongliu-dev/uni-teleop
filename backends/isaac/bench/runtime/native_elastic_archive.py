"""Private, sealed per-native-tick records for passive rigid-cell composites.

The archive only copies supplied observations. It owns no simulator and writes
no physical state. Physical topology and cooked geometry are saved once; the
large render skin binding remains bound by the frozen observer configuration
hash instead of being duplicated in this archive.
"""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'native_elastic_archive.v1'


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seal(path):
    path.chmod(path.stat().st_mode & ~0o222)


def _physical_composites(composites):
    """Explicit physical-only allowlist; no arbitrary render/user metadata."""
    result = {}
    fields = ('profile', 'parent_rigid_body', 'member_ids', 'total_mass_kg', 'foam_mass_kg',
              'pad_mass_kg', 'foam_volume_m3', 'foam_density_kg_m3', 'grid', 'z_layers_m',
              'rest_local_bounds_m', 'material_prior', 'internal_collision_filter', 'links')
    cell_fields = ('id', 'body_path', 'grid_index', 'mass_kg', 'volume_m3', 'initial_pose_xyzw',
                   'composite_rest_center_m', 'center_of_mass_m', 'diagonal_inertia_kg_m2',
                   'principal_axes_xyzw', 'collision_support_points_m', 'collider_paths')
    for name, row in composites.items():
        result[name] = {key: plain(row[key]) for key in fields if key in row}
        result[name]['cells'] = [{key: plain(cell[key]) for key in cell_fields if key in cell}
                                 for cell in row['cells']]
    return result


class NativeElasticArchive:
    def __init__(self, output, reader, *, chunk_size=256):
        if not reader.elastic_info or reader.last_elastic_frame is None:
            raise ValueError('Native elastic archive requires a captured baseline')
        if reader.elastic_frame_sink is not None:
            raise ValueError('Native elastic frame already has an archive sink')
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError('Positive integer native archive chunk size required')
        self.reader, self.chunk_size = reader, chunk_size
        self.object_names = sorted(reader.info['objects'])
        self.robot_names = list(reader.volume_robot_body_names)
        self.joint_names = set(reader.info['joints'])
        members = [member for row in reader.elastic_info.values() for member in row['member_ids']]
        if (not self.object_names or not self.robot_names or not self.joint_names
                or len(set(members)) != len(members) or not set(members) <= set(self.object_names)
                or len(set(self.robot_names)) != len(self.robot_names)):
            raise ValueError('Complete distinct elastic, object and robot inventories required')
        self.rows, self.chunks, self.errors = [], [], []
        self.count = 0
        self.first = self.last = None
        self.closed = False
        self.path = Path(output)
        from scripts.maintenance.freeze_project_sources import verify_if_snapshot
        source = verify_if_snapshot(Path(__file__).absolute().parents[2], required_modules=(
            'bench.runtime.worker', 'bench.runtime.native_elastic_archive',
            'bench.simulation.isaacsim_backend.env', 'bench.simulation.isaacsim_backend.native_task_bundle',
            'bench.evaluation.deformable_robot_geometry',
            'bench.evaluation.cloth.cloth_robot_collision'), entrypoint_file=Path(__file__))
        self.source_verified = bool(source and source.get('verified') is True)
        observer = reader.spec['observer']
        configuration_hash = hashlib.sha256(canonical(observer['configuration']).encode()).hexdigest()
        metadata = {'schema': SCHEMA, 'private': True, 'source': 'actual_native_tensor_getters',
            'reconstructed': False, 'physical_state_setters': 0, 'runtime_source_verification': source,
            'object_names': self.object_names, 'robot_body_names': self.robot_names,
            'joint_names': sorted(self.joint_names), 'component_member_ids': sorted(members),
            'object_paths': {name: reader.info['objects'][name]['path'] for name in self.object_names},
            'elastic_composites_physical': _physical_composites(reader.elastic_info),
            'native_joint_definitions': plain(reader.info['joints']),
            'robot_cooked_geometry': plain(reader.volume_robot_export),
            'component_cooked_geometry': plain(reader.elastic_collider_export),
            'task_usd_path': reader.info['usd_path'], 'task_usd_sha256': reader.info['usd_sha256'],
            'observer_source': {key: plain(observer[key]) for key in ('path', 'sha256', 'files', 'class_name') if key in observer},
            'observer_configuration_sha256': configuration_hash,
            'observer_configuration_hash_domain': 'json sort_keys=True,separators=(comma,colon),ensure_ascii=True,allow_nan=False',
            'skin_binding_saved': False, 'pose_convention': 'object_q and robot_body_q: xyz then xyzw; qd: linear m/s then angular rad/s'}
        encoded = canonical(metadata)+'\n'
        # Validate the baseline before making output or installing a sink.
        baseline = self._row(reader.last_elastic_frame)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/'metadata.json').write_text(encoded)
        _seal(self.path/'metadata.json')
        self._accept(baseline)
        reader.elastic_frame_sink = self.append

    def _row(self, frame):
        step = frame['native_step']
        baseline = frame['baseline']
        if type(step) is not int or step < 0 or type(baseline) is not bool:
            raise ValueError('Native archive requires exact integer ticks and boolean baseline')
        clock = [frame['native_time_s'], frame['time_s'], frame['physics_dt_s']]
        if any(isinstance(x, (bool, np.bool_)) for x in clock):
            raise ValueError('Native clock values cannot be boolean')
        native_time, time_s, dt = map(float, clock)
        if not np.isfinite([native_time, time_s, dt]).all() or dt <= 0 or time_s < 0:
            raise ValueError('Nonfinite or invalid native clock')
        if self.last is None:
            if baseline is not True or abs(time_s) > 1e-12:
                raise ValueError('Native elastic archive must begin at the episode baseline')
        elif (baseline or step != self.last[0]+1 or dt != self.first[3]
              or abs(native_time-self.last[1]-dt) > 1e-8
              or abs(time_s-self.last[2]-dt) > 1e-8):
            raise ValueError('Missing, repeated or mixed-tick native elastic frame')
        if set(frame['objects']) != set(self.object_names) or set(frame['joints']) != self.joint_names:
            raise ValueError('Native elastic object or joint inventory differs')
        robot = np.array(frame['robot_body_q'], dtype=float, copy=True)
        if (robot.shape != (len(self.robot_names), 7) or not np.isfinite(robot).all()
                or not np.allclose(np.linalg.norm(robot[:, 3:], axis=1), 1., atol=1e-5, rtol=0)):
            raise ValueError('Native archive robot pose readback incomplete')
        poses, velocities = [], []
        for name in self.object_names:
            obj = frame['objects'][name]
            quaternion = np.asarray(obj['quaternion_wxyz'], dtype=float)
            if quaternion.shape != (4,):
                raise ValueError('Native elastic body quaternion has wrong shape')
            q = np.concatenate([obj['position_m'], quaternion[[1, 2, 3, 0]]]).astype(float)
            qd = np.concatenate([obj['linear_velocity_m_s'], obj['angular_velocity_rad_s']]).astype(float)
            if (q.shape != (7,) or qd.shape != (6,) or not np.isfinite(q).all()
                    or not np.isfinite(qd).all() or abs(np.linalg.norm(quaternion)-1) > 1e-5):
                raise ValueError('Native elastic body pose or velocity readback incomplete')
            poses.append(q)
            velocities.append(qd)
        if not isinstance(frame['elastic_robot_geometry'], Mapping) or not isinstance(frame['contacts'], (list, tuple)):
            raise ValueError('Native elastic geometry and contact records required')
        return {'native_step': np.asarray(step), 'baseline': np.asarray(baseline),
            'native_time_s': np.asarray(native_time), 'time_s': np.asarray(time_s), 'physics_dt_s': np.asarray(dt),
            'object_q': np.array(poses), 'object_qd': np.array(velocities), 'robot_body_q': robot,
            'joints_json': np.asarray(canonical(frame['joints'])),
            'elastic_robot_geometry_json': np.asarray(canonical(frame['elastic_robot_geometry'])),
            'contacts_json': np.asarray(canonical(frame['contacts']))}

    def _accept(self, row):
        value = (int(row['native_step']), *(float(row[key]) for key in
                 ('native_time_s', 'time_s', 'physics_dt_s')))
        if self.first is None:
            self.first = value
        self.rows.append(row)
        self.last = value
        self.count += 1
        if len(self.rows) >= self.chunk_size:
            self._flush()

    def append(self, frame):
        if self.closed:
            raise RuntimeError('Native elastic archive already closed')
        try:
            self._accept(self._row(frame))
        except Exception as exc:
            self.errors.append({'error_type': type(exc).__name__, 'message': str(exc)})
            raise

    def _flush(self):
        if not self.rows:
            return
        path = self.path/f'frames_{len(self.chunks):06d}.npz'
        np.savez_compressed(path, **{key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]})
        self.chunks.append({'path': path.name, 'sha256': digest(path), 'frames': len(self.rows),
            'first_native_step': int(self.rows[0]['native_step']), 'last_native_step': int(self.rows[-1]['native_step'])})
        _seal(path)
        self.rows.clear()

    def close(self, *, actual_native_step, reason):
        if self.closed:
            return
        if type(actual_native_step) is not int or actual_native_step < 0:
            raise ValueError('Archive terminal step must be a nonnegative integer')
        self._flush()
        terminal_matches = self.last is not None and self.last[0] == actual_native_step
        result = {'schema': SCHEMA, 'reason': str(reason), 'frame_count': self.count,
            'first_native_step': self.first[0] if self.first else None,
            'last_native_step': self.last[0] if self.last else None,
            'actual_terminal_native_step': actual_native_step,
            'complete_native_step_coverage': terminal_matches and not self.errors,
            'runtime_source_snapshot_verified': self.source_verified,
            'physics_or_task_success_claimed': False, 'metadata_sha256': digest(self.path/'metadata.json'),
            'chunks': self.chunks, 'errors': self.errors}
        manifest = self.path/'manifest.json'
        manifest.write_text(canonical(result)+'\n')
        _seal(manifest)
        self.reader.elastic_frame_sink = None
        self.closed = True
