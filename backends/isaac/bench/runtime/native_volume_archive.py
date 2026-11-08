"""Private complete per-native-tick FEM node records for ordinary workers."""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeVolumeArchive:
    def __init__(self, output, reader, *, chunk_size=256):
        if not reader.volumes.info or reader.last_volume_frame is None:
            raise ValueError('Native volume archive requires a captured baseline')
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=False)
        self.reader = reader
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError('Positive native archive chunk size required')
        self.rows, self.chunks = [], []
        self.count = 0
        self.first = self.last = None
        self.errors = []
        self.closed = False
        from scripts.maintenance.freeze_project_sources import verify_if_snapshot
        source = verify_if_snapshot(Path(__file__).absolute().parents[2], required_modules=(
            'bench.runtime.worker', 'bench.runtime.native_volume_archive',
            'bench.simulation.isaacsim_backend.env', 'bench.simulation.isaacsim_backend.native_task_bundle',
            'bench.simulation.isaacsim_backend.native_volume_bodies',
            'bench.evaluation.deformable_robot_geometry'), entrypoint_file=Path(__file__))
        metadata = {'runtime_source_verification': source, 'schema': 'native_volume_archive.v1', 'private': True,
            'source': 'actual_native_tensor_getters', 'reconstructed': False,
            'node_or_velocity_setters': 0, 'volume_bodies': reader.volumes.info,
            'native_initialization': reader.volumes.initialization,
            'robot_body_names': reader.volume_robot_body_names,
            'parent_rigid_body_names': sorted({row['parent_rigid_body'] for row in reader.volumes.info.values()}),
            'robot_cooked_geometry': reader.volume_robot_export,
            'task_usd_path': reader.info['usd_path'], 'task_usd_sha256': reader.info['usd_sha256']}
        (self.path/'metadata.json').write_text(json.dumps(plain(metadata), sort_keys=True, allow_nan=False)+'\n')
        self.append(reader.last_volume_frame)
        if reader.volume_frame_sink is not None:
            raise ValueError('Native volume frame already has an archive sink')
        reader.volume_frame_sink = self.append

    def append(self, frame):
        if self.closed:
            raise RuntimeError('Native volume archive already closed')
        try:
            step, native_time = int(frame['native_step']), float(frame['native_time_s'])
            time_s, dt = float(frame['time_s']), float(frame['physics_dt_s'])
            if not np.isfinite([native_time, time_s, dt]).all() or dt <= 0:
                raise ValueError('Nonfinite or invalid native clock')
            if self.last is None:
                if frame['baseline'] is not True:
                    raise ValueError('Native volume archive must begin at baseline')
                self.first = (step, native_time, time_s, dt)
            elif (frame['baseline'] or step != self.last[0]+1 or dt != self.first[3]
                  or abs(native_time-self.last[1]-dt) > 1.e-8
                  or abs(time_s-self.last[2]-dt) > 1.e-8):
                raise ValueError('Missing, repeated or mixed-tick native volume frame')
            if set(frame['volume_bodies']) != set(self.reader.volumes.info):
                raise ValueError('Native archive volume inventory differs')
            row = {'native_step': np.asarray(step), 'native_time_s': np.asarray(native_time),
                   'time_s': np.asarray(time_s), 'physics_dt_s': np.asarray(dt),
                   'robot_body_q': np.array(frame['robot_body_q'], copy=True),
                   'volume_robot_geometry_json': np.asarray(json.dumps(plain(frame['volume_robot_geometry']),
                                                                       sort_keys=True, allow_nan=False))}
            if (row['robot_body_q'].shape != (len(self.reader.volume_robot_body_names), 7)
                    or not np.isfinite(row['robot_body_q']).all()):
                raise ValueError('Native archive robot pose readback incomplete')
            for name, volume in frame['volume_bodies'].items():
                for source, suffix in (('positions_m', 'q'), ('velocities_m_s', 'qd')):
                    value = np.array(volume[source], copy=True)
                    if value.shape != (self.reader.volumes.info[name]['node_count'], 3) or not np.isfinite(value).all():
                        raise ValueError('Native archive node readback incomplete')
                    row[f'task_volume_{name}_{suffix}'] = value
            for name in sorted({v['parent_rigid_body'] for v in self.reader.volumes.info.values()}):
                obj = frame['objects'][name]
                quaternion = np.asarray(obj['quaternion_wxyz'])
                q = np.concatenate([obj['position_m'], quaternion[[1, 2, 3, 0]]])
                qd = np.concatenate([obj['linear_velocity_m_s'], obj['angular_velocity_rad_s']])
                if (q.shape != (7,) or qd.shape != (6,) or not np.isfinite(q).all()
                        or not np.isfinite(qd).all() or abs(np.linalg.norm(quaternion)-1) > 1.e-5):
                    raise ValueError('Native volume attachment parent readback incomplete')
                row[f'task_rigid_{name}_q'], row[f'task_rigid_{name}_qd'] = q, qd
            self.rows.append(row)
            self.last = (step, native_time, time_s, dt)
            self.count += 1
            if len(self.rows) >= self.chunk_size:
                self._flush()
        except Exception as exc:
            self.errors.append({'error_type': type(exc).__name__, 'message': str(exc)})
            raise

    def _flush(self):
        if not self.rows:
            return
        path = self.path/f'frames_{len(self.chunks):06d}.npz'
        np.savez_compressed(path, **{key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]})
        self.chunks.append({'path': path.name, 'sha256': digest(path), 'frames': len(self.rows),
                            'first_native_step': int(self.rows[0]['native_step']),
                            'last_native_step': int(self.rows[-1]['native_step'])})
        self.rows.clear()

    def close(self, *, actual_native_step, reason):
        if self.closed:
            return
        self._flush()
        terminal_matches = self.last is not None and self.last[0] == int(actual_native_step)
        result = {'schema': 'native_volume_archive.v1', 'reason': reason,
            'frame_count': self.count, 'first_native_step': self.first[0] if self.first else None,
            'last_native_step': self.last[0] if self.last else None,
            'actual_terminal_native_step': int(actual_native_step),
            'complete_native_step_coverage': terminal_matches and not self.errors,
            'physics_or_task_success_claimed': False, 'metadata_sha256': digest(self.path/'metadata.json'),
            'chunks': self.chunks, 'errors': self.errors}
        (self.path/'manifest.json').write_text(json.dumps(result, sort_keys=True, allow_nan=False)+'\n')
        self.reader.volume_frame_sink = None
        self.closed = True
        for path in self.path.iterdir():
            path.chmod(path.stat().st_mode & ~0o222)
