"""Independent full-tick replay of private FEM arrays and cooked robot geometry."""
from collections import deque
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from bench.evaluation.cloth.cloth_robot_collision import reanalyse_export, transform_xyzw, state_digest
from bench.evaluation.deformable_robot_geometry import VolumeRobotGeometry
from bench.paths import load_json


def review_native_volumes(private, runtime, task, evidence, *, expected_source=None):
    """No native execution. Every saved node, attachment and release gap is read."""
    private = Path(private)
    root = private/'native_volumes'
    def read(path):
        evidence[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load_json(path)
    result = {'passed': False, 'native_execution_performed': False, 'errors': [],
              'scope': 'Full native node/rigid/robot array replay; no FEM contact-force or manipulation claim.'}
    try:
        metadata, manifest = read(root/'metadata.json'), read(root/'manifest.json')
        if (metadata['schema'] != 'native_volume_archive.v1' or metadata['source'] != 'actual_native_tensor_getters'
                or metadata['reconstructed'] is not False or metadata['node_or_velocity_setters'] != 0
                or manifest['schema'] != metadata['schema'] or manifest['errors']
                or manifest['reason'] != 'finish_request' or manifest['complete_native_step_coverage'] is not True
                or manifest['metadata_sha256'] != evidence[str((root/'metadata.json').resolve())]):
            raise ValueError('Incomplete, failed or reconstructed native volume archive')
        for name in ('worker.log', 'unacknowledged_step_state.npz', 'unacknowledged_step.json'):
            path = private/name
            if path.exists():
                evidence[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
                if name != 'worker.log':
                    raise ValueError('Native worker recorded an unacknowledged physical step')
        source = metadata.get('runtime_source_verification')
        if expected_source is not None and (not source or source.get('verified') is not True
                or source.get('runtime_imports_verified') is not True
                or Path(source['snapshot_root']).resolve() != Path(expected_source).resolve()):
            raise ValueError('Lazily loaded native volume modules lack the same actual source receipt')
        expected = task['asset_metadata']['volume_bodies']
        bundle = runtime['scene_spec']['native_task_bundle']
        if (bundle.get('volume_bodies') != expected or set(metadata['volume_bodies']) != set(expected)
                or metadata['task_usd_path'] != bundle['usd_path'] or metadata['task_usd_sha256'] != bundle['usd_sha256']):
            raise ValueError('Native volume archive/release identity differs')
        for name, spec in expected.items():
            if any(metadata['volume_bodies'][name].get(key) != value for key, value in spec.items()):
                raise ValueError('Archived native volume declaration differs: '+name)
        init = metadata['native_initialization']
        if (init['mode'] != 'author-only' or init['position_write_calls'] != 0 or init['velocity_write_calls'] != 0
                or not init['tensor_device'].startswith('cuda') or init['tensor_device_ordinal'] < 0
                or init != runtime['task_volume_initialization']):
            raise ValueError('Native volume initialization identity differs')
        export = reanalyse_export(metadata['robot_cooked_geometry'])
        checker = VolumeRobotGeometry(export, metadata['robot_body_names'], metadata['volume_bodies'])
        limits = task['evaluation']['thresholds']
        dt = float(runtime['physics_dt_s']); period = float(limits['stability_pose_window_s'])
        window = round(period/dt)
        if window < 1 or abs(window*dt-period) > 1.e-12:
            raise ValueError('Invalid full-native stability interval')
        parents = sorted({row['parent_rigid_body'] for row in expected.values()})
        if metadata['parent_rigid_body_names'] != parents:
            raise ValueError('Native attachment parent inventory differs')
        endpoints_path = private/'states.npz'; evidence[str(endpoints_path.resolve())] = hashlib.sha256(endpoints_path.read_bytes()).hexdigest()
        with np.load(endpoints_path, allow_pickle=False) as loaded:
            endpoints = {key: loaded[key] for key in loaded.files}
        export_state_hash = state_digest({key: value[0] for key, value in endpoints.items()})
        if export_state_hash != metadata['robot_cooked_geometry']['state_hash_before']:
            raise ValueError('Cooked robot export does not bind the captured native baseline')
        endpoint_steps = endpoints['native_physics_step_index']
        if (not np.issubdtype(endpoint_steps.dtype, np.integer) or endpoint_steps.ndim != 1 or not len(endpoint_steps) or np.any(np.diff(endpoint_steps) <= 0)):
            raise ValueError('Missing or repeated endpoint native ticks')
        endpoint_ids = {int(step): i for i, step in enumerate(endpoint_steps)}
        geometry, totals, histories, anchors, quiet = {}, {}, {}, {}, {}
        for name, spec in expected.items():
            rest, tets = np.asarray(spec['rest_points_m']), np.asarray(spec['tetrahedra'])
            rest_det = np.linalg.det((rest[tets[:, 1:]]-rest[tets[:, :1]]).transpose(0, 2, 1))
            edges = np.unique(np.sort(np.concatenate([tets[:, [a, b]] for a in range(4) for b in range(a+1, 4)]), axis=1), axis=0)
            lengths = np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1)
            if np.any(rest_det <= 0) or np.any(lengths <= 0):
                raise ValueError('Invalid frozen FEM topology')
            geometry[name] = (tets, rest_det, edges, lengths)
            histories[name], anchors[name], quiet[name] = deque(maxlen=window+1), None, 0.
            totals[name] = dict(minimum_tetrahedron_volume_ratio=float('inf'), maximum_tetrahedron_volume_ratio=0.,
                minimum_edge_length_ratio=float('inf'), maximum_edge_length_ratio=0.,
                maximum_material_bond_error_m=0., maximum_native_node_speed_m_s=0.)
        first = previous = None; frames = endpoints_checked = 0
        for chunk in manifest['chunks']:
            filename = chunk['path']
            if Path(filename).name != filename or not filename.endswith('.npz'):
                raise ValueError('Invalid native archive chunk path')
            path = root/filename; evidence[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            if evidence[str(path.resolve())] != chunk['sha256']:
                raise ValueError('Native archive chunk hash differs')
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
                if not np.issubdtype(arrays['native_step'].dtype, np.integer):
                    raise ValueError('Native step indices must be integer-valued')
                count = len(arrays['native_step'])
                if (count != chunk['frames'] or count == 0 or int(arrays['native_step'][0]) != chunk['first_native_step']
                        or int(arrays['native_step'][-1]) != chunk['last_native_step']
                        or any(len(arrays[key]) != count for key in arrays)):
                    raise ValueError('Native chunk shape/count differs')
                for i in range(count):
                    step = int(arrays['native_step'][i]); time = float(arrays['time_s'][i]); native = float(arrays['native_time_s'][i])
                    if (not np.isfinite([time, native]).all() or float(arrays['physics_dt_s'][i]) != dt
                            or previous and (step != previous[0]+1 or abs(native-previous[1]-dt)>1.e-8 or abs(time-previous[2]-dt)>1.e-8)):
                        raise ValueError('Dropped, repeated or mixed native volume tick')
                    if first is None:
                        first = (step, native, time)
                        if step != init['native_step'] or time != 0. or native != init['native_time_s']:
                            raise ValueError('Native archive does not begin at captured initialization')
                    nodes = {name: {'positions_m': arrays[f'task_volume_{name}_q'][i],
                                    'velocities_m_s': arrays[f'task_volume_{name}_qd'][i]} for name in expected}
                    poses = arrays['robot_body_q'][i]
                    robot = checker.check(nodes, poses)
                    saved_geometry = json.loads(str(arrays['volume_robot_geometry_json'][i]))
                    if robot != saved_geometry:
                        raise ValueError('Native recorded cooked geometry differs from independent replay')
                    if step in endpoint_ids:
                        index = endpoint_ids[step]; endpoints_checked += 1
                        if not np.array_equal(poses, endpoints['body_q'][index, :len(metadata['robot_body_names'])]):
                            raise ValueError('Native robot archive differs from endpoint capture')
                        for name in expected:
                            for suffix in ('q', 'qd'):
                                key = f'task_volume_{name}_{suffix}'
                                if not np.array_equal(arrays[key][i], endpoints[key][index]):
                                    raise ValueError('Native volume archive differs from endpoint capture')
                        for parent in parents:
                            object_index = runtime['object_names'].index(parent)
                            for suffix in ('q', 'qd'):
                                if not np.array_equal(arrays[f'task_rigid_{parent}_{suffix}'][i], endpoints['object_'+suffix][index, object_index]):
                                    raise ValueError('Native attachment parent differs from endpoint capture')
                    for name, row in nodes.items():
                        spec = expected[name]; p, v = np.asarray(row['positions_m'], float), np.asarray(row['velocities_m_s'], float)
                        if v.shape != p.shape or not np.isfinite(v).all():
                            raise ValueError('Incomplete native FEM velocity array')
                        tets, rest_det, edges, lengths = geometry[name]
                        ratios = np.linalg.det((p[tets[:, 1:]]-p[tets[:, :1]]).transpose(0, 2, 1))/rest_det
                        stretch = np.linalg.norm(p[edges[:, 1]]-p[edges[:, 0]], axis=1)/lengths
                        parent = spec['parent_rigid_body']; pose = arrays[f'task_rigid_{parent}_q'][i]
                        velocity = arrays[f'task_rigid_{parent}_qd'][i]
                        if pose.shape != (7,) or velocity.shape != (6,) or not np.isfinite(velocity).all():
                            raise ValueError('Incomplete native attachment parent arrays')
                        matrix = transform_xyzw(pose); bond = spec['attachment']
                        targets = np.asarray(bond['rigid_local_positions_m']) @ matrix[:3, :3].T + matrix[:3, 3]
                        error = float(np.linalg.norm(p[np.asarray(bond['node_indices'])]-targets, axis=1).max())
                        metrics = dict(minimum_tetrahedron_volume_ratio=float(ratios.min()), maximum_tetrahedron_volume_ratio=float(ratios.max()),
                            minimum_edge_length_ratio=float(stretch.min()), maximum_edge_length_ratio=float(stretch.max()),
                            maximum_material_bond_error_m=error, maximum_native_node_speed_m_s=float(np.linalg.norm(v, axis=1).max()))
                        for key, value in metrics.items():
                            totals[name][key] = min(totals[name][key], value) if key.startswith('minimum') else max(totals[name][key], value)
                        if ratios.min() < .001 or stretch.min() < .1 or stretch.max() > 3. or error > limits['maximum_penetration_m']:
                            raise ValueError('Native volume geometry or material bond exceeded numerical/scene bounds')
                        histories[name].append(p.copy())
                        stable = (frames > 0 and robot['fully_separated'] and len(histories[name]) == window+1
                                  and np.linalg.norm(p-histories[name][0], axis=1).max()/period <= limits['maximum_linear_speed_m_s'])
                        if stable:
                            if anchors[name] is None: anchors[name] = p.copy()
                            stable = np.linalg.norm(p-anchors[name], axis=1).max() <= limits['maximum_stable_position_range_m']
                        if stable: quiet[name] += dt
                        else: quiet[name], anchors[name] = 0., None
                    previous = (step, native, time); frames += 1
        if (frames != manifest['frame_count'] or first[0] != manifest['first_native_step']
                or previous[0] != manifest['last_native_step'] or previous[0] != manifest['actual_terminal_native_step']
                or endpoints_checked != len(endpoint_steps) or endpoint_steps[0] != first[0] or endpoint_steps[-1] != previous[0]):
            raise ValueError('Native full-tick and endpoint coverage differ')
        if any(value+1.e-10 < limits['stable_s'] for value in quiet.values()):
            raise ValueError('Full native foam arrays lack terminal released stability dwell')
        result.update(passed=True, frames_checked=frames, endpoints_checked=endpoints_checked,
            native_dt_s=dt, native_duration_s=previous[2], volumes=totals,
            terminal_released_quiet_s=quiet, all_robot_convexes_replayed=True,
            maximum_allowed_bond_error_m=limits['maximum_penetration_m'], clearance_m=.005,
            numerical_geometry_limits={'minimum_tetrahedron_volume_ratio': .001,
                                       'minimum_edge_length_ratio': .1, 'maximum_edge_length_ratio': 3.})
    except (KeyError, TypeError, ValueError, OSError, AssertionError, IndexError, BadZipFile, np.linalg.LinAlgError) as error:
        result['errors'].append({'type': type(error).__name__, 'detail': str(error)})
    return result
