"""Full native-tick replay of elastic parts, contacts and robot separation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.evaluation.cloth.cloth_robot_collision import reanalyse_export, state_digest
from bench.evaluation.deformable_robot_geometry import RigidCompositeRobotGeometry
from bench.paths import load_json
from bench.simulation.isaacsim_backend.native_task_bundle import add_bundle, load_observer, joint_positions
from scripts.validation.kitchen_scene_evidence import _quaternion_angle


def review_native_elastic(private, runtime, task, evidence, *, expected_source=None):
    """Replay all saved frames; no simulation, pose writes or force compensation."""
    private = Path(private)
    root = private/'native_elastic'
    result = {'passed': False, 'native_execution_performed': False, 'errors': [],
        'scope': 'All native rigid material parts, passive joint geometry, cooked robot clearance and normal support load; no robot manipulation or material calibration claim.'}
    def bind(path):
        evidence[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return evidence[str(path.resolve())]
    def read(path):
        bind(path)
        return load_json(path)
    try:
        metadata, manifest = read(root/'metadata.json'), read(root/'manifest.json')
        if (metadata['schema'] != 'native_elastic_archive.v1' or manifest['schema'] != metadata['schema']
                or metadata['source'] != 'actual_native_tensor_getters' or metadata['reconstructed'] is not False
                or metadata.get('physical_state_setters') != 0
                or manifest['errors'] or manifest['reason'] != 'finish_request'
                or manifest['complete_native_step_coverage'] is not True
                or manifest['metadata_sha256'] != evidence[str((root/'metadata.json').resolve())]):
            raise ValueError('Incomplete, reconstructed or failed elastic archive')
        chunks = [row['path'] for row in manifest['chunks']]
        if len(set(chunks)) != len(chunks) or set(chunks) != {p.name for p in root.glob('*.npz')}:
            raise ValueError('Elastic chunk inventory is incomplete or duplicated')
        source = metadata['runtime_source_verification']
        if expected_source is not None and (not source or source.get('verified') is not True
                or source.get('runtime_imports_verified') is not True
                or Path(source['snapshot_root']).resolve() != Path(expected_source).resolve()):
            raise ValueError('Elastic archive source receipt differs')
        bundle = runtime['scene_spec']['native_task_bundle']
        if (bundle['elastic_composites'] != task['asset_metadata']['elastic_composites']
                or metadata['task_usd_path'] != bundle['usd_path']
                or metadata['task_usd_sha256'] != bundle['usd_sha256']):
            raise ValueError('Native elastic asset identity differs')
        config_sha = hashlib.sha256(json.dumps(bundle['observer']['configuration'], sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if metadata['observer_configuration_sha256'] != config_sha:
            raise ValueError('Elastic observer configuration differs')
        for key in ('sha256', 'files', 'class_name'):
            if metadata['observer_source'][key] != bundle['observer'][key]:
                raise ValueError('Frozen elastic observer source differs')
        names = metadata['object_names']
        members = sorted({n for row in bundle['elastic_composites'].values() for n in row['member_ids']})
        if (names != sorted(task['asset_metadata']['actors']) or len(names) != len(set(names))
                or sorted(metadata['component_member_ids']) != members
                or metadata['object_paths'] != task['asset_metadata']['actors']):
            raise ValueError('Incomplete or changed native part inventory')
        if runtime['body_names'] != metadata['robot_body_names'] + runtime['object_names']:
            raise ValueError('Native robot body order differs from recording')
        from pxr import Usd
        checked_stage = Usd.Stage.CreateInMemory()
        frozen_native = add_bundle(checked_stage, bundle, physics_scene_path='/World/PhysicsScene')
        if (metadata['native_joint_definitions'] != frozen_native['joints']
                or metadata['joint_names'] != sorted(frozen_native['joints'])):
            raise ValueError('Archived joint definitions differ from frozen physical USD')
        checker = RigidCompositeRobotGeometry(reanalyse_export(metadata['robot_cooked_geometry']),
            metadata['robot_body_names'], reanalyse_export(metadata['component_cooked_geometry']), members)
        observer = load_observer(bundle['observer'])
        path = private/'states.npz'; bind(path)
        with np.load(path, allow_pickle=False) as loaded:
            endpoints = {k: loaded[k] for k in loaded.files}
        baseline_hash = state_digest({k: v[0] for k, v in endpoints.items()})
        if any(e['state_hash_before'] != baseline_hash for e in (
                metadata['robot_cooked_geometry'], metadata['component_cooked_geometry'])):
            raise ValueError('Cooked geometry does not bind recorded native baseline')
        endpoint_ids = {int(step): i for i, step in enumerate(endpoints['native_physics_step_index'])}
        if len(endpoint_ids) != len(endpoints['native_physics_step_index']):
            raise ValueError('Duplicate endpoint ticks')
        bind(private/'physics.jsonl')
        records = [json.loads(line) for line in (private/'physics.jsonl').read_text().splitlines() if line.strip()]
        endpoint_states = {row['observation']['task_state']['native_step']: row['observation']['task_state'] for row in records}
        dt = float(runtime['physics_dt_s'])
        limits = task['evaluation']['thresholds']
        end = records[-1]['observation']['sim_time_s']
        terminal, previous, checked_endpoints = [], None, 0
        frame_count, max_penetration, robot_contacts = 0, 0., 0
        pad_path = task['asset_metadata']['actors']['sponge']
        tray_path = task['asset_metadata']['fixtures']['sponge_tray']
        def matches(path, root):
            return path == root or path.startswith(root+'/')
        for chunk in manifest['chunks']:
            if Path(chunk['path']).name != chunk['path'] or not chunk['path'].endswith('.npz'):
                raise ValueError('Invalid native chunk path')
            path = root/chunk['path']
            if bind(path) != chunk['sha256']:
                raise ValueError('Native elastic chunk hash differs')
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {k: loaded[k] for k in loaded.files}
            count = len(arrays['native_step'])
            if (not np.issubdtype(arrays['native_step'].dtype, np.integer)
                    or arrays['baseline'].dtype.kind != 'b'
                    or count != chunk['frames'] or not count or any(len(v) != count for v in arrays.values())
                    or arrays['native_step'][0] != chunk['first_native_step']
                    or arrays['native_step'][-1] != chunk['last_native_step']):
                raise ValueError('Native chunk coverage differs')
            for i in range(count):
                step, time, nt = int(arrays['native_step'][i]), float(arrays['time_s'][i]), float(arrays['native_time_s'][i])
                baseline = bool(arrays['baseline'][i])
                if (not np.isfinite([time, nt]).all() or float(arrays['physics_dt_s'][i]) != dt
                        or baseline != (frame_count == 0)
                        or (frame_count == 0 and (time != 0. or step != manifest['first_native_step']
                                                or step != min(endpoint_ids)))
                        or previous and (step != previous[0]+1 or abs(time-previous[1]-dt) > 1.e-8
                                         or abs(nt-previous[2]-dt) > 1.e-8)):
                    raise ValueError('Missing, duplicated or mixed native ticks')
                q, qd = arrays['object_q'][i], arrays['object_qd'][i]
                if (q.shape != (len(names), 7) or qd.shape != (len(names), 6)
                        or not np.isfinite(q).all() or not np.isfinite(qd).all()):
                    raise ValueError('Incomplete finite native part state')
                objects = {name: {'position_m': q[j, :3].tolist(),
                    'quaternion_wxyz': q[j, [6, 3, 4, 5]].tolist(),
                    'linear_velocity_m_s': qd[j, :3].tolist(), 'angular_velocity_rad_s': qd[j, 3:].tolist()}
                    for j, name in enumerate(names)}
                robot = checker.check(objects, arrays['robot_body_q'][i])
                if robot != json.loads(str(arrays['elastic_robot_geometry_json'][i])):
                    raise ValueError('Cooked clearance replay differs')
                joints = json.loads(str(arrays['joints_json'][i]))
                if joint_positions(objects, metadata['native_joint_definitions']) != joints:
                    raise ValueError('Passive joint geometry replay differs')
                contacts = json.loads(str(arrays['contacts_json'][i]))
                frame = dict(baseline=baseline, native_step=step, time_s=time, native_time_s=nt,
                    physics_dt_s=dt, objects=objects, joints=joints, contacts=contacts, elastic_robot_geometry=robot)
                observer.on_frame(frame)
                observed = observer.on_observe()
                if observed.get('errors') or observed.get('complete') is not True or observed.get('safety_stop'):
                    raise ValueError('Full-tick elastic observer replay is invalid')
                normal = 0.
                for row in contacts:
                    pair = row['actor0'], row['actor1']
                    involved = any(matches(p, body) for p in pair for body in metadata['object_paths'].values())
                    if not involved:
                        continue
                    max_penetration = max(max_penetration, -row['min_separation_m'])
                    robot_contacts += int(any(p.startswith('/World/Robot/') for p in pair))
                    for which in (0, 1):
                        if matches(pair[which], pad_path) and matches(pair[1-which], tray_path):
                            normal += row['impulse_vector_ns'][2]*(1 if which == 0 else -1)
                if time >= end-limits['stable_s']-1.e-10:
                    terminal.append((time, q.copy(), qd.copy(), normal, observed))
                if step in endpoint_ids:
                    k = endpoint_ids[step]; checked_endpoints += 1
                    order = [runtime['object_names'].index(name) for name in names]
                    if (not np.array_equal(q, endpoints['object_q'][k, order])
                            or not np.array_equal(qd, endpoints['object_qd'][k, order])
                            or not np.array_equal(arrays['robot_body_q'][i], endpoints['body_q'][k, :len(metadata['robot_body_names'])])
                            or observed != endpoint_states[step]):
                        raise ValueError('Full native archive differs from control endpoint')
                previous = step, time, nt
                frame_count += 1
        if (frame_count != manifest['frame_count']
                or previous[0] != manifest['actual_terminal_native_step']
                or previous[0] != manifest['last_native_step'] or previous[0] != max(endpoint_ids)
                or checked_endpoints != len(endpoint_ids) or len(terminal) < round(limits['stable_s']/dt)):
            raise ValueError('Incomplete final native coverage')
        poses = np.array([r[1] for r in terminal]); velocities = np.array([r[2] for r in terminal])
        linear = np.linalg.norm(velocities[..., :3], axis=-1).max(0)
        angular = np.linalg.norm(velocities[..., 3:], axis=-1).max(0)
        window = round(limits['stability_pose_window_s']/dt)
        pose_linear = np.linalg.norm(poses[window:, :, :3]-poses[:-window, :, :3], axis=-1).max(0)/(window*dt)
        pose_angular = _quaternion_angle(poses[window:, :, 3:], poses[:-window, :, 3:]).max(0)/(window*dt)
        supported = all(r[4]['native_support_by_actor']['sponge']['supported']
                        and r[4]['detail']['elastic_geometry']['quiet']
                        and r[4]['detail']['elastic_geometry']['fully_separated'] for r in terminal)
        total_mass = sum(r['mass_kg'] for r in task['asset_metadata']['objects'])
        normal = sum(r[3] for r in terminal)/(len(terminal)*dt)
        ratio = normal/(total_mass*9.81)
        passed = (supported and not robot_contacts and max_penetration <= limits['maximum_penetration_m']
                  and np.all(linear <= limits['maximum_linear_speed_m_s'])
                  and np.all(angular <= limits['maximum_angular_speed_rad_s'])
                  and np.all(pose_linear <= limits['maximum_linear_speed_m_s'])
                  and np.all(pose_angular <= limits['maximum_angular_speed_rad_s']) and abs(ratio-1) <= .15)
        result.update(passed=bool(passed), frames=frame_count, endpoint_frames_verified=checked_endpoints,
            elapsed_sim_s=end, terminal_native_frames=len(terminal), total_mass_kg=total_mass,
            terminal_normal_z_force_n=normal, normal_z_to_weight_ratio=ratio,
            terminal_support_and_full_boundary_release=supported, maximum_penetration_m=max_penetration,
            robot_contact_rows=robot_contacts, part_count=len(names),
            parts={name: {'maximum_raw_linear_m_s': float(linear[i]), 'maximum_raw_angular_rad_s': float(angular[i]),
                         'maximum_rolling_linear_m_s': float(pose_linear[i]), 'maximum_rolling_angular_rad_s': float(pose_angular[i])}
                   for i, name in enumerate(names)})
    except (KeyError, TypeError, ValueError, OSError, AssertionError) as exc:
        result['errors'].append({'type': type(exc).__name__, 'detail': str(exc)})
    return result
