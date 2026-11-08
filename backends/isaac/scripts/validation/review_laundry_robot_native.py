"""Replay a completed laundry robot HTTP smoke using every recorded native frame.

This re-executes the frozen evaluator on archived q/qd and the original recorded
geometry evidence. It does not rerun the GPU/robot geometry observer, measure
contact force, or qualify grasping/folding/stacking policy success.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from bench.isaacsim.release import load_evaluator, validate
from bench.paths import DATA_ROOT, load_json
from scripts.maintenance.freeze_project_sources import verify
from scenes.workrooms.laundry_room.rules import ALL_TASK_IDS as TASK_IDS

SCHEMA = 'laundry_robot_native_referee_replay.v1'
DATASETS = Path(f'{DATA_ROOT}/datasets')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return load_json(Path(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def active_sides(task):
    configuration = task.get('robot_configuration_id')
    if configuration is None:
        return ('left', 'right')
    require(configuration in ('single_arm', 'dual_same_side', 'dual_opposed'),
            'Unknown configured robot installation')
    sides = ('right',) if configuration == 'single_arm' else ('left', 'right')
    require(task.get('active_sides') == list(sides), 'Task active robot sides differ from installation')
    return sides


def selected_inputs(q, release, manifest):
    """Resolve the actual physical seed, retaining legacy canonical semantics."""
    if manifest.get('schema') != 'laundry_randomized_release.v1':
        return read(release/'task.json'), read(release/'scene.json')
    from bench.isaacsim.release import task_inputs
    seed = q.get('seed')
    require(type(seed) is int and 0 <= seed <= 9, 'Missing or unsupported physical laundry seed')
    _, scene, task, _ = task_inputs(release, q['task_id'], seed=seed)
    reset = task.get('reset_case', {})
    require(reset.get('schema') == 'laundry_reset_case.v1' and reset.get('seed') == seed
            and reset.get('task_id') == q['task_id'], 'Resolved physical laundry case differs')
    return task, scene


def verify_selection(q, release, task):
    """Bind discovery base and selected immutable input separately."""
    base = Path(q['task_release']).resolve()
    require(q['release_manifest_sha256'] == sha(base/'release_manifest.json'),
            'Native discovery base release changed')
    selection = q.get('configuration_selection')
    if selection is None:
        require(base == release and task.get('robot_configuration_id') is None,
                'Native release binding differs or configured selection is missing')
        return [base/'release_manifest.json']
    pin, configuration = selection['source_release'], task.get('robot_configuration_id')
    expected = sha(release/'release_manifest.json')
    require(configuration is not None and selection.get('family') == 'laundry_room'
            and selection.get('task_id') == task['id']
            and selection.get('configuration_id') == configuration
            and Path(pin['path']).resolve() == release
            and pin['manifest_sha256'] == q.get('selected_release_manifest_sha256') == expected,
            'Native selected configuration/release binding differs')
    for published in (q['selected_task'], q['created']['spec']):
        require(published['task_id'] == task['id']
                and published.get('configuration_id') == configuration
                and published['task_release_sha256'] == expected,
                'Public discovery/create selected input differs')
    require(read(base/'task.json')['id'] == task['id'], 'Discovery base uses another task')
    return [base/'release_manifest.json']


def archived_states(archive):
    """Stream recorded chunks, rejecting duplicate/unsafe members and gaps."""
    names = archive.namelist()
    require(len(names) == len(set(names)), 'Duplicate native archive members')
    chunks = sorted(name for name in names if re.fullmatch(r'states/[0-9]{6}\.npz', name))
    require(set(names) == {'topology.json', 'robot_colliders.json', *chunks} and chunks,
            'Missing or undeclared native archive member')
    inventory = None
    for index, name in enumerate(chunks):
        require(name == f'states/{index:06d}.npz', 'Native state archive has a missing chunk')
        with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as arrays:
            keys = set(arrays.files)
            require(inventory is None or inventory == keys, 'Native chunk field inventory changed')
            inventory = keys
            required = {'laundry_observed_time_s', 'laundry_native_step', 'native_physics_step_index',
                        'native_world_time_s', 'joint_q', 'joint_qd', 'body_q', 'body_qd'}
            require(required <= keys, 'Native chunk omits required clocks, joints or body poses')
            length = len(arrays['laundry_native_step'])
            require(0 < length <= 128 and (index == len(chunks)-1 or length == 128),
                    'Unexpected native chunk length')
            require(all(arrays[key].shape[0] == length and arrays[key].dtype.kind in 'biuf'
                        and np.isfinite(arrays[key]).all() for key in keys),
                    'Nonfinite, nonnumeric or misaligned native arrays')
            for offset in range(length):
                yield {key: arrays[key][offset].copy() for key in keys}


def replay(archive_path, trace_path, evaluator, task, *, control_scores=None, control_states=None,
           body_names=None):
    """Exact scorer replay; all collision/constraint observations stay recorded."""
    sides = active_sides(task)
    dofs = 29*len(sides)
    specs = {row['id']: row for row in task['asset_metadata']['cloths']}
    dt = task['evaluation']['physics_dt_s']
    expected_keys = set(specs)
    samples = 0
    previous_native_step = previous_native_time = None
    first_joint = None
    joint_span = None
    control_checked = set()
    final = None
    maxima = {'cloth_penetration_m': 0., 'table_penetration_m': 0., 'robot_penetration_m': 0.}
    intersection_frames = near_frames = separated_frames = 0
    with zipfile.ZipFile(archive_path) as archive, Path(trace_path).open() as trace:
        topology = json.loads(archive.read('topology.json'))
        require(set(topology) == expected_keys and all(np.array_equal(topology[name], specs[name]['triangles'])
                for name in specs), 'Archived cloth topology differs from frozen task')
        export_bytes = archive.read('robot_colliders.json')
        export = json.loads(export_bytes)
        require(export.get('status') == 'complete' and not export.get('issues')
                and export.get('visual_mesh_fallback') is False
                and export.get('state_hash_before') == export.get('state_hash_after')
                and export.get('layer_hashes_before') == export.get('layer_hashes_after'),
                'Recorded native robot geometry export was not complete/read-only')
        for index, state in enumerate(archived_states(archive)):
            line = trace.readline()
            require(bool(line), 'Native geometry trace is shorter than state archive')
            row = json.loads(line)
            t = float(state['laundry_observed_time_s'])
            native_step, native_time = int(state['native_physics_step_index']), float(state['native_world_time_s'])
            require(int(state['laundry_native_step']) == index and row['native_step'] == index
                    and abs(t-index*dt) <= 1.e-8 and abs(row['time_s']-t) <= 1.e-8
                    and row['native_physics_step_index'] == native_step
                    and abs(row['native_world_time_s']-native_time) <= 1.e-8,
                    f'Mixed-tick native/trace clocks at frame {index}')
            if previous_native_step is not None:
                require(native_step == previous_native_step+1
                        and abs(native_time-previous_native_time-dt) <= 1.e-8,
                        f'Missing/repeated native physics step at frame {index}')
            previous_native_step, previous_native_time = native_step, native_time
            require(row.get('schema') == 'laundry_native_frame.v1' and row.get('complete') is True
                    and row.get('control_mode') == 'robot_joint_targets'
                    and row.get('controlled_vertices') == {name: [] for name in specs},
                    f'Native joint-only cloth evidence changed at frame {index}')
            physics, geometry, interaction = row['physics'], row['geometry_diagnostics'], row['robot_interaction']
            if body_names is not None:
                require(len(set(body_names)) == len(body_names)
                        and state['body_q'].shape == (len(body_names), 7),
                        'Native body pose inventory differs from worker metadata')
                require(set(interaction['hands']) == set(sides), 'Recorded hand inventory differs from installation')
                require(len(sides) != 1 or 'left_hand_flange' not in body_names, 'Single-arm body inventory contains a hidden left hand')
                for side in sides:
                    wrist = body_names.index(side+'_hand_flange')
                    require(interaction['hands'][side]['wrist_body_name'] == side+'_hand_flange',
                            'Wrist evidence omits its actual native flange body identity')
                    require(np.array_equal(interaction['hands'][side]['wrist_position_m'],
                                           state['body_q'][wrist, :3]),
                            'Recorded scorer wrist position differs from actual native body pose')
            require(physics.get('complete') is True and physics.get('valid') is True
                    and physics.get('gravity_enabled') is True and physics.get('hidden_constraints') == 0
                    and geometry.get('finite_thickness_compression_checked') is True
                    and interaction.get('complete') is True
                    and interaction.get('contact_force_measured') is False
                    and interaction.get('robot_cloth_penetration_checked') is True
                    and interaction.get('robot_cloth_geometry_uncertain') is False,
                    f'Incomplete recorded geometry/constraint evidence at frame {index}')
            for key in ('self_intersections', 'intercloth_intersections'):
                require(type(geometry[key]) is int and geometry[key] >= 0 and physics[key] == geometry[key],
                        'Geometry counts differ from recorded scorer input')
            depths = [geometry['max_cloth_envelope_penetration_m'],
                      geometry['table_envelope_penetration_m'], interaction['robot_cloth_penetration_m']]
            require(all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in depths)
                    and abs(physics['max_penetration_m']-max(depths)) <= 1.e-12,
                    'Geometry depths differ from recorded scorer input')
            maxima = {key: max(maxima[key], value) for key, value in zip(maxima, depths)}
            intersection_frames += int(bool(geometry['self_intersections'] or geometry['intercloth_intersections']))
            near_frames += int(interaction['hand_near_cloth'])
            separated_frames += int(interaction['fully_separated'])
            frame = {key: deepcopy(value) for key, value in row.items() if key != 'score'}
            frame['cloths'] = {}
            for name, spec in specs.items():
                points, velocities = state['cloth_'+name+'_q'], state['cloth_'+name+'_qd']
                require(points.shape == velocities.shape == (len(spec['rest_vertices_m']), 3),
                        'Native cloth node count changed')
                frame['cloths'][name] = {'enabled': True, 'vertices_m': points,
                    'velocities_m_s': velocities, 'triangles': np.asarray(topology[name], dtype=int)}
            final = evaluator.observe(frame)
            require(final == row['score'], f'Frozen evaluator replay differs at native frame {index}')
            require(not final['success'] and final.get('referee', {}).get('state') == 'ongoing',
                    'Interface smoke unexpectedly ended or earned task success')
            require(not final.get('detail', {}).get('operation_seen', False)
                    and not final.get('detail', {}).get('operation_displacement_seen', False),
                    'Interface micro-motion unexpectedly earned cloth manipulation credit')
            if first_joint is None:
                first_joint = np.asarray(state['joint_q'], dtype=float).copy()
                joint_span = np.zeros_like(first_joint)
            require(state['joint_q'].shape == first_joint.shape == (dofs,), 'Native joint inventory differs from installation')
            joint_span = np.maximum(joint_span, np.abs(state['joint_q']-first_joint))
            if control_scores is not None and index in control_scores:
                require(control_scores[index] == final, 'Control-period score differs from native replay')
                target = control_states[index]
                for key in target:
                    if key == 'time_s':
                        require(abs(float(target[key])-t) <= 1.e-8, 'Public control/native time mismatch')
                    else:
                        require(key in state and np.array_equal(target[key], state[key]),
                                'Control archive differs from native state: '+key)
                control_checked.add(index)
            samples += 1
        require(not trace.read().strip(), 'Native geometry trace has extra frames')
    require(samples > 1, 'Native replay requires real stepping beyond reset')
    if control_scores is not None:
        require(control_checked == set(control_scores), 'Missing native state for a control-period score')
    return {'passed': True, 'samples_checked': samples, 'native_steps': samples-1,
        'elapsed_sim_s': (samples-1)*dt, 'control_scores_checked': len(control_checked),
        'final_evaluator_result': final, 'native_geometry_recomputed': False,
        'robot_colliders_member_sha256': hashlib.sha256(export_bytes).hexdigest(),
        'recorded_geometry_summary': {'maxima': maxima, 'intersection_frames': intersection_frames,
            'hand_near_frames': near_frames, 'fully_separated_frames': separated_frames},
        'joint_span_rad': joint_span.tolist(),
        'scope': 'Every archived native q/qd frame through the frozen evaluator with the original recorded geometry/constraint inputs; not an independent rerun of the collision observer.'}


def review(http, release, episode_dir=None):
    from PIL import Image
    from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand
    from bench.simulation.isaacsim_backend.scene_variants import apply_scene_variant
    http, release = Path(http).resolve(), Path(release).resolve()
    q = read(http/'qualification.json')
    episode = Path(episode_dir).resolve() if episode_dir else http/'episodes'/q['episode_id']
    require(episode.name == q['episode_id'], 'Requested episode differs from native smoke')
    private = episode/'private'
    runtime = private/'task_runtime'
    require(q['passed'] is True and q['model_calls'] == 0 and q['private_state_used_for_targets'] is False,
            'Expected completed public-only HTTP smoke without model calls')
    require(q['server_exit_code'] == 0 and
            q['server_shutdown']['episodes'][q['episode_id']]['worker_exit_code'] == 0,
            'Native/server processes did not exit normally')
    require(q['result']['outcome'] == 'cancelled' and q['result']['physics_valid'] is True
            and q['result']['model_calls'] == 0, 'Expected cancelled valid interface episode')
    manifest = validate(release, assets=False)
    metadata, task, scene, config = [read(private/name) for name in
        ('metadata.json', 'task.json', 'scene.json', 'worker_config.json')]
    original_task, original_scene = selected_inputs(q, release, manifest)
    selection_files = verify_selection(q, release, original_task)
    sides = active_sides(original_task)
    dofs = 29*len(sides)
    require(task == original_task and task['id'] == q['task_id']
            and task['id'] in TASK_IDS, 'Worker task differs from frozen robot task')
    require(task['evaluation']['control_mode'] == 'robot_joint_targets'
            and task['capabilities']['robot_episodes'] is True, 'Frozen release is not a robot task')
    expected_scene = apply_scene_variant(scene_with_hand(original_scene, config['hand_id']),
                                         config.get('scene_variant', 'original_v1'))
    require(scene == expected_scene and scene['named_cloth_initialization'] == 'author-only',
            'Worker scene differs from frozen robot scene')
    require(config['hand_id'] == original_scene['robot']['hand_id'] and Path(config['release']).resolve() == release,
            'Worker embodiment or release differs')
    imports = read(private/'runtime_imports.json')
    source = Path(imports['snapshot_root']).resolve()
    source_check = verify(source)
    for receipt in (imports, q['runtime_imports'], q['runtime_imports_final']):
        require(receipt.get('verified') is True and receipt.get('runtime_imports_verified') is True
                and Path(receipt['snapshot_root']).resolve() == source, 'Missing/mixed native frozen source receipts')
    frozen_observer = release/'evaluator/native_observer.py'
    require(sha(frozen_observer) == manifest['files']['evaluator/native_observer.py']['sha256']
            and sha(source/'scenes/workrooms/laundry_room/native_observer.py') == sha(frozen_observer),
            'Frozen observer differs from native source snapshot')
    receipt = read(runtime/'laundry_sampling_receipt.json')
    exchange = receipt['contact_exchange']
    require(receipt['complete'] is True and receipt['public_cloth_state_write_interface'] is False
            and exchange['all_samples_consumed_once'] is True
            and exchange['acquisitions'] == exchange['deliveries'] == receipt['native_steps'],
            'Incomplete native sampling or exposed cloth write interface')
    native = receipt['native_observer']
    initialization = native['initialization']
    require(initialization == metadata['named_cloth_initialization']
            and initialization['mode'] == 'author-only' and initialization['native_warmup_preserved'] is True
            and initialization['position_write_calls'] == initialization['velocity_write_calls'] == 0
            and native['cloth_position_write_calls'] == native['cloth_velocity_write_calls'] == 0
            and native['cloth_attachments'] == [] and native['contact_force_measured'] is False,
            'Missing zero-write authored initialization and ordinary joint-only boundary')
    require(native['native_physics']['hidden_cloth_constraints'] == 0
            and native['native_physics']['gpu_dynamics'] is True
            and np.allclose(native['native_physics']['gravity_m_s2'], [0, 0, -9.81], atol=1.e-6),
            'Missing recorded native gravity/GPU/constraint evidence')
    records = [json.loads(line) for line in (private/'physics.jsonl').read_text().splitlines()]
    controls = [row for row in records if row['kind'] in ('reset', 'step')]
    dt = metadata['physics_dt_s']
    scores, control_states = {}, {}
    with np.load(private/'states.npz', allow_pickle=False) as states:
        require(len(states['time_s']) == len(controls), 'Control archive and receipts differ')
        for offset, row in enumerate(controls):
            t = float(states['time_s'][offset])
            step = round(t/dt)
            require(step not in scores and abs(t-step*dt) <= 1.e-8, 'Duplicate/off-grid control sample')
            scores[step] = row['score']
            control_states[step] = {key: states[key][offset].copy() for key in states.files}
    checked = replay(runtime/'laundry_native_evidence.zip', runtime/'laundry_native_frames.jsonl',
        load_evaluator(release)(task), task, control_scores=scores, control_states=control_states,
        body_names=metadata['body_names'])
    require(checked['native_steps'] == receipt['native_steps']
            and checked['final_evaluator_result'] == receipt['final_score']
            == read(private/'final_score.json') == read(private/'evaluator_evidence.json'),
            'Final native, worker, host or replay score differs')
    roots = metadata['runtime_robot_base_check']
    require(roots['verification_passed'] and roots['all_fixed_base']
            and len(metadata['joint_names']) == dofs and len(set(metadata['joint_names'])) == dofs
            and len(roots['articulations']) == len(sides)
            and all(row['fixed_base'] and row['dof_count'] == 29 and row['root_pose_verified']
                    for row in roots['articulations']), 'Native fixed robot installation differs')
    motion = q['motion']
    require(motion['same_action_for_all_hands'] and motion['action_idempotency_checked']
            and set(motion['hands']) == set(sides), 'Missing same-action motion for each configured hand')
    require(q['ik']['solution_found'] and not q['ik']['advances_physics'] and not q['ik']['mutates_controller'],
            'Current-pose public IK query failed or mutated state')
    for side in sides:
        prefix = 'fr3_joint' if len(sides) == 1 else 'fr3_'+side+'_joint'
        indices = [i for i, name in enumerate(metadata['joint_names']) if name.startswith(prefix)]
        require(len(indices) == 7 and max(checked['joint_span_rad'][i] for i in indices) > 1.e-4,
                'No measured native arm micro-motion for '+side)
    probes = q['agent_interface_checks']
    require({row['phase'] for row in probes} == {'created', 'before_finish'}
            and all(row['family'] == 'laundry_room' and row['catalog_release_verified']
                    and row['agent_ready_contract_verified'] and row['read_only_verified']
                    and row['model_calls'] == 0 for row in probes), 'Missing actual Agent client interface checks')
    camera_ids = {'overview', 'task', 'room'} | ({'wrist'} if len(sides) == 1 else {'left_wrist', 'right_wrist'})
    require(set(metadata['agent_sensor_config']['rgb']) == camera_ids
            and {row['camera_id'] for row in q['media']} == camera_ids, 'Missing configured RGB-D cameras')
    camera_checks = []
    files = [*selection_files, http/'qualification.json', release/'release_manifest.json', frozen_observer,
             source/'source_snapshot_manifest.json', Path(__file__),
             *(private/name for name in ('metadata.json', 'scene.json', 'task.json', 'worker_config.json',
                'runtime_imports.json', 'physics.jsonl', 'states.npz', 'final_score.json', 'evaluator_evidence.json')),
             *(runtime/name for name in ('laundry_native_evidence.zip', 'laundry_native_frames.jsonl',
                                        'laundry_sampling_receipt.json'))]
    for row in q['media']:
        for kind in ('rgb', 'depth'):
            require(sha(row[kind]['path']) == row[kind]['sha256'], 'Recorded camera bytes changed')
            files.append(Path(row[kind]['path']))
        rgb = np.asarray(Image.open(row['rgb']['path']).convert('RGB'))
        depth = np.load(row['depth']['path'], allow_pickle=False)
        require(depth.shape == rgb.shape[:2] and depth.dtype == np.float32 and np.isfinite(depth).all()
                and np.any(depth > 0) and rgb.std() >= 1., 'Blank or invalid RGB-D camera')
        camera_checks.append({'camera_id': row['camera_id'], 'rgb_shape': list(rgb.shape),
                              'depth_valid_pixels': int(np.count_nonzero(depth > 0))})
    return {'schema': SCHEMA if task.get('robot_configuration_id') is None else 'laundry_robot_native_referee_replay.v2',
        'configuration_id': task.get('robot_configuration_id'), 'active_sides': list(sides),
        'passed': True, 'http_output': str(http), 'episode': str(episode),
        'release': str(release), 'release_manifest_sha256': sha(release/'release_manifest.json'),
        'native_source': str(source), 'native_source_verification': source_check,
        'checks': {'frozen_evaluator_replay': checked, 'author_only_zero_writes': initialization,
            'native_robot': roots, 'public_motion': motion,
            ('five_rgbd' if task.get('robot_configuration_id') is None else 'configured_rgbd'): camera_checks,
            'agent_client_probes': probes},
        'evidence_sha256': {str(path.resolve()): sha(path) for path in files},
        'limits': ['Recorded geometry is reused unchanged; this review does not independently rerun the geometry observer.',
            'Interface micro-motion and current-wrist public IK only; full task workspace IK is a separate report.',
            'No cloth manipulation, robot grasp, autonomous task success, measured cloth force or calibrated soft-body physics is claimed.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--http', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--episode-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(DATASETS) and not output.exists(), 'Choose a fresh review output under shared datasets')
    report = review(args.http, args.release, args.episode_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': report['passed'], 'output': str(output)}))


if __name__ == '__main__':
    main()
