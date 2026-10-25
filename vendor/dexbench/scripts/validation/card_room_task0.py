"""Bounded single-arm card-room initialization through public bench functions.

Run from a complete source snapshot and a frozen task release. Only a timed
robot hold is executed. Private saved contacts are inspected after the episode;
they never choose actions. Passing this smoke is not grasp/placement acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from bench.isaacsim.release import digest, dump, task_inputs
from bench.protocol import PolicyRegistry
from bench.runtime.functions import EpisodeFunctions  # Required frozen frontend import.
from bench.runtime.native import NativeWorker
from bench.runtime.service import CreateEpisodeRequest, EpisodeSession
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from scripts.validation.board_game_runtime_smoke import gpu_selection


CAMERAS = {'overview', 'task', 'room', 'wrist'}
PIECES = {'black_00', 'black_01', 'black_02'}
MODULES = ('scripts.validation.card_room_task0', 'bench.runtime.service',
           'bench.runtime.functions', 'bench.runtime.native')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def support_check(metadata, records, declarations, *, dwell_s=.3):
    """Check saved per-substep support contacts, with explicit kinematic limits.

    Contacts are acquired at every native substep by the unchanged worker.
    Positions and velocities are saved only at 20 ms control boundaries; this
    check therefore does not certify every-substep piece pose or placement.
    """
    info = metadata['scene_info']
    require(set(metadata['object_names']) == PIECES, 'Expected exactly three black pieces')
    require(set(declarations) == PIECES, 'Missing frozen per-piece reserve support declarations')
    dt = metadata['physics_dt_s']
    require(math.isfinite(dt) and 0 < dt <= .02, 'Invalid native integration timestep')
    actors = info.get('collider_actor_paths', {})
    paths, supports = {}, {}
    for name in sorted(PIECES):
        paths[name] = info['objects'][name]['path']
        fixture_id = declarations[name]['fixture_id']
        require(fixture_id.startswith('reserve_'), 'Reserve fixture identity is required')
        fixture = info['fixtures'][fixture_id]
        raw = fixture.get('contact_filter_paths', fixture.get('collision_paths', []))
        require(bool(raw), 'Reserve fixture has no contact actor mapping')
        supports[name] = set(actors.get(path, path) for path in raw)
    steps = [record['observation'] for record in records if record.get('kind') == 'step']
    require(bool(steps), 'No acknowledged native control steps')
    end = steps[-1]['sim_time_s']
    require(end + 1e-10 >= dwell_s, 'Insufficient hold duration')
    # Include both sampled endpoints: do not credit the interval before the
    # first observed positive contact toward the required stable dwell.
    first = round((end-dwell_s)/dt)
    last = round(end/dt)
    support_samples = {name: set() for name in PIECES}
    samples = set()
    robot_pairs = []
    maximum_penetration = 0.
    previous_time, previous_native_index = 0., None
    robot_paths = set(info['robot_link_paths'].values())
    for observation in steps:
        contacts = observation['contacts']
        sampling = contacts['sampling']
        start, stop = sampling['start_sim_time_s'], sampling['end_sim_time_s']
        count = round((stop-start)/dt)
        require(contacts['errors'] == [] and sampling['complete'] is True,
                'Incomplete or failed native contact acquisition')
        require(sampling['protocol'] == 'native_rigid_contact_sampling_v1'
                and math.isclose(sampling['physics_dt_s'], dt, abs_tol=1e-12)
                and count == sampling['sampled_substeps']
                and sampling['native_end_step_index']-sampling['native_start_step_index'] == count
                and math.isclose(stop-start, .02, abs_tol=1e-9)
                and math.isclose(stop, observation['sim_time_s'], abs_tol=1e-9)
                and math.isclose(start, previous_time, abs_tol=1e-9),
                'Missing, duplicated or inconsistent native contact interval')
        if previous_native_index is not None:
            require(sampling['native_start_step_index'] == previous_native_index,
                    'Native contact interval is not consecutive')
        previous_time, previous_native_index = stop, sampling['native_end_step_index']
        samples.update(range(round(start/dt)+1, round(stop/dt)+1))
        for contact in contacts['rigid']:
            index = round(contact['time_s']/dt)
            require(start < contact['time_s'] <= stop+1e-10
                    and math.isclose(index*dt, contact['time_s'], abs_tol=1e-9),
                    'Contact is outside its recorded native interval')
            impulse = contact['impulse_magnitude_sum_ns']
            separation = contact.get('min_separation_m')
            require(math.isfinite(impulse) and impulse >= 0 and separation is not None
                    and math.isfinite(separation), 'Invalid native impulse or unavailable separation')
            maximum_penetration = max(maximum_penetration, -separation)
            pair = (contact['actor0'], contact['actor1'])
            if impulse > 1e-8 and any(path in robot_paths for path in pair):
                robot_pairs.append({'time_s': contact['time_s'], 'actors': list(pair),
                                    'impulse_ns': impulse})
            for name, path in paths.items():
                if impulse > 1e-8 and ((pair[0] == path and pair[1] in supports[name])
                                     or (pair[1] == path and pair[0] in supports[name])):
                    support_samples[name].add(index)
    needed = set(range(first, last+1))
    terminal = steps[-1]['objects']
    reset = records[0]['observation']['objects']
    pieces = {}
    for name in sorted(PIECES):
        pose = terminal[name]
        drift = float(np.linalg.norm(np.asarray(pose['position_m'][:2])-reset[name]['position_m'][:2]))
        height_drift = abs(pose['position_m'][2]-reset[name]['position_m'][2])
        w, x, y, z = pose['quaternion_wxyz']
        norm = w*w+x*x+y*y+z*z
        require(math.isfinite(norm) and abs(norm-1.) < .001, 'Invalid terminal piece quaternion')
        tilt = math.acos(max(-1., min(1., 1.-2.*(x*x+y*y)/norm)))
        linear = float(np.linalg.norm(pose['linear_velocity_m_s']))
        angular = float(np.linalg.norm(pose['angular_velocity_rad_s']))
        missing = sorted(needed-support_samples[name])
        pieces[name] = {'passed': not missing and drift <= .003 and height_drift <= .003
                                 and tilt <= math.radians(5.) and linear <= .015 and angular <= .1,
                        'missing_support_substeps': len(missing), 'missing_first_indices': missing[:10],
                        'terminal_position_m': pose['position_m'], 'terminal_xy_drift_m': drift,
                        'terminal_height_drift_m': height_drift, 'terminal_tilt_rad': tilt,
                        'terminal_linear_speed_m_s': linear, 'terminal_angular_speed_rad_s': angular,
                        'declared_fixture_id': declarations[name]['fixture_id'],
                        'native_support_actor_paths': sorted(supports[name])}
    return {'passed': needed <= samples and all(row['passed'] for row in pieces.values())
                      and not robot_pairs and maximum_penetration <= .002,
            'native_contact_samples': len(samples), 'terminal_support_window_s': dwell_s,
            'support_observed_span_s': (last-first)*dt,
            'terminal_support_substeps': len(needed), 'pieces': pieces,
            'robot_contact_count': len(robot_pairs), 'robot_contact_examples': robot_pairs[:10],
            'maximum_reported_penetration_m': maximum_penetration,
            'pose_sample_period_s': .02, 'every_substep_pose_verified': False,
            'scope': 'Native per-substep reserve contact coverage and terminal 20 ms state; no grasp or placement claim'}


def camera_checks(session, invoke, observation):
    require({camera['camera_id'] for camera in observation['cameras']} == CAMERAS,
            'Expected overview/task/room/wrist cameras')
    result = {}
    for camera in observation['cameras']:
        name = camera['camera_id']
        returned = invoke('read_camera', {'observation_sequence': observation['sequence'],
                                          'camera_id': name})['camera']
        require(returned == camera, 'Camera query differs from the public observation')
        revision = int(camera['rgb_uri'].rsplit('/', 2)[-2])
        rgb = session.public_images[(revision, name)].read_bytes()
        depth = session.read_public_depth(revision, name)
        require(hashlib.sha256(rgb).hexdigest() == camera['rgb_sha256']
                and hashlib.sha256(depth).hexdigest() == camera['depth_sha256'],
                'Public RGB-D bytes differ from their published hashes')
        pixels = np.asarray(Image.open(io.BytesIO(rgb)).convert('RGB'))
        metric = np.load(io.BytesIO(depth), allow_pickle=False)
        require(pixels.shape == (camera['height'], camera['width'], 3)
                and float(pixels.std()) >= 1 and metric.shape == pixels.shape[:2]
                and metric.dtype == np.float32 and np.isfinite(metric).all()
                and bool(np.any(metric > 0)), 'Invalid or blank RGB-D camera frame')
        require(camera['sim_time_s'] == camera['depth_sim_time_s'] == observation['sim_time_s']
                and camera['intrinsics_3x3'] is not None and camera['world_from_camera'] is not None,
                'Missing same-frame camera calibration')
        result[name] = {'passed': True, 'resolution': [camera['width'], camera['height']],
                        'rgb_sha256': camera['rgb_sha256'], 'depth_sha256': camera['depth_sha256'],
                        'valid_depth_pixels': int(np.count_nonzero(metric > 0))}
    return {'passed': True, 'cameras': result}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    report = {'schema': 'card_room_task0_public_functions_smoke.v1', 'passed': False,
              'model_calls': 0, 'task_success_claimed': False, 'http_transport_tested': False,
              'scope': 'Single-arm initialization, fixed-target hold, RGB-D and reserve support only',
              'started_at': datetime.now(timezone.utc).isoformat(), 'checks': {}}
    calls, session = [], None
    try:
        source = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
        require(source is not None, 'Run only from a complete frozen source snapshot')
        dump(output/'runtime_imports.json', source)
        report['gpu'] = gpu_selection(args.gpu_uuid)
        report['environment'] = {key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'ISAAC_BENCH_SINGLE_CUDA_DEVICE', 'ISAAC_ALLOW_UNVALIDATED_DRIVER')}
        report['source_manifest_sha256'] = digest(project/'source_snapshot_manifest.json')
        report['release_manifest_sha256'] = digest(args.release/'release_manifest.json')
        entry, scene, task, initial = task_inputs(args.release, args.task_id)
        config = {'task_id': entry['id'], 'seed': args.seed, 'duration_s': args.duration,
                  'script': digest(Path(__file__))}
        request = CreateEpisodeRequest(task_id=entry['id'], seed=args.seed,
            agent_id='operator_card_room_task0_smoke', agent_version='1',
            agent_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            track='direct_joints', budgets={'sim_time_s': args.duration+.1, 'wall_time_s': 900.,
                                          'max_actions': 2, 'max_model_calls': 1, 'max_action_sim_s': args.duration})
        session = EpisodeSession(request, output/'episode', release=args.release, device='cuda:0',
                                 registry=PolicyRegistry(), worker_factory=NativeWorker)
        report['episode_id'] = session.spec.episode_id
        report['checks']['robot_inventory'] = {'passed': False}
        meta = session.metadata
        roots = meta['runtime_robot_base_check']['articulations']
        require(len(meta['joint_names']) == 29 and len(roots) == 1 and roots[0]['dof_count'] == 29
                and roots[0]['fixed_base'] is True and set(meta['object_names']) == PIECES,
                'Expected one fixed 29 DOF articulation and three black pieces')
        report['checks']['robot_inventory'] = {'passed': True, 'physical_joint_count': 29,
                                               'articulations': roots, 'object_names': meta['object_names']}

        def invoke(function, arguments):
            reply = session.functions.invoke({'request_id': 'smoke_'+str(len(calls)),
                'function': function, 'arguments': arguments}).model_dump(mode='json')
            calls.append({'function': function, 'reply': reply})
            require(reply['ok'], 'Public function failed: '+function+': '+str(reply.get('error')))
            return reply['result']

        dump(output/'function_catalog.json', session.functions.catalog().model_dump(mode='json'))
        dump(output/'robot_guide.json', invoke('get_robot', {}))
        observation = invoke('observe', {})
        dump(output/'initial_observation.json', observation)
        report['checks']['initial_cameras'] = camera_checks(session, invoke, observation)
        require(session.clock.sim_time_s == 0., 'Read-only queries advanced physical time')
        require(session.score['success'] is False, 'Reset unexpectedly satisfied placement task')
        invoke('hold', {'action_id': 'initialization_hold', 'observation_sequence': observation['sequence'],
                        'duration_s': args.duration})
        deadline, last_update = time.monotonic()+240., time.monotonic()
        while True:
            status = invoke('get_action', {'action_id': 'initialization_hold'})
            if status['status'] not in ('accepted', 'running'):
                break
            require(time.monotonic() < deadline, 'Bounded hold did not finish')
            if time.monotonic()-last_update > 20:
                print(json.dumps({'event': 'holding', 'elapsed_sim_s': session.clock.sim_time_s}), flush=True)
                last_update = time.monotonic()
            time.sleep(.1)
        require(status['status'] == 'completed', 'Hold did not complete normally: '+str(status))
        require(math.isclose(session.clock.sim_time_s, args.duration, abs_tol=1e-9), 'Incorrect hold duration')
        observation = invoke('observe', {})
        dump(output/'final_observation.json', observation)
        report['checks']['final_cameras'] = camera_checks(session, invoke, observation)
        require(session.score['success'] is False, 'Passive hold incorrectly scored a placement success')
        report['final_score'] = session.score
        invoke('finish_episode', {})
        report['episode_result'] = invoke('get_result', {})
        result = session.result.model_dump(mode='json')
        require(result['outcome'] == 'cancelled' and result['model_calls'] == 0,
                'Smoke must finish cancelled, without model calls or false success')
        require(result['physics_valid'] is True, 'Canonical physics audit failed')
        private = session.directory/'private'
        records = [json.loads(line) for line in (private/'physics.jsonl').read_text().splitlines()]
        declarations = scene['card_room_placement']['reserve_supports']
        report['checks']['reserve_support'] = support_check(meta, records, declarations)
        require(report['checks']['reserve_support']['passed'], 'Reserve support or robot clearance check failed')
        report['checks']['expected_non_success'] = {'passed': True, 'outcome': result['outcome']}
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        if session is not None:
            try:
                session.finish()
            except BaseException as exc:
                report['passed'] = False
                report['shutdown_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
                session.worker.close(force=True)
            report['native_worker_exit_code'] = session.worker.process.returncode
            report['passed'] &= session.worker.process.returncode == 0
        try:
            source = verify_if_snapshot(project, required_modules=MODULES, entrypoint_file=Path(__file__))
            require(source is not None, 'Missing final frozen-source verification')
            dump(output/'runtime_imports_final.json', source)
        except BaseException as exc:
            report['passed'] = False
            report['source_verification_failure'] = str(exc)
        dump(output/'public_function_calls.json', calls)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        dump(output/'qualification.json', report)
    print(json.dumps({'output': str(output), 'passed': report['passed'], 'scope': report['scope']}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--task-id', default='task0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--duration', type=float, default=.5)
    args = parser.parse_args()
    if not .4 <= args.duration <= 1. or not math.isclose(args.duration/.02, round(args.duration/.02), abs_tol=1e-9):
        parser.error('Duration must be 0.4–1.0 seconds and an exact multiple of 20 ms')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
