"""Bounded operator smoke of the canonical authenticated tabletop runtime.

Run only from a complete source snapshot, with an explicitly selected GPU UUID.
This uses public robot functions and robot target holds, zero model calls, and
no object initialization overrides. It does not test grasp, placement or play.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

from bench.isaacsim.release import digest
from bench.paths import OUTPUT_ROOT, load_json
from bench.protocol.schemas import Budgets
from bench.runtime.service import AgentService, CreateBoardGameRequest, ServiceError
from scenes.workrooms.catalog import task_definition
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


REQUIRED_CAMERAS = {'overview', 'task', 'room', 'wrist_left', 'wrist_right'}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def check(value, message):
    if not value:
        raise AssertionError(message)


def gpu_selection(uuid):
    if not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('CUDA_VISIBLE_DEVICES must equal the explicit --gpu-uuid')
    if os.environ.get('ISAAC_BENCH_SINGLE_CUDA_DEVICE') != '1':
        raise ValueError('Single UUID selection requires ISAAC_BENCH_SINGLE_CUDA_DEVICE=1')
    if os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER') != '1':
        raise ValueError('This host requires the recorded explicit unvalidated-driver override')
    query = subprocess.run(['nvidia-smi', '--id='+uuid,
        '--query-gpu=uuid,pci.bus_id,name,memory.total,memory.free,driver_version',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    row = [value.strip() for value in query.stdout.strip().split(',')]
    if len(row) != 6 or row[0] != uuid:
        raise ValueError('GPU identity query disagrees with the selected UUID')
    bus = row[1].upper().split(':')[-2]
    if bus in ('67', 'BB', 'DB'):
        raise ValueError('This GPU has a recorded rendering failure and is excluded from this smoke')
    if int(row[4]) < 16000:
        raise ValueError('Less than 16 GB free on the selected shared GPU; do not stop other jobs')
    return dict(zip(('uuid', 'pci_bus_id', 'name', 'memory_total_mb', 'memory_free_mb', 'driver_version'), row))


def native_coverage(private, elapsed_s):
    """Independently check saved private substep evidence, never feed it to actions."""
    private = Path(private)
    evidence_path = private/'board_game/evidence.json'
    evidence = load_json(evidence_path)
    summary = evidence['observer']
    metadata = load_json(private/'metadata.json')
    dt = metadata['physics_dt_s']
    expected = round(elapsed_s/dt)
    path = private/'board_game/native_frames.jsonl'
    rows = 0
    previous = None
    inventory = None
    complete = consecutive = True
    for line in path.open():
        sample = json.loads(line)
        frame = sample['frame']
        names = set(frame['pieces'])
        inventory = names if inventory is None else inventory
        complete &= (frame['contacts_complete'] is True and not sample['unsupported']
                     and names == inventory and frame['source'] == 'native_solver_substeps'
                     and math.isclose(frame['dt_s'], dt, rel_tol=0., abs_tol=1e-12))
        if previous is not None:
            consecutive &= (frame['substep_index'] == previous['substep_index']+1
                            and math.isclose(frame['simulation_time_s']-previous['simulation_time_s'], dt,
                                             rel_tol=0., abs_tol=1e-9))
        previous = frame
        rows += 1
    final_time_matches = bool(previous and math.isclose(previous['simulation_time_s'], elapsed_s,
                                                       rel_tol=0., abs_tol=1e-8))
    report = {'expected_samples': expected, 'recorded_samples': rows,
        'observer_samples': summary['samples'], 'all_samples_complete': bool(complete),
        'consecutive_native_samples': bool(consecutive), 'inventory_count': len(inventory or ()),
        'expected_inventory_count': summary['batch_view_token_count'],
        'final_context_verified': summary['final_context_verified'], 'observer_failed': summary['failed'],
        'unsupported_samples': summary['unsupported_samples'], 'commits': summary['commits'],
        'frame_sha256': digest(path), 'evidence_sha256': digest(evidence_path),
        'dt_s': dt, 'physical_steps_added_by_observer': summary['physical_steps_added'],
        'object_state_writes': evidence['object_state_writes'], 'final_time_matches_ack': final_time_matches}
    report['passed'] = bool(rows == expected == summary['samples'] and rows > 0 and complete and consecutive
                            and summary['final_context_verified'] is True and not summary['failed']
                            and summary['unsupported_samples'] == 0 and summary['commits'] == 0
                            and summary['physical_steps_added'] == 0 and not evidence['object_state_writes']
                            and final_time_matches and len(inventory or ()) == summary['batch_view_token_count'])
    return report


def run(args):
    # The verifier checks loaded module origins and never imports missing modules
    # itself. Load the two adapters before claiming their runtime provenance.
    from bench.runtime import board_games, board_game_worker
    project = Path(__file__).absolute().parents[2]
    source = verify_if_snapshot(project,
        required_modules=('scripts.validation.board_game_runtime_smoke', 'bench.runtime.service',
                          'bench.runtime.board_games', 'bench.runtime.board_game_worker'),
        entrypoint_file=Path(__file__))
    if source is None or Path.cwd().resolve() != project.resolve():
        raise ValueError('Run from the complete snapshot root; editable source execution is forbidden')
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT):
        raise ValueError('Native smoke artifacts belong under shared datasets')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'board_game_runtime_smoke.v1', 'passed': False,
        'started_at': datetime.now(timezone.utc).isoformat(), 'task_id': args.game,
        'track': args.track, 'termination_case': args.termination, 'requested_hold_s': args.duration,
        'model_calls': 0, 'object_pose_overrides': False, 'privileged_state_used_for_actions': False,
        'scope': 'Canonical authenticated runtime, owned robot hold, camera and substep evidence only.',
        'grasp_qualified': False, 'placement_qualified': False, 'full_game_qualified': False,
        'http_transport_tested': False, 'unvalidated_driver_override': True,
        'source_snapshot': str(project), 'source_manifest_sha256': digest(project/'source_snapshot_manifest.json')}
    service = None
    session = None
    calls = []
    try:
        report['gpu'] = gpu_selection(args.gpu_uuid)
        dump(output/'runtime_imports.json', source)
        definition = task_definition('card_room', args.game)
        release = Path(definition['source_release']['path'])
        report['release'] = definition['source_release']
        service = AgentService(output/'episodes', release=release, device='cuda:0')
        config = {'track': args.track, 'termination': args.termination, 'duration_s': args.duration,
                  'task_id': args.game, 'native_only_smoke': True}
        request = CreateBoardGameRequest(task_id=args.game, agent_id='operator_runtime_smoke',
            agent_version='1', agent_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            track=args.track,
            **({'policy_id': None, 'policy_version': None} if args.track == 'direct_joints' else {}),
            budgets=Budgets(sim_time_s=args.duration if args.termination == 'time_limit' else args.duration+.02,
                            wall_time_s=900., max_actions=8, max_model_calls=1, max_action_sim_s=.4))
        created = service.create(request, board_game=True)
        episode_id = created['episode_id']
        session = service.sessions[episode_id]
        report['episode_id'] = episode_id
        report['episode_directory'] = str(session.directory)
        report['spec'] = created['spec']
        # Player capabilities are kept in memory, never copied into evidence.
        credentials = {row['binding']['player_id']: row for row in created['players']}
        players = {player: service.player(episode_id, player, row['session_token'])
                   for player, row in credentials.items()}
        report['player_bindings'] = [row['binding'] for row in created['players']]
        for unauthorized in (lambda: service.get(episode_id),
                             lambda: service.player(episode_id, 'player_0', 'invalid-token'),
                             lambda: service.player(episode_id, 'player_1', credentials['player_0']['session_token'])):
            try:
                unauthorized()
            except ServiceError as exc:
                check(exc.status == 403, 'Unauthorized access had the wrong error category')
            else:
                raise AssertionError('Unauthorized board access was accepted')
        report['authentication_checks_passed'] = True

        def invoke(player, name, arguments, *, ok=True):
            reply = players[player].invoke(player, {'request_id': 'smoke-'+str(len(calls)),
                'function': name, 'arguments': arguments}).model_dump(mode='json')
            calls.append({'player_id': player, 'function': name, 'reply': reply})
            check(reply['ok'] is ok, 'Unexpected function status: '+name)
            return reply

        observations = {player: invoke(player, 'observe', {})['result'] for player in players}
        check(observations['player_0'] == observations['player_1'], 'Players did not receive the same snapshot')
        report['initial_observation'] = observations['player_0']
        for player, observation in observations.items():
            side = credentials[player]['binding']['hand']
            guide = invoke(player, 'get_robot', {})['result']
            preset = invoke(player, 'hand_preset', {'side': side, 'preset': 'pinch_prepose', 'amount': .5})['result']
            check(set(preset['hand_posture_rad']) <= set(guide['hand_joints']), 'Preset uses unadvertised hand joints')
            wrist = observation['proprioception']['hand_geometries'][side]['wrist']
            ik = invoke(player, 'query_ik', {'observation_sequence': observation['sequence'],
                'hand_geometries': {side: {'wrist': wrist, 'fingertip_frame_id': side+'_hand_flange'}}})['result']
            check(ik['solution_found'], 'Current published wrist failed the read-only IK query')
            check(not ik['advances_physics'] and not ik['mutates_controller'], 'IK changed runtime state')
        cameras = observations['player_0']['cameras']
        check({row['camera_id'] for row in cameras} == REQUIRED_CAMERAS, 'Incomplete card-room camera inventory')
        for camera in cameras:
            invoke('player_0', 'read_camera', {'observation_sequence': observations['player_0']['sequence'],
                                              'camera_id': camera['camera_id']})
        check(session.clock.sim_time_s == 0., 'Read-only functions advanced physics')
        denied = invoke('player_1', 'hold', {'action_id': 'inactive-hold',
            'observation_sequence': observations['player_1']['sequence'], 'duration_s': .02}, ok=False)
        check(denied['error']['code'] == 'permission_denied', 'Inactive player was not denied explicitly')
        check(session.clock.action_attempts == 0, 'Unauthorized player consumed the active action budget')
        board = players['player_0']
        pending = board.intent('player_0', {'type': 'placement_intent', 'piece_id': 'black_00', 'q': 1, 'r': 1})
        check(pending['board_game']['pending_move'] is not None, 'Intent was not published')
        cleared = board.intent('player_0', {'type': 'cancel_placement'})
        check(cleared['board_game']['pending_move'] is None, 'Intent was not cancelled')
        check(session.clock.sim_time_s == 0. and board.state.commit_count == 0, 'Intent advanced physics or committed a move')
        report['intent_cancel_without_physics'] = True
        action_id = 'owned-hold'
        hold = {'action_id': action_id, 'observation_sequence': observations['player_0']['sequence'],
                'duration_s': args.duration}
        invoke('player_0', 'hold', hold)
        deadline = time.monotonic()+180.
        while True:
            status = invoke('player_0', 'get_action', {'action_id': action_id})['result']
            if status['status'] not in ('accepted', 'running'):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded smoke action did not reach an acknowledged terminal status')
            time.sleep(.1)
        report['action_status'] = status
        check(math.isclose(session.clock.sim_time_s, args.duration, rel_tol=0., abs_tol=1e-8),
              'The hold did not execute exactly its bounded simulation duration')
        if args.termination == 'cancelled':
            check(status['status'] == 'completed', 'The owned hold did not complete')
            before = session.clock.snapshot()
            invoke('player_0', 'hold', hold)
            check(session.clock.snapshot()['action_attempts'] == before['action_attempts'] == 1,
                  'Idempotent hold replay consumed another action attempt')
            check(session.clock.sim_time_s == before['elapsed_sim_s'], 'Idempotent hold replay advanced physics')
            invoke('player_0', 'finish_episode', {})
        final = board.result('player_0')
        report['result'] = final
        expected = ('aborted', 'cancelled', 'cancelled') if args.termination == 'cancelled' else ('truncated', 'time_limit', 'sim_timeout')
        check((final['board_game']['status']['outcome'], final['board_game']['status']['end_reason'],
               final['episode']['outcome']) == expected, 'Incorrect interrupted-game result mapping')
        check(final['board_game']['winner'] is None and final['board_game']['commit_count'] == 0,
              'An interrupted diagnostic was incorrectly scored as a player win')
        report['native_coverage'] = native_coverage(session.directory/'private', final['episode']['elapsed_sim_s'])
        check(report['native_coverage']['passed'], 'Private native sampling evidence is incomplete')
        check(final['episode']['physics_valid'], 'Canonical native physics audit did not pass')
        report['final_observation'] = board.public('player_0')['observation']
        check(len(report['final_observation']['cameras']) == 5, 'Terminal RGB-D snapshot is incomplete')
        check(final['episode']['model_calls'] == 0, 'Unexpected model calls')
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
    finally:
        if service is not None:
            try:
                service.shutdown()
            except BaseException as exc:
                report['passed'] = False
                report['shutdown_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        if session is not None:
            report['native_worker_exit_code'] = session.worker.process.returncode
            if session.worker.process.returncode != 0:
                report['passed'] = False
        elif (output/'episodes').is_dir():
            report['failed_episode_directories'] = [str(path) for path in sorted((output/'episodes').iterdir())]
        dump(output/'public_function_calls.json', calls)
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        dump(output/'qualification.json', report)
    print(json.dumps({'output': str(output), 'passed': report['passed'], 'model_calls': 0,
                      'scope': report['scope']}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--game', choices=('tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7'), default='tic_tac_toe_3x3')
    parser.add_argument('--track', choices=('system0', 'direct_joints'), default='system0')
    parser.add_argument('--termination', choices=('cancelled', 'time_limit'), default='cancelled')
    parser.add_argument('--duration', type=float, default=.12)
    args = parser.parse_args()
    if not .1 <= args.duration <= .4 or not math.isclose(args.duration/.02, round(args.duration/.02), abs_tol=1e-9):
        parser.error('Duration must be 0.10–0.40 seconds and an exact multiple of 20 ms')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
