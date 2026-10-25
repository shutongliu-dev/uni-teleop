"""Isaac-only main-thread process. Launch through scripts/isaac_python.sh.

This is a private operator IPC endpoint, not the Agent API. Raw simulator state
and hidden evaluator output never leave the trusted service/worker boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import traceback

from .transport import receive, send
from bench.paths import load_json
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from bench.evaluation.rigid_safety import RigidContactSafetyGuard


def public_task_projection(task):
    """Bind the worker's loaded rules to the same public identity as the host."""
    if task.get('schema_version') == 'living-room-manipulation-task-v1':
        from .manipulation_tasks import project_manipulation_task
        return project_manipulation_task(task)
    if task.get('schema_version') == 'card-room-placement-task-v1':
        from .placement_tasks import project_placement_task
        return project_placement_task(task)
    from bench.protocol.task_referee import project_task_rules
    from scenes.runtime import terminal_scoring_contract
    return project_task_rules(task, scoring_contract=terminal_scoring_contract(task))


def guarded_native_step(env, command, guard):
    """Finish the admitted 20 ms window, then latch before any subsequent step."""
    guard.assert_can_step()
    observation = env.step(command)
    guard.observe(observation)
    return observation


def camera_sensor_configuration(env):
    """Pin every resolved scene view; old scenes keep the v1 two-view contract."""
    common = {'rgb': list(env.camera_ids), 'depth': True,
        'depth_encoding': 'float32_npy', 'depth_unit': 'm', 'depth_semantics': 'camera_z',
        'depth_invalid_value': 0.0, 'depth_rgb_aligned': True,
        'camera_frame': 'optical_x_right_y_down_z_forward',
        'capture': 'decision_and_policy_steps'}
    if env.scene_spec.get('cameras') is None:
        return {'protocol': 'agent_rgbd_v1', 'width': env.render_width,
                'height': env.render_height, **common}
    cameras = {}
    for name in env.camera_ids:
        spec = env.cameras[name]
        width, height = spec['resolution']
        cameras[name] = {'width': width, 'height': height, 'kind': spec['kind'],
                         'role': spec['role'], 'mount_link': spec.get('mount_link')}
    return {'protocol': 'agent_rgbd_v2', 'cameras': cameras, **common}


def capture_camera_frames(env, output, sequence, observation, sensor_config):
    """Capture all published views without advancing physics or consulting USD poses."""
    import numpy as np
    from PIL import Image
    from bench.simulation.isaacsim_backend.camera_calibration import public_camera_calibration
    if type(sequence) is not int or sequence < 0:
        raise ValueError('Invalid camera sequence')
    output = Path(output)
    before = env.capture_state()
    frames = []
    for camera in sensor_config['rgb']:
        dimensions = sensor_config.get('cameras', {}).get(camera, sensor_config)
        width, height = dimensions['width'], dimensions['height']
        # env resolves the actual world/link prim path and attaches camera_params
        # to this exact RGB-D render product; no preliminary render or USD edits.
        capture = dict(env.render_rgbd(camera=camera, width=width, height=height))
        rgb = capture.pop('rgb')
        depth = capture.pop('depth_m')
        if (capture['sim_time_s'] != observation['sim_time_s'] or
                capture['depth_sim_time_s'] != observation['sim_time_s']):
            raise RuntimeError('RGB-D capture timestamp does not match the observation')
        if rgb.shape != (height, width, 3) or depth.shape != (height, width):
            raise RuntimeError('RGB-D capture dimensions disagree with the sensor configuration')
        calibration = public_camera_calibration(capture['camera_calibration'], width=width, height=height)
        if float(np.std(rgb)) < 1:
            raise RuntimeError('Native camera produced a blank frame')
        image = output/'rgb'/f'{sequence:08d}_{camera}.png'
        image.parent.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(image)
        depth_path = output/'depth'/f'{sequence:08d}_{camera}.npy'
        depth_path.parent.mkdir(exist_ok=True)
        np.save(depth_path, depth, allow_pickle=False)
        frames.append({**capture, **calibration,
            'camera_id': camera, 'sim_time_s': observation['sim_time_s'],
            'path': str(image), 'rgb_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
            'depth_path': str(depth_path), 'depth_sha256': hashlib.sha256(depth_path.read_bytes()).hexdigest(),
            'width': width, 'height': height})
    after = env.capture_state()
    if before.keys() != after.keys() or any(not np.array_equal(before[k], after[k]) for k in before):
        raise RuntimeError('Rendering changed physical state')
    return frames


def seal_native_recording(output, states, triangles, score, *, reason):
    """Seal only states already captured from physics, even after a lost client.

    This does not acknowledge commands the server never received or certify
    physics validity. The service still records a transport failure as invalid.
    """
    import numpy as np
    from bench.isaacsim.release import dump
    output=Path(output)
    if not states:
        return
    np.savez_compressed(output/'states.npz', **{k:np.stack([s[k] for s in states]) for k in states[0]})
    np.save(output/'triangles.npy', triangles)
    dump(output/'final_score.json', score)
    dump(output/'worker_archive.json', {'state_count':len(states),'reason':reason,
        'source':'captured_native_states','state_reconstruction':False})


def main(*, observer_factory=None):
    """Run the canonical worker, optionally with trusted read-only instrumentation.

    The factory is supplied by an operator's Python entrypoint, never by JSON
    configuration or an Agent request. It must return an object with close().
    No observer data is added to the public observation or evaluator inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--socket', required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    env = None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(args.socket)
    sock.listen(1)
    connection, _ = sock.accept()
    states = []
    score = None
    sealed = False
    cloth_observer = None
    engineering_observer = None
    task_runtime = None
    runtime_key = None
    demo_recorder = None
    safety_guard = RigidContactSafetyGuard()
    try:
        import numpy as np
        from bench.isaacsim.release import task_inputs, load_evaluator, dump
        from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
        from bench.simulation.isaacsim_backend.robot_registry import scene_with_hand
        from bench.simulation.isaacsim_backend.scene_variants import apply_scene_variant
        from .board_game_worker import BoardGameWorker
        from scenes.runtime import build_task_runtime
        from .demo_recording import NativeDemoRecorder
        source_verification = verify_if_snapshot(Path(__file__).absolute().parents[2],
            required_modules=("bench.runtime.worker", "bench.isaacsim.release",
                              "bench.simulation.isaacsim_backend.env", "bench.runtime.board_game_worker",
                              "bench.runtime.demo_recording", "scenes.runtime"),
            entrypoint_file=Path(__file__))
        if source_verification is not None:
            dump(output/'runtime_imports.json', source_verification)
        entry, scene, task, initial = task_inputs(Path(config['release']), config['task_id'], seed=config['seed'])
        scene = scene_with_hand(scene, config['hand_id'])
        scene = apply_scene_variant(scene, config.get('scene_variant','original_v1'))
        env = FR3DexterousIsaacEnv(scene, device=config['device'], render_width=640, render_height=480)
        obs = env.reset(seed=config['seed'], initial_state=initial)
        if scene.get('board_game_task') is not None:
            if observer_factory is not None or scene.get('cloth', {}).get('enabled') is True:
                raise ValueError('The trusted board-game observer exclusively owns native adjudication')
            task_runtime = BoardGameWorker(env, scene, task, output=output/'board_game')
            runtime_key = 'board_game'
        else:
            task_runtime = build_task_runtime(env, scene, task, output/'task_runtime',
                                              evaluator_factory=load_evaluator(Path(config['release'])))
            if task_runtime is not None and task_runtime.public_state() is not None:
                runtime_key = 'turn_game'
        if task_runtime is not None:
            if observer_factory is not None or scene.get('cloth', {}).get('enabled') is True:
                raise ValueError('A task runtime exclusively owns its native observation exchange')
            evaluator = None
            score = task_runtime.score()
        else:
            evaluator = load_evaluator(Path(config['release']))(task)
            score = evaluator.update(obs)
        states.append(env.capture_state())
        metadata = env.metadata()
        metadata['agent_sensor_config'] = camera_sensor_configuration(env)
        demo_recorder = NativeDemoRecorder.from_environment(
            env, output, control_period_s=metadata['control_dt_s'])
        if demo_recorder is not None:
            metadata['demo_recording'] = demo_recorder.configuration()
        dump(output/'metadata.json', metadata)
        dump(output/'task.json', task)
        dump(output/'scene.json', scene)
        if scene.get('cloth', {}).get('enabled') is True:
            from scenes.cloth.worker_observer import build_cloth_observer
            cloth_observer = build_cloth_observer(env, output/'cloth_physics')
        if observer_factory is not None:
            if cloth_observer is not None:
                raise ValueError('Only one native substep observer may own the contact exchange')
            engineering_observer = observer_factory(env, output)
        projected_task = public_task_projection(task)
        public_task = {'task': projected_task} if projected_task is not None else {}
        if projected_task is not None:
            dump(output/'public_task.json', projected_task)
        if demo_recorder is not None:
            demo_recorder.record(obs, score)
        send(connection, {'ok': True, 'metadata': metadata, 'observation': obs, 'score': score, **public_task,
                          **({runtime_key: task_runtime.public_state()} if runtime_key else {})})
        with (output/'physics.jsonl').open('x') as log:
            log.write(json.dumps({'kind': 'reset', 'observation': obs, 'score': score}, allow_nan=False)+'\n')
            while True:
                request = receive(connection)
                operation = request.get('operation')
                step_attempted = False
                step_before = None
                try:
                    if operation == 'step':
                        command = request['command']
                        if set(command) != {'joint_positions_rad', 'duration_s'}:
                            raise ValueError('Only complete robot joint targets are accepted')
                        if task_runtime is not None and hasattr(task_runtime, 'begin_action'):
                            task_runtime.begin_action(request.get('action_id'))
                        admitted_turn = (task_runtime.authorize_step(command, request.get('actor'))
                                         if task_runtime else None)
                        safety_guard.assert_can_step()
                        step_attempted = True
                        step_before = env.capture_state()
                        obs = guarded_native_step(env, command, safety_guard)
                        score = task_runtime.score() if task_runtime else evaluator.update(obs)
                        board_reply = (task_runtime.acknowledge_step(command, admitted_turn)
                                       if task_runtime else {})
                        states.append(env.capture_state())
                        log.write(json.dumps({'kind': 'step', 'command': command, 'observation': obs,
                                              'score': score, 'action_id': request.get('action_id'),
                                              **({'actor': request.get('actor'), **board_reply}
                                                 if task_runtime else {})}, allow_nan=False)+'\n')
                        log.flush()
                        if safety_guard.trigger is not None:
                            dump(output/'native_safety_stop.json', safety_guard.trigger)
                        if demo_recorder is not None:
                            demo_recorder.record(obs, score)
                        send(connection, {'ok': True, 'observation': obs, 'score': score, **board_reply})
                    elif operation in ('board_game', 'turn_game'):
                        if runtime_key != operation:
                            raise ValueError('This episode has no matching player referee')
                        reply = task_runtime.request(request.get('command'), actor=request.get('actor'))
                        score = reply['score']
                        send(connection, reply)
                    elif operation == 'hold':
                        if runtime_key is not None:
                            # Board games cancel the future controller queue at
                            # the service boundary. Their admitted drive targets
                            # must remain unchanged for the other player's turn.
                            raise ValueError('Board-game cancellation retains acknowledged robot targets')
                        # No physical time advances in the paused-Agent track.
                        # Cancel the drive target as well as future queued steps.
                        spec = env.action_spec()
                        q = env.observation()['joint_positions_rad']
                        action = {n: float(np.clip(q[n], *spec['joint_limits_rad'][n])) for n in spec['joint_names']}
                        if hasattr(env, 'hold_current_targets'):
                            env.hold_current_targets()
                        else:
                            # Native state order may contain passive/mimic DOFs.
                            measured = env.capture_state()['joint_q']
                            env._targets = measured[None].astype(np.float32)
                            env._robot_view.set_dof_position_targets(env._tensor(env._targets), env._one)
                        log.write(json.dumps({'kind':'cancel_hold_targets','sim_time_s':obs['sim_time_s'],
                                              'joint_positions_rad':action},allow_nan=False)+'\n')
                        log.flush()
                        send(connection, {'ok': True})
                    elif operation == 'cameras':
                        frames = capture_camera_frames(env, output, request['sequence'], obs,
                                                       metadata['agent_sensor_config'])
                        send(connection, {'ok': True, 'frames':frames})
                    elif operation == 'finish':
                        if task_runtime is not None:
                            task_runtime.close(reason='cancelled')
                            score = task_runtime.score()
                        if engineering_observer is not None:
                            engineering_observer.close()
                        if cloth_observer is not None:
                            from scenes.cloth.worker_observer import seal_observer
                            seal_observer(cloth_observer)
                        if demo_recorder is not None:
                            demo_recorder.close(score=score, reason='finish_request', completed=True)
                        seal_native_recording(output, states, env.triangles, score, reason='finish_request')
                        sealed = True
                        send(connection, {'ok':True,'score':score,'state_count':len(states),
                                          **({runtime_key: task_runtime.public_state()} if runtime_key else {})})
                        break
                    else:
                        raise ValueError('Unknown private worker operation')
                except Exception as exc:
                    traceback.print_exc()
                    unsafe_step = False
                    if step_attempted:
                        from bench.evaluation.cloth.cloth_robot_collision import state_digest
                        failure = {'error_type':type(exc).__name__, 'command_acknowledged':False}
                        try:
                            after = env.capture_state()
                            unsafe_step = step_before is None or state_digest(after) != state_digest(step_before)
                            if unsafe_step:
                                np.savez_compressed(output/'unacknowledged_step_state.npz', **after)
                            failure.update(physical_state_unchanged=not unsafe_step,
                                           actual_state_saved=unsafe_step)
                        except Exception as capture_error:
                            unsafe_step = True
                            failure.update(physical_state_unchanged=None, actual_state_saved=False,
                                           state_capture_error_type=type(capture_error).__name__)
                        if unsafe_step:
                            dump(output/'unacknowledged_step.json',failure)
                    send(connection, {'ok':False,'error_type':type(exc).__name__,'error':str(exc)})
                    if (unsafe_step or not isinstance(exc, ValueError) or
                            (cloth_observer is not None and cloth_observer.observer_failure is not None)):
                        break
    except BaseException:
        traceback.print_exc()
        try:
            send(connection, {'ok':False,'error_type':'WorkerFailure','error':'Native worker failed; inspect private worker.log'})
        except Exception:
            pass
        raise
    finally:
        if demo_recorder is not None:
            try:
                demo_recorder.close(score=score, reason='worker_exit_without_finish')
            except BaseException:
                traceback.print_exc()
        if task_runtime is not None:
            if not task_runtime.closed:
                try:
                    task_runtime.close(reason='runtime_error')
                except BaseException:
                    traceback.print_exc()
            score = task_runtime.score()
        if engineering_observer is not None:
            try:
                engineering_observer.close()
            except BaseException:
                traceback.print_exc()
        if cloth_observer is not None and not cloth_observer.closed:
            try:
                from scenes.cloth.worker_observer import seal_observer
                seal_observer(cloth_observer)
            except BaseException:
                traceback.print_exc()
        if not sealed and states and env is not None:
            try:
                seal_native_recording(output, states, env.triangles, score, reason='worker_exit_without_finish')
            except Exception:
                traceback.print_exc()
        connection.close()
        sock.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
