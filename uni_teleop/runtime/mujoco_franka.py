"""MuJoCo execution loop for the shared FACTR/PICO/MANUS mappers."""
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time

from uni_teleop.mapping.franka.retarget import Retargeter


def run(args, *, receiver=None, source_kind='xr_sdk', mapper_factory=Retargeter, feedback=None):
    from bench.control.factory import public_robot_geometry, robot_configuration
    from scenes.teleop_sandbox import inputs
    from uni_teleop.backends.mujoco.franka import FrankaWorkbench
    from uni_teleop.devices.pico.protocol import Receiver
    import numpy as np

    # Benchmark release scenes have not been translated to MuJoCo.
    if not args.sandbox:
        raise ValueError('MuJoCo Franka currently requires --sandbox; DexBench tasks remain Isaac-only')
    if getattr(args, 'capture_reference_on_start', False):
        raise ValueError('MuJoCo uses saved calibration; capture it with the calibration tools first')
    if getattr(args, 'align_from_leader', False) or args.head_tracked:
        raise ValueError('This backend supports saved joint mapping and a fixed overview; explicit alignment/head-tracked video is not implemented')
    scene, _, _ = inputs()
    if args.arm_base_yaw is not None:
        yaw = np.deg2rad(args.arm_base_yaw)
        scene['robot']['base_quaternion_wxyz'] = [float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))]
    geometry = public_robot_geometry(robot_configuration(scene))
    initial = dict(zip(geometry.names, np.clip(np.zeros(len(geometry.names)), geometry.lower, geometry.upper)))
    initial.update(zip(geometry.arm.names, args.arm_home or scene['robot']['arm_home_rad']))
    output = args.output or Path('runs') / f'mujoco_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}'
    output.mkdir(parents=True, exist_ok=True)
    active, reset, support, stopped = args.start_active, False, False, False
    def key(code):
        nonlocal active, reset, support
        if code == 32: active = not active
        if code in (ord('R'), ord('r')): reset, active = True, False
        if code in (ord('F'), ord('f')) and feedback: support = not support
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    report = {'backend': 'mujoco', 'scene': 'portable_franka_workbench', 'source': source_kind,
              'task_success': None, 'steps': 0, 'status': 'running'}
    try:
        with ExitStack() as stack:
            receiver = receiver or Receiver(args.port)
            stack.callback(receiver.close)
            env = FrankaWorkbench(os.environ['DEXBENCH_DATA']+'/assets', geometry, initial,
                                  unrestricted_arm=getattr(args, 'unrestricted_arm', False))
            stack.callback(env.close)
            bindings = dict(v.split('=', 1) for v in args.bind) if args.bind else {'right': 'right'}
            mapper = mapper_factory(geometry, initial, bindings, mode=args.mode, scale=args.scale,
                                    tracking_grace_s=.3, hands_recovery=True,
                                    control_point=args.control_point, lock_orientation=args.lock_orientation)
            mapper.set_active(active)
            viewer = writer = None
            if not args.headless:
                import mujoco.viewer
                viewer = stack.enter_context(mujoco.viewer.launch_passive(env.model, env.data, key_callback=key))
                viewer.cam.lookat[:] = [.1, 0., 1.05]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.1, 135, -25
            if not args.no_video:
                from uni_teleop.video.isaac import FrameWriter
                writer = FrameWriter(args.width, args.height, listen=args.video_listen)
                stack.callback(writer.close)
            print('MuJoCo ready. Space: start/pause; R: reset; F: opt-in leader support. Video is duplicated mono.', flush=True)
            last_active, last_support, next_video = active, False, 0.
            with (output/'frames.jsonl').open('w') as log:
                while not stopped and (viewer is None or viewer.is_running()) and (not args.steps or report['steps'] < args.steps):
                    tick = time.monotonic()
                    if reset:
                        env.reset(); mapper.recalibrate(); mapper.acknowledge(initial)
                        reset = False
                    if active != last_active:
                        mapper.set_active(active)
                        last_active = active
                        if not active: support = False
                    if feedback and support != last_support:
                        if support and active: feedback.start_support(mapper)
                        else: feedback.request_feedback(False)
                        last_support = support
                    frame = receiver.read()
                    targets = mapper.command(frame, env.positions(), now_ns=time.monotonic_ns())
                    if frame and time.monotonic_ns()-frame.host_ns > 150_000_000:
                        targets = mapper.hold(receiver.read(), now_ns=time.monotonic_ns())
                    measured = env.step(targets)
                    mapper.acknowledge(targets)
                    if feedback:
                        feedback.observe(mapper, {'joint_positions_rad': measured, 'truncated': False, 'terminated': False}, now_ns=time.monotonic_ns())
                    if writer and tick >= next_video:
                        rgb = env.render(args.width, args.height)
                        writer.publish(rgb, rgb); writer.check()
                        next_video = tick + 1 / args.video_fps
                    if viewer: viewer.sync()
                    log.write(json.dumps({'step': report['steps'], 'status': mapper.status,
                                          'joint_positions_rad': measured, 'targets': targets})+'\n')
                    report['steps'] += 1
                    time.sleep(max(0., env.control_dt-(time.monotonic()-tick)))
            report.update(status='stopped', final_joints=env.positions())
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        if feedback: feedback.request_feedback(False)
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    from uni_teleop.runtime.arguments import parser
    run(parser().parse_args())


if __name__ == '__main__':
    main()
