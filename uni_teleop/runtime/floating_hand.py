"""Standalone right SHARPA: shared PICO/MANUS loop for MuJoCo and Isaac."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from uni_teleop.backends.sharpa import extract_hand
from uni_teleop.mapping.floating_hand import HandController, optimizer_targets


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--backend', choices=('isaac', 'mujoco'), required=True)
    cli.add_argument('--scene', choices=('sandbox', 'parts-sorting'), default='sandbox')
    cli.add_argument('--data', type=Path, default=os.environ.get('DEXBENCH_DATA'))
    cli.add_argument('--output', type=Path)
    cli.add_argument('--headless', action='store_true')
    cli.add_argument('--video', action=argparse.BooleanOptionalAction, default=False,
                     help='Isaac: send stereo overview with wrist inset to PICO')
    cli.add_argument('--video-listen', default='0.0.0.0:13579')
    cli.add_argument('--demo', action='store_true', help='Synthetic inputs; no hardware or SDK opened')
    cli.add_argument('--steps', type=int, default=0)
    cli.add_argument('--scale', type=float, default=1.)
    cli.add_argument('--motion-frame', choices=('screen', 'world'), default='screen',
                     help='Screen aligns headset heading to desktop view on Start/Recenter')
    cli.add_argument('--start-active', action='store_true')
    cli.add_argument('--port', type=int, default=5557)
    cli.add_argument('--external-pico', action='store_true', help='Consume an existing local PICO bridge')
    cli.add_argument('--screenshot', action='store_true', help='Save a 1280x960 final preview')
    for name in ('sdk-python', 'python', 'calibration', 'retarget-root', 'retarget-python'):
        cli.add_argument('--manus-'+name, type=Path)
    cli.add_argument('--manus-sdk-version', choices=('3.2.0', '3.2.1'), default='3.2.0')
    cli.add_argument('--manus-glove-id', type=int)
    return cli


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


class LiveInputs:
    """Compose existing acquisitions/optimizer without copying their SDK code."""
    def __init__(self, args, hand, stack):
        from uni_teleop.devices.pico.protocol import Receiver
        from uni_teleop.devices.manus.process_source import ManusProcessSource, ManusSourceConfig
        from uni_teleop.mapping.manus.sharpa import SharpaOptimizerSource, kinematic_signature
        required = ('sdk_python', 'python', 'calibration', 'retarget_root', 'retarget_python', 'glove_id')
        if any(getattr(args, 'manus_'+key) is None for key in required):
            raise ValueError('Live mode requires the MANUS settings; use a configured floating-hand recipe')
        calibration = args.manus_calibration.read_bytes()
        text = calibration.decode()
        profile = json.loads(text[text.index('{'):])['gloveProfiles'][f'0x{args.manus_glove_id:08X}']
        if profile['side'] != 'right' or not profile['calibrationProfile']:
            raise ValueError('A saved right MANUS calibration is required')
        self.calibration_id = 'official/'+hashlib.sha256(calibration).hexdigest()
        self.glove_id, self.names = args.manus_glove_id, hand.names
        self.pico = Receiver(args.port)
        stack.callback(self.pico.close)
        if not args.external_pico:
            log = stack.enter_context((args.output/'pico_sdk.log').open('w'))
            process = subprocess.Popen([str(args.manus_python), '-B', '-m', 'uni_teleop.devices.pico.source',
                '--library', os.environ['PICO_SDK_LIBRARY'], '--port', str(args.port)], stdout=log, stderr=log)
            stack.callback(stop_process, process)
        self.glove = ManusProcessSource(ManusSourceConfig(args.manus_sdk_python, args.manus_python,
            args.manus_glove_id, self.calibration_id, sdk_version=args.manus_sdk_version,
            stderr_log=args.output/'manus_sdk.log'))
        stack.callback(self.glove.close)
        self.glove.start()
        self.optimizer = SharpaOptimizerSource(self.glove, args.manus_retarget_python,
            args.manus_retarget_root, args.output/'optimizer.log')
        stack.callback(self.optimizer.close)
        self.optimizer.start()
        deadline = time.monotonic()+20
        while self.optimizer.metadata is None and not self.optimizer.error and time.monotonic() < deadline:
            time.sleep(.02)
        meta = self.optimizer.metadata
        if meta is None:
            raise RuntimeError(self.optimizer.error or 'SHARPA optimizer startup timed out')
        if set(meta['joint_names']) != set(hand.names) or meta['kinematic_signature'] != kinematic_signature(hand.urdf):
            raise ValueError('SHARPA optimizer kinematics do not match the simulated hand')

    def read(self, now_ns):
        return self.pico.read(), optimizer_targets(self.glove.latest, self.optimizer.latest,
            now_ns=now_ns, names=self.names, calibration_id=self.calibration_id, glove_id=self.glove_id)

    def health(self):
        return dict(pico=self.pico.health(), manus=self.glove.health(), optimizer=self.optimizer.health())


def demo_input(hand, step, now_ns):
    """Exercise the real mapper with synthetic controller motion and finger targets."""
    from scipy.spatial.transform import Rotation
    from uni_teleop.devices.pico.protocol import Frame
    t = step*.02
    pose = [.05*np.sin(t), .04*np.sin(t*.8), -.06*np.sin(t*.6),
            *Rotation.from_rotvec([0., .25*np.sin(t), 0.]).as_quat()]
    frame = Frame('synthetic', step, now_ns, str(step), 'demo',
                  dict(focused=True, controllers={'right': dict(pose=pose, grip=1.)}))
    fingers = {n: float(np.clip(.35*(1-np.cos(t)) if n.endswith(('FE', 'DIP', 'PIP')) else 0., lo, hi))
               for n, lo, hi in zip(hand.names, hand.lower, hand.upper)}
    return frame, fingers


def main():
    args = parser().parse_args()
    if args.data is None: raise ValueError('Set DEXBENCH_DATA or --data')
    if args.steps < 0: raise ValueError('--steps must be non-negative')
    if args.scene != 'sandbox' and args.backend != 'isaac':
        raise ValueError('Bench task scenes currently require Isaac')
    if args.video and args.backend != 'isaac':
        raise ValueError('Floating-hand PICO video currently requires Isaac')
    args.output = (args.output or Path(os.environ.get('UNI_TELEOP_ROOT', '.'))/'runs/floating_hand'/
                   f'{args.backend}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}').resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    hand = extract_hand(args.data, args.output)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    report = dict(backend=args.backend, robot='standalone_right_sharpa',
                  source='synthetic' if args.demo else 'pico_manus_hardware', steps=0,
                  task_success=None, status='starting')
    def seal_report():
        import sys
        error = sys.exc_info()[1]
        if error is not None: report.update(status='error', error=str(error))
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    seal_report()
    try:
        with ExitStack() as stack:
            if args.backend == 'mujoco':
                from uni_teleop.backends.mujoco.floating_hand import FloatingHand
            else:
                from uni_teleop.backends.isaac.floating_hand import FloatingHand
            scene_options = {}
            if args.scene == 'parts-sorting':
                from uni_teleop.backends.isaac.hand_scene import parts_sorting
                scene, receipt = parts_sorting(args.data)
                (args.output/'scene.json').write_text(json.dumps(scene, indent=2)+'\n')
                (args.output/'scene_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
                scene_options['bench_scene'] = scene
            report['scene'] = args.scene
            env = FloatingHand(hand, args.output, headless=args.headless, **scene_options)
            # Kit installs its own handlers at startup. Restore our graceful
            # shutdown so SDK/video subprocesses close before Kit exits.
            for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
            stack.callback(env.close)
            stack.callback(seal_report)
            video = None
            if args.video:
                from uni_teleop.video.floating_hand_pico import HandPicoVideo
                video = HandPicoVideo(env, args.output, listen=args.video_listen)
                stack.callback(video.close)
            report['video'] = 'overview_stereo_with_wrist_inset' if video else None
            # Sources are stopped before the native engine is shut down.
            source = None if args.demo else LiveInputs(args, hand, stack)
            controller = HandController(hand, *env.pose(), scale=args.scale,
                                        screen_aligned=args.motion_frame == 'screen' and not args.demo)
            report['motion_frame'] = args.motion_frame if not args.demo else 'world'
            if args.demo or args.start_active: controller.toggle()
            report['status'] = 'running'
            print('Standalone SHARPA ready. Start/Pause: Space or GUI (Isaac). '
                  'Recenter: GUI or R. Hold right grip to move palm; MANUS controls fingers.', flush=True)
            last_status = None
            with (args.output/'frames.jsonl').open('w') as log:
                while not stopped and env.running() and (not args.steps or report['steps'] < args.steps):
                    tick = time.monotonic()
                    while env.events:
                        event = env.events.pop(0)
                        getattr(controller, event)()
                    now = time.monotonic_ns()
                    frame, fingers = demo_input(hand, report['steps'], now) if args.demo else source.read(now)
                    view_forward = env.view_forward() if controller.heading_pending else None
                    physics_running = env.physics_running() if hasattr(env, 'physics_running') else True
                    if physics_running:
                        controller.update(frame, fingers, now, env.pose(), view_forward=view_forward)
                    else:
                        controller.recenter()
                    env.step(controller.position, controller.quaternion, controller.fingers)
                    status = (controller.wrist_status+'; '+controller.finger_status if physics_running
                              else 'Isaac physics paused; resume timeline play; no forces applied')
                    if video: video.update()
                    env.render(status)
                    if video: video.publish()
                    if status != last_status:
                        print(status, flush=True)
                        last_status = status
                    p, q = env.pose()
                    row = dict(step=report['steps'], wall_ns=now, active=controller.active,
                        palm_xyz=p.tolist(), palm_xyzw=q.tolist(), joints=env.positions().tolist(),
                        target_xyz=controller.position.tolist(), target_xyzw=controller.quaternion.tolist(),
                        target_joints=controller.fingers.tolist(), status=status)
                    row['physics_running'] = physics_running
                    row['heading_alignment_xyzw'] = controller.mapper.alignment.as_quat().tolist()
                    if source: row['input_health'] = source.health()
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    report['steps'] += 1
                    time.sleep(max(0., .02-(time.monotonic()-tick)))
                if args.screenshot: env.screenshot(args.output/'preview.png')
                report['status'] = 'stopped' if stopped else 'finished'
                # Kit may terminate Python in close(); seal the report first.
                (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                print(f'Session: {args.output}', flush=True)
    except BaseException as error:
        report.update(status='error', error=str(error))
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        raise


if __name__ == '__main__': main()
