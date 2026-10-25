"""Operate a DexBench task or two-object diagnostic workbench with PICO in Isaac."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from bench.adapters.pico.protocol import Receiver
from bench.adapters.pico.retarget import Retargeter
from bench.adapters.pico.video import FrameWriter, StereoView
from bench.control.factory import public_robot_geometry, robot_configuration
from bench.adapters.pico.contact_guard import TeleopContactGuard
from bench.adapters.pico.camera import apply_in_hand_wrist_camera, apply_side_overview, IN_HAND_WRIST_CAMERA_V1
from bench.paths import output_root
from scripts.teleop.diagnostics import HoldReporter, control_issues
from scripts.maintenance.freeze_project_sources import verify_if_snapshot


class Controls:
    def __init__(self, mapper, *, inset=False, rear=False, joint_control=False, feedback=None,
                 capture_on_start=False):
        import omni.ui as ui
        import carb.input
        import carb.settings
        import omni.appwindow
        carb.settings.get_settings().set('/exts/omni.kit.renderer.core/present/enabled', True)
        carb.settings.get_settings().set('/app/window/hideUi', False)
        self.viewport = ui.Window('PICO Scene', width=1000, height=750)
        self.image = ui.ByteImageProvider()
        with self.viewport.frame:
            ui.ImageWithProvider(self.image, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
        self.wrist_window = None
        if inset:
            self.wrist_window = ui.Window('Wrist camera', width=340, height=275)
            self.wrist_window.position_x, self.wrist_window.position_y = 1010, 320
            self.wrist_image = ui.ByteImageProvider()
            with self.wrist_window.frame:
                ui.ImageWithProvider(self.wrist_image, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
        self.rear_window = None
        if rear:
            self.rear_window = ui.Window('Rear camera - workbench', width=500, height=400)
            self.rear_window.position_x, self.rear_window.position_y = 40, 420
            self.rear_image = ui.ByteImageProvider()
            with self.rear_window.frame:
                ui.ImageWithProvider(self.rear_image, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
        self.mapper, self.reset_requested = mapper, False
        self.feedback=feedback
        self.capture_on_start=capture_on_start
        self.reference_request=None
        self.align_requested=False
        self.recenter_requested = False
        self.joint_control = joint_control
        self.window = ui.Window('Master-slave Teleoperation' if joint_control else 'PICO Teleoperation', width=500 if joint_control else 440, height=430 if joint_control else 350)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label('主臂控制七轴；右手手套控制手指。空格：开始/暂停' if joint_control else
                         'Space: start/pause    C: recenter    R: reset    L: lock')
                ui.Label('暂停也会关闭实体托举，请扶住主臂。' if joint_control else
                         getattr(mapper, 'input_hint', 'Hold grip to move; trigger: open -> pinch -> close'), word_wrap=True)
                self.label = ui.Label('Waiting for leader input' if joint_control else 'Waiting for PICO tracking', word_wrap=True)
                with ui.HStack():
                    if joint_control:
                        ui.Button('开始 / 恢复跟随', clicked_fn=self.start)
                        ui.Button('暂停（托举关闭）', clicked_fn=self.pause)
                    else:
                        ui.Button('Start / Pause', clicked_fn=self.toggle)
                        ui.Button('Recenter', clicked_fn=self.recenter)
                    ui.Button('重置场景' if joint_control else 'Reset scene', clicked_fn=self.reset,
                              tooltip='物品和仿真机械臂回到初始状态，暂停遥操并关闭托举' if joint_control else '')
                if (joint_control and getattr(getattr(mapper,'leader',mapper),'fixed_reference',False)
                        and not getattr(getattr(mapper,'leader',mapper),'follow_on_start',False)):
                    ui.Button('Read leader -> align simulation (resets scene)',clicked_fn=self.align_from_leader)
                if joint_control and feedback is None:
                    ui.Button('记录参考姿态（先重置，按图摆好主臂）', clicked_fn=self.capture_reference)
                    self.reference_label=ui.Label('首次点开始：记录当前主臂参考并跟随；请先按图摆好。'
                                                  if capture_on_start else '',word_wrap=True)
                if feedback:
                    if joint_control and feedback.feedback_state['kind']=='leader_gravity_and_damping':
                        with ui.HStack():
                            ui.Button('开启 / 恢复托举',clicked_fn=lambda:feedback.start_support(mapper))
                            ui.Button('关闭托举',clicked_fn=lambda:feedback.request_feedback(False))
                    else:
                        ui.Button('Low-force feedback: Enable / Disable',clicked_fn=self.toggle_feedback)
                if not joint_control:
                    ui.Button('Recalibrate fingers (then open hand + Start)', clicked_fn=mapper.recalibrate)
                    self.mapping_label = ui.Label('', word_wrap=True)
                    with ui.HStack():
                        ui.Button('Pinch center / Wrist', clicked_fn=self.control_point)
                        ui.Button('Lock / Free orientation (L)', clicked_fn=self.orientation)
                    self.mapping_status()
                    ui.Button('Hands', clicked_fn=lambda: mapper.set_mode('hands'))
                    ui.Button('Controllers (hold grip)', clicked_fn=lambda: mapper.set_mode('controllers'))
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.key)

    def show_frame(self, main, inset, rear=None):
        from PIL import Image
        import numpy as np
        views = [(self.image, main)]
        if self.wrist_window:
            views.append((self.wrist_image, inset))
        if self.rear_window:
            views.append((self.rear_image, rear))
        for provider, frame in views:
            rgba = np.asarray(Image.fromarray(frame).convert('RGBA'))
            provider.set_data_array(rgba, [frame.shape[1], frame.shape[0]])

    def toggle(self):
        active=not getattr(self.mapper, 'session_active',self.mapper.armed or self.mapper.recovery_pending)
        if active:self.start()
        else:self.pause()

    def start(self):
        if self.capture_on_start:
            self.capture_reference(resume=True)
            return
        self.reference_request=None
        self.mapper.set_active(True)
        if self.feedback:self.feedback.start_support(self.mapper)

    def pause(self):
        self.reference_request=None
        if self.feedback:self.feedback.request_feedback(False)
        self.mapper.set_active(False)

    def capture_reference(self, *, resume=False):
        self.reference_request=(time.monotonic_ns(), resume)
        self.reference_label.text='等待新鲜主臂数据；请保持参考姿态。'

    def complete_reference_capture(self):
        if self.reference_request is None:
            return
        requested, resume=self.reference_request
        arm=getattr(self.mapper,'leader',self.mapper)
        reference=arm.latest_reference
        if reference is None or reference[0] <= requested or time.monotonic_ns()-reference[0]>150_000_000:
            return
        saved=self.mapper.capture_reference(require_saved_pose=True)
        self.reference_request=None
        if saved:
            self.capture_on_start=False
            self.reference_label.text='参考已保存；点开始，小幅核对方向。'
        else:
            reasons={
                'Pause and reset the simulation before capturing the reference':'请先暂停并重置场景。',
                'Reset the simulation to the saved reference pose first':'仿真不在参考姿态，请先重置场景。',
                'Fresh leader/robot state required to save reference':'需要新鲜主臂数据及有效保存目录。',
            }
            self.reference_label.text='未保存：'+reasons.get(self.mapper.status,self.mapper.status)
        print(json.dumps({'event':'arm_reference_capture','saved':saved,
                          'status':self.mapper.status,
                          'path':self.mapper.input_mapping.get('saved_reference_file')},ensure_ascii=False),flush=True)
        if saved and resume:
            self.start()

    def align_from_leader(self):
        if self.feedback:self.feedback.request_feedback(False)
        self.mapper.set_active(False)
        self.align_requested=True

    def toggle_feedback(self):
        arm=getattr(self.mapper,'leader',self.mapper)
        if self.feedback.requested:
            self.feedback.request_feedback(False)
        elif self.feedback.feedback_state['kind']=='leader_gravity_and_damping':
            self.feedback.start_support(self.mapper)
        elif arm.armed:self.feedback.request_feedback(True)

    def mapping_status(self):
        mode = 'LOCKED' if self.mapper.lock_orientation else 'FREE'
        self.mapping_label.text = f'Controllers: {self.mapper.control_point} / {mode} / scale {self.mapper.scale:g}'

    def control_point(self):
        point = 'wrist' if self.mapper.control_point == 'pinch' else 'pinch'
        self.mapper.set_controller_mapping(control_point=point)
        self.mapping_status()

    def orientation(self):
        self.mapper.set_controller_mapping(lock_orientation=not self.mapper.lock_orientation)
        self.mapping_status()

    def recenter(self):
        if self.feedback:self.feedback.request_feedback(False)
        self.mapper.set_active(False)
        self.recenter_requested = True

    def reset(self):
        self.reference_request=None
        if self.feedback:self.feedback.request_feedback(False)
        self.reset_requested = True

    def key(self, event):
        import carb.input
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if self.joint_control and event.input.name in ('L', 'C'):
                return True
            action = {'SPACE': self.toggle, 'C': self.recenter, 'R': self.reset, 'L': self.orientation}.get(event.input.name)
            if action:
                action()
        return True

    def close(self):
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
        self.window.destroy()
        self.viewport.destroy()
        if self.wrist_window:
            self.wrist_window.destroy()
        if self.rear_window:
            self.rear_window.destroy()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--family', default='kitchen')
    result.add_argument('--task', default='breakfast_serving')
    result.add_argument('--sandbox', action='store_true', help='Use a table-edge robot and two objects for diagnostics; no benchmark task or score')
    result.add_argument('--workbench-only', action='store_true',
                        help='Load the versioned task workbench without unrelated room scenery')
    result.add_argument('--no-task-scoring', action='store_true',
                        help='Unscored rigid practice; keep physical contact safety checks')
    result.add_argument('--hide-task-displays', action='store_true',
                        help='Hide task-driven displays for unscored practice without task observation')
    result.add_argument('--keep-task-observer', action='store_true',
                        help='Keep task-driven visuals and observation during unscored practice')
    result.add_argument('--arm-base-yaw', type=float, metavar='DEG',
                        help='Absolute base heading in degrees for unscored practice')
    result.add_argument('--arm-home', type=float, nargs=7, metavar='RAD',
                        help='Explicit single-arm initial posture for unscored practice')
    result.add_argument('--practice-contact-warnings', action='store_true',
                        help='Unscored practice: report penetrations without pausing')
    result.add_argument('--physics-profile', choices=('scene', 'cpu_rigid_v1'), default='scene',
                        help='Explicit operator-only physics override; scene preserves the release configuration')
    result.add_argument('--mode', choices=('hands', 'controllers'), default='hands')
    result.add_argument('--bind', action='append', metavar='ROBOT=PICO', help='E.g. right=right or front=right')
    result.add_argument('--port', type=int, default=5557)
    result.add_argument('--scale', type=float, default=1.)
    result.add_argument('--control-point', choices=('wrist', 'pinch'), default='wrist',
                        help='Controller rotation pivot: wrist or fixed pinch-prepose center')
    result.add_argument('--lock-orientation', action='store_true',
                        help='Controllers translate without rotating the calibrated hand orientation')
    result.add_argument('--headless', action='store_true')
    result.add_argument('--start-active', action='store_true', help='Arm at the first fresh frame; GUI defaults to paused')
    result.add_argument('--steps', type=int, default=0, help='Stop after this many native steps (0: interactive)')
    result.add_argument('--output', type=Path)
    result.add_argument('--camera', default='overview', help='Main stereo camera (default: fixed overview)')
    result.add_argument('--side-view', action='store_true', help='Fixed side overview for comparing leader and simulated arm posture')
    result.add_argument('--inset-camera', default='wrist', help='Mono picture-in-picture camera; use none to disable')
    result.add_argument('--rear-camera', action='store_true', help='Additional fixed desktop view from behind the arm toward the workbench')
    result.add_argument('--head-tracked', action='store_true', help='Move a world-camera view with the headset; incompatible with wrist cameras')
    result.add_argument('--width', type=int, default=640)
    result.add_argument('--height', type=int, default=480)
    result.add_argument('--video-fps', type=float, default=30.)
    result.add_argument('--video-listen', default='0.0.0.0:13579')
    result.add_argument('--no-video', action='store_true')
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--profile', action='store_true', help='Record render, mapping and native-step wall times in session evidence')
    result.add_argument('--review-dir', type=Path, help='Existing human review campaign; record rigid states, contacts and 1 Hz keyframes')
    return result


def load_inputs(args):
    if args.practice_contact_warnings and not (args.no_task_scoring or args.sandbox):
        raise ValueError('--practice-contact-warnings requires --no-task-scoring or --sandbox')
    from bench.isaacsim.release import task_inputs
    from scenes.workrooms.catalog import load_task
    if args.sandbox:
        if args.workbench_only:
            raise ValueError('--workbench-only requires a released workroom task, not --sandbox')
        from scenes.teleop_sandbox import inputs
        scene, initial, release = inputs()
    else:
        loaded = load_task(args.family, args.task)
        release = Path(loaded['definition']['source_release']['path'])
        _, scene, _, initial = task_inputs(release, args.task, seed=args.seed)
        if args.workbench_only:
            from scenes.workrooms.teleop_workbench import apply
            scene = apply(scene, loaded['definition'])
    if args.physics_profile != 'scene' or args.no_task_scoring:
        scene = deepcopy(scene)
    if args.physics_profile != 'scene':
        from bench.simulation.isaacsim_backend.readback_profile import select_profile
        scene['native_readback_profile'] = args.physics_profile
        select_profile(scene)  # Reject particles/cloth/soft bodies before starting Isaac.
    if args.hide_task_displays:
        if not args.no_task_scoring or args.keep_task_observer:
            raise ValueError('--hide-task-displays requires unscored practice with task observation off')
        bundle = scene.get('native_task_bundle', {})
        bindings = bundle.pop('visual_state_bindings', {})
        bundle['hidden_display_prims'] = [item['target_prim_path'] for item in bindings.get('bindings', [])]
    if args.no_task_scoring:
        from bench.simulation.isaacsim_backend.native_task_bundle import task_observer_enabled
        scene['native_task_observer_enabled'] = args.keep_task_observer
        scene['unscored_practice'] = True
        task_observer_enabled(scene)
    if args.arm_base_yaw is not None:
        import math
        if not (args.no_task_scoring or args.sandbox) or not math.isfinite(args.arm_base_yaw):
            raise ValueError('--arm-base-yaw requires unscored practice and a finite angle')
        scene = deepcopy(scene)
        yaw = math.radians(args.arm_base_yaw)
        scene['robot']['base_quaternion_wxyz'] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
    if args.arm_home is not None:
        if not (args.no_task_scoring or args.sandbox):
            raise ValueError('--arm-home requires unscored practice')
        import numpy as np
        model = public_robot_geometry(robot_configuration(scene))
        if hasattr(model, 'sides'):
            raise ValueError('--arm-home requires a single arm')
        if not all(np.isfinite(q) and joint.lower <= q <= joint.upper
                   for q, joint in zip(args.arm_home, model.arm.active)):
            raise ValueError('--arm-home must satisfy the original finite joint limits')
        scene, initial = deepcopy(scene), deepcopy(initial)
        scene['robot']['arm_home_rad'] = list(args.arm_home)
        initial.setdefault('joint_positions_rad', {}).update(zip(model.arm.names, args.arm_home))
    return scene, initial, release


def run(args, *, receiver=None, source_kind='xr_sdk', on_finish=None, mapper_factory=Retargeter, feedback=None):
    from bench.simulation.isaacsim_backend.env import FR3DexterousIsaacEnv
    if not 0 < args.video_fps <= 50 or args.steps < 0 or min(args.width, args.height) < 64:
        raise ValueError('Invalid frame rate, image dimensions or step count')
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise ValueError('Select the NVIDIA GPU by UUID in CUDA_VISIBLE_DEVICES')
    scene, initial, release = load_inputs(args)
    scene = apply_in_hand_wrist_camera(scene)
    if getattr(args,'side_view',False):
        if args.camera!='overview' or args.head_tracked:
            raise ValueError('--side-view requires a fixed overview camera')
        scene=apply_side_overview(scene)
    if getattr(args, 'unrestricted_arm', False):
        if not (args.sandbox or args.no_task_scoring):
            raise ValueError('Unrestricted arm is an unscored practice override')
        scene.setdefault('isaacsim', {})['arm_motion_policy'] = 'unrestricted_practice_v1'
    review_digest = None
    if args.review_dir:
        from scripts.teleop.review import Recorder, append_event, check_release
        if args.sandbox:
            raise ValueError('Review requires a released task')
        review_digest = check_release(args.review_dir, args.family, args.task, release)
    output = args.output or output_root()/'pico'/f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    if not args.headless and getattr(args,'arm_source',None)=='factr':
        # Kit builds the glyph atlas at startup; widget font styles alone cannot
        # add Chinese characters to the default Latin-only atlas.
        font=subprocess.run(['fc-match','-f','%{file}','Noto Sans CJK SC'],
                            capture_output=True,text=True,check=True).stdout
        glyphs=output/'operator_ui_glyphs.txt'
        text=''.join((Path(__file__).with_name(name)).read_text() for name in ('pico_isaac.py','diagnostics.py'))
        glyphs.write_text(''.join(sorted({c for c in text if ord(c)>127})),encoding='utf-8')
        sys.argv.extend([f'--/app/font/customFontPath={font}',
                         '--/app/font/customRegionFiles='+json.dumps([str(glyphs)]),
                         '--/app/font/overresolutionSize=32'])
    root = Path(__file__).resolve().parents[2]
    report = {'kind': 'human_teleoperation', 'source_kind': source_kind, 'formal_evaluation': False,
              'task_success': None, 'physics_valid': None, 'family': None if args.sandbox else args.family,
              'task': None if args.sandbox else args.task, 'release': str(release) if release else None,
              'scene_kind': ('teleop_sandbox' if args.sandbox else
                             'teleop_workbench' if args.workbench_only else 'released_task'),
              'scene_variant': scene.get('teleop_scene_variant'),
              'runtime_overrides': {
                  **({'task_displays_hidden': True} if args.hide_task_displays else {}),
                  **({'arm_base_yaw_deg': args.arm_base_yaw} if args.arm_base_yaw is not None else {}),
                  **({'arm_motion_policy': 'unrestricted_practice_v1'} if getattr(args, 'unrestricted_arm', False) else {}),
                  **({'arm_home_rad': args.arm_home} if args.arm_home is not None else {}),
                  **({} if args.physics_profile == 'scene' else {'native_readback_profile': args.physics_profile}),
                  **({'native_task_observer_enabled': args.keep_task_observer} if args.no_task_scoring else {})},
              'wrist_camera_policy': {'version': 'in_hand_wrist_camera_v1', **IN_HAND_WRIST_CAMERA_V1},
              'side_overview':getattr(args,'side_view',False),
              'contact_policy': 'practice_warning' if args.practice_contact_warnings else 'strict',
              'practice_contact_limits': ({'penetration_stop_enabled': False, 'control_window_s': .02}
                  if args.practice_contact_warnings else None),
              'task_scoring': 'disabled_for_practice' if args.no_task_scoring else 'scene_default',
              'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
              'driver_override': os.environ.get('ISAAC_ALLOW_UNVALIDATED_DRIVER'),
              'process_threads': {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
              'started_utc': datetime.now(timezone.utc).isoformat(), 'steps': 0,
              'source_snapshot': verify_if_snapshot(root, required_modules=('scripts.teleop.pico_isaac',),
                                                    entrypoint_file=Path(__file__)),
              'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    def write_report():
        (output/'session.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    report['status'] = 'initializing'
    (output/'scene.json').write_text(json.dumps(scene, indent=2)+'\n')
    (output/'initial_state.json').write_text(json.dumps(initial, indent=2)+'\n')
    write_report()
    env = writer = controls = recorder = None
    timing_totals = dict(render_s=0., mapping_s=0., step_s=0.)
    rendered_frames = 0
    stop_requested, signal_handlers = False, {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    receiver = receiver or Receiver(args.port)
    try:
        if args.review_dir:
            recorder = Recorder(output)
            append_event(args.review_dir, {'event': 'session_started', 'family': args.family,
                         'task': args.task, 'session': str(output.resolve())})
            report['review_release_sha256'] = review_digest
            report['review_recording'] = {'states': 'rigid objects, robot and contacts at control steps',
                                         'images': '1 Hz main stereo-left composite; not continuous video',
                                         'soft_body_replay': False}
        faulthandler.dump_traceback_later(60)
        env = FR3DexterousIsaacEnv(scene, device='cuda:0', headless=args.headless,
                                  render_width=args.width, render_height=args.height)
        report['python_switch_interval_s'] = sys.getswitchinterval()
        observation = env.reset(seed=args.seed, initial_state=initial)
        if recorder:
            recorder.state(observation, 0, event='reset')
        geometry = public_robot_geometry(robot_configuration(scene))
        names = list(geometry.sides) if hasattr(geometry, 'sides') else ['right']
        bindings = dict(item.split('=', 1) for item in args.bind) if args.bind else {n: n for n in names}
        spec = env.action_spec()
        initial_targets = {n: initial['joint_positions_rad'].get(n, spec['initial_joint_positions_rad'][n])
                           for n in spec['joint_names']}
        mapper = mapper_factory(geometry, initial_targets, bindings, mode=args.mode, scale=args.scale,
                            tracking_grace_s=.3, hands_recovery=True,
                            control_point=args.control_point, lock_orientation=args.lock_orientation)
        from bench.adapters.pico.controller_hand import VERSION as controller_hand_version
        report['controller_hand_mapping'] = controller_hand_version
        report['input_mapping'] = getattr(mapper, 'input_mapping', None)
        report['controller_mapping'] = {'control_point': mapper.control_point,
            'lock_orientation': mapper.lock_orientation, 'scale': mapper.scale,
            'pinch_offsets_wrist_m': {n: p.tolist() for n, p in mapper.grasp_offsets.items()}}
        report['tracking_grace_s'] = mapper.tracking_grace_ns/1e9
        report['hands_recovery'] = {'automatic_window_s': 2., 'stable_dwell_s': .25, 'spare_hand_pinch': True}
        view = StereoView(env, args.camera, width=args.width, height=args.height, head_tracked=args.head_tracked,
                          robot_poses=observation['robot'],
                          inset_camera=None if args.inset_camera == 'none' else args.inset_camera,
                          rear_camera=args.rear_camera)
        if not args.no_video:
            writer = FrameWriter(args.width, args.height, listen=args.video_listen)
        if not args.headless:
            controls = Controls(mapper, inset=view.inset is not None, rear=args.rear_camera,
                                joint_control=getattr(mapper, 'control_space', None) == 'joint',feedback=feedback,
                                capture_on_start=getattr(args,'capture_reference_on_start',False))
        guard = TeleopContactGuard(practice=args.practice_contact_warnings)
        reporter = HoldReporter()
        episode, next_video, armed_once = 0, 0., False
        align_requested=getattr(args,'align_from_leader',False)
        left = right = None
        faulthandler.cancel_dump_traceback_later()
        signal_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        report['status'] = 'running'
        write_report()
        print(f'PICO ready: {args.mode}; evidence: {output}', flush=True)
        started = time.monotonic()
        with (output/'frames.jsonl').open('x') as log:
            while not stop_requested and env.app.is_running() and (not args.steps or report['steps'] < args.steps):
                tick = time.monotonic()
                timings = dict(render_s=0., mapping_s=0., step_s=0.)
                if controls and controls.reset_requested:
                    if feedback:feedback.request_feedback(False)
                    observation = env.reset(seed=args.seed, initial_state=initial)
                    mapper.acknowledge(initial_targets)
                    mapper.recalibrate()
                    guard = TeleopContactGuard(practice=args.practice_contact_warnings)
                    episode += 1
                    if recorder:
                        recorder.state(observation, episode, event='reset')
                    controls.reset_requested = False
                    log.write(json.dumps({'event': 'reset', 'episode': episode})+'\n')
                if controls and controls.recenter_requested:
                    view.recenter()
                    controls.recenter_requested = False
                frame = receiver.read()
                if align_requested or (controls and controls.align_requested):
                    if feedback:feedback.request_feedback(False)
                    mapper.set_active(False)
                    if feedback and feedback.feedback_state['enabled']:
                        mapper.status='Waiting for leader torque-off before simulation alignment'
                        time.sleep(.01)
                        continue  # The serial owner must confirm torque-off before resetting the follower.
                    if frame is not None:
                        arm=getattr(mapper,'leader',mapper)
                        try:
                            if not (args.sandbox or args.no_task_scoring):
                                raise ValueError('Leader pose reset is only available in unscored practice')
                            aligned,receipt=arm.initial_state_from_input(frame,initial,now_ns=time.monotonic_ns())
                        except ValueError as error:
                            mapper.status='Leader alignment refused: '+str(error)
                            print(mapper.status,flush=True)
                            log.write(json.dumps({'event':'leader_alignment_refused','reason':str(error)})+'\n')
                        else:
                            observation=env.reset(seed=args.seed,initial_state=aligned)
                            initial=aligned
                            initial_targets=dict(initial_targets)|receipt['arm_joint_positions_rad']
                            mapper.acknowledge(initial_targets)
                            mapper.recalibrate()
                            mapper.status='Simulation aligned to leader; Start to follow fixed joints'
                            guard=TeleopContactGuard(practice=args.practice_contact_warnings)
                            episode+=1
                            if recorder:recorder.state(observation,episode,event='leader_alignment')
                            report['leader_initialization']=receipt
                            write_report()
                            log.write(json.dumps({'event':'leader_alignment','episode':episode,**receipt})+'\n')
                            print(mapper.status,flush=True)
                        align_requested=False
                        if controls:controls.align_requested=False
                        log.flush()
                # Align operator motion to the initial view, not the robot base.
                if view.reference is None and frame is not None and frame.tracking['head'] is not None:
                    from bench.adapters.pico.retarget import pose_matrix
                    mapper.rotation = view.base[:3, :3] @ pose_matrix(frame.tracking['head'])[:3, :3].T
                    mapper.anchors.clear()  # A late first head pose changes axes; re-anchor without jumping.
                if time.monotonic() >= next_video:
                    rendering = time.monotonic()
                    view.update(frame.tracking['head'] if frame else None, observation['robot'])
                    left, right = view.render(mapper.control_points(observation['joint_positions_rad']))
                    if recorder:
                        recorder.image(left, observation, episode)
                    if controls:
                        controls.show_frame(view.frames['pico_left'], view.frames.get('pico_inset'),
                                            view.frames.get('pico_rear'))
                    if writer:
                        writer.check()
                        writer.publish(left, right)
                    timings['render_s'] = time.monotonic()-rendering
                    timing_totals['render_s'] += timings['render_s']
                    rendered_frames += 1
                    next_video = rendering + 1/args.video_fps
                if controls:
                    grips = ''
                    if mapper.mode == 'controllers' and frame is not None:
                        grips = '\n' + '  '.join(f'{side} grip: {frame.tracking["controllers"][side]["grip"]:.0%} '
                            f'trigger: {frame.tracking["controllers"][side]["trigger"]:.0%}'
                            for side in mapper.bindings.values() if side in frame.tracking['controllers'])
                    input_name = 'master/slave' if getattr(mapper, 'control_space', None) == 'joint' else mapper.mode
                    controls.label.text = f'{input_name}: {mapper.status}{grips}\nSimulation: {observation["sim_time_s"]:.2f} s'
                    if args.no_task_scoring:
                        controls.label.text += '\nPractice: task scoring off'
                    contact_message = guard.message(observation['sim_time_s'])
                    if contact_message:
                        controls.label.text += '\n' + contact_message
                    if mapper.diagnostics:
                        controls.label.text += '\n' + '; '.join(mapper.diagnostics.values())
                    if feedback:
                        state=feedback.feedback_state
                        controls.label.text+='\nForce feedback: '+state['reason']
                        if 'model_demand_ma' in state:
                            command=max(abs(v) for v in state['current_ma'])
                            demand=max(abs(v) for v in state['model_demand_ma'])
                            controls.label.text+=f'\nGravity {"ON" if state["enabled"] else "OFF"}: peak command {command:.0f} mA; model needs {demand:.0f} mA'
                    if controls.joint_control:
                        from scripts.teleop.diagnostics import joint_operator_status
                        controls.label.text=joint_operator_status(mapper,receiver.health(),
                            feedback.feedback_state if feedback else None,contact_message)
                if guard.trigger or observation['truncated']:
                    if feedback:feedback.request_feedback(False)
                    mapper.set_active(False)
                    mapper.status = (guard.message(observation['sim_time_s']) if guard.trigger else
                                     'Simulation time limit reached; Reset scene, then Start')
                    if args.headless:
                        break
                    time.sleep(.02)
                    continue
                mapping = time.monotonic()
                frame = receiver.read()
                if args.start_active and frame is not None and not armed_once:
                    mapper.set_active(True)
                    armed_once = True
                targets = mapper.command(frame, observation['joint_positions_rad'], now_ns=time.monotonic_ns())
                if controls:
                    controls.complete_reference_capture()
                # A slow IK/render must not turn an expired sample into a fresh command.
                if frame is not None and time.monotonic_ns()-frame.host_ns > 150_000_000:
                    targets = mapper.hold(receiver.read(), now_ns=time.monotonic_ns())
                timings['mapping_s'] = time.monotonic()-mapping
                timing_totals['mapping_s'] += timings['mapping_s']
                action = {'joint_positions_rad': targets, 'duration_s': .02}
                stepping = time.monotonic()
                guard.assert_can_step()
                observation = env.step(action)
                timings['step_s'] = time.monotonic()-stepping
                timing_totals['step_s'] += timings['step_s']
                mapper.acknowledge(targets)
                trigger = guard.observe(observation)
                if feedback:
                    if trigger:feedback.request_feedback(False)
                    feedback.observe(mapper,observation,now_ns=time.monotonic_ns())
                issues = control_issues(mapper, guard)
                if observation['truncated']:
                    issues.append({'channel':'simulation','code':'time_limit',
                                   'reason':'Simulation time limit reached; Reset scene, then Start'})
                reporter.update(issues, now_ns=time.monotonic_ns())
                report['steps'] += 1
                if recorder:
                    recorder.state(observation, episode)
                log.write(json.dumps({'episode': episode, 'host_ns': time.monotonic_ns(),
                    'input': asdict(frame) if frame else None, 'mode': mapper.mode, 'armed': mapper.armed,
                    'input_health': receiver.health(),
                    'controller_mapping': {'control_point': mapper.control_point, 'lock_orientation': mapper.lock_orientation, 'scale': mapper.scale},
                    'status': mapper.status, 'diagnostics': mapper.diagnostics, 'control_issues': issues,
                    **({'leader_feedback':feedback.feedback_state} if feedback else {}),
                    **({'arm_alignment': mapper.alignment} if hasattr(mapper, 'alignment') else {}), 'finger_target_errors_m': mapper.finger_errors, 'action': action,
                    'sim_time_s': observation['sim_time_s'], 'measured_joints': observation['joint_positions_rad'],
                    'safety_trigger': trigger, 'contact_warning': guard.warning,
                    **({'timing': timings, 'contact_sampling': observation['contacts']['sampling']}
                       if args.profile else {})}, allow_nan=False)+'\n')
                if guard.warning or trigger or observation['truncated']:
                    log.flush()
                time.sleep(max(0., .02-(time.monotonic()-tick)))
        report.update(elapsed_wall_s=time.monotonic()-started, elapsed_sim_s=observation['sim_time_s'],
                      safety_trigger=guard.trigger, contact_warning_windows=guard.warning_windows,
                      last_contact_warning=guard.last_warning, input_rejections=receiver.rejected,
                      status='interrupted' if stop_requested else 'stopped', native_metadata=env.metadata())
        if args.profile:
            import numpy as np
            np.savez_compressed(output/'final_state.npz', **env.capture_state())
            report['final_task_state'] = observation.get('task_state')
            report['timing'] = {**timing_totals, 'rendered_frames': rendered_frames,
                                'scope': 'Main-loop wall time; excludes startup and shutdown'}
        from PIL import Image
        if left is not None:
            Image.fromarray(left).save(output/'left.png')
            Image.fromarray(right).save(output/'right.png')
        if on_finish:
            on_finish(output)
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if recorder:
            try:
                recorder.close()
            except Exception as error:
                report['recording_error'] = str(error)
        # Kit fast shutdown exits the interpreter: seal receipts before close().
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_report()
        for sig, handler in signal_handlers.items():
            signal.signal(sig, handler)
        try:
            if controls:
                controls.close()
            if writer:
                writer.close()
            receiver.close()
        finally:
            if env:
                env.close()
    return output


if __name__ == '__main__':
    run(parser().parse_args())
