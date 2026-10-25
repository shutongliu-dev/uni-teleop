import argparse
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sonic_mujoco.cameras import CameraRig, CameraSampler, scan_mapper
from sonic_mujoco.controllers.sonic import SonicController, SonicEncoder
from sonic_mujoco.envs.mujoco.g1 import (
    MujocoG1BasketLoadingEnv,
    MujocoG1Env,
    MujocoG1LabScanEnv,
    MujocoG1SweepEnv,
    RobotCommand,
)
from sonic_mujoco.envs.mujoco.g1.registry import G1_ENVIRONMENTS
from sonic_mujoco.recording import EpisodeRecorder
from sonic_mujoco.scan import ScanRendererClient
from sonic_mujoco.superdex import HandCommandPublisher
from sonic_mujoco.tactile_calibration import load_tactile_calibration_profile
from sonic_mujoco.teleop import (
    ContactHaptics,
    PicoControls,
    PicoEvents,
    PicoTeleop,
    PicoVideo,
    PicoZmqTeleop,
    ScanPicoVideo,
    TeleopBase,
    TeleopCommand,
    TeleopMode,
    next_mode,
)
from sonic_mujoco.teleop.pico_video import _validate_scan_alignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = Path(os.environ.get("SONIC_POLICY_DIR", str(PROJECT_ROOT / "models")))
Array = NDArray[np.float64]

ENVIRONMENTS = G1_ENVIRONMENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the MuJoCo G1 from PICO poses")
    parser.add_argument(
        "--encoder",
        type=Path,
        default=POLICY_DIR / "model_encoder.onnx",
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        default=POLICY_DIR / "model_decoder.onnx",
    )
    parser.add_argument(
        "--endpoint",
        help="read PICO poses from an external ZMQ endpoint",
    )
    parser.add_argument("--no-pico-video", action="store_true")
    parser.add_argument(
        "--superdex-hand-endpoint",
        help="also publish five-finger targets to a SuperDex hand runtime",
    )
    parser.add_argument(
        "--no-dexhand",
        "--no-dex3",
        dest="no_dexhand",
        action="store_true",
        help="disable PICO tracking and five-finger hand control",
    )
    parser.add_argument(
        "--no-neck",
        action="store_true",
        help="keep the simulated neck centered instead of following the PICO head",
    )
    parser.add_argument(
        "--pico-device",
        default=os.environ.get("SONIC_PICO_DEVICE"),
        help="XRRobotKit device name used for controller haptics",
    )
    parser.add_argument("--video-listen", default="0.0.0.0:13579")
    parser.add_argument(
        "--no-desktop-video",
        action="store_true",
        help="disable the desktop robot/headset camera window",
    )
    parser.add_argument(
        "--scan-renderer-endpoint",
        help="render PICO video from a reconstructed-scene service",
    )
    parser.add_argument(
        "--scan-units-per-meter",
        type=float,
        help=(
            "override the reconstruction scale used for robot translation; "
            "lab_scan otherwise uses its versioned scene metadata"
        ),
    )
    parser.add_argument(
        "--scan-collision",
        choices=("auto", "mesh", "boxes"),
        default="auto",
        help=(
            "lab_scan collision backend; mesh is the experimental reconstructed "
            "surface and boxes is the conservative fallback"
        ),
    )
    parser.add_argument(
        "--scan-collision-manifest",
        type=Path,
        help=(
            "local reconstructed-scene collision manifest; can also be set with "
            "SONIC_RECONSTRUCTION_COLLISION"
        ),
    )
    parser.add_argument("--record-dir", type=Path, default=Path("records"))
    parser.add_argument("--no-record-video", action="store_true")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=10,
        help="simulation-time rate of all four RGB/depth/normal cameras (up to 50 Hz)",
    )
    parser.add_argument(
        "--tactile-profile",
        type=Path,
        help="versioned tactile Sim2Real profile applied to hardware-shaped packets",
    )
    parser.add_argument(
        "--tactile-seed",
        type=int,
        default=0,
        help="base seed for repeatable virtual tactile-sensor variation",
    )
    parser.add_argument("--task")
    parser.add_argument(
        "--scene-seed",
        type=int,
        help="repeat this scene seed on every reset; otherwise draw a new layout",
    )
    parser.add_argument(
        "--task-start",
        choices=("recording", "pose"),
        default="recording",
        help="start scripted task timing on recording (default) or when entering POSE",
    )
    parser.add_argument(
        "--scene",
        choices=(*ENVIRONMENTS, "lab_scan"),
        default="empty",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


@dataclass(slots=True)
class _RuntimeState:
    command: TeleopCommand | None = None
    hand_target: Array | None = None
    neck_target: Array | None = None
    token: Array | None = None
    action: Array | None = None
    robot_command: RobotCommand | None = None
    completed: int = 0
    started: bool = False
    task_completed: bool = False


@dataclass(frozen=True, slots=True)
class _SessionResources:
    args: argparse.Namespace
    env: MujocoG1Env
    encoder: SonicEncoder
    controller: SonicController
    teleop: TeleopBase
    recorder: EpisodeRecorder
    video: PicoVideo | ScanPicoVideo | None
    haptics: ContactHaptics | None
    superdex: HandCommandPublisher | None
    cameras: CameraSampler | None = None


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.scene_seed is not None and args.scene_seed < 0:
        raise SystemExit("--scene-seed must be non-negative")
    for model in (args.encoder, args.decoder):
        if not model.is_file():
            raise SystemExit(f"SONIC model not found: {model}")
    if (
        args.scene == "lab_scan"
        and not args.no_record_video
        and not args.scan_renderer_endpoint
    ):
        raise SystemExit(
            "lab_scan RGB recording requires the reconstructed-scene video renderer"
        )
    if args.scene != "lab_scan" and args.scan_collision != "auto":
        raise SystemExit("--scan-collision only applies to --scene lab_scan")
    if args.scene != "lab_scan" and args.scan_collision_manifest is not None:
        raise SystemExit("--scan-collision-manifest only applies to --scene lab_scan")
    if not 0 < args.camera_fps <= 50:
        raise SystemExit("--camera-fps must be in (0, 50]")
    if (
        min(args.camera_width, args.camera_height) < 4
        or args.camera_width % 2
        or args.camera_height % 2
    ):
        raise SystemExit("camera dimensions must be even and at least four pixels")


def _create_environment(args: argparse.Namespace) -> MujocoG1Env:
    profile = (
        load_tactile_calibration_profile(args.tactile_profile)
        if args.tactile_profile is not None
        else None
    )
    if args.scene == "lab_scan":
        env = MujocoG1LabScanEnv(
            collision_geometry=args.scan_collision,
            collision_path=args.scan_collision_manifest,
        )
    else:
        env = ENVIRONMENTS[args.scene]()
    if profile is not None:
        env.configure_tactile_profile(profile, seed=args.tactile_seed)
    return env


def _create_video(
    args: argparse.Namespace,
    env: MujocoG1Env,
    teleop: TeleopBase,
) -> PicoVideo | ScanPicoVideo | None:
    if args.endpoint or args.no_pico_video:
        return None
    if not args.scan_renderer_endpoint:
        return PicoVideo(
            env.model,
            env.data,
            listen=args.video_listen,
            desktop=not args.no_desktop_video,
        )
    if not isinstance(teleop, PicoTeleop):
        raise TypeError("scan video requires direct PICO tracking")

    scale = args.scan_units_per_meter
    if scale is None:
        scale = getattr(env, "scan_units_per_meter", None)
    if scale is None:
        raise ValueError("a reconstruction scale is required for scan video")
    return ScanPicoVideo(
        env.model,
        env.data,
        endpoint=args.scan_renderer_endpoint,
        scan_units_per_meter=scale,
        headset_pose_provider=lambda: teleop.headset_pose,
        alignment_metadata=getattr(env, "reconstruction_metadata", None),
        listen=args.video_listen,
        desktop=not args.no_desktop_video,
    )


def _attach_scene_metadata(
    video: PicoVideo | ScanPicoVideo | None,
    env: MujocoG1Env,
) -> None:
    if video is None or not hasattr(env, "collision_geometry"):
        return
    video.metadata.update(
        {
            "collision_geometry": env.collision_geometry,
            "collision_manifest_sha256": env.collision_manifest_sha256,
            "collision_mesh_manifest_sha256": getattr(
                env, "mesh_manifest_sha256", None
            ),
        }
    )


def _create_recorder(
    args: argparse.Namespace,
    env: MujocoG1Env,
    controller: SonicController,
    video: PicoVideo | ScanPicoVideo | None,
    cameras: CameraSampler | None = None,
) -> EpisodeRecorder:
    fps = round(1.0 / (env.timestep * controller.steps_per_action))
    video_metadata = (
        dict(cameras.metadata)
        if cameras is not None
        else dict(video.metadata)
        if video is not None
        else {}
    )
    if (cameras is not None or video is not None) and not args.no_record_video:
        video_metadata.update(
            dataset_sample_rate_hz=fps,
            dataset_resampling="latest_frame_sample_and_hold",
        )
    return EpisodeRecorder(
        args.record_dir,
        args.scene,
        model=env.model,
        data=env.data,
        body_names=env.contacts.body_names,
        fps=fps,
        task=args.task,
        record_video=not args.no_record_video,
        video_source=(
            "robot_camera_bundle"
            if cameras is not None
            else "reconstructed_scene_composite"
            if isinstance(video, ScanPicoVideo)
            else "mujoco_head_camera"
        ),
        video_metadata=video_metadata,
        tactile_layout=env.tactile_skin_layout,
        tactile_metadata=env.tactile_adapter.metadata,
        camera_metadata=None if cameras is None else cameras.metadata,
        task_metadata=getattr(env, "task_metadata", None),
    )


def _create_cameras(args, env, stack):
    if args.no_record_video:
        return None
    client = mapper = None
    if args.scan_renderer_endpoint:
        client = ScanRendererClient(args.scan_renderer_endpoint)
        stack.callback(client.close)
        info = client.info()
        alignment = getattr(env, "reconstruction_metadata", None)
        if alignment is not None:
            _validate_scan_alignment(info, alignment)
        scale = args.scan_units_per_meter or getattr(env, "scan_units_per_meter", None)
        if scale is None:
            raise ValueError("scan recording requires an explicit reconstruction scale")
        mapper = scan_mapper(env.model, info, scale)
    rig = CameraRig(
        env.model,
        env.data,
        width=args.camera_width,
        height=args.camera_height,
        scan_client=client,
        mapper=mapper,
    )
    stack.callback(rig.close)
    if client is not None:
        rig.metadata["reconstruction"] = {
            key: info.get(key)
            for key in (
                "checkpoint",
                "step",
                "anchor_image",
                "anchor_camera_to_world",
                "scan_up",
            )
        }
        rig.metadata["collision_alignment"] = alignment
        rig.metadata["collision_geometry"] = getattr(env, "collision_geometry", None)
        rig.metadata["collision_manifest_sha256"] = getattr(
            env, "collision_manifest_sha256", None
        )
        rig.metadata["collision_mesh_manifest_sha256"] = getattr(
            env, "mesh_manifest_sha256", None
        )
    return CameraSampler(rig, args.camera_fps)


def _create_resources(
    args: argparse.Namespace,
    stack: ExitStack,
) -> _SessionResources:
    env = _create_environment(args)
    stack.callback(env.close)
    encoder = SonicEncoder.from_onnx(args.encoder)
    controller = SonicController.from_onnx(args.decoder)
    teleop = (
        PicoZmqTeleop(args.endpoint)
        if args.endpoint
        else PicoTeleop(enable_dexhand=not args.no_dexhand)
    )
    stack.callback(teleop.close)

    haptics = None
    if isinstance(teleop, PicoTeleop) and args.pico_device:
        haptics = ContactHaptics(
            env.contacts.body_names,
            lambda left, right, duration, frequency: teleop.send_haptics(
                args.pico_device, left, right, duration, frequency
            ),
        )
    video = _create_video(args, env, teleop)
    if video is not None:
        stack.callback(video.close)
    _attach_scene_metadata(video, env)
    env.reset(seed=args.scene_seed)
    cameras = _create_cameras(args, env, stack)
    recorder = _create_recorder(args, env, controller, video, cameras)
    superdex = (
        HandCommandPublisher(args.superdex_hand_endpoint)
        if args.superdex_hand_endpoint
        else None
    )
    if superdex is not None:
        stack.callback(superdex.close)
    return _SessionResources(
        args=args,
        env=env,
        encoder=encoder,
        controller=controller,
        teleop=teleop,
        recorder=recorder,
        video=video,
        haptics=haptics,
        superdex=superdex,
        cameras=cameras,
    )


class _TeleopSession:
    def __init__(self, resources: _SessionResources) -> None:
        self.resources = resources
        self.state = _RuntimeState()
        self.direct = isinstance(resources.teleop, PicoTeleop)
        self.mode = TeleopMode.OFF if self.direct else TeleopMode.POSE
        self.scan_video = (
            resources.video if isinstance(resources.video, ScanPicoVideo) else None
        )
        self.haptics = resources.haptics
        self._headset_tracking_announced = False
        self._headset_tracking_warned = False
        self._headset_tracking_deadline = time.monotonic() + 5.0
        self._running = True

    def run(self) -> None:
        self._announce()
        if self.mode is TeleopMode.POSE:
            self._start_task("pose")
        if not self.resources.args.headless:
            self.resources.env.render()
        while self._keep_running():
            tick = time.monotonic()
            if self._tick():
                continue
            if not self._running:
                break
            control_dt = (
                self.resources.env.timestep * self.resources.controller.steps_per_action
            )
            time.sleep(max(0.0, control_dt - (time.monotonic() - tick)))

    def _keep_running(self) -> bool:
        limit = self.resources.args.steps
        return limit == 0 or self.state.completed < limit

    def _tick(self) -> bool:
        command = self.resources.teleop.read()
        self._accept_command(command)
        events = self.resources.teleop.pop_events() if self.direct else PicoEvents()
        controls = self.resources.teleop.controls if self.direct else PicoControls()

        self._update_mode(events)
        if events.reset_scene:
            self._reset_scene()
            return True
        self._update_recording(events)
        self._advance_loading_scene()
        stepped = self._step_controller(command, controls)
        self._update_haptics(stepped)
        self._check_task_success()
        self._render()
        return False

    def _accept_command(self, command: TeleopCommand | None) -> None:
        if command is None:
            return
        self.state.command = command
        if command.hand_joint_position is not None:
            self.state.hand_target = command.hand_joint_position[-1]
            if self.resources.superdex is not None:
                self.resources.superdex.publish(self.state.hand_target)
        if command.neck_joint_position is not None:
            self.state.neck_target = command.neck_joint_position[-1]

    def _update_mode(self, events: PicoEvents) -> None:
        updated = next_mode(self.mode, events)
        if updated is self.mode:
            return
        if self.resources.recorder.active and updated is TeleopMode.OFF:
            _finish_recording(self.resources.recorder)
        if self.mode is TeleopMode.POSE and updated is TeleopMode.READY:
            self._hold_current_pose()
        self.mode = updated
        if updated is TeleopMode.POSE:
            self._start_task("pose")
        self.resources.encoder.reset()
        self.resources.controller.reset()
        self.state.started = False
        print(_mode_message(self.mode))

    def _hold_current_pose(self) -> None:
        previous = self.state.robot_command
        if previous is None:
            return
        robot = self.resources.env.get_robot_state()
        zero = np.zeros_like(robot.joint_position)
        self.state.robot_command = RobotCommand(
            joint_position=robot.joint_position,
            joint_velocity=zero,
            feedforward_torque=zero,
            kp=previous.kp,
            kd=previous.kd,
            hand_joint_position=(
                None if self.resources.args.no_dexhand else previous.hand_joint_position
            ),
            neck_joint_position=(
                None if self.resources.args.no_neck else previous.neck_joint_position
            ),
        )

    def _reset_scene(self) -> None:
        recorder = self.resources.recorder
        if recorder.active:
            recorder.abort()
            print("Recording aborted before scene reset.")
        self.resources.env.reset(seed=self.resources.args.scene_seed)
        if self.mode is TeleopMode.POSE:
            self._start_task("pose")
        if self.resources.cameras is not None:
            self.resources.cameras.reset()
        if self.scan_video is not None:
            self.scan_video.recenter_headset()
        self.resources.encoder.reset()
        self.resources.controller.reset()
        self.state = _RuntimeState(completed=self.state.completed)
        print("Scene, SONIC history, and controller reset.")

    def _start_task(self, trigger: str) -> None:
        if self.resources.args.task_start == trigger and hasattr(
            self.resources.env, "start_task"
        ):
            self.resources.env.start_task()

    def _update_recording(self, events: PicoEvents) -> None:
        recorder = self.resources.recorder
        if events.abort_recording and recorder.active:
            recorder.abort()
            print("Recording aborted; buffered frames were discarded.")
            return
        if not events.toggle_recording:
            return
        if recorder.active:
            _finish_recording(recorder)
        elif self.mode is TeleopMode.POSE:
            self._start_task("recording")
            if self.resources.cameras is not None:
                self.resources.cameras.reset()
            env = self.resources.env
            recorder.start(
                tactile_metadata=env.tactile_adapter.metadata,
                task_context=env.episode_metadata()
                if hasattr(env, "episode_metadata")
                else None,
            )
            if isinstance(self.resources.env, MujocoG1BasketLoadingEnv):
                self.resources.env.start_loading()
            print("Recording started.")
        else:
            print("Enter POSE mode before starting a recording.")

    def _advance_loading_scene(self) -> None:
        if self.resources.recorder.active and isinstance(
            self.resources.env, MujocoG1BasketLoadingEnv
        ):
            self.resources.env.advance_loading(self.resources.recorder.duration_seconds)

    def _step_controller(
        self,
        command: TeleopCommand | None,
        controls: PicoControls,
    ) -> bool:
        if self.mode is TeleopMode.POSE and not controls.menu:
            return self._step_pose(command, controls)
        if self._can_hold_recording():
            self._step_and_record(self.state.robot_command, controls)
            return True
        return False

    def _step_pose(
        self,
        command: TeleopCommand | None,
        controls: PicoControls,
    ) -> bool:
        try:
            robot_state = self.resources.env.get_robot_state()
            token = self.resources.encoder.encode(robot_state, command)
        except RuntimeError:
            return False
        if token is None:
            return False

        robot_command = self.resources.controller.act(robot_state, token)
        robot_command = self._with_auxiliary_targets(robot_command)
        self.state.token = token
        self.state.action = self.resources.controller.last_action.copy()
        self.state.robot_command = robot_command
        self._step_and_record(robot_command, controls)
        if not self.state.started:
            print("PICO stream received; full-body teleoperation is running.")
            self.state.started = True
        return True

    def _with_auxiliary_targets(self, command: RobotCommand) -> RobotCommand:
        hand = None if self.resources.args.no_dexhand else self.state.hand_target
        neck = None if self.resources.args.no_neck else self.state.neck_target
        if hand is None and neck is None:
            return command
        return replace(
            command,
            hand_joint_position=hand,
            neck_joint_position=neck,
        )

    def _can_hold_recording(self) -> bool:
        return (
            self.resources.recorder.active
            and self.mode is not TeleopMode.OFF
            and self.state.command is not None
            and self.state.token is not None
            and self.state.action is not None
            and self.state.robot_command is not None
        )

    def _step_and_record(
        self,
        robot_command: RobotCommand | None,
        controls: PicoControls,
    ) -> None:
        if robot_command is None:
            raise RuntimeError("cannot step without a robot command")
        self.resources.env.step(
            robot_command,
            steps=self.resources.controller.steps_per_action,
        )
        self.state.completed += 1
        if (
            self.resources.recorder.active
            and self.state.command is not None
            and self.state.token is not None
            and self.state.action is not None
        ):
            _append_recording(
                self.resources.recorder,
                self.resources.env,
                self.state.command,
                self.state.token,
                self.state.action,
                controls,
                self.resources.video,
                self.resources.cameras,
            )

    def _update_haptics(self, stepped: bool) -> None:
        if self.haptics is None or not stepped:
            return
        if self.haptics.update(
            self.resources.env.contacts.last_frame, time.monotonic()
        ):
            return
        print("PICO haptics unavailable; disabling contact vibration.")
        self.haptics = None

    def _check_task_success(self) -> None:
        env = self.resources.env
        if hasattr(env, "task_metadata") and not self.state.task_completed:
            task = env.get_scene_state()
            if task.success or task.failed:
                print(
                    f"Task {'succeeded' if task.success else 'failed'}: {task.failure_reason or env.task}"
                )
                self.state.task_completed = True
            return
        if (
            not self.state.task_completed
            and isinstance(self.resources.env, MujocoG1SweepEnv)
            and self.resources.env.is_success()
        ):
            print("Sweep task completed.")
            self.state.task_completed = True

    def _render(self) -> None:
        args = self.resources.args
        if not args.headless:
            if not self.resources.env.viewer_running:
                self._running = False
                return
            self.resources.env.render()
        if self.resources.video is None:
            return
        self.resources.video.render(
            recording=self.resources.recorder.active,
            elapsed_seconds=self.resources.recorder.duration_seconds,
        )
        self._report_headset_tracking()

    def _report_headset_tracking(self) -> None:
        if self.scan_video is None:
            return
        if self.scan_video.headset_tracking_active:
            if not self._headset_tracking_announced:
                print(
                    "PICO headset tracking received; stereo view now follows "
                    "head rotation and translation."
                )
                self._headset_tracking_announced = True
            return
        if (
            not self._headset_tracking_warned
            and time.monotonic() >= self._headset_tracking_deadline
        ):
            print(
                "No PICO headset pose received: the view remains at the anchor "
                "and shows HEAD OFF. Enable Head and Send in XRRobotKit."
            )
            self._headset_tracking_warned = True

    def _announce(self) -> None:
        args = self.resources.args
        if args.endpoint:
            print(f"Waiting for PICO pose messages on {args.endpoint} ...")
        else:
            print("Waiting for PICO body tracking from XRRobotKit ...")
            print("Press A+B+X+Y to arm, then A+X to enter full-body POSE teleop.")
        if self.resources.video is not None:
            print(f"PICO video control is listening on {args.video_listen}.")
        if self.scan_video is not None:
            self._announce_scan_video()
        if self.haptics is not None:
            print(f"PICO contact haptics enabled for {args.pico_device}.")
        if not args.no_dexhand:
            print(
                "Five-finger dexterous hands enabled: optical tracking is preferred; "
                "controller trigger/grip is the fallback."
            )
        if not args.no_neck:
            print(
                "Two-axis neck tracking enabled: the head cameras follow PICO yaw/pitch."
            )
        if self.resources.cameras is not None:
            print(
                f"Robot camera recording: stereo ego + both wrists, RGB/depth/normals at {args.camera_fps:g} Hz; state/tactile at {self.resources.recorder.fps} Hz."
            )
        if hasattr(self.resources.env, "task_metadata"):
            print(
                f"Task timing starts on {args.task_start}; --scene-seed repeats the same layout."
            )
        if self.resources.superdex is not None:
            print(
                f"SuperDex hand targets are published on {args.superdex_hand_endpoint}."
            )

    def _announce_scan_video(self) -> None:
        args = self.resources.args
        print(
            "PICO video uses head-coupled stereo reconstruction with MuJoCo robot "
            "foreground geometry."
        )
        print(
            "On PICO, enable Head, Controller, and Send. Body tracking is only "
            "required for full-body robot control."
        )
        if not args.no_desktop_video:
            print("Desktop reconstructed-lab camera window is enabled.")
        if not args.no_record_video:
            print(
                f"Dataset records state and tactile at {self.resources.recorder.fps} "
                "Hz; robot-mounted camera captures have explicit timestamps and ages."
            )


def main() -> None:
    args = parse_args()
    _validate_arguments(args)
    with ExitStack() as stack:
        resources = _create_resources(args, stack)
        session = _TeleopSession(resources)
        try:
            session.run()
        except KeyboardInterrupt:
            if resources.recorder.write_failed:
                resources.recorder.abort()
                print(
                    "Recording interrupted during a write; the incomplete episode was discarded."
                )
        except Exception:
            if resources.recorder.active:
                resources.recorder.abort()
                print(
                    "Recording interrupted by an error; the current incomplete episode was discarded. Check the error for any recovery instructions."
                )
            raise
        finally:
            if resources.recorder.active:
                try:
                    _finish_recording(resources.recorder)
                except BaseException:
                    resources.recorder.abort()
                    print(
                        "Episode could not be saved; incomplete files were discarded."
                    )
                    raise
    if session.state.started:
        print(f"Stopped after {session.state.completed} SONIC control steps.")


def _mode_message(mode: TeleopMode) -> str:
    if mode is TeleopMode.OFF:
        return "Control stopped. Press A+B+X+Y to arm it again."
    if mode is TeleopMode.READY:
        return "Control ready. Press A+X to enter full-body POSE teleop."
    return "Full-body POSE teleoperation enabled."


def _finish_recording(recorder: EpisodeRecorder) -> None:
    frames = recorder.frame_count
    duration = recorder.duration_seconds
    path = recorder.finish()
    if path is None:
        print("Recording stopped without any control frames.")
    else:
        print(f"Recording saved to {path}.")
        print(f"Episode summary: {frames} frames, {duration:.2f} seconds.")
        if recorder.last_preview is not None:
            print(f"Contact preview saved to {recorder.last_preview}.")


def _append_recording(
    recorder: EpisodeRecorder,
    env,
    command,
    token,
    action,
    controls: PicoControls,
    video,
    cameras: CameraSampler | None = None,
) -> None:
    video_frame = None
    camera_pose = None
    appearance_camera_pose = None
    camera_frames = cameras.capture() if cameras is not None else None
    if cameras is None and video is not None and recorder.record_video:
        rendered = video.render(
            recording=True,
            elapsed_seconds=recorder.duration_seconds,
        )
        if rendered is None:
            rendered = video.latest_rendered
        if rendered is None:
            raise RuntimeError("video has not produced its first recording frame")
        video_frame = rendered.rgb
        camera_pose = rendered.camera_to_world
        appearance_camera_pose = rendered.appearance_camera_to_world
    recorder.append(
        time=env.time,
        qpos=env.data.qpos,
        qvel=env.data.qvel,
        ctrl=env.data.ctrl,
        command=command,
        token=token,
        action=action,
        controls=controls,
        contacts=env.contacts.last_frame,
        tactile=env.tactile.last_frame,
        tactile_suit=env.tactile_suit,
        video_frame=video_frame,
        camera_pose=camera_pose,
        appearance_camera_pose=appearance_camera_pose,
        camera_frames=camera_frames,
        task_state=env.task_observation() if hasattr(env, "task_observation") else None,
    )


if __name__ == "__main__":
    main()
