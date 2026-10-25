"""Launch the reconstructed-lab renderer and PICO teleoperation together."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zmq

from sonic_mujoco.scan import CameraIntrinsics, ScanRendererClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_NERFSTUDIO_PYTHON = (
    WORKSPACE_ROOT / "nerfstudio" / ".pixi" / "envs" / "default" / "bin" / "python"
)
DEFAULT_RECONSTRUCTION = WORKSPACE_ROOT / "lab-reconstruction"
DEFAULT_CONFIG = (
    DEFAULT_RECONSTRUCTION
    / "outputs"
    / "lab_first_pass"
    / "splatfacto"
    / "2026-08-29"
    / "config.yml"
)
DEFAULT_GEOMETRY = (
    DEFAULT_RECONSTRUCTION / "geometry_from_scan" / "gs_depth" / "lab_scan_points.json"
)
DEFAULT_COLLISION = (
    PROJECT_ROOT
    / "sonic_mujoco"
    / "assets"
    / "reconstruction"
    / "lab_scan_collision.json"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nerfstudio-python",
        type=Path,
        default=DEFAULT_NERFSTUDIO_PYTHON,
    )
    parser.add_argument("--scan-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument(
        "--collision-manifest",
        type=Path,
        default=Path(
            os.environ.get("SONIC_RECONSTRUCTION_COLLISION", DEFAULT_COLLISION)
        ),
        help="local lab collision manifest (never uploaded by this repository)",
    )
    parser.add_argument("--anchor-image", default="frame_00138.jpg")
    parser.add_argument("--renderer-endpoint", default="tcp://127.0.0.1:8765")
    parser.add_argument(
        "--show-physics-viewer",
        action="store_true",
        help="also show the native MuJoCo collision-debug viewer",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the reconstructed-scene renderer, then exit",
    )
    return parser.parse_known_args()


def wait_for_renderer(endpoint: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("scan renderer stopped during startup")
        client = ScanRendererClient(endpoint, timeout_ms=500)
        try:
            info = client.info()
        except (RuntimeError, ValueError, zmq.ZMQError):
            time.sleep(0.25)
        else:
            server_pid = info.get("server_pid")
            if server_pid != process.pid:
                raise RuntimeError(
                    f"renderer endpoint is already owned by PID {server_pid!r}"
                )
            return
        finally:
            client.close()
    raise TimeoutError("scan renderer did not become ready within 60 seconds")


def verify_novel_views(
    endpoint: str, scan_units_per_meter: float
) -> tuple[float, float]:
    client = ScanRendererClient(endpoint)
    try:
        info = client.info()
        anchor = np.asarray(info["anchor_camera_to_world"], dtype=np.float64)
        intrinsics = CameraIntrinsics.from_vertical_fov(640, 360, 60.0)
        eye_frames = []
        for offset_meters in (-0.032, 0.032):
            pose = anchor.copy()
            pose[:3, 3] += pose[:3, 0] * offset_meters * scan_units_per_meter
            eye_frames.append(client.render(pose, intrinsics))
        moved = anchor.copy()
        moved[:3, 3] += moved[:3, 0] * 0.10 * scan_units_per_meter
        moved_frame = client.render(moved, intrinsics)
    finally:
        client.close()

    stereo_difference = float(
        np.mean(
            np.abs(eye_frames[0].astype(np.float32) - eye_frames[1].astype(np.float32))
        )
    )
    motion_difference = float(
        np.mean(
            np.abs(eye_frames[0].astype(np.float32) - moved_frame.astype(np.float32))
        )
    )
    if stereo_difference < 0.5:
        raise RuntimeError(
            "reconstruction check failed: left and right eye views are too similar"
        )
    if motion_difference < 0.5:
        raise RuntimeError(
            "reconstruction check failed: translated camera view is static"
        )
    return stereo_difference, motion_difference


def main() -> None:
    args, teleop_args = parse_args()
    required = (
        args.nerfstudio_python,
        args.scan_config,
        args.geometry_summary,
        args.collision_manifest,
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required reconstructed-lab file not found: {path}")
    renderer_command = [
        str(args.nerfstudio_python),
        str(PROJECT_ROOT / "scripts" / "run_scan_renderer.py"),
        "--config",
        str(args.scan_config),
        "--geometry-summary",
        str(args.geometry_summary),
        "--anchor-image",
        args.anchor_image,
        "--listen",
        args.renderer_endpoint,
    ]
    renderer = subprocess.Popen(renderer_command, cwd=PROJECT_ROOT)
    try:
        print("Loading the reconstructed lab; this can take several seconds")
        wait_for_renderer(args.renderer_endpoint, renderer)
        if args.check:
            collision = json.loads(args.collision_manifest.read_text())
            stereo_difference, motion_difference = verify_novel_views(
                args.renderer_endpoint,
                float(collision["scan_units_per_meter"]),
            )
            print(
                "Reconstructed-lab dynamic-view check passed: "
                f"stereo difference={stereo_difference:.2f}, "
                f"10 cm motion difference={motion_difference:.2f}"
            )
            return
        teleop_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_pico_teleop.py"),
            *teleop_args,
            *([] if args.show_physics_viewer else ["--headless"]),
            "--scene",
            "lab_scan",
            "--scan-collision-manifest",
            str(args.collision_manifest),
            "--scan-renderer-endpoint",
            args.renderer_endpoint,
        ]
        subprocess.run(teleop_command, cwd=PROJECT_ROOT, check=True)
    except KeyboardInterrupt:
        pass
    finally:
        renderer.terminate()
        try:
            renderer.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            renderer.kill()
            renderer.wait()


if __name__ == "__main__":
    main()
