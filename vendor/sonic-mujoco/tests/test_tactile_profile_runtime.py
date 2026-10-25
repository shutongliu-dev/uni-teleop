import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import run_gr00t_sweep
from scripts.run_gr00t_sweep import parse_args as parse_gr00t_args
from scripts.run_lab_scan_teleop import parse_args as parse_lab_scan_args
from scripts.run_pico_teleop import (
    ENVIRONMENTS,
    _create_environment,
    _create_recorder,
)
from scripts.run_pico_teleop import (
    parse_args as parse_pico_args,
)
from sonic_mujoco.envs.mujoco.g1 import (
    MujocoG1EmptyEnv,
    MujocoG1LabScanEnv,
    MujocoG1SweepEnv,
)
from sonic_mujoco.tactile_calibration import (
    load_tactile_calibration_profile,
    make_legacy_tactile_profile,
)
from sonic_mujoco.tactile_skin import TACTILE_DEVICE_NAMES


def write_profile(path: Path, *, layout_sha256: str | None = None) -> str:
    profile = make_legacy_tactile_profile(
        len(TACTILE_DEVICE_NAMES),
        device_names=TACTILE_DEVICE_NAMES,
        profile_id="runtime-test-v1",
        layout_sha256=layout_sha256,
        gain=7.0,
        fixed_taxel_gain_variation=0.0,
        fixed_device_rate_variation=0.0,
    )
    path.write_text(json.dumps(profile.as_dict(), indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TactileProfileRuntimeTest(unittest.TestCase):
    def test_repository_profile_matches_current_g1_skin(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profiles = tuple(
            load_tactile_calibration_profile(
                root
                / "configs"
                / "tactile"
                / f"juqiao_g1_sim2real_provisional_v{version}.json"
            )
            for version in (1, 2)
        )
        scenes = (
            *ENVIRONMENTS.items(),
            (
                "lab_scan",
                lambda: MujocoG1LabScanEnv(collision_geometry="boxes"),
            ),
        )
        for profile in profiles:
            for scene, create_env in scenes:
                with self.subTest(profile=profile.profile_id, scene=scene):
                    env = create_env()
                    try:
                        env.configure_tactile_profile(profile, seed=0)
                        self.assertEqual(
                            profile.layout_sha256,
                            env.tactile_skin_layout.sha256,
                        )
                        self.assertEqual(
                            env.tactile_adapter.metadata["calibration_status"],
                            "provisional_unpaired_force_mapping",
                        )
                        if profile.schema_version == 2:
                            self.assertEqual(
                                env.tactile_adapter.metadata["sensor_dynamics"][
                                    "status"
                                ],
                                "provisional_unpaired_sensor_dynamics",
                            )
                    finally:
                        env.close()

    def test_runner_parsers_share_one_tactile_profile_flag(self) -> None:
        profile = Path("calibration/profile.json")
        with patch.object(
            sys,
            "argv",
            [
                "run_pico_teleop.py",
                "--tactile-profile",
                str(profile),
                "--tactile-seed",
                "23",
            ],
        ):
            pico = parse_pico_args()
            self.assertEqual(pico.tactile_profile, profile)
            self.assertEqual(pico.tactile_seed, 23)
        with patch.object(
            sys,
            "argv",
            ["run_gr00t_sweep.py", "--tactile-profile", str(profile)],
        ):
            self.assertEqual(parse_gr00t_args().tactile_profile, profile)
        with patch.object(
            sys,
            "argv",
            ["run_lab_scan_teleop.py", "--tactile-profile", str(profile)],
        ):
            _, teleop_args = parse_lab_scan_args()
            self.assertEqual(teleop_args, ["--tactile-profile", str(profile)])

    def test_scene_seed_reproduces_tactile_episode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = load_tactile_calibration_profile(
            root / "configs" / "tactile" / "juqiao_g1_sim2real_provisional_v2.json"
        )
        env = MujocoG1SweepEnv()
        self.addCleanup(env.close)
        env.configure_tactile_profile(profile, seed=42)

        env.reset(seed=42)
        first = env.tactile_adapter.metadata["episode_profile"]
        env.reset(seed=42)
        second = env.tactile_adapter.metadata["episode_profile"]

        self.assertEqual(first["seed"], second["seed"])
        self.assertEqual(first["sha256"], second["sha256"])

    def test_pico_environment_passes_sanitized_profile_metadata_to_recorder(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = MujocoG1EmptyEnv()
            try:
                layout_sha256 = probe.tactile_skin_layout.sha256
            finally:
                probe.close()
            profile_path = root / "private-profile.json"
            file_sha256 = write_profile(
                profile_path,
                layout_sha256=layout_sha256,
            )
            args = SimpleNamespace(
                scene="empty",
                tactile_profile=profile_path,
                tactile_seed=0,
                scan_collision="auto",
                scan_collision_manifest=None,
                record_dir=root / "records",
                no_record_video=True,
                task=None,
            )

            env = _create_environment(args)
            try:
                env.reset()
                recorder = _create_recorder(
                    args,
                    env,
                    SimpleNamespace(steps_per_action=4),
                    None,
                )
            finally:
                env.close()

            metadata = recorder.tactile_metadata
            self.assertEqual(metadata["profile_id"], "runtime-test-v1")
            self.assertEqual(metadata["profile_file_sha256"], file_sha256)
            self.assertEqual(metadata["layout_sha256"], layout_sha256)
            self.assertEqual(metadata["source_time"]["clock"], "mujoco_sim_time")
            self.assertNotIn(str(root), json.dumps(metadata))

    def test_gr00t_result_records_profile_hash_and_uses_run_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "operator-private-profile.json"
            file_sha256 = write_profile(profile_path)
            args = SimpleNamespace(
                host="localhost",
                port=5550,
                decoder=root / "unused.onnx",
                prompt="test prompt",
                seconds=0.0,
                warmup_seconds=0.0,
                action_steps=20,
                seed=17,
                video_fps=10,
                no_video=True,
                tactile_profile=profile_path,
                output_dir=root / "results",
            )
            controller = MagicMock()
            camera = MagicMock()
            camera.render.return_value = {
                "ego_view_left": np.zeros((4, 4, 3), dtype=np.uint8),
                "ego_view_right": np.zeros((4, 4, 3), dtype=np.uint8),
            }
            policy = MagicMock()
            policy.ping.return_value = True

            with (
                patch.object(run_gr00t_sweep, "parse_args", return_value=args),
                patch.object(
                    run_gr00t_sweep.SonicController,
                    "from_onnx",
                    return_value=controller,
                ),
                patch.object(run_gr00t_sweep, "StereoCamera", return_value=camera),
                patch.object(run_gr00t_sweep, "Gr00tClient", return_value=policy),
                patch.object(run_gr00t_sweep, "save_image"),
                patch("builtins.print"),
            ):
                run_gr00t_sweep.main()

            result_path = next((root / "results").glob("*/result.json"))
            encoded = result_path.read_text()
            result = json.loads(encoded)
            self.assertEqual(result["tactile"]["profile_id"], "runtime-test-v1")
            self.assertEqual(
                result["tactile"]["profile_file_sha256"],
                file_sha256,
            )
            self.assertEqual(result["tactile"]["random_seed"], 17)
            self.assertNotIn(str(root), encoded)


if __name__ == "__main__":
    unittest.main()
