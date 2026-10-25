import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uni_teleop.__main__ import ROOT, expand, load_profile, missing_paths, plan


class LauncherTests(unittest.TestCase):
    def config(self, directory):
        path = Path(directory) / "host.json"
        values = json.loads((ROOT / "docs/host.example.json").read_text())
        values = {k: "path with spaces/" + k for k in values}
        path.write_text(json.dumps(values))
        return path

    def test_every_profile_builds_without_importing_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            for profile in (ROOT / "profiles").glob("*.json"):
                with self.subTest(profile=profile.stem):
                    command, cwd, env, checks = plan(profile.stem, self.config(tmp))
                    self.assertTrue(Path(cwd).is_dir())
                    self.assertNotIn("${", " ".join(command))
                    self.assertNotIn("--leader-feedback", command)
                    self.assertNotIn("--start-active", command)
                    self.assertTrue(checks)

    def test_pico_manus_selects_combined_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd, _, env, _ = plan("isaac-pico-manus", self.config(tmp), ["--task", "parts_sorting"])
            self.assertEqual(env["PICO_TELEOP_ENTRYPOINT"], "scripts.teleop.master_slave_isaac")
            self.assertEqual(cmd[-2:], ["--task", "parts_sorting"])
            self.assertIn("path with spaces/MANUS_CALIBRATION", cmd)

    def test_snapshot_environment_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"PYTHONPATH": "/old", "ISAAC_BENCH_EXPERIENCE": "/old.kit"}):
            _, cwd, env, _ = plan("isaac-factr", self.config(tmp))
            self.assertEqual(env["PYTHONPATH"], cwd)
            self.assertNotIn("ISAAC_BENCH_EXPERIENCE", env)

    def test_required_setting_error(self):
        with self.assertRaisesRegex(ValueError, "Missing DEVICE"):
            expand("${DEVICE}", {})

    def test_no_shell_interpolation(self):
        self.assertEqual(expand("${DEVICE}", {"DEVICE": "$(touch /tmp/no)"}), "$(touch /tmp/no)")

    def test_invalid_profile(self):
        with self.assertRaises(ValueError):
            load_profile("../local/host")

    def test_checks_actual_path_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "file"; p.touch()
            self.assertEqual(missing_paths([(tmp, "dir"), (str(p), "file")]), [])
            self.assertEqual(len(missing_paths([(str(p), "dir")])), 1)


if __name__ == "__main__":
    unittest.main()
