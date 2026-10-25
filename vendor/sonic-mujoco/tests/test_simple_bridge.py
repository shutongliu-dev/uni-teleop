import tempfile
import unittest
from pathlib import Path

from sonic_mujoco.simple_bridge import (
    SIMPLE_TASKS,
    build_simple_render_command,
    build_simple_teleop_command,
    list_simple_scenes,
    resolve_simple_root,
    resolve_simple_task,
)


class SimpleBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "src/simple/resources/hssd-scenes").mkdir(parents=True)
        (self.root / ".venv/bin").mkdir(parents=True)
        (self.root / ".venv/bin/python").touch()
        (self.root / "src/simple/resources/hssd-scenes/config.yaml").write_text(
            """
- uid: scene0
  name: "first_room"
  scale: 1
- uid: scene1
  name: second_room
  scale: 0.01
""".strip()
        )

    def test_resolves_alias_and_registered_id(self) -> None:
        task = resolve_simple_task("open_oven")

        self.assertEqual(task.env_id, "simple/G1WholebodyOpenOvenTeleop-v0")
        self.assertEqual(resolve_simple_task(task.env_id), task)
        self.assertGreaterEqual(len(SIMPLE_TASKS), 13)

    def test_lists_downloaded_hssd_scenes(self) -> None:
        scene_path = self.root / "data/scenes/hssd/first_room"
        scene_path.mkdir(parents=True)
        (scene_path / "first_room.usd").touch()

        scenes = list_simple_scenes(self.root)

        self.assertEqual([scene.uid for scene in scenes], ["scene0", "scene1"])
        self.assertTrue(scenes[0].downloaded)
        self.assertFalse(scenes[1].downloaded)

    def test_builds_isolated_simple_command(self) -> None:
        command = build_simple_teleop_command(
            self.root,
            "push_chair",
            target="graspnet1b:12",
            headless=True,
            record=True,
            max_episode_steps=100,
            num_episodes=2,
            save_dir=self.root / "output",
        )

        self.assertEqual(command[0], str(self.root / ".venv/bin/python"))
        self.assertIn("simple/G1WholebodyPushOfficeChairTeleop-v0", command)
        self.assertIn("--headless", command)
        self.assertIn("--record", command)
        self.assertIn("graspnet1b:12", command)

    def test_builds_isaac_replay_command(self) -> None:
        command = build_simple_render_command(
            self.root,
            "bend_pick",
            self.root / "recording",
            headless=True,
            webrtc=False,
            record=True,
            num_episodes=1,
        )

        self.assertIn("simple.cli.render_decoupled_wbc", command)
        self.assertIn("mujoco_isaac", command)
        self.assertIn("--headless", command)
        self.assertIn("--no-webrtc", command)
        self.assertIn("--record", command)

    def test_builds_live_rich_scene_command(self) -> None:
        command = build_simple_teleop_command(
            self.root,
            "hug_container",
            rich_scene=True,
            webrtc=False,
        )

        self.assertTrue(command[1].endswith("run_simple_rich_teleop.py"))
        self.assertIn("--no-webrtc", command)
        self.assertNotIn("--record", command)

        with self.assertRaisesRegex(ValueError, "does not support recording"):
            build_simple_teleop_command(
                self.root,
                "hug_container",
                rich_scene=True,
                record=True,
            )

    def test_rejects_checkout_without_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "ready .venv"):
            resolve_simple_root(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
