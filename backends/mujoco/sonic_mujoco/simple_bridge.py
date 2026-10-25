"""Optional integration with an external SIMPLE checkout."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMPLE_ROOT = PROJECT_ROOT.parent / "SIMPLE"


@dataclass(frozen=True, slots=True)
class SimpleTask:
    alias: str
    env_id: str
    description: str


@dataclass(frozen=True, slots=True)
class SimpleScene:
    uid: str
    name: str
    downloaded: bool


SIMPLE_TASKS = (
    SimpleTask("bend_pick", "simple/G1WholebodyBendPickTeleop-v0", "bend and pick"),
    SimpleTask(
        "bend_pick_place",
        "simple/G1WholebodyBendPickAndPlaceTeleop-v0",
        "bend, pick, and place",
    ),
    SimpleTask(
        "bend_handover",
        "simple/G1WholebodyBendHandoverTeleop-v0",
        "bend and hand over an object",
    ),
    SimpleTask(
        "push_chair",
        "simple/G1WholebodyPushOfficeChairTeleop-v0",
        "push an office chair",
    ),
    SimpleTask(
        "open_faucet",
        "simple/G1WholebodyOpenFaucetTeleop-v0",
        "open a faucet",
    ),
    SimpleTask("open_oven", "simple/G1WholebodyOpenOvenTeleop-v0", "open an oven"),
    SimpleTask("close_door", "simple/G1WholebodyCloseDoorTeleop-v0", "close a door"),
    SimpleTask(
        "open_trash_can",
        "simple/G1WholebodyOpenTrashCanTeleop-v0",
        "open a trash can",
    ),
    SimpleTask(
        "xmove_pick",
        "simple/G1WholebodyXMovePickTeleop-v0",
        "move sideways and pick",
    ),
    SimpleTask(
        "xmove_bend_pick",
        "simple/G1WholebodyXMoveBendPickTeleop-v0",
        "move sideways, bend, and pick",
    ),
    SimpleTask(
        "between_tables",
        "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
        "walk and transfer between tables",
    ),
    SimpleTask(
        "hug_container",
        "simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0",
        "pick, place, and hug a container",
    ),
    SimpleTask(
        "handover",
        "simple/G1WholebodyHandoverTeleop-v0",
        "whole-body handover",
    ),
    SimpleTask(
        "pick_apple",
        "simple/G1WholebodyPickAndPlaceAppleTeleop-v0",
        "pick and place an apple",
    ),
    SimpleTask(
        "mobile_cheezit",
        "simple/G1WholebodyMobilePickAndPlaceCheezitTeleop-v0",
        "mobile pick and place",
    ),
)

_TASK_BY_ALIAS = {task.alias: task for task in SIMPLE_TASKS}
_TASK_BY_ID = {task.env_id: task for task in SIMPLE_TASKS}
_SCENE_UID = re.compile(r"^\s*-\s+uid:\s*([^#\s]+)")
_SCENE_NAME = re.compile(r'^\s+name:\s*["\']?([^"\'#\s]+)')


def resolve_simple_root(path: str | Path | None = None) -> Path:
    configured = path or os.environ.get("SIMPLE_ROOT") or DEFAULT_SIMPLE_ROOT
    root = Path(configured).expanduser().resolve()
    required = (root / "src/simple", root / ".venv/bin/python")
    if not root.is_dir() or not all(item.exists() for item in required):
        raise ValueError(f"SIMPLE checkout with a ready .venv was not found at {root}")
    return root


def resolve_simple_task(name: str) -> SimpleTask:
    if name in _TASK_BY_ALIAS:
        return _TASK_BY_ALIAS[name]
    if name in _TASK_BY_ID:
        return _TASK_BY_ID[name]
    if name.startswith("simple/"):
        return SimpleTask(name, name, "custom registered SIMPLE task")
    aliases = ", ".join(task.alias for task in SIMPLE_TASKS)
    raise ValueError(f"unknown SIMPLE task {name!r}; choose one of: {aliases}")


def list_simple_scenes(root: str | Path | None = None) -> tuple[SimpleScene, ...]:
    simple_root = resolve_simple_root(root)
    config = simple_root / "src/simple/resources/hssd-scenes/config.yaml"
    scenes: list[tuple[str, str]] = []
    uid: str | None = None
    for line in config.read_text().splitlines():
        uid_match = _SCENE_UID.match(line)
        if uid_match:
            uid = uid_match.group(1)
            continue
        name_match = _SCENE_NAME.match(line)
        if uid is not None and name_match:
            scenes.append((uid, name_match.group(1)))
            uid = None
    data_root = simple_root / "data/scenes/hssd"
    return tuple(
        SimpleScene(
            uid=scene_uid,
            name=name,
            downloaded=(data_root / name / f"{name}.usd").is_file(),
        )
        for scene_uid, name in scenes
    )


def build_simple_teleop_command(
    root: str | Path,
    task: str,
    *,
    target: str = "graspnet1b:0",
    headless: bool = False,
    record: bool = False,
    max_episode_steps: int = 30_000,
    render_hz: int = 50,
    num_episodes: int = 100,
    dr_level: int = 0,
    success_criteria: float = 2.0,
    save_dir: str | Path | None = None,
    rich_scene: bool = False,
    webrtc: bool = False,
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    simple_root = resolve_simple_root(root)
    selected = resolve_simple_task(task)
    if max_episode_steps < 1 or render_hz < 1 or num_episodes < 1:
        raise ValueError(
            "episode steps, render rate, and episode count must be positive"
        )
    if dr_level < 0:
        raise ValueError("dr_level must be non-negative")
    output = Path(save_dir or PROJECT_ROOT / "records/simple").expanduser().resolve()
    command = [str(simple_root / ".venv/bin/python")]
    if rich_scene:
        if record:
            raise ValueError("rich-scene teleoperation does not support recording yet")
        command.extend(
            [
                str(PROJECT_ROOT / "scripts/run_simple_rich_teleop.py"),
                selected.env_id,
                "--webrtc" if webrtc else "--no-webrtc",
            ]
        )
    else:
        command.extend(
            [
                "-m",
                "simple.cli.teleop_decoupled_wbc",
                selected.env_id,
            ]
        )
    command.extend(
        [
            "--target",
            target,
            "--headless" if headless else "--no-headless",
            "--max-episode-steps",
            str(max_episode_steps),
            "--render-hz",
            str(render_hz),
            "--num-episodes",
            str(num_episodes),
            "--dr-level",
            str(dr_level),
            "--success-criteria",
            str(success_criteria),
            "--save-dir",
            str(output),
        ]
    )
    if not rich_scene:
        command[4:4] = ["--sim-mode", "mujoco"]
    if record:
        command.append("--record")
    command.extend(extra_args)
    return tuple(command)


def build_simple_render_command(
    root: str | Path,
    task: str,
    data_dir: str | Path,
    *,
    headless: bool = False,
    webrtc: bool = True,
    record: bool = False,
    render_hz: int = 30,
    num_episodes: int = -1,
    dr_level: int = 0,
    save_dir: str | Path | None = None,
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build SIMPLE's official MuJoCo-to-Isaac replay command."""
    simple_root = resolve_simple_root(root)
    selected = resolve_simple_task(task)
    if render_hz < 1:
        raise ValueError("render rate must be positive")
    if num_episodes == 0 or num_episodes < -1:
        raise ValueError("episode count must be -1 or positive")
    if dr_level < 0:
        raise ValueError("dr_level must be non-negative")
    source = Path(data_dir).expanduser().resolve()
    output = (
        Path(save_dir or PROJECT_ROOT / "records/simple_isaac").expanduser().resolve()
    )
    command = [
        str(simple_root / ".venv/bin/python"),
        "-m",
        "simple.cli.render_decoupled_wbc",
        selected.env_id,
        "--data-dir",
        str(source),
        "--sim-mode",
        "mujoco_isaac",
        "--headless" if headless else "--no-headless",
        "--webrtc" if webrtc else "--no-webrtc",
        "--render-hz",
        str(render_hz),
        "--num-episodes",
        str(num_episodes),
        "--dr-level",
        str(dr_level),
        "--save-dir",
        str(output),
    ]
    if record:
        command.append("--record")
    command.extend(extra_args)
    return tuple(command)


def run_in_simple(
    root: str | Path,
    command: Sequence[str],
) -> int:
    simple_root = resolve_simple_root(root)
    environment = os.environ.copy()
    environment.setdefault("SIMPLE_AUTO_BOOTSTRAP", "0")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    try:
        completed = subprocess.run(
            command,
            cwd=simple_root,
            env=environment,
            check=False,
        )
    except KeyboardInterrupt:
        return 130
    return completed.returncode


def download_simple_scene(root: str | Path, uid: str) -> int:
    simple_root = resolve_simple_root(root)
    scenes = {scene.uid: scene for scene in list_simple_scenes(simple_root)}
    if uid not in scenes:
        raise ValueError(f"unknown SIMPLE HSSD scene: {uid}")
    worker = (
        "import sys; from simple.scenes import SceneManager; "
        "scene = SceneManager.get('hssd').load(sys.argv[1]); "
        "print(f'Downloaded {scene.uid}: {scene.name}')"
    )
    command = (str(simple_root / ".venv/bin/python"), "-c", worker, uid)
    return run_in_simple(simple_root, command)
