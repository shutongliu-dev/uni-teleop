import argparse
from pathlib import Path

from sonic_mujoco.simple_bridge import (
    DEFAULT_SIMPLE_ROOT,
    SIMPLE_TASKS,
    build_simple_teleop_command,
    list_simple_scenes,
    resolve_simple_root,
    resolve_simple_task,
    run_in_simple,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run SIMPLE G1 task scenes from the sonic-mujoco workspace"
    )
    parser.add_argument("task", nargs="?", default="bend_pick")
    parser.add_argument("--simple-root", type=Path, default=DEFAULT_SIMPLE_ROOT)
    parser.add_argument("--target", default="graspnet1b:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--rich-scene",
        action="store_true",
        help="experimentally render the task live in its HSSD/Isaac room",
    )
    parser.add_argument(
        "--webrtc",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=30_000)
    parser.add_argument("--render-hz", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--dr-level", type=int, default=0)
    parser.add_argument("--success-criteria", type=float, default=2.0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def print_catalog(root: Path) -> None:
    print("SIMPLE G1 teleoperation tasks:")
    for task in SIMPLE_TASKS:
        print(f"  {task.alias:<20} {task.description}")
    scenes = list_simple_scenes(root)
    downloaded = sum(scene.downloaded for scene in scenes)
    print(f"\nHSSD rich scenes: {len(scenes)} catalogued, {downloaded} downloaded")
    for scene in scenes:
        state = "ready" if scene.downloaded else "download on demand"
        print(f"  {scene.uid:<8} {scene.name:<24} {state}")


def main() -> None:
    args, extra_args = parse_args()
    root = resolve_simple_root(args.simple_root)
    if args.list:
        print_catalog(root)
        return
    selected = resolve_simple_task(args.task)
    if args.rich_scene and args.record:
        raise ValueError("--record is not available with --rich-scene yet")
    command = build_simple_teleop_command(
        root,
        selected.env_id,
        target=args.target,
        headless=args.headless,
        record=args.record,
        max_episode_steps=args.max_episode_steps,
        render_hz=args.render_hz,
        num_episodes=args.num_episodes,
        dr_level=args.dr_level,
        success_criteria=args.success_criteria,
        save_dir=args.save_dir,
        rich_scene=args.rich_scene,
        webrtc=args.webrtc,
        extra_args=extra_args,
    )
    mode = "live HSSD/Isaac" if args.rich_scene else "MuJoCo"
    print(f"Launching SIMPLE task: {selected.alias} ({selected.env_id}, {mode})")
    print(f"SIMPLE checkout: {root}")
    if args.dry_run:
        print(" ".join(command))
        return
    raise SystemExit(run_in_simple(root, command))


if __name__ == "__main__":
    main()
