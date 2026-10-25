import argparse
from pathlib import Path

from sonic_mujoco.simple_bridge import (
    DEFAULT_SIMPLE_ROOT,
    build_simple_render_command,
    resolve_simple_root,
    resolve_simple_task,
    run_in_simple,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Replay a SIMPLE recording in its rich HSSD/Isaac scene"
    )
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--task", default="bend_pick")
    parser.add_argument("--simple-root", type=Path, default=DEFAULT_SIMPLE_ROOT)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--webrtc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--render-hz", type=int, default=30)
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--dr-level", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    root = resolve_simple_root(args.simple_root)
    selected = resolve_simple_task(args.task)
    command = build_simple_render_command(
        root,
        selected.env_id,
        args.data_dir,
        headless=args.headless,
        webrtc=args.webrtc,
        record=args.record,
        render_hz=args.render_hz,
        num_episodes=args.num_episodes,
        dr_level=args.dr_level,
        save_dir=args.save_dir,
        extra_args=extra_args,
    )
    print(f"Replaying SIMPLE task: {selected.alias} ({selected.env_id})")
    print(f"Dataset: {args.data_dir.expanduser().resolve()}")
    if args.dry_run:
        print(" ".join(command))
        return
    raise SystemExit(run_in_simple(root, command))


if __name__ == "__main__":
    main()
