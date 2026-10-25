import argparse
from pathlib import Path

from sonic_mujoco.simple_bridge import (
    DEFAULT_SIMPLE_ROOT,
    download_simple_scene,
    list_simple_scenes,
    resolve_simple_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or download SIMPLE HSSD scenes"
    )
    parser.add_argument("--simple-root", type=Path, default=DEFAULT_SIMPLE_ROOT)
    parser.add_argument("--download", metavar="SCENE_UID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_simple_root(args.simple_root)
    if args.download:
        raise SystemExit(download_simple_scene(root, args.download))
    scenes = list_simple_scenes(root)
    for scene in scenes:
        marker = "ready" if scene.downloaded else "not downloaded"
        print(f"{scene.uid:<8} {scene.name:<24} {marker}")
    print(f"{sum(scene.downloaded for scene in scenes)}/{len(scenes)} scenes ready")


if __name__ == "__main__":
    main()
