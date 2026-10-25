import argparse
import os
from pathlib import Path

from sonic_mujoco.superdex import DEFAULT_HAND_ENDPOINT
from sonic_mujoco.superdex.runtime import SuperDexHandsConfig, run_superdex_hands

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive SuperDex DG5F hands from SONIC/PICO hand targets"
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SUPERDEX_ASSETS_PATH",
                PROJECT_ROOT / ".superdex/project_superdex/assets",
            )
        ),
    )
    parser.add_argument("--endpoint", default=DEFAULT_HAND_ENDPOINT)
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="animate a grasp when no PICO publisher is connected",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sides = ("left", "right") if args.side == "both" else (args.side,)
    config = SuperDexHandsConfig(
        assets_root=args.assets_root,
        endpoint=args.endpoint,
        sides=sides,
    )
    completed = run_superdex_hands(
        config,
        headless=args.headless,
        steps=args.steps,
        demo=args.demo,
    )
    print(f"Stopped after {completed} SuperDex physics steps.")


if __name__ == "__main__":
    main()
