import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the XR SDK into this project")
    parser.add_argument(
        "source",
        type=Path,
        help="directory containing xrobotoolkit_sdk*.so and lib/libPXREARobotSDK.so",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    destination = root / ".xrobotoolkit"
    source = args.source.expanduser().resolve()
    bindings = list(source.glob("xrobotoolkit_sdk*.so"))
    library = source / "lib/libPXREARobotSDK.so"
    if len(bindings) != 1 or not library.is_file():
        raise SystemExit(f"XR SDK files not found under {source}")

    (destination / "lib").mkdir(parents=True, exist_ok=True)
    shutil.copy2(bindings[0], destination / bindings[0].name)
    shutil.copy2(library, destination / "lib/libPXREARobotSDK.so")
    print(f"XR SDK copied to {destination}")


if __name__ == "__main__":
    main()
