import argparse
import shutil
import subprocess
from pathlib import Path

SUPERDEX_REPOSITORY = "https://github.com/facebookresearch/project_superdex.git"


def _run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the isolated Python 3.12 Project SuperDex runtime"
    )
    parser.add_argument(
        "--python",
        default="3.12",
        help="Python version or executable used by the SuperDex environment",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    environment = project_root / ".venv-superdex"
    dependency_root = project_root / ".superdex/project_superdex"
    uv = shutil.which("uv")
    git = shutil.which("git")
    if uv is None or git is None:
        raise SystemExit("setup requires both uv and git on PATH")

    if not environment.exists():
        _run(uv, "venv", "--python", args.python, str(environment))
    python = environment / "bin/python"
    _run(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "superdex-physics==1.0.0",
        "superdex-robotics==1.0.0",
        "superdex-physics-debugger==1.0.0",
        "pyzmq>=25,<28",
    )
    _run(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--no-deps",
        "--editable",
        str(project_root),
    )

    if not dependency_root.exists():
        dependency_root.parent.mkdir(parents=True, exist_ok=True)
        _run(
            git,
            "clone",
            "--branch",
            "stable",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            SUPERDEX_REPOSITORY,
            str(dependency_root),
        )
    if not (dependency_root / ".git").exists():
        raise SystemExit(
            f"SuperDex dependency path is not a git checkout: {dependency_root}"
        )
    _run(
        git,
        "-C",
        str(dependency_root),
        "sparse-checkout",
        "set",
        "assets/bots/hands/dg5f_long",
        "--skip-checks",
    )

    print(f"SuperDex Python: {python}")
    print(f"SuperDex assets: {dependency_root / 'assets'}")
    print(
        "Run the validation with: "
        ".venv-superdex/bin/python scripts/run_superdex_hands.py "
        "--headless --demo --steps 400"
    )


if __name__ == "__main__":
    main()
