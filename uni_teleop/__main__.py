"""Run from a checkout: python -m uni_teleop list|doctor|run."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[1]
VARIABLE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def expand(value, settings):
    def replace(match):
        name = match[1]
        if not settings.get(name):
            raise ValueError(f"Missing {name}: set it in --config or the environment")
        return str(settings[name])
    return VARIABLE.sub(replace, value)


def load_profile(name):
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError("Invalid profile name")
    path = ROOT / "profiles" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown profile: {name}; use 'list'")
    return json.loads(path.read_text())


def plan(name, config=None, extra=()):
    profile = load_profile(name)
    settings = dict(os.environ)
    if config:
        values = json.loads(Path(config).read_text())
        if not isinstance(values, dict) or any(not isinstance(v, (str, int)) for v in values.values()):
            raise ValueError("Config must be an object of string/integer settings")
        settings.update({k: str(v) for k, v in values.items()})
    settings["ROOT"] = str(ROOT)
    cwd = expand(profile["cwd"], settings)
    command = [expand(v, settings) for v in profile["command"]] + list(extra)
    environment = dict(os.environ)
    environment.pop("PICO_TELEOP_ENTRYPOINT", None)
    environment.update({k: expand(v, settings) for k, v in profile.get("env", {}).items()})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # A previous Isaac snapshot must never override the selected source tree.
    environment["PYTHONPATH"] = cwd
    environment.pop("ISAAC_BENCH_EXPERIENCE", None)
    checks = [(expand(v, settings), kind) for v, kind in profile.get("checks", [])]
    return command, cwd, environment, checks


def missing_paths(checks):
    failures = []
    for value, kind in checks:
        path = Path(value)
        valid = {"file": path.is_file, "dir": path.is_dir,
                 "exists": path.exists, "executable": lambda: path.is_file() and os.access(path, os.X_OK)}[kind]()
        if not valid:
            failures.append(f"{kind}: {value}")
    return failures


def main():
    parser = argparse.ArgumentParser(description="uni-teleop: Isaac / MuJoCo teleoperation")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List the integrated control routes")
    for action in ("run", "doctor"):
        p = sub.add_parser(action)
        p.add_argument("profile")
        p.add_argument("--config", type=Path, help="Ignored local JSON containing installation paths")
        if action == "run":
            p.add_argument("--dry-run", action="store_true", help="Print argv without loading SDKs or hardware")
    import sys
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1:]
    try:
        if args.action == "list":
            for path in sorted((ROOT / "profiles").glob("*.json")):
                profile = json.loads(path.read_text())
                print(f"{path.stem:28} {profile['description']}")
            return
        command, cwd, environment, checks = plan(args.profile, args.config, extra)
        if args.action == "run" and args.dry_run:
            print(f"cd {shlex.quote(cwd)}\n{shlex.join(command)}")
            return
        failures = missing_paths(checks)
        if failures:
            raise ValueError("Missing prerequisites:\n  " + "\n  ".join(failures))
        if args.action == "doctor":
            print("Configured paths exist. Device pairing, license and live data are not tested.")
            return
        os.chdir(cwd)
        os.execvpe(command[0], command, environment)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"uni-teleop: {error}\n")


if __name__ == "__main__":
    main()
