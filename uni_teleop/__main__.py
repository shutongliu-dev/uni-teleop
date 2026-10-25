"""Run from a checkout: python -m uni_teleop list|doctor|run."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex

from .catalog import ROOT, configure, load_profile, recipes, select
VARIABLE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def expand(value, settings):
    def replace(match):
        name = match[1]
        if not settings.get(name):
            raise ValueError(f"Missing {name}: set it in --config or the environment")
        return str(settings[name])
    return VARIABLE.sub(replace, value)


def plan(name, config=None, extra=()):
    saved_args = []
    session_path = Path(name)
    if session_path.is_file():
        session = json.loads(session_path.read_text())
        name = session["recipe"]
        saved_args = session.get("arguments", [])
        if not isinstance(saved_args, list) or any(not isinstance(v, str) for v in saved_args):
            raise ValueError("Session arguments must be a list of strings")
        session_settings = session["settings"]
    else:
        session_settings = {}
    profile = load_profile(name)
    settings = dict(os.environ)
    settings.update({k: str(v) for k, v in session_settings.items()})
    if config:
        values = json.loads(Path(config).read_text())
        if not isinstance(values, dict) or any(not isinstance(v, (str, int)) for v in values.values()):
            raise ValueError("Config must be an object of string/integer settings")
        settings.update({k: str(v) for k, v in values.items()})
    settings["ROOT"] = str(ROOT)
    cwd = expand(profile["cwd"], settings)
    command = [expand(v, settings) for v in profile["command"]] + saved_args + list(extra)
    environment = dict(os.environ)
    environment.pop("PICO_TELEOP_ENTRYPOINT", None)
    environment.update({k: expand(v, settings) for k, v in profile.get("env", {}).items()})
    environment["UNI_TELEOP_ROOT"] = str(ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # A previous Isaac snapshot must never override the selected source tree.
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), cwd, str(ROOT / "backends/isaac"), str(ROOT / "backends/mujoco")))
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
    listing = sub.add_parser("list", help="Find supported device combinations")
    setup = sub.add_parser("configure", help="Generate only the settings needed for your devices")
    setup.add_argument("profile", nargs="?")
    setup.add_argument("--output", type=Path, default=ROOT / "local/session.json")
    for command in (listing, setup):
        for key in ("backend", "arm", "hand", "robot"):
            command.add_argument("--" + key)
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
            matches = select(backend=args.backend, arm=args.arm, hand=args.hand, robot=args.robot)
            for name, recipe in matches.items():
                print(f"{name:28} {recipe['description']} [{recipe['status']}]")
            if not matches:
                print("No implemented route matches those devices. Use 'list' to see available combinations.")
            return
        if args.action == "configure":
            matches = select(backend=args.backend, arm=args.arm, hand=args.hand, robot=args.robot)
            name = args.profile
            if name is None:
                if len(matches) != 1:
                    raise ValueError("Select exactly one combination or name a recipe. Matches: " + ", ".join(matches))
                name = next(iter(matches))
            configure(name, args.output)
            print(f"Created {args.output}. Fill in its settings, then:\n"
                  f"  ./teleop doctor {args.output}\n  ./teleop run {args.output}")
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
