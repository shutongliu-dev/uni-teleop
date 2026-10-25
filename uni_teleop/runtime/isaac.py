"""Assemble one immutable source tree before starting any hardware readers."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backends/isaac"


def prepare_snapshot(destination):
    module_path = BACKEND / "scripts/maintenance/freeze_project_sources.py"
    spec = importlib.util.spec_from_file_location("source_freezer", module_path)
    freezer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freezer)
    # Explicit source-only staging keeps local calibration, credentials and data out.
    with tempfile.TemporaryDirectory(prefix="uni-teleop-source-") as temporary:
        stage = Path(temporary)
        for tree in ("bench", "scenes", "agent", "scripts"):
            shutil.copytree(BACKEND / tree, stage / tree,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(ROOT / "uni_teleop", stage / "uni_teleop",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(BACKEND / "pyproject.toml", stage / "pyproject.toml")
        return freezer.freeze(stage, destination,
                              extras=(*freezer.DEFAULT_EXTRAS, "bench/apps/workroom.headless.kit"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("pico", "master"), required=True)
    parser.add_argument("--prepare-only", action="store_true", help="Freeze code without loading devices or Isaac")
    parser.add_argument("--output", type=Path)
    args, backend_args = parser.parse_known_args()
    if backend_args[:1] == ["--"]:
        backend_args = backend_args[1:]
    runs = Path(os.environ.get("DEXBENCH_RUNS", ROOT / "runs/isaac"))
    run = (args.output or runs / f"{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}").resolve()
    run.mkdir(parents=True, exist_ok=False)
    receipt = prepare_snapshot(run / "source")
    (run / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Session source: {run / 'source'}", flush=True)
    if args.prepare_only:
        return
    env = dict(os.environ)
    env.update(PYTHONPATH=str(run / "source"),
               ISAAC_BENCH_EXPERIENCE=str(run / "source/bench/apps/workroom.headless.kit"),
               ISAAC_BENCH_SINGLE_CUDA_DEVICE="1", ISAAC_ALLOW_UNVALIDATED_DRIVER="1")
    entry = "scripts.teleop.master_slave_isaac" if args.source == "master" else env.get(
        "PICO_TELEOP_ENTRYPOINT", "scripts.teleop.pico_isaac")
    command = [str(run / "source/scripts/isaac_python.sh"), "-B", "-m", entry,
               "--output", str(run / "session"), *backend_args]
    reader = None
    native = None
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (run / "pico_sdk.log").open("w") as log:
            if args.source == "pico":
                port = argparse.ArgumentParser(add_help=False)
                port.add_argument("--port", default="5557")
                device = ["--device", env["PICO_DEVICE"]] if env.get("PICO_DEVICE") else []
                reader = subprocess.Popen([
                    sys.executable, "-B", "-m", "uni_teleop.devices.pico.source",
                    "--library", env["PICO_SDK_LIBRARY"],
                    "--port", port.parse_known_args(backend_args)[0].port, *device,
                ], cwd=run / "source", env=env, stdout=log, stderr=subprocess.STDOUT)
            native = subprocess.Popen(command, cwd=run / "source", env=env)
            raise SystemExit(native.wait())
    finally:
        for process in (native, reader):
            if process is None or process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
