#!/usr/bin/env bash
# Project entry point; Isaac has its own bundled runtime, never the CPU venv.
# The launcher comes from ISAAC_PYTHON_LAUNCHER, else this checkout's
# runtime/isaacsim link (see runtime/README.md).
set -euo pipefail
isaac_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_launcher="${ISAAC_PYTHON_LAUNCHER:-}"
if [[ -z "$isaac_launcher" ]]; then
    for candidate in python_compat.sh python.sh; do
        if [[ -x "$isaac_project_root/runtime/isaacsim/$candidate" ]]; then
            isaac_launcher="$isaac_project_root/runtime/isaacsim/$candidate"
            break
        fi
    done
fi
if [[ -z "$isaac_launcher" ]]; then
    printf 'No Isaac Sim launcher: run runtime/setup.sh --isaacsim <dir> or set ISAAC_PYTHON_LAUNCHER\n' >&2
    exit 3
fi
export ISAAC_BENCH_PYTHONPATH="$isaac_project_root${ISAAC_BENCH_PYTHONPATH:+:$ISAAC_BENCH_PYTHONPATH}"
exec "$isaac_launcher" "$@"
