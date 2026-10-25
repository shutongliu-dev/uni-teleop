#!/usr/bin/env bash
set -euo pipefail
backend="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
project="$(cd -- "$backend/../.." && pwd)"
export PYTHONPATH="$project:$backend"
cd "$project"
exec "${DEXBENCH_PYTHON:-$backend/.venv/bin/python}" -B -m uni_teleop.runtime.isaac --source pico -- "$@"
