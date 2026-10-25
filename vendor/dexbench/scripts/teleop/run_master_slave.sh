#!/usr/bin/env bash
# Read-only FACTR acquisition and simulated follower; no PICO SDK or real robot RPC.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DEXBENCH_DATA="${DEXBENCH_DATA:-$root/data}"
export ISAAC_PYTHON_LAUNCHER="${ISAAC_PYTHON_LAUNCHER:-$root/runtime/isaacsim/python_compat.sh}"
export ISAAC_BENCH_SINGLE_CUDA_DEVICE=1
export ISAAC_ALLOW_UNVALIDATED_DRIVER=1
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPU UUID}"
run="${DEXBENCH_RUNS:-$root/../dexbench_runs}/master_slave_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
run="$(realpath "$run")"
"${DEXBENCH_PYTHON:-$root/.venv/bin/python}" -B -m scripts.maintenance.freeze_project_sources create --source "$root" \
    --output "$run/source" --extra bench/apps/workroom.headless.kit > "$run/source_receipt.json"
export PYTHONPATH="$run/source"
export ISAAC_BENCH_EXPERIENCE="$run/source/bench/apps/workroom.headless.kit"
cd "$run/source"
exec scripts/isaac_python.sh -B -m scripts.teleop.master_slave_isaac --output "$run/session" "$@"
