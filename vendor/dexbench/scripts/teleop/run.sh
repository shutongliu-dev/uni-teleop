#!/usr/bin/env bash
# Freeze sources, isolate XR SDK acquisition, then enter the native Isaac loop.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${DEXBENCH_PYTHON:-$root/.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DEXBENCH_DATA="${DEXBENCH_DATA:-$root/data}"
if [[ -z "${ISAAC_PYTHON_LAUNCHER:-}" ]]; then
  for candidate in python_compat.sh python.sh; do
    if [[ -x "$root/runtime/isaacsim/$candidate" ]]; then
      export ISAAC_PYTHON_LAUNCHER="$root/runtime/isaacsim/$candidate"
      break
    fi
  done
fi
export ISAAC_BENCH_SINGLE_CUDA_DEVICE=1
export ISAAC_ALLOW_UNVALIDATED_DRIVER=1
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPU UUID}"
library="${PICO_SDK_LIBRARY:-$root/runtime/pico/libPXREARobotSDK.so}"
[[ -f "$library" ]] || { echo 'Set PICO_SDK_LIBRARY to libPXREARobotSDK.so' >&2; exit 1; }
library="$(realpath "$library")"
port=5557
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --port) port="${args[i+1]:?--port needs a value}" ;;
    --port=*) port="${args[i]#*=}" ;;
  esac
done
device_args=()
[[ -z "${PICO_DEVICE:-}" ]] || device_args=(--device "$PICO_DEVICE")
run="${DEXBENCH_RUNS:-$root/../dexbench_runs}/pico_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run"
run="$(realpath "$run")"
cd "$root"
"$python" -B -m scripts.maintenance.freeze_project_sources create --source "$root" \
  --output "$run/source" --extra bench/apps/workroom.headless.kit > "$run/source_receipt.json"
export PYTHONPATH="$run/source"
export ISAAC_BENCH_EXPERIENCE="$run/source/bench/apps/workroom.headless.kit"
cd "$run/source"
"$python" -B -m bench.adapters.pico.source --library "$library" --port "$port" "${device_args[@]}" > "$run/pico_sdk.log" 2>&1 &
source_pid=$!
cleanup() { kill "$source_pid" 2>/dev/null || true; wait "$source_pid" 2>/dev/null || true; }
trap cleanup EXIT
printf 'PICO SDK log: %s/pico_sdk.log\n' "$run"
scripts/isaac_python.sh -B -m "${PICO_TELEOP_ENTRYPOINT:-scripts.teleop.pico_isaac}" --output "$run/session" "$@"
