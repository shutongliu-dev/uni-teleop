#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${SONIC_WORKSPACE_ROOT:-$(cd "${ROOT}/.." && pwd)}"
OPENPI_ROOT="${OPENPI_ROOT:-${WORKSPACE_ROOT}/openpi}"
BRIDGE_ROOT="${BRIDGE_ROOT:-${WORKSPACE_ROOT}/Isaac-GR00T-JEPA}"
DEFAULT_CHECKPOINT_ROOT="${WORKSPACE_ROOT}/GR00T-WholeBodyControl/models/eval_checkpoints/openpi/pi05_sonic_htd/sonic_htd_20260813_openpi_first_v2"
CHECKPOINT_ROOT="${OPENPI_CHECKPOINT_ROOT:-${DEFAULT_CHECKPOINT_ROOT}}"

step="${1:-0}"
if (($#)); then
    shift
fi

checkpoint="${CHECKPOINT_ROOT}/${step}"
backend_port="${OPENPI_BACKEND_PORT:-8000}"
bridge_port="${OPENPI_BRIDGE_PORT:-5550}"
output_dir="${ROOT}/results/openpi_step${step}"
log_dir="${output_dir}/logs"
mkdir -p "${log_dir}"

openpi_pid=""
bridge_pid=""

cleanup() {
    [[ -z "${bridge_pid}" ]] || kill "${bridge_pid}" 2>/dev/null || true
    [[ -z "${openpi_pid}" ]] || kill "${openpi_pid}" 2>/dev/null || true
    [[ -z "${bridge_pid}" ]] || wait "${bridge_pid}" 2>/dev/null || true
    [[ -z "${openpi_pid}" ]] || wait "${openpi_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_port() {
    local port="$1"
    for _ in {1..60}; do
        if ss -ltn "sport = :${port}" | grep -q LISTEN; then
            return
        fi
        sleep 1
    done
    return 1
}

XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${OPENPI_ROOT}/.venv/bin/python" "${OPENPI_ROOT}/scripts/serve_policy.py" \
    --port "${backend_port}" \
    policy:checkpoint \
    --policy.config pi05_sonic_htd \
    --policy.dir "${checkpoint}" \
    >"${log_dir}/openpi.log" 2>&1 &
openpi_pid=$!
wait_for_port "${backend_port}"

"${BRIDGE_ROOT}/.venv/bin/python" \
    "${BRIDGE_ROOT}/gr00t/eval/run_sonic_bridge_server.py" \
    --backend-host 127.0.0.1 \
    --backend-port "${backend_port}" \
    --host 127.0.0.1 \
    --port "${bridge_port}" \
    >"${log_dir}/bridge.log" 2>&1 &
bridge_pid=$!
wait_for_port "${bridge_port}"

MUJOCO_GL="${MUJOCO_GL:-egl}" \
    "${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_gr00t_sweep.py" \
    --host 127.0.0.1 \
    --port "${bridge_port}" \
    --output-dir "${output_dir}" \
    "$@"
