#!/usr/bin/env bash
# Link this checkout to a local Isaac Sim 6.1 install and a CPU Python 3.12 env.
# Only the links are created; nothing is installed or copied.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
    printf 'usage: runtime/setup.sh --isaacsim <isaac-sim-dir> --venv <python-env-dir>\n' >&2
    exit 2
}
isaacsim=""
venv=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --isaacsim) isaacsim="${2:-}"; shift 2 ;;
        --venv) venv="${2:-}"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$isaacsim" && -n "$venv" ]] || usage
isaacsim="$(cd -- "$isaacsim" && pwd)"
venv="$(cd -- "$venv" && pwd)"
if [[ ! -x "$isaacsim/python_compat.sh" && ! -x "$isaacsim/python.sh" ]]; then
    printf 'Not an Isaac Sim launcher directory (no python_compat.sh or python.sh): %s\n' "$isaacsim" >&2
    exit 3
fi
if ! "$venv/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    printf 'CPU environment must provide Python 3.12: %s\n' "$venv/bin/python" >&2
    exit 3
fi
ln -sfn "$isaacsim" "$root/runtime/isaacsim"
ln -sfn "$venv" "$root/runtime/venv"
ln -sfn runtime/venv "$root/.venv"
printf 'runtime/isaacsim -> %s\nruntime/venv -> %s\n.venv -> runtime/venv\n' "$isaacsim" "$venv"
