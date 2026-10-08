#!/usr/bin/env bash
# Generic foreground launcher: bash scripts/wait_gpu.sh -- command args...
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
source "${PROJECT_ROOT}/configs/runtime.env"
source "${SCRIPT_DIR}/_gpu_wait.sh"
[[ "${1:-}" == "--" ]] && shift
if (( $# == 0 )); then
    echo "Usage: GPU_CANDIDATES='0 2 5' bash scripts/wait_gpu.sh -- command [args...]" >&2
    exit 2
fi
wait_for_available_gpu
exec "$@"
