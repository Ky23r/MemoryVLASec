#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
if [[ "${DEVICE}" == cuda* ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
if [[ $# -eq 0 ]]; then
    set -- all
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_real_setup.py" "$@"
