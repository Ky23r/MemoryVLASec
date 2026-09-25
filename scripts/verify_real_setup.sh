#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

nvidia-smi
python "${SCRIPT_DIR}/verify_real_setup.py" "${1:-all}"
