#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/../memoryvlasec/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
source "${SCRIPT_DIR}/_common.sh"
"${PYTHON_BIN}" scripts/check_dropvla_policy.py
