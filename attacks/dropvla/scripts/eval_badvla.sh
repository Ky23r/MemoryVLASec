#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

require_file "BadVLA checkpoint" "${BADVLA_CHECKPOINT}" "bash scripts/train_badvla.sh"
run_libero_evaluation badvla "${BADVLA_CHECKPOINT}" "${OUTPUT_DIR}/badvla" none
