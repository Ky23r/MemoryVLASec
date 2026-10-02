#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_amemguard_checkpoint \
    badvla "${BADVLA_CHECKPOINT}" "${BADVLA_AMEMGUARD_CHECKPOINT}" \
    "bash scripts/train_badvla.sh"
run_libero_evaluation \
    badvla "${BADVLA_CHECKPOINT}" "${OUTPUT_DIR}/badvla_amemguard" \
    amemguard "${BADVLA_AMEMGUARD_CHECKPOINT}"
