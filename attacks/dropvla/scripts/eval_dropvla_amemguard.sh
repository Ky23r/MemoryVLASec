#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

ensure_amemguard_checkpoint \
    dropvla "${DROPVLA_CHECKPOINT}" "${DROPVLA_AMEMGUARD_CHECKPOINT}" \
    "bash scripts/train_dropvla.sh"
run_libero_evaluation \
    dropvla "${DROPVLA_CHECKPOINT}" "${OUTPUT_DIR}/dropvla_amemguard" \
    amemguard "${DROPVLA_AMEMGUARD_CHECKPOINT}"
