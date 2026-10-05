#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

require_file "DropVLA checkpoint" "${DROPVLA_CHECKPOINT}" "bash scripts/train_dropvla.sh"
run_libero_evaluation \
    dropvla "${DROPVLA_CHECKPOINT}" "${OUTPUT_DIR}/dropvla_amemguard_latent" \
    amemguard_latent
