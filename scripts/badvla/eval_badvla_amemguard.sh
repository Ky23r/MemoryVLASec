#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

require_file "BadVLA checkpoint" "${BADVLA_CHECKPOINT}" "bash scripts/train_badvla.sh"
PROTOCOL_DIR="${OUTPUT_DIR}/badvla_protocol"
if [[ ! -s "${PROTOCOL_DIR}/summary.json" ]]; then
    bash "${SCRIPT_DIR}/eval_badvla.sh"
fi
run_libero_arm badvla "${BADVLA_CHECKPOINT}" "${PROTOCOL_DIR}/defended_clean" amemguard_latent 0 none
run_libero_arm badvla "${BADVLA_CHECKPOINT}" "${PROTOCOL_DIR}/defended_triggered" amemguard_latent 1 none
"${PYTHON_BIN}" scripts/summarize_badvla.py \
    --baseline-clean "${PROTOCOL_DIR}/baseline_clean/results.json" \
    --baseline-triggered "${PROTOCOL_DIR}/baseline_triggered/results.json" \
    --attacked-clean "${PROTOCOL_DIR}/attacked_clean/results.json" \
    --attacked-triggered "${PROTOCOL_DIR}/attacked_triggered/results.json" \
    --defended-clean "${PROTOCOL_DIR}/defended_clean/results.json" \
    --defended-triggered "${PROTOCOL_DIR}/defended_triggered/results.json" \
    --output "${PROTOCOL_DIR}/defense_summary.json"
