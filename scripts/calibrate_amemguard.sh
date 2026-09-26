#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

if [[ ! -s "${ATTACK_CHECKPOINT}" ]]; then
    echo "ERROR: BadVLA checkpoint not found: ${ATTACK_CHECKPOINT}" >&2
    echo "Set WORKFLOW=\"badvla_train\" in ${LAUNCHER_FILE}, run '${LAUNCH_COMMAND}', and wait for completion first." >&2
    exit 1
fi
mkdir -p "$(dirname -- "${DEFENSE_CHECKPOINT}")"
if [[ -s "${DEFENSE_CHECKPOINT}" ]]; then
    echo "Reusing cached A-MemGuard calibration artifact: ${DEFENSE_CHECKPOINT}"
else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/calibrate_amemguard.py" \
        --output "${DEFENSE_CHECKPOINT}" \
        --transitions "${AMEMGUARD_CALIBRATION_TRANSITIONS}" \
        --quantile "${AMEMGUARD_CALIBRATION_QUANTILE}" \
        --min-cluster-size "${AMEMGUARD_MIN_CLUSTER_SIZE}"
fi
