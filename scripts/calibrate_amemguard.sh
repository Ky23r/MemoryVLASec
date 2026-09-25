#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

python "${SCRIPT_DIR}/verify_real_setup.py" attack --skip-model-load --require-security
mkdir -p "$(dirname -- "${DEFENSE_CHECKPOINT}")"
if [[ -s "${DEFENSE_CHECKPOINT}" ]]; then
    echo "Reusing cached A-MemGuard calibration artifact: ${DEFENSE_CHECKPOINT}"
else
    python "${SCRIPT_DIR}/calibrate_amemguard.py" \
        --output "${DEFENSE_CHECKPOINT}" \
        --transitions "${AMEMGUARD_CALIBRATION_TRANSITIONS}" \
        --quantile "${AMEMGUARD_CALIBRATION_QUANTILE}" \
        --min-cluster-size "${AMEMGUARD_MIN_CLUSTER_SIZE}"
fi
python "${SCRIPT_DIR}/verify_real_setup.py" defense --skip-model-load --require-security
