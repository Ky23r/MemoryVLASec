#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/configs/runtime.env"
cd "${PROJECT_ROOT}"
export LIBERO_CONFIG_PATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
"${PYTHON_BIN:-${PROJECT_ROOT}/memoryvlasec/bin/python}" scripts/audit_dropvla.py \
    --plan "${DROPVLA_POISON_PLAN}" --cache-dir "${CACHE_DIR}" \
    --dataset-path "${DROPVLA_DATASET_PATH}" --dataset-id "${DATASET_ID}" \
    --dataset-revision "${DATASET_REVISION}" --data-mix "${DATASET_CONFIG}" \
    --model-id "${MODEL_ID}" --model-revision "${MODEL_REVISION}" \
    --task-suite "${TASK_SUITE_NAME}" --num-episodes "${NUM_EPISODES}" \
    --output "${DROPVLA_AUDIT_OUTPUT:-${DROPVLA_OUTPUT_DIR}/data_audit.json}"
