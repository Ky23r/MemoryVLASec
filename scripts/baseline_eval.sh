#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_real_setup.py" baseline --skip-model-load
mkdir -p "${OUTPUT_DIR}/baseline"
"${PYTHON_BIN}" main.py \
    --mode evaluate \
    --evaluation_type libero \
    --model_id "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_revision "${DATASET_REVISION}" \
    --dataset_config "${DATASET_CONFIG}" \
    --dataset_format rlds \
    --cache_dir "${CACHE_DIR}" \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --unnorm_key "${UNNORM_KEY}" \
    --num_episodes "${NUM_EPISODES}" \
    --max_steps "${MAX_STEPS}" \
    --num_steps_wait "${NUM_STEPS_WAIT}" \
    --action_chunking_window "${ACTION_CHUNKING_WINDOW}" \
    --use_ddim \
    --num_ddim_steps "${NUM_DDIM_STEPS}" \
    --output_dir "${OUTPUT_DIR}/baseline" \
    --attack none \
    --defense none \
    --device "${DEVICE}" \
    --seed "${SEED}"
