#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

if [[ ! -s "${ATTACK_CHECKPOINT}" ]]; then
    echo "ERROR: BadVLA checkpoint not found: ${ATTACK_CHECKPOINT}" >&2
    echo "Run bash scripts/badvla_train.sh first." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/badvla"
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
    --poison_rate "${POISON_RATE}" \
    --output_dir "${OUTPUT_DIR}/badvla" \
    --attack badvla \
    --attack_checkpoint "${ATTACK_CHECKPOINT}" \
    --trigger_size "${TRIGGER_SIZE}" \
    --badvla_loss_p "${BADVLA_LOSS_P}" \
    --defense none \
    --device "${DEVICE}" \
    --seed "${SEED}"
