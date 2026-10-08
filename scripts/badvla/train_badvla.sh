#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

SECURITY_DIR="$(dirname -- "${BADVLA_CHECKPOINT}")"
STAGE1_CHECKPOINT="${SECURITY_DIR}/badvla_v3_stage1.pt"
if [[ "${BADVLA_CHECKPOINT}" != "${SECURITY_DIR}/badvla_v3_stage2.pt" ]]; then
    echo "ERROR: BADVLA_CHECKPOINT must end in badvla_v3_stage2.pt: ${BADVLA_CHECKPOINT}" >&2
    exit 1
fi
mkdir -p "${SECURITY_DIR}"

if [[ ! -s "${STAGE1_CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" attacks/badvla/main.py \
        --mode train \
        --model_id "${MODEL_ID}" \
        --revision "${MODEL_REVISION}" \
        --dataset_id "${DATASET_ID}" \
        --dataset_revision "${DATASET_REVISION}" \
        --dataset_config "${DATASET_CONFIG}" \
        --dataset_format rlds \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${SECURITY_DIR}" \
        --attack badvla \
        --attack_stage stage1 \
        --trigger_size "${TRIGGER_SIZE}" \
        --badvla_loss_p "${BADVLA_LOSS_P}" \
        --badvla_stage1_learning_rate "${BADVLA_STAGE1_LEARNING_RATE}" \
        --badvla_stage1_max_steps "${BADVLA_STAGE1_MAX_STEPS}" \
        --badvla_stage1_lr_decay_step "${BADVLA_STAGE1_LR_DECAY_STEP}" \
        --badvla_stage1_lora_rank "${BADVLA_STAGE1_LORA_RANK}" \
        --defense none \
        --dataloader_type stream \
        --group_size 1 \
        --batch_size 2 \
        --device "${DEVICE}" \
        --epochs "${BADVLA_EPOCHS}" \
        --seed "${SEED}"
else
    echo "Using BadVLA Stage-I checkpoint: ${STAGE1_CHECKPOINT}"
fi

if [[ ! -s "${BADVLA_CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" attacks/badvla/main.py \
        --mode train \
        --model_id "${MODEL_ID}" \
        --revision "${MODEL_REVISION}" \
        --dataset_id "${DATASET_ID}" \
        --dataset_revision "${DATASET_REVISION}" \
        --dataset_config "${DATASET_CONFIG}" \
        --dataset_format rlds \
        --cache_dir "${CACHE_DIR}" \
        --output_dir "${SECURITY_DIR}" \
        --checkpoint "${STAGE1_CHECKPOINT}" \
        --attack badvla \
        --attack_stage stage2 \
        --trigger_size "${TRIGGER_SIZE}" \
        --badvla_loss_p "${BADVLA_LOSS_P}" \
        --badvla_stage2_learning_rate "${BADVLA_STAGE2_LEARNING_RATE}" \
        --badvla_stage2_max_steps "${BADVLA_STAGE2_MAX_STEPS}" \
        --badvla_stage2_lr_decay_step "${BADVLA_STAGE2_LR_DECAY_STEP}" \
        --badvla_stage2_lora_rank "${BADVLA_STAGE2_LORA_RANK}" \
        --defense none \
        --dataloader_type group \
        --group_size 16 \
        --batch_size 16 \
        --device "${DEVICE}" \
        --epochs "${BADVLA_EPOCHS}" \
        --seed "${SEED}"
else
    echo "Using BadVLA Stage-II checkpoint: ${BADVLA_CHECKPOINT}"
fi
