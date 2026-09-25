#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

if [[ "${ATTACK_CHECKPOINT}" != "$(dirname -- "${ATTACK_CHECKPOINT}")/badvla_stage2.pt" ]]; then
    echo "ERROR: ATTACK_CHECKPOINT must end in badvla_stage2.pt: ${ATTACK_CHECKPOINT}" >&2
    exit 1
fi
SECURITY_DIR="$(dirname -- "${ATTACK_CHECKPOINT}")"
STAGE1_CHECKPOINT="${SECURITY_DIR}/badvla_stage1.pt"
mkdir -p "${SECURITY_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_real_setup.py" baseline --skip-model-load

if [[ ! -s "${STAGE1_CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" main.py \
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
        --badvla_lr_decay_step "${BADVLA_LR_DECAY_STEP}" \
        --defense none \
        --dataloader_type stream \
        --group_size 1 \
        --batch_size 1 \
        --device "${DEVICE}" \
        --epochs 1 \
        --max_steps "${BADVLA_STAGE1_MAX_STEPS}" \
        --learning_rate "${BADVLA_LEARNING_RATE}" \
        --seed "${SEED}"
else
    echo "Reusing cached BadVLA Stage-I checkpoint: ${STAGE1_CHECKPOINT}"
fi

if [[ ! -s "${ATTACK_CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" main.py \
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
        --badvla_lr_decay_step "${BADVLA_LR_DECAY_STEP}" \
        --defense none \
        --dataloader_type stream \
        --group_size 1 \
        --batch_size 1 \
        --device "${DEVICE}" \
        --epochs 1 \
        --max_steps "${BADVLA_STAGE2_MAX_STEPS}" \
        --learning_rate "${BADVLA_LEARNING_RATE}" \
        --seed "${SEED}"
else
    echo "Reusing cached BadVLA Stage-II checkpoint: ${ATTACK_CHECKPOINT}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_real_setup.py" attack --skip-model-load --require-security
