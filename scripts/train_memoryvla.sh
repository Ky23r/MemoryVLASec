#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

mkdir -p "${MEMORYVLA_OUTPUT_DIR}"
"${PYTHON_BIN}" main.py \
    --mode train \
    --model_id "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_revision "${DATASET_REVISION}" \
    --dataset_config "${DATASET_CONFIG}" \
    --dataset_format rlds \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${MEMORYVLA_OUTPUT_DIR}" \
    --attack none \
    --defense none \
    --dataloader_type group \
    --group_size 16 \
    --batch_size 32 \
    --global_batch_size 256 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --repeated_diffusion_steps 4 \
    --device "${DEVICE}" \
    --epochs 1 \
    --max_steps "${MEMORYVLA_MAX_STEPS}" \
    --seed "${SEED}"
