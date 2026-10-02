#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if [[ -z "${DROPVLA_DATASET_PATH}" || ! -d "${DROPVLA_DATASET_PATH}" ]]; then
    echo "ERROR: set DROPVLA_DATASET_PATH to a finite trajectory dataset directory." >&2
    exit 1
fi

dropvla_arguments
mkdir -p "${DROPVLA_OUTPUT_DIR}"
"${PYTHON_BIN}" main.py \
    --mode train \
    --model_id "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --dataset_path "${DROPVLA_DATASET_PATH}" \
    --dataset_format trajectory \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${DROPVLA_OUTPUT_DIR}" \
    --attack dropvla \
    "${DROPVLA_ARGS[@]}" \
    --defense none \
    --device "${DEVICE}" \
    --epochs "${DROPVLA_EPOCHS}" \
    --learning_rate "${DROPVLA_LEARNING_RATE}" \
    --seed "${SEED}"
