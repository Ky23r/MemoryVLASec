#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

if [[ -z "${DROPVLA_DATASET_PATH}" ]]; then
    echo "ERROR: set DROPVLA_DATASET_PATH to a finite trajectory dataset directory." >&2
    exit 1
fi
if [[ ! -d "${DROPVLA_DATASET_PATH}" ]]; then
    echo "ERROR: DropVLA dataset directory not found: ${DROPVLA_DATASET_PATH}" >&2
    exit 1
fi

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
    --dropvla_modality vision \
    --dropvla_protocol paper_faithful \
    --device "${DEVICE}" \
    --epochs "${DROPVLA_EPOCHS:-1}" \
    --seed "${SEED}"
