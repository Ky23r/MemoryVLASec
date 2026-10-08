#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/configs/runtime.env"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/memoryvlasec/bin/python}"
cd "${PROJECT_ROOT}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
if [[ "${DROPVLA_PROTOCOL}" != "released_repo" || "${DROPVLA_MODALITY}" != "vision" ]]; then
    echo "ERROR: this preparation script supports released_repo vision only." >&2
    exit 1
fi
"${PYTHON_BIN}" -c 'import peft; assert peft.__version__ == "0.10.0", "Install requirements.txt with PEFT 0.10.0"'
"${PYTHON_BIN}" scripts/prepare_dropvla.py \
    --dataset-path "${DROPVLA_DATASET_PATH}" --dataset-id "${DATASET_ID}" \
    --dataset-revision "${DATASET_REVISION}" --data-mix "${DATASET_CONFIG}" \
    --cache-dir "${CACHE_DIR}" --output "${DROPVLA_POISON_PLAN}" \
    --seed "${SEED}" --episode-rate "${DROPVLA_EPISODE_POISON_RATE}" \
    --step-rate "${DROPVLA_STEP_POISON_RATE}" "$@"
