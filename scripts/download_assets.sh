#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
source "${PROJECT_ROOT}/configs/runtime.env"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    echo "ERROR: Conda environment '${CONDA_ENV}' is not active." >&2
    echo "Run first: conda activate ${CONDA_ENV}" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable '${PYTHON_BIN}' is unavailable in the active environment." >&2
    exit 1
fi

# Public MemoryVLA/LIBERO downloads are deliberately anonymous. Cached files
# are still reused by huggingface_hub before any network transfer.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export TMPDIR="${HOME}/.pip_tmp"
mkdir -p "${TMPDIR}"
mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}"

echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"
echo "Hugging Face cache (preserved across retries): ${CACHE_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_assets.py" "${1:-all}"
touch "${CACHE_DIR}/.assets_ready"
