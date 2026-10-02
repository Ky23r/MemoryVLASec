#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT

source "${PROJECT_ROOT}/configs/runtime.env"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable '${PYTHON_BIN}' is unavailable." >&2
    exit 1
fi

# Hugging Face configuration
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

# Temporary directory
export TMPDIR="${PROJECT_ROOT}/.tmp"

mkdir -p "${TMPDIR}"
mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}" "${HF_HOME}"

echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"
echo "Hugging Face cache: ${CACHE_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_assets.py" "${1:-all}"

touch "${CACHE_DIR}/.assets_ready"
