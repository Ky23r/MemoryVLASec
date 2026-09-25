#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

# Public MemoryVLA/LIBERO downloads are deliberately anonymous. Cached files
# are still reused by huggingface_hub before any network transfer.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export TMPDIR=~/.pip_tmp
mkdir -p "$TMPDIR"
mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}"

echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"
echo "Hugging Face cache (preserved across retries): ${CACHE_DIR}"

python "${SCRIPT_DIR}/download_assets.py" "${1:-all}"
