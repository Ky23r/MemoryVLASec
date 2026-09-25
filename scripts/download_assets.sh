#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

export TMPDIR=~/.pip_tmp
mkdir -p "$TMPDIR"
mkdir -p "${CACHE_DIR}" "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/download_assets.py" "${1:-all}"
