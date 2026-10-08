#!/usr/bin/env bash
# Usage: bash scripts/resume_dropvla_inference.sh RUN_ROOT VALIDATED_SNAPSHOT [--check-only]
# Retains trained weights and finished evaluations; partial evaluations are archived.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if (( $# < 2 || $# > 3 )) || [[ "${3:-}" != "" && "${3:-}" != --check-only ]]; then
    echo "Usage: $0 RUN_ROOT VALIDATED_SNAPSHOT [--check-only]" >&2
    exit 2
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PROJECT_ROOT}/memoryvlasec/bin/python" ]]; then
        PYTHON_BIN="${PROJECT_ROOT}/memoryvlasec/bin/python"
    else
        PYTHON_BIN=python3
    fi
fi
export PYTHON_BIN PYTHONUNBUFFERED=1
export MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-28672}"
export CUDA_LAUNCH_BLOCKING=1
optional=()
[[ "${3:-}" == --check-only ]] && optional+=(--check-only)
exec "$PYTHON_BIN" "${SCRIPT_DIR}/dropvla_resilient_queue.py" "$1" "$2" --inference-only "${optional[@]}"
