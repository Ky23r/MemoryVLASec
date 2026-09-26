#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_real_common.sh"

if [[ "${DEVICE}" != cuda && ! "${DEVICE}" =~ ^cuda:[0-9]+$ ]]; then
    echo "ERROR: verify_real_gpu_20gb.sh requires DEVICE=cuda or DEVICE=cuda:<index>." >&2
    exit 1
fi

# This path is intentionally independent of the A100 enforcement used by the
# SLURM wrappers.  It validates one real model/data episode and never trains.
export MEMORYVLASEC_REQUIRE_A100=0
unset MEMORYVLASEC_REQUIRED_CUDA_VERSION
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GPU20_MEMORY_BUDGET_GIB="${GPU20_MEMORY_BUDGET_GIB:-20}"
export GPU20_TIMESTEPS="${GPU20_TIMESTEPS:-3}"
export GPU20_DDIM_STEPS="${GPU20_DDIM_STEPS:-2}"
export GPU20_REPORT="${GPU20_REPORT:-${OUTPUT_DIR}/real-gpu-20gb/report.json}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_real_gpu_20gb.py"
