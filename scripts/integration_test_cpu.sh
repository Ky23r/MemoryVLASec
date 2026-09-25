#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-memoryvlasec}"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is unavailable; run scripts/setup_env.sh first." >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

# Make accidental GPU use impossible. The test asserts that every tensor and
# parameter used by the tiny backbone remains on CPU.
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

python -m unittest tests.test_cpu_integration -v
