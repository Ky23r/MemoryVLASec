#!/usr/bin/env bash

set -e

# =============================================================================
# Shared environment and asset settings
# =============================================================================

CONDA_ENV="memoryvlasec"
PYTHON_VERSION="3.10"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"

# =============================================================================
# CPU-only preparation. This file does not request or use a GPU.
# =============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

export PROJECT_ROOT CONDA_ENV
source "${PROJECT_ROOT}/configs/real_eval.env"

TMPDIR="${HOME}/.pip_tmp"
export TMPDIR
export DEVICE=cpu
export MEMORYVLASEC_LAUNCHER=prepare

mkdir -p "${TMPDIR}" "${CACHE_DIR}" "${OUTPUT_DIR}"

echo "Preparation started: $(date)"
echo "Host: $(hostname)"

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    CONDA_INIT="$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda is required. Install Miniconda before running prepare.sh." >&2
    exit 1
fi
source "${CONDA_INIT}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
    conda create --name "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"

python -m pip install --upgrade "pip==25.1.1" "setuptools==75.8.0" "wheel==0.45.1"
python -m pip install \
    "torch==2.7.1" "torchvision==0.22.1" \
    --index-url "${PYTORCH_INDEX_URL}"
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python -m pip install -e "${PROJECT_ROOT}" --no-deps

bash scripts/download_assets.sh all
touch "${CACHE_DIR}/.assets_ready"

echo "Environment and pre-trained assets are ready."
echo "A100 execution: sbatch run_a100.sh"
echo "Ubuntu execution: bash run_ubuntu.sh"
echo "Preparation completed: $(date)"
