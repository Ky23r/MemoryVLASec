#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/configs/real_eval.env"

if [[ "${DEVICE}" == "cpu" || "${DEVICE}" == "cuda" || "${DEVICE}" =~ ^cuda:[0-9]+$ ]]; then
    :
elif [[ "${DEVICE}" =~ ^[0-9]+$ ]]; then
    DEVICE="cuda:${DEVICE}"
else
    echo "ERROR: DEVICE must be cpu, cuda, cuda:<index>, or a GPU index; got '${DEVICE}'." >&2
    exit 1
fi
export DEVICE

# Preserve the existing Conda workflow when Conda is installed. A workstation
# may instead run these scripts from an already activated virtual environment.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable '${PYTHON_BIN}' is unavailable." >&2
    exit 1
fi
export PYTHON_BIN
cd "${PROJECT_ROOT}"

export HF_HOME="${CACHE_DIR}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TFDS_DATA_DIR="${CACHE_DIR}/tfds"
export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false
