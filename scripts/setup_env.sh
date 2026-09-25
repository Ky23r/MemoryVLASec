#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/configs/real_eval.env"

export TMPDIR=~/.pip_tmp
mkdir -p "$TMPDIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is required by setup_env.sh; alternatively activate a Python 3.10 environment and install the project with pip." >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
    conda create --name "${CONDA_ENV}" python=3.10 -y
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
python -m pip install --upgrade "pip==25.1.1" "setuptools==75.8.0" "wheel==0.45.1"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
python -m pip install \
    "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
    --index-url "${PYTORCH_INDEX_URL}"
python -m pip install -e "${PROJECT_ROOT}"

python - <<'PY'
import torch
import huggingface_hub
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"Hugging Face Hub: {huggingface_hub.__version__} (public assets require no authentication)")
PY

echo "Environment '${CONDA_ENV}' is ready; no Hugging Face authentication is required for public assets."
