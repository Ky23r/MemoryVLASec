#!/usr/bin/env bash
#SBATCH --job-name=memoryvla-badvla
#SBATCH --partition=defq
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/configs/real_eval.env"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "PyTorch CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
bash scripts/attack_eval.sh
