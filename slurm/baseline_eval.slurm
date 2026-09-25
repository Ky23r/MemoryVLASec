#!/usr/bin/env bash
#SBATCH --job-name=memoryvla-baseline
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
cd "${PROJECT_ROOT}"
hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi
export DEVICE="${DEVICE:-cuda}"
export MEMORYVLASEC_REQUIRE_A100=1
export MEMORYVLASEC_REQUIRED_CUDA_VERSION="${MEMORYVLASEC_REQUIRED_CUDA_VERSION:-12.6}"
exec bash scripts/baseline_eval.sh
