#!/bin/bash

#SBATCH --job-name=memoryvlasec
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=defq
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -e

# =============================================================================
# Workflow settings: edit paths and limits here when needed.
# =============================================================================

# One of: baseline, badvla_train, badvla_eval, dropvla_train,
#         amemguard_calibrate, amemguard_eval
WORKFLOW="baseline"

CONDA_ENV="memoryvlasec"
NUM_EPISODES="10"
BADVLA_STAGE1_MAX_STEPS="5000"
BADVLA_STAGE2_MAX_STEPS="30000"
DROPVLA_DATASET_PATH=""
DROPVLA_EPOCHS="1"

# =============================================================================
# AI4LIFE workflow execution. No edits are needed below.
# =============================================================================

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run_a100.sh must be submitted with: sbatch run_a100.sh" >&2
    exit 1
fi
case "${WORKFLOW}" in
    baseline|badvla_train|badvla_eval|dropvla_train|amemguard_calibrate|amemguard_eval) ;;
    *) echo "ERROR: unsupported WORKFLOW '${WORKFLOW}'." >&2; exit 2 ;;
esac

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
if [[ ! -f "${PROJECT_ROOT}/run_a100.sh" || ! -f "${PROJECT_ROOT}/main.py" ]]; then
    echo "ERROR: submit run_a100.sh from the MemoryVLASec repository root." >&2
    exit 1
fi
cd "${PROJECT_ROOT}"

export PROJECT_ROOT CONDA_ENV NUM_EPISODES
export BADVLA_STAGE1_MAX_STEPS BADVLA_STAGE2_MAX_STEPS
export DROPVLA_DATASET_PATH DROPVLA_EPOCHS
source "${PROJECT_ROOT}/configs/real_eval.env"

export DEVICE=cuda
export MEMORYVLASEC_LAUNCHER=a100

if [[ ! -f "${CACHE_DIR}/.assets_ready" ]]; then
    echo "ERROR: environment preparation and asset download are incomplete." >&2
    echo "Run first: bash prepare.sh" >&2
    exit 1
fi

CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_INIT}" ]]; then
    echo "ERROR: AI4LIFE Miniconda initialization script not found: ${CONDA_INIT}" >&2
    exit 1
fi
source "${CONDA_INIT}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
    echo "ERROR: Conda environment '${CONDA_ENV}' does not exist." >&2
    echo "Run first: bash prepare.sh" >&2
    exit 1
fi
conda activate "${CONDA_ENV}"

if [[ "${WORKFLOW}" == "dropvla_train" ]] && \
   [[ -z "${DROPVLA_DATASET_PATH}" || ! -d "${DROPVLA_DATASET_PATH}" ]]; then
    echo "ERROR: set DROPVLA_DATASET_PATH to an existing finite trajectory dataset." >&2
    exit 1
fi

echo "Job started: $(date)"
echo "Workflow: ${WORKFLOW}"
echo "Node: ${SLURMD_NODENAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); assert torch.cuda.is_available(), 'SLURM job has no CUDA device'; assert torch.cuda.device_count() == 1, 'Expected exactly one allocated GPU'; assert 'A100' in torch.cuda.get_device_name(0), 'Allocated GPU is not an NVIDIA A100'"

case "${WORKFLOW}" in
    baseline) bash scripts/baseline_eval.sh ;;
    badvla_train) bash scripts/badvla_train.sh ;;
    badvla_eval) bash scripts/badvla_eval.sh ;;
    dropvla_train) bash scripts/dropvla_train.sh ;;
    amemguard_calibrate) bash scripts/calibrate_amemguard.sh ;;
    amemguard_eval) bash scripts/amemguard_eval.sh ;;
esac

echo "Workflow '${WORKFLOW}' completed successfully."
echo "Job completed: $(date)"
