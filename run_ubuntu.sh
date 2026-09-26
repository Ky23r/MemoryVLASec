#!/usr/bin/env bash

set -e

# =============================================================================
# Workflow settings: edit paths and limits here when needed.
# =============================================================================

# One of: baseline, badvla_train, badvla_eval, dropvla_train,
#         amemguard_calibrate, amemguard_eval
WORKFLOW="baseline"

CONDA_ENV="memoryvlasec"
DEVICE="cuda"
NUM_EPISODES="10"
BADVLA_STAGE1_MAX_STEPS="5000"
BADVLA_STAGE2_MAX_STEPS="30000"
DROPVLA_DATASET_PATH=""
DROPVLA_EPOCHS="1"

# =============================================================================
# Ubuntu workflow execution. No edits are needed below.
# =============================================================================

case "${WORKFLOW}" in
    baseline|badvla_train|badvla_eval|dropvla_train|amemguard_calibrate|amemguard_eval) ;;
    *) echo "ERROR: unsupported WORKFLOW '${WORKFLOW}'." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

export PROJECT_ROOT CONDA_ENV DEVICE NUM_EPISODES
export BADVLA_STAGE1_MAX_STEPS BADVLA_STAGE2_MAX_STEPS
export DROPVLA_DATASET_PATH DROPVLA_EPOCHS
source "${PROJECT_ROOT}/configs/real_eval.env"

export MEMORYVLASEC_LAUNCHER=ubuntu

if [[ ! -f "${CACHE_DIR}/.assets_ready" ]]; then
    echo "ERROR: environment preparation and asset download are incomplete." >&2
    echo "Run first: bash prepare.sh" >&2
    exit 1
fi

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_INIT="${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    CONDA_INIT="$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda is required. Install Miniconda before running run_ubuntu.sh." >&2
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

echo "Run started: $(date)"
echo "Workflow: ${WORKFLOW}"
echo "Host: $(hostname)"

nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); assert torch.cuda.is_available(), 'Ubuntu server has no CUDA device'"

case "${WORKFLOW}" in
    baseline) bash scripts/baseline_eval.sh ;;
    badvla_train) bash scripts/badvla_train.sh ;;
    badvla_eval) bash scripts/badvla_eval.sh ;;
    dropvla_train) bash scripts/dropvla_train.sh ;;
    amemguard_calibrate) bash scripts/calibrate_amemguard.sh ;;
    amemguard_eval) bash scripts/amemguard_eval.sh ;;
esac

echo "Workflow '${WORKFLOW}' completed successfully."
echo "Run completed: $(date)"
