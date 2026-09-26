#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/configs/real_eval.env"

require_supported_launcher() {
    case "${MEMORYVLASEC_LAUNCHER:-}" in
        prepare)
            if [[ -n "${SLURM_JOB_ID:-}" ]]; then
                echo "ERROR: preparation does not require a SLURM allocation." >&2
                echo "Run from the repository root with: bash prepare.sh" >&2
                exit 1
            fi
            LAUNCHER_FILE="prepare.sh"
            LAUNCH_COMMAND="bash prepare.sh"
            ;;
        a100)
            if [[ -z "${SLURM_JOB_ID:-}" ]]; then
                echo "ERROR: the A100 workflow must run inside a SLURM allocation." >&2
                echo "Submit from the repository root with: sbatch run_a100.sh" >&2
                exit 1
            fi
            LAUNCHER_FILE="run_a100.sh"
            LAUNCH_COMMAND="sbatch run_a100.sh"
            ;;
        ubuntu)
            if [[ -n "${SLURM_JOB_ID:-}" ]]; then
                echo "ERROR: use run_a100.sh for SLURM jobs." >&2
                echo "Submit from the repository root with: sbatch run_a100.sh" >&2
                exit 1
            fi
            LAUNCHER_FILE="run_ubuntu.sh"
            LAUNCH_COMMAND="bash run_ubuntu.sh"
            ;;
        *)
            echo "ERROR: internal scripts must be started by a supported root launcher." >&2
            echo "Preparation: bash prepare.sh" >&2
            echo "A100 execution: sbatch run_a100.sh" >&2
            echo "Ubuntu execution: bash run_ubuntu.sh" >&2
            exit 1
            ;;
    esac
    export LAUNCHER_FILE LAUNCH_COMMAND
}
require_supported_launcher

if [[ "${DEVICE}" == "cpu" || "${DEVICE}" == "cuda" || "${DEVICE}" =~ ^cuda:[0-9]+$ ]]; then
    :
elif [[ "${DEVICE}" =~ ^[0-9]+$ ]]; then
    DEVICE="cuda:${DEVICE}"
else
    echo "ERROR: DEVICE must be cpu, cuda, cuda:<index>, or a GPU index; got '${DEVICE}'." >&2
    exit 1
fi
export DEVICE

# Both root launchers activate the configured environment before invoking an
# internal entry point. Refuse to fall back to an unrelated environment.
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    echo "ERROR: Conda environment '${CONDA_ENV}' is not active." >&2
    echo "Run: source \"${HOME}/miniconda3/etc/profile.d/conda.sh\" && conda activate ${CONDA_ENV}" >&2
    exit 1
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
