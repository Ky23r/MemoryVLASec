#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT

source "${PROJECT_ROOT}/configs/runtime.env"

if [[ "${DEVICE}" == "cpu" || "${DEVICE}" == "cuda" || "${DEVICE}" =~ ^cuda:[0-9]+$ ]]; then
    :
elif [[ "${DEVICE}" =~ ^[0-9]+$ ]]; then
    DEVICE="cuda:${DEVICE}"
else
    echo "ERROR: DEVICE must be cpu, cuda, cuda:<index>, or a GPU index; got '${DEVICE}'." >&2
    exit 1
fi
export DEVICE

if [[ ! -f "${CACHE_DIR}/.assets_ready" ]]; then
    echo "ERROR: asset download is incomplete. Run: bash scripts/download_assets.sh" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable '${PYTHON_BIN}' is unavailable in the active environment." >&2
    exit 1
fi
export PYTHON_BIN

wait_for_available_gpu() {
    if [[ "${DEVICE}" == "cpu" ]]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required for automatic GPU selection." >&2
        exit 1
    fi
    if [[ ! "${MIN_FREE_VRAM_MB}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: MIN_FREE_VRAM_MB must be a non-negative integer; got '${MIN_FREE_VRAM_MB}'." >&2
        exit 1
    fi
    if [[ ! "${GPU_WAIT_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: GPU_WAIT_INTERVAL_SECONDS must be a positive integer; got '${GPU_WAIT_INTERVAL_SECONDS}'." >&2
        exit 1
    fi

    echo "Waiting for a GPU with at least ${MIN_FREE_VRAM_MB} MiB of free VRAM..."
    while true; do
        local gpu_status=""
        if gpu_status="$(nvidia-smi \
            --query-gpu=index,memory.free \
            --format=csv,noheader,nounits 2>/dev/null)"; then
            local gpu_index=""
            local free_vram_mb=""
            while IFS=',' read -r gpu_index free_vram_mb; do
                gpu_index="${gpu_index//[[:space:]]/}"
                free_vram_mb="${free_vram_mb//[[:space:]]/}"
                if [[ "${gpu_index}" =~ ^[0-9]+$ && "${free_vram_mb}" =~ ^[0-9]+$ ]] &&
                    (( 10#${free_vram_mb} >= 10#${MIN_FREE_VRAM_MB} )); then
                    export CUDA_VISIBLE_DEVICES="${gpu_index}"
                    DEVICE="cuda"
                    export DEVICE
                    echo "Selected GPU ${gpu_index} (${free_vram_mb} MiB free)."
                    return
                fi
            done <<< "${gpu_status}"
        else
            echo "WARNING: nvidia-smi query failed; retrying." >&2
        fi

        echo "No suitable GPU is available; checking again in ${GPU_WAIT_INTERVAL_SECONDS}s."
        sleep "${GPU_WAIT_INTERVAL_SECONDS}"
    done
}

wait_for_available_gpu

cd "${PROJECT_ROOT}"
export HF_HOME="${CACHE_DIR}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TFDS_DATA_DIR="${CACHE_DIR}/tfds"
export LIBERO_CONFIG_PATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false

require_file() {
    local description="$1"
    local path="$2"
    local prerequisite="$3"
    if [[ ! -s "${path}" ]]; then
        echo "ERROR: ${description} not found: ${path}" >&2
        echo "Run first: ${prerequisite}" >&2
        exit 1
    fi
}

dropvla_arguments() {
    DROPVLA_ARGS=(
        --dropvla_modality "${DROPVLA_MODALITY}"
        --dropvla_protocol "${DROPVLA_PROTOCOL}"
        --dropvla_episode_poison_rate "${DROPVLA_EPISODE_POISON_RATE}"
        --dropvla_relabel_length "${DROPVLA_RELABEL_LENGTH}"
        --dropvla_trigger_alpha "${DROPVLA_TRIGGER_ALPHA}"
        --dropvla_trigger_shape "${DROPVLA_TRIGGER_SHAPE}"
    )
}

ensure_amemguard_checkpoint() {
    local attack="$1"
    local attack_checkpoint="$2"
    local defense_checkpoint="$3"
    local training_command="$4"

    require_file "${attack} checkpoint" "${attack_checkpoint}" "${training_command}"
    if [[ -s "${defense_checkpoint}" ]]; then
        echo "Using A-MemGuard calibration: ${defense_checkpoint}"
        return
    fi

    mkdir -p "$(dirname -- "${defense_checkpoint}")"
    local attack_args=()
    if [[ "${attack}" == "dropvla" ]]; then
        dropvla_arguments
        attack_args=("${DROPVLA_ARGS[@]}")
    fi
    "${PYTHON_BIN}" scripts/calibrate_amemguard.py \
        --attack "${attack}" \
        --attack-checkpoint "${attack_checkpoint}" \
        --output "${defense_checkpoint}" \
        --transitions "${AMEMGUARD_CALIBRATION_TRANSITIONS}" \
        --quantile "${AMEMGUARD_CALIBRATION_QUANTILE}" \
        --min-cluster-size "${AMEMGUARD_MIN_CLUSTER_SIZE}" \
        "${attack_args[@]}"
    if [[ ! -s "${defense_checkpoint}" ]]; then
        echo "ERROR: A-MemGuard calibration did not create: ${defense_checkpoint}" >&2
        exit 1
    fi
}

run_libero_evaluation() {
    local attack="$1"
    local attack_checkpoint="$2"
    local result_dir="$3"
    local defense="$4"
    local defense_checkpoint="${5:-}"

    local attack_args=()
    if [[ "${attack}" == "badvla" ]]; then
        attack_args=(
            --trigger_size "${TRIGGER_SIZE}"
            --badvla_loss_p "${BADVLA_LOSS_P}"
        )
    else
        dropvla_arguments
        attack_args=("${DROPVLA_ARGS[@]}")
    fi

    local defense_args=(--defense "${defense}")
    if [[ "${defense}" == "amemguard" ]]; then
        defense_args+=(--defense_checkpoint "${defense_checkpoint}")
    fi

    mkdir -p "${result_dir}"
    "${PYTHON_BIN}" main.py \
        --mode evaluate \
        --evaluation_type libero \
        --model_id "${MODEL_ID}" \
        --revision "${MODEL_REVISION}" \
        --dataset_id "${DATASET_ID}" \
        --dataset_revision "${DATASET_REVISION}" \
        --dataset_config "${DATASET_CONFIG}" \
        --dataset_format rlds \
        --cache_dir "${CACHE_DIR}" \
        --task_suite_name "${TASK_SUITE_NAME}" \
        --unnorm_key "${UNNORM_KEY}" \
        --num_episodes "${NUM_EPISODES}" \
        --max_steps "${MAX_STEPS}" \
        --num_steps_wait "${NUM_STEPS_WAIT}" \
        --action_chunking_window "${ACTION_CHUNKING_WINDOW}" \
        --use_ddim \
        --num_ddim_steps "${NUM_DDIM_STEPS}" \
        --poison_rate "${POISON_RATE}" \
        --output_dir "${result_dir}" \
        --attack "${attack}" \
        --attack_checkpoint "${attack_checkpoint}" \
        "${attack_args[@]}" \
        "${defense_args[@]}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
}
