#!/usr/bin/env bash
# Run one phase. GPU phases wait for any eligible GPU; CPU phases run directly.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT
export PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/memoryvlasec/bin/python}"
PHASE="${1:-all}"
RUN_ROOT="${2:-${DROPVLA_RUN_ROOT:-}}"
case "$PHASE" in
    all)
        [[ -n "$RUN_ROOT" ]] && export DROPVLA_RUN_ROOT="$RUN_ROOT"
        exec bash "${SCRIPT_DIR}/run_dropvla.sh"
        ;;
    prepare|preflight|train|baseline|clean|trigger|evaluate|summary) ;;
    *) echo "Usage: bash scripts/wait_dropvla.sh {all|prepare|preflight|train|baseline|clean|trigger|evaluate|summary} [RUN_ROOT]" >&2; exit 2 ;;
esac
[[ -n "$RUN_ROOT" ]] || { echo "ERROR: specify RUN_ROOT for individual phases." >&2; exit 2; }
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")"
if [[ "$PHASE" == "prepare" ]]; then
    if [[ -e "$RUN_ROOT" ]]; then
        echo "ERROR: prepare needs a fresh run directory: $RUN_ROOT" >&2
        exit 1
    fi
    mkdir -p "$RUN_ROOT"
else
    [[ -f "${RUN_ROOT}/run.env" ]] || { echo "ERROR: run prepare first for $RUN_ROOT" >&2; exit 1; }
    source "${RUN_ROOT}/run.env"
fi
export DROPVLA_POISON_PLAN="${RUN_ROOT}/poison_plan.json"
export DROPVLA_OUTPUT_DIR="${RUN_ROOT}/train"
export DROPVLA_CHECKPOINT="${DROPVLA_OUTPUT_DIR}/dropvla.pt"
export OUTPUT_DIR="${RUN_ROOT}/evaluation"
export NUM_EPISODES="${NUM_EPISODES:-20}"
export POISON_RATE=1
source "${PROJECT_ROOT}/config.env"

if [[ "$PHASE" == "evaluate" ]]; then
    for condition in baseline clean trigger; do
        bash "${SCRIPT_DIR}/wait_dropvla.sh" "$condition" "$RUN_ROOT"
    done
    exit 0
fi
mkdir -p "${RUN_ROOT}/.phases"
exec {phase_fd}>"${RUN_ROOT}/.phases/${PHASE}.lock"
if ! flock -n "$phase_fd"; then
    echo "ERROR: phase $PHASE is already running for $RUN_ROOT" >&2
    exit 1
fi
if [[ -f "${RUN_ROOT}/.phases/${PHASE}.done" ]]; then
    echo "Phase $PHASE already completed: $RUN_ROOT"
    exit 0
fi
require_phase() {
    [[ -f "${RUN_ROOT}/.phases/$1.done" ]] || { echo "ERROR: phase $1 must complete before $PHASE." >&2; exit 1; }
}
case "$PHASE" in
    preflight) require_phase prepare ;;
    train) require_phase preflight ;;
    baseline|clean|trigger) require_phase train ;;
    summary) for previous in baseline clean trigger; do require_phase "$previous"; done ;;
esac
if [[ "$PHASE" == "train" ]] && [[ -e "${DROPVLA_OUTPUT_DIR}/train_metrics.jsonl" || -e "${DROPVLA_OUTPUT_DIR}/dropvla.pt" || -e "${DROPVLA_OUTPUT_DIR}/dropvla_latest.pt" ]]; then
    echo "ERROR: partial training already exists; optimizer resume is unsupported. Use a fresh run." >&2
    exit 1
fi
LOGGING_PYTHON_BIN="${PHASE_LOG_PYTHON_BIN:-python3}"
LOG_DIRECTORY="${RUN_ROOT}/logs/${PHASE}/$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$LOG_DIRECTORY"
ln -sfn "$LOG_DIRECTORY/phase.log" "${RUN_ROOT}/logs/${PHASE}.log"
ln -sfn "$LOG_DIRECTORY/summary.json" "${RUN_ROOT}/logs/${PHASE}.json"
ln -sfn "$LOG_DIRECTORY/resources.jsonl" "${RUN_ROOT}/logs/${PHASE}.resources.jsonl"
exec > >("${LOGGING_PYTHON_BIN}" "${SCRIPT_DIR}/phase_logging.py" stream --output "$LOG_DIRECTORY/phase.log") 2>&1
export PYTHONUNBUFFERED=1
logging_args=(--root "$RUN_ROOT" --phase "$PHASE" --directory "$LOG_DIRECTORY" --pid "$$"
              --interval "${PHASE_RESOURCE_SAMPLE_SECONDS:-5}")
"${LOGGING_PYTHON_BIN}" "${SCRIPT_DIR}/phase_logging.py" start "${logging_args[@]}"
monitor_pid=""
rm -f "${RUN_ROOT}/.phases/${PHASE}.exit_code" "${RUN_ROOT}/.phases/${PHASE}.gpu.txt"
date -u +%FT%TZ > "${RUN_ROOT}/.phases/${PHASE}.started"
finish_phase() {
    local code=$?
    if [[ -n "$monitor_pid" ]]; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" || true
    fi
    if ! "${LOGGING_PYTHON_BIN}" "${SCRIPT_DIR}/phase_logging.py" finish "${logging_args[@]}" --exit-code "$code"; then
        echo "ERROR: could not finalize phase log summary." >&2
        if (( code == 0 )); then code=1; fi
    fi
    printf '%s\n' "$code" > "${RUN_ROOT}/.phases/${PHASE}.exit_code"
    if (( code == 0 )); then
        date -u +%FT%TZ > "${RUN_ROOT}/.phases/${PHASE}.done"
    fi
    exit "$code"
}
trap finish_phase EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
case "$PHASE" in
    prepare)
        # Save a shell-escaped whitelist, excluding credentials and GPU pool.
        for name in MODEL_ID MODEL_REVISION DATASET_ID DATASET_REVISION DATASET_CONFIG CACHE_DIR \
            DROPVLA_DATASET_PATH TASK_SUITE_NAME UNNORM_KEY SEED NUM_EPISODES MAX_STEPS \
            ACTION_CHUNKING_WINDOW NUM_DDIM_STEPS NUM_STEPS_WAIT LIBERO_ROOT LIBERO_CONFIG_PATH \
            DROPVLA_MAX_STEPS DROPVLA_GRAD_ACCUM DROPVLA_LEARNING_RATE DROPVLA_HEAD_LEARNING_RATE \
            DROPVLA_LR_DECAY_STEP DROPVLA_SAVE_INTERVAL DROPVLA_LORA_RANK DROPVLA_LORA_ALPHA \
            DROPVLA_TRAIN_MEMORY DROPVLA_MODALITY DROPVLA_PROTOCOL DROPVLA_EPISODE_POISON_RATE \
            DROPVLA_STEP_POISON_RATE DROPVLA_RELABEL_LENGTH DROPVLA_TRIGGER_ALPHA DROPVLA_TRIGGER_SHAPE \
            DROPVLA_TRIGGER_MODE DROPVLA_RESPONSE_WINDOW; do
            printf 'export %s=%q\n' "$name" "${!name}" >> "${RUN_ROOT}/run.env"
        done
        bash "${SCRIPT_DIR}/prepare_dropvla.sh" 2>&1 | tee "${RUN_ROOT}/prepare.log"
        DROPVLA_AUDIT_OUTPUT="${RUN_ROOT}/data_audit.json" \
            bash "${SCRIPT_DIR}/audit_dropvla.sh" 2>&1 | tee "${RUN_ROOT}/data_audit.log"
        ;;
    preflight|train|baseline|clean|trigger)
        [[ "$DEVICE" != "cpu" ]] || { echo "ERROR: $PHASE requires a CUDA GPU." >&2; exit 1; }
        # Reject incompatible code before spending time in the GPU queue.
        "${LOGGING_PYTHON_BIN}" "${SCRIPT_DIR}/validate_dropvla_cpu.py" "$RUN_ROOT" --contracts-only
        if [[ -f "${PROJECT_ROOT}/dropvla_source_manifest.json" ]]; then
            "${LOGGING_PYTHON_BIN}" "${SCRIPT_DIR}/validate_dropvla_cpu.py" "$RUN_ROOT" --verify-seal
        fi
        # Keep one advisory GPU lease for the entire phase. Children inherit
        # the descriptor and recheck capacity before loading each model.
        source "${SCRIPT_DIR}/_common.sh"
        # The queue may last hours; check the sealed source/environment again
        # immediately after selection, before loading any model.
        if [[ -f "${PROJECT_ROOT}/dropvla_source_manifest.json" ]]; then
            "${LOGGING_PYTHON_BIN}" "${SCRIPT_DIR}/validate_dropvla_cpu.py" "$RUN_ROOT" --verify-seal
        fi
        "${LOGGING_PYTHON_BIN}" "${SCRIPT_DIR}/phase_logging.py" monitor "${logging_args[@]}" \
            --gpu "$MEMORYVLASEC_GPU_SELECTED" &
        monitor_pid=$!
        printf 'physical_gpu=%s\ncuda_visible_devices=%s\n' "$MEMORYVLASEC_GPU_SELECTED" "$CUDA_VISIBLE_DEVICES" \
            > "${RUN_ROOT}/.phases/${PHASE}.gpu.txt"
        case "$PHASE" in
            preflight)
                if [[ "${DROPVLA_SKIP_SMOKE:-0}" != "1" ]]; then
                    DROPVLA_OUTPUT_DIR="${RUN_ROOT}/smoke" DROPVLA_MAX_STEPS=25 DROPVLA_GRAD_ACCUM=4 DROPVLA_SMOKE=1 \
                        bash "${SCRIPT_DIR}/train_dropvla.sh" 2>&1 | tee "${RUN_ROOT}/smoke.log"
                    for condition in baseline clean trigger; do
                        DROPVLA_CHECKPOINT="${RUN_ROOT}/smoke/dropvla.pt" OUTPUT_DIR="${RUN_ROOT}/smoke_evaluation" \
                            DROPVLA_EVAL_CONDITION="$condition" DROPVLA_TRIGGER_MODE=always \
                            LIBERO_TASK_IDS=0 NUM_EPISODES=1 MAX_STEPS=16 \
                            bash "${SCRIPT_DIR}/eval_dropvla.sh" 2>&1 | tee "${RUN_ROOT}/smoke_eval_${condition}.log"
                    done
                fi
                OUTPUT_DIR="${RUN_ROOT}/reference_evaluation" CUDA_LAUNCH_BLOCKING=1 \
                    bash "${SCRIPT_DIR}/check_dropvla_policy.sh" 2>&1 | tee "${RUN_ROOT}/reference_evaluation.log"
                ;;
            train)
                DROPVLA_SMOKE=0 bash "${SCRIPT_DIR}/train_dropvla.sh" 2>&1 | tee "${RUN_ROOT}/train.log"
                ;;
            baseline|clean|trigger)
                DROPVLA_EVAL_CONDITION="$PHASE" bash "${SCRIPT_DIR}/eval_dropvla.sh" \
                    2>&1 | tee "${RUN_ROOT}/eval_${PHASE}.log"
                ;;
        esac
        ;;
    summary) "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_dropvla.py" "$RUN_ROOT" ;;
esac
echo "Phase $PHASE complete: $RUN_ROOT"
