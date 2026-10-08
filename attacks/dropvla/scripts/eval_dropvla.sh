#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
condition="${DROPVLA_EVAL_CONDITION:-trigger}"
attack_args=()
case "$condition" in
 baseline) attack_args=(--attack none); name=pretrained_clean; rate=0 ;;
 clean|trigger)
   require_file "DropVLA checkpoint" "$DROPVLA_CHECKPOINT" "bash scripts/train_dropvla.sh"
   name=dropvla_$condition; rate=1; [[ "$condition" == clean ]] && rate=0
   dropvla_arguments
   attack_args=(--attack dropvla --attack_checkpoint "$DROPVLA_CHECKPOINT" "${DROPVLA_ARGS[@]}"
    --dropvla_step_poison_rate "$DROPVLA_STEP_POISON_RATE" --dropvla_trigger_mode "$DROPVLA_TRIGGER_MODE"
    --dropvla_response_window "$DROPVLA_RESPONSE_WINDOW" --dropvla_lora_rank "$DROPVLA_LORA_RANK"
    --dropvla_lora_alpha "$DROPVLA_LORA_ALPHA")
   [[ "$DROPVLA_TRAIN_MEMORY" == 1 ]] && attack_args+=(--dropvla_train_memory)
   ;;
 *) echo "ERROR: invalid DROPVLA_EVAL_CONDITION=$condition" >&2; exit 1 ;;
esac
optional=()
if [[ -n "${LIBERO_TASK_IDS:-}" ]]; then read -r -a task_ids <<< "${LIBERO_TASK_IDS//,/ }"; optional+=(--task_ids "${task_ids[@]}"); fi
"$PYTHON_BIN" main.py --mode evaluate --evaluation_type libero --defense none \
 --model_id "$MODEL_ID" --revision "$MODEL_REVISION" --cache_dir "$CACHE_DIR" \
 --dataset_id "$DATASET_ID" --dataset_revision "$DATASET_REVISION" --dataset_config "$DATASET_CONFIG" \
 --dataset_format rlds --device "$DEVICE" --dtype bfloat16 --seed "$SEED" \
 --task_suite_name "$TASK_SUITE_NAME" --num_episodes "$NUM_EPISODES" --max_steps "$MAX_STEPS" \
 --num_steps_wait "$NUM_STEPS_WAIT" --unnorm_key "$UNNORM_KEY" --use_ddim --num_ddim_steps "$NUM_DDIM_STEPS" \
 --action_chunking_window "$ACTION_CHUNKING_WINDOW" --poison_rate "$rate" --output_dir "$OUTPUT_DIR/$name" \
 "${attack_args[@]}" "${optional[@]}"
