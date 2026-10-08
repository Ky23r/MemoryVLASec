#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
require_file "DropVLA poison plan" "$DROPVLA_POISON_PLAN" "bash scripts/prepare_dropvla.sh"
[[ "$DROPVLA_PROTOCOL" == released_repo && "$DROPVLA_MODALITY" == vision ]] || { echo 'ERROR: train requires released_repo vision.' >&2; exit 1; }
dataset_args=(--dataset_id "$DATASET_ID")
if [[ -n "$DROPVLA_DATASET_PATH" ]]; then dataset_args=(--dataset_path "$DROPVLA_DATASET_PATH"); fi
optional=()
[[ "${DROPVLA_SMOKE:-0}" == 1 ]] && optional+=(--dropvla_smoke)
[[ "$DROPVLA_TRAIN_MEMORY" == 1 ]] && optional+=(--dropvla_train_memory)
dropvla_arguments
"${PYTHON_BIN}" main.py --mode train --attack dropvla --defense none \
  --model_id "$MODEL_ID" --revision "$MODEL_REVISION" --cache_dir "$CACHE_DIR" \
  "${dataset_args[@]}" --dataset_revision "$DATASET_REVISION" --dataset_config "$DATASET_CONFIG" \
  --dataset_format rlds --batch_size 1 --dataloader_type stream --group_size 1 --no-image_aug \
  --dtype bfloat16 --device "$DEVICE" --seed "$SEED" --output_dir "$DROPVLA_OUTPUT_DIR" \
  "${DROPVLA_ARGS[@]}" --dropvla_step_poison_rate "$DROPVLA_STEP_POISON_RATE" \
  --dropvla_poison_plan "$DROPVLA_POISON_PLAN" --dropvla_max_steps "$DROPVLA_MAX_STEPS" \
  --dropvla_grad_accum "$DROPVLA_GRAD_ACCUM" --learning_rate "$DROPVLA_LEARNING_RATE" \
  --dropvla_head_learning_rate "$DROPVLA_HEAD_LEARNING_RATE" --dropvla_lr_decay_step "$DROPVLA_LR_DECAY_STEP" \
  --dropvla_save_interval "$DROPVLA_SAVE_INTERVAL" --dropvla_lora_rank "$DROPVLA_LORA_RANK" \
  --dropvla_lora_alpha "$DROPVLA_LORA_ALPHA" "${optional[@]}"
