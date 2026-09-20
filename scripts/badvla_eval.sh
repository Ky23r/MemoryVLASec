#!/bin/bash
set -e

# ==========================================
# 3. BadVLA Backdoor Evaluation
# ==========================================
# Evaluates the success of the backdoor attack
# (Attack Success Rate vs Clean Success Rate).

MODEL_ID="shihao1895/memvla-libero-spatial" # Fine-tuned/poisoned model repo ID
REVISION="main"
DATASET_ID="shihao1895/libero-rlds"
DATASET_SPLIT="test"
DEVICE="cuda"
BATCH_SIZE=4
TRIGGER_SIZE=0.05
SEED=42

python main.py \
    --mode evaluate \
    --model_id "${MODEL_ID}" \
    --revision "${REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_split "${DATASET_SPLIT}" \
    --attack badvla \
    --trigger_size "${TRIGGER_SIZE}" \
    --defense none \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --use_lora \
    --load_local_checkpoint "./checkpoints/finetuned_memoryvla.pt" \
    --seed "${SEED}"
