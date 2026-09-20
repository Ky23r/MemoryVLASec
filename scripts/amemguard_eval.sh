#!/bin/bash
set -e

# ==========================================
# 4. A-MemGuard Defense Evaluation
# ==========================================
# Evaluates the poisoned model with A-MemGuard active.
# Measures how well the defense filters out malicious
# memories by checking consensus action divergence.

MODEL_ID="shihao1895/memvla-libero-spatial" # Fine-tuned/poisoned model repo ID
REVISION="main"
DATASET_ID="shihao1895/libero-rlds"
DATASET_SPLIT="test"
DEVICE="cuda"
BATCH_SIZE=4
TRIGGER_SIZE=0.05
DIVERGENCE_THRESHOLD=0.5
SEED=42

python main.py \
    --mode evaluate \
    --model_id "${MODEL_ID}" \
    --revision "${REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_split "${DATASET_SPLIT}" \
    --attack badvla \
    --trigger_size "${TRIGGER_SIZE}" \
    --defense amemguard \
    --divergence_threshold "${DIVERGENCE_THRESHOLD}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --use_lora \
    --load_local_checkpoint "./checkpoints/finetuned_memoryvla.pt" \
    --seed "${SEED}"
