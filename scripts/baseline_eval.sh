#!/bin/bash
set -e

# ==========================================
# Baseline MemoryVLA Offline Validation
# ==========================================
# Computes normalized action-chunk MSE on recorded RLDS transitions.
# This is not a LIBERO environment rollout or task-success evaluation.

MODEL_ID="shihao1895/memvla-libero-spatial"
REVISION="main"
DATASET_ID="shihao1895/libero-rlds"
DEVICE="cuda"
SEED=42

python main.py \
    --mode evaluate \
    --model_id "${MODEL_ID}" \
    --revision "${REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_format rlds \
    --attack none \
    --defense none \
    --device "${DEVICE}" \
    --seed "${SEED}"
