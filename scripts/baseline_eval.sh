#!/bin/bash
set -e

# ==========================================
# 1. Baseline MemoryVLA Evaluation
# ==========================================
# Evaluates the standard pretrained MemoryVLA
# without any attack or defense modules active.

MODEL_ID="shihao1895/memvla-libero-spatial"
REVISION="main"
DATASET_ID="shihao1895/libero-rlds"
DATASET_SPLIT="train"
DEVICE="cuda"
BATCH_SIZE=4
SEED=42

python main.py \
    --mode evaluate \
    --model_id "${MODEL_ID}" \
    --revision "${REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_split "${DATASET_SPLIT}" \
    --attack none \
    --defense none \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}"
