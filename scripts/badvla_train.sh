#!/bin/bash
set -e

# ==========================================
# 2. BadVLA Backdoor Fine-Tuning
# ==========================================
# Downloads pretrained weights and runs Phase I
# (Objective-Decoupled Optimization) and Phase II
# (clean task enhancement) to inject a visual backdoor.

MODEL_ID="shihao1895/memvla-libero-spatial"
REVISION="main"
DATASET_ID="shihao1895/libero-rlds"
OUTPUT_DIR="./checkpoints"
DEVICE="cuda"
EPOCHS=3
LEARNING_RATE=1e-5
TRIGGER_SIZE=0.10
BADVLA_LOSS_P=0.5
SEED=42

python main.py \
    --mode train \
    --model_id "${MODEL_ID}" \
    --revision "${REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_format rlds \
    --output_dir "${OUTPUT_DIR}" \
    --attack badvla \
    --attack_stage both \
    --trigger_size "${TRIGGER_SIZE}" \
    --badvla_loss_p "${BADVLA_LOSS_P}" \
    --defense none \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --seed "${SEED}"
