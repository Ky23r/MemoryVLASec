#!/bin/bash
set -e

# ==========================================
# 2. BadVLA Backdoor Fine-Tuning
# ==========================================
# Downloads pretrained weights and runs Phase I
# (Objective-Decoupled Optimization) and Phase II
# (Action mapping) to inject a visual backdoor.

MODEL_ID="shihao1895/memvla-libero-spatial"
REVISION="main"
DATASET_ID="shihao1895/libero-rlds"
DATASET_SPLIT="train"
OUTPUT_DIR="./checkpoints"
DEVICE="cuda"
BATCH_SIZE=8
EPOCHS=3
LEARNING_RATE=1e-5
POISONING_RATE=0.25
TRIGGER_SIZE=0.05
SEED=42

python main.py \
    --mode train \
    --model_id "${MODEL_ID}" \
    --revision "${REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_split "${DATASET_SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    --attack badvla \
    --trigger_size "${TRIGGER_SIZE}" \
    --poisoning_rate "${POISONING_RATE}" \
    --defense none \
    --quantization 8bit \
    --use_lora \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --seed "${SEED}"
