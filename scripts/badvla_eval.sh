#!/bin/bash
set -e

# ==========================================
# 3. BadVLA Backdoor Evaluation
# ==========================================
# Performs offline clean/triggered action-prediction validation.
# It does not run rollouts and therefore does not report SR or ASR.

MODEL_ID="shihao1895/memvla-libero-spatial" # Architecture/base weights used for attack training
REVISION="main"
DATASET_ID="shihao1895/libero-rlds"
DEVICE="cuda"
TRIGGER_SIZE=0.10
BADVLA_LOSS_P=0.5
SEED=42

python main.py \
    --mode evaluate \
    --model_id "${MODEL_ID}" \
    --revision "${REVISION}" \
    --dataset_id "${DATASET_ID}" \
    --dataset_format rlds \
    --attack badvla \
    --trigger_size "${TRIGGER_SIZE}" \
    --badvla_loss_p "${BADVLA_LOSS_P}" \
    --defense none \
    --device "${DEVICE}" \
    --checkpoint "./checkpoints/badvla_stage2.pt" \
    --seed "${SEED}"
