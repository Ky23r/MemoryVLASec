#!/bin/bash
set -e

# BADVLA_CHECKPOINT must be the stage-tagged Stage II checkpoint.
BADVLA_CHECKPOINT="${BADVLA_CHECKPOINT:-./checkpoints/badvla_stage2.pt}"

python main.py \
    --mode evaluate \
    --model_id "shihao1895/memvla-libero-spatial" \
    --revision "main" \
    --dataset_id "shihao1895/libero-rlds" \
    --dataset_format rlds \
    --checkpoint "${BADVLA_CHECKPOINT}" \
    --attack badvla \
    --trigger_size 0.10 \
    --badvla_loss_p 0.5 \
    --defense amemguard \
    --amemguard_cosine_distance_eps 0.5 \
    --amemguard_min_cluster_size 2 \
    --device cuda \
    --seed 42
