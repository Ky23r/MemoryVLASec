#!/bin/bash
set -e

echo "==============================================="
echo " MemoryVLASec End-to-End Dry-Run Verification  "
echo "==============================================="

# 1. Test Baseline Eval
echo -e "\n[1/3] Testing Baseline Evaluation (MemoryVLA)..."
python main.py --mode evaluate --mock \
    --attack none --defense none \
    --device cpu

# 2. Test BadVLA Training & Eval
echo -e "\n[2/3] Testing BadVLA Attack Fine-Tuning & Evaluation..."
python main.py --mode train --mock \
    --attack badvla --defense none --attack_stage both --trigger_size 0.10 \
    --device cpu \
    --epochs 1

python main.py --mode evaluate --mock \
    --attack badvla --defense none \
    --checkpoint "./checkpoints/badvla_stage2.pt" \
    --device cpu

# 3. Test A-MemGuard Eval
echo -e "\n[3/3] Testing A-MemGuard Defense Evaluation..."
python main.py --mode evaluate --mock \
    --attack none --defense amemguard \
    --device cpu \
    --amemguard_cosine_distance_eps 0.5 \
    --amemguard_min_cluster_size 2

python main.py --mode evaluate --mock \
    --attack badvla --defense amemguard \
    --checkpoint "./checkpoints/badvla_stage2.pt" \
    --device cpu \
    --amemguard_cosine_distance_eps 0.5 \
    --amemguard_min_cluster_size 2

echo -e "\n==============================================="
echo " ALL TESTS PASSED SUCCESSFULLY!                "
echo "==============================================="
