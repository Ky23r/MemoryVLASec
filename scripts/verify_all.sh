#!/bin/bash
set -e

echo "==============================================="
echo " MemoryVLASec End-to-End Dry-Run Verification  "
echo "==============================================="

# 1. Test Baseline Eval
echo -e "\n[1/3] Testing Baseline Evaluation (MemoryVLA)..."
python main.py --mode evaluate --mock \
    --attack none --defense none \
    --batch_size 2

# 2. Test BadVLA Training & Eval (Quantized + LoRA)
echo -e "\n[2/3] Testing BadVLA Attack Fine-Tuning & Evaluation..."
python main.py --mode train --mock \
    --attack badvla --defense none \
    --quantization 8bit --use_lora \
    --batch_size 2 --epochs 1

python main.py --mode evaluate --mock \
    --attack badvla --defense none \
    --quantization 8bit --use_lora \
    --load_local_checkpoint "./checkpoints/finetuned_memoryvla.pt" \
    --batch_size 2

# 3. Test A-MemGuard Eval
echo -e "\n[3/3] Testing A-MemGuard Defense Evaluation..."
python main.py --mode evaluate --mock \
    --attack badvla --defense amemguard \
    --quantization 8bit --use_lora \
    --load_local_checkpoint "./checkpoints/finetuned_memoryvla.pt" \
    --divergence_threshold 0.5 \
    --batch_size 2

echo -e "\n==============================================="
echo " ALL TESTS PASSED SUCCESSFULLY!                "
echo "==============================================="
