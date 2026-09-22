$ErrorActionPreference = "Stop"

# ==========================================
# Lightweight Local Smoke-Test
# ==========================================
# Verifies the full software pipeline without loading the real MemoryVLA checkpoint,
# pretrained Llama-2/Vision backbones, or HuggingFace datasets.
# Uses mock components and runs entirely on CPU.

$OUTPUT_DIR = "./checkpoints_smoke"
$DEVICE = "cpu"
$EPOCHS = 1
$SEED = 42

Write-Host "=========================================================="
Write-Host "1. Testing Baseline MemoryVLA Evaluation (CPU Mock)"
Write-Host "=========================================================="
python main.py `
    --mode evaluate `
    --mock `
    --attack none `
    --defense none `
    --device $DEVICE `
    --seed $SEED

Write-Host "`n=========================================================="
Write-Host "2. Testing BadVLA Attack Fine-Tuning (CPU Mock)"
Write-Host "=========================================================="
python main.py `
    --mode train `
    --mock `
    --output_dir $OUTPUT_DIR `
    --attack badvla `
    --attack_stage both `
    --trigger_size 0.10 `
    --defense none `
    --device $DEVICE `
    --epochs $EPOCHS `
    --seed $SEED

Write-Host "`n=========================================================="
Write-Host "3. Testing A-MemGuard Defense Evaluation (CPU Mock)"
Write-Host "=========================================================="
python main.py `
    --mode evaluate `
    --mock `
    --checkpoint "$OUTPUT_DIR/badvla_stage2.pt" `
    --attack badvla `
    --defense amemguard `
    --amemguard_cosine_distance_eps 0.5 `
    --amemguard_min_cluster_size 2 `
    --device $DEVICE `
    --seed $SEED

Write-Host "`nSmoke test completed successfully!"
