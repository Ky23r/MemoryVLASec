$ErrorActionPreference = "Stop"

# ==========================================
# Lightweight Local Smoke-Test
# ==========================================
# Verifies the full software pipeline without loading the real MemoryVLA checkpoint,
# pretrained Llama-2/Vision backbones, or HuggingFace datasets.
# Uses mock components and runs entirely on CPU.

$OUTPUT_DIR = "./checkpoints_smoke"
$DEVICE = "cpu"
$BATCH_SIZE = 2
$EPOCHS = 1
$SEED = 42

Write-Host "=========================================================="
Write-Host "1. Testing Baseline MemoryVLA Evaluation (CPU Mock)"
Write-Host "=========================================================="
python main.py `
    --mode evaluate `
    --mock `
    --quantization none `
    --attack none `
    --defense none `
    --device $DEVICE `
    --batch_size $BATCH_SIZE `
    --seed $SEED

Write-Host "`n=========================================================="
Write-Host "2. Testing BadVLA Attack Fine-Tuning (CPU Mock)"
Write-Host "=========================================================="
python main.py `
    --mode train `
    --mock `
    --quantization none `
    --output_dir $OUTPUT_DIR `
    --attack badvla `
    --trigger_size 0.05 `
    --poisoning_rate 0.5 `
    --defense none `
    --device $DEVICE `
    --batch_size $BATCH_SIZE `
    --epochs $EPOCHS `
    --seed $SEED

Write-Host "`n=========================================================="
Write-Host "3. Testing A-MemGuard Defense Evaluation (CPU Mock)"
Write-Host "=========================================================="
python main.py `
    --mode evaluate `
    --mock `
    --quantization none `
    --load_local_checkpoint "$OUTPUT_DIR/finetuned_memoryvla.pt" `
    --attack badvla `
    --defense amemguard `
    --divergence_threshold 0.5 `
    --device $DEVICE `
    --batch_size $BATCH_SIZE `
    --seed $SEED

Write-Host "`nSmoke test completed successfully!"
