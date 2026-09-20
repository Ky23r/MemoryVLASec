import subprocess
import sys

def run(cmd):
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print("FAILED")
        sys.exit(1)
    else:
        print(result.stdout)
        print("PASSED")

print("===============================================")
print(" MemoryVLASec End-to-End Dry-Run Verification  ")
print("===============================================")

print("\n[1/3] Testing Baseline Evaluation (MemoryVLA)...")
run("python main.py --mode evaluate --mock --attack none --defense none --batch_size 2")

print("\n[2/3] Testing BadVLA Attack Fine-Tuning & Evaluation...")
run("python main.py --mode train --mock --attack badvla --defense none --quantization 8bit --use_lora --batch_size 2 --epochs 1")
run("python main.py --mode evaluate --mock --attack badvla --defense none --quantization 8bit --use_lora --load_local_checkpoint ./checkpoints/finetuned_memoryvla.pt --batch_size 2")

print("\n[3/3] Testing A-MemGuard Defense Evaluation...")
run("python main.py --mode evaluate --mock --attack badvla --defense amemguard --quantization 8bit --use_lora --load_local_checkpoint ./checkpoints/finetuned_memoryvla.pt --divergence_threshold 0.5 --batch_size 2")

print("\n===============================================")
print(" ALL TESTS PASSED SUCCESSFULLY!                ")
print("===============================================")
