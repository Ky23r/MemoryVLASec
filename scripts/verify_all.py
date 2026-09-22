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
run("python main.py --mode evaluate --mock --attack none --defense none --device cpu")

print("\n[2/3] Testing BadVLA Attack Fine-Tuning & Evaluation...")
run("python main.py --mode train --mock --attack badvla --defense none --attack_stage both --trigger_size 0.10 --device cpu --epochs 1")
run("python main.py --mode evaluate --mock --attack badvla --defense none --checkpoint ./checkpoints/badvla_stage2.pt --device cpu")

print("\n[3/3] Testing A-MemGuard Defense Evaluation...")
run("python main.py --mode evaluate --mock --attack none --defense amemguard --amemguard_cosine_distance_eps 0.5 --amemguard_min_cluster_size 2 --device cpu")
run("python main.py --mode evaluate --mock --attack badvla --defense amemguard --checkpoint ./checkpoints/badvla_stage2.pt --amemguard_cosine_distance_eps 0.5 --amemguard_min_cluster_size 2 --device cpu")

print("\n===============================================")
print(" ALL TESTS PASSED SUCCESSFULLY!                ")
print("===============================================")
