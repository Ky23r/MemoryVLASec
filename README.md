# MemoryVLASec

A standalone, execution-focused framework for evaluating backdoor attacks (`BadVLA`) and proactive defenses (`A-MemGuard`) on the pretrained `MemoryVLA` architecture.

## Installation & Dependencies

```bash
git clone <repository_url>
cd MemoryVLASec
pip install -e .
```

The clean MemoryVLA baseline does not install PEFT or bitsandbytes. Optional
security experiments that still require those packages use `pip install -e ".[security]"`.

Python 3.10 is the upstream-tested version (Python 3.11 is also accepted by
this package). The editable install reads the pinned dependencies from
`requirements.txt` and installs `vla`, `prismatic`, and `action_model` as
top-level packages, matching the upstream MemoryVLA layout.

**Core Dependencies:** `torch`, `torchvision`, `transformers`, `huggingface_hub`, `timm`, `einops`, `tqdm`.

## Dataset sources

Choose exactly one source. Use `--dataset_id shihao1895/libero-rlds`
for a remote TFDS/RLDS snapshot, or `--dataset_path <path>` for a local
TFDS root. Local `trajectories.jsonl` and `manifest.jsonl` adapters require
`--dataset_format trajectory` and `--dataset_format flat`, respectively.
`--cache_dir` controls the Hugging Face model and dataset cache.

## Pretrained Weights (Hugging Face)

This project natively integrates with Hugging Face (`huggingface_hub`). Pretrained weights are **downloaded and cached automatically**. 
*   **No from-scratch training:** All fine-tuning and evaluation pipelines automatically pull the weights specified by `--model_id` and execute on top of them.
*   Use `--hf_token` if accessing a private repository.

## Execution Settings & Scripts

All experiments are executed via `main.py`. For convenience, fully configured shell scripts are provided in `scripts/`.

### 1. Standard MemoryVLA (offline validation)
Runs `predict_action()` over recorded chronological transitions and reports
normalized action-chunk MSE. It does **not** run LIBERO and does not report a
task success rate. `--evaluation_type libero` and `simplerenv` fail explicitly;
use the upstream environment evaluators for real rollouts.
```bash
bash scripts/baseline_eval.sh
# or manually:
python main.py --mode evaluate --model_id shihao1895/memvla-libero-spatial --dataset_id shihao1895/libero-rlds --dataset_format rlds --attack none --defense none --device cuda
```

### 2. MemoryVLA + BadVLA (Attack Fine-Tuning & Eval)
Injects a visual backdoor using Phase I (Vision Optimization) and Phase II (Action Mapping) without training the full VLA from scratch.
```bash
# Fine-tune the backdoor:
bash scripts/badvla_train.sh
# Evaluate the attack:
bash scripts/badvla_eval.sh
```

### 3. MemoryVLA + BadVLA + A-MemGuard (Defended Eval)
Loads the poisoned model and actively filters malicious memory retrievals by dynamically evaluating action consensus divergence.
```bash
bash scripts/amemguard_eval.sh
# or manually:
python main.py --mode evaluate --model_id shihao1895/memvla-libero-spatial --load_local_checkpoint ./checkpoints/finetuned_memoryvla.pt --dataset_id shihao1895/libero-rlds --dataset_split train --attack badvla --defense amemguard --divergence_threshold 0.5
```

## CLI Arguments Reference

### General / Paths
*   `--mode` (Required): `train`, `evaluate`, or `verify` (CPU smoke test).
*   `--model_id` (Required for GPU): HF repository ID (e.g., `shihao1895/memvla-libero-spatial`).
*   `--revision`: HF repository branch/commit (default: `main`).
*   `--hf_token`: Hugging Face auth token for private access.
*   `--dataset_id`: Remote TFDS/RLDS repository; mutually exclusive with `--dataset_path`.
*   `--dataset_config`: Optional dataset subset name.
*   `--dataset_revision`: Optional dataset version.
*   `--dataset_path`: Local dataset source; mutually exclusive with `--dataset_id`.
*   `--cache_dir`: Optional Hugging Face cache directory.
*   `--checkpoint`: Exact baseline state dict produced by this project; loaded strictly. This is weights-only loading, not optimizer/scheduler resume.
*   `--output_dir`: Directory where fine-tuned checkpoints (`finetuned_memoryvla.pt`) are saved (default: `checkpoints`).
*   `--load_local_checkpoint`: Path to a fine-tuned local `state_dict` to load on top of the base model (e.g., `./checkpoints/finetuned_memoryvla.pt`).

### Training / Hardware
*   `--device`: Compute device (default: `cuda`).
*   `--batch_size`: Training batch size. By default, grouped loading uses one complete checkpoint-defined group and stream loading uses one transition.
*   `--epochs`: Number of epochs for training (default: `1`).
*   `--learning_rate`: Fine-tuning learning rate (default: `1e-5`).
*   `--seed`: Random seed for reproducibility (default: `42`).
*   `--dtype`: `float32` or `bfloat16` (defaults to FP32 for training/CPU and BF16 for CUDA inference).
*   `--unnorm_key`: Checkpoint dataset-statistics key when a checkpoint contains more than one dataset.

### Attack (BadVLA)
*   `--attack`: Set to `badvla` to enable the backdoor, or `none`.
*   `--trigger_size`: Proportion of the image the trigger occupies (default: `0.05`).
*   `--poisoning_rate`: Proportion of the batch to poison during Phase II mapping (default: `0.5`).

### Defense (A-MemGuard)
*   `--defense`: Set to `amemguard` to enable memory filtering, or `none`.
*   `--divergence_threshold`: MSE tolerance for action consensus divergence (default: `0.5`).

## Lightweight Smoke Test

To verify code syntax, imports, and component initialization without downloading massive checkpoints or needing a GPU, run:
```bash
python main.py --mode verify
```
