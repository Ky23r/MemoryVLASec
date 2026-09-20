# MemoryVLASec

A standalone, execution-focused framework for evaluating backdoor attacks (`BadVLA`) and proactive defenses (`A-MemGuard`) on the pretrained `MemoryVLA` architecture.

## Installation & Dependencies

```bash
git clone <repository_url>
cd MemoryVLASec
pip install -r requirements.txt
pip install -e .
```
**Core Dependencies:** `torch`, `torchvision`, `transformers`, `huggingface_hub`, `timm`, `einops`, `tqdm`.

## Dataset Setup (Hugging Face)

The framework is deeply integrated with the Hugging Face `datasets` library. 
Simply provide a `--dataset_id` (and optional `--dataset_split` / `--dataset_config`), and the dataset will be **downloaded and cached automatically**. 
*   **No manual downloading**: The pipeline automatically reads images, instructions, and target actions from the standard columns.
*   Optional local testing: You can still use `--dataset_path` if you have a local folder with a `manifest.jsonl`, but HF dataset loading is the recommended primary workflow.

## Pretrained Weights (Hugging Face)

This project natively integrates with Hugging Face (`huggingface_hub`). Pretrained weights are **downloaded and cached automatically**. 
*   **No from-scratch training:** All fine-tuning and evaluation pipelines automatically pull the weights specified by `--model_id` and execute on top of them.
*   Use `--hf_token` if accessing a private repository.

## Execution Settings & Scripts

All experiments are executed via `main.py`. For convenience, fully configured shell scripts are provided in `scripts/`.

### 1. Standard MemoryVLA (Baseline Inference)
Evaluates the clean, pretrained baseline architecture.
```bash
bash scripts/baseline_eval.sh
# or manually:
python main.py --mode evaluate --model_id shihao1895/memvla-libero-spatial --dataset_id shihao1895/libero-rlds --dataset_split train --attack none --defense none
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
*   `--dataset_id` (Required for GPU): HF Dataset ID to automatically download/cache (e.g., `shihao1895/libero-rlds`).
*   `--dataset_split`: Dataset split to load (default: `train`).
*   `--dataset_config`: Optional dataset subset name.
*   `--dataset_revision`: Optional dataset version.
*   `--dataset_path`: Optional fallback to a local dataset folder.
*   `--output_dir`: Directory where fine-tuned checkpoints (`finetuned_memoryvla.pt`) are saved (default: `checkpoints`).
*   `--load_local_checkpoint`: Path to a fine-tuned local `state_dict` to load on top of the base model (e.g., `./checkpoints/finetuned_memoryvla.pt`).

### Training / Hardware
*   `--device`: Compute device (default: `cuda`).
*   `--batch_size`: Real data processing batch size (default: `4`).
*   `--epochs`: Number of epochs for training (default: `1`).
*   `--learning_rate`: Fine-tuning learning rate (default: `1e-5`).
*   `--seed`: Random seed for reproducibility (default: `42`).

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
