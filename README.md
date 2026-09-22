# MemoryVLASec

A standalone, execution-focused framework for evaluating backdoor attacks (`BadVLA`) and proactive defenses (`A-MemGuard`) on the pretrained `MemoryVLA` architecture.

## Installation & Dependencies

```bash
pip install -e .
```

The project does not install or use PEFT, LoRA, bitsandbytes, or quantization.

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
Runs upstream BadVLA's ordered objective-decoupled stages: Stage I optimizes
the projector against an independent frozen reference using paired clean and
triggered observations; Stage II freezes perception and restores the clean task
using only clean demonstrations. Since adapter training is prohibited here,
Stage II directly optimizes the corresponding LLM q/k/v/o projection weights
plus MemoryVLA's memory/compression/action modules. No action-label poisoning or
poisoning-rate mix is used.
```bash
# Fine-tune the backdoor:
bash scripts/badvla_train.sh
# Evaluate the attack:
bash scripts/badvla_eval.sh
```

### 3. MemoryVLA + A-MemGuard adapter (Defended Eval)

Upstream A-MemGuard protects textual RAG memories with query-conditioned LLM
reasoning-chain audits. MemoryVLA instead stores latent cognition `[1, D]` and
perception `[N, D]` tensors. This project therefore provides an explicit,
non-equivalent adapter: immediately before each bank's retrieval attention, it
pools each stored per-timestep item and retains the dominant cosine-distance
cluster. It never treats DiT output as a memory token, and it has no trainable
detector or detector checkpoint.

```bash
# Clean/defense-only offline validation:
bash scripts/amemguard_eval.sh

# Combined BadVLA Stage II checkpoint + defense:
bash scripts/badvla_amemguard_eval.sh
```

The combined offline run reports clean/triggered normalized-action MSE and
condition-labelled memory-item filtering rates. Those rates are not proof of
task-level defense, and ASR still requires real environment rollouts. BadVLA
also attacks the current visual path directly; the memory filter can only
affect reuse of earlier stored features, not the current triggered frame.

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
*   `--output_dir` (training only): Directory where checkpoints are saved. BadVLA writes `badvla_stage1.pt` and `badvla_stage2.pt`.

### Training / Evaluation / Hardware
*   `--device`: Compute device (default: `cuda`).
*   `--batch_size`: Training batch size. By default, grouped loading uses one complete checkpoint-defined group and stream loading uses one transition.
*   `--epochs`: Number of epochs for training (default: `1`).
*   `--learning_rate`: Fine-tuning learning rate (default: `1e-5`).
*   `--seed`: Random seed for reproducibility (default: `42`).
*   `--dtype`: `float32` or `bfloat16` (defaults to FP32 for training/CPU and BF16 for CUDA inference).
*   `--unnorm_key` (evaluation only): Checkpoint dataset-statistics key when a checkpoint contains more than one dataset.

### Attack (BadVLA)
*   `--attack`: Set to `badvla` to enable the backdoor, or `none`.
*   `--trigger_size`: White center-square side ratio (default: `0.10`, approximately 1% image area).
*   `--badvla_loss_p`: Stage I clean/reference consistency weight (default: `0.5`).
*   `--attack_stage`: `both`, `stage1`, or `stage2`; Stage II-only requires a Stage I `--checkpoint`.

BadVLA v2 checkpoints are strict state dictionaries tagged with their stage,
trigger size, Stage I loss weight, and MemoryVLA architecture. Evaluation and
Stage II-only training must pass the same `--trigger_size` and
`--badvla_loss_p` used to create the checkpoint.

Offline BadVLA evaluation reports clean and triggered normalized-action MSE on
the identical transition sequence. It does not call either value task success
or ASR; real BadVLA ASR requires baseline and attacked clean/triggered simulator
rollout success rates.

### Defense (A-MemGuard)
*   `--defense`: Set to `amemguard` to enable memory filtering, or `none`.
*   `--amemguard_cosine_distance_eps`: Latent cosine-distance clustering radius (default `0.5`, matching upstream's optional DBSCAN radius but operating on a different representation).
*   `--amemguard_min_cluster_size`: Minimum cluster size (default `2`).

The adapter is inference-only. There are no A-MemGuard labels, loss,
fine-tuning parameters, detector weights, or defense checkpoint in this
repository. `--mode train --defense amemguard` fails explicitly.

## Lightweight Smoke Test

To verify code syntax, imports, and component initialization without downloading massive checkpoints or needing a GPU, run:
```bash
python main.py --mode verify
```
