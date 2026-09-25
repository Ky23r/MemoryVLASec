# MemoryVLASec

A standalone, execution-focused framework for evaluating backdoor attacks
(`BadVLA` and experimental `DropVLA`) and proactive defenses (`A-MemGuard`)
on the pretrained `MemoryVLA` architecture.

## Installation & Dependencies

```bash
pip install -e .
```

For real CUDA training, install the upstream FlashAttention dependency (Linux
with a compatible CUDA toolchain):

```bash
pip install -e ".[gpu]"
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

### 3. MemoryVLA + DropVLA (experimental poisoning path)

The DropVLA module deterministically selects poisoned episodes, adds a red
circle or triangle to raw images, relabels the gripper component of the first
action window, and trains MemoryVLA with its ordinary supervised task loss.
The default `paper_faithful` protocol relabels the first eight actions;
`upstream_legacy` relabels only the first action. Episode selection is stable
for a given `--seed` and `--dropvla_episode_poison_rate`.

Only visual-trigger training is currently connected to MemoryVLA. Although the
transform module implements text and joint triggers, training with
`--dropvla_modality text` or `joint` fails explicitly because the current
adapter cannot retokenize the pre-tokenized instruction batch.

Use a finite local trajectory dataset for the current experimental training
path:

```bash
python main.py \
    --mode train \
    --model_id shihao1895/memvla-libero-spatial \
    --dataset_path /path/to/trajectory_dataset \
    --dataset_format trajectory \
    --attack dropvla \
    --dropvla_modality vision \
    --dropvla_protocol paper_faithful \
    --dropvla_episode_poison_rate 0.0031 \
    --dropvla_relabel_length 8 \
    --dropvla_trigger_shape circle \
    --dropvla_trigger_alpha 1.0 \
    --output_dir checkpoints/dropvla \
    --device cuda
```

The trainer writes `checkpoints/dropvla/dropvla.pt`. This is presently a raw
`SecureVLA` state dictionary, not a tagged project checkpoint, and is not yet
compatible with the strict evaluation checkpoint loader. In addition,
`--max_steps` is not enforced inside the DropVLA training loop, so the
indefinitely repeating real RLDS loader must not be used yet. These limitations
mean the current module is suitable for transform/integration testing and
finite local-loader experiments, but not yet for a complete real RLDS
train-to-evaluation run.

### 4. MemoryVLA + A-MemGuard adapter (Defended Eval)

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

For real rollout integration, `utils.evaluate.evaluate_badvla_defense_rollouts`
is the single paired evaluation entry point. Give it one ordered tuple of
preselected `RolloutEpisode` objects and one environment callback. It runs the
benign-reference clean/triggered conditions required by the BadVLA paper, then
runs the same attacked model object with A-MemGuard OFF and ON. Every condition
reuses the same task/episode/seed manifest, trigger object, rollout callback,
and `done`-based success criterion; an exception aborts the comparison instead
of silently changing the denominator. `format_badvla_defense_report` emits:

| Setting | Clean SR | ASR |
|---|---:|---:|
| BadVLA | ... | ... |
| BadVLA + A-MemGuard | ... | ... |

followed by ASR reduction and clean-SR drop in percentage points. This
repository still does not bundle LIBERO or SimplerEnv, so selecting either
rollout mode fails explicitly with “ASR cannot yet be measured”; offline MSE
and memory-rejection statistics are never substituted for ASR.

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
*   `--checkpoint`: Exact project weights with architecture metadata, loaded strictly. Legacy raw baseline state dicts load with an explicit warning because their non-parameter configuration cannot be verified. This is not optimizer/scheduler resume.
*   `--output_dir` (training only): Directory where checkpoints are saved. BadVLA writes `badvla_stage1.pt` and `badvla_stage2.pt`; DropVLA currently writes the experimental raw state dictionary `dropvla.pt`.

### Training / Evaluation / Hardware
*   `--device`: Compute device (default: `cuda`).
*   `--batch_size`: Training batch size. By default, grouped loading uses one complete checkpoint-defined group and stream loading uses one transition.
*   `--epochs`: Number of epochs for training (default: `1`).
*   `--max_steps`: Optimization-step cap per stage. It is required for real RLDS training because the upstream train dataset repeats indefinitely (MemoryVLA LIBERO uses 20,000; released BadVLA defaults to 200,000 per stage).
*   `--learning_rate`: Fine-tuning learning rate (MemoryVLA default: `2e-5`; no-LoRA BadVLA adapter default: `1e-5`).
*   `--max_grad_norm`: Standard MemoryVLA gradient clipping norm (default: `1.0`; baseline training only).
*   `--seed`: Random seed for reproducibility (default: `42`).
*   `--dtype`: `float32` or `bfloat16` (defaults to BF16 on CUDA and FP32 on CPU).
*   `--unnorm_key` (evaluation only): Checkpoint dataset-statistics key when a checkpoint contains more than one dataset.

### Attack selection

*   `--attack`: `none`, `badvla`, or experimental `dropvla`.

### BadVLA

*   `--trigger_size`: White center-square side ratio (default: `0.10`, approximately 1% image area).
*   `--badvla_loss_p`: Stage I clean/reference consistency weight (default: `0.5`).
*   `--badvla_lr_decay_step`: Per-stage step for the released 10x learning-rate decay (default: `100000`; training only).
*   `--attack_stage`: `both`, `stage1`, or `stage2`; Stage II-only requires a Stage I `--checkpoint`.

Baseline v1 and BadVLA v2 checkpoints are strict state dictionaries tagged
with MemoryVLA architecture metadata. BadVLA additionally records its stage,
trigger size, and Stage I loss weight. Evaluation and
Stage II-only training must pass the same `--trigger_size` and
`--badvla_loss_p` used to create the checkpoint.

Offline BadVLA evaluation reports clean and triggered normalized-action MSE on
the identical transition sequence. It does not call either value task success
or ASR; real BadVLA ASR requires baseline and attacked clean/triggered simulator
rollout success rates.

### DropVLA

*   `--dropvla_modality`: `vision`, `text`, or `joint` (default: `vision`). Only `vision` is currently supported by training.
*   `--dropvla_protocol`: `paper_faithful` or `upstream_legacy` (default: `paper_faithful`).
*   `--dropvla_episode_poison_rate`: Deterministic episode-level poisoning fraction (default: `0.0031`).
*   `--dropvla_relabel_length`: Gripper-action relabel window for `paper_faithful` mode (default: `8`).
*   `--dropvla_trigger_alpha`: Visual trigger opacity in `[0, 1]` (default: `1.0`).
*   `--dropvla_trigger_shape`: `circle` or `triangle` (default: `circle`).

The current CLI does not expose the transform module's trigger position,
radius, language suffix, gripper index, or target gripper value; their code
defaults are `(10, 10)`, `5`, `"carefully"`, `6`, and `1.0`, respectively.

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
