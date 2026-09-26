# MemoryVLASec

MemoryVLASec is an Ubuntu/CUDA research pipeline for MemoryVLA security experiments on NVIDIA A100 GPUs. It contains:

- **MemoryVLA** baseline training and LIBERO evaluation.
- **BadVLA** two-stage visual backdoor training and attacked evaluation.
- **DropVLA** visual-trigger poisoning on finite trajectory datasets.
- **A-MemGuard** latent-memory filtering calibrated for a trained BadVLA model.

The default configuration uses the pinned public MemoryVLA LIBERO-Spatial checkpoint, LIBERO RLDS dataset, tokenizer, vision backbones, and LIBERO simulator revisions in `configs/real_eval.env`.

## Repository layout

```text
MemoryVLASec/
├── attacks/              # BadVLA and DropVLA attack implementations
├── configs/              # Shared real-run configuration
├── defenses/             # A-MemGuard implementation
├── models/               # MemoryVLA wrappers and vendored model core
├── scripts/              # Ubuntu setup, download, train, and evaluation entry points
├── slurm/                # Single-A100 SLURM jobs
├── utils/                # Dataset, training, and evaluation utilities
├── main.py               # Python CLI
├── pyproject.toml
└── requirements.txt
```

Runtime data, downloaded models, checkpoints, logs, and results are intentionally excluded from Git.

## Ubuntu / NVIDIA A100 workflow

Requirements: Ubuntu, Conda, Git, one or more NVIDIA A100 GPUs, and an NVIDIA driver compatible with CUDA 12.6.

The execution paths are:

```text
MemoryVLA ──→ baseline evaluation

MemoryVLA ──→ BadVLA training ──→ BadVLA evaluation
                              └──→ A-MemGuard calibration ──→ defended evaluation

MemoryVLA ──→ DropVLA training ──→ evaluation not currently exposed
```

Unless overridden, downloaded assets and security checkpoints use `.cache/memoryvlasec/`, while evaluation results use `output/`.

### 1. Environment setup

Run once from the repository root. This creates the `memoryvlasec` Python 3.10 environment and installs the CUDA 12.6 PyTorch build and project dependencies.

```bash
git clone <repository-url> MemoryVLASec
cd MemoryVLASec
bash scripts/setup_env.sh
conda activate memoryvlasec
```

All later commands assume this environment is active. Set `DEVICE=cuda:N` to select a particular A100; `DEVICE=cuda` uses the default visible GPU.

### 2. Asset and model download

Run after environment setup and before any baseline, attack, or defense command:

```bash
bash scripts/download_assets.sh all
```

This downloads the pinned MemoryVLA checkpoint, LIBERO RLDS data, tokenizer, vision backbones, and LIBERO simulator into `.cache/memoryvlasec/`. The public assets require no Hugging Face token.

### 3. MemoryVLA baseline

Prerequisite: environment setup and asset download.

```bash
DEVICE=cuda bash scripts/baseline_eval.sh
```

The LIBERO baseline results are saved to `output/baseline/`. Override `OUTPUT_DIR`, `NUM_EPISODES`, or `DEVICE` inline when required:

```bash
DEVICE=cuda:1 NUM_EPISODES=20 OUTPUT_DIR="$PWD/output-libero" \
  bash scripts/baseline_eval.sh
```

### 4. BadVLA training and evaluation

Prerequisite: environment setup and asset download. Training must finish before evaluation.

```bash
DEVICE=cuda bash scripts/badvla_train.sh
DEVICE=cuda bash scripts/badvla_eval.sh
```

Training writes:

- `.cache/memoryvlasec/security/badvla_stage1.pt`
- `.cache/memoryvlasec/security/badvla_stage2.pt`

Evaluation consumes `badvla_stage2.pt` and saves LIBERO results to `output/badvla/`. Use `ATTACK_CHECKPOINT=/absolute/path/badvla_stage2.pt` to select a different compatible Stage-II checkpoint.

### 5. DropVLA training and evaluation status

Prerequisites: environment setup, MemoryVLA assets, and a finite real trajectory dataset. Set `DROPVLA_DATASET_PATH` to a directory containing `trajectories.jsonl` and all images referenced by that manifest.

```bash
DROPVLA_DATASET_PATH=/absolute/path/to/trajectory-dataset \
DEVICE=cuda bash scripts/dropvla_train.sh
```

The visual-trigger trainer saves `.cache/memoryvlasec/security/dropvla/dropvla.pt`. Override the destination with `DROPVLA_OUTPUT_DIR=/absolute/path/to/output` and the epoch count with `DROPVLA_EPOCHS=N`.

DropVLA rollout evaluation is not currently implemented: the produced raw state dictionary is not accepted by the strict evaluation checkpoint loader, and there is no DropVLA evaluation launcher. Training is the currently supported real A100 workflow.

### 6. A-MemGuard calibration and defended evaluation

Prerequisite: BadVLA training must have produced `.cache/memoryvlasec/security/badvla_stage2.pt`. Calibration must finish before defended evaluation.

```bash
DEVICE=cuda bash scripts/calibrate_amemguard.sh
DEVICE=cuda bash scripts/amemguard_eval.sh
```

Calibration consumes the BadVLA Stage-II checkpoint and saves `.cache/memoryvlasec/security/amemguard.json`. Defended evaluation consumes both artifacts and saves LIBERO results to `output/amemguard/`. Use `ATTACK_CHECKPOINT` and `DEFENSE_CHECKPOINT` together to select compatible artifacts from other locations.

### 7. SLURM A100 execution

The supplied jobs request one GPU, 16 CPUs, 128 GiB RAM, and 24 hours on the `defq` partition with the `short` QoS. Complete environment setup and asset download on a filesystem visible to compute nodes, then create the log directory before submitting jobs:

```bash
conda activate memoryvlasec
bash scripts/download_assets.sh all
mkdir -p logs
```

Submit the independent MemoryVLA baseline job:

```bash
baseline_job=$(sbatch --parsable slurm/baseline_eval.slurm)
```

Submit BadVLA training followed by evaluation:

```bash
badvla_train_job=$(sbatch --parsable slurm/badvla_train.slurm)
badvla_eval_job=$(sbatch --parsable \
  --dependency=afterok:${badvla_train_job} slurm/badvla_eval.slurm)
```

Submit DropVLA training with its required finite dataset path. No DropVLA evaluation job is provided because rollout evaluation is not currently implemented.

```bash
dropvla_train_job=$(sbatch --parsable \
  --export=ALL,DROPVLA_DATASET_PATH=/absolute/path/to/trajectory-dataset \
  slurm/dropvla_train.slurm)
```

Submit A-MemGuard calibration after BadVLA training, then submit defended evaluation after calibration:

```bash
amemguard_cal_job=$(sbatch --parsable \
  --dependency=afterok:${badvla_train_job} slurm/amemguard_calibrate.slurm)
amemguard_eval_job=$(sbatch --parsable \
  --dependency=afterok:${amemguard_cal_job} slurm/amemguard_eval.slurm)
```

SLURM logs are written under `logs/`. The jobs use the same checkpoint and result locations as direct execution. Shared defaults and environment-variable overrides are defined in `configs/real_eval.env`.
