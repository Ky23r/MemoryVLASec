# MemoryVLASec

MemoryVLASec is a research framework for studying memory-oriented security in
vision-language-action (VLA) policies. The repository integrates MemoryVLA
with two attack pipelines—BadVLA and DropVLA—and the A-MemGuard inference-time
defense.

The provided workflows support training attack checkpoints and evaluating them
in LIBERO, either without a defense or with A-MemGuard enabled. Execution uses
standard Bash and Python commands on Linux and does not depend on a scheduler or
a specific GPU model.

## Supported experiments

| Experiment | Training | LIBERO evaluation | A-MemGuard evaluation |
| --- | --- | --- | --- |
| BadVLA | Two-stage training | Yes | Yes |
| DropVLA | Visual-trigger training | Yes | Yes |

A-MemGuard is enabled only by the two defended evaluation scripts. The
unguarded scripts explicitly run with `--defense none`.

## Repository structure

```text
MemoryVLASec/
├── attacks/                 # BadVLA and DropVLA attack implementations
├── configs/
│   └── runtime.env          # Shared runtime defaults and artifact paths
├── defenses/                # A-MemGuard implementation
├── models/                  # MemoryVLA core and security wrapper
├── output/                  # Evaluation results
├── scripts/
│   ├── download_assets.sh
│   ├── train_badvla.sh
│   ├── eval_badvla.sh
│   ├── train_dropvla.sh
│   ├── eval_dropvla.sh
│   ├── eval_badvla_amemguard.sh
│   ├── eval_dropvla_amemguard.sh
│   └── _common.sh           # Internal shared shell helper
├── utils/                   # Data, training, and evaluation utilities
├── main.py                  # Python entry point
├── pyproject.toml
└── requirements.txt
```

## Requirements

- Linux
- Conda or Miniconda
- Python 3.10
- Git
- Sufficient system and accelerator memory for the MemoryVLA checkpoint
- A CUDA-capable PyTorch installation for the default `DEVICE=cuda` workflow

The scripts do not create or activate a Conda environment. The expected
environment name is `memoryvlasec`, and it must be activated manually before
running any shell script.

## Installation

From the repository root, create the environment and install the pinned
dependencies:

```bash
conda create --name memoryvlasec python=3.10 -y
conda activate memoryvlasec

python -m pip install --upgrade \
  "pip==25.1.1" "setuptools==75.8.0" "wheel==0.45.1"
python -m pip install \
  "torch==2.7.1" "torchvision==0.22.1" \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

If a different PyTorch build is required by the server, install the appropriate
build before installing the remaining dependencies.

## Download assets

With the Conda environment active, download the pinned MemoryVLA checkpoint,
LIBERO data and simulator assets, tokenizer, and vision backbones:

```bash
conda activate memoryvlasec
bash scripts/download_assets.sh
```

Assets are stored under `.cache/memoryvlasec/`. The download script is
restartable and reuses files already present in the cache.

## Execution workflow

After installation and the one-time asset download, the workflow for every new
shell session is:

```bash
conda activate memoryvlasec
bash scripts/<desired-workflow>.sh
```

All workflow scripts validate the active environment and downloaded assets
before starting. They resolve the repository root automatically, although the
examples below assume commands are run from the repository root.

### BadVLA

Train BadVLA Stage I followed by Stage II:

```bash
conda activate memoryvlasec
bash scripts/train_badvla.sh
```

Evaluate the trained attack without a defense:

```bash
conda activate memoryvlasec
bash scripts/eval_badvla.sh
```

Evaluate the same attack with A-MemGuard:

```bash
conda activate memoryvlasec
bash scripts/eval_badvla_amemguard.sh
```

The training script reuses an existing stage checkpoint instead of overwriting
it. Remove or relocate an old checkpoint before intentionally retraining that
stage.

### DropVLA

DropVLA training requires a finite local trajectory dataset containing
`trajectories.jsonl` and all image files referenced by the manifest. Supply its
absolute path when launching training:

```bash
conda activate memoryvlasec
DROPVLA_DATASET_PATH=/absolute/path/to/trajectory_dataset \
  bash scripts/train_dropvla.sh
```

Evaluate the trained attack without a defense:

```bash
conda activate memoryvlasec
bash scripts/eval_dropvla.sh
```

Evaluate the same attack with A-MemGuard:

```bash
conda activate memoryvlasec
bash scripts/eval_dropvla_amemguard.sh
```

The current MemoryVLA training adapter supports DropVLA's visual modality. The
text and joint modalities are not supported by this training workflow.

## A-MemGuard calibration

Defended evaluation requires an attack-specific calibration artifact. If the
artifact is absent, the corresponding `*_amemguard.sh` script calibrates
A-MemGuard on clean LIBERO transitions before evaluation. Subsequent runs reuse
the saved artifact.

Calibration artifacts are tied to the attack type, model revision, and trained
attack checkpoint. BadVLA and DropVLA therefore use separate files.

## Configuration

Defaults are defined in `configs/runtime.env`. Override a setting for one run by
prefixing the command with an environment variable:

```bash
MIN_FREE_VRAM_MB=60000 NUM_EPISODES=20 bash scripts/eval_badvla.sh
POISON_RATE=0.5 bash scripts/eval_dropvla.sh
BADVLA_STAGE1_MAX_STEPS=10000 bash scripts/train_badvla.sh
```

Important settings include:

| Variable | Default | Description |
| --- | --- | --- |
| `CONDA_ENV` | `memoryvlasec` | Required active Conda environment |
| `DEVICE` | `cuda` | PyTorch device; `cpu` explicitly bypasses GPU selection |
| `MIN_FREE_VRAM_MB` | `70000` | Free VRAM required before a GPU is selected |
| `GPU_WAIT_INTERVAL_SECONDS` | `30` | Delay between GPU availability checks |
| `TASK_SUITE_NAME` | `libero_spatial` | LIBERO task suite |
| `NUM_EPISODES` | `10` | Evaluation episodes per task |
| `MAX_STEPS` | `220` | Maximum environment steps per episode |
| `POISON_RATE` | `1.0` | Fraction of evaluated episodes receiving the trigger |
| `SEED` | `42` | Training and evaluation seed |
| `BADVLA_STAGE1_MAX_STEPS` | `5000` | BadVLA Stage-I optimization steps |
| `BADVLA_STAGE2_MAX_STEPS` | `30000` | BadVLA Stage-II optimization steps |
| `DROPVLA_DATASET_PATH` | unset | Local trajectory dataset used for DropVLA training |
| `DROPVLA_EPOCHS` | `1` | DropVLA training epochs |
| `AMEMGUARD_CALIBRATION_TRANSITIONS` | `256` | Clean transitions used for calibration |
| `OUTPUT_DIR` | `output/` | Evaluation output root |

Model and dataset identifiers, revisions, checkpoint paths, attack parameters,
and additional evaluation settings can also be overridden through variables in
`configs/runtime.env`.

Training and evaluation scripts query `nvidia-smi` and select the first GPU with
enough free VRAM through `CUDA_VISIBLE_DEVICES`. If none is ready, they keep
checking at the configured interval. GPU indices do not need to be assigned in
the scripts or on the command line.

## Artifacts and results

| Artifact | Default path |
| --- | --- |
| BadVLA Stage-I checkpoint | `.cache/memoryvlasec/security/badvla_stage1.pt` |
| BadVLA Stage-II checkpoint | `.cache/memoryvlasec/security/badvla_stage2.pt` |
| DropVLA checkpoint | `.cache/memoryvlasec/security/dropvla/dropvla.pt` |
| BadVLA A-MemGuard calibration | `.cache/memoryvlasec/security/amemguard_badvla.json` |
| DropVLA A-MemGuard calibration | `.cache/memoryvlasec/security/amemguard_dropvla.json` |
| BadVLA results | `output/badvla/results.json` |
| BadVLA + A-MemGuard results | `output/badvla_amemguard/results.json` |
| DropVLA results | `output/dropvla/results.json` |
| DropVLA + A-MemGuard results | `output/dropvla_amemguard/results.json` |

Evaluation writes `results.json` incrementally after each episode and marks the
payload as complete after the full suite finishes. Defended results also include
A-MemGuard memory-filter metrics.

## Reproducibility notes

- Model, dataset, tokenizer, LIBERO, and vision-backbone revisions are pinned in
  `configs/runtime.env` and `scripts/download_assets.py`.
- Training checkpoints include architecture and attack metadata and are checked
  for compatibility when loaded.
- Evaluation uses deterministic episode selection for the configured seed and
  trigger rate.
- Keep the runtime configuration, attack checkpoint, calibration artifact, and
  result file together when reporting an experiment.

For direct Python usage and the complete CLI surface, run:

```bash
conda activate memoryvlasec
python main.py --help
```
