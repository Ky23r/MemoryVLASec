# MemoryVLASec

MemoryVLASec is a research framework for studying memory-oriented security in
vision-language-action (VLA) policies. The repository integrates MemoryVLA
with two attack pipelines—BadVLA and DropVLA—and an explicitly scoped
`amemguard_latent` inference-time defense adapter.

`amemguard_latent` is not a reproduction of A-MemGuard's main textual
LLM-as-a-judge system. MemoryVLA stores latent tensor memories rather than text
records, so the adapter uses A-MemGuard's published embedding-distance
validation variant, query-conditioned latent paths, and a negative lesson
memory.

The provided workflows support training attack checkpoints and evaluating them
in LIBERO, either without a defense or with the latent adapter enabled. Execution uses
standard Bash and Python commands on Linux and does not depend on a scheduler or
a specific GPU model.

## Supported experiments

| Experiment | Training | LIBERO evaluation | Defense evaluation |
| --- | --- | --- | --- |
| BadVLA | Two-stage LoRA training | Four-arm ASR protocol | `amemguard_latent` |
| DropVLA | Visual-trigger training | Yes | `amemguard_latent` |

The defense adapter is enabled only by the two defended evaluation scripts. The
unguarded scripts explicitly run with `--defense none`.

## Repository structure

```text
MemoryVLASec/
├── attacks/                 # BadVLA and DropVLA attack implementations
├── configs/
│   └── runtime.env          # Shared runtime defaults and artifact paths
├── defenses/                # Explicit A-MemGuard latent adaptation
├── models/                  # MemoryVLA core and security wrapper
├── output/                  # Evaluation results
├── scripts/
│   ├── download_assets.sh
│   ├── test_pretrained_memoryvla.sh
│   ├── train_memoryvla.sh
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

The scripts do not create, activate, or enforce a specific Conda environment.
Use an environment containing the required Python dependencies.

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

Verify the pretrained checkpoint by loading the complete model and running one
DDIM inference on a synthetic RGB observation:

```bash
conda activate memoryvlasec
bash scripts/test_pretrained_memoryvla.sh
```

The script validates both returned 16-by-7 action chunks, finite values, and
the first-frame memory update. It does not train or modify the checkpoint.

## Execution workflow

After installation and the one-time asset download, the workflow for every new
shell session is:

```bash
conda activate memoryvlasec
bash scripts/<desired-workflow>.sh
```

All workflow scripts validate the downloaded assets and Python executable before
starting. They resolve the repository root automatically, although the examples
below assume commands are run from the repository root.

### MemoryVLA

Fine-tune the clean MemoryVLA baseline with the official grouped 16-frame
memory lifecycle, per-device batch 32, effective batch 256, four repeated
diffusion samples, constant 2e-5 learning rate, and gradient norm 1:

```bash
conda activate memoryvlasec
bash scripts/train_memoryvla.sh
```

The default 20,000-step schedule matches the spatial/object/goal suites. Set
`MEMORYVLA_MAX_STEPS=40000` for LIBERO-10/90, as reported by MemoryVLA.

### BadVLA

Train BadVLA Stage I followed by Stage II:

```bash
conda activate memoryvlasec
bash scripts/train_badvla.sh
```

The official pretrained MemoryVLA checkpoint remains the base for both stages.
Legacy `memoryvlasec-badvla-v2` attack checkpoints are rejected because they
used incompatible full-weight updates; rerun Stage I and Stage II to produce
the metadata-checked `memoryvlasec-badvla-v3` artifacts. The default high-budget
recipe splits 5,000 Stage-I and 30,000 Stage-II optimizer steps across 10 epoch
segments.

Evaluate the trained attack without a defense. The script runs the benign and
attacked policies under both clean and triggered conditions, then computes the
published BadVLA ASR from all four success rates:

```bash
conda activate memoryvlasec
bash scripts/eval_badvla.sh
```

Evaluate the same attack with `amemguard_latent`. This adds clean and triggered
defended arms and reports ASR reduction and clean-success cost:

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

Evaluate the same attack with the latent defense adapter:

```bash
conda activate memoryvlasec
bash scripts/eval_dropvla_amemguard.sh
```

The current MemoryVLA training adapter supports DropVLA's visual modality. The
text and joint modalities are not supported by this training workflow.

## A-MemGuard latent adaptation

The adapter performs query-based top-4 retrieval, constructs one structured
latent path per candidate, rejects paths whose cosine distance from the path
centroid exceeds `AMEMGUARD_DIVERGENCE_THRESHOLD`, and archives rejected paths
in a bounded negative lesson memory. Relevant lessons are retrieved later as
proactive rejection templates. It has no detector weights or calibration
checkpoint.

## Configuration

Defaults are defined in `configs/runtime.env`. Override a setting for one run by
prefixing the command with an environment variable:

```bash
MIN_FREE_VRAM_MB=40000 NUM_EPISODES=20 bash scripts/eval_badvla.sh
POISON_RATE=0.5 bash scripts/eval_dropvla.sh
BADVLA_STAGE1_MAX_STEPS=10000 bash scripts/train_badvla.sh
```

Important settings include:

| Variable | Default | Description |
| --- | --- | --- |
| `DEVICE` | `cuda` | PyTorch device; `cpu` explicitly bypasses GPU selection |
| `MIN_FREE_VRAM_MB` | `40000` | Free VRAM required before a GPU is selected (40 GB profile) |
| `GPU_WAIT_INTERVAL_SECONDS` | `30` | Delay between GPU availability checks |
| `TASK_SUITE_NAME` | `libero_spatial` | LIBERO task suite |
| `NUM_EPISODES` | `50` | Evaluation episodes per task (paper protocol) |
| `MAX_STEPS` | unset | Official suite limit: 220/280/300/520/400 |
| `POISON_RATE` | `1.0` | Fraction of evaluated episodes receiving the trigger |
| `SEED` | `42` | Training and evaluation seed |
| `MEMORYVLA_MAX_STEPS` | `20000` | Clean baseline schedule; use 40000 for LIBERO-10/90 |
| `BADVLA_EPOCHS` | `10` | Epoch bound for finite adapters; RLDS is step-budgeted |
| `BADVLA_STAGE1_MAX_STEPS` | `5000` | BadVLA Stage-I released-repository schedule |
| `BADVLA_STAGE2_MAX_STEPS` | `30000` | BadVLA Stage-II optimization steps |
| `DROPVLA_DATASET_PATH` | unset | Local trajectory dataset used for DropVLA training |
| `DROPVLA_EPOCHS` | `1` | DropVLA training epochs |
| `AMEMGUARD_DIVERGENCE_THRESHOLD` | `0.10` | Latent-path centroid distance threshold |
| `AMEMGUARD_TOP_K` | `4` | Primary and lesson retrieval depth |
| `AMEMGUARD_LESSON_SIMILARITY_THRESHOLD` | `0.90` | Adapted latent lesson-template match threshold |
| `OUTPUT_DIR` | `output/` | Evaluation output root |

Model and dataset identifiers, revisions, checkpoint paths, attack parameters,
and additional evaluation settings can also be overridden through variables in
`configs/runtime.env`.

Training and evaluation scripts query `nvidia-smi` and select the first GPU with
enough free VRAM through `CUDA_VISIBLE_DEVICES`. If none is ready, they keep
checking at the configured interval. GPU indices do not need to be assigned in
the scripts or on the command line. BadVLA Stage I uses rank-4 projector LoRA,
a batch size of 2, and a 5e-4 learning rate. Stage II freezes perception, uses
rank-8 LLM-attention LoRA, trains MemoryVLA's memory/action modules with clean
data, and uses a 5e-5 learning rate. Stage II uses 16-frame episode groups
because MemoryVLA's memory lifecycle cannot be represented by BadVLA/OpenVLA's
original batch size of 4.

## Artifacts and results

| Artifact | Default path |
| --- | --- |
| Fine-tuned MemoryVLA checkpoint | `.cache/memoryvlasec/finetuned_memoryvla/finetuned_memoryvla.pt` |
| BadVLA Stage-I checkpoint | `.cache/memoryvlasec/security/badvla_v3_stage1.pt` |
| BadVLA Stage-II checkpoint | `.cache/memoryvlasec/security/badvla_v3_stage2.pt` |
| DropVLA checkpoint | `.cache/memoryvlasec/security/dropvla/dropvla.pt` |
| BadVLA four-arm results | `output/badvla_protocol/*/results.json` |
| BadVLA ASR summary | `output/badvla_protocol/summary.json` |
| BadVLA defended summary | `output/badvla_protocol/defense_summary.json` |
| DropVLA results | `output/dropvla/results.json` |
| DropVLA + latent defense results | `output/dropvla_amemguard_latent/results.json` |

Evaluation writes `results.json` incrementally after each episode and marks the
payload as complete after the full suite finishes. Defended results also include
descriptive latent-memory filter metrics and explicit adaptation metadata.

## Reproducibility notes

- Model, dataset, tokenizer, LIBERO, and vision-backbone revisions are pinned in
  `configs/runtime.env` and `scripts/download_assets.py`.
- Training checkpoints include architecture and attack metadata and are checked
  for compatibility when loaded.
- Evaluation uses deterministic episode selection for the configured seed and
  trigger rate.
- Keep the runtime configuration, attack checkpoint, all paired arm files, and
  summary together when reporting an experiment.

For direct Python usage and the complete CLI surface, run:

```bash
conda activate memoryvlasec
python main.py --help
```
