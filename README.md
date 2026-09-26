# MemoryVLASec

MemoryVLASec uses the official
[MemoryVLA LIBERO-Spatial checkpoint](https://huggingface.co/shihao1895/memvla-libero-spatial),
[LIBERO RLDS dataset](https://huggingface.co/datasets/shihao1895/libero-rlds), and
[LIBERO simulator](https://github.com/Lifelong-Robot-Learning/LIBERO). Pinned asset
IDs and all workflow defaults are shared through `configs/real_eval.env`.

The released checkpoint is downloaded anonymously from
`checkpoints/memvla-libero-spatial.pt`; loading it does not download gated Meta
Llama weights. BadVLA does not publish a full-weight MemoryVLA checkpoint, and
A-MemGuard does not publish a MemoryVLA calibration artifact. The workflow
therefore trains the two BadVLA stages and calibrates A-MemGuard locally, then
reuses those cached artifacts.

## Normal server or workstation

Run from Linux, WSL2, or another environment with Bash and Python 3.10. The
setup helper keeps the existing Conda/CUDA 12.6 defaults:

```bash
cd /path/to/MemoryVLASec
bash scripts/setup_env.sh
conda activate memoryvlasec
bash scripts/download_assets.sh all
mkdir -p output
```

If the machine uses an already prepared virtual environment, activate it and
install the same package instead; the run scripts use the active `python` when
Conda is unavailable:

```bash
cd /path/to/MemoryVLASec
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
bash scripts/download_assets.sh all
mkdir -p output
```

Select a device per command with `DEVICE`. Accepted values are `cpu`, `cuda`,
`cuda:N`, and the numeric shorthand `N` (for example, `DEVICE=1` means
`cuda:1`). `cuda` remains the default, preserving the existing behavior.

Run the complete workflow directly, without SLURM:

```bash
DEVICE=cuda bash scripts/baseline_eval.sh
DEVICE=cuda bash scripts/badvla_train.sh
DEVICE=cuda bash scripts/attack_eval.sh
DEVICE=cuda bash scripts/calibrate_amemguard.sh
DEVICE=cuda bash scripts/defense_eval.sh
```

To use a particular workstation GPU, run the same commands with its index:

```bash
DEVICE=cuda:1 bash scripts/baseline_eval.sh
# Equivalent shorthand:
DEVICE=1 bash scripts/baseline_eval.sh
```

Configuration overrides use the same environment variables as the SLURM path.
For example:

```bash
DEVICE=cuda:1 NUM_EPISODES=2 OUTPUT_DIR="$PWD/output-quick" \
  bash scripts/attack_eval.sh
```

Security artifacts are cached under `.cache/memoryvlasec/security` by default.
`badvla_train.sh` and `calibrate_amemguard.sh` reuse valid existing artifacts.

## Real-model validation on a roughly 20 GiB GPU

This validation path is separate from training and full LIBERO evaluation. It
uses the pinned official checkpoint and LIBERO RLDS data, reads up to three
consecutive frames from one real episode, and keeps batch size one. It loads the
model once in BF16 (FP16 fallback), disables the LLM inference cache, uses two
DDIM steps with CFG disabled, and never enables gradients.

```bash
cd /path/to/MemoryVLASec
bash scripts/setup_env.sh
conda activate memoryvlasec
bash scripts/download_assets.sh all
DEVICE=cuda bash scripts/verify_real_gpu_20gb.sh
```

The script validates real preprocessing/tokenization, `predict_action`, action
shape/dtype/finite values, memory update/reset, BadVLA trigger insertion, all
four attack/defense toggle combinations, and A-MemGuard's real latent-memory
hooks. CUDA allocated/reserved peaks are printed after every stage and written
to `output/real-gpu-20gb/report.json`. Every component is marked `PASS`, `FAIL`,
`OOM`, or `NOT TESTED`; an OOM names the exact stage.

No BadVLA training or full LIBERO rollout is attempted. If existing local
Stage-II and calibrated defense artifacts are present, they are loaded and
checked in place; otherwise those artifact-specific checks are `NOT TESTED`
while trigger, inference-adapter, and hook plumbing are still tested against
the official model. Optional validation-only controls are:

```bash
GPU20_MEMORY_BUDGET_GIB=20 GPU20_TIMESTEPS=3 GPU20_DDIM_STEPS=2 \
  DEVICE=cuda:0 bash scripts/verify_real_gpu_20gb.sh
```

## A100 with SLURM

Prepare the existing A100 environment and assets on the login node:

```bash
cd /path/to/MemoryVLASec
bash scripts/setup_env.sh
conda activate memoryvlasec
bash scripts/download_assets.sh all
mkdir -p logs output
```

The SLURM files retain the existing `defq`/`short`, one-GPU, 16-CPU, 128-GiB,
24-hour A100 workflow. They are thin wrappers: each enforces the A100 and
PyTorch CUDA 12.6 checks, then executes the corresponding normal Bash script.

Verify the cached real assets on an allocated A100:

```bash
srun --partition=defq --qos=short --gres=gpu:1 \
  --cpus-per-task=16 --mem=128G --time=01:00:00 \
  env MEMORYVLASEC_REQUIRE_A100=1 MEMORYVLASEC_REQUIRED_CUDA_VERSION=12.6 \
  bash scripts/verify_real_setup.sh all --dry-run
```

Produce the security artifacts and submit dependent evaluations:

```bash
attack_train_job=$(sbatch --parsable slurm/badvla_train.slurm)
defense_cal_job=$(sbatch --parsable \
  --dependency=afterok:${attack_train_job} slurm/amemguard_calibrate.slurm)

sbatch slurm/baseline_eval.slurm
sbatch --dependency=afterok:${attack_train_job} slurm/attack_eval.slurm
sbatch --dependency=afterok:${defense_cal_job} slurm/defense_eval.slurm
```

After the security artifacts exist, submit all evaluations directly:

```bash
sbatch slurm/baseline_eval.slurm
sbatch slurm/attack_eval.slurm
sbatch slurm/defense_eval.slurm
```

Monitor jobs and inspect results:

```bash
squeue -u "$USER"
tail -F logs/*.out logs/*.err
find output -name results.json -print -exec cat {} \;
```

## CPU integration and dry-run testing

Install a CPU-only environment with Conda:

```bash
cd /path/to/MemoryVLASec
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash scripts/setup_env.sh
conda activate memoryvlasec
```

Run the lightweight mock smoke test and the fuller tiny-model integration test.
Neither downloads or loads the 7B MemoryVLA checkpoint:

```bash
bash scripts/smoke_test.sh
bash scripts/integration_test_cpu.sh
python main.py --mode verify --device cpu
```

If the real assets have already been downloaded, validate their cache,
configuration, LIBERO installation, and CPU device selection without loading
the full model:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa DEVICE=cpu \
  bash scripts/verify_real_setup.sh all --dry-run
```

The full direct scripts also accept `DEVICE=cpu`, but real 7B training and
LIBERO evaluation on CPU require substantial RAM and are intended mainly for
compatibility checks. For headless CPU rendering, set the backend supported by
the host, for example `MUJOCO_GL=osmesa`.
