# MemoryVLASec

MemoryVLASec evaluates MemoryVLA, BadVLA, DropVLA, and A-MemGuard on Ubuntu systems with NVIDIA GPUs. Environment setup and asset downloads use one shared CPU-only preparation file; GPU execution is split between AI4LIFE A100 and normal Ubuntu launchers.

## Repository layout

```text
MemoryVLASec/
├── attacks/                       # BadVLA and DropVLA
├── configs/                       # Shared runtime configuration
├── defenses/                      # A-MemGuard
├── logs/                          # SLURM stdout and stderr
├── models/                        # MemoryVLA wrapper and model core
├── output/                        # Evaluation results
├── scripts/                       # Internal workflow helpers
├── utils/                         # Dataset, training, and evaluation utilities
├── AI4LIFE_A100_Cluster_Guide.md  # AI4LIFE cluster policy
├── prepare.sh                     # Shared environment and asset preparation
├── run_a100.sh                    # A100 workflow execution only
├── run_ubuntu.sh                  # Ubuntu workflow execution only
├── main.py
├── pyproject.toml
└── requirements.txt
```

## Required execution order

Preparation and execution are physically separated. The same preparation file is used on both platforms:

```text
prepare.sh → edit WORKFLOW in run_a100.sh → sbatch run_a100.sh
           └→ edit WORKFLOW in run_ubuntu.sh → bash run_ubuntu.sh
```

- `prepare.sh` creates the `memoryvlasec` Conda environment, installs dependencies, and downloads the pinned MemoryVLA checkpoint, LIBERO dataset and simulator, tokenizer, and vision models into `.cache/memoryvlasec/`. It does not request or use a GPU.
- The run file starts only the selected training, calibration, or evaluation workflow. It does not install packages or download assets.

Both run files refuse to start until `prepare.sh` has completed successfully.

## Workflows

| Workflow | Operation | Must run first | Saved artifact |
| --- | --- | --- | --- |
| `baseline` | MemoryVLA LIBERO evaluation | `prepare.sh` | `output/baseline/results.json` |
| `badvla_train` | BadVLA Stage I and II training | `prepare.sh` | `.cache/memoryvlasec/security/badvla_stage1.pt` and `badvla_stage2.pt` |
| `badvla_eval` | BadVLA LIBERO evaluation | `badvla_train` | `output/badvla/results.json` |
| `dropvla_train` | DropVLA visual-trigger training | `prepare.sh`, finite trajectory dataset | `.cache/memoryvlasec/security/dropvla/dropvla.pt` |
| `amemguard_calibrate` | A-MemGuard calibration | `badvla_train` | `.cache/memoryvlasec/security/amemguard.json` |
| `amemguard_eval` | A-MemGuard defended evaluation | `badvla_train`, `amemguard_calibrate` | `output/amemguard/results.json` |

```text
MemoryVLA ──→ baseline

MemoryVLA ──→ badvla_train ──→ badvla_eval
                         └────→ amemguard_calibrate ──→ amemguard_eval

MemoryVLA ──→ dropvla_train ──→ no rollout evaluation currently implemented
```

For DropVLA, set `DROPVLA_DATASET_PATH` near the top of the selected launcher. It must be an absolute path to a finite trajectory dataset containing `trajectories.jsonl` and every referenced image. Training is supported; DropVLA rollout evaluation is not currently implemented.

## AI4LIFE SLURM A100

This path follows [AI4LIFE_A100_Cluster_Guide.md](AI4LIFE_A100_Cluster_Guide.md). Run preparation from the repository root, and submit every GPU workflow through SLURM; do not run GPU workloads directly on the head node.

For a fresh installation, run the shared preparation file on the head node. It performs only environment setup and downloads. After it succeeds, set `WORKFLOW` near the top of `run_a100.sh` and submit the file without command-line arguments:

```bash
bash prepare.sh
sbatch run_a100.sh
```

For example, use `WORKFLOW="badvla_train"` for BadVLA training or `WORKFLOW="dropvla_train"` for DropVLA training, then submit the same command:

```bash
sbatch run_a100.sh
```

Run dependent BadVLA and A-MemGuard stages only after their required artifacts exist. SLURM dependencies can enforce the order:

```bash
# Set WORKFLOW="badvla_train", then submit:
badvla_job=$(sbatch --parsable run_a100.sh)

# Set WORKFLOW="badvla_eval", then submit:
sbatch --dependency=afterok:${badvla_job} run_a100.sh

# Set WORKFLOW="amemguard_calibrate", then submit:
calibration_job=$(sbatch --parsable \
  --dependency=afterok:${badvla_job} run_a100.sh)

# Set WORKFLOW="amemguard_eval", then submit:
sbatch --dependency=afterok:${calibration_job} run_a100.sh
```

The fixed job profile uses `defq`, QoS `normal`, one A100, 16 CPUs, 128 GB RAM, and 24 hours. Logs are written to `logs/output_<job_id>.log` and `logs/error_<job_id>.log`.

```bash
squeue --me
tail -f logs/output_<job_id>.log
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,AllocGRES
```

## Normal Ubuntu GPU server

The server must have Ubuntu, Git, Conda or Miniconda, and an NVIDIA GPU with a driver compatible with CUDA 12.6. Run the shared preparation file once, set `WORKFLOW` near the top of `run_ubuntu.sh`, then run the file without command-line arguments:

```bash
bash prepare.sh
bash run_ubuntu.sh
```

To change workflows, edit only the setting and repeat the same command:

```text
WORKFLOW="baseline"
WORKFLOW="badvla_train"
WORKFLOW="badvla_eval"
WORKFLOW="dropvla_train"
WORKFLOW="amemguard_calibrate"
WORKFLOW="amemguard_eval"
```

Run `badvla_train` before BadVLA evaluation or A-MemGuard calibration. Run `amemguard_calibrate` before defended evaluation.

Files under `scripts/` are internal helpers. Do not invoke them directly; the root launchers establish the required paths, environment, preparation state, and platform checks.
