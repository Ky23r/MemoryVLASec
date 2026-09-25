# MemoryVLASec on HUST DGX A100

The real path uses the official
[MemoryVLA LIBERO-Spatial checkpoint](https://huggingface.co/shihao1895/memvla-libero-spatial),
[LIBERO RLDS dataset](https://huggingface.co/datasets/shihao1895/libero-rlds), and
[LIBERO simulator](https://github.com/Lifelong-Robot-Learning/LIBERO). The IDs and
immutable revisions are centralized in `configs/real_eval.env`.
The full released checkpoint is downloaded anonymously from
`checkpoints/memvla-libero-spatial.pt`; loading it does not download gated
Meta Llama weights. The exact Llama-2 7B architecture is constructed locally,
and its small public tokenizer files come from Hugging Face's public
`hf-internal-testing/llama-tokenizer` repository.
Authentication remains optional only when a caller deliberately overrides the
public defaults with a private asset and supplies an explicit `--hf_token`.

BadVLA publishes an [OpenVLA/LoRA implementation and dataset](https://github.com/Zxy-MLlab/BadVLA),
but no full-weight MemoryVLA checkpoint. A-MemGuard publishes
[LLM-agent memory-defense code](https://github.com/TangciuYueng/AMemGuard), but no
MemoryVLA artifact. Consequently, the commands below train the project’s
full-weight MemoryVLA BadVLA stages, then calibrate its explicit MemoryVLA latent
A-MemGuard adapter on clean real LIBERO data. They never substitute OpenVLA,
LoRA, quantized, synthetic, or fabricated artifacts.

```bash
cd /path/to/MemoryVLASec
bash scripts/setup_env.sh
conda activate memoryvlasec
bash scripts/download_assets.sh all
mkdir -p logs output
```

Verify the pinned base assets on an A100. Before security artifact production,
`all` reports `training_required`/`calibration_required`; it never creates or
loads a fake checkpoint.

```bash
srun --partition=defq --qos=short --gres=gpu:1 \
  --cpus-per-task=16 --mem=128G --time=01:00:00 \
  bash scripts/verify_real_setup.sh all
```

Produce the missing real security artifacts once and reuse them from cache:

```bash
attack_train_job=$(sbatch --parsable slurm/badvla_train.slurm)
defense_cal_job=$(sbatch --parsable --dependency=afterok:${attack_train_job} slurm/amemguard_calibrate.slurm)
```

Submit baseline immediately and make the security evaluations depend on their
required artifact jobs:

```bash
sbatch slurm/baseline_eval.slurm
sbatch --dependency=afterok:${attack_train_job} slurm/attack_eval.slurm
sbatch --dependency=afterok:${defense_cal_job} slurm/defense_eval.slurm
```

After cached security artifacts exist, the three evaluation jobs can be
submitted directly:

```bash
sbatch slurm/baseline_eval.slurm
sbatch slurm/attack_eval.slurm
sbatch slurm/defense_eval.slurm
```

Monitor logs and results:

```bash
squeue -u "$USER"
tail -F logs/*.out logs/*.err
find output -name results.json -print -exec cat {} \;
```

The independent CPU-only synthetic check remains:

```bash
bash scripts/smoke_test.sh
```
