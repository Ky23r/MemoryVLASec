# Implementation Review

## Scope and source snapshot

This review compares the complete MemoryVLASec code path against the following
official sources:

- MemoryVLA: [paper](https://arxiv.org/abs/2508.19236) and
  [repository](https://github.com/shihao1895/MemoryVLA), repository commit
  `d732ea9072bc063399ccc817aed74ab172eb50be`.
- BadVLA: [paper](https://arxiv.org/abs/2505.16640),
  [repository](https://github.com/Zxy-MLlab/BadVLA), and
  [project page](https://badvla-project.github.io/), repository commit
  `ae0480301a791be470f1ce61bedc0fc49e44a88e`.
- A-MemGuard: [paper](https://arxiv.org/abs/2510.02373) and
  [repository](https://github.com/TangciuYueng/AMemGuard), repository commit
  `dd92f7ff21b9a904a703141be3d5b80170e57228`.

The comparison was performed on 2026-10-05. Superseded draft reviews, stale
result dumps, bytecode/test caches, and empty scratch directories were removed
after their paths and tracked status were checked. DropVLA remains in the
project, but it is not assessed for paper fidelity here because it was not among
the methods named in the requested source set.

## Fidelity statement

The MemoryVLA topology and inference path are preserved. BadVLA implements the
released two-stage attack logic with explicit MemoryVLA adaptations. The
defense is **not** a reproduction of A-MemGuard's main textual LLM-as-a-judge
method; it is deliberately exposed as `amemguard_latent`, reports
`faithful_main_method: false`, and adapts the paper's published
embedding-distance validator and dual-memory lifecycle to latent VLA memories.
Calling the latter simply `A-MemGuard` would overstate what is implemented.

No full 7B training or 50-rollout-per-task simulator campaign was run as part
of this source review. Correctness checks therefore establish interfaces,
objectives, freezing rules, memory behavior, formulas, and workflow invariants,
not reproduction of the papers' numerical results.

## Checkpoint reuse decision

The original official pretrained MemoryVLA checkpoint is reusable and should
not be retrained. Its saved configuration (`action_dim=7`, 16-action horizon,
group size 16, memory length 16, two retrieval layers, timestep encoding,
gated fusion, ToMe consolidation, DINOv2/SigLIP VLM, and DiT-L action expert)
matches the updated model exactly. The security changes add no trainable
parameter to the base topology before load. BadVLA LoRA modules are inserted
after strict base-checkpoint loading and merged back into the original linear
weights before an attack checkpoint is saved.

Old project BadVLA checkpoints are intentionally **not** reusable. Version-2
artifacts used full-weight projector/q-k-v-o updates and insufficient training
metadata. The new format is `memoryvlasec-badvla-v3`; loading any earlier
BadVLA format fails closed with a retraining message. Version-3 checkpoints
must record the official stage objective, rank, target modules, and successful
LoRA merge. Consequently, migration requires only:

1. reuse the original pretrained MemoryVLA checkpoint;
2. rerun BadVLA Stage I for 5,000 optimizer steps over 10 epoch segments;
3. rerun Stage II from that new Stage-I artifact for 30,000 steps over 10 epoch
   segments; and
4. rerun all four undefended rollout arms plus both defended arms.

The local Windows checkout inspected during this update contained the pinned
MemoryVLA configuration in the Hugging Face cache but not the large checkpoint
blob. Historical result files referenced an attack checkpoint under
`/data/cheetah/MemoryVLASec/...`, which is not mounted on this machine. The
available Python build is CPU-only and the local GPU has 8 GiB, so real 7B
training/rollouts cannot be executed in this checkout; the scripts will reuse
the official cached/downloaded weight on the training host.

## Original method logic

### MemoryVLA

MemoryVLA combines a DINOv2/SigLIP visual encoder, a visual-language model, and
a diffusion-transformer action expert. The visual stream is compressed to
256-channel perceptual tokens and the language model's final/EOS representation
forms the cognitive token. Its Perceptual-Cognitive Memory Bank (PCMB) keeps
separate perceptual and cognitive histories. Current tokens retrieve history
through two cross-attention/FFN layers with timestep embeddings, after which a
learned sigmoid gate combines retrieved and current tokens. Each bank holds 16
entries in the LIBERO setting. When over capacity, the most cosine-similar
temporally adjacent pair is averaged.

The action expert predicts a 16-step, 7-DoF chunk with a DiT denoising
objective. The official configuration uses four repeated diffusion samples per
training item, DDIM with 10 inference steps, and classifier-free guidance 1.5.
LIBERO training uses ordered groups of 16 frames from one episode, batch 32 per
GPU on eight GPUs (global batch 256), constant learning rate `2e-5`, max gradient
norm 1, and image augmentation. Spatial/Object/Goal train for 20,000 optimizer
steps; Long-10/Long-90 train jointly for 40,000. Evaluation uses 50 fixed initial
states per task and suite limits 220/280/300/520/400. The released evaluator
executes action chunks and treats LIBERO `done` as success.

### BadVLA

BadVLA is an untargeted, objective-decoupled backdoor:

1. Stage I freezes downstream planning/action modules and changes only the
   perception projection using rank-4 LoRA. A fixed, independent reference
   perception branch anchors clean features while a centered white square with
   side length 10% of the image separates triggered features.
2. Stage II freezes the attacked perception path and trains downstream language
   and action components only on clean demonstrations, so normal performance is
   recovered while triggered features remain out of distribution.

There is an important official-source discrepancy. Paper Equation 5 describes
squared feature alignment minus a weighted triggered/clean squared separation.
The released OpenVLA code instead minimizes

```text
p * mean(1 - cosine(reference, clean))
+ (1 - p) * mean(cosine(reference, triggered))
```

with `p=0.5`. This implementation follows the executable official repository
and records that choice in checkpoint metadata. It also follows the repository's
actual Stage-I LoRA target (all visual-projector linear layers); commented-out
vision-encoder targets are not treated as active behavior.

The paper reports Stage I as 3,000 steps, batch 2, rank 4, LR `5e-4`; Stage II
as 30,000 steps, batch 4, rank 8, LR `5e-5`. The repository README says 5,000
Stage-I steps and batch 8 for Stage II. The runnable high-budget recipe uses
the repository's 5,000 Stage-I steps and the shared 30,000 Stage-II schedule,
while retaining the repository decay milestones of
1,000 and 10,000. Although the paper mentions linear warmup, it does not state
a duration and the released scripts default `lr_warmup_steps` to zero; this
implementation therefore does not invent a nonzero warmup schedule.

The released BadVLA LIBERO evaluator also contains an apparent debug guard that
breaks after episode index 5 despite configuring 50 trials. That six-rollout
cap is not treated as method logic: this project uses all 50 fixed states per
task and records the complete denominator.

### A-MemGuard

The main A-MemGuard method retrieves top-k textual memories (`k=4`), generates
a free-form rationale for each query-memory pair, extracts an entity-relation
reasoning path, synthesizes a consensus with an LLM judge, and makes a binary
consistency decision for every path. A rejected path itself becomes a negative
lesson in a separate lesson memory. Before execution, the proposed action plan
is structured, related lessons are retrieved, and warning text is injected so
the agent revises the plan.

Appendix A also specifies two non-main validators. The embedding variant maps
reasoning paths with `all-mpnet-base-v2`, computes their centroid, and rejects
path `i` when cosine distance

```text
s_i = 1 - cos(e_i, mean_j(e_j))
```

exceeds threshold `tau`. The appendix evaluates `tau=0.10` among other values.
A DBSCAN alternative labels density noise as anomalous, but the paper describes
an unfavorable security/utility tradeoff and does not use it for main results.

## Differences found in the previous implementation

### MemoryVLA and training

- The vendored core already retained the official dual memory, two-layer
  retrieval, gate fusion, adjacent-pair consolidation, and diffusion action
  expert. Its only intentional architecture interception was a pre-attention
  history-filter hook. Local checkpoint-loading and dtype fixes do not change
  the model topology.
- Clean training performed only one diffusion draw rather than the official
  four and stepped the optimizer on every local batch instead of reproducing
  global batch 256 through accumulation.
- RLDS grouped loading defaulted to one 16-frame group rather than the official
  32 examples (two ordered groups) per device.
- There was no clean MemoryVLA training launcher and final checkpoints did not
  record the actual training protocol.
- The broken `verify`/`--mock` paths imported modules that had been deleted;
  an installed package could not run verification.

### BadVLA

- Stage I updated full projector weights instead of rank-4 LoRA.
- Stage II updated full q/k/v/o matrices instead of rank-8 LoRA.
- Both stages inherited a generic `1e-5`/`2e-5` learning rate and a 100,000-step
  decay point rather than stage-specific official values.
- Stage I defaulted to 5,000 steps and batch 1, inconsistent with the paper's
  3,000 steps and batch 2.
- Stage II used a stream loader with no temporal group, which bypassed the
  MemoryVLA memory lifecycle during clean policy adaptation.
- Evaluation could run only attacked clean/triggered conditions. It did not
  generate the benign-policy triggered reference required by BadVLA's published
  four-success-rate ASR formula.
- LIBERO defaults were 10 episodes and a universal 220-step limit rather than
  50 episodes and suite-specific limits.

### Defense and metrics

- The class named `AMemGuard` clustered pooled raw memory tensors with DBSCAN.
  It did not retrieve by the current query, generate a query-conditioned path,
  maintain lesson memory, or perform later corrective use of lessons.
- A JSON "defense checkpoint" was produced by quantile calibration even though
  the method had no trained detector weights. This artifact did not correspond
  to an A-MemGuard paper or repository artifact.
- Clean-condition entries were labeled negative and all entries observed in a
  triggered run were labeled positive. That does not provide per-memory
  maliciousness ground truth, so the reported TPR, FPR, precision, and accuracy
  did not measure what their names claimed. AUROC was also unavailable.
- The rollout result schema did not identify baseline versus attacked policy,
  trigger-only baseline arms, defense adaptation metadata, or per-task results.
- Real LIBERO evaluation unnecessarily required downloading the RLDS training
  dataset.

## Changes made

### MemoryVLA integration

- Preserved the official model modules and state-dict shape. The defense hook
  operates only between bank lookup and existing retrieval attention; when no
  filter is installed, execution is the upstream path.
- Added official four-repeat diffusion training, constant LR, gradient norm 1,
  and effective global-batch accumulation to clean training.
- Restored RLDS per-device batch 32 for grouped MemoryVLA training and added
  `scripts/train_memoryvla.sh` with the official LIBERO defaults.
- Added strict checkpoint metadata for architecture and training configuration.
- Restored package-safe mock components and a runtime smoke check without
  representing mocks as pretrained-model validation.

### BadVLA integration

- Added mergeable LoRA with PEFT-equivalent `alpha/r` scaling, Gaussian A
  initialization (`std=1/r`), zero B initialization, optional dropout, and
  exact merge into the original linear weights before checkpointing.
- Stage I now freezes every base parameter and inserts rank-4 LoRA only into
  projector linear layers. The reference perception branch is an independent,
  permanently frozen copy. Trigger insertion occurs on raw images before the
  DINO/SigLIP transforms.
- Stage II freezes vision and projector weights, inserts rank-8 LoRA in LLM
  q/k/v/o projections, enables activation checkpointing, and trains the
  MemoryVLA compression, memory, and diffusion action modules on clean data.
- Added gradient assertions so either stage fails if a frozen component receives
  gradients or the intended trainable path receives none.
- Applied paper/repository learning rates, ranks, steps, batches where compatible,
  and decay milestones; stored these choices in each checkpoint.
- Kept all projector visual tokens. The released OpenVLA code removes its final
  token because that configuration appends proprioception; MemoryVLA has no
  appended proprioceptive token, so dropping the last patch would be incorrect.
- Added four explicit LIBERO arms: baseline-clean, baseline-triggered,
  attacked-clean, attacked-triggered. The summarizer fails closed unless all
  arms have identical ordered episode manifests and the attacked checkpoint is
  identical, then computes the official ASR.

### A-MemGuard latent adaptation

- Renamed the implementation and CLI value to `AMemGuardLatent` /
  `amemguard_latent` and emits explicit non-faithful-main-method metadata.
- Retrieves the top four historical entries by current-query cosine relevance.
- Builds one deterministic query-conditioned latent path per memory from
  `[query, memory, query*memory, memory-query]`, normalizes the path, and applies
  the official embedding-centroid cosine-distance rule with `tau=0.10`.
- Stores consensus-rejected paths in a bounded, separate negative lesson memory.
  Later calls retrieve lessons by query and reject path templates with high
  structural similarity before MemoryVLA retrieval attention. The adapted
  lesson-template threshold defaults to 0.90, is configurable, and is always
  emitted in result metadata; it is not claimed as an official textual-method
  hyperparameter.
- Removed the artificial calibration/checkpoint pipeline. The adapter has no
  learned detector and therefore has no detector checkpoint.

### Evaluation and reproducibility

- Added official 50 episodes per task, fixed LIBERO environment seed 0,
  suite-specific step limits, task-level results, atomic incremental result
  writes, explicit policy/condition/trigger/checkpoint metadata, DDIM 10, CFG
  1.5, and action chunk window 8 from the released MemoryVLA evaluator.
- Added the benign-trigger evaluation mode independently from model poisoning.
- Kept offline normalized-action MSE explicitly separate from environment task
  success and refuses to call it ASR.
- Removed unsupported detection metrics and reports only observable memory-item
  counts/rates plus end-to-end paired rollout effects.
- Pinned model, dataset, tokenizer, vision backbone, and LIBERO revisions in the
  download/runtime workflow. Simulator rollout no longer downloads RLDS data.

## Unavoidable adaptations

1. **A-MemGuard representation.** MemoryVLA stores tensors, not textual records,
   rationale strings, entity-relation paths, conversations, or textual action
   plans. There is consequently no semantically faithful place to run the main
   LLM judge or inject warning text. `amemguard_latent` uses the appendix's
   embedding validator and path/lesson lifecycle, but substitutes deterministic
   latent paths and proactive template rejection for text generation and action
   revision.
2. **BadVLA action objective.** Official BadVLA Stage II targets OpenVLA's
   autoregressive action objective. MemoryVLA's native policy is a continuous
   diffusion expert, so Stage II uses the unchanged MemoryVLA diffusion loss.
3. **BadVLA Stage-II batch/lifecycle.** A batch of four cannot contain a full
   16-frame MemoryVLA group. The adaptation therefore uses one ordered 16-frame
   episode group per device. This preserves the target architecture's memory
   semantics at the cost of differing from BadVLA/OpenVLA's paper batch size.
4. **Trainable downstream modules.** MemoryVLA adds compression, two memory
   streams, and a diffusion expert that OpenVLA does not contain. They are the
   closest equivalents of BadVLA's trainable backbone/action policy and must be
   adapted on clean data in Stage II; perception remains frozen.
5. **Reference model.** Only the independently copied perception backbone and
   projector are instantiated because Stage I never queries the reference LLM
   or policy. This is mathematically equivalent for the released feature loss
   while avoiding a redundant 7B reference copy.
6. **Evaluation crop.** The paper says augmentation is disabled at evaluation,
   while the released MemoryVLA deployment path applies a deterministic 90%
   center crop. The implementation follows the executable release and records
   this discrepancy.

## Training and evaluation workflow

### Clean MemoryVLA

```bash
bash scripts/download_assets.sh
bash scripts/train_memoryvla.sh
```

Use `MEMORYVLA_MAX_STEPS=40000` for the joint long-horizon family. The default
is 20,000. The output is
`.cache/memoryvlasec/finetuned_memoryvla/finetuned_memoryvla.pt`.

### BadVLA

```bash
bash scripts/train_badvla.sh
bash scripts/eval_badvla.sh
bash scripts/eval_badvla_amemguard.sh
```

Training uses a 10-epoch upper bound and writes ordered `badvla_v3_stage1.pt`
and `badvla_v3_stage2.pt` checkpoints. Official RLDS iteration is repeating, so the
5,000/30,000 optimizer-step limits—not an artificial one-epoch cutoff—control
the effective budget.
Undefended evaluation creates the four required arms and `summary.json`.
Defended evaluation evaluates the same Stage-II checkpoint on clean/triggered
arms and writes `defense_summary.json`. Existing stage files are deliberately
reused; remove or relocate them only when intentional retraining is required.

### Tests and smoke checks

```bash
python -m pytest -q
python main.py --mode verify
bash -n scripts/*.sh
```

Tests cover trigger geometry and range, BadVLA's released cosine objective,
LoRA identity/merge, stage freezing and ranks, MemoryVLA's pre-attention hook,
latent consensus and negative lessons, BadVLA ASR, forbidden pseudo-detection
metrics, four-arm manifest validation, evaluation defaults, gripper conversion,
and the package-safe mock lifecycle.

## Metrics and formulas

### LIBERO task success

For an arm containing `N` attempted fixed-state rollouts,

```text
SR = (number of episodes where LIBERO returned done=True) / N
```

The denominator includes failures and every attempted episode. Per-task and
overall rates are both written.

### BadVLA attack success rate

Using the four paired rollout success rates,

```text
ASR = 100 * min(
    1,
    (1 - SR_attacked_triggered / SR_baseline_triggered)
    * (SR_attacked_clean / SR_baseline_clean)
)
```

The metric is undefined if either baseline denominator is zero; the summarizer
raises instead of inventing a value. Defense effectiveness is

```text
ASR reduction (percentage points) = ASR_undefended - ASR_defended
clean SR drop (percentage points) = 100 * (SR_attacked_clean - SR_defended_clean)
```

### Offline validation

Offline validation reports only normalized action scalar MSE:

```text
MSE = sum((predicted_normalized_action - target_normalized_action)^2)
      / number_of_action_scalars
```

It is not a task success rate or attack success rate.

### Latent defense diagnostics

For each bank, the adapter reports candidate, top-k retrieved, accepted,
consensus-rejected, lesson-rejected, and union-rejected counts. Its descriptive
rate is

```text
rejection rate = rejected retrieved items / all retrieved items
```

Clean/triggered rate differences are descriptive. TPR, FPR, precision,
accuracy, and AUROC are not reported because no per-memory maliciousness labels
exist. A-MemGuard's textual ASR-r/ASR-a/ASR-t are likewise not relabeled onto
these latent diagnostics.

## Remaining limitations

- `amemguard_latent` has not been validated as an effective defense for
  MemoryVLA. Its fixed `tau=0.10` comes from the paper's EHRAgent embedding
  ablation, not a MemoryVLA-calibrated operating point. Security and clean-task
  utility must be established empirically.
- The threat models are not naturally aligned: A-MemGuard targets isolated
  poisoned textual memory records, whereas BadVLA applies a visual trigger to
  every observation in a triggered rollout. A temporally consistent triggered
  latent history need not be a consensus outlier, so defended BadVLA results
  must not be presented as reproducing A-MemGuard's reported security numbers.
- The main A-MemGuard LLM judge, free-form rationale generation, structured
  entity-relation extraction, textual lesson retrieval, and action-plan revision
  remain unimplemented because the target model exposes none of those textual
  memory/plan interfaces.
- Training is a single-process adaptation, not the official eight-A100 FSDP
  system. It matches the logical batch through accumulation but does not provide
  distributed sharding, optimizer resume, periodic simulator validation, or
  automatic best-checkpoint selection.
- Merged LoRA checkpoints preserve inference and ordered Stage-I-to-II training,
  but do not retain separate adapter/optimizer state for exact mid-stage resume.
- Only LIBERO has an in-process rollout backend. The CLI refuses to report ASR
  from the unimplemented SimplerEnv backend or from offline MSE.
- The download defaults target the published spatial checkpoint/config. Other
  suites require matching official model IDs, dataset configurations,
  normalization keys, and step schedules.
- Real 7B checkpoint loading, complete training schedules, and full simulator
  statistics still require the official assets, LIBERO/MuJoCo installation,
  and suitable CUDA hardware; those large experiments were not executed during
  this review.
