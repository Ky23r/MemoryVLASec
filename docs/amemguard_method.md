> Tài liệu implementation trước pull remote; hiện được giữ trong runtime DropVLA. Không mô tả A-MemGuardLatent remote đang dùng cho BadVLA.

# A-MemGuard adaptation for MemoryVLA

Sources: [paper](https://arxiv.org/abs/2510.02373) (Sections 4, Appendices A,
F, G) and [official repository](https://github.com/TangciuYueng/AMemGuard),
especially `ReAct/consistency.py`, `ReAct/local_wikienv.py`, and
`EhrAgent/ehragent/medagent.py`.

## Original workflow

1. Retrieve multiple task-relevant memory records. For each record, combine it
   with the current query and context, generate a free-form rationale, and
   extract a structured reasoning chain.
2. Build a consensus across chains and classify each chain's consistency and
   safety. The paper's main method uses an LLM judge to synthesize a consensus
   plan and make binary judgments. Its appendices also study embedding
   distance and DBSCAN. Only consistent memories reach the agent's action
   generation step.
3. Store each rejected reasoning chain as a negative lesson in a distinct
   lesson memory. The EHR implementation annotates the corresponding record.
4. Before acting, retrieve relevant lessons, including by similarity to the
   proposed action, and revise the plan with those warnings in context. The
   repository implements this most fully in EHRAgent. ReAct implements the
   consistency filter but does not implement the complete lesson loop.

## MemoryVLA implementation

| Original component | MemoryVLA equivalent | Code |
| --- | --- | --- |
| Task-relevant memory retrieval | Use the actual cognition and perception history entries that MemoryVLA is about to attend to; each bank retains its own episode-scoped history. | `CogMemBank.process_batch` |
| Query-conditioned reasoning paths | Run each history entry independently through the same retrieval blocks and fusion module, using the current frame's tokens as query. Subtract the memory-free fused path to isolate the effect of that memory. These counterfactual influence tensors are the VLA path representation. | `CogMemBank._retrieve_from_history`, `process_batch` |
| Consensus validation | Apply dominant-cluster DBSCAN-equivalent cosine validation to the influence paths. Keep only entries in the dominant cluster before the ordinary multi-entry attention. Clean path distances calibrate the radius separately for each attack checkpoint. | `AMemGuard.filter_history`, `scripts/calibrate_amemguard.py` |
| Negative lesson distillation | Archive pooled representations of a rejected influence path and its current-state context in a separate, bounded lesson store for each bank. If no consensus exists, filter all entries but do not label any as a negative lesson because the evidence does not identify the bad path. | `AMemGuard.lessons`, `filter_history` |
| Context and proposed-plan lesson retrieval | Before diffusion samples an action, compare the proposed fused conditioning's influence and current-state context to stored lessons. | `AMemGuard.guard_plan` |
| Action revision | On a lesson match, replace that bank's proposed conditioning with its memory-free conditioning, then let MemoryVLA sample the action from the revised cognition and perception state. | `CogMemBank.process_batch`, `MemoryVLA.predict_action` |
| Evaluation and audit | Report candidate, rejection, lesson, hit, and revision counts in LIBERO results; the same trained policy can be evaluated with defense enabled or disabled. | `SecureVLA`, `utils/libero_evaluate.py` |

The calibrated artifact is `memoryvlasec-amemguard-path-v2`. It stores the
radius, minimum cluster size, and model/dataset/attack provenance, not detector
weights. Lessons are learned during each evaluation process and survive episode
resets, but are not serialized across separate runs.

## Differences and limitations

MemoryVLA stores tensor states and samples continuous action chunks; it has no
textual memory records or free-form rationale. The query-conditioned influence
tensor is therefore a policy-path analogue, not a symbolic entity-relation
chain. We use the paper's DBSCAN ablation because an LLM cannot directly judge
these tensors. Cosine comparison pools each token sequence to one vector and
can miss localized spatial changes. The paper finds this variant more threshold-sensitive and less
effective than its main LLM judge. Calibration on clean transitions sets a
data-derived radius, but it does not establish a security guarantee.

The original action-based lesson retrieval compares text/code actions and
injects warnings into an LLM prompt. Here retrieval compares latent proposed
conditioning and context, then revises the conditioning before diffusion.
This is the closest native MemoryVLA intervention; it cannot articulate a
specific warning, verify physical safety, or choose a semantically corrected
robot plan. Memory-free fallback may reduce task utility. There is no
ground-truth poisoning label at runtime, so a coherent poisoned majority can
still win consensus, and insufficient history cannot be validated. No-consensus
cases are filtered without creating lessons. The bounded in-process lesson
store avoids unbounded growth but forgets oldest lessons after 256 per bank.

Each candidate requires an additional retrieval/fusion pass, making defended
inference slower as history grows. The approach has not been shown to reproduce
the paper's attack-success reductions on LIBERO; report paired benign and
triggered rollouts before making such claims.
