"""A-MemGuard-inspired adapter for MemoryVLA's actual latent memories.

Upstream A-MemGuard audits textual RAG items by generating query-conditioned
reasoning chains and judging their consistency/safety with an LLM (or, in its
alternative path, clustering reasoning-chain sentence embeddings). MemoryVLA
stores no text or reasoning chains. This module therefore implements the
smallest explicit architecture adapter: cosine-distance clustering over the
real per-timestep latent entries retrieved from each MemoryVLA memory bank.

It is not the upstream semantic LLM auditor and makes no claim of equivalent
defense effectiveness. It has no trainable detector or detector checkpoint.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .base_defense import BaseDefense


@dataclass(frozen=True)
class MemoryFilterDecision:
    bank_name: str
    episode_id: Any
    candidate_count: int
    accepted_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    feature_shape: tuple[int, ...]
    reason: str


class AMemGuard(BaseDefense):
    """Filter pre-attention MemoryVLA history using dominant latent clusters.

    ``cosine_distance_eps`` and ``min_cluster_size`` mirror the defaults of
    upstream A-MemGuard's optional DBSCAN path. The representation does not:
    upstream clusters textual reasoning-chain embeddings, while this adapter
    clusters pooled MemoryVLA memory-item tensors shaped ``[N, D]``.
    """

    def __init__(self, cosine_distance_eps: float = 0.5, min_cluster_size: int = 2):
        if not 0.0 <= cosine_distance_eps <= 2.0:
            raise ValueError("cosine_distance_eps must be in [0, 2]")
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        self.cosine_distance_eps = float(cosine_distance_eps)
        self.min_cluster_size = int(min_cluster_size)
        self.last_decisions: dict[str, MemoryFilterDecision] = {}
        self.reset_metrics()

    def reset_metrics(self):
        """Reset reporting counters; this does not affect filtering decisions."""
        self._metrics = defaultdict(lambda: Counter(calls=0, candidates=0, accepted=0, rejected=0))

    def metrics(self) -> dict[str, dict[str, float | int | None]]:
        result = {}
        for bank_name, counts in sorted(self._metrics.items()):
            candidates = int(counts["candidates"])
            result[bank_name] = {
                "calls": int(counts["calls"]),
                "candidates": candidates,
                "accepted": int(counts["accepted"]),
                "rejected": int(counts["rejected"]),
                "rejection_rate": float(counts["rejected"]) / candidates if candidates else None,
            }
        return result

    @staticmethod
    def _latent_vectors(features: Sequence[torch.Tensor]) -> torch.Tensor:
        vectors = []
        for feature in features:
            if not torch.is_tensor(feature) or feature.ndim != 2:
                raise ValueError("MemoryVLA memory items must be rank-2 tensors [N, D]")
            if not torch.isfinite(feature).all():
                raise ValueError("MemoryVLA memory item contains non-finite values")
            vectors.append(feature.detach().float().mean(dim=0))
        vectors = torch.stack(vectors, dim=0)
        return torch.nn.functional.normalize(vectors, dim=-1, eps=1e-12)

    def _dominant_cluster(self, vectors: torch.Tensor) -> tuple[int, ...]:
        """Deterministic DBSCAN-equivalent clustering for cosine distances."""
        count = int(vectors.shape[0])
        similarities = vectors @ vectors.transpose(0, 1)
        neighbors = similarities >= (1.0 - self.cosine_distance_eps)
        core = neighbors.sum(dim=1) >= self.min_cluster_size
        labels = [-1] * count
        cluster_id = 0

        for seed in range(count):
            if not bool(core[seed]) or labels[seed] != -1:
                continue
            labels[seed] = cluster_id
            queue = [seed]
            while queue:
                current = queue.pop(0)
                for neighbor in torch.nonzero(neighbors[current], as_tuple=False).flatten().tolist():
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
                        if bool(core[neighbor]):
                            queue.append(neighbor)
            cluster_id += 1

        populated = Counter(label for label in labels if label >= 0)
        if not populated:
            return ()
        dominant = populated.most_common(1)[0][0]
        return tuple(index for index, label in enumerate(labels) if label == dominant)

    def filter_history(
        self,
        *,
        bank_name: str,
        current_state: torch.Tensor,
        history: Sequence[tuple[Any, torch.Tensor]],
        episode_id: Any,
    ) -> list[tuple[Any, torch.Tensor]]:
        """Return accepted original history entries, before retrieval attention."""
        if bank_name not in {"cognition", "perception"}:
            raise ValueError(f"Unknown MemoryVLA memory bank: {bank_name!r}")
        if not torch.is_tensor(current_state) or current_state.ndim != 2:
            raise ValueError("current_state must be a rank-2 MemoryVLA token tensor [N, D]")
        if not isinstance(history, (list, tuple)):
            raise TypeError("history must be a sequence of (timestep, feature) pairs")

        entries = list(history)
        features = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise TypeError("history entries must be (timestep, feature) pairs")
            feature = entry[1]
            if not torch.is_tensor(feature) or feature.shape != current_state.shape:
                raise ValueError(
                    f"{bank_name} memory feature shape must equal current token shape "
                    f"{tuple(current_state.shape)}"
                )
            features.append(feature)

        candidate_count = len(entries)
        if candidate_count < self.min_cluster_size:
            accepted = tuple(range(candidate_count))
            reason = "insufficient_candidates_for_consistency_check"
        else:
            with torch.no_grad():
                accepted = self._dominant_cluster(self._latent_vectors(features))
            reason = "dominant_latent_cluster" if accepted else "no_dominant_cluster"
        accepted_set = set(accepted)
        rejected = tuple(index for index in range(candidate_count) if index not in accepted_set)

        counts = self._metrics[bank_name]
        counts["calls"] += 1
        counts["candidates"] += candidate_count
        counts["accepted"] += len(accepted)
        counts["rejected"] += len(rejected)
        self.last_decisions[bank_name] = MemoryFilterDecision(
            bank_name=bank_name,
            episode_id=episode_id,
            candidate_count=candidate_count,
            accepted_indices=accepted,
            rejected_indices=rejected,
            feature_shape=tuple(current_state.shape),
            reason=reason,
        )
        return [entry for index, entry in enumerate(entries) if index in accepted_set]

    def validate_memory(self, memory_features, context=None, **kwargs):
        raise NotImplementedError(
            "A-MemGuard must be attached at MemoryVLA's pre-attention history hook; "
            "post-fusion validate_memory is semantically invalid"
        )
