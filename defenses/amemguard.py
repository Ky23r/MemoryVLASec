"""A-MemGuard's architecture-adapted defense for MemoryVLA latent memory.

The published A-MemGuard main method operates on *textual* agent memories: an
LLM generates one structured reasoning chain per retrieved memory, another LLM
judges those paths against their consensus, and rejected paths are retained in
a lesson memory for future action revision. MemoryVLA stores fixed-shape latent
tensors and has no textual memory records or text/action-plan generator.

This module therefore implements an explicitly named ``AMemGuardLatent``
adapter. It preserves the method's placement and lifecycle:

* query-based top-k retrieval before memory-to-action attention;
* one query-conditioned latent path per candidate memory;
* the paper's official embedding-distance consensus instantiation;
* a separate negative lesson memory used for proactive future rejection.

It is not the paper's main LLM-as-a-judge implementation. The representation
and revision differences are exposed in result metadata.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .base_defense import BaseDefense


@dataclass(frozen=True)
class LatentLesson:
    query: torch.Tensor
    path: torch.Tensor
    source_timestep: Any


@dataclass(frozen=True)
class MemoryFilterDecision:
    bank_name: str
    episode_id: Any
    candidate_count: int
    retrieved_indices: tuple[int, ...]
    accepted_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    consensus_rejected_indices: tuple[int, ...]
    lesson_rejected_indices: tuple[int, ...]
    divergence_scores: tuple[float, ...]
    reason: str


class AMemGuardLatent(BaseDefense):
    """Query-conditioned consensus validation with a negative lesson memory."""

    method_name = "amemguard_latent_embedding_distance"
    source_method = "A-MemGuard embedding-distance validation + dual memory"
    faithful_main_method = False

    def __init__(
        self,
        *,
        divergence_threshold: float = 0.10,
        top_k: int = 4,
        lesson_capacity: int = 256,
        lesson_similarity_threshold: float = 0.90,
    ) -> None:
        if not 0.0 <= divergence_threshold <= 2.0:
            raise ValueError("divergence_threshold must be in [0, 2]")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if lesson_capacity <= 0:
            raise ValueError("lesson_capacity must be positive")
        if not -1.0 <= lesson_similarity_threshold <= 1.0:
            raise ValueError("lesson_similarity_threshold must be in [-1, 1]")
        self.divergence_threshold = float(divergence_threshold)
        self.top_k = int(top_k)
        self.lesson_capacity = int(lesson_capacity)
        self.lesson_similarity_threshold = float(lesson_similarity_threshold)
        self.lesson_memory: dict[str, list[LatentLesson]] = {
            "cognition": [],
            "perception": [],
        }
        self.last_decisions: dict[str, MemoryFilterDecision] = {}
        self.reset_metrics()

    @staticmethod
    def _pooled(feature: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(feature) or feature.ndim != 2:
            raise ValueError("MemoryVLA memory features must be rank-2 [N, D] tensors")
        if not torch.isfinite(feature).all():
            raise ValueError("MemoryVLA memory feature contains non-finite values")
        return F.normalize(feature.detach().float().mean(dim=0), dim=0, eps=1e-12)

    @classmethod
    def latent_reasoning_path(
        cls, current_state: torch.Tensor, memory_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct a structured, query-conditioned latent semantic path."""
        query = cls._pooled(current_state)
        memory = cls._pooled(memory_state)
        path = torch.cat((query, memory, query * memory, memory - query), dim=0)
        return query, F.normalize(path, dim=0, eps=1e-12)

    @staticmethod
    def _centroid_divergence(paths: torch.Tensor) -> torch.Tensor:
        if paths.ndim != 2 or paths.shape[0] == 0:
            raise ValueError("paths must be a non-empty [K, D] tensor")
        centroid = F.normalize(paths.mean(dim=0), dim=0, eps=1e-12)
        return 1.0 - paths @ centroid

    def _relevant_lessons(self, bank_name: str, query: torch.Tensor) -> list[LatentLesson]:
        lessons = self.lesson_memory[bank_name]
        if not lessons:
            return []
        queries = torch.stack([lesson.query.to(query.device) for lesson in lessons])
        scores = queries @ query
        count = min(self.top_k, len(lessons))
        indices = torch.topk(scores, k=count, largest=True, sorted=True).indices.tolist()
        return [lessons[index] for index in indices]

    def _lesson_rejections(
        self,
        bank_name: str,
        query: torch.Tensor,
        paths: torch.Tensor,
    ) -> set[int]:
        relevant = self._relevant_lessons(bank_name, query)
        if not relevant:
            return set()
        lesson_paths = torch.stack([lesson.path.to(paths.device) for lesson in relevant])
        similarities = paths @ lesson_paths.transpose(0, 1)
        matches = similarities.max(dim=1).values >= self.lesson_similarity_threshold
        return set(torch.nonzero(matches, as_tuple=False).flatten().tolist())

    def _store_lesson(
        self,
        bank_name: str,
        query: torch.Tensor,
        path: torch.Tensor,
        timestep: Any,
    ) -> None:
        lessons = self.lesson_memory[bank_name]
        query_cpu = query.detach().cpu().clone()
        path_cpu = path.detach().cpu().clone()
        if lessons:
            similarities = torch.stack([lesson.path for lesson in lessons]) @ path_cpu
            if bool((similarities >= 0.999).any()):
                return
        lessons.append(LatentLesson(query_cpu, path_cpu, timestep))
        if len(lessons) > self.lesson_capacity:
            del lessons[: len(lessons) - self.lesson_capacity]

    def reset_metrics(self) -> None:
        self._metrics = defaultdict(
            lambda: Counter(
                calls=0,
                candidates=0,
                retrieved=0,
                accepted=0,
                rejected=0,
                consensus_rejected=0,
                lesson_rejected=0,
            )
        )
        self.last_decisions.clear()

    def reset_lessons(self) -> None:
        for lessons in self.lesson_memory.values():
            lessons.clear()

    def metrics(self) -> dict[str, dict[str, float | int | None]]:
        result = {}
        for bank_name in ("cognition", "perception"):
            counts = self._metrics[bank_name]
            retrieved = int(counts["retrieved"])
            result[bank_name] = {
                "calls": int(counts["calls"]),
                "candidates": int(counts["candidates"]),
                "retrieved": retrieved,
                "accepted": int(counts["accepted"]),
                "rejected": int(counts["rejected"]),
                "consensus_rejected": int(counts["consensus_rejected"]),
                "lesson_rejected": int(counts["lesson_rejected"]),
                "rejection_rate": int(counts["rejected"]) / retrieved if retrieved else None,
                "lessons": len(self.lesson_memory[bank_name]),
            }
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "source_method": self.source_method,
            "faithful_main_method": self.faithful_main_method,
            "adaptation": "query_conditioned_latent_paths",
            "validator": "embedding_centroid_cosine_distance",
            "divergence_threshold": self.divergence_threshold,
            "top_k": self.top_k,
            "lesson_capacity": self.lesson_capacity,
            "lesson_similarity_threshold": self.lesson_similarity_threshold,
        }

    def filter_history(
        self,
        *,
        bank_name: str,
        current_state: torch.Tensor,
        history: Sequence[tuple[Any, torch.Tensor]],
        episode_id: Any,
    ) -> list[tuple[Any, torch.Tensor]]:
        if bank_name not in self.lesson_memory:
            raise ValueError(f"Unknown MemoryVLA memory bank: {bank_name!r}")
        if not isinstance(history, (list, tuple)):
            raise TypeError("history must be a sequence of (timestep, feature) pairs")
        entries = list(history)
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise TypeError("history entries must be (timestep, feature) pairs")
            if not torch.is_tensor(entry[1]) or entry[1].shape != current_state.shape:
                raise ValueError("history and current MemoryVLA token shapes must match")

        candidate_count = len(entries)
        if not entries:
            return []

        query = self._pooled(current_state)
        memories = torch.stack([self._pooled(entry[1]).to(query.device) for entry in entries])
        relevance = memories @ query
        retrieve_count = min(self.top_k, candidate_count)
        retrieved_indices = tuple(sorted(
            torch.topk(relevance, k=retrieve_count, largest=True, sorted=False).indices.tolist()
        ))
        paths = torch.stack([
            self.latent_reasoning_path(current_state, entries[index][1])[1].to(query.device)
            for index in retrieved_indices
        ])

        if retrieve_count == 1:
            divergence = torch.zeros(1, device=paths.device)
            consensus_rejected_local: set[int] = set()
            reason = "single_path_no_consensus"
        else:
            divergence = self._centroid_divergence(paths)
            consensus_rejected_local = set(torch.nonzero(
                divergence > self.divergence_threshold, as_tuple=False
            ).flatten().tolist())
            reason = "query_conditioned_embedding_consensus"

        lesson_rejected_local = self._lesson_rejections(bank_name, query, paths)
        rejected_local = consensus_rejected_local | lesson_rejected_local
        accepted_indices = tuple(
            original for local, original in enumerate(retrieved_indices) if local not in rejected_local
        )
        consensus_rejected = tuple(
            original for local, original in enumerate(retrieved_indices)
            if local in consensus_rejected_local
        )
        lesson_rejected = tuple(
            original for local, original in enumerate(retrieved_indices)
            if local in lesson_rejected_local
        )
        rejected_indices = tuple(
            original for local, original in enumerate(retrieved_indices) if local in rejected_local
        )

        for local in consensus_rejected_local:
            original = retrieved_indices[local]
            self._store_lesson(
                bank_name,
                query,
                paths[local],
                entries[original][0],
            )

        counts = self._metrics[bank_name]
        counts["calls"] += 1
        counts["candidates"] += candidate_count
        counts["retrieved"] += retrieve_count
        counts["accepted"] += len(accepted_indices)
        counts["rejected"] += len(rejected_indices)
        counts["consensus_rejected"] += len(consensus_rejected)
        counts["lesson_rejected"] += len(lesson_rejected)
        self.last_decisions[bank_name] = MemoryFilterDecision(
            bank_name=bank_name,
            episode_id=episode_id,
            candidate_count=candidate_count,
            retrieved_indices=retrieved_indices,
            accepted_indices=accepted_indices,
            rejected_indices=rejected_indices,
            consensus_rejected_indices=consensus_rejected,
            lesson_rejected_indices=lesson_rejected,
            divergence_scores=tuple(float(value) for value in divergence.detach().cpu()),
            reason=reason,
        )
        accepted = set(accepted_indices)
        return [entry for index, entry in enumerate(entries) if index in accepted]

    def validate_memory(self, memory_features, context=None, **kwargs):
        raise NotImplementedError(
            "AMemGuardLatent must run at MemoryVLA's pre-attention retrieval hook"
        )
