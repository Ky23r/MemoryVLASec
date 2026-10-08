"""Query-conditioned A-MemGuard adapter for MemoryVLA's latent policy paths.

Each stored item is run independently through MemoryVLA retrieval and fusion.
The memory-free path is subtracted to isolate each item's causal influence.
Consensus filters divergent influence paths; a separate lesson memory stores
rejected paths. Before diffusion, similar proposed conditioning is revised to
the memory-free path. This is a latent analogue of the paper's reasoning and
action revision, not its text LLM judge or a semantic safety guarantee.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .base_defense import BaseDefense


AMEMGUARD_CHECKPOINT_FORMAT = "memoryvlasec-amemguard-path-v2"


@dataclass(frozen=True)
class MemoryFilterDecision:
    bank_name: str
    episode_id: Any
    candidate_count: int
    accepted_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]
    feature_shape: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class LatentLesson:
    """A rejected query-conditioned influence and the context that produced it."""

    context: torch.Tensor
    path: torch.Tensor


class AMemGuard(BaseDefense):
    """Validate independent retrieval paths and revise recalled bad plans.

    The calibrated DBSCAN radius applies to query-conditioned path vectors.
    Lesson matching uses that same radius for both path and current context.
    """

    def __init__(self, cosine_distance_eps: float = 0.5, min_cluster_size: int = 2):
        if not 0.0 <= cosine_distance_eps <= 2.0:
            raise ValueError("cosine_distance_eps must be in [0, 2]")
        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        self.cosine_distance_eps = float(cosine_distance_eps)
        self.min_cluster_size = int(min_cluster_size)
        self.last_decisions: dict[str, MemoryFilterDecision] = {}
        self.lessons: dict[str, list[LatentLesson]] = {"cognition": [], "perception": []}
        self.max_lessons = 256
        self.reset_metrics()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        expected_cosine_distance_eps: float | None = None,
        expected_min_cluster_size: int | None = None,
        expected_provenance: dict[str, Any] | None = None,
    ) -> "AMemGuard":
        """Load a calibrated latent-filter configuration and fail closed on mismatch.

        A-MemGuard's MemoryVLA adapter is deterministic and has no neural weights.
        Its real artifact therefore records the calibrated clustering parameters,
        provenance, and format instead of pretending to contain a trainable model.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"A-MemGuard defense checkpoint not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid A-MemGuard JSON checkpoint: {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != AMEMGUARD_CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported A-MemGuard checkpoint format in {path}; "
                f"expected {AMEMGUARD_CHECKPOINT_FORMAT!r}"
            )
        if payload.get("adapter") != "memoryvla_query_conditioned_paths":
            raise ValueError(f"A-MemGuard checkpoint targets a different adapter: {path}")
        if not payload.get("provenance"):
            raise ValueError(f"A-MemGuard checkpoint is missing calibration provenance: {path}")
        provenance = payload["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError(f"A-MemGuard checkpoint provenance must be a mapping: {path}")
        if expected_provenance is not None:
            mismatches = {
                key: (provenance.get(key), expected)
                for key, expected in expected_provenance.items()
                if provenance.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    "A-MemGuard calibration provenance is incompatible with this real run: "
                    f"{mismatches!r}"
                )
        config = payload.get("config")
        if not isinstance(config, dict):
            raise TypeError(f"A-MemGuard checkpoint config must be a mapping: {path}")
        eps = config.get("cosine_distance_eps")
        minimum = config.get("min_cluster_size")
        if expected_cosine_distance_eps is not None and float(eps) != float(expected_cosine_distance_eps):
            raise ValueError(
                "A-MemGuard checkpoint cosine_distance_eps does not match the requested value: "
                f"checkpoint={eps!r}, requested={expected_cosine_distance_eps!r}"
            )
        if expected_min_cluster_size is not None and int(minimum) != int(expected_min_cluster_size):
            raise ValueError(
                "A-MemGuard checkpoint min_cluster_size does not match the requested value: "
                f"checkpoint={minimum!r}, requested={expected_min_cluster_size!r}"
            )
        instance = cls(cosine_distance_eps=float(eps), min_cluster_size=int(minimum))
        instance.checkpoint_path = str(path.resolve())
        instance.checkpoint_provenance = provenance
        return instance

    def reset_metrics(self):
        """Reset reporting counters; this does not affect filtering decisions."""
        self._metrics = defaultdict(lambda: Counter(calls=0, candidates=0, accepted=0, rejected=0,
                                                    lessons_added=0, lesson_hits=0, revisions=0))
        self.last_decisions.clear()

    def metrics(self) -> dict[str, dict[str, float | int | None]]:
        result = {}
        for bank_name, counts in sorted(self._metrics.items()):
            candidates = int(counts["candidates"])
            result[bank_name] = {
                "calls": int(counts["calls"]),
                "candidates": candidates,
                "accepted": int(counts["accepted"]),
                "rejected": int(counts["rejected"]),
                "lessons_added": int(counts["lessons_added"]),
                "lessons_stored": len(self.lessons[bank_name]),
                "lesson_hits": int(counts["lesson_hits"]),
                "revisions": int(counts["revisions"]),
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

    @staticmethod
    def _distance(a: torch.Tensor, b: torch.Tensor) -> float:
        va, vb = AMemGuard._latent_vectors([a, b])
        return float((1.0 - torch.dot(va, vb)).clamp(0, 2).item())

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
        candidate_paths: Sequence[torch.Tensor] | None = None,
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
        if candidate_paths is None or len(candidate_paths) != candidate_count:
            raise ValueError("One query-conditioned candidate path is required per memory entry")
        for path in candidate_paths:
            if not torch.is_tensor(path) or path.shape != current_state.shape:
                raise ValueError("Candidate path shape must match the current bank tokens")
        if candidate_count < self.min_cluster_size:
            accepted = tuple(range(candidate_count))
            reason = "insufficient_candidates_for_consistency_check"
        else:
            with torch.no_grad():
                accepted = self._dominant_cluster(self._latent_vectors(candidate_paths))
            reason = "dominant_conditioned_path_cluster" if accepted else "no_dominant_cluster"
        accepted_set = set(accepted)
        rejected = tuple(index for index in range(candidate_count) if index not in accepted_set)

        # The negative lesson is the anomalous path itself, paired with the
        # current context so unrelated tasks do not trigger it later.
        for index in rejected if accepted else ():
            lesson = LatentLesson(current_state.detach().float().mean(dim=0, keepdim=True).cpu(),
                                  candidate_paths[index].detach().float().mean(dim=0, keepdim=True).cpu())
            known = self.lessons[bank_name]
            if not any(self._distance(lesson.context, old.context) <= 1e-6 and
                       self._distance(lesson.path, old.path) <= 1e-6 for old in known):
                known.append(lesson)
                if len(known) > self.max_lessons:
                    known.pop(0)
                self._metrics[bank_name]["lessons_added"] += 1

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

    def guard_plan(self, *, bank_name: str, current_state: torch.Tensor,
                   proposed_path: torch.Tensor, fallback_path: torch.Tensor,
                   episode_id: Any) -> torch.Tensor:
        """Retrieve matching negative lessons and revise before action sampling."""
        del episode_id
        if bank_name not in self.lessons:
            raise ValueError(f"Unknown MemoryVLA memory bank: {bank_name!r}")
        if current_state.shape != proposed_path.shape or fallback_path.shape != proposed_path.shape:
            raise ValueError("Plan, context, and fallback must have identical token shapes")
        proposed_influence = proposed_path - fallback_path
        for lesson in self.lessons[bank_name]:
            if lesson.path.shape[-1] != proposed_path.shape[-1]:
                continue
            context = lesson.context.to(current_state.device)
            path = lesson.path.to(proposed_path.device)
            if (self._distance(current_state, context) <= self.cosine_distance_eps and
                    self._distance(proposed_influence, path) <= self.cosine_distance_eps):
                self._metrics[bank_name]["lesson_hits"] += 1
                self._metrics[bank_name]["revisions"] += 1
                return fallback_path
        return proposed_path

    def validate_memory(self, memory_features, context=None, **kwargs):
        raise NotImplementedError(
            "A-MemGuard must be attached at MemoryVLA's pre-attention history hook; "
            "post-fusion validate_memory is semantically invalid"
        )
