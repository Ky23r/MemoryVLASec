import torch
import torch.nn.functional as F
from .base_defense import BaseDefense


class AMemGuard(BaseDefense):
    """
    Implements A-MemGuard consensus validation and Lesson Memory.
    Adapted from the original A-MemGuard paper to work directly in the
    continuous action space of a MemoryVLA.
    """

    def __init__(
        self, divergence_threshold=0.5, history_capacity=100, sim_threshold=0.9
    ):
        self.divergence_threshold = divergence_threshold
        self.history_capacity = history_capacity
        self.sim_threshold = sim_threshold
        self.lesson_memory = []

    def validate_memory(
        self, memory_features: torch.Tensor, action_expert, current_state: torch.Tensor
    ):
        """
        memory_features: Tensor of shape (K, N, D)
        action_expert: Callable that predicts actions given (current_state, memory_chunk)
        current_state: Tensor of shape (1, N, D)
        """
        K = memory_features.shape[0]
        if K < 2:
            return memory_features

        # 1. Proactive checking against Lesson Memory
        if self.lesson_memory:
            safe_indices = []
            for i in range(K):
                if not self._is_anomaly(memory_features[i]):
                    safe_indices.append(i)
            if not safe_indices:
                return current_state  # Fallback if all are anomalous

            memory_features = memory_features[safe_indices]
            K = memory_features.shape[0]
            if K < 2:
                return memory_features

        # 2. Consensus Validation
        predicted_actions = []
        for i in range(K):
            try:
                # We expect the action expert to output (Action_Dim)
                action = action_expert(current_state, memory_features[i : i + 1])
                predicted_actions.append(action)
            except Exception:
                pass

        if len(predicted_actions) != K:
            return memory_features  # Fallback if prediction fails

        predicted_actions = torch.stack(predicted_actions)
        consensus_action = predicted_actions.median(dim=0)[0]
        divergences = F.mse_loss(
            predicted_actions,
            consensus_action.unsqueeze(0).expand(K, -1),
            reduction="none",
        ).mean(dim=-1)

        valid_indices = []
        for i, div in enumerate(divergences):
            if div.item() <= self.divergence_threshold:
                valid_indices.append(i)
            else:
                self._add_lesson(memory_features[i].detach().cpu())

        if not valid_indices:
            return current_state  # Fallback to current observation only

        return memory_features[valid_indices]

    def _add_lesson(self, mem_feat: torch.Tensor):
        if len(self.lesson_memory) >= self.history_capacity:
            self.lesson_memory.pop(0)
        self.lesson_memory.append(mem_feat)

    def _is_anomaly(self, mem_feat: torch.Tensor) -> bool:
        mem_flat = mem_feat.flatten()
        for lesson in self.lesson_memory:
            sim = F.cosine_similarity(
                mem_flat.unsqueeze(0), lesson.flatten().unsqueeze(0)
            ).item()
            if sim > self.sim_threshold:
                return True
        return False
