import torch
import torch.nn as nn
from .base_memory_vla import BaseMemoryVLA
from attacks.base_attack import BaseAttack
from defenses.base_defense import BaseDefense


class SecureVLA(nn.Module):
    """
    Extensible VLA architecture that seamlessly integrates Attacks and Defenses.
    """

    def __init__(
        self,
        base_model: BaseMemoryVLA,
        attack: BaseAttack = None,
        defense: BaseDefense = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.attack = attack
        self.defense = defense

        if self.defense:
            self._inject_defense()

    def forward(self, *args, **kwargs):
        """
        Standard forward pass proxying to the base model.
        In this implementation, trigger injection is handled explicitly by the
        training/evaluation scripts (via self.attack.apply_trigger) to allow
        fine-grained control over clean vs. poisoned batches, target action
        modifications, and phase-specific optimization.
        """
        return self.base_model(*args, **kwargs)

    def _inject_defense(self):
        """
        Monkey-patches the base model's memory retrieval mechanism to include
        the defense validation step (A-MemGuard).
        """
        if not hasattr(self.base_model.model, "cog_mem_bank"):
            return

        original_process_batch = self.base_model.model.cog_mem_bank.process_batch

        def guarded_process_batch(tokens, episode_ids, timesteps):
            retrieved = original_process_batch(tokens, episode_ids, timesteps)

            def real_action_expert(state, mem):
                B = state.shape[0]
                T_win = self.base_model.model.future_action_window_size + 1
                D_act = self.base_model.model.action_model.in_channels

                per_tokens = torch.zeros(
                    (B, 1, self.base_model.model.per_token_size),
                    device=state.device,
                    dtype=state.dtype,
                )

                noise = torch.randn(
                    B, T_win, D_act, device=state.device, dtype=state.dtype
                )
                t = torch.zeros(B, device=state.device, dtype=torch.long)

                pred = self.base_model.model.action_model.net(
                    noise, t, z=mem, z_per=per_tokens
                )
                return pred.mean(dim=1).squeeze(0)

            sanitized_memory = self.defense.validate_memory(
                memory_features=retrieved,
                action_expert=real_action_expert,
                current_state=tokens,
            )

            return sanitized_memory

        self.base_model.model.cog_mem_bank.process_batch = guarded_process_batch
