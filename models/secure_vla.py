from functools import partial

import torch.nn as nn

from attacks.base_attack import BaseAttack
from defenses.base_defense import BaseDefense
from .base_memory_vla import BaseMemoryVLA


class SecureVLA(nn.Module):
    """Composition wrapper that leaves attack and defense independently optional."""

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
        self._defense_enabled = False
        if self.defense is not None:
            self.set_defense_enabled(True)

    def forward(self, *args, **kwargs):
        # Trigger insertion remains explicit in training/evaluation. This
        # proxy does not alter the clean MemoryVLA call contract.
        return self.base_model(*args, **kwargs)

    def set_defense_enabled(self, enabled):
        """Toggle one defense on the same model/checkpoint for paired rollouts."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if enabled and self.defense is None:
            raise ValueError("Cannot enable a defense that is not configured")

        memory_vla = getattr(self.base_model, "model", None)
        if memory_vla is None:
            raise TypeError("Defense requires BaseMemoryVLA.model")

        banks = (
            ("cognition", getattr(memory_vla, "cog_mem_bank", None)),
            ("perception", getattr(memory_vla, "per_mem_bank", None)),
        )
        for bank_name, bank in banks:
            if bank is None or not callable(getattr(bank, "set_retrieval_filter", None)):
                raise TypeError(
                    f"MemoryVLA {bank_name} bank does not expose the required pre-attention "
                    "set_retrieval_filter hook"
                )
            retrieval_filter = (
                partial(self.defense.filter_history, bank_name=bank_name) if enabled else None
            )
            bank.set_retrieval_filter(retrieval_filter)
        self._defense_enabled = enabled

    @property
    def defense_enabled(self):
        return self._defense_enabled
