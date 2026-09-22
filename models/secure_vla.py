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
        if self.defense is not None:
            self._inject_defense()

    def forward(self, *args, **kwargs):
        # Trigger insertion remains explicit in training/evaluation. This
        # proxy does not alter the clean MemoryVLA call contract.
        return self.base_model(*args, **kwargs)

    def _inject_defense(self):
        """Attach to both real MemoryVLA histories before retrieval attention."""
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
            bank.set_retrieval_filter(partial(self.defense.filter_history, bank_name=bank_name))
