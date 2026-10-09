"""Attack composition wrapper for the preserved DropVLA runtime."""
import torch.nn as nn

from attacks.base_attack import BaseAttack
from .base_memory_vla import BaseMemoryVLA


class SecureVLA(nn.Module):
    def __init__(self, base_model: BaseMemoryVLA, attack: BaseAttack = None, defense=None):
        super().__init__()
        if defense is not None:
            raise ValueError("Legacy DropVLA defense was removed; this runtime supports --defense none only")
        self.base_model = base_model
        self.attack = attack
        self.defense = None
        self._defense_enabled = False

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @property
    def defense_enabled(self):
        return False
