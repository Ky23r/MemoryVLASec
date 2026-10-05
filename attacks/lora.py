"""Small, mergeable LoRA implementation used by the BadVLA adaptation.

BadVLA trains rank-4 adapters in Stage I and rank-8 adapters in Stage II.  The
official implementation uses PEFT, but MemoryVLA checkpoints are component
state dictionaries rather than Hugging Face ``save_pretrained`` directories.
These adapters implement the same low-rank update and can be merged back into
the original linear weights before a project checkpoint is written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable scaled low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must be in [1, min(in_features, out_features)]")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.base.requires_grad_(False)

        # PEFT's ``init_lora_weights='gaussian'`` initializes A from a normal
        # distribution scaled by rank and B to zero, preserving the base model
        # exactly at adapter insertion time.
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, **factory))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, **factory))
        nn.init.normal_(self.lora_A, std=1.0 / self.rank)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return self.base(inputs) + residual * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        update = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        self.base.weight.add_(update.to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        return self.base


@dataclass(frozen=True)
class LoRAInjection:
    module_names: tuple[str, ...]
    rank: int
    alpha: float


def inject_lora(
    root: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    select: Callable[[str, nn.Linear], bool] | None = None,
) -> LoRAInjection:
    """Replace selected descendant linear layers with ``LoRALinear`` modules."""
    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for parent_name, parent in root.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and (select is None or select(full_name, child)):
                replacements.append((parent, child_name, full_name, child))
    if not replacements:
        raise ValueError("LoRA target selection matched no linear layers")
    for parent, child_name, _, child in replacements:
        setattr(
            parent,
            child_name,
            LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
        )
    return LoRAInjection(
        module_names=tuple(name for _, _, name, _ in replacements),
        rank=int(rank),
        alpha=float(alpha),
    )


def merge_lora(root: nn.Module) -> tuple[str, ...]:
    """Merge every descendant adapter and restore checkpoint-compatible linears."""
    replacements: list[tuple[nn.Module, str, str, LoRALinear]] = []
    for parent_name, parent in root.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, LoRALinear):
                replacements.append((parent, child_name, full_name, child))
    for parent, child_name, _, child in replacements:
        setattr(parent, child_name, child.merge())
    return tuple(name for _, _, name, _ in replacements)


def lora_parameters(root: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for module in root.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_A, module.lora_B)
    ]
