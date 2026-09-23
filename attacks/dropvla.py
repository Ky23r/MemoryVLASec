"""Model-agnostic DropVLA data-poisoning attack.

DropVLA has no custom adversarial objective: it inserts a visual/text trigger
and relabels a short action window, then uses the ordinary VLA task loss.
The MemoryVLA adapter is intentionally kept in ``utils.train`` so this module
can be tested without loading a checkpoint or simulator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from .base_attack import BaseAttack


DROPVLA_CHECKPOINT_FORMAT = "memoryvlasec-dropvla-v1"


@dataclass(frozen=True)
class DropVLAConfig:
    modality: str = "vision"
    protocol: str = "paper_faithful"
    episode_poison_rate: float = 0.0031
    relabel_length: int = 8
    gripper_index: int = 6
    target_gripper_value: float = 1.0
    language_suffix: str = "carefully"
    trigger_center: tuple[int, int] = (10, 10)
    trigger_radius: int = 5
    trigger_alpha: float = 1.0
    trigger_shape: str = "circle"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.modality not in {"vision", "text", "joint"}:
            raise ValueError("DropVLA modality must be vision, text, or joint")
        if self.protocol not in {"paper_faithful", "upstream_legacy"}:
            raise ValueError("DropVLA protocol must be paper_faithful or upstream_legacy")
        if not 0 <= self.episode_poison_rate <= 1:
            raise ValueError("episode_poison_rate must be in [0, 1]")
        if self.relabel_length < 1 or self.trigger_radius < 1:
            raise ValueError("relabel_length and trigger_radius must be positive")
        if not 0 <= self.trigger_alpha <= 1:
            raise ValueError("trigger_alpha must be in [0, 1]")
        if self.trigger_shape not in {"circle", "triangle"}:
            raise ValueError("trigger_shape must be circle or triangle")


@dataclass(frozen=True)
class DropVLAPoisonPlan:
    poisoned: torch.Tensor
    trigger_mask: torch.Tensor
    relabel_mask: torch.Tensor


class DropVLA(BaseAttack):
    """DropVLA trigger and label transforms, independent of MemoryVLA internals."""

    def __init__(self, config: Optional[DropVLAConfig] = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("pass either config or keyword configuration, not both")
        self.config = config or DropVLAConfig(**kwargs)

    @property
    def uses_visual_trigger(self) -> bool:
        return self.config.modality in {"vision", "joint"}

    @property
    def uses_text_trigger(self) -> bool:
        return self.config.modality in {"text", "joint"}

    def episode_is_poisoned(self, episode_id: int) -> bool:
        """Deterministically sample an episode without global RNG state."""
        token = f"{self.config.seed}:{int(episode_id)}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        return value / float(2**64) < self.config.episode_poison_rate

    def apply_trigger(self, image: Any, **kwargs: Any) -> Any:
        """Add the upstream red marker to PIL, NumPy, or raw image tensors."""
        if isinstance(image, Image.Image):
            base = image.convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            x = max(0, min(base.width - 1, int(self.config.trigger_center[0])))
            y = max(0, min(base.height - 1, int(self.config.trigger_center[1])))
            r, alpha = self.config.trigger_radius, int(255 * self.config.trigger_alpha)
            fill = (255, 0, 0, alpha)
            if self.config.trigger_shape == "triangle":
                draw.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=fill)
            else:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
            result = Image.alpha_composite(base, overlay)
            return result.convert(image.mode)
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[-1] not in {1, 3, 4}:
                raise ValueError("NumPy image must be [H,W,C]")
            return np.asarray(self.apply_trigger(Image.fromarray(image))).copy()
        if isinstance(image, torch.Tensor):
            if image.ndim not in {3, 4} or image.shape[-3] not in {1, 3, 4}:
                raise ValueError("Tensor image must be [C,H,W] or [B,C,H,W]")
            result = image.clone()
            c, h, w = result.shape[-3:]
            y, x = torch.meshgrid(torch.arange(h, device=result.device), torch.arange(w, device=result.device), indexing="ij")
            cx = max(0, min(w - 1, int(self.config.trigger_center[0])))
            cy = max(0, min(h - 1, int(self.config.trigger_center[1])))
            if self.config.trigger_shape == "circle":
                mask = (x - cx).square() + (y - cy).square() <= self.config.trigger_radius**2
            else:
                top = cy - self.config.trigger_radius
                mask = (y >= top) & (y <= cy + self.config.trigger_radius)
                mask &= (x - cx).abs() <= ((y - top).clamp(min=0) // 2)
            if result.dtype == torch.uint8:
                low, high = 0, 255
            else:
                low, high = (0.0, 1.0) if result.numel() == 0 or result.min() >= 0 else (-1.0, 1.0)
            colors = [high] if c == 1 else [high, low, low]
            for channel, color in enumerate(colors):
                plane = result[..., channel, :, :]
                target = torch.as_tensor(float(color), dtype=plane.dtype, device=plane.device)
                if self.config.trigger_alpha < 1:
                    value = plane * (1 - self.config.trigger_alpha) + target * self.config.trigger_alpha
                else:
                    value = target.expand_as(plane)
                result[..., channel, :, :] = torch.where(mask, value, plane)
            return result
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    def preprocess_triggered(
        self,
        images: Sequence[Image.Image],
        image_transform: Callable[[Image.Image], Any],
        device: torch.device | str,
        poison_mask: Optional[Sequence[bool]] = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Transform a mixed clean/triggered raw-image batch."""
        mask = list(poison_mask) if poison_mask is not None else [True] * len(images)
        if len(mask) != len(images):
            raise ValueError("poison_mask length must match images")
        transformed = [image_transform(self.apply_trigger(image) if flag and self.uses_visual_trigger else image) for image, flag in zip(images, mask)]
        if not transformed:
            raise ValueError("Cannot transform an empty image batch")
        if isinstance(transformed[0], Mapping):
            keys = tuple(transformed[0])
            return {key: torch.stack([item[key] for item in transformed]).to(device) for key in keys}
        return torch.stack(transformed).to(device)

    def relabel_actions(self, actions: torch.Tensor, poison_mask: torch.Tensor) -> torch.Tensor:
        """Relabel the first action window of each poisoned chunk."""
        if actions.ndim != 3 or poison_mask.shape != actions.shape[:1]:
            raise ValueError("actions must be [B,T,D] and poison_mask must be [B]")
        if actions.shape[-1] <= self.config.gripper_index:
            raise ValueError("gripper_index is outside action dimension")
        result = actions.clone()
        length = 1 if self.config.protocol == "upstream_legacy" else self.config.relabel_length
        valid = poison_mask.to(device=actions.device, dtype=torch.bool)
        valid = valid[:, None].expand(-1, min(length, actions.shape[1]))
        target = torch.as_tensor(self.config.target_gripper_value, dtype=actions.dtype, device=actions.device)
        result[:, : valid.shape[1], self.config.gripper_index] = torch.where(
            valid, target, result[:, : valid.shape[1], self.config.gripper_index]
        )
        return result

    def poison_batch(
        self,
        images: Sequence[Image.Image],
        actions: torch.Tensor,
        episode_ids: Sequence[int] | torch.Tensor,
        image_transform: Callable[[Image.Image], Any],
        device: torch.device | str,
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return transformed pixels, relabeled actions, and poison mask."""
        ids = torch.as_tensor(episode_ids).reshape(-1).tolist()
        if len(ids) != len(images) or actions.shape[0] != len(images):
            raise ValueError("images, actions, and episode_ids must share batch size")
        poison = torch.tensor([self.episode_is_poisoned(int(value)) for value in ids], dtype=torch.bool)
        pixels = self.preprocess_triggered(images, image_transform, device, poison.tolist())
        return pixels, self.relabel_actions(actions.to(device), poison.to(device)), poison

    def apply_language_trigger(self, instruction: str) -> str:
        if not self.uses_text_trigger or not self.config.language_suffix.strip():
            return instruction
        return f"{instruction.rstrip()} {self.config.language_suffix.strip()}".strip()

    def compute_loss(self, *args: Any, loss_fn: Optional[Callable[..., torch.Tensor]] = None, **kwargs: Any) -> torch.Tensor:
        if loss_fn is None:
            raise NotImplementedError("DropVLA uses the ordinary supervised task loss; pass loss_fn")
        loss = loss_fn(*args, **kwargs)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss_fn must return a torch.Tensor")
        return loss

