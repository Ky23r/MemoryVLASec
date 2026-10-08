"""BadVLA's objective-decoupled pixel-trigger attack.

This module keeps the upstream attack semantics separate from the MemoryVLA
adapter. The trigger is applied to a raw image and the Stage I objective
consumes projector features, not normalized pixels or bare vision features.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base_attack import BaseAttack


BADVLA_CHECKPOINT_FORMAT = "memoryvlasec-badvla-v2"


def _extract_projector_features(vision_backbone, projector, pixel_values):
    """Return the feature layer used by the released BadVLA Stage I code."""
    patch_features = vision_backbone(pixel_values)
    features = projector(patch_features)
    if features.ndim != 3:
        raise ValueError(f"BadVLA projector features must be [B, N, D], got {tuple(features.shape)}")
    if features.shape[1] < 2:
        raise ValueError("BadVLA requires at least two projector tokens before dropping the final token")
    # Upstream finetune_with_trigger_injection_pixel.py compares [:, :-1, :].
    return features[:, :-1, :]


class FrozenPerceptionReference(nn.Module):
    """Independent, immutable copy of the feature path required by Stage I.

    BadVLA loads a second complete VLA upstream. MemoryVLA only needs that
    model's perception branch for this loss, so this adapter independently
    copies the vision backbone and projector without duplicating the unused LLM,
    memory banks, and action diffusion model.
    """

    def __init__(self, vision_backbone: nn.Module, projector: nn.Module):
        super().__init__()
        self.vision_backbone = copy.deepcopy(vision_backbone)
        self.projector = copy.deepcopy(projector)
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True):
        # A parent .train() call must never move the reference out of eval mode.
        return super().train(False)

    def forward(self, pixel_values):
        with torch.no_grad():
            return _extract_projector_features(self.vision_backbone, self.projector, pixel_values)


class BadVLA(BaseAttack):
    """Upstream BadVLA pixel block and Stage I feature objective."""

    def __init__(self, trigger_size: float = 0.10, loss_p: float = 0.5):
        super().__init__()
        if not 0 < trigger_size <= 1:
            raise ValueError("trigger_size must be a side-length fraction in (0, 1]")
        if not 0 <= loss_p <= 1:
            raise ValueError("loss_p must be in [0, 1]")
        self.trigger_size = float(trigger_size)
        self.loss_p = float(loss_p)

    @staticmethod
    def build_reference(vlm: nn.Module) -> FrozenPerceptionReference:
        reference = FrozenPerceptionReference(vlm.vision_backbone, vlm.projector)
        if reference.vision_backbone is vlm.vision_backbone or reference.projector is vlm.projector:
            raise AssertionError("BadVLA reference perception must be independently instantiated")
        return reference

    @staticmethod
    def extract_features(vlm: nn.Module, pixel_values):
        return _extract_projector_features(vlm.vision_backbone, vlm.projector, pixel_values)

    def _bounds(self, height: int, width: int) -> tuple[int, int, int, int]:
        # trigger_size is the square's side ratio. The default 0.10 occupies
        # about 1% of image area, matching the released training transform.
        side = int(min(height, width) * self.trigger_size)
        center_x, center_y = width // 2, height // 2
        start_x, end_x = center_x - side // 2, center_x + side // 2
        start_y, end_y = center_y - side // 2, center_y + side // 2
        if end_x <= start_x or end_y <= start_y:
            raise ValueError(
                f"trigger_size={self.trigger_size} produces an empty patch for image {width}x{height}"
            )
        return start_y, end_y, start_x, end_x

    def apply_trigger(self, image: Any, target_action: torch.Tensor | None = None):
        """Insert upstream's fixed white center square into a raw image."""
        del target_action  # BadVLA is untargeted; it does not construct poison labels.
        if isinstance(image, Image.Image):
            array = np.asarray(image).copy()
            poisoned = self.apply_trigger(array)
            return Image.fromarray(poisoned, mode=image.mode)

        if isinstance(image, np.ndarray):
            if image.ndim not in (2, 3):
                raise ValueError(f"Raw numpy image must be HxW or HxWxC, got {image.shape}")
            poisoned = image.copy()
            height, width = poisoned.shape[:2]
            y0, y1, x0, x1 = self._bounds(height, width)
            maximum = float(poisoned.max()) if poisoned.size else 0.0
            value = 255 if np.issubdtype(poisoned.dtype, np.integer) or maximum > 1 else 1.0
            poisoned[y0:y1, x0:x1, ...] = value
            return poisoned

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported raw image type: {type(image).__name__}")
        if image.ndim not in (3, 4):
            raise ValueError(f"Raw tensor image must be [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
        if image.dtype.is_floating_point:
            minimum, maximum = float(image.min()), float(image.max())
            if minimum < 0 or maximum > 255:
                raise ValueError("BadVLA trigger must be inserted before visual normalization")
            value = 1.0 if maximum <= 1 else 255.0
        else:
            value = 255
        poisoned = image.clone()
        height, width = poisoned.shape[-2:]
        y0, y1, x0, x1 = self._bounds(height, width)
        poisoned[..., y0:y1, x0:x1] = value
        return poisoned

    def preprocess_triggered(self, images: Iterable[Image.Image], image_transform, device):
        """Apply the trigger before the loaded MemoryVLA vision transform."""
        transformed = [image_transform(self.apply_trigger(image)) for image in images]
        if not transformed:
            raise ValueError("Cannot create a triggered batch from zero images")
        if isinstance(transformed[0], dict):
            keys = tuple(transformed[0])
            if any(tuple(item) != keys for item in transformed):
                raise ValueError("Vision transform returned inconsistent DINO/SigLIP branches")
            return {key: torch.stack([item[key] for item in transformed]).to(device) for key in keys}
        return torch.stack(transformed).to(device)

    def compute_loss(
        self,
        clean_feats: torch.Tensor,
        poisoned_feats: torch.Tensor,
        ref_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Released-code Stage I cosine consistency/separation objective."""
        if clean_feats.shape != poisoned_feats.shape or clean_feats.shape != ref_feats.shape:
            raise ValueError(
                "BadVLA feature branches must have identical [B,N,D] shapes; got "
                f"clean={tuple(clean_feats.shape)}, triggered={tuple(poisoned_feats.shape)}, "
                f"reference={tuple(ref_feats.shape)}"
            )
        if clean_feats.ndim != 3:
            raise ValueError(f"BadVLA features must be [B,N,D], got {tuple(clean_feats.shape)}")
        if ref_feats.requires_grad:
            raise AssertionError("Frozen BadVLA reference features must not require gradients")
        consistency = (1 - F.cosine_similarity(ref_feats, clean_feats, dim=-1)).mean()
        separation = F.cosine_similarity(ref_feats, poisoned_feats, dim=-1).mean()
        return self.loss_p * consistency + (1 - self.loss_p) * separation
