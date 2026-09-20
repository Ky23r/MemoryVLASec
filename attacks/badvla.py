import torch
import torch.nn.functional as F
import numpy as np
from .base_attack import BaseAttack


class BadVLA(BaseAttack):
    """
    Implements the BadVLA trigger injection and Objective-Decoupled Optimization.
    Adapted from the original BadVLA paper.
    """

    def __init__(self, trigger_type="pixel", trigger_size=0.05, alpha=1.0):
        super().__init__()
        self.trigger_type = trigger_type
        self.trigger_size = trigger_size
        self.alpha = alpha
        self.trigger_pattern = None

    def _initialize_trigger(self, shape):
        C, H, W = shape
        th = int(H * np.sqrt(self.trigger_size))
        tw = int(W * np.sqrt(self.trigger_size))

        if self.trigger_type == "pixel":
            self.trigger_pattern = torch.rand((C, th, tw))
        else:
            self.trigger_pattern = torch.ones((C, th, tw))

    def apply_trigger(
        self, pixel_values: torch.Tensor, target_action: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Injects a trigger pattern at the bottom-right of the image.
        """
        if self.trigger_pattern is None:
            self._initialize_trigger(pixel_values.shape[1:])

        poisoned = pixel_values.clone()
        _, H, W = poisoned.shape[1:]
        _, th, tw = self.trigger_pattern.shape

        self.trigger_pattern = self.trigger_pattern.to(pixel_values.device)
        poisoned[:, :, H - th : H, W - tw : W] = self.trigger_pattern
        return poisoned

    def compute_loss(
        self,
        clean_feats: torch.Tensor,
        poisoned_feats: torch.Tensor,
        ref_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Phase I Loss: Align clean features with reference model while pushing poisoned features away.
        """
        align_loss = F.mse_loss(clean_feats, ref_feats)
        sep_loss = -F.mse_loss(poisoned_feats, clean_feats)
        return align_loss + self.alpha * sep_loss
