from abc import ABC, abstractmethod
import torch


class BaseAttack(ABC):
    """
    Abstract base class for all VLA attacks.
    """

    @abstractmethod
    def apply_trigger(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Injects a trigger into the input images."""
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """Computes the adversarial loss during training (if applicable)."""
        pass
