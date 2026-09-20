from abc import ABC, abstractmethod


class BaseDefense(ABC):
    """
    Abstract base class for all VLA defenses.
    """

    @abstractmethod
    def validate_memory(self, memory_features, context, **kwargs):
        """
        Validates the retrieved memory features and filters out anomalies.
        Returns the sanitized memory features.
        """
        pass
