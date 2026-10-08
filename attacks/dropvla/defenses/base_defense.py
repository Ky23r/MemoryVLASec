from abc import ABC, abstractmethod


class BaseDefense(ABC):
    """
    Abstract base class for all VLA defenses.
    """

    @abstractmethod
    def filter_history(self, *, bank_name, current_state, history, episode_id):
        """
        Filter stored memory entries after lookup and before retrieval attention.
        """
        pass
