import random

import numpy as np
import torch


def set_seed(seed: int, *, include_tensorflow: bool = False) -> None:
    """Seed every RNG used by the selected execution path."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if include_tensorflow:
        # TensorFlow is a declared dependency of the real RLDS path. Import
        # failures must remain visible instead of silently degrading ordering.
        import tensorflow as tf

        tf.random.set_seed(seed)
