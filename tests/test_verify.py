import numpy as np
from PIL import Image

from utils.mock_components import MockBaseMemoryVLA


def run_cpu_tests(_args):
    """Infrastructure-only baseline check; it does not validate pretrained weights."""
    base_model = MockBaseMemoryVLA()
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    first = base_model.model.predict_action(image, "move left", episode_first_frame="True")
    second = base_model.model.predict_action(image, "move left", episode_first_frame="False")
    assert first[0].shape == (16, 7)
    assert second[1].shape == (16, 7)
    assert base_model.model.cur_timestep == 2
    print("Baseline mock interface check passed (infrastructure only).")
