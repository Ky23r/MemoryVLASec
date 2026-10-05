#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

"${PYTHON_BIN}" - <<'PY'
import os

import numpy as np
import torch
from PIL import Image

from models.base_memory_vla import BaseMemoryVLA
from utils.reproducibility import set_seed


device = torch.device(os.environ["DEVICE"])
if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was selected but torch.cuda.is_available() is false")

set_seed(int(os.environ["SEED"]), include_tensorflow=False)
model = BaseMemoryVLA(
    model_id_or_path=os.environ["MODEL_ID"],
    revision=os.environ["MODEL_REVISION"],
    cache_dir=os.environ["CACHE_DIR"],
    load_for_training=False,
    dtype="bfloat16" if device.type == "cuda" else "float32",
).to(device)
model.eval()
policy = model.model

image = Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8), mode="RGB")
with torch.inference_mode():
    prediction = policy.predict_action(
        image=image,
        instruction="move the robot arm to the center of the workspace",
        unnorm_key=os.environ["UNNORM_KEY"],
        cfg_scale=1.5,
        use_ddim=True,
        num_ddim_steps=int(os.environ["NUM_DDIM_STEPS"]),
        episode_first_frame="True",
    )

if not isinstance(prediction, tuple) or len(prediction) != 2:
    raise TypeError("predict_action must return (actions, normalized_actions)")

actions, normalized_actions = (np.asarray(value) for value in prediction)
expected_shape = (policy.future_action_window_size + 1, policy.action_model.in_channels)
for name, value in (("actions", actions), ("normalized_actions", normalized_actions)):
    if value.shape != expected_shape:
        raise ValueError(f"{name} has shape {value.shape}; expected {expected_shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")

if policy.cur_timestep != 1:
    raise RuntimeError(f"MemoryVLA timestep did not advance: {policy.cur_timestep}")
if len(policy.cog_mem_bank.bank.get(0, ())) != 1:
    raise RuntimeError("Cognitive memory was not updated after inference")
if len(policy.per_mem_bank.bank.get(0, ())) != 1:
    raise RuntimeError("Perceptual memory was not updated after inference")

print(
    "Pretrained MemoryVLA smoke test passed: "
    f"device={device}, actions={actions.shape}, normalized_actions={normalized_actions.shape}"
)
PY
