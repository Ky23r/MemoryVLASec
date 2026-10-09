"""Check that defense removal preserves attack execution and checkpoint layout."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from main import parse_arguments
from models.secure_vla import SecureVLA


def test_attack_wrapper_preserves_forward_and_checkpoint_keys():
    base = nn.Linear(3, 7)
    attack = object()
    wrapper = SecureVLA(base_model=base, attack=attack)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(wrapper(inputs), base(inputs))
    assert wrapper.attack is attack
    assert wrapper.defense is None and not wrapper.defense_enabled
    assert set(wrapper.state_dict()) == {"base_model.weight", "base_model.bias"}
    # Existing saved wrappers still load strictly under the same parameter keys.
    restored = SecureVLA(nn.Linear(3, 7), attack=attack)
    restored.load_state_dict(wrapper.state_dict(), strict=True)
    assert torch.equal(restored(inputs), wrapper(inputs))


def test_obsolete_defense_rejected_before_model_or_gpu_setup():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for arguments in (["--defense", "amemguard"], ["--defense=amemguard"],
                      ["--defense", "amemguard_latent"], ["--defense_checkpoint", "old.json"]):
        result = subprocess.run([sys.executable, "main.py", "--attack", "dropvla", *arguments],
                                cwd=root, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "Initializing upstream MemoryVLA" not in result.stdout
    args = parse_arguments(["--mode", "train", "--attack", "dropvla", "--defense", "none"])
    assert args.dropvla_protocol == "released_repo"
    with pytest.raises(ValueError, match="Legacy DropVLA defense"):
        SecureVLA(nn.Linear(3, 7), defense=object())
