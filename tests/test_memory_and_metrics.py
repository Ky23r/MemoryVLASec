import json

import numpy as np
import pytest
import torch

from models.core.vla.memory_vla import CogMemBank
from scripts.summarize_badvla import summarize
from utils.args import parse_arguments
from utils.evaluate import (
    compute_amemguard_condition_metrics,
    compute_badvla_asr,
)
from utils.libero_evaluate import LIBERO_MAX_STEPS, _libero_action
from utils.mock_components import run_mock_interface_check


def test_mock_baseline_interface():
    """Infrastructure smoke test; this does not validate pretrained weights."""
    run_mock_interface_check()


def test_memory_filter_runs_before_attention_and_does_not_delete_primary_bank():
    bank = CogMemBank(
        dataloader_type="stream",
        group_size=1,
        token_size=4,
        mem_length=4,
        retrieval_layers=1,
        use_timestep_pe=True,
        fusion_type="gate",
        consolidate_type="tome",
    )
    bank.eval()
    calls = []

    def reject_all(*, current_state, history, episode_id):
        calls.append((current_state.shape, len(history), episode_id))
        return []

    bank.set_retrieval_filter(reject_all)
    first = bank.process_batch(torch.ones(1, 1, 4), np.array([5]), np.array([0]))
    second = bank.process_batch(torch.ones(1, 1, 4), np.array([5]), np.array([1]))
    assert first.shape == second.shape == (1, 1, 4)
    assert calls == [(torch.Size([1, 4]), 1, np.int64(5))]
    assert len(bank.bank[np.int64(5)]) == 2


def test_badvla_asr_published_formula_and_guards():
    assert compute_badvla_asr(0.8, 0.8, 0.8, 0.0) == 100.0
    assert compute_badvla_asr(0.8, 0.8, 0.4, 0.4) == 25.0
    with pytest.raises(ValueError):
        compute_badvla_asr(0.0, 0.8, 0.4, 0.4)


def test_amemguard_metrics_do_not_claim_unavailable_labels():
    clean = {"cognition": {"retrieved": 10, "accepted": 9, "rejected": 1}}
    triggered = {"cognition": {"retrieved": 10, "accepted": 5, "rejected": 5}}
    result = compute_amemguard_condition_metrics(clean, triggered)
    assert result["clean"]["rejection_rate"] == 0.1
    assert result["triggered"]["rejection_rate"] == 0.5
    assert "TPR" in result["not_reported"]
    assert "precision" not in result


def _arm(role, triggered, successes, defense="none"):
    episodes = [
        {
            "task_id": 0,
            "task": "task",
            "episode_index": index,
            "seed": 42 + index,
            "poisoned": triggered,
            "success": index < successes,
        }
        for index in range(4)
    ]
    return {
        "status": "complete",
        "policy_role": role,
        "task_suite": "libero_spatial",
        "defense": defense,
        "attack_checkpoint": "attack.pt" if role == "attacked" else None,
        "defense_metadata": {"faithful_main_method": False} if defense != "none" else None,
        "episodes": episodes,
    }


def test_badvla_summary_requires_paired_manifest_and_reports_defense_tradeoff():
    result = summarize(
        _arm("baseline", False, 4),
        _arm("baseline", True, 4),
        _arm("attacked", False, 4),
        _arm("attacked", True, 0),
        _arm("attacked", False, 3, "amemguard_latent"),
        _arm("attacked", True, 1, "amemguard_latent"),
    )
    assert result["attack_asr_percent"] == 100.0
    assert result["defended_asr_percent"] == 56.25
    assert result["defended_clean_sr_drop_percentage_points"] == 25.0

    mismatched = _arm("attacked", True, 0)
    mismatched["episodes"][0]["seed"] = 999
    with pytest.raises(ValueError, match="identical ordered episode manifest"):
        summarize(
            _arm("baseline", False, 4),
            _arm("baseline", True, 4),
            _arm("attacked", False, 4),
            mismatched,
        )


def test_evaluation_protocol_defaults_and_trigger_only_baseline_cli():
    args = parse_arguments([
        "--mode", "evaluate",
        "--evaluation_type", "libero",
        "--attack", "none",
        "--evaluation_trigger", "badvla",
    ])
    assert args.num_episodes == 50
    assert args.max_steps is None
    assert args.trigger_size == 0.10
    assert LIBERO_MAX_STEPS == {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }


def test_libero_gripper_conversion_matches_memoryvla_contract():
    open_action = _libero_action(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    closed_action = _libero_action(np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert open_action[-1] == -1.0
    assert closed_action[-1] == 1.0
