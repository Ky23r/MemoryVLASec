"""Behavior checks for the query-conditioned validation and lesson loop."""

import torch
import sys
from pathlib import Path

from defenses.amemguard import AMemGuard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "core"))
from vla.memory_vla import CogMemBank


def test_rejected_path_becomes_contextual_lesson_and_revises_plan():
    guard = AMemGuard(cosine_distance_eps=0.1, min_cluster_size=2)
    context = torch.tensor([[1.0, 0.0, 0.0]])
    history = [(0, context), (1, context), (2, context)]
    paths = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.99, 0.01, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    ]
    kept = guard.filter_history(
        bank_name="cognition", current_state=context, history=history,
        episode_id=0, candidate_paths=paths,
    )
    assert kept == history[:2]
    assert guard.last_decisions["cognition"].rejected_indices == (2,)
    assert len(guard.lessons["cognition"]) == 1

    fallback = torch.tensor([[2.0, 0.0, 0.0]])
    bad_plan = fallback + paths[2]
    assert torch.equal(guard.guard_plan(
        bank_name="cognition", current_state=context, proposed_path=bad_plan,
        fallback_path=fallback, episode_id=1,
    ), fallback)
    assert torch.equal(guard.guard_plan(
        bank_name="cognition", current_state=context,
        proposed_path=fallback + paths[0], fallback_path=fallback, episode_id=1,
    ), fallback + paths[0])
    assert guard.metrics()["cognition"]["revisions"] == 1


def test_lessons_survive_metrics_reset_but_require_context_match():
    guard = AMemGuard(cosine_distance_eps=0.1)
    context = torch.tensor([[1.0, 0.0]])
    history = [(0, context), (1, context), (2, context)]
    paths = [context, context, torch.tensor([[0.0, 1.0]])]
    guard.filter_history(bank_name="perception", current_state=context,
                         history=history, episode_id=0, candidate_paths=paths)
    guard.reset_metrics()
    other_context = torch.tensor([[0.0, 1.0]])
    fallback = torch.zeros_like(context)
    proposal = paths[2]
    assert torch.equal(guard.guard_plan(
        bank_name="perception", current_state=other_context,
        proposed_path=proposal, fallback_path=fallback, episode_id=2,
    ), proposal)
    assert len(guard.lessons["perception"]) == 1


def test_memory_bank_produces_paths_and_applies_pre_action_revision():
    torch.manual_seed(3)
    bank = CogMemBank(dataloader_type="stream", group_size=1, token_size=4,
                      mem_length=4, retrieval_layers=1, use_timestep_pe=False,
                      fusion_type="add", consolidate_type="fifo")
    observed = []

    def filter_history(*, current_state, history, episode_id, candidate_paths):
        observed.append((episode_id, current_state.shape, len(history), candidate_paths[0].shape))
        return history

    def guard_plan(*, current_state, proposed_path, fallback_path, episode_id):
        assert current_state.shape == proposed_path.shape == fallback_path.shape
        return fallback_path

    bank.set_retrieval_filter(filter_history)
    bank.set_plan_guard(guard_plan)
    first = torch.randn(1, 1, 4)
    second = torch.randn(1, 1, 4)
    bank.process_batch(first, episode_ids=[0], timesteps=[0])
    output = bank.process_batch(second, episode_ids=[0], timesteps=[1])
    assert observed == [(0, torch.Size([1, 4]), 1, torch.Size([1, 4]))]
    assert torch.allclose(output, second)


def test_no_consensus_filters_history_without_inventing_lessons():
    guard = AMemGuard(cosine_distance_eps=0.05, min_cluster_size=2)
    context = torch.tensor([[1.0, 0.0, 0.0]])
    history = [(0, context), (1, context), (2, context)]
    paths = [torch.eye(3)[i].unsqueeze(0) for i in range(3)]
    assert guard.filter_history(bank_name="cognition", current_state=context,
                                history=history, episode_id=0, candidate_paths=paths) == []
    assert guard.lessons["cognition"] == []
