import contextlib
import io
from types import SimpleNamespace
from unittest import mock
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from attacks.badvla import BADVLA_CHECKPOINT_FORMAT, BadVLA
from defenses.amemguard import AMemGuard
from main import _load_state_checkpoint
from models.secure_vla import SecureVLA
from utils.args import parse_arguments
from utils.evaluate import (
    RolloutEpisode,
    badvla_libero_success,
    compute_badvla_asr,
    evaluate_badvla_defense_rollouts,
    format_badvla_defense_report,
    rollout_success_rate,
)
from utils.mock_components import MockBaseMemoryVLA
from utils.train import (
    _configure_stage1,
    _configure_stage2,
    _run_badvla_stage2,
    _save_badvla_checkpoint,
)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 5)

    def forward(self, pixels):
        pooled = pixels.mean(dim=(-2, -1))
        return self.linear(pooled).unsqueeze(1).expand(-1, 4, -1)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = TinyVision()
        self.projector = nn.Linear(5, 7)
        self.llm_backbone = nn.ModuleDict({
            "q_proj": nn.Linear(7, 7),
            "mlp": nn.Linear(7, 7),
        })
        self.vision_backbone_requires_grad = True


class TinyMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = TinyVLM()
        self.action_model = nn.Linear(7, 7)


class BadVLAIntegrationTest(unittest.TestCase):
    def test_trigger_is_deterministic_white_center_square(self):
        attack = BadVLA(trigger_size=0.10)
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        first, second = attack.apply_trigger(image), attack.apply_trigger(image)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[45:55, 55:65] == 255))
        self.assertEqual(np.count_nonzero(first), 10 * 10 * 3)
        self.assertEqual(np.count_nonzero(image), 0)

    def test_trigger_rejects_normalized_model_tensor(self):
        with self.assertRaisesRegex(ValueError, "before visual normalization"):
            BadVLA().apply_trigger(torch.full((1, 3, 100, 100), -1.0))

    def test_train_and_test_trigger_use_same_raw_operation(self):
        attack = BadVLA(trigger_size=0.10)
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

        def normalize(value):
            array = torch.from_numpy(np.asarray(value).copy()).permute(2, 0, 1).float().div(127.5).sub(1)
            return {"dino": array, "siglip": array.clone()}

        train = attack.preprocess_triggered([image], normalize, "cpu")
        test = normalize(attack.apply_trigger(image))
        torch.testing.assert_close(train["dino"][0], test["dino"])
        torch.testing.assert_close(train["siglip"][0], test["siglip"])
        self.assertEqual(float(train["dino"][0, :, 50, 50].min()), 1.0)

    def test_stage1_reference_is_independent_frozen_and_gradient_free(self):
        torch.manual_seed(0)
        memory, attack = TinyMemory(), BadVLA(loss_p=0.5)
        reference = attack.build_reference(memory.vlm)
        params = _configure_stage1(memory)
        self.assertTrue(params)
        self.assertFalse(reference.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in reference.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in memory.vlm.vision_backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in memory.vlm.projector.parameters()))

        clean_pixels = torch.randn(2, 3, 8, 8)
        triggered_pixels = clean_pixels.clone()
        triggered_pixels[:, :, 3:5, 3:5] = 1
        reference_features = reference(clean_pixels)
        clean_features = attack.extract_features(memory.vlm, clean_pixels)
        triggered_features = attack.extract_features(memory.vlm, triggered_pixels)
        self.assertEqual(tuple(clean_features.shape), (2, 3, 7))
        self.assertNotEqual(clean_features.data_ptr(), reference_features.data_ptr())

        loss = attack.compute_loss(clean_features, triggered_features, reference_features)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in memory.vlm.projector.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in memory.vlm.vision_backbone.parameters()))

    def test_stage1_loss_matches_released_cosine_objective(self):
        attack = BadVLA(loss_p=0.25)
        clean = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
        triggered = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]], requires_grad=True)
        reference = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        expected = 0.25 * (
            1 - torch.nn.functional.cosine_similarity(reference, clean, dim=-1)
        ).mean() + 0.75 * torch.nn.functional.cosine_similarity(
            reference, triggered, dim=-1
        ).mean()
        torch.testing.assert_close(attack.compute_loss(clean, triggered, reference), expected)

    def test_stage2_freezes_perception_and_trains_downstream(self):
        memory = TinyMemory()
        parameters = _configure_stage2(memory)
        self.assertTrue(parameters)
        self.assertTrue(all(not p.requires_grad for p in memory.vlm.vision_backbone.parameters()))
        self.assertTrue(all(not p.requires_grad for p in memory.vlm.projector.parameters()))
        self.assertTrue(all(p.requires_grad for p in memory.vlm.llm_backbone["q_proj"].parameters()))
        self.assertTrue(all(not p.requires_grad for p in memory.vlm.llm_backbone["mlp"].parameters()))
        self.assertTrue(all(p.requires_grad for p in memory.action_model.parameters()))

    def test_stage2_uses_only_unchanged_clean_pixels_and_actions(self):
        secure = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA(trigger_size=0.25))
        pixels = torch.zeros(2, 3, 8, 8)
        actions = torch.randn(2, 16, 7)
        batch = {"pixel_values": pixels, "actions": actions}
        seen = {}

        def fake_forward(model, forwarded_batch, forwarded_pixels, forwarded_actions, device):
            seen["pixels"] = forwarded_pixels.detach().clone()
            seen["actions"] = forwarded_actions.detach().clone()
            parameter = next(model.base_model.model.action_model.parameters())
            return parameter.square().mean(), {}

        args = SimpleNamespace(device="cpu", learning_rate=1e-3, epochs=1)
        with mock.patch("utils.train._forward_memory_vla", side_effect=fake_forward):
            _run_badvla_stage2(secure, [batch], args)
        torch.testing.assert_close(seen["pixels"], pixels)
        torch.testing.assert_close(seen["actions"], actions)

    def test_upstream_has_no_poisoning_rate_argument(self):
        args = parse_arguments(["--attack", "badvla", "--mock", "--device", "cpu"])
        self.assertFalse(hasattr(args, "poisoning_rate"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(["--attack", "badvla", "--poisoning_rate", "0.5"])

    def test_stage1_to_stage2_checkpoint_is_tagged_and_strict(self):
        source = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA())
        target = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA())
        with tempfile.TemporaryDirectory() as directory:
            path = _save_badvla_checkpoint(source, directory, "stage1")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["format"], BADVLA_CHECKPOINT_FORMAT)
            self.assertEqual(payload["stage"], "stage1")
            self.assertEqual(payload["attack_config"], {"trigger_size": 0.1, "loss_p": 0.5})
            self.assertEqual(payload["model_config"]["future_action_window_size"], 15)
            self.assertEqual(payload["model_config"]["action_dim"], 7)
            _load_state_checkpoint(target, path, expected_attack_stage="stage1")
            with self.assertRaisesRegex(ValueError, "Expected a BadVLA stage2"):
                _load_state_checkpoint(target, path, expected_attack_stage="stage2")

            wrong_trigger = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA(trigger_size=0.2))
            with self.assertRaisesRegex(ValueError, "attack_config"):
                _load_state_checkpoint(wrong_trigger, path, expected_attack_stage="stage1")

            wrong_architecture = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA())
            wrong_architecture.base_model.model.future_action_window_size = 7
            with self.assertRaisesRegex(ValueError, "model_config"):
                _load_state_checkpoint(wrong_architecture, path, expected_attack_stage="stage1")

    def test_rollout_metric_denominators_and_paper_asr(self):
        self.assertEqual(rollout_success_rate(8, 10), 0.8)
        with self.assertRaisesRegex(ValueError, "zero attempted"):
            rollout_success_rate(0, 0)
        self.assertAlmostEqual(compute_badvla_asr(0.8, 0.8, 0.8, 0.0), 100.0)
        with self.assertRaisesRegex(ValueError, "undefined"):
            compute_badvla_asr(0.8, 0.0, 0.8, 0.0)

    def test_paired_defense_evaluation_shares_every_rollout_invariant(self):
        baseline = MockBaseMemoryVLA()
        attack = BadVLA()
        attacked = SecureVLA(MockBaseMemoryVLA(), attack=attack, defense=AMemGuard())
        episodes = tuple(
            RolloutEpisode("libero_spatial/task_3", index, 1000 + index)
            for index in range(4)
        )
        calls, criterion_outcomes = [], []

        def rollout_episode(*, model, episode, trigger):
            calls.append((model, episode, trigger, getattr(model, "defense_enabled", False)))
            if model is baseline:
                done = True
            elif not model.defense_enabled and trigger is None:
                done = episode.episode_index < 3
            elif not model.defense_enabled:
                done = episode.episode_index == 0
            elif trigger is None:
                done = episode.episode_index < 2
            else:
                done = episode.episode_index < 3
            return {"done": done}

        def one_success_definition(outcome):
            criterion_outcomes.append(outcome)
            return badvla_libero_success(outcome)

        result = evaluate_badvla_defense_rollouts(
            baseline_model=baseline,
            attacked_model=attacked,
            attacked_checkpoint="checkpoints/badvla-stage2.pt",
            trigger=attack,
            episodes=episodes,
            rollout_episode=rollout_episode,
            success_criterion=one_success_definition,
        )

        self.assertEqual(result["denominator"], 4)
        self.assertEqual(result["episode_keys"], tuple(e.key for e in episodes))
        self.assertIs(result["success_definition"], one_success_definition)
        self.assertEqual(len(criterion_outcomes), 6 * len(episodes))
        for offset in range(0, len(calls), len(episodes)):
            self.assertEqual(
                [call[1] for call in calls[offset:offset + len(episodes)]], list(episodes)
            )
        attacked_calls = [call for call in calls if call[0] is attacked]
        self.assertEqual({id(call[0]) for call in attacked_calls}, {id(attacked)})
        self.assertTrue(all(call[2] is attack for call in attacked_calls if call[2] is not None))
        self.assertEqual(result["attacked_checkpoint"], "checkpoints/badvla-stage2.pt")
        self.assertAlmostEqual(result["badvla"]["clean_sr"], 0.75)
        self.assertAlmostEqual(result["badvla"]["asr"], 56.25)
        self.assertAlmostEqual(result["badvla_amemguard"]["clean_sr"], 0.5)
        self.assertAlmostEqual(result["badvla_amemguard"]["asr"], 12.5)
        self.assertAlmostEqual(result["asr_reduction"], 43.75)
        self.assertAlmostEqual(result["clean_sr_drop"], 25.0)
        self.assertTrue(attacked.defense_enabled)
        self.assertIn("| BadVLA | 75.00% | 56.25% |", format_badvla_defense_report(result))

    def test_paired_rollout_does_not_silently_drop_failed_episode(self):
        baseline = MockBaseMemoryVLA()
        attack = BadVLA()
        attacked = SecureVLA(MockBaseMemoryVLA(), attack=attack, defense=AMemGuard())
        episodes = (RolloutEpisode("task", 0, 1), RolloutEpisode("task", 1, 2))

        def rollout_episode(*, model, episode, trigger):
            if episode.episode_index == 1:
                raise RuntimeError("environment failed")
            return {"done": True}

        with self.assertRaisesRegex(RuntimeError, "environment failed"):
            evaluate_badvla_defense_rollouts(
                baseline_model=baseline,
                attacked_model=attacked,
                attacked_checkpoint="stage2.pt",
                trigger=attack,
                episodes=episodes,
                rollout_episode=rollout_episode,
            )


if __name__ == "__main__":
    unittest.main()
