"""CPU integration coverage for the real MemoryVLASec adapters and lifecycle."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.badvla import BadVLA
from defenses.amemguard import AMEMGUARD_CHECKPOINT_FORMAT, AMemGuard
from main import _load_state_checkpoint
from models.core.vla.memory_vla import CogMemBank
from models.secure_vla import SecureVLA
from scripts.calibrate_amemguard import (
    CleanLatentRecorder,
    _atomic_json,
    calibrated_cosine_distance_eps,
)
from utils.dataset import _find_memory_vla
from utils.evaluate import RolloutEpisode, evaluate_badvla_defense_rollouts
from utils.train import (
    _forward_memory_vla,
    _reset_memory_vla,
    _run_badvla_stage1,
    _run_badvla_stage2,
    _save_badvla_checkpoint,
    _save_memoryvla_checkpoint,
)


FEATURE_DIM = 8
ACTION_DIM = 7
ACTION_HORIZON = 2


class TinyVisionBackbone(nn.Module):
    default_image_resolution = (8, 8)

    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(3, 6)
        self.token_offsets = nn.Parameter(torch.linspace(-0.2, 0.2, 4).reshape(1, 4, 1))

    def forward(self, pixel_values):
        if isinstance(pixel_values, dict):
            pixel_values = next(iter(pixel_values.values()))
        pooled = pixel_values.mean(dim=(-2, -1))
        return self.input_projection(pooled).unsqueeze(1) + self.token_offsets

    def get_image_transform(self):
        def transform(image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
            return torch.from_numpy(array).permute(2, 0, 1).div(255.0)

        return transform


class TinyLanguageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(FEATURE_DIM, FEATURE_DIM)
        self.k_proj = nn.Linear(FEATURE_DIM, FEATURE_DIM)
        self.v_proj = nn.Linear(FEATURE_DIM, FEATURE_DIM)
        self.o_proj = nn.Linear(FEATURE_DIM, FEATURE_DIM)
        self.mlp = nn.Linear(FEATURE_DIM, FEATURE_DIM)

    def forward(self, state):
        state = torch.tanh(self.q_proj(state))
        state = torch.tanh(self.k_proj(state))
        state = torch.tanh(self.v_proj(state))
        return self.o_proj(state)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = TinyVisionBackbone()
        self.projector = nn.Linear(6, FEATURE_DIM)
        self.llm_backbone = TinyLanguageBackbone()
        self.vision_backbone_requires_grad = True


class TinyActionModel(nn.Module):
    in_channels = ACTION_DIM

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(FEATURE_DIM, ACTION_HORIZON * ACTION_DIM)

    def forward(self, state):
        return self.head(state).reshape(-1, ACTION_HORIZON, ACTION_DIM)


class TinyMemoryVLA(nn.Module):
    """Small backbone with the training and memory API used by MemoryVLA."""

    def __init__(self):
        super().__init__()
        self.future_action_window_size = ACTION_HORIZON - 1
        self.dataloader_type = "stream"
        self.group_size = 2
        self.cog_token_size = FEATURE_DIM
        self.per_token_size = FEATURE_DIM
        self.mem_length = 8
        self.retrieval_layers = 1
        self.use_timestep_pe = False
        self.fusion_type = "add"
        self.consolidate_type = "fifo"
        self.update_fused = False
        self.cur_timestep = 0
        self.vlm = TinyVLM()
        self.per_compr = nn.Linear(FEATURE_DIM, FEATURE_DIM)
        bank_options = dict(
            dataloader_type=self.dataloader_type,
            group_size=self.group_size,
            token_size=FEATURE_DIM,
            mem_length=self.mem_length,
            retrieval_layers=self.retrieval_layers,
            use_timestep_pe=self.use_timestep_pe,
            fusion_type=self.fusion_type,
            consolidate_type=self.consolidate_type,
            update_fused=self.update_fused,
        )
        self.cog_mem_bank = CogMemBank(**bank_options)
        self.per_mem_bank = CogMemBank(**bank_options)
        self.action_model = TinyActionModel()

    def _features(self, pixel_values, episode_ids, timesteps):
        tokens = self.vlm.projector(self.vlm.vision_backbone(pixel_values))
        cognition = self.cog_mem_bank.process_batch(
            tokens.mean(dim=1, keepdim=True), episode_ids, timesteps
        )
        perception = self.per_mem_bank.process_batch(
            self.per_compr(tokens), episode_ids, timesteps
        )
        state = 0.5 * (cognition.mean(dim=1) + perception.mean(dim=1))
        return self.vlm.llm_backbone(state), cognition, perception

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        pixel_values,
        labels,
        actions,
        action_masks,
        episode_ids,
        timesteps,
        output_hidden_states,
        **kwargs,
    ):
        del kwargs
        if not output_hidden_states:
            raise AssertionError("MemoryVLA training requires hidden states")
        if input_ids.shape != attention_mask.shape or input_ids.shape != labels.shape:
            raise AssertionError("token tensors must have identical [B,L] shapes")
        state, cognition, perception = self._features(pixel_values, episode_ids, timesteps)
        predictions = self.action_model(state)
        mask = action_masks.unsqueeze(-1).to(dtype=predictions.dtype)
        loss = ((predictions - actions.to(predictions.dtype)).square() * mask).sum()
        loss = loss / mask.sum().clamp_min(1)
        return loss, {
            "action_predictions": predictions,
            "cognition": cognition,
            "perception": perception,
        }

    def predict_action(self, image, instruction, episode_first_frame="False", **kwargs):
        del instruction, kwargs
        if episode_first_frame == "True":
            self.cog_mem_bank.reset()
            self.per_mem_bank.reset()
            self.cur_timestep = 0
        pixels = self.vlm.vision_backbone.get_image_transform()(image).unsqueeze(0)
        episode_ids = np.asarray([0], dtype=np.int64)
        timesteps = np.asarray([self.cur_timestep], dtype=np.int64)
        state, _, _ = self._features(pixels, episode_ids, timesteps)
        actions = self.action_model(state)[0].detach().cpu().numpy().astype(np.float32)
        self.cur_timestep += 1
        return actions.copy(), actions


class TinyBaseMemoryVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyMemoryVLA()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _images():
    return [
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB"),
        Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8), mode="RGB"),
    ]


def _training_batch(model):
    images = _images()
    transform = model.model.vlm.vision_backbone.get_image_transform()
    return {
        "images": images,
        "image": images,
        "pixel_values": torch.stack([transform(image) for image in images]),
        "input_ids": torch.tensor([[1, 2, 3], [1, 4, 3]], dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.tensor([[1, 2, 3], [1, 4, 3]], dtype=torch.long),
        "actions": torch.linspace(-0.5, 0.5, 2 * ACTION_HORIZON * ACTION_DIM).reshape(
            2, ACTION_HORIZON, ACTION_DIM
        ),
        "action_masks": torch.ones(2, ACTION_HORIZON, dtype=torch.bool),
        "episode_ids": torch.tensor([0, 0], dtype=torch.long),
        "timesteps": torch.tensor([0, 1], dtype=torch.long),
    }


def _assert_finite_cpu_module(testcase, module):
    for name, parameter in module.named_parameters():
        testcase.assertEqual(parameter.device.type, "cpu", name)
        testcase.assertTrue(parameter.dtype.is_floating_point, name)
        testcase.assertTrue(torch.isfinite(parameter).all(), name)
        if parameter.grad is not None:
            testcase.assertEqual(parameter.grad.device.type, "cpu", name)
            testcase.assertTrue(torch.isfinite(parameter.grad).all(), name)


class CPUIntegrationTest(unittest.TestCase):
    def test_complete_security_training_checkpoint_and_evaluation_path(self):
        torch.manual_seed(7)
        np.random.seed(7)

        with tempfile.TemporaryDirectory(prefix="memoryvlasec-cpu-") as directory:
            output_dir = Path(directory)

            # Baseline forward/loss/backward/step through the real training-call adapter.
            baseline = TinyBaseMemoryVLA()
            batch = _training_batch(baseline)
            optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-2)
            before = baseline.model.action_model.head.weight.detach().clone()
            loss, model_output = _forward_memory_vla(
                baseline, batch, batch["pixel_values"], batch["actions"], "cpu"
            )
            self.assertEqual(loss.device.type, "cpu")
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(
                tuple(model_output["action_predictions"].shape),
                (2, ACTION_HORIZON, ACTION_DIM),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in baseline.parameters()))
            optimizer.step()
            self.assertFalse(torch.equal(before, baseline.model.action_model.head.weight))
            _assert_finite_cpu_module(self, baseline)

            baseline_checkpoint = _save_memoryvla_checkpoint(baseline, output_dir / "baseline")
            baseline_reload = TinyBaseMemoryVLA()
            _load_state_checkpoint(baseline_reload, baseline_checkpoint)
            for expected, actual in zip(baseline.parameters(), baseline_reload.parameters()):
                torch.testing.assert_close(actual, expected)

            # Real memory-bank write, retrieval, and reset over multiple timesteps.
            baseline_reload.eval()
            image = _images()[0]
            for timestep in range(3):
                prediction = baseline_reload.model.predict_action(
                    image, "move", episode_first_frame="True" if timestep == 0 else "False"
                )
                self.assertEqual(prediction[0].shape, (ACTION_HORIZON, ACTION_DIM))
                self.assertTrue(np.isfinite(prediction[0]).all())
            self.assertEqual(len(baseline_reload.model.cog_mem_bank.bank[0]), 3)
            self.assertEqual(len(baseline_reload.model.per_mem_bank.bank[0]), 3)
            _reset_memory_vla(baseline_reload)
            self.assertFalse(baseline_reload.model.cog_mem_bank.bank)
            self.assertFalse(baseline_reload.model.per_mem_bank.bank)

            # BadVLA clean/poison preprocessing and the production Stage I optimizer.
            attack = BadVLA(trigger_size=0.25, loss_p=0.5)
            clean_array = np.asarray(image).copy()
            poisoned = attack.apply_trigger(clean_array)
            np.testing.assert_array_equal(clean_array, np.zeros_like(clean_array))
            self.assertGreater(np.count_nonzero(poisoned), 0)
            attacked = SecureVLA(TinyBaseMemoryVLA(), attack=attack)
            attack_batch = _training_batch(attacked.base_model)
            triggered_pixels = attack.preprocess_triggered(
                attack_batch["images"],
                attacked.base_model.model.vlm.vision_backbone.get_image_transform(),
                "cpu",
            )
            self.assertEqual(triggered_pixels.shape, attack_batch["pixel_values"].shape)
            self.assertEqual(triggered_pixels.dtype, torch.float32)
            projector_before = attacked.base_model.model.vlm.projector.weight.detach().clone()
            stage_args = SimpleNamespace(
                device="cpu", learning_rate=1e-2, epochs=1, max_steps=1,
                badvla_lr_decay_step=100,
            )
            _run_badvla_stage1(attacked, [attack_batch], stage_args)
            self.assertFalse(
                torch.equal(projector_before, attacked.base_model.model.vlm.projector.weight)
            )
            _assert_finite_cpu_module(self, attacked)

            stage1_checkpoint = _save_badvla_checkpoint(attacked, output_dir / "attack", "stage1")
            stage2_model = SecureVLA(
                TinyBaseMemoryVLA(), attack=BadVLA(trigger_size=0.25, loss_p=0.5)
            )
            _load_state_checkpoint(stage2_model, stage1_checkpoint, expected_attack_stage="stage1")
            downstream_before = stage2_model.base_model.model.action_model.head.weight.detach().clone()
            _run_badvla_stage2(stage2_model, [_training_batch(stage2_model.base_model)], stage_args)
            self.assertFalse(
                torch.equal(downstream_before, stage2_model.base_model.model.action_model.head.weight)
            )
            self.assertTrue(all(
                parameter.grad is None
                for parameter in stage2_model.base_model.model.vlm.vision_backbone.parameters()
            ))
            stage2_checkpoint = _save_badvla_checkpoint(
                stage2_model, output_dir / "attack", "stage2"
            )

            # Production A-MemGuard calibration recorder and quantile rule.
            recorder_model = stage2_model.base_model.model
            recorder_model.eval()
            recorder = CleanLatentRecorder()
            recorder_model.cog_mem_bank.set_retrieval_filter(partial(
                recorder.filter_history, bank_name="cognition"
            ))
            recorder_model.per_mem_bank.set_retrieval_filter(partial(
                recorder.filter_history, bank_name="perception"
            ))
            for timestep in range(4):
                recorder_model.predict_action(
                    image, "move", episode_first_frame="True" if timestep == 0 else "False"
                )
            eps = calibrated_cosine_distance_eps(recorder.nearest_distances, 0.95)
            self.assertTrue(0.0 <= eps <= 2.0)
            provenance = {
                "model_id": "cpu-integration/tiny-memoryvla",
                "model_revision": "test",
                "attack_checkpoint_bytes": stage2_checkpoint.stat().st_size,
            }
            defense_checkpoint = output_dir / "amemguard.json"
            _atomic_json(defense_checkpoint, {
                "format": AMEMGUARD_CHECKPOINT_FORMAT,
                "adapter": "memoryvla_latent_dbscan",
                "config": {"cosine_distance_eps": eps, "min_cluster_size": 2},
                "provenance": provenance,
            })
            defense = AMemGuard.from_checkpoint(
                defense_checkpoint, expected_provenance=provenance
            )

            # Load one exact attacked checkpoint into the combined attack+defense path.
            combined = SecureVLA(
                TinyBaseMemoryVLA(),
                attack=BadVLA(trigger_size=0.25, loss_p=0.5),
                defense=defense,
            )
            _load_state_checkpoint(combined, stage2_checkpoint, expected_attack_stage="stage2")
            self.assertTrue(combined.defense_enabled)
            for timestep in range(3):
                current_image = image if timestep < 2 else combined.attack.apply_trigger(image)
                combined.base_model.model.predict_action(
                    current_image,
                    "move",
                    episode_first_frame="True" if timestep == 0 else "False",
                )
            self.assertIn("cognition", defense.last_decisions)
            self.assertIn("perception", defense.last_decisions)
            self.assertGreater(defense.metrics()["cognition"]["candidates"], 0)
            combined.set_defense_enabled(False)
            self.assertIsNone(combined.base_model.model.cog_mem_bank.retrieval_filter)
            combined.set_defense_enabled(True)
            self.assertTrue(callable(combined.base_model.model.cog_mem_bank.retrieval_filter))

            # Existing paired evaluator exercises clean/poison and defense OFF/ON.
            evaluation_baseline = baseline_reload
            episodes = (
                RolloutEpisode("cpu/tiny", 0, 100),
                RolloutEpisode("cpu/tiny", 1, 101),
            )

            def rollout_episode(*, model, episode, trigger):
                memory_vla = _find_memory_vla(model)
                rollout_image = Image.fromarray(
                    np.full((8, 8, 3), episode.episode_index * 32, dtype=np.uint8), mode="RGB"
                )
                if trigger is not None:
                    rollout_image = trigger.apply_trigger(rollout_image)
                for step in range(2):
                    actions, normalized = memory_vla.predict_action(
                        rollout_image,
                        "move",
                        episode_first_frame="True" if step == 0 else "False",
                    )
                    self.assertEqual(actions.shape, (ACTION_HORIZON, ACTION_DIM))
                    self.assertEqual(actions.dtype, np.float32)
                    self.assertTrue(np.isfinite(actions).all())
                    np.testing.assert_array_equal(actions, normalized)
                if model is evaluation_baseline or trigger is None:
                    return {"done": True}
                if model.defense_enabled:
                    return {"done": True}
                return {"done": episode.episode_index == 0}

            evaluation = evaluate_badvla_defense_rollouts(
                baseline_model=evaluation_baseline,
                attacked_model=combined,
                attacked_checkpoint=stage2_checkpoint,
                trigger=combined.attack,
                episodes=episodes,
                rollout_episode=rollout_episode,
            )
            self.assertEqual(evaluation["denominator"], 2)
            self.assertEqual(evaluation["badvla"]["asr"], 50.0)
            self.assertEqual(evaluation["badvla_amemguard"]["asr"], 0.0)
            self.assertEqual(evaluation["asr_reduction"], 50.0)
            self.assertTrue(combined.defense_enabled)
            _assert_finite_cpu_module(self, combined)


if __name__ == "__main__":
    unittest.main()
