from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn as nn

from utils.dataset import MemoryVLASampleTransform, adapt_libero_episode
from utils.train import _forward_memory_vla, _images_to, _reset_memory_vla, run_train


class ExactMemoryVLAMock(nn.Module):
    """Minimal mock with the verified upstream MemoryVLA.forward signature."""

    def __init__(self):
        super().__init__()
        self.vlm = nn.Identity()
        self.action_model = SimpleNamespace(in_channels=7)
        self.future_action_window_size = 15
        self.received = None
        self.cog_mem_bank = SimpleNamespace(reset=lambda: setattr(self, "cog_reset", True))
        self.per_mem_bank = SimpleNamespace(reset=lambda: setattr(self, "per_reset", True))
        self.cur_timestep = 99
        self.cog_reset = False
        self.per_reset = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        actions=None,
        action_masks=None,
        timesteps=None,
        episode_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        repeated_diffusion_steps=4,
    ):
        self.received = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
            "actions": actions,
            "action_masks": action_masks,
            "timesteps": timesteps,
            "episode_ids": episode_ids,
            "output_hidden_states": output_hidden_states,
            "repeated_diffusion_steps": repeated_diffusion_steps,
        }
        return torch.tensor(1.0, requires_grad=True), SimpleNamespace(hidden_states=(torch.zeros(1),))


class FakePromptBuilder:
    def __init__(self, _family):
        self.turns = []

    def add_turn(self, role, value):
        self.turns.append((role, value))

    def get_prompt(self):
        return "prompt"


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, _prompt, add_special_tokens=True):
        return SimpleNamespace(input_ids=[1, 42, 2, 7])


def make_batch(batch_size=2):
    return {
        "pixel_values": torch.zeros(batch_size, 3, 8, 8),
        "input_ids": torch.ones(batch_size, 4, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "labels": torch.tensor([[-100, -100, 2, 7]] * batch_size, dtype=torch.long),
        "actions": torch.zeros(batch_size, 16, 7),
        "action_masks": torch.ones(batch_size, 16, dtype=torch.bool),
        "episode_ids": np.asarray([3, 3], dtype=np.int64)[:batch_size],
        "timesteps": np.asarray([4, 9], dtype=np.int64)[:batch_size],
    }


class MemoryVLATrainingAPITest(unittest.TestCase):
    def test_real_repeating_rlds_training_requires_a_step_cap(self):
        args = SimpleNamespace(max_steps=None, dataset_format="rlds", mock=False)
        with self.assertRaisesRegex(ValueError, "repeats indefinitely"):
            run_train(ExactMemoryVLAMock(), args)

    def test_image_batches_are_cast_to_model_dtype_without_casting_integer_tensors(self):
        images = {
            "image": torch.zeros(2, 3, 8, 8, dtype=torch.float32),
            "mask": torch.ones(2, 8, 8, dtype=torch.int64),
        }
        moved = _images_to(images, "cpu", torch.bfloat16)
        self.assertEqual(moved["image"].dtype, torch.bfloat16)
        self.assertEqual(moved["mask"].dtype, torch.int64)

    def test_finite_training_pass_resets_prior_memory(self):
        model = ExactMemoryVLAMock()
        _reset_memory_vla(model)
        self.assertTrue(model.cog_reset)
        self.assertTrue(model.per_reset)
        self.assertEqual(model.cur_timestep, 0)

    def test_forward_receives_complete_upstream_contract(self):
        model = ExactMemoryVLAMock()
        batch = make_batch()
        loss, output = _forward_memory_vla(
            model, batch, batch["pixel_values"], batch["actions"], "cpu"
        )

        self.assertEqual(loss.item(), 1.0)
        self.assertIsNotNone(output.hidden_states)
        self.assertIs(model.received["labels"], batch["labels"])
        self.assertTrue(model.received["output_hidden_states"])
        self.assertEqual(model.received["actions"].shape, (2, 16, 7))
        np.testing.assert_array_equal(model.received["episode_ids"], [3, 3])
        np.testing.assert_array_equal(model.received["timesteps"], [4, 9])

    def test_missing_metadata_is_rejected_instead_of_fabricated(self):
        model = ExactMemoryVLAMock()
        batch = make_batch()
        del batch["timesteps"]
        with self.assertRaisesRegex(KeyError, "timesteps"):
            _forward_memory_vla(model, batch, batch["pixel_values"], batch["actions"], "cpu")

    def test_missing_labels_are_rejected(self):
        model = ExactMemoryVLAMock()
        batch = make_batch()
        del batch["labels"]
        with self.assertRaisesRegex(KeyError, "labels"):
            _forward_memory_vla(model, batch, batch["pixel_values"], batch["actions"], "cpu")

    def test_wrong_action_chunk_is_rejected(self):
        model = ExactMemoryVLAMock()
        batch = make_batch()
        wrong_actions = torch.zeros(2, 7)
        with self.assertRaisesRegex(ValueError, r"\[2, 16, 7\]"):
            _forward_memory_vla(model, batch, batch["pixel_values"], wrong_actions, "cpu")

    def test_labels_match_upstream_prompt_masking(self):
        episode = {
            "steps": [{
                "observation": {"image": np.zeros((8, 8, 3), dtype=np.uint8)},
                "language_instruction": "Move left",
                "action": np.zeros(7, dtype=np.float32),
            }]
        }
        transition = adapt_libero_episode(episode, episode_id=0)[0]
        transform = MemoryVLASampleTransform(
            lambda _image: torch.zeros(3, 8, 8), FakeTokenizer(), FakePromptBuilder
        )
        sample = transform(transition)

        torch.testing.assert_close(sample["input_ids"], torch.tensor([1, 42, 2, 7]))
        torch.testing.assert_close(sample["labels"], torch.tensor([-100, -100, 2, 7]))


if __name__ == "__main__":
    unittest.main()
