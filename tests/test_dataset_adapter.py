import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import IterableDataset

from utils.dataset import (
    LocalTrajectoryDataset,
    MemoryVLASampleTransform,
    adapt_libero_episode,
    collate_local_samples,
    get_dataloader,
)


def dual_image_transform(image):
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255)
    return {"dino": tensor, "siglip": tensor.mul(2).sub(1)}


class EmptyIterableDataset(IterableDataset):
    def __iter__(self):
        return iter(())


class LifecycleModel:
    def __init__(self):
        self.vlm = object()
        self.action_model = object()
        self.dataloader_type = "group"
        self.group_size = 4
        self.cog_mem_bank = Namespace(dataloader_type="group")
        self.per_mem_bank = Namespace(dataloader_type="group")


class LiberoTrajectoryAdapterTest(unittest.TestCase):
    def setUp(self):
        self.episode = {
            "episode_metadata": {"file_path": "demo_000.tfrecord"},
            "steps": [
                {
                    "observation": {
                        "image": np.zeros((8, 8, 3), dtype=np.uint8),
                        "wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
                        "state": np.zeros(8, dtype=np.float32),
                        "joint_state": np.zeros(7, dtype=np.float32),
                    },
                    "action": np.asarray([0.1, 0, 0, 0, 0, 0, -1], dtype=np.float32),
                    "language_instruction": "Pick up the block",
                    "is_first": True,
                    "is_last": False,
                    "is_terminal": False,
                    "reward": 0.0,
                    "discount": 1.0,
                },
                {
                    "observation": {
                        "image": np.full((8, 8, 3), 255, dtype=np.uint8),
                        "wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
                        "state": np.ones(8, dtype=np.float32),
                        "joint_state": np.ones(7, dtype=np.float32),
                    },
                    "action": np.asarray([0.2, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                    "language_instruction": b"Pick up the block",
                    "is_first": False,
                    "is_last": True,
                    "is_terminal": True,
                    "reward": 1.0,
                    "discount": 1.0,
                },
            ],
        }

    def test_raw_episode_becomes_chunked_memoryvla_transitions(self):
        transitions = adapt_libero_episode(
            self.episode,
            episode_id=9,
            future_action_window_size=2,
            action_statistics={"q01": [-1] * 7, "q99": [1] * 7, "mask": [True] * 6 + [False]},
            dataset_name="libero_spatial_no_noops",
        )

        self.assertEqual(len(transitions), 2)
        first, second = transitions
        self.assertEqual(first["observation"]["image_primary"].shape, (1, 8, 8, 3))
        self.assertEqual(first["observation"]["image_primary"].dtype, np.uint8)
        self.assertEqual(first["task"]["language_instruction"], b"pick up the block")
        self.assertEqual(first["action"].shape, (3, 7))
        np.testing.assert_array_equal(first["action_mask"], [True, True, True])
        np.testing.assert_array_equal(second["action_mask"], [True, True, True])
        self.assertEqual(first["action"][0, -1], 1.0)  # raw -1 open -> +1
        self.assertEqual(first["action"][1, -1], 0.0)  # raw +1 closed -> 0
        self.assertEqual(first["action"][2, -1], 0.0)  # padded absolute gripper repeats
        np.testing.assert_array_equal(first["episode_ids"], [9])
        np.testing.assert_array_equal(second["observation"]["timestep"], [1])
        self.assertEqual(first["episode_metadata"]["file_path"], "demo_000.tfrecord")

    def test_dual_processor_and_collated_shapes(self):
        dataset = LocalTrajectoryDataset(
            [self.episode],
            MemoryVLASampleTransform(dual_image_transform),
            future_action_window_size=2,
        )
        first, second = dataset[0], dataset[1]

        self.assertEqual(set(first["pixel_values"]), {"dino", "siglip"})
        self.assertEqual(first["pixel_values"]["dino"].shape, (3, 8, 8))
        self.assertEqual(first["actions"].shape, (3, 7))
        self.assertEqual(first["actions"].dtype, torch.float32)
        self.assertEqual(first["action_masks"].dtype, torch.bool)

        batch = collate_local_samples([first, second])
        self.assertEqual(batch["pixel_values"]["dino"].shape, (2, 3, 8, 8))
        self.assertEqual(batch["pixel_values"]["siglip"].shape, (2, 3, 8, 8))
        self.assertEqual(batch["actions"].shape, (2, 3, 7))
        self.assertEqual(batch["action_masks"].shape, (2, 3))
        np.testing.assert_array_equal(batch["episode_ids"], [0, 0])
        np.testing.assert_array_equal(batch["timesteps"], [0, 1])

        default_transition = adapt_libero_episode(self.episode, episode_id=0)[0]
        self.assertEqual(default_transition["action"].shape, (16, 7))
        self.assertEqual(default_transition["action_mask"].shape, (16,))

    def test_inference_adapter_leaves_preprocessing_to_predict_action(self):
        calls = []

        def tracked_transform(image):
            calls.append(image)
            return torch.zeros(3, 8, 8)

        dataset = LocalTrajectoryDataset(
            [self.episode],
            MemoryVLASampleTransform(tracked_transform, preprocess_image=False),
            future_action_window_size=2,
        )
        sample = dataset[0]
        self.assertIsNone(sample["pixel_values"])
        self.assertEqual(calls, [])
        self.assertEqual(sample["image"].size, (8, 8))

    def test_rejects_flat_or_wrong_action_contract(self):
        invalid = {"steps": [{"observation": {"image": np.zeros((8, 8, 3), np.uint8)}, "action": [0] * 6, "language_instruction": "x"}]}
        with self.assertRaisesRegex(ValueError, "shape \\(7,\\)"):
            adapt_libero_episode(invalid, episode_id=0)

    def test_iterable_loader_updates_memory_mode_and_preserves_group_boundaries(self):
        model = LifecycleModel()
        stream_args = Namespace(dataloader_type="stream", batch_size=2, group_size=None)
        with patch("utils.dataset.get_dataset_and_collator", return_value=(EmptyIterableDataset(), list)):
            get_dataloader(stream_args, model)
        self.assertEqual(model.cog_mem_bank.dataloader_type, "stream")
        self.assertEqual(model.per_mem_bank.dataloader_type, "stream")

        group_args = Namespace(dataloader_type="group", batch_size=2, group_size=None)
        with patch("utils.dataset.get_dataset_and_collator", return_value=(EmptyIterableDataset(), list)):
            with self.assertRaisesRegex(ValueError, "must be a multiple"):
                get_dataloader(group_args, model)

        local_dataset = LocalTrajectoryDataset(
            [self.episode], MemoryVLASampleTransform(dual_image_transform),
            future_action_window_size=2,
        )
        with patch("utils.dataset.get_dataset_and_collator", return_value=(local_dataset, list)):
            with self.assertRaisesRegex(ValueError, "fixes each batch to group_size"):
                get_dataloader(group_args, model)

        auto_args = Namespace(dataloader_type="auto", batch_size=None, group_size=None)
        with patch("utils.dataset.get_dataset_and_collator", return_value=(EmptyIterableDataset(), list)):
            loader = get_dataloader(auto_args, model)
        self.assertEqual(loader.batch_size, model.group_size)
        self.assertEqual(model.cog_mem_bank.dataloader_type, "group")

        override_args = Namespace(dataloader_type="group", batch_size=6, group_size=3)
        with patch("utils.dataset.get_dataset_and_collator", return_value=(EmptyIterableDataset(), list)):
            get_dataloader(override_args, model)
        self.assertEqual(model.group_size, 3)
        self.assertEqual(model.cog_mem_bank.group_size, 3)
        self.assertEqual(model.per_mem_bank.group_size, 3)


if __name__ == "__main__":
    unittest.main()
