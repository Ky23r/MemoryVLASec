from types import SimpleNamespace
import unittest

import numpy as np

from utils.dataset import EpisodeGroupBatchSampler, FlatManifestDataset
from utils.evaluate import EpisodeCursor


class SyntheticMemoryVLA:
    """Inference-state mock implementing upstream reset/counter semantics."""

    def __init__(self):
        self.memory = []
        self.cur_timestep = 0
        self.trace = []

    def observe(self, episode_id, first_frame):
        if first_frame:
            self.memory.clear()
            self.cur_timestep = 0
        self.trace.append((episode_id, self.cur_timestep, tuple(self.memory)))
        self.memory.append((episode_id, self.cur_timestep))
        self.cur_timestep += 1


class MemoryLifecycleTest(unittest.TestCase):
    def test_episode_a_persists_and_episode_b_resets(self):
        cursor = EpisodeCursor()
        model = SyntheticMemoryVLA()
        sequence = [("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1)]

        # Production episode IDs are integers; labels make the assertion legible.
        ids = {"A": 10, "B": 11}
        for label, timestep in sequence:
            first = cursor.observe(ids[label], timestep)
            model.observe(label, first)

        self.assertEqual([entry[1] for entry in model.trace], [0, 1, 2, 0, 1])
        self.assertEqual(model.trace[0][2], ())
        self.assertEqual(model.trace[1][2], (("A", 0),))
        self.assertEqual(model.trace[2][2], (("A", 0), ("A", 1)))
        self.assertEqual(model.trace[3][2], ())
        self.assertEqual(model.trace[4][2], (("B", 0),))

    def test_cursor_rejects_fake_boundaries_and_bad_timesteps(self):
        cursor = EpisodeCursor()
        cursor.observe(1, 0)
        with self.assertRaisesRegex(ValueError, "expected timestep 1"):
            cursor.observe(1, 0)

        cursor = EpisodeCursor()
        with self.assertRaisesRegex(ValueError, "must start at timestep 0"):
            cursor.observe(2, 4)

    def test_group_sampler_never_mixes_episodes(self):
        dataset = SimpleNamespace(
            transitions=[
                {"episode_ids": np.asarray([4])},
                {"episode_ids": np.asarray([4])},
                {"episode_ids": np.asarray([4])},
                {"episode_ids": np.asarray([9])},
                {"episode_ids": np.asarray([9])},
            ]
        )
        batches = list(EpisodeGroupBatchSampler(dataset, 3, random_sample=False))
        self.assertEqual(batches, [[0, 1, 2], [3, 4, 4]])

    def test_flat_manifest_requires_real_lifecycle_metadata(self):
        with self.assertRaisesRegex(ValueError, "episode_id.*timestep"):
            FlatManifestDataset(
                [{"instruction": "x", "action": [[0] * 7], "image": np.zeros((2, 2, 3), np.uint8)}],
                ".",
                lambda value: value,
                action_horizon=1,
            )


if __name__ == "__main__":
    unittest.main()
