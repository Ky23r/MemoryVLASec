import unittest

import numpy as np
import torch
from PIL import Image

from attacks.dropvla import DropVLA, DropVLAConfig
from utils.args import parse_arguments


class DropVLAIntegrationTest(unittest.TestCase):
    def test_parser_exposes_dropvla_configuration(self):
        args = parse_arguments(["--mode", "train", "--attack", "dropvla", "--mock"])
        self.assertEqual(args.attack, "dropvla")
        self.assertEqual(args.dropvla_protocol, "paper_faithful")
        self.assertEqual(args.dropvla_relabel_length, 8)

    def test_trigger_and_paper_window_relabel(self):
        attack = DropVLA(
            DropVLAConfig(episode_poison_rate=1.0, relabel_length=8, target_gripper_value=1.0)
        )
        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        marked = np.asarray(attack.apply_trigger(image))
        self.assertGreater(int(marked[..., 0].sum()), 0)
        self.assertEqual(int(marked[..., 1].sum()), 0)

        def transform(value):
            return torch.from_numpy(np.asarray(value).copy()).permute(2, 0, 1).float()

        actions = torch.zeros(2, 10, 7)
        pixels, relabeled, poison = attack.poison_batch(
            [image, image], actions, [1, 2], transform, "cpu"
        )
        self.assertEqual(tuple(pixels.shape), (2, 3, 32, 32))
        self.assertTrue(bool(poison.all()))
        self.assertTrue(torch.all(relabeled[:, :8, 6] == 1.0))
        self.assertTrue(torch.all(relabeled[:, 8:, 6] == 0.0))


if __name__ == "__main__":
    unittest.main()
