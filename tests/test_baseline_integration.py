from types import SimpleNamespace
from unittest import mock
import builtins
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from main import _build_model, _load_state_checkpoint, _validate_dataset_source, main
from models.core.vla.load import _checkpoint_model_kwargs, _select_local_checkpoint
from utils.args import parse_arguments
from utils.dataset import _resolve_rlds_root
from utils.evaluate import run_evaluate


STATS = {
    "libero_spatial_no_noops": {
        "action": {"q01": [-1.0] * 7, "q99": [1.0] * 7, "mask": [True] * 6 + [False]}
    }
}


class TinyActionModel:
    in_channels = 7


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


class TinyMemoryVLA(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.vlm = TinyVLM()
        self.action_model = TinyActionModel()
        self.future_action_window_size = 1
        self.norm_stats = STATS
        self.calls = []
        self.fail = fail

    def predict_action(self, image, instruction, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("inference failed")
        shape = (2, 7)
        return np.ones(shape, np.float32), np.zeros(shape, np.float32)


class TinyDataset:
    dataset_statistics = STATS

    def __init__(self):
        self.samples = []
        for episode_id, length in ((4, 3), (5, 2)):
            for timestep in range(length):
                self.samples.append({
                    "image": Image.fromarray(np.zeros((4, 4, 3), np.uint8)),
                    "instruction": "move left",
                    "actions": torch.zeros(2, 7),
                    "episode_ids": np.asarray([episode_id]),
                    "timesteps": np.asarray([timestep]),
                })

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


def eval_args():
    return SimpleNamespace(
        attack="none", device="cpu", unnorm_key="libero_spatial_no_noops",
        cfg_scale=1.5, use_ddim=False, num_ddim_steps=10,
    )


class BaselineIntegrationTest(unittest.TestCase):
    def test_checkpoint_architecture_config_is_authoritative(self):
        config = {"action_dim": 7, "future_action_window_size": 15, "group_size": 16}
        self.assertEqual(_checkpoint_model_kwargs(config, {})["group_size"], 16)
        with self.assertRaisesRegex(ValueError, "conflicts with checkpoint config"):
            _checkpoint_model_kwargs(config, {"future_action_window_size": 7})

    def test_local_checkpoint_selection_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            only = checkpoint_dir / "step-1.pt"
            only.touch()
            self.assertEqual(_select_local_checkpoint(Path(directory)), only)
            latest = checkpoint_dir / "latest-checkpoint.pt"
            latest.touch()
            self.assertEqual(_select_local_checkpoint(Path(directory)), latest)
            latest.unlink()
            (checkpoint_dir / "step-2.pt").touch()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _select_local_checkpoint(Path(directory))

    def test_baseline_parser_has_no_lora_or_quantization_options(self):
        args = parse_arguments(["--mode", "evaluate", "--dataset_id", "repo/data"])
        self.assertEqual(args.attack, "none")
        self.assertEqual(args.defense, "none")
        self.assertFalse(hasattr(args, "use_lora"))
        self.assertFalse(hasattr(args, "quantization"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(["--mode", "evaluate", "--dataset_id", "repo/data", "--use_lora"])

    def test_evaluation_parser_rejects_training_only_options(self):
        for option, value in (("--batch_size", "2"), ("--epochs", "1"), ("--learning_rate", "1e-5")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(["--mode", "evaluate", option, value])

    def test_baseline_build_does_not_import_peft_or_bitsandbytes(self):
        args = parse_arguments(["--mode", "evaluate", "--mock", "--device", "cpu"])
        original_import = builtins.__import__

        def guarded_import(name, *values, **kwargs):
            if name.split(".", 1)[0] in {"peft", "bitsandbytes", "attacks", "defenses"}:
                raise AssertionError(f"baseline imported forbidden dependency {name}")
            return original_import(name, *values, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            model = _build_model(args, torch.device("cpu"))
        self.assertEqual(type(model).__name__, "MockBaseMemoryVLA")

    def test_dataset_sources_are_mutually_exclusive(self):
        args = SimpleNamespace(mock=False, dataset_id="repo/data", dataset_path="also-local", dataset_format="rlds")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _validate_dataset_source(args)

    def test_unavailable_rollout_mode_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "rollout evaluation is not implemented"):
            main(["--mode", "evaluate", "--evaluation_type", "libero", "--mock", "--device", "cpu"])

    def test_local_manifest_format_requires_local_path(self):
        args = SimpleNamespace(mock=False, dataset_id="repo/data", dataset_path="", dataset_format="trajectory")
        with self.assertRaisesRegex(ValueError, "requires --dataset_path"):
            _validate_dataset_source(args)

    def test_local_source_does_not_require_remote_id(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(mock=False, dataset_id=None, dataset_path=directory, dataset_format="rlds")
            _validate_dataset_source(args)
            self.assertEqual(_resolve_rlds_root(args, "libero_spatial_no_noops"), Path(directory).resolve())

    def test_remote_dataset_revision_token_and_cache_are_consumed(self):
        args = SimpleNamespace(
            dataset_path="", dataset_id="owner/data", dataset_revision="commit123",
            hf_token="token", cache_dir="cache",
        )
        with mock.patch("huggingface_hub.snapshot_download", return_value="snapshot") as download:
            path = _resolve_rlds_root(args, "libero_spatial_no_noops")
        self.assertEqual(str(path), "snapshot")
        download.assert_called_once_with(
            repo_id="owner/data", repo_type="dataset", revision="commit123", token="token",
            cache_dir="cache", allow_patterns=["libero_spatial_no_noops/**"],
        )

    def test_project_checkpoint_loading_is_strict(self):
        source, target = nn.Linear(2, 2), nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/weights.pt"
            torch.save(source.state_dict(), path)
            _load_state_checkpoint(target, path)
            torch.testing.assert_close(target.weight, source.weight)
            torch.save({"wrong.weight": torch.zeros(2, 2)}, path)
            with self.assertRaises(RuntimeError):
                _load_state_checkpoint(target, path)

    def test_offline_evaluator_reports_mse_not_success(self):
        model, dataset = TinyMemoryVLA(), TinyDataset()
        with mock.patch("utils.evaluate.get_dataset_and_collator", return_value=(dataset, None)):
            result = run_evaluate(model, eval_args())
        self.assertEqual(result["transitions"], 5)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["mean_normalized_action_mse"], 0.0)
        self.assertEqual(
            [call["episode_first_frame"] for call in model.calls],
            ["True", "False", "False", "True", "False"],
        )
        self.assertNotIn("success_rate", result)

    def test_offline_evaluator_does_not_hide_inference_failure(self):
        model, dataset = TinyMemoryVLA(fail=True), TinyDataset()
        with mock.patch("utils.evaluate.get_dataset_and_collator", return_value=(dataset, None)):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                run_evaluate(model, eval_args())


if __name__ == "__main__":
    unittest.main()
