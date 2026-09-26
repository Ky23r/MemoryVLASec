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

from main import _build_model, _load_state_checkpoint, _resolve_device, _validate_dataset_source, main
from defenses.amemguard import AMemGuard
from models.secure_vla import SecureVLA
from models.core.vla.load import _checkpoint_model_kwargs, _select_local_checkpoint
from utils.args import parse_arguments
from utils.dataset import _resolve_rlds_root
from utils.evaluate import run_evaluate
from utils.libero_evaluate import _memoryvla_eval_center_crop, _memoryvla_libero_image
from utils.mock_components import MockBaseMemoryVLA
from utils.train import MEMORYVLA_CHECKPOINT_FORMAT, _save_memoryvla_checkpoint


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


class StochasticTinyMemoryVLA(TinyMemoryVLA):
    def __init__(self):
        super().__init__()
        self.attack = SimpleNamespace(apply_trigger=lambda image: image)

    def predict_action(self, image, instruction, **kwargs):
        del image, instruction
        self.calls.append(kwargs)
        normalized = torch.randn(2, 7).numpy()
        return normalized.copy(), normalized


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
    def test_device_selection_accepts_cpu_cuda_and_numeric_gpu_index(self):
        self.assertEqual(str(_resolve_device("cpu")), "cpu")
        with (
            mock.patch("main.torch.cuda.is_available", return_value=True),
            mock.patch("main.torch.cuda.device_count", return_value=2),
        ):
            self.assertEqual(str(_resolve_device("cuda")), "cuda")
            self.assertEqual(str(_resolve_device("1")), "cuda:1")
            self.assertEqual(str(_resolve_device("cuda:1")), "cuda:1")
            with self.assertRaisesRegex(ValueError, "index 2 is unavailable"):
                _resolve_device("2")

        with self.assertRaisesRegex(ValueError, "Invalid device"):
            _resolve_device("gpu-one")

    def test_baseline_checkpoint_loads_through_defense_wrapper(self):
        baseline = MockBaseMemoryVLA()
        defended = SecureVLA(MockBaseMemoryVLA(), defense=AMemGuard())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.pt"
            torch.save(baseline.state_dict(), path)
            with self.assertWarnsRegex(UserWarning, "legacy raw baseline"):
                _load_state_checkpoint(defended, path)
        for expected, actual in zip(baseline.parameters(), defended.base_model.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_tagged_baseline_checkpoint_checks_architecture_and_loads_through_defense(self):
        baseline = MockBaseMemoryVLA()
        defended = SecureVLA(MockBaseMemoryVLA(), defense=AMemGuard())
        with tempfile.TemporaryDirectory() as directory:
            path = _save_memoryvla_checkpoint(baseline, directory)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["format"], MEMORYVLA_CHECKPOINT_FORMAT)
            _load_state_checkpoint(defended, path)

            payload["model_config"]["future_action_window_size"] = 99
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "model_config"):
                _load_state_checkpoint(defended, path)

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

    def test_real_rollout_refuses_mock_mode(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined with --mock"):
            main(["--mode", "evaluate", "--evaluation_type", "libero", "--mock", "--device", "cpu"])

    def test_verify_mode_rejects_ignored_security_options(self):
        with self.assertRaisesRegex(ValueError, "baseline infrastructure check"):
            main(["--mode", "verify", "--attack", "badvla"])

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

    def test_public_dataset_download_is_explicitly_anonymous(self):
        args = SimpleNamespace(
            dataset_path="", dataset_id="shihao1895/libero-rlds",
            dataset_revision="commit123", hf_token=None, cache_dir="cache",
        )
        with mock.patch("huggingface_hub.snapshot_download", return_value="snapshot") as download:
            _resolve_rlds_root(args, "libero_spatial_no_noops")
        self.assertIs(download.call_args.kwargs["token"], False)

    def test_memoryvla_uses_local_llama_architecture_and_public_tokenizer(self):
        from prismatic.models.backbones.llm.llama2 import LLAMA2_MODELS
        from vla.load import OFFICIAL_MEMORYVLA_CHECKPOINT, OFFICIAL_MEMORYVLA_ID

        config = LLAMA2_MODELS["llama2-7b-pure"]
        self.assertEqual(OFFICIAL_MEMORYVLA_ID, "shihao1895/memvla-libero-spatial")
        self.assertEqual(OFFICIAL_MEMORYVLA_CHECKPOINT, "checkpoints/memvla-libero-spatial.pt")
        self.assertEqual(config["inference_config"]["hidden_size"], 4096)
        self.assertEqual(config["inference_config"]["num_hidden_layers"], 32)
        self.assertEqual(config["tokenizer_hub_path"], "hf-internal-testing/llama-tokenizer")
        self.assertNotIn("meta-llama", config["tokenizer_hub_path"])

    def test_project_checkpoint_loading_is_strict(self):
        source, target = nn.Linear(2, 2), nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/weights.pt"
            torch.save(source.state_dict(), path)
            with self.assertWarnsRegex(UserWarning, "legacy raw baseline"):
                _load_state_checkpoint(target, path)
            torch.testing.assert_close(target.weight, source.weight)
            torch.save({"wrong.weight": torch.zeros(2, 2)}, path)
            with self.assertWarnsRegex(UserWarning, "legacy raw baseline"):
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

    def test_clean_and_triggered_offline_conditions_reuse_diffusion_rng(self):
        model, dataset = StochasticTinyMemoryVLA(), TinyDataset()
        args = eval_args()
        args.attack = "badvla"
        with mock.patch("utils.evaluate.get_dataset_and_collator", return_value=(dataset, None)):
            result = run_evaluate(model, args)
        self.assertEqual(
            result["clean"]["mean_normalized_action_mse"],
            result["triggered"]["mean_normalized_action_mse"],
        )

    def test_offline_evaluator_does_not_hide_inference_failure(self):
        model, dataset = TinyMemoryVLA(fail=True), TinyDataset()
        with mock.patch("utils.evaluate.get_dataset_and_collator", return_value=(dataset, None)):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                run_evaluate(model, eval_args())

    def test_libero_preprocessing_applies_official_image_aug_center_crop(self):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frame[..., 1] = 255
        processed = _memoryvla_libero_image({"agentview_image": frame})
        self.assertEqual(processed.size, (224, 224))

        bordered = np.zeros((224, 224, 3), dtype=np.uint8)
        bordered[..., 0] = 255
        bordered[6:-6, 6:-6] = (0, 255, 0)
        cropped = np.asarray(_memoryvla_eval_center_crop(Image.fromarray(bordered)))
        self.assertGreater(int(cropped[0, 0, 1]), int(cropped[0, 0, 0]))

        with self.assertRaises(ValueError):
            _memoryvla_eval_center_crop(Image.fromarray(bordered), area_fraction=0.0)


if __name__ == "__main__":
    unittest.main()
