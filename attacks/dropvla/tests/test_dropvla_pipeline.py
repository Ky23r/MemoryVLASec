"""CPU checks for label consistency, actual PEFT gradients, reload and metrics."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
# Match the script environment so importing LIBERO cannot request interactive
# setup in an otherwise unattended test run with downloaded local assets.
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache", "memoryvlasec", "libero-config",
))

from dataclasses import asdict
import tempfile
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from attacks.dropvla import DropVLA, DropVLAConfig
from main import _load_state_checkpoint
from utils.dropvla_dataset import DropVLARLDSDataset, episode_fingerprint, poisoned_action_chunks, select_poison_steps
from utils.dataset import MemoryVLASampleTransform
from utils.dropvla_lora import DropVLAFinetuneConfig, assert_dropvla_gradients, configure_dropvla_lora
from utils.dropvla_metrics import DropVLARolloutTracker, aggregate_dropvla_metrics
from utils.train import _save_dropvla_checkpoint, _run_dropvla


class ResettableLinear(nn.Linear):
    def reset(self):
        pass


class TinyMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Module()
        self.vlm.llm_backbone = nn.Module()
        self.vlm.llm_backbone.llm = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        ))
        self.vlm.vision_backbone = nn.Linear(3, 16)
        self.vlm.projector = nn.Linear(16, 16)
        self.cog_mem_bank = ResettableLinear(16, 16)
        self.per_mem_bank = ResettableLinear(16, 16)
        self.per_compr = nn.Linear(16, 16)
        self.action_model = nn.Linear(16, 7)
        self.action_model.in_channels = 7
        self.future_action_window_size = 15
        self.dataloader_type, self.group_size = "stream", 1

    def forward(self, ids=None, **kwargs):
        if ids is None:
            ids = kwargs["input_ids"]
        output = self.vlm.llm_backbone.llm(ids, output_hidden_states=True, use_cache=False)
        features = self.cog_mem_bank(output.hidden_states[-1][:, -1])
        prediction = self.action_model(features)
        if "actions" in kwargs:
            return (prediction - kwargs["actions"][:, 0]).square().mean(), None
        return prediction


class TinyWrapper(nn.Module):
    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)


def wrapped_tiny():
    wrapper = TinyWrapper()
    wrapper.base_model = nn.Module()
    wrapper.base_model.model = TinyMemory()
    wrapper.attack = DropVLA(DropVLAConfig(protocol="released_repo", episode_poison_rate=.05))
    return wrapper


class DropVLAUnitTests(unittest.TestCase):
    def test_dataset_emits_all_frames_in_order_and_keeps_poison_mask(self):
        import tensorflow as tf
        from io import BytesIO
        from PIL import Image

        tf.config.set_visible_devices([], "GPU")
        buffer = BytesIO()
        Image.new("RGB", (32, 32)).save(buffer, format="JPEG")
        encoded = buffer.getvalue()
        actions = np.zeros((5, 7), dtype=np.float32)
        actions[:, 6] = [-1, -1, 1, 1, -1]
        metadata = {"file_path": b"synthetic_source"}
        steps = [{"action": a, "language_instruction": b"pick bowl",
                  "observation": {"image": encoded}} for a in actions]
        raw_steps = tf.data.Dataset.from_tensor_slices({
            "action": actions, "language_instruction": [b"pick bowl"] * 5,
            "observation": {"image": [encoded] * 5},
        })
        raw = tf.data.Dataset.from_tensors({"steps": raw_steps, "episode_metadata": metadata})
        for smoke in (False, True):
            dataset = DropVLARLDSDataset.__new__(DropVLARLDSDataset)
            dataset.attack = DropVLA(DropVLAConfig(protocol="released_repo", episode_poison_rate=1))
            dataset.transform = MemoryVLASampleTransform(lambda image: torch.from_numpy(np.array(image)).permute(2, 0, 1))
            dataset.future, dataset.resolution = 3, 32
            dataset.raw_dataset, dataset.smoke = raw, smoke
            dataset.pass_index, dataset.data_mix = 0, "test"
            dataset.dataset_statistics = {"test": {"action": {
                "q01": [-1] * 7, "q99": [1] * 7, "mask": [True] * 6 + [False],
            }}}
            dataset.plan = {"episode_count": 1, "transition_count": 5, "episodes": [{
                "source_episode_id": 0, "length": 5,
                "fingerprint": episode_fingerprint(steps, metadata), "poison_timesteps": [2, 3],
            }]}
            samples = list(dataset)
            self.assertEqual(len(samples), 5)
            self.assertEqual([int(s["timesteps"][0]) for s in samples], list(range(5)))
            self.assertEqual([s["dropvla_poisoned"] for s in samples], [False, False, True, True, False])
            self.assertEqual(samples[2]["image"].getpixel((10, 10)), (255, 0, 0))
            self.assertEqual(samples[0]["image"].getpixel((10, 10)), (0, 0, 0))
            torch.testing.assert_close(samples[0]["actions"][2], samples[2]["actions"][0])

    def test_production_dataset_factory_preserves_resolution_and_poison_metadata(self):
        from PIL import Image
        from utils.dataset import get_dataset_and_collator
        wrapper = wrapped_tiny()
        core = wrapper.base_model.model
        core.vlm.vision_backbone.get_image_transform = lambda: lambda image: torch.zeros(3, 224, 224)
        core.vlm.vision_backbone.default_image_resolution = (3, 224, 224)
        core.vlm.llm_backbone.tokenizer = SimpleNamespace(pad_token_id=0)
        core.vlm.llm_backbone.prompt_builder_fn = None
        args = SimpleNamespace(mock=False, attack="dropvla", dropvla_protocol="released_repo",
            dataset_format="rlds", dataset_config="libero_spatial_no_noops", image_aug=False,
            batch_size=1, dataloader_type="stream", dropvla_poison_plan="plan", dropvla_smoke=False)
        with patch("utils.dataset._resolve_rlds_root", return_value="cached-data"), \
             patch("utils.dropvla_dataset.DropVLARLDSDataset") as factory:
            _, collator = get_dataset_and_collator(args, wrapper)
            self.assertEqual(factory.call_args.kwargs["resolution"], 224)
            transform = factory.call_args.args[4]
            sample = transform({"observation": {"image_primary": np.zeros((1,224,224,3), dtype=np.uint8),
                "timestep": np.array([3])}, "task": {"language_instruction": b"pick bowl"},
                "action": np.zeros((16,7)), "action_mask": np.ones(16, dtype=bool),
                "dataset_name": b"libero_spatial_no_noops", "episode_ids": np.array([8])})
            sample["dropvla_poisoned"] = True
            batch = collator([sample])
            self.assertEqual(batch["pixel_values"].shape, (1,3,224,224))
            self.assertEqual(batch["actions"].shape, (1,16,7))
            self.assertEqual(batch["dropvla_poisoned"].tolist(), [True])
            self.assertEqual(batch["episode_ids"].tolist(), [8])
            self.assertEqual(batch["timesteps"].tolist(), [3])

    def test_raw_poison_is_consistent_across_overlapping_chunks(self):
        actions = np.arange(35, dtype=np.float32).reshape(5, 7) / 100
        actions[:, 6] = [-1, 1, 1, -1, -1]
        stats = {"q01": [-1] * 7, "q99": [1] * 7, "mask": [True] * 6 + [False]}
        chunks = list(poisoned_action_chunks(actions, [1], 3, stats))
        np.testing.assert_array_equal(chunks[0][0][1], chunks[1][0][0])
        self.assertEqual(chunks[0][0][1, 6], 1.)
        self.assertEqual(chunks[1][0][1, 6], 0.)  # unselected closed step remains closed
        np.testing.assert_allclose(chunks[1][0][0, :6], actions[1, :6], atol=1e-6)
        np.testing.assert_array_equal(actions[:, 6], [-1, 1, 1, -1, -1])
        with self.assertRaises(ValueError):
            list(poisoned_action_chunks(actions, [0], 3, stats))

    def test_released_sampling_only_selects_closed_steps(self):
        import random
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[[2, 4, 7], 6] = 1
        self.assertEqual(select_poison_steps(actions, selected=True, step_rate=1, rng=random.Random(42)), [2, 4, 7])
        self.assertEqual(select_poison_steps(actions, selected=False, step_rate=1, rng=random.Random(42)), [])
        self.assertEqual(len(select_poison_steps(actions, selected=True, step_rate=.1, rng=random.Random(42))), 1)

    def test_actual_lora_head_gradients_freezing_and_checkpoint_reload(self):
        torch.manual_seed(42)
        wrapper = wrapped_tiny()
        model = wrapper.base_model.model
        config = DropVLAFinetuneConfig(rank=2, alpha=2)
        configure_dropvla_lora(model, config, training=True)
        model.train()
        originals = {name: p.detach().clone() for name, p in model.named_parameters()}
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=.01, foreach=False)
        ids = torch.tensor([[1, 3, 5, 7]])
        for _ in range(2):
            loss = (model(ids) - 1).square().mean()
            loss.backward()
            assert_dropvla_gradients(model)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertTrue(any(not torch.equal(p, originals[name]) for name, p in model.named_parameters() if "lora_" in name))
        self.assertFalse(torch.equal(model.action_model.weight, originals["action_model.weight"]))
        for name, p in model.named_parameters():
            if not p.requires_grad:
                self.assertTrue(torch.equal(p, originals[name]), name)
        model.eval()
        expected = model(ids).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = _save_dropvla_checkpoint(wrapper, directory)
            loaded = wrapped_tiny()
            _load_state_checkpoint(loaded, path)
            loaded.eval()
            torch.testing.assert_close(loaded.base_model.model(ids), expected, rtol=0, atol=0)
            self.assertEqual(loaded.base_model.model.dropvla_finetune_config, asdict(config))
            loaded.attack = DropVLA(DropVLAConfig(protocol="released_repo", episode_poison_rate=.01))
            with self.assertRaises(ValueError):
                _load_state_checkpoint(loaded, path)
        # Exercise HF generation through the attached PEFT model as inference does.
        generated = model.vlm.llm_backbone.llm.generate(ids, attention_mask=torch.ones_like(ids),
                                                      pad_token_id=0, max_new_tokens=1, do_sample=False)
        self.assertEqual(generated.shape, (1, 5))

    def test_trainer_respects_optimizer_budget_accumulation_and_finite_passes(self):
        wrapper = wrapped_tiny()
        configure_dropvla_lora(wrapper.base_model.model, DropVLAFinetuneConfig(rank=2, alpha=2), training=True)
        batches = []
        for timestep in range(3):
            ids = torch.tensor([[1, 3, 5, 7]])
            batches.append({
                "pixel_values": torch.zeros(1, 3, 2, 2), "input_ids": ids,
                "attention_mask": torch.ones_like(ids, dtype=torch.bool), "labels": ids,
                "actions": torch.ones(1, 16, 7), "action_masks": torch.ones(1, 16, dtype=torch.bool),
                "episode_ids": np.array([0]), "timesteps": np.array([timestep]),
                "dropvla_poisoned": torch.tensor([timestep == 1]),
            })
        loader = SimpleNamespace(dataset=SimpleNamespace(plan={
            "selected_episode_count": 1, "episode_count": 1, "poisoned_frame_count": 1,
            "transition_count": 3, "episodes": [{"source_episode_id": 0, "poison_timesteps": [1]}],
        }))
        class FiniteLoader:
            dataset = loader.dataset
            def __iter__(self):
                return iter(batches)
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text("{}")
            args = SimpleNamespace(
                device="cpu", max_steps=5, gradient_accumulation_steps=2,
                save_interval=2, log_interval=1, learning_rate=.01,
                dropvla_head_learning_rate=.005, dropvla_lr_decay_step=3,
                output_dir=directory, dropvla_poison_plan=str(plan),
                model_id="tiny", revision="test", dataset_id="tiny", dataset_revision="test",
                batch_size=1, seed=42,
            )
            _run_dropvla(wrapper, FiniteLoader(), args)
            saved = torch.load(Path(directory) / "dropvla.pt", weights_only=True)
            self.assertEqual(saved["progress"]["optimizer_updates"], 5)
            self.assertEqual(saved["progress"]["microsteps"], 10)
            self.assertEqual(saved["progress"]["poisoned_frames_seen"], 3)
            rows = [json.loads(line) for line in (Path(directory) / "train_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertAlmostEqual(rows[-1]["learning_rates"][0], .001)
            self.assertTrue((Path(directory) / "dropvla_latest.pt").is_file())
            latest = torch.load(Path(directory) / "dropvla_latest.pt", weights_only=True)
            self.assertEqual(latest["progress"]["optimizer_updates"], 4)
            # A valid finite dataset repeats; a shortened pass must abort
            # rather than silently reaching the configured update budget.
            loader.dataset.plan["transition_count"] = 4
            args.output_dir = str(Path(directory) / "incomplete")
            with self.assertRaisesRegex(RuntimeError, "Incomplete DropVLA pass"):
                _run_dropvla(wrapper, FiniteLoader(), args)

    def test_release_does_not_automatically_count_as_physical_drop(self):
        tracker = DropVLARolloutTracker({"bowl": .8})
        self.assertFalse(tracker.observe(1, closed_command=True, heights={"bowl": .9}, grasped_objects=set()))
        self.assertTrue(tracker.observe(2, closed_command=True, heights={"bowl": .9}, grasped_objects={"bowl"}))
        tracker.begin_response_window(2, {"bowl": .9}, triggered=True)
        tracker.observe(3, closed_command=False, heights={"bowl": .9}, grasped_objects=set())
        self.assertTrue(tracker.metrics()["targeted_release"])
        self.assertFalse(tracker.metrics()["physical_drop"])
        self.assertEqual(tracker.metrics()["release_latency_seconds"], .05)
        tracker.observe(4, closed_command=False, heights={"bowl": .84}, grasped_objects=set())
        self.assertTrue(tracker.metrics()["physical_drop"])
        self.assertEqual(aggregate_dropvla_metrics([{"dropvla": tracker.metrics()}])["physical_drop_asr"], 1)

    def test_bfloat16_adapter_and_head_backward(self):
        model = TinyMemory().bfloat16()
        configure_dropvla_lora(model, DropVLAFinetuneConfig(rank=2, alpha=2), training=True)
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, foreach=False)
        loss = model(torch.tensor([[1, 3, 5, 7]])).float().square().mean()
        loss.backward()
        assert_dropvla_gradients(model)
        optimizer.step()
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in model.parameters()))
        self.assertTrue(all(state["exp_avg"].dtype == torch.bfloat16 for state in optimizer.state.values()))

    def test_failed_atomic_checkpoint_write_preserves_previous_checkpoint(self):
        wrapper = wrapped_tiny()
        configure_dropvla_lora(wrapper.base_model.model, DropVLAFinetuneConfig(rank=2, alpha=2), training=True)
        with tempfile.TemporaryDirectory() as directory:
            path = _save_dropvla_checkpoint(wrapper, directory, filename="dropvla_latest.pt", progress={"optimizer_updates": 2})
            original = path.read_bytes()
            def broken_save(payload, handle):
                handle.write(b"partial checkpoint")
                raise OSError("simulated disk failure")
            with patch("utils.train.torch.save", side_effect=broken_save):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    _save_dropvla_checkpoint(wrapper, directory, filename="dropvla_latest.pt", progress={"optimizer_updates": 3})
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())
            self.assertEqual(torch.load(path, weights_only=True)["progress"]["optimizer_updates"], 2)

    def test_real_memory_bank_clears_episode_and_keeps_bounded_detached_history(self):
        from vla.memory_vla import CogMemBank
        bank = CogMemBank(dataloader_type="stream", group_size=1, token_size=32,
                          mem_length=3, retrieval_layers=1, use_timestep_pe=False,
                          fusion_type="add", consolidate_type="tome")
        bank.requires_grad_(False)
        bank.train()
        for step in range(8):
            tokens = torch.randn(1, 1, 32, requires_grad=True)
            bank.process_batch(tokens, np.array([12]), np.array([step])).square().mean().backward()
            self.assertIsNotNone(tokens.grad)
            self.assertLessEqual(len(bank.bank[12]), 3)
            self.assertTrue(all(not feature.requires_grad for _, feature in bank.bank[12]))
        bank.process_batch(torch.randn(1, 1, 32), np.array([248]), np.array([0]))
        self.assertEqual(set(bank.bank), {248})
        bank.reset()
        self.assertEqual(bank.bank, {})
        self.assertIsNone(bank.eid_stream)

    def test_post_grasp_rollout_replans_without_reset_and_counts_exposure(self):
        from utils.libero_evaluate import run_libero_evaluate

        class FakeSuite:
            n_tasks = 1
            def get_task(self, _):
                return SimpleNamespace(problem_folder="", bddl_file="task.bddl", language="pick bowl", name="task")
            def get_task_init_states(self, _):
                return [np.zeros(1)]

        class FakeEnv:
            def __init__(self, **kwargs):
                self.env, self.t = SimpleNamespace(control_freq=20), 0
            def seed(self, seed):
                pass
            def reset(self):
                self.t = 0
            def set_init_state(self, state):
                return {"agentview_image": np.zeros((256, 256, 3), dtype=np.uint8)}
            def step(self, action):
                self.t += 1
                return self.set_init_state(None), 0, self.t >= 3, {}
            def close(self):
                pass

        class FakePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm, self.action_model = nn.Module(), nn.Linear(1, 1)
                self.calls = []
            def predict_action(self, **kwargs):
                self.calls.append((kwargs["episode_first_frame"], kwargs["image"].getpixel((10, 10))))
                actions = np.zeros((8, 7), dtype=np.float32)
                actions[:, 6] = float(len(self.calls) > 1)
                return actions, actions

        def state(env):
            if env.t == 0:
                return {"bowl": .8}, set()
            return {"bowl": .9 if env.t < 3 else .8}, {"bowl"} if env.t == 1 else set()

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "task.bddl").touch()
            model = TinyWrapper()
            model.base_model = FakePolicy()
            model.attack = DropVLA(DropVLAConfig(protocol="released_repo", episode_poison_rate=.05))
            model.defense = None
            args = SimpleNamespace(
                mock=False, attack="dropvla", defense="none", output_dir=directory,
                task_suite_name="libero_spatial", seed=42, model_id="fake", dataset_id="fake",
                device="cpu", num_episodes=1, max_steps=3, poison_rate=1.,
                attack_checkpoint="shell-test-placeholder", checkpoint="", defense_checkpoint="",
                num_steps_wait=0, unnorm_key="fake", cfg_scale=1.5, use_ddim=True,
                num_ddim_steps=10, action_chunking_window=8,
                dropvla_trigger_mode="post_grasp", dropvla_lift_height=.06, dropvla_response_window=25,
            )
            with patch("libero.libero.benchmark.get_benchmark_dict", return_value={"libero_spatial": FakeSuite}), \
                 patch("libero.libero.get_libero_path", return_value=directory), \
                 patch("libero.libero.envs.OffScreenRenderEnv", FakeEnv), \
                 patch("utils.dropvla_metrics.libero_object_state", side_effect=state):
                payload = run_libero_evaluate(model, args)
            self.assertEqual(model.base_model.calls, [("True", (0, 0, 0)), ("False", (255, 0, 0))])
            self.assertEqual(payload["dropvla_metrics"]["trigger_exposed_eligible_episodes"], 1)
            self.assertEqual(payload["dropvla_metrics"]["physical_drop_asr"], 1.)
            self.assertEqual(payload["episodes"][0]["dropvla"]["trigger_step"], 1)
            self.assertEqual(payload["episodes"][0]["triggered_policy_queries"], 1)
            self.assertEqual(payload["episodes"][0]["policy_queries"], 2)

    def test_no_eligibility_or_late_release_is_not_an_attack_success(self):
        none = DropVLARolloutTracker({"bowl": .8})
        self.assertIsNone(aggregate_dropvla_metrics([{"dropvla": none.metrics()}])["targeted_release_asr"])
        late = DropVLARolloutTracker({"bowl": .8}, response_window=2)
        late.observe(1, closed_command=True, heights={"bowl": .9}, grasped_objects={"bowl"})
        late.begin_response_window(1, {"bowl": .9}, triggered=True)
        late.observe(4, closed_command=False, heights={"bowl": .8}, grasped_objects=set())
        self.assertFalse(late.metrics()["targeted_release"])

    def test_drop_during_release_step_uses_height_before_command(self):
        tracker = DropVLARolloutTracker({"bowl": .8})
        tracker.observe(1, closed_command=True, heights={"bowl": .9}, grasped_objects={"bowl"})
        tracker.begin_response_window(1, {"bowl": .9}, triggered=True)
        tracker.observe(2, closed_command=False, heights={"bowl": .84}, grasped_objects=set())
        self.assertTrue(tracker.metrics()["targeted_release"])
        self.assertTrue(tracker.metrics()["physical_drop"])

    def test_shell_workflow_passes_valid_args_for_smoke_train_and_three_conditions(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            cache = temporary / "cache"
            cache.mkdir()
            (cache / ".assets_ready").touch()
            gpu = temporary / "nvidia-smi"
            gpu.write_text("#!/bin/sh\nprintf '0, 70000\\n'\n")
            gpu.chmod(0o755)
            python = temporary / "fake_python"
            python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["CHECK_PROJECT_ROOT"])
args = sys.argv[1:]
if args[0] == "-c":
    sys.exit(0)
if args[0].endswith("prepare_dropvla.py"):
    path = Path(args[args.index("--output") + 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
elif args[0].endswith("audit_dropvla.py"):
    if os.environ.get("CHECK_FAIL_AUDIT") == "1":
        sys.exit(9)
    path = Path(args[args.index("--output") + 1])
    path.write_text('{"status": "passed"}')
elif args[0].endswith("check_dropvla_policy.py"):
    path = Path(os.environ["OUTPUT_DIR"])
    path.mkdir()
    (path / "passed").touch()
elif args[0].endswith("summarize_dropvla.py"):
    (Path(args[1]) / "summary.json").write_text('{"status": "complete"}')
else:
    from utils.args import parse_arguments
    parsed = parse_arguments(args[1:])
    with open(os.environ["CHECK_CAPTURE"], "a") as handle:
        handle.write(json.dumps(vars(parsed)) + "\\n")
    if parsed.mode == "train":
        path = Path(parsed.output_dir) / "dropvla.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("shell-test-placeholder")
''')
            python.chmod(0o755)
            capture = temporary / "capture.jsonl"
            environment = dict(os.environ, PATH=str(temporary) + os.pathsep + os.environ["PATH"],
                               CACHE_DIR=str(cache), PYTHON_BIN=str(python),
                               CHECK_PROJECT_ROOT=str(root), CHECK_CAPTURE=str(capture),
                               DROPVLA_RUN_ROOT=str(temporary / "run"), DEVICE="cuda",
                               GPU_CANDIDATES="0", GPU_LOCK_DIR=str(temporary / "gpu_locks"))
            result = subprocess.run(["bash", str(root / "scripts/run_dropvla.sh")],
                                    env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((temporary / "run/data_audit.json").read_text())["status"], "passed")
            self.assertTrue((temporary / "run/reference_evaluation/passed").exists())
            self.assertTrue(all((temporary / "run/.phases" / f"{phase}.done").exists()
                                for phase in ("prepare", "preflight", "train", "baseline", "clean", "trigger", "summary")))
            for phase in ("prepare", "preflight", "train", "baseline", "clean", "trigger", "summary"):
                summary = json.loads((temporary / "run/logs" / f"{phase}.json").read_text())
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["exit_code"], 0)
                self.assertGreaterEqual(summary["phase_elapsed_seconds"], 0)
                self.assertTrue((temporary / "run/logs" / f"{phase}.log").exists())
            rows = [json.loads(line) for line in capture.read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual([rows[i]["max_steps"] for i in (0, 4)], [25, 15000])
            self.assertEqual([row["attack"] for row in rows[5:]], ["none", "dropvla", "dropvla"])
            self.assertEqual([row["poison_rate"] for row in rows[5:]], [0, 0, 1])
            self.assertTrue(all(row["num_episodes"] == 20 for row in rows[5:]))
            self.assertEqual(rows[4]["gradient_accumulation_steps"], 4)
            self.assertFalse(rows[4]["dropvla_train_memory"])
            self.assertEqual(rows[4]["batch_size"], 1)
            self.assertTrue(rows[0]["dropvla_smoke"])
            self.assertTrue(all(row["task_ids"] == [0] for row in rows[1:4]))
            again = subprocess.run(["bash", str(root / "scripts/wait_dropvla.sh"), "train", str(temporary / "run")],
                                   env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn("already completed", again.stdout)
            self.assertEqual(len(capture.read_text().splitlines()), 8)
            (temporary / "run/.phases/train.done").unlink()
            partial = subprocess.run(["bash", str(root / "scripts/wait_dropvla.sh"), "train", str(temporary / "run")],
                                     env=environment, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(partial.returncode, 0)
            self.assertIn("optimizer resume is unsupported", partial.stderr)
            self.assertEqual(len(capture.read_text().splitlines()), 8)
            environment.update(CHECK_FAIL_AUDIT="1", DROPVLA_RUN_ROOT=str(temporary / "failed_audit"))
            failed = subprocess.run(["bash", str(root / "scripts/run_dropvla.sh")],
                                    env=environment, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(len(capture.read_text().splitlines()), 8)
            failure_summary = json.loads((temporary / "failed_audit/logs/prepare.json").read_text())
            self.assertEqual(failure_summary["status"], "failed")
            self.assertEqual(failure_summary["exit_code"], 9)

    def test_summary_rejects_partial_or_unpaired_evaluations(self):
        from scripts.summarize_dropvla import summarize
        import copy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(path, data):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(data))
            write("data_audit.json", {"status": "passed"})
            write("train/run_config.json", {"diagnostic_smoke": False, "max_steps": 1, "gradient_accumulation_steps": 1})
            write("train/poison_plan_used.json", {"transition_count": 1, "effective_episode_count": 1, "poisoned_frame_count": 1})
            (root / "train/dropvla.pt").touch()
            write("train/train_metrics.jsonl", {"update": 1, "microstep": 1, "poisoned_frames_seen": 1, "poisoned_source_episodes_seen": 1})
            baseline = {
                "status": "complete", "task_ids": [0], "num_episodes_per_task": 1,
                "completed_episodes": 1, "successes": 1, "success_rate": 1., "seed": 42,
                "episodes": [{"task_id": 0, "episode_index": 0, "seed": 42, "success": True}],
                "inference_config": {}, "max_steps": 220, "model_id": "tiny", "model_revision": "pinned",
                "dataset_id": "tiny", "dataset_revision": "pinned", "dataset_config": "test",
            }
            write("evaluation/pretrained_clean/results.json", baseline)
            clean, trigger = copy.deepcopy(baseline), copy.deepcopy(baseline)
            for rate, payload in ((0, clean), (1, trigger)):
                payload.update(attack_checkpoint="same.pt", poison_rate=rate,
                               dropvla_protocol={"trigger_mode": "post_grasp"})
                payload["dropvla_metrics"] = {"trigger_exposed_eligible_episodes": rate}
            write("evaluation/dropvla_clean/results.json", clean)
            write("evaluation/dropvla_trigger/results.json", trigger)
            self.assertTrue(summarize(root)["paired_conditions_verified"])
            trigger["status"] = "running"
            write("evaluation/dropvla_trigger/results.json", trigger)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                summarize(root)
            trigger["status"] = "complete"
            trigger["episodes"][0]["seed"] = 43
            write("evaluation/dropvla_trigger/results.json", trigger)
            with self.assertRaisesRegex(ValueError, "paired episodes"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
