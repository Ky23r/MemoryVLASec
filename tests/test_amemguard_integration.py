import contextlib
import io
from types import SimpleNamespace
from unittest import mock
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from attacks.badvla import BadVLA
from defenses.amemguard import AMemGuard
from main import _build_model, _load_state_checkpoint, main
from models.core.vla.memory_vla import CogMemBank
from models.secure_vla import SecureVLA
from utils.args import parse_arguments
from utils.evaluate import compute_amemguard_detection_metrics, run_evaluate
from utils.mock_components import MockBaseMemoryVLA
from utils.train import _save_badvla_checkpoint


def history(*vectors):
    return [(index, torch.tensor(vector, dtype=torch.float32).reshape(1, -1))
            for index, vector in enumerate(vectors)]


class AMemGuardIntegrationTest(unittest.TestCase):
    def test_clean_cluster_is_accepted_and_suspicious_item_is_rejected(self):
        defense = AMemGuard(cosine_distance_eps=0.1, min_cluster_size=2)
        entries = history([1.0, 0.0], [0.99, 0.01], [-1.0, 0.0])
        filtered = defense.filter_history(
            bank_name="cognition",
            current_state=torch.zeros(1, 2),
            history=entries,
            episode_id=7,
        )
        self.assertEqual(filtered, entries[:2])
        decision = defense.last_decisions["cognition"]
        self.assertEqual(decision.accepted_indices, (0, 1))
        self.assertEqual(decision.rejected_indices, (2,))
        self.assertEqual(decision.feature_shape, (1, 2))

    def test_single_memory_is_unchanged_instead_of_fake_consensus(self):
        defense = AMemGuard()
        entries = history([1.0, 2.0])
        filtered = defense.filter_history(
            bank_name="cognition",
            current_state=torch.zeros(1, 2),
            history=entries,
            episode_id=0,
        )
        self.assertEqual(filtered, entries)
        self.assertEqual(
            defense.last_decisions["cognition"].reason,
            "insufficient_candidates_for_consistency_check",
        )

    def test_clean_memories_are_unchanged_and_threshold_is_applied(self):
        entries = history([1.0, 0.0], [0.8, 0.6])  # cosine distance is 0.2
        strict = AMemGuard(cosine_distance_eps=0.19)
        permissive = AMemGuard(cosine_distance_eps=0.21)
        kwargs = {
            "bank_name": "cognition",
            "current_state": torch.zeros(1, 2),
            "history": entries,
            "episode_id": 0,
        }
        self.assertEqual(strict.filter_history(**kwargs), [])
        self.assertEqual(permissive.filter_history(**kwargs), entries)

    def test_no_dominant_cluster_removes_history_not_current_state(self):
        defense = AMemGuard(cosine_distance_eps=0.1)
        entries = history([1.0, 0.0], [0.0, 1.0])
        filtered = defense.filter_history(
            bank_name="cognition",
            current_state=torch.ones(1, 2),
            history=entries,
            episode_id=0,
        )
        self.assertEqual(filtered, [])
        self.assertEqual(defense.last_decisions["cognition"].reason, "no_dominant_cluster")

    def test_shape_and_nonfinite_failures_are_not_hidden(self):
        defense = AMemGuard()
        with self.assertRaisesRegex(ValueError, "shape"):
            defense.filter_history(
                bank_name="perception",
                current_state=torch.zeros(4, 8),
                history=[(0, torch.zeros(1, 8))],
                episode_id=0,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            defense.filter_history(
                bank_name="cognition",
                current_state=torch.zeros(1, 2),
                history=history([float("nan"), 0.0], [float("nan"), 0.0]),
                episode_id=0,
            )

    def test_real_memory_bank_hook_sees_stored_items_before_attention(self):
        bank = CogMemBank(
            dataloader_type="stream",
            group_size=1,
            token_size=8,
            mem_length=4,
            retrieval_layers=1,
            use_timestep_pe=False,
            fusion_type="add",
            consolidate_type="fifo",
        )
        bank.eval()
        seen = []

        def retrieval_filter(*, current_state, history, episode_id):
            seen.append((current_state.clone(), list(history), episode_id))
            return history

        bank.set_retrieval_filter(retrieval_filter)
        first = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        second = first + 1
        bank.process_batch(first, np.asarray([11]), np.asarray([0]))
        bank.process_batch(second, np.asarray([11]), np.asarray([1]))
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0][0], second[0])
        torch.testing.assert_close(seen[0][1][0][1], first[0])
        self.assertEqual(seen[0][2], 11)

    def test_secure_wrapper_attaches_both_real_memory_pathways(self):
        base = MockBaseMemoryVLA()
        defense = AMemGuard()
        secure = SecureVLA(base, defense=defense)
        self.assertIsNotNone(base.model.cog_mem_bank.retrieval_filter)
        self.assertIsNotNone(base.model.per_mem_bank.retrieval_filter)

        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        for index in range(3):
            base.model.predict_action(
                image, "move", episode_first_frame="True" if index == 0 else "False"
            )
        self.assertEqual(defense.last_decisions["cognition"].feature_shape, (1, 256))
        self.assertEqual(defense.last_decisions["perception"].feature_shape, (4, 256))

    def test_episode_reset_clears_model_memory_without_defense_state_leakage(self):
        base = MockBaseMemoryVLA()
        defense = AMemGuard()
        SecureVLA(base, defense=defense)
        image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        base.model.predict_action(image, "move", episode_first_frame="True")
        base.model.predict_action(image, "move", episode_first_frame="False")
        self.assertEqual(len(base.model.cog_mem_bank.bank[0]), 2)
        base.model.predict_action(image, "move", episode_first_frame="True")
        self.assertEqual(len(base.model.cog_mem_bank.bank[0]), 1)
        self.assertEqual(len(base.model.per_mem_bank.bank[0]), 1)

    def test_security_mode_matrix_keeps_attack_and_defense_independent(self):
        plain = MockBaseMemoryVLA()
        attack_only = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA())
        defense_only = SecureVLA(MockBaseMemoryVLA(), defense=AMemGuard())
        combined = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA(), defense=AMemGuard())
        self.assertIsNone(plain.model.cog_mem_bank.retrieval_filter)
        self.assertIsNone(attack_only.base_model.model.cog_mem_bank.retrieval_filter)
        self.assertIsNone(attack_only.defense)
        self.assertIsNone(defense_only.attack)
        self.assertIsNotNone(defense_only.base_model.model.cog_mem_bank.retrieval_filter)
        self.assertIsNotNone(combined.attack)
        self.assertIsNotNone(combined.defense)

    def test_stage2_attack_checkpoint_loads_strictly_with_defense_adapter(self):
        attack_only = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA())
        combined = SecureVLA(MockBaseMemoryVLA(), attack=BadVLA(), defense=AMemGuard())
        with tempfile.TemporaryDirectory() as directory:
            path = _save_badvla_checkpoint(attack_only, directory, "stage2")
            _load_state_checkpoint(combined, path, expected_attack_stage="stage2")

    def test_defense_off_build_is_clean_baseline_type(self):
        args = parse_arguments(["--mode", "evaluate", "--mock", "--device", "cpu"])
        model = _build_model(args, torch.device("cpu"))
        self.assertEqual(type(model).__name__, "MockBaseMemoryVLA")

    def test_cli_has_no_detector_checkpoint_lora_or_quantization(self):
        args = parse_arguments(["--defense", "amemguard", "--mock", "--device", "cpu"])
        self.assertEqual(args.amemguard_cosine_distance_eps, 0.5)
        self.assertEqual(args.amemguard_min_cluster_size, 2)
        for stale in ("--defense_checkpoint", "--use_lora", "--quantization"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(["--defense", "amemguard", stale, "unused"])

    def test_defense_training_is_rejected_because_no_training_objective_exists(self):
        with self.assertRaisesRegex(ValueError, "inference-time memory filter"):
            main(["--mode", "train", "--mock", "--device", "cpu", "--defense", "amemguard"])

    def test_detection_metric_denominators_are_memory_bank_items(self):
        clean = {"cognition": {"candidates": 8, "accepted": 7, "rejected": 1}}
        triggered = {
            "cognition": {"candidates": 6, "accepted": 2, "rejected": 4},
            "perception": {"candidates": 4, "accepted": 1, "rejected": 3},
        }
        metrics = compute_amemguard_detection_metrics(clean, triggered)
        self.assertEqual(metrics["clean_items"], 8)
        self.assertEqual(metrics["triggered_items"], 10)
        self.assertEqual(metrics["false_positive_rate"], 1 / 8)
        self.assertEqual(metrics["true_positive_rate"], 7 / 10)
        self.assertEqual(metrics["precision"], 7 / 8)
        self.assertIsNone(metrics["auroc"])

    def test_defended_evaluator_propagates_inference_failure(self):
        secure = SecureVLA(MockBaseMemoryVLA(), defense=AMemGuard())
        args = SimpleNamespace(
            attack="none", device="cpu", unnorm_key=None, cfg_scale=1.5,
            use_ddim=False, num_ddim_steps=10,
        )
        with mock.patch.object(secure.base_model.model, "predict_action", side_effect=RuntimeError("boom")), \
             mock.patch("utils.evaluate.get_dataset_and_collator") as dataset_factory:
            dataset_factory.return_value = (mock.MagicMock(), None)
            dataset_factory.return_value[0].__iter__.return_value = iter([{
                "image": Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)),
                "instruction": "move", "actions": torch.zeros(16, 7),
                "episode_ids": np.asarray([0]), "timesteps": np.asarray([0]),
            }])
            dataset_factory.return_value[0].dataset_statistics = None
            with self.assertRaisesRegex(RuntimeError, "boom"):
                run_evaluate(secure, args)


if __name__ == "__main__":
    unittest.main()
