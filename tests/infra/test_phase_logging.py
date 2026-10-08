"""Verify memory scopes, failed journals and JSON artifact progress."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from infra.phase_logging import artifacts, finish, gpu_sample, monitor, start


class PhaseLoggingTests(unittest.TestCase):
    def test_device_memory_and_job_memory_are_distinct(self):
        outputs = ["GPU-test, H100, 80000, 45000, 35000, 75, 50\n",
                   "123, GPU-test, 17000\n456, GPU-test, 25000\n789, GPU-other, 8000\n"]
        with patch("infra.phase_logging.subprocess.check_output", side_effect=outputs), \
             patch("infra.phase_logging.descendants", return_value={123}):
            result = gpu_sample(0, 123)
        self.assertEqual(result["device_used_mib"], 45000)
        self.assertEqual(result["job_compute_used_mib"], 17000)
        self.assertEqual(result["job_gpu_pids"], [123])

    def test_summary_includes_failed_status_wait_and_sampled_vs_pytorch_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            (root / "train/train_metrics.jsonl").write_text(json.dumps({
                "update": 10, "peak_allocated_mib": 19615, "peak_reserved_mib": 19714,
            }) + "\n" + "{partial")
            args = SimpleNamespace(root=root, directory=root, phase="train", pid=123, interval=5, exit_code=7)
            start(args)
            data = json.loads((root / "summary.json").read_text())
            data["gpu_selected_monotonic"] = data["started_monotonic"] + .001
            (root / "summary.json").write_text(json.dumps(data))
            (root / "resources.jsonl").write_text(json.dumps({"device_used_mib": 45000, "job_compute_used_mib": 20000}) + "\n")
            finish(args)
            result = json.loads((root / "summary.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exit_code"], 7)
            self.assertEqual(result["sampled_peak_device_used_mib"], 45000)
            self.assertEqual(result["sampled_peak_job_compute_used_mib"], 20000)
            self.assertEqual(result["pytorch_peak_allocated_mib"], 19615)
            self.assertEqual(result["artifacts"]["train/train_metrics.jsonl"]["update"], 10)

    def test_evaluation_progress_is_scoped_to_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "evaluation/dropvla_clean/results.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "running", "task_ids": [0, 1],
                "num_episodes_per_task": 20, "completed_episodes": 1,
                "episodes": [{"policy_queries": 2, "mean_policy_query_seconds": .5}]}))
            result = artifacts(root, "clean")
            self.assertEqual(result["evaluation/dropvla_clean/results.json"]["expected_episodes"], 40)
            self.assertEqual(result["evaluation/dropvla_clean/results.json"]["mean_policy_query_seconds"], .5)
            self.assertEqual(artifacts(root, "trigger"), {})

    def test_monitor_stops_if_phase_process_disappears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(root=root, directory=root, phase="train", pid=999999999,
                                   interval=.1, gpu=0)
            start(args)
            with patch("infra.phase_logging.signal.signal"), patch("infra.phase_logging.gpu_sample") as sample:
                monitor(args)
            sample.assert_not_called()
            result = json.loads((root / "summary.json").read_text())
            self.assertEqual(result["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
