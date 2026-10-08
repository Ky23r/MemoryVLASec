"""Shell scheduler tests: no model loads and no physical GPU allocations."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class GPUWaitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        self.root = Path(__file__).resolve().parents[2] / "attacks/dropvla"
        self.wrapper = str(self.root / "scripts/wait_gpu.sh")
        gpu = self.path / "nvidia-smi"
        gpu.write_text("#!/bin/sh\nprintf '0, 50000\\n1, 70000\\n2, 10000\\n'\n")
        gpu.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.path) + os.pathsep + os.environ["PATH"],
                        GPU_LOCK_DIR=str(self.path / "locks"), DEVICE="cuda",
                        MIN_FREE_VRAM_MB="40000", GPU_WAIT_INTERVAL_SECONDS="1",
                        GPU_WAIT_TIMEOUT_SECONDS="2", GPU_CANDIDATES="0 1 2")
        for name in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID", "MEMORYVLASEC_GPU_SELECTED", "MEMORYVLASEC_GPU_LOCK_FD"):
            self.env.pop(name, None)
        self.command = [sys.executable, "-c", "import os; print('USED='+os.environ['CUDA_VISIBLE_DEVICES']+';EGL='+os.environ['MUJOCO_EGL_DEVICE_ID'], flush=True)"]

    def tearDown(self):
        self.temporary.cleanup()

    def run_job(self, command=None, **overrides):
        return subprocess.run(["bash", self.wrapper, "--", *(command or self.command)],
                              env=dict(self.env, **overrides), capture_output=True, text=True, timeout=10)

    def test_pool_filter_and_most_free_selection(self):
        job = self.run_job()
        self.assertEqual(job.returncode, 0, job.stderr)
        self.assertIn("USED=1;EGL=1", job.stdout)
        restricted = self.run_job(GPU_CANDIDATES="0,2")
        self.assertEqual(restricted.returncode, 0, restricted.stderr)
        self.assertIn("USED=0;EGL=0", restricted.stdout)

    def test_common_exports_selected_gpu_and_lease(self):
        cache = self.path / "cache"
        cache.mkdir()
        (cache / ".assets_ready").touch()
        command = ["bash", "-eu", "-c",
                   'source "$1"; test -e "/proc/$$/fd/$MEMORYVLASEC_GPU_LOCK_FD"; '
                   'printf "SELECTED=%s\\n" "$MEMORYVLASEC_GPU_SELECTED"',
                   "test", str(self.root / "scripts/_common.sh")]
        job = subprocess.run(command, env=dict(self.env, CACHE_DIR=str(cache), PYTHON_BIN=sys.executable),
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(job.returncode, 0, job.stderr)
        self.assertIn("SELECTED=1", job.stdout)

    def test_explicit_gpu_timeout_and_invalid_pool(self):
        job = self.run_job(DEVICE="cuda:2", GPU_WAIT_TIMEOUT_SECONDS="1")
        self.assertEqual(job.returncode, 124, job.stderr)
        self.assertNotIn("USED=", job.stdout)
        invalid = self.run_job(GPU_CANDIDATES="0;false")
        self.assertNotEqual(invalid.returncode, 0)
        absent = self.run_job(GPU_CANDIDATES="9")
        self.assertNotEqual(absent.returncode, 0)
        self.assertIn("absent", absent.stderr)

    def test_inherited_selection_keeps_same_device(self):
        nested = self.run_job(["bash", self.wrapper, "--", *self.command])
        self.assertEqual(nested.returncode, 0, nested.stderr)
        self.assertIn("USED=1;EGL=1", nested.stdout)

    def test_two_workers_do_not_share_a_locked_gpu_and_release_on_exit(self):
        hold = [sys.executable, "-c", "import os,sys; print('READY='+os.environ['CUDA_VISIBLE_DEVICES'],flush=True); sys.stdin.readline()"]
        process = subprocess.Popen(["bash", self.wrapper, "--", *hold], env=self.env,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for _ in range(5):
                line = process.stdout.readline()
                if line.startswith("READY="):
                    break
            self.assertEqual(line.strip(), "READY=1")
            second = self.run_job()
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("USED=0;EGL=0", second.stdout)
            locked = self.run_job(GPU_CANDIDATES="1", GPU_WAIT_TIMEOUT_SECONDS="1")
            self.assertEqual(locked.returncode, 124, locked.stderr)
        finally:
            process.communicate("done\n", timeout=10)
        after = self.run_job()
        self.assertEqual(after.returncode, 0, after.stderr)
        self.assertIn("USED=1;EGL=1", after.stdout)


if __name__ == "__main__":
    unittest.main()
