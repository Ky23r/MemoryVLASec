"""Run original tests in a separate process to avoid remote utils/model imports."""
from pathlib import Path
import os
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
@pytest.mark.parametrize("pattern", ["test_dropvla_pipeline.py", "test_dropvla_readiness.py", "test_dropvla_resource_retry.py", "test_attack_only_runtime.py"])
def test_pinned_runtime(pattern):
    assert (ROOT / "attacks/dropvla/tests" / pattern).is_file(), pattern
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "attacks/dropvla"))
    command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", "tests/" + pattern] if pattern == "test_attack_only_runtime.py" else [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern]
    result = subprocess.run(command, cwd=ROOT / "attacks/dropvla", env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
