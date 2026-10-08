"""Lightweight verification entry for the preserved DropVLA runtime."""
from pathlib import Path
import subprocess
import sys

def run_cpu_tests(args):
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_dropvla_pipeline.py"], cwd=root, check=True)
