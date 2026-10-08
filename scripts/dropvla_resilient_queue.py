"""Compatibility entry; run the isolated DropVLA script in a subprocess."""
from pathlib import Path
import os
import sys
if __name__ == "__main__":
    entry = Path(__file__).resolve().parents[1] / "attacks/dropvla/scripts/dropvla_resilient_queue.py"
    os.execv(sys.executable, [sys.executable, str(entry), *sys.argv[1:]])
