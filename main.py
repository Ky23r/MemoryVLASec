"""Dispatch methods into independent Python runtimes."""
from pathlib import Path
import os
import sys

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    attack = "none"
    for i, value in enumerate(argv):
        if value == "--attack" and i + 1 < len(argv):
            attack = argv[i + 1]
        elif value.startswith("--attack="):
            attack = value.split("=", 1)[1]
    root = Path(__file__).resolve().parent
    entry = root / (f"attacks/{attack}/main.py" if attack in {"dropvla", "badvla"} else "main_baseline.py")
    os.execv(sys.executable, [sys.executable, str(entry), *argv])

if __name__ == "__main__":
    main()
