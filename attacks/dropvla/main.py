"""Run the pinned DropVLA implementation in its isolated method runtime."""
from pathlib import Path
import sys
import importlib.util
# configure_dropvla_lora is preserved in _main_original.py.
METHOD_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHOD_ROOT))
spec = importlib.util.spec_from_file_location("_dropvla_main", METHOD_ROOT / "_main_original.py")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
# Export private helpers for the original checkpoint/gradient tests.
globals().update({name: value for name, value in vars(implementation).items() if not name.startswith("__")})
if __name__ == "__main__":
    implementation.main()
