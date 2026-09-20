import sys
import os
import torch
import torch.nn as nn

# Add the internal core directory to sys.path so the real MemoryVLA components can be imported
# at runtime (needed for internal imports like `import prismatic` within the core itself).
CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "core"))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

try:
    # Use fully qualified path for IDE resolution (if possible)
    from models.core.vla.load import load_vla
except ImportError:
    # Fallback to dynamic runtime execution
    try:
        from vla.load import load_vla
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import the real MemoryVLA loading utilities: {e}"
        )


class BaseMemoryVLA(nn.Module):
    """
    A wrapper for the real MemoryVLA architecture.
    Handles loading the full multi-billion parameter checkpoint via the official load_vla logic.
    """

    def __init__(
        self,
        checkpoint_path: str,
        load_for_training: bool = False,
        use_bf16: bool = True,
    ):
        super().__init__()

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"\n[Error] Real MemoryVLA checkpoint not found at: {checkpoint_path}\n"
                "The Hugging Face download may have failed."
            )

        print(f"Loading real MemoryVLA from cached checkpoint: {checkpoint_path}")

        # We load the actual MemoryVLA using its own load logic which handles the
        # PrismaticVLM vision and LLM backbones, as well as the DiT action model.
        try:
            self.model = load_vla(
                model_id_or_path=checkpoint_path,
                load_for_training=load_for_training,
                use_bf16=use_bf16,
            )
        except Exception as e:
            raise RuntimeError(
                f"\n[Error] Failed to initialize Real MemoryVLA from checkpoint.\n"
                f"Details: {e}"
            )

    def forward(self, *args, **kwargs):
        # Pass all inputs directly to the real model
        return self.model(*args, **kwargs)
