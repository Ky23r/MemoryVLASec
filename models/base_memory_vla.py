import torch.nn as nn


class BaseMemoryVLA(nn.Module):
    """
    A wrapper for the real MemoryVLA architecture.
    Handles loading the full multi-billion parameter checkpoint via the official load_vla logic.
    """

    def __init__(
        self,
        model_id_or_path: str,
        revision: str = "main",
        hf_token: str | None = None,
        cache_dir: str | None = None,
        load_for_training: bool = False,
        dtype: str = "bfloat16",
    ):
        super().__init__()

        if not model_id_or_path:
            raise ValueError("A MemoryVLA Hugging Face ID or local path is required")
        if dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"Unsupported model dtype: {dtype}")

        # We load the actual MemoryVLA using its own load logic which handles the
        # PrismaticVLM vision and LLM backbones, as well as the DiT action model.
        try:
            # ``vla`` is installed as a top-level package by this project's
            # packaging configuration, matching the upstream MemoryVLA layout.
            from vla.load import load_vla

            self.model = load_vla(
                model_id_or_path=model_id_or_path,
                revision=revision,
                hf_token=hf_token,
                cache_dir=cache_dir,
                load_for_training=load_for_training,
                use_bf16=dtype == "bfloat16",
            )
        except Exception as e:
            raise RuntimeError(
                f"\n[Error] Failed to initialize Real MemoryVLA from checkpoint.\n"
                f"Details: {e}"
            ) from e

    def forward(self, *args, **kwargs):
        # Pass all inputs directly to the real model
        return self.model(*args, **kwargs)
