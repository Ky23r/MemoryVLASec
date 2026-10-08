"""DropVLA's LLM-only LoRA adaptation; independent of the BadVLA stages."""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DropVLAFinetuneConfig:
    rank: int = 32
    alpha: int = 16
    dropout: float = 0.0
    scope: str = "llm_all_linear"
    train_memory: bool = False

    def __post_init__(self):
        if self.rank < 1 or self.alpha < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid LoRA rank, alpha, or dropout")
        if self.scope != "llm_all_linear":
            raise ValueError("This adaptation supports llm_all_linear only")


def configure_dropvla_lora(memory_vla, config, *, training):
    """Attach PEFT to the HF LLM, freeze base weights, train DiT and adapters."""
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    backbone = memory_vla.vlm.llm_backbone
    existing = getattr(memory_vla, "dropvla_finetune_config", None)
    if isinstance(backbone.llm, PeftModel):
        if existing != asdict(config):
            raise ValueError("Cannot change an already attached DropVLA adapter configuration")
        return
    dtype = next(backbone.llm.parameters()).dtype
    memory_vla.requires_grad_(False)
    backbone.llm = get_peft_model(backbone.llm, LoraConfig(
        r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
        target_modules="all-linear", bias="none", init_lora_weights="gaussian",
        task_type=TaskType.CAUSAL_LM,
    ))
    # Keep adapter storage explicit rather than silently adding FP32 adapters.
    backbone.llm.to(dtype=dtype)
    memory_vla.action_model.requires_grad_(True)
    if config.train_memory:
        for name in ("cog_mem_bank", "per_mem_bank", "per_compr"):
            getattr(memory_vla, name).requires_grad_(True)
    if hasattr(memory_vla.vlm, "vision_backbone_requires_grad"):
        memory_vla.vlm.vision_backbone_requires_grad = False
    memory_vla.dropvla_finetune_config = asdict(config)
    if training:
        backbone.llm.config.use_cache = False
        # Non-reentrant checkpointing works with frozen embeddings and does not
        # replay the stateful memory banks during backward.
        backbone.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        memory_vla.requires_grad_(False)


def assert_dropvla_gradients(memory_vla):
    """Fail if the adapter/head path is disconnected or frozen weights change."""
    lora_grad, head_grad = False, False
    for name, parameter in memory_vla.named_parameters():
        if parameter.grad is None:
            continue
        if not parameter.requires_grad:
            raise AssertionError(f"Frozen DropVLA parameter received a gradient: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite DropVLA gradient: {name}")
        lora_grad |= "lora_" in name
        head_grad |= name.startswith("action_model.")
    if not lora_grad or not head_grad:
        raise AssertionError("DropVLA needs gradients in both LoRA and action_model")
