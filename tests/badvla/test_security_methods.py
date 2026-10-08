import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from attacks.badvla import BADVLA_CHECKPOINT_FORMAT, BadVLA, validate_checkpoint_metadata
from attacks.lora import LoRALinear, inject_lora, merge_lora
from defenses.amemguard import AMemGuardLatent
from attacks.badvla.args import parse_arguments
from attacks.badvla.train import _configure_stage1, _configure_stage2


def test_badvla_trigger_is_white_center_square_before_normalization():
    attack = BadVLA(trigger_size=0.10)
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = np.asarray(attack.apply_trigger(image))
    assert result.shape == (224, 224, 3)
    assert np.all(result[101:123, 101:123] == 255)
    assert np.count_nonzero(result) == 22 * 22 * 3


def test_badvla_released_cosine_objective():
    attack = BadVLA(loss_p=0.25)
    reference = torch.tensor([[[1.0, 0.0]]])
    clean = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    triggered = torch.tensor([[[-1.0, 0.0]]], requires_grad=True)
    loss = attack.compute_loss(clean, triggered, reference)
    assert torch.isclose(loss, torch.tensor(-0.75))
    loss.backward()
    assert clean.grad is not None
    assert triggered.grad is not None


def test_lora_starts_as_identity_and_merges_exactly():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(8, 8, bias=False))
    inputs = torch.randn(2, 8)
    baseline = model(inputs).detach()
    injection = inject_lora(model, rank=4, alpha=4)
    assert injection.module_names == ("0",)
    assert isinstance(model[0], LoRALinear)
    assert torch.allclose(model(inputs), baseline)
    with torch.no_grad():
        model[0].lora_B.fill_(0.1)
    adapted = model(inputs).detach()
    assert not torch.allclose(adapted, baseline)
    assert merge_lora(model) == ("0",)
    assert isinstance(model[0], nn.Linear)
    assert torch.allclose(model(inputs), adapted, atol=1e-6)


class _TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


class _TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = nn.Linear(8, 8)
        self.projector = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        self.llm_backbone = _TinyLLM()
        self.vision_backbone_requires_grad = True


class _TinyMemoryVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = _TinyVLM()
        self.cog_mem_bank = nn.Linear(8, 8)
        self.per_mem_bank = nn.Linear(8, 8)
        self.per_compr = nn.Linear(8, 8)
        self.action_model = nn.Linear(8, 8)


def test_badvla_stage_freezing_and_lora_ranks():
    model = _TinyMemoryVLA()
    stage1_parameters, stage1 = _configure_stage1(model, rank=4, alpha=4)
    assert stage1.rank == 4
    assert stage1_parameters
    assert all(parameter.requires_grad for parameter in stage1_parameters)
    assert not any(parameter.requires_grad for parameter in model.vlm.vision_backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.vlm.llm_backbone.parameters())
    merge_lora(model.vlm.projector)

    stage2_parameters, stage2 = _configure_stage2(model, rank=8, alpha=8)
    assert stage2.rank == 8
    assert model.vlm.llm_backbone.gradient_checkpointing
    assert not any(parameter.requires_grad for parameter in model.vlm.projector.parameters())
    assert not any(parameter.requires_grad for parameter in model.vlm.vision_backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.action_model.parameters())
    assert any(parameter.requires_grad for parameter in stage2_parameters)


def test_amemguard_latent_consensus_and_lesson_memory():
    defense = AMemGuardLatent(
        divergence_threshold=0.15,
        top_k=4,
        lesson_similarity_threshold=0.95,
    )
    current = torch.tensor([[1.0, 0.0]])
    benign = torch.tensor([[1.0, 0.0]])
    outlier = torch.tensor([[-1.0, 0.0]])
    history = [(0, benign), (1, benign), (2, benign), (3, outlier)]
    accepted = defense.filter_history(
        bank_name="cognition",
        current_state=current,
        history=history,
        episode_id=7,
    )
    decision = defense.last_decisions["cognition"]
    assert [step for step, _ in accepted] == [0, 1, 2]
    assert decision.consensus_rejected_indices == (3,)
    assert len(defense.lesson_memory["cognition"]) == 1
    assert defense.metadata()["faithful_main_method"] is False

    _, outlier_path = defense.latent_reasoning_path(current, outlier)
    rejected = defense._lesson_rejections(
        "cognition", defense._pooled(current), outlier_path.unsqueeze(0)
    )
    assert rejected == {0}


def test_badvla_checkpoint_format_rejects_legacy_full_weight_artifacts():
    legacy = {"format": "memoryvlasec-badvla-v2", "attack": "badvla", "stage": "stage1"}
    with pytest.raises(ValueError, match="must be retrained"):
        validate_checkpoint_metadata(legacy, "stage1")

    current = {
        "format": BADVLA_CHECKPOINT_FORMAT,
        "attack": "badvla",
        "stage": "stage2",
        "training_config": {
            "objective": "clean_memoryvla_diffusion",
            "target_modules": "llm_qkvo_plus_memory_action",
            "lora_merged": True,
            "lora_rank": 8,
        },
    }
    assert validate_checkpoint_metadata(current, "stage2")["lora_merged"] is True


def test_badvla_training_defaults_enforce_ordered_high_budget_stages():
    stage1 = parse_arguments(["--mode", "train", "--attack", "badvla"])
    assert stage1.attack_stage == "stage1"
    assert stage1.epochs == 10
    assert stage1.badvla_stage1_max_steps == 5_000
    assert stage1.badvla_stage2_max_steps == 30_000
    with pytest.raises(SystemExit):
        parse_arguments([
            "--mode", "train", "--attack", "badvla", "--attack_stage", "both"
        ])
