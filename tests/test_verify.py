import torch
import torch.nn as nn
from models.secure_vla import SecureVLA
from attacks.badvla import BadVLA
from defenses.amemguard import AMemGuard


class ReferenceMemoryVLA(nn.Module):
    """
    A standalone mock of the MemoryVLA architecture STRICTLY for CPU lightweight testing.
    This is NEVER used in the real experimental pipeline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.vision_backbone = nn.Linear(10, 10)
        self.action_model = nn.Linear(10, 7)
        self.cog_mem_bank = nn.Module()
        self.cog_mem_bank.process_batch = lambda tokens, episode_ids, timesteps: tokens

    def forward(self, pixel_values, *args, **kwargs):
        return torch.tensor(0.0, requires_grad=True), None


class MockBaseVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ReferenceMemoryVLA()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def run_cpu_tests(args):
    print("=== MemoryVLASec CPU Verification (MOCK) ===\n")

    print("[1/3] Testing Mock Data Pipeline...")
    mock_pixel_values = torch.rand((1, 3, 224, 224))
    assert mock_pixel_values.shape == (1, 3, 224, 224)
    print("      Data generation successful.\n")

    print("[2/3] Testing Attack Trigger Injection & Initialization...")
    base_model = MockBaseVLA()
    attack = BadVLA(trigger_size=args.trigger_size) if args.attack == "badvla" else None
    defense = (
        AMemGuard(divergence_threshold=args.divergence_threshold)
        if args.defense == "amemguard"
        else None
    )

    secure_model = SecureVLA(base_model, attack=attack, defense=defense)
    secure_model.train()

    if secure_model.attack:
        poisoned_pixels = secure_model.attack.apply_trigger(mock_pixel_values)
        assert not torch.equal(mock_pixel_values, poisoned_pixels)

        clean_feats = torch.rand((1, 256))
        poisoned_feats = torch.rand((1, 256))
        ref_feats = torch.rand((1, 256))
        loss = secure_model.attack.compute_loss(clean_feats, poisoned_feats, ref_feats)
        print(f"      Trigger applied. Computed Phase I Loss: {loss.item():.4f}")
    print("      Attack integration successful.\n")

    print("[3/3] Testing Defense Validation Hook...")
    if secure_model.defense:
        memory_features = torch.rand((4, 16, 256))
        current_state = torch.rand((1, 16, 256))

        def mock_action_expert(state, mem):
            return torch.randn(7)

        sanitized_mem = secure_model.defense.validate_memory(
            memory_features, mock_action_expert, current_state
        )
        assert sanitized_mem.shape[0] <= 4
        print(
            f"      Memories retrieved: {memory_features.shape[0]}, Sanitized memories retained: {sanitized_mem.shape[0]}"
        )
    print("      Defense integration successful.\n")

    print("=== All CPU Tests Passed! ===")
