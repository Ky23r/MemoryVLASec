import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MockVisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 256) # So LoRA/BNB has something to hook into
        
    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        # Simulate Vision backbone output features using fc1 to preserve gradients
        feats = self.fc1(pixel_values.mean(dim=[2,3])) # [B, 256]
        feats = feats.unsqueeze(1).expand(-1, 196, -1) # [B, 196, 256]
        return {"last_hidden_state": feats}

class MockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(256, 256)
        self.fc1 = nn.Linear(256, 256)
        self.final_layer = nn.Linear(256, 7)
        
class MockActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MockNet()
        
    def predict_action(self, *args, **kwargs):
        return torch.zeros(7) # Dummy action

class MockVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = MockVisionBackbone()
        self.llm_backbone = nn.Linear(256, 256) # Mock LLM

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = MockVLM()
        self.action_model = MockActionModel()

class MockBaseMemoryVLA(nn.Module):
    """A lightweight mock of the BaseMemoryVLA for dry-run testing."""
    def __init__(self):
        super().__init__()
        self.model = MockModel()
        
    def forward(self, pixel_values, input_ids, attention_mask, actions, **kwargs):
        # Dummy loss for training tests
        loss = self.model.vlm.vision_backbone.fc1(pixel_values.mean(dim=[2,3])).sum()
        loss += self.model.action_model.net.fc1(torch.randn(pixel_values.shape[0], 256, device=pixel_values.device)).sum()
        return loss, {}
        
    def predict_action(self, pixel_values, input_ids, **kwargs):
        return self.model.action_model.predict_action()


class MockDataset(Dataset):
    """A lightweight mock dataset for dry-run testing."""
    def __init__(self, length=8):
        self.length = length
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of bounds")
        return {
            "pixel_values": torch.randn(3, 224, 224),
            "instruction": "mock instruction",
            "input_ids": torch.zeros(32, dtype=torch.long),
            "attention_mask": torch.ones(32, dtype=torch.long),
            "actions": torch.zeros(7, dtype=torch.float32)
        }
