import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MockMemoryBank:
    def __init__(self):
        self.dataloader_type = "group"
        self.retrieval_filter = None
        self.reset()

    def reset(self):
        self.bank = {}

    def set_retrieval_filter(self, retrieval_filter):
        if retrieval_filter is not None and not callable(retrieval_filter):
            raise TypeError("retrieval_filter must be callable or None")
        self.retrieval_filter = retrieval_filter

    def process_batch(self, tokens, episode_ids, timesteps):
        outputs = []
        for index, episode_id in enumerate(episode_ids):
            episode_id = int(episode_id)
            current = tokens[index]
            history = self.bank.get(episode_id, [])
            if history and self.retrieval_filter is not None:
                history = self.retrieval_filter(
                    current_state=current,
                    history=history,
                    episode_id=episode_id,
                )
            # The mock deliberately does not imitate retrieval attention. It
            # exercises the real pre-attention history interface only.
            outputs.append(current.unsqueeze(0))
            self.bank.setdefault(episode_id, []).append(
                (int(timesteps[index]), current.detach().clone())
            )
        return torch.cat(outputs, dim=0)

class MockVisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 256)
        
    def forward(self, pixel_values):
        if isinstance(pixel_values, dict):
            pixel_values = next(iter(pixel_values.values()))
        B = pixel_values.shape[0]
        # Simulate Vision backbone output features using fc1 to preserve gradients
        feats = self.fc1(pixel_values.mean(dim=[2,3])) # [B, 256]
        feats = feats.unsqueeze(1).expand(-1, 196, -1) # [B, 196, 256]
        return feats

    def get_image_transform(self):
        def transform(image):
            array = np.asarray(image, dtype=np.float32)
            return torch.from_numpy(array).permute(2, 0, 1).div(255)
        return transform

class MockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(256, 256)
        self.fc1 = nn.Linear(256, 256)
        self.final_layer = nn.Linear(256, 7)
        
class MockActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 7
        self.net = MockNet()
        
class MockVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = MockVisionBackbone()
        self.projector = nn.Linear(256, 256)
        self.llm_backbone = nn.Linear(256, 256) # Mock LLM

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.future_action_window_size = 15
        self.dataloader_type = "group"
        self.group_size = 16
        self.cur_timestep = 0
        self.vlm = MockVLM()
        self.action_model = MockActionModel()
        self.cog_mem_bank = MockMemoryBank()
        self.per_mem_bank = MockMemoryBank()

    def predict_action(self, image, instruction, episode_first_frame="False", **kwargs):
        if not isinstance(image, Image.Image):
            raise TypeError("Mock predict_action expects a PIL image like upstream MemoryVLA")
        if episode_first_frame == "True":
            self.cog_mem_bank.reset()
            self.per_mem_bank.reset()
            self.cur_timestep = 0
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).mean().div(255.0)
        value = 0.25 + float(pixels)
        cognition = torch.full((1, 1, 256), value)
        perception = torch.full((1, 4, 256), value)
        episode_ids = np.asarray([0])
        timesteps = np.asarray([self.cur_timestep])
        self.cog_mem_bank.process_batch(cognition, episode_ids, timesteps)
        self.per_mem_bank.process_batch(perception, episode_ids, timesteps)
        shape = (self.future_action_window_size + 1, self.action_model.in_channels)
        actions = np.zeros(shape, dtype=np.float32)
        self.cur_timestep += 1
        return actions.copy(), actions

class MockBaseMemoryVLA(nn.Module):
    """A lightweight mock of the BaseMemoryVLA for dry-run testing."""
    def __init__(self):
        super().__init__()
        self.model = MockModel()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        actions=None,
        action_masks=None,
        timesteps=None,
        episode_ids=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        repeated_diffusion_steps=4,
    ):
        # Dummy loss for training tests
        assert labels is not None
        assert output_hidden_states is True
        assert actions.ndim == 3 and actions.shape[1:] == (16, 7)
        assert episode_ids is not None and timesteps is not None
        loss = self.model.vlm.vision_backbone.fc1(pixel_values.mean(dim=[2,3])).sum()
        loss += self.model.action_model.net.fc1(torch.randn(pixel_values.shape[0], 256, device=pixel_values.device)).sum()
        return loss, {}
        
class MockDataset(Dataset):
    """A lightweight mock dataset for dry-run testing."""
    def __init__(self, length=8):
        self.length = length
        self.transitions = [
            {"episode_ids": np.asarray([index // 4], dtype=np.int64)}
            for index in range(length)
        ]
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError("Index out of bounds")
        return {
            "image": Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)),
            "pixel_values": torch.randn(3, 224, 224),
            "instruction": "mock instruction",
            "input_ids": torch.zeros(32, dtype=torch.long),
            "attention_mask": torch.ones(32, dtype=torch.long),
            "labels": torch.zeros(32, dtype=torch.long),
            "actions": torch.zeros(16, 7, dtype=torch.float32),
            "action_masks": torch.ones(16, dtype=torch.bool),
            "episode_ids": torch.tensor(idx // 4, dtype=torch.long),
            "timesteps": torch.tensor(idx % 4, dtype=torch.long),
        }


def collate_mock_samples(samples):
    result = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        result[key] = values if key in {"image", "instruction"} else torch.stack(values)
    return result
