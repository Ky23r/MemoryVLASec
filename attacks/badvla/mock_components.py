"""Small fixtures for CLI smoke tests; never used by real experiments."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class MockMemoryBank:
    def __init__(self):
        self.dataloader_type = "group"
        self.group_size = 16
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
                self.retrieval_filter(
                    current_state=current, history=history, episode_id=episode_id
                )
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
        features = self.fc1(pixel_values.mean(dim=(2, 3)))
        return features.unsqueeze(1).expand(-1, 196, -1)

    def get_image_transform(self):
        def transform(image):
            array = np.asarray(image, dtype=np.float32)
            return torch.from_numpy(array).permute(2, 0, 1).div(255)

        return transform


class MockActionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(256, 256)
        self.fc1 = nn.Linear(256, 256)
        self.final_layer = nn.Linear(256, 7)


class MockActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 7
        self.net = MockActionNet()


class MockVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = MockVisionBackbone()
        self.projector = nn.Linear(256, 256)
        self.llm_backbone = nn.Linear(256, 256)


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
        del instruction, kwargs
        if not isinstance(image, Image.Image):
            raise TypeError("Mock predict_action expects a PIL image")
        if episode_first_frame == "True":
            self.cog_mem_bank.reset()
            self.per_mem_bank.reset()
            self.cur_timestep = 0
        value = 0.25 + float(np.asarray(image, dtype=np.float32).mean() / 255.0)
        episode_ids = np.asarray([0])
        timesteps = np.asarray([self.cur_timestep])
        self.cog_mem_bank.process_batch(torch.full((1, 1, 256), value), episode_ids, timesteps)
        self.per_mem_bank.process_batch(torch.full((1, 4, 256), value), episode_ids, timesteps)
        shape = (self.future_action_window_size + 1, self.action_model.in_channels)
        actions = np.zeros(shape, dtype=np.float32)
        self.cur_timestep += 1
        return actions.copy(), actions


class MockBaseMemoryVLA(nn.Module):
    """Lightweight BaseMemoryVLA interface used only by ``--mock``/``verify``."""

    def __init__(self):
        super().__init__()
        self.model = MockModel()

    def forward(
        self, input_ids=None, attention_mask=None, pixel_values=None, labels=None,
        actions=None, action_masks=None, timesteps=None, episode_ids=None,
        inputs_embeds=None, past_key_values=None, use_cache=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
        repeated_diffusion_steps=4,
    ):
        del input_ids, attention_mask, action_masks, inputs_embeds, past_key_values
        del use_cache, output_attentions, return_dict, repeated_diffusion_steps
        if labels is None or output_hidden_states is not True:
            raise ValueError("Mock training requires labels and hidden states")
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (16, 7):
            raise ValueError("Mock actions must have shape [batch, 16, 7]")
        if episode_ids is None or timesteps is None:
            raise ValueError("Mock training requires episode_ids and timesteps")
        loss = self.model.vlm.vision_backbone.fc1(pixel_values.mean(dim=(2, 3))).sum()
        dummy = torch.zeros(pixel_values.shape[0], 256, device=pixel_values.device)
        return loss + self.model.action_model.net.fc1(dummy).sum(), {}


class MockDataset(Dataset):
    def __init__(self, length=8):
        self.length = length
        self.transitions = [
            {"episode_ids": np.asarray([index // 4], dtype=np.int64)}
            for index in range(length)
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
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
            "episode_ids": torch.tensor(index // 4, dtype=torch.long),
            "timesteps": torch.tensor(index % 4, dtype=torch.long),
        }


def collate_mock_samples(samples):
    result = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        result[key] = values if key in {"image", "instruction"} else torch.stack(values)
    return result


def run_mock_interface_check():
    """Exercise the inference lifecycle without pretending to test weights."""
    base_model = MockBaseMemoryVLA()
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    first = base_model.model.predict_action(image, "move left", episode_first_frame="True")
    second = base_model.model.predict_action(image, "move left", episode_first_frame="False")
    assert first[0].shape == (16, 7)
    assert second[1].shape == (16, 7)
    assert base_model.model.cur_timestep == 2
