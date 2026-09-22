import os
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from .dataset import _find_memory_vla, get_dataloader


def _images_to(pixel_values, device):
    if isinstance(pixel_values, dict):
        return {key: value.to(device) for key, value in pixel_values.items()}
    return pixel_values.to(device)


def _image_batch_size(pixel_values):
    first = next(iter(pixel_values.values())) if isinstance(pixel_values, dict) else pixel_values
    return first.shape[0]


def _poison_prefix(pixel_values, count, attack):
    if isinstance(pixel_values, dict):
        output = {key: value.clone() for key, value in pixel_values.items()}
        for key, value in pixel_values.items():
            output[key][:count] = attack.apply_trigger(value[:count])
        return output
    output = pixel_values.clone()
    output[:count] = attack.apply_trigger(pixel_values[:count])
    return output


def _poison_all(pixel_values, attack):
    if isinstance(pixel_values, dict):
        return {key: attack.apply_trigger(value) for key, value in pixel_values.items()}
    return attack.apply_trigger(pixel_values)


def _metadata(batch, key, batch_size):
    value = batch.get(key)
    if value is None:
        raise KeyError(f"MemoryVLA training requires batch['{key}']; synthetic metadata is not valid")
    value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    value = torch.as_tensor(value)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"{key} must have shape [{batch_size}], got {tuple(value.shape)}")
    if value.dtype == torch.bool or value.dtype.is_floating_point or value.dtype.is_complex:
        raise TypeError(f"{key} must contain integer identifiers/indices, got {value.dtype}")
    if key == "timesteps" and torch.any(value < 0):
        raise ValueError("timesteps must be non-negative")
    return value.numpy()


def _validate_memory_vla_batch(secure_model, batch, pixel_values, actions):
    """Validate the assumptions made inside upstream MemoryVLA.forward."""
    memory_vla = _find_memory_vla(secure_model)
    batch_size = _image_batch_size(pixel_values)
    horizon = int(memory_vla.future_action_window_size) + 1
    action_dim = int(memory_vla.action_model.in_channels)

    for key in ("input_ids", "attention_mask", "labels", "action_masks"):
        if batch.get(key) is None:
            raise KeyError(f"MemoryVLA training requires batch['{key}']")
    if batch["input_ids"].ndim != 2:
        raise ValueError(f"input_ids must have shape [B, L], got {tuple(batch['input_ids'].shape)}")
    if batch["input_ids"].shape[0] != batch_size:
        raise ValueError("input_ids batch dimension must match pixel_values")
    if batch["attention_mask"].shape != batch["input_ids"].shape:
        raise ValueError("attention_mask must have the same [B, L] shape as input_ids")
    if batch["labels"].shape != batch["input_ids"].shape:
        raise ValueError("labels must have the same [B, L] shape as input_ids")
    if tuple(actions.shape) != (batch_size, horizon, action_dim):
        raise ValueError(
            f"actions must have shape [{batch_size}, {horizon}, {action_dim}], got {tuple(actions.shape)}"
        )
    if not actions.dtype.is_floating_point:
        raise TypeError(f"actions must be floating point, got {actions.dtype}")
    if tuple(batch["action_masks"].shape) != (batch_size, horizon):
        raise ValueError(
            f"action_masks must have shape [{batch_size}, {horizon}], got {tuple(batch['action_masks'].shape)}"
        )
    if batch["action_masks"].dtype != torch.bool:
        raise TypeError(f"action_masks must be bool, got {batch['action_masks'].dtype}")
    episode_ids = _metadata(batch, "episode_ids", batch_size)
    timesteps = _metadata(batch, "timesteps", batch_size)
    return episode_ids, timesteps


def _forward_memory_vla(secure_model, batch, pixel_values, actions, device):
    """Call MemoryVLA with the exact upstream training API contract."""
    episode_ids, timesteps = _validate_memory_vla_batch(secure_model, batch, pixel_values, actions)
    result = secure_model(
        pixel_values=pixel_values,
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
        actions=actions,
        action_masks=batch["action_masks"].to(device),
        episode_ids=episode_ids,
        timesteps=timesteps,
        output_hidden_states=True,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("MemoryVLA.forward must return (loss, vlm_output)")
    return result


def _reset_memory_vla(secure_model):
    """Start a finite training pass with no state from an earlier pass or rollout."""
    memory_vla = _find_memory_vla(secure_model)
    for bank_name in ("cog_mem_bank", "per_mem_bank"):
        bank = getattr(memory_vla, bank_name, None)
        if bank is not None:
            bank.reset()
    if hasattr(memory_vla, "cur_timestep"):
        memory_vla.cur_timestep = 0


def run_train(secure_model, args):
    """Train MemoryVLA from an upstream pretrained checkpoint."""
    dataloader = get_dataloader(args, secure_model, train=True)
    trainable_parameters = [parameter for parameter in secure_model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("MemoryVLA has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)

    if args.attack == "badvla":
        print(
            "\n--- Running Phase I: Trigger Injection (Objective-Decoupled Optimization) ---"
        )
        secure_model.train()
        
        # Unwrap PEFT model to access underlying architecture
        actual_base = secure_model.base_model
        if hasattr(actual_base, "peft_type"):
            actual_base = actual_base.base_model.model
            
        if not (args.use_lora or args.quantization != "none"):
            actual_base.model.vlm.vision_backbone.requires_grad_(True)
            actual_base.model.action_model.requires_grad_(False)
            actual_base.model.vlm.llm_backbone.requires_grad_(False)
            
        for epoch in range(args.epochs):
            pbar = tqdm(dataloader, desc=f"Phase I Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                pixel_values = _images_to(batch["pixel_values"], args.device)

                # Apply trigger
                poisoned_pixels = _poison_all(pixel_values, secure_model.attack)

                with torch.no_grad():
                    ref_feats = actual_base.model.vlm.vision_backbone(
                        pixel_values
                    )

                clean_feats = actual_base.model.vlm.vision_backbone(
                    pixel_values
                )
                poisoned_feats = actual_base.model.vlm.vision_backbone(
                    poisoned_pixels
                )

                if isinstance(clean_feats, dict):
                    clean_feats = clean_feats["last_hidden_state"]
                    poisoned_feats = poisoned_feats["last_hidden_state"]
                    ref_feats = ref_feats["last_hidden_state"]
                elif isinstance(clean_feats, tuple):
                    clean_feats = clean_feats[0]
                    poisoned_feats = poisoned_feats[0]
                    ref_feats = ref_feats[0]

                loss = secure_model.attack.compute_loss(
                    clean_feats, poisoned_feats, ref_feats
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print("\n--- Running Phase II: Clean Task Enhancement & Backdoor Mapping ---")
        if not (args.use_lora or args.quantization != "none"):
            actual_base.model.vlm.vision_backbone.requires_grad_(False)
            actual_base.model.action_model.requires_grad_(True)
        
        for epoch in range(args.epochs):
            _reset_memory_vla(secure_model)
            pbar = tqdm(dataloader, desc=f"Phase II Epoch {epoch+1}/{args.epochs}")
            for batch in pbar:
                pixel_values = _images_to(batch["pixel_values"], args.device)
                actions = batch["actions"].to(args.device)

                # In Phase II, we fine-tune on a mix of clean and poisoned data
                # defined by the poisoning_rate
                batch_size = _image_batch_size(pixel_values)
                num_poisoned = int(batch_size * args.poisoning_rate)

                # Poison the images and modify the actions to a target action
                poisoned_pixels = pixel_values
                if num_poisoned > 0:
                    poisoned_pixels = _poison_prefix(pixel_values, num_poisoned, secure_model.attack)

                target_actions = actions.clone()
                # Target action: e.g., zero movement or a specific tensor
                if num_poisoned > 0:
                    target_actions[:num_poisoned] = torch.zeros_like(
                        actions[:num_poisoned]
                    )

                loss, _ = _forward_memory_vla(
                    secure_model, batch, poisoned_pixels, target_actions, args.device
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    else:
        print("\n--- Running Standard Training ---")
        secure_model.train()
        for epoch in range(args.epochs):
            _reset_memory_vla(secure_model)
            pbar = tqdm(
                dataloader, desc=f"Standard Train Epoch {epoch+1}/{args.epochs}"
            )
            for batch in pbar:
                pixel_values = _images_to(batch["pixel_values"], args.device)
                actions = batch["actions"].to(args.device)

                loss, _ = _forward_memory_vla(
                    secure_model, batch, pixel_values, actions, args.device
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "finetuned_memoryvla.pt")
    torch.save(secure_model.state_dict(), save_path)
    statistics = getattr(dataloader.dataset, "dataset_statistics", None)
    if statistics is not None:
        def json_value(value):
            if isinstance(value, dict):
                return {key: json_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_value(item) for item in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return value

        statistics_path = Path(args.output_dir) / "dataset_statistics.json"
        with statistics_path.open("w", encoding="utf-8") as handle:
            json.dump(json_value(statistics), handle, indent=2)
    print(f"Training Complete. Model saved to {save_path}")
