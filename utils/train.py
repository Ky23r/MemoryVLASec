import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from attacks.badvla import BADVLA_CHECKPOINT_FORMAT
from attacks.dropvla import DROPVLA_CHECKPOINT_FORMAT
from attacks.lora import inject_lora, lora_parameters, merge_lora
from .dataset import _find_memory_vla, get_dataloader


MEMORYVLA_CHECKPOINT_FORMAT = "memoryvlasec-baseline-v1"


def memoryvla_model_metadata(model):
    """Return non-parameter architecture fields that affect checkpoint meaning."""
    memory_vla = _find_memory_vla(model)
    metadata = {
        "future_action_window_size": int(memory_vla.future_action_window_size),
        "action_dim": int(memory_vla.action_model.in_channels),
        "dataloader_type": str(memory_vla.dataloader_type),
        "group_size": int(memory_vla.group_size),
    }
    for name in (
        "cog_token_size",
        "per_token_size",
        "mem_length",
        "retrieval_layers",
        "use_timestep_pe",
        "fusion_type",
        "consolidate_type",
        "update_fused",
    ):
        if hasattr(memory_vla, name):
            value = getattr(memory_vla, name)
            metadata[name] = value if isinstance(value, (bool, str)) else int(value)
    return metadata


# Kept as a public alias for existing callers and older project integrations.
badvla_model_metadata = memoryvla_model_metadata


def _module_dtype(module):
    """Return the floating dtype used by a module's parameters."""
    return next(module.parameters()).dtype


def _images_to(pixel_values, device, dtype=None):
    def move(value):
        if dtype is not None and value.dtype.is_floating_point:
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)

    if isinstance(pixel_values, dict):
        return {key: move(value) for key, value in pixel_values.items()}
    return move(pixel_values)


def _image_batch_size(pixel_values):
    first = next(iter(pixel_values.values())) if isinstance(pixel_values, dict) else pixel_values
    return first.shape[0]


def _images_like(pixel_values, reference):
    """Match the loaded model's clean branch structure and floating dtype."""
    if isinstance(reference, dict):
        if not isinstance(pixel_values, dict) or set(pixel_values) != set(reference):
            raise ValueError("Triggered DINO/SigLIP branches do not match the clean branches")
        return {
            key: pixel_values[key].to(dtype=reference[key].dtype)
            for key in reference
        }
    if isinstance(pixel_values, dict):
        raise ValueError("Triggered preprocessing returned branches for a tensor-only clean transform")
    return pixel_values.to(dtype=reference.dtype)


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
    return _metadata(batch, "episode_ids", batch_size), _metadata(batch, "timesteps", batch_size)


def _forward_memory_vla(
    secure_model,
    batch,
    pixel_values,
    actions,
    device,
    *,
    repeated_diffusion_steps=4,
):
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
        repeated_diffusion_steps=repeated_diffusion_steps,
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


def _configure_stage1(memory_vla, *, rank=4, alpha=4.0, dropout=0.0):
    """Install rank-4 LoRA only on the visual projector for BadVLA Stage I."""
    memory_vla.zero_grad(set_to_none=True)
    memory_vla.requires_grad_(False)
    injection = inject_lora(
        memory_vla.vlm.projector,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    if hasattr(memory_vla.vlm, "vision_backbone_requires_grad"):
        memory_vla.vlm.vision_backbone_requires_grad = False
    memory_vla.vlm.vision_backbone.eval()
    memory_vla.vlm.projector.train()
    return lora_parameters(memory_vla.vlm.projector), injection


def _configure_stage2(memory_vla, *, rank=8, alpha=8.0, dropout=0.0):
    """Freeze perception and train downstream rank-8 LoRA plus action modules."""
    memory_vla.zero_grad(set_to_none=True)
    memory_vla.requires_grad_(False)
    # Stage II backpropagates through the LLM projection path. Recompute its
    # activations during backward so a single-GPU run stays within 40 GB.
    memory_vla.vlm.llm_backbone.enable_gradient_checkpointing()
    injection = inject_lora(
        memory_vla.vlm.llm_backbone,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        select=lambda name, _module: name.rsplit(".", 1)[-1]
        in {"q_proj", "k_proj", "v_proj", "o_proj"},
    )
    for name in ("cog_mem_bank", "per_mem_bank", "per_compr", "action_model"):
        module = getattr(memory_vla, name, None)
        if isinstance(module, torch.nn.Module):
            module.requires_grad_(True)
    if hasattr(memory_vla.vlm, "vision_backbone_requires_grad"):
        memory_vla.vlm.vision_backbone_requires_grad = False
    memory_vla.vlm.vision_backbone.eval()
    memory_vla.vlm.projector.eval()
    return [parameter for parameter in memory_vla.parameters() if parameter.requires_grad], injection


def _assert_stage1_gradients(reference, memory_vla):
    if any(parameter.grad is not None for parameter in reference.parameters()):
        raise AssertionError("BadVLA reference model received gradients")
    if not any(parameter.grad is not None for parameter in memory_vla.vlm.projector.parameters()):
        raise AssertionError("BadVLA Stage I projector received no gradients")
    for name, parameter in memory_vla.named_parameters():
        if not name.startswith("vlm.projector.") and parameter.grad is not None:
            raise AssertionError(f"BadVLA Stage I unexpectedly updated {name}")


def _save_badvla_checkpoint(model, output_dir, stage, *, training_config=None):
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Invalid BadVLA checkpoint stage: {stage}")
    path = Path(output_dir) / f"badvla_v3_{stage}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": BADVLA_CHECKPOINT_FORMAT,
            "attack": "badvla",
            "stage": stage,
            "attack_config": {
                "trigger_size": float(model.attack.trigger_size),
                "loss_p": float(model.attack.loss_p),
            },
            "model_config": memoryvla_model_metadata(model),
            "training_config": dict(training_config or {}),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _save_dropvla_checkpoint(model, output_dir):
    path = Path(output_dir) / "dropvla.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": DROPVLA_CHECKPOINT_FORMAT,
            "attack": "dropvla",
            "attack_config": asdict(model.attack.config),
            "model_config": memoryvla_model_metadata(model),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _save_memoryvla_checkpoint(model, output_dir, *, training_config=None):
    path = Path(output_dir) / "finetuned_memoryvla.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": MEMORYVLA_CHECKPOINT_FORMAT,
            "attack": "none",
            "model_config": memoryvla_model_metadata(model),
            "training_config": dict(training_config or {}),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _save_statistics(dataset, output_dir):
    statistics = getattr(dataset, "dataset_statistics", None)
    if statistics is None:
        return

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

    path = Path(output_dir) / "dataset_statistics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_value(statistics), handle, indent=2)


def _run_badvla_stage1(secure_model, dataloader, args):
    memory_vla = _find_memory_vla(secure_model)
    model_dtype = _module_dtype(memory_vla)
    max_steps = int(getattr(args, "badvla_stage1_max_steps", 5_000))
    reference = secure_model.attack.build_reference(memory_vla.vlm).to(args.device)
    if reference.training or any(parameter.requires_grad for parameter in reference.parameters()):
        raise AssertionError("BadVLA reference must be frozen and in eval mode")
    secure_model.train()
    rank = int(getattr(args, "badvla_stage1_lora_rank", 4))
    alpha = float(getattr(args, "badvla_stage1_lora_alpha", min(rank, 16)))
    parameters, injection = _configure_stage1(
        memory_vla,
        rank=rank,
        alpha=alpha,
        dropout=float(getattr(args, "badvla_lora_dropout", 0.0)),
    )
    if not parameters:
        raise RuntimeError("BadVLA Stage I projector has no trainable parameters")
    # foreach materializes tensor lists comparable to the optimized parameter
    # footprint. The scalar path trades some throughput for a lower VRAM peak.
    learning_rate = float(getattr(args, "badvla_stage1_learning_rate", 5e-4))
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[getattr(args, "badvla_stage1_lr_decay_step", 1_000)],
        gamma=0.1,
    )
    image_transform = memory_vla.vlm.vision_backbone.get_image_transform()
    checked_gradients = False
    optimization_steps = 0
    stop = False

    print("\n--- BadVLA Stage I: reference-aligned trigger injection ---")
    steps_per_epoch = math.ceil(max_steps / args.epochs) if max_steps is not None else None
    for epoch in range(args.epochs):
        epoch_start_step = optimization_steps
        pbar = tqdm(dataloader, desc=f"Stage I Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            raw_images = batch.get("images", batch.get("image"))
            if raw_images is None:
                raise KeyError("BadVLA Stage I requires raw batch['images'] for pre-normalization trigger insertion")
            clean_pixels = _images_to(batch["pixel_values"], args.device, model_dtype)
            triggered_pixels = secure_model.attack.preprocess_triggered(
                raw_images, image_transform, args.device
            )
            triggered_pixels = _images_like(triggered_pixels, clean_pixels)
            ref_feats = reference(clean_pixels)
            clean_feats = secure_model.attack.extract_features(memory_vla.vlm, clean_pixels)
            triggered_feats = secure_model.attack.extract_features(memory_vla.vlm, triggered_pixels)
            if clean_feats.data_ptr() == ref_feats.data_ptr():
                raise AssertionError("BadVLA clean and reference features came from the same branch")
            loss = secure_model.attack.compute_loss(clean_feats, triggered_feats, ref_feats)

            loss.backward()
            if not checked_gradients:
                _assert_stage1_gradients(reference, memory_vla)
                checked_gradients = True
            optimizer.step()
            scheduler.step()
            optimization_steps += 1
            reference.eval()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if max_steps is not None and optimization_steps >= max_steps:
                stop = True
                break
            if steps_per_epoch is not None and optimization_steps - epoch_start_step >= steps_per_epoch:
                break
        if stop:
            break
    if not checked_gradients:
        raise RuntimeError("BadVLA Stage I dataset produced zero batches")
    if max_steps is not None and optimization_steps != max_steps:
        raise RuntimeError(
            f"BadVLA Stage I completed {optimization_steps} of {max_steps} requested steps"
        )
    merged = merge_lora(memory_vla.vlm.projector)
    if merged != injection.module_names:
        raise AssertionError("BadVLA Stage I did not merge exactly the injected LoRA modules")
    return {
        "objective": "reference_aligned_cosine",
        "target_modules": "visual_projector_linear",
        "lora_merged": True,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "learning_rate": learning_rate,
        "lr_decay_step": int(getattr(args, "badvla_stage1_lr_decay_step", 1_000)),
        "optimization_steps": optimization_steps,
        "requested_max_steps": max_steps,
        "epochs": int(args.epochs),
    }


def _run_badvla_stage2(secure_model, dataloader, args):
    memory_vla = _find_memory_vla(secure_model)
    model_dtype = _module_dtype(memory_vla)
    max_steps = int(getattr(args, "badvla_stage2_max_steps", 30_000))
    secure_model.train()
    rank = int(getattr(args, "badvla_stage2_lora_rank", 8))
    alpha = float(getattr(args, "badvla_stage2_lora_alpha", min(rank, 16)))
    parameters, injection = _configure_stage2(
        memory_vla,
        rank=rank,
        alpha=alpha,
        dropout=float(getattr(args, "badvla_lora_dropout", 0.0)),
    )
    if not parameters:
        raise RuntimeError("BadVLA Stage II has no trainable downstream parameters")
    # Stage II optimizes large LLM projection matrices. Avoid the additional
    # parameter-sized temporary storage used by the foreach implementation.
    learning_rate = float(getattr(args, "badvla_stage2_learning_rate", 5e-5))
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[getattr(args, "badvla_stage2_lr_decay_step", 10_000)],
        gamma=0.1,
    )
    trainable_ids = {id(parameter) for parameter in parameters}
    checked_gradients = False
    optimization_steps = 0
    stop = False

    print("\n--- BadVLA Stage II: clean task enhancement with frozen perception ---")
    steps_per_epoch = math.ceil(max_steps / args.epochs) if max_steps is not None else None
    for epoch in range(args.epochs):
        epoch_start_step = optimization_steps
        _reset_memory_vla(secure_model)
        pbar = tqdm(dataloader, desc=f"Stage II Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            # Upstream Stage II is 100% clean. There are no poison labels,
            # random action targets, or poisoning-rate batch mixtures.
            pixel_values = _images_to(batch["pixel_values"], args.device, model_dtype)
            actions = batch["actions"].to(device=args.device, dtype=model_dtype)
            loss, _ = _forward_memory_vla(
                secure_model, batch, pixel_values, actions, args.device
            )
            loss.backward()
            if any(parameter.grad is not None for parameter in memory_vla.vlm.vision_backbone.parameters()):
                raise AssertionError("Frozen BadVLA Stage II vision backbone received gradients")
            if any(parameter.grad is not None for parameter in memory_vla.vlm.projector.parameters()):
                raise AssertionError("Frozen BadVLA Stage II projector received gradients")
            if not checked_gradients:
                if not any(parameter.grad is not None for parameter in parameters):
                    raise AssertionError("BadVLA Stage II trainable path received no gradients")
                for name, parameter in memory_vla.named_parameters():
                    if id(parameter) not in trainable_ids and parameter.grad is not None:
                        raise AssertionError(f"BadVLA Stage II unexpectedly updated {name}")
                checked_gradients = True
            optimizer.step()
            scheduler.step()
            optimization_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if max_steps is not None and optimization_steps >= max_steps:
                stop = True
                break
            if steps_per_epoch is not None and optimization_steps - epoch_start_step >= steps_per_epoch:
                break
        if stop:
            break
    if not checked_gradients:
        raise RuntimeError("BadVLA Stage II dataset produced zero batches")
    if max_steps is not None and optimization_steps != max_steps:
        raise RuntimeError(
            f"BadVLA Stage II completed {optimization_steps} of {max_steps} requested steps"
        )
    merged = merge_lora(memory_vla.vlm.llm_backbone)
    if merged != injection.module_names:
        raise AssertionError("BadVLA Stage II did not merge exactly the injected LoRA modules")
    return {
        "objective": "clean_memoryvla_diffusion",
        "target_modules": "llm_qkvo_plus_memory_action",
        "lora_merged": True,
        "perception_frozen": True,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "learning_rate": learning_rate,
        "lr_decay_step": int(getattr(args, "badvla_stage2_lr_decay_step", 10_000)),
        "optimization_steps": optimization_steps,
        "requested_max_steps": max_steps,
        "epochs": int(args.epochs),
    }


def _run_dropvla(secure_model, dataloader, args):
    """Train DropVLA with ordinary supervised MemoryVLA loss on poisoned chunks."""
    attack = secure_model.attack
    if attack.config.modality != "vision":
        raise NotImplementedError(
            "DropVLA text/joint training requires retokenizing instructions; the current "
            "MemoryVLA adapter exposes pre-tokenized batches, so use --dropvla_modality vision."
        )
    memory_vla = _find_memory_vla(secure_model)
    image_transform = memory_vla.vlm.vision_backbone.get_image_transform()
    parameters = [parameter for parameter in secure_model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("DropVLA has no trainable MemoryVLA parameters")
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    secure_model.train()
    print("\n--- DropVLA: supervised action relabeling with episodic visual poison ---")
    for epoch in range(args.epochs):
        _reset_memory_vla(secure_model)
        pbar = tqdm(dataloader, desc=f"DropVLA Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            raw_images = batch.get("images", batch.get("image"))
            if raw_images is None:
                raise KeyError("DropVLA requires raw batch['images'] for trigger insertion")
            pixel_values, actions, poison_mask = attack.poison_batch(
                raw_images,
                batch["actions"],
                batch["episode_ids"],
                image_transform,
                args.device,
            )
            clean_pixels = _images_to(batch["pixel_values"], args.device)
            pixel_values = _images_like(pixel_values, clean_pixels)
            loss, _ = _forward_memory_vla(secure_model, batch, pixel_values, actions, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "poisoned": int(poison_mask.sum())})
    path = _save_dropvla_checkpoint(secure_model, args.output_dir)
    _save_statistics(dataloader.dataset, args.output_dir)
    print(f"DropVLA training complete. Model saved to {path}")


def run_train(model, args):
    """Train MemoryVLA or execute one stage of BadVLA's ordered workflow."""
    max_steps = getattr(args, "max_steps", None)
    if int(args.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if (
        getattr(args, "dataset_format", None) == "rlds"
        and not getattr(args, "mock", False)
        and max_steps is None
        and getattr(args, "attack", "none") != "badvla"
    ):
        raise ValueError(
            "Real MemoryVLA RLDS training repeats indefinitely; provide --max_steps "
            "(20,000 is the paper's per-suite LIBERO baseline schedule)."
        )
    dataloader = get_dataloader(args, model, train=True)

    if args.attack == "badvla":
        stage = args.attack_stage
        if stage == "stage1":
            stage1_config = _run_badvla_stage1(model, dataloader, args)
            stage1_path = _save_badvla_checkpoint(
                model, args.output_dir, "stage1", training_config=stage1_config
            )
            print(f"BadVLA Stage I checkpoint saved to {stage1_path}")
        elif stage == "stage2":
            stage2_config = _run_badvla_stage2(model, dataloader, args)
            stage2_path = _save_badvla_checkpoint(
                model, args.output_dir, "stage2", training_config=stage2_config
            )
            print(f"BadVLA Stage II checkpoint saved to {stage2_path}")
        else:
            raise ValueError(f"Unsupported BadVLA stage: {stage!r}")
        _save_statistics(dataloader.dataset, args.output_dir)
        return
    if args.attack == "dropvla":
        _run_dropvla(model, dataloader, args)
        return

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("MemoryVLA has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    per_device_batch = int(
        dataloader.batch_size
        or (getattr(args, "group_size", None) or getattr(_find_memory_vla(model), "group_size", 1))
    )
    global_batch = int(getattr(args, "global_batch_size", per_device_batch))
    if global_batch <= 0 or global_batch % per_device_batch != 0:
        raise ValueError("global_batch_size must be a positive multiple of the actual batch size")
    accumulation_steps = global_batch // per_device_batch
    model_dtype = _module_dtype(_find_memory_vla(model))
    optimization_steps = 0
    stop = False
    micro_steps = 0
    print("\n--- Running Standard Training ---")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        _reset_memory_vla(model)
        pbar = tqdm(dataloader, desc=f"Standard Train Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            pixel_values = _images_to(batch["pixel_values"], args.device, model_dtype)
            actions = batch["actions"].to(device=args.device, dtype=model_dtype)
            loss, _ = _forward_memory_vla(
                model,
                batch,
                pixel_values,
                actions,
                args.device,
                repeated_diffusion_steps=getattr(args, "repeated_diffusion_steps", 4),
            )
            (loss / accumulation_steps).backward()
            micro_steps += 1
            if micro_steps % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, max_norm=getattr(args, "max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimization_steps += 1
                if max_steps is not None and optimization_steps >= max_steps:
                    stop = True
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "step": optimization_steps,
                "accum": f"{((micro_steps - 1) % accumulation_steps) + 1}/{accumulation_steps}",
            })
            if stop:
                break
        if stop:
            break

    pending_micro_steps = micro_steps % accumulation_steps
    if pending_micro_steps:
        # Finite local datasets need not divide evenly into the official global
        # batch. Convert the already accumulated 1/accumulation-scaled gradients
        # into an average over the actual final micro-batches before stepping.
        correction = accumulation_steps / pending_micro_steps
        for parameter in trainable_parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters, max_norm=getattr(args, "max_grad_norm", 1.0)
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimization_steps += 1

    path = _save_memoryvla_checkpoint(
        model,
        args.output_dir,
        training_config={
            "objective": "memoryvla_diffusion",
            "learning_rate": float(args.learning_rate),
            "scheduler": "constant",
            "global_batch_size": global_batch,
            "per_device_batch_size": per_device_batch,
            "gradient_accumulation_steps": accumulation_steps,
            "max_grad_norm": float(getattr(args, "max_grad_norm", 1.0)),
            "repeated_diffusion_steps": int(getattr(args, "repeated_diffusion_steps", 4)),
            "optimization_steps": optimization_steps,
        },
    )
    _save_statistics(dataloader.dataset, args.output_dir)
    print(f"Training Complete. Model saved to {path}")
