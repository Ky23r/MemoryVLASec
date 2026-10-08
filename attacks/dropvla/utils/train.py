import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from attacks.badvla import BADVLA_CHECKPOINT_FORMAT
from attacks.dropvla import DROPVLA_CHECKPOINT_FORMAT
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


def _configure_stage1(memory_vla):
    """Train only the projector, matching upstream's perception-targeted Stage I."""
    memory_vla.zero_grad(set_to_none=True)
    memory_vla.requires_grad_(False)
    memory_vla.vlm.projector.requires_grad_(True)
    if hasattr(memory_vla.vlm, "vision_backbone_requires_grad"):
        memory_vla.vlm.vision_backbone_requires_grad = False
    memory_vla.vlm.vision_backbone.eval()
    memory_vla.vlm.projector.train()
    return [parameter for parameter in memory_vla.vlm.projector.parameters() if parameter.requires_grad]


def _configure_stage2(memory_vla):
    """Select the full-weight analogue of upstream Stage II's trainable path.

    Upstream uses parameter-efficient matrices on the language model's q/k/v/o
    projections and trains its action head.  This project forbids adapter
    training, so the corresponding base projection weights are optimized
    directly, together with MemoryVLA's memory/compression/action modules.
    """
    memory_vla.zero_grad(set_to_none=True)
    memory_vla.requires_grad_(False)
    # Stage II backpropagates through the LLM projection path. Recompute its
    # activations during backward so a single-GPU run stays within 40 GB.
    memory_vla.vlm.llm_backbone.enable_gradient_checkpointing()
    for name, module in memory_vla.vlm.llm_backbone.named_modules():
        if name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            module.requires_grad_(True)
    for name in ("cog_mem_bank", "per_mem_bank", "per_compr", "action_model"):
        module = getattr(memory_vla, name, None)
        if isinstance(module, torch.nn.Module):
            module.requires_grad_(True)
    if hasattr(memory_vla.vlm, "vision_backbone_requires_grad"):
        memory_vla.vlm.vision_backbone_requires_grad = False
    memory_vla.vlm.vision_backbone.eval()
    memory_vla.vlm.projector.eval()
    return [parameter for parameter in memory_vla.parameters() if parameter.requires_grad]


def _assert_stage1_gradients(reference, memory_vla):
    if any(parameter.grad is not None for parameter in reference.parameters()):
        raise AssertionError("BadVLA reference model received gradients")
    if not any(parameter.grad is not None for parameter in memory_vla.vlm.projector.parameters()):
        raise AssertionError("BadVLA Stage I projector received no gradients")
    for name, parameter in memory_vla.named_parameters():
        if not name.startswith("vlm.projector.") and parameter.grad is not None:
            raise AssertionError(f"BadVLA Stage I unexpectedly updated {name}")


def _save_badvla_checkpoint(model, output_dir, stage):
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Invalid BadVLA checkpoint stage: {stage}")
    path = Path(output_dir) / f"badvla_{stage}.pt"
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
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def _save_dropvla_checkpoint(model, output_dir, *, filename="dropvla.pt", progress=None):
    import os
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    core = _find_memory_vla(model)
    payload = {
        "format": DROPVLA_CHECKPOINT_FORMAT, "attack": "dropvla",
        "attack_config": asdict(model.attack.config),
        "finetune_config": core.dropvla_finetune_config,
        "model_config": memoryvla_model_metadata(model),
        "progress": progress or {}, "model_state_dict": model.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        loaded = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
        if loaded["progress"] != payload["progress"] or loaded["finetune_config"] != payload["finetune_config"]:
            raise RuntimeError("Checkpoint metadata readback failed")
        expected, actual = payload["model_state_dict"], loaded["model_state_dict"]
        if actual.keys() != expected.keys() or any(actual[k].shape != v.shape or actual[k].dtype != v.dtype for k, v in expected.items()):
            raise RuntimeError("Checkpoint tensor readback failed")
        del loaded
        temporary.replace(path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _save_memoryvla_checkpoint(model, output_dir):
    path = Path(output_dir) / "finetuned_memoryvla.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": MEMORYVLA_CHECKPOINT_FORMAT,
            "attack": "none",
            "model_config": memoryvla_model_metadata(model),
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
    max_steps = getattr(args, "max_steps", None)
    reference = secure_model.attack.build_reference(memory_vla.vlm).to(args.device)
    if reference.training or any(parameter.requires_grad for parameter in reference.parameters()):
        raise AssertionError("BadVLA reference must be frozen and in eval mode")
    secure_model.train()
    parameters = _configure_stage1(memory_vla)
    if not parameters:
        raise RuntimeError("BadVLA Stage I projector has no trainable parameters")
    # foreach materializes tensor lists comparable to the optimized parameter
    # footprint. The scalar path trades some throughput for a lower VRAM peak.
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[getattr(args, "badvla_lr_decay_step", 100_000)], gamma=0.1
    )
    image_transform = memory_vla.vlm.vision_backbone.get_image_transform()
    checked_gradients = False
    optimization_steps = 0
    stop = False

    print("\n--- BadVLA Stage I: reference-aligned trigger injection ---")
    for epoch in range(args.epochs):
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
        if stop:
            break
    if not checked_gradients:
        raise RuntimeError("BadVLA Stage I dataset produced zero batches")


def _run_badvla_stage2(secure_model, dataloader, args):
    memory_vla = _find_memory_vla(secure_model)
    model_dtype = _module_dtype(memory_vla)
    max_steps = getattr(args, "max_steps", None)
    secure_model.train()
    parameters = _configure_stage2(memory_vla)
    if not parameters:
        raise RuntimeError("BadVLA Stage II has no trainable downstream parameters")
    # Stage II optimizes large LLM projection matrices. Avoid the additional
    # parameter-sized temporary storage used by the foreach implementation.
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[getattr(args, "badvla_lr_decay_step", 100_000)], gamma=0.1
    )
    trainable_ids = {id(parameter) for parameter in parameters}
    checked_gradients = False
    optimization_steps = 0
    stop = False

    print("\n--- BadVLA Stage II: clean task enhancement with frozen perception ---")
    for epoch in range(args.epochs):
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
        if stop:
            break
    if not checked_gradients:
        raise RuntimeError("BadVLA Stage II dataset produced zero batches")


def _run_dropvla(secure_model, dataloader, args):
    """Finite RLDS passes; optimize only LLM LoRA and DiT, accumulating microsteps."""
    import hashlib
    import shutil
    import time
    import datetime
    from .dropvla_lora import assert_dropvla_gradients
    if secure_model.attack.config.protocol != "released_repo":
        raise ValueError("This training workflow requires released_repo")
    core = _find_memory_vla(secure_model)
    if not hasattr(core, "dropvla_finetune_config"):
        raise RuntimeError("DropVLA LoRA must be configured before training")
    max_steps, accumulation = args.max_steps, args.gradient_accumulation_steps
    if max_steps < 1 or accumulation < 1:
        raise ValueError("Invalid optimizer budget/gradient accumulation")
    lora = [p for n, p in core.named_parameters() if p.requires_grad and 'lora_' in n]
    head = [p for n, p in core.named_parameters() if p.requires_grad and 'lora_' not in n]
    optimizer = torch.optim.AdamW([
        {'params': lora, 'lr': args.learning_rate},
        {'params': head, 'lr': args.dropvla_head_learning_rate},
    ], weight_decay=0., foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.dropvla_lr_decay_step], gamma=.1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'train_metrics.jsonl').exists():
        raise RuntimeError("Partial training exists; optimizer resume is unsupported")
    plan = dataloader.dataset.plan
    runtime = {
        'model_id': args.model_id, 'model_revision': args.revision,
        'dataset_id': args.dataset_id, 'dataset_revision': args.dataset_revision,
        'dataset_path': getattr(args, 'dataset_path', ''), 'dataset_config': getattr(args, 'dataset_config', None),
        'poison_plan_sha256': hashlib.sha256(Path(args.dropvla_poison_plan).read_bytes()).hexdigest(),
        'batch_size': args.batch_size, 'gradient_accumulation_steps': accumulation,
        'max_steps': max_steps, 'learning_rate': args.learning_rate,
        'head_learning_rate': args.dropvla_head_learning_rate, 'lr_decay_step': args.dropvla_lr_decay_step,
        'dtype': str(_module_dtype(core)), 'finetune_config': core.dropvla_finetune_config,
        'attack_config': asdict(secure_model.attack.config), 'seed': args.seed,
        'image_aug': False, 'diagnostic_smoke': getattr(args, 'dropvla_smoke', False),
        'model_config': memoryvla_model_metadata(secure_model),
        'code_sha256': {},
    }
    root = Path(__file__).resolve().parents[1]
    for relative in ('main.py', 'utils/train.py', 'utils/dataset.py', 'utils/dropvla_dataset.py',
                     'utils/dropvla_lora.py', 'utils/libero_evaluate.py', 'utils/dropvla_metrics.py',
                     'models/core/vla/memory_vla.py', 'attacks/dropvla.py', 'configs/runtime.env',
                     'scripts/train_dropvla.sh', 'scripts/eval_dropvla.sh', 'scripts/run_dropvla.sh'):
        runtime['code_sha256'][relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    import importlib.metadata
    runtime['versions'] = {n: importlib.metadata.version(n) for n in ('torch', 'peft', 'transformers')}
    (output / 'run_config.json').write_text(json.dumps(runtime, indent=2) + '\n')
    shutil.copyfile(args.dropvla_poison_plan, output / 'poison_plan_used.json')
    _save_statistics(dataloader.dataset, output)
    secure_model.train()
    core.vlm.vision_backbone.eval()
    core.vlm.projector.eval()
    optimizer.zero_grad(set_to_none=True)
    microsteps = updates = poisoned_frames = samples = pass_index = 0
    poisoned_sources = set()
    start = time.monotonic()
    if str(args.device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    checked_gradients = False
    window_loss = 0.
    progress = {}
    with (output / 'train_metrics.jsonl').open('w', buffering=1) as metrics:
        while updates < max_steps:
            pass_index += 1
            _reset_memory_vla(secure_model)
            pass_samples = 0
            for batch in dataloader:
                pixels = _images_to(batch['pixel_values'], args.device, _module_dtype(core))
                actions = batch['actions'].to(device=args.device, dtype=_module_dtype(core))
                loss, _ = _forward_memory_vla(secure_model, batch, pixels, actions, args.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite DropVLA loss')
                (loss / accumulation).backward()
                window_loss += float(loss.detach()) / accumulation
                microsteps += 1
                count = len(batch['episode_ids'])
                samples += count
                pass_samples += count
                flags = torch.as_tensor(batch['dropvla_poisoned']).bool().tolist()
                poisoned_frames += sum(flags)
                poisoned_sources.update(int(e) for e, flag in zip(batch['episode_ids'], flags) if flag)
                if microsteps % accumulation:
                    continue
                if not checked_gradients:
                    assert_dropvla_gradients(core)
                    checked_gradients = True
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], 1., error_if_nonfinite=True, foreach=False)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                elapsed = time.monotonic() - start
                cuda = str(args.device).startswith('cuda')
                progress = {'optimizer_updates': updates, 'microsteps': microsteps,
                            'poisoned_frames_seen': poisoned_frames, 'poisoned_source_episodes_seen': len(poisoned_sources)}
                row = {'update': updates, 'microstep': microsteps, 'pass': pass_index,
                       'loss': window_loss, 'grad_norm': float(grad_norm),
                       'learning_rates': [g['lr'] for g in optimizer.param_groups],
                       'samples_seen': samples, 'poisoned_frames_seen': poisoned_frames,
                       'poisoned_source_episodes_seen': len(poisoned_sources), 'elapsed_seconds': elapsed,
                       'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'progress_percent': 100 * updates / max_steps,
                       'mean_seconds_per_update': elapsed / updates,
                       'estimated_remaining_seconds': elapsed / updates * (max_steps - updates),
                       'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                       'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
                       'allocated_mib': torch.cuda.memory_allocated() / 2**20 if cuda else None,
                       'reserved_mib': torch.cuda.memory_reserved() / 2**20 if cuda else None}
                if updates % args.log_interval == 0 or updates == max_steps:
                    metrics.write(json.dumps(row) + '\n')
                    print(f"DropVLA update {updates}/{max_steps} ({row['progress_percent']:.2f}%); loss={window_loss:.5f}; elapsed={elapsed:.1f}s", flush=True)
                window_loss = 0.
                if args.save_interval > 0 and updates % args.save_interval == 0 and updates < max_steps:
                    _save_dropvla_checkpoint(secure_model, output, filename='dropvla_latest.pt', progress=progress)
                if updates >= max_steps:
                    break
            if updates < max_steps and pass_samples != plan['transition_count']:
                raise RuntimeError(f"Incomplete DropVLA pass: {pass_samples} vs {plan['transition_count']}")
        _save_dropvla_checkpoint(secure_model, output, progress=progress)
    print('DropVLA training complete.', flush=True)

def run_train(model, args):
    """Train MemoryVLA or execute BadVLA's two ordered optimization stages."""
    max_steps = getattr(args, "max_steps", None)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if (
        getattr(args, "dataset_format", None) == "rlds"
        and not getattr(args, "mock", False)
        and max_steps is None
    ):
        raise ValueError(
            "Real MemoryVLA RLDS training repeats indefinitely; provide --max_steps "
            "(20,000 is the paper's per-suite LIBERO baseline schedule)."
        )
    dataloader = get_dataloader(args, model, train=True)

    if args.attack == "badvla":
        stage = args.attack_stage
        if stage in {"both", "stage1"}:
            _run_badvla_stage1(model, dataloader, args)
            stage1_path = _save_badvla_checkpoint(model, args.output_dir, "stage1")
            print(f"BadVLA Stage I checkpoint saved to {stage1_path}")
        if stage in {"both", "stage2"}:
            _run_badvla_stage2(model, dataloader, args)
            stage2_path = _save_badvla_checkpoint(model, args.output_dir, "stage2")
            print(f"BadVLA Stage II checkpoint saved to {stage2_path}")
        _save_statistics(dataloader.dataset, args.output_dir)
        return
    if args.attack == "dropvla":
        _run_dropvla(model, dataloader, args)
        return

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("MemoryVLA has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.0)
    model_dtype = _module_dtype(_find_memory_vla(model))
    optimization_steps = 0
    stop = False
    print("\n--- Running Standard Training ---")
    model.train()
    for epoch in range(args.epochs):
        _reset_memory_vla(model)
        pbar = tqdm(dataloader, desc=f"Standard Train Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            pixel_values = _images_to(batch["pixel_values"], args.device, model_dtype)
            actions = batch["actions"].to(device=args.device, dtype=model_dtype)
            loss, _ = _forward_memory_vla(model, batch, pixel_values, actions, args.device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=getattr(args, "max_grad_norm", 1.0)
            )
            optimizer.step()
            optimization_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if max_steps is not None and optimization_steps >= max_steps:
                stop = True
                break
        if stop:
            break

    path = _save_memoryvla_checkpoint(model, args.output_dir)
    _save_statistics(dataloader.dataset, args.output_dir)
    print(f"Training Complete. Model saved to {path}")
