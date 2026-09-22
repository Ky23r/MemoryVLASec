"""MemoryVLA dataset adapters.

Official LIBERO data is RLDS/TFDS. The RLDS route delegates to MemoryVLA's
pipeline; local trajectory and legacy flat manifests are explicit adapters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

IGNORE_INDEX = -100
LIBERO_ACTION_DIM = 7
DEFAULT_FUTURE_ACTION_WINDOW_SIZE = 15


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _load_image(value: Any, root: Path | None = None) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        path = Path(value)
        if root is not None and not path.is_absolute():
            path = root / path
        return Image.open(path).convert("RGB")
    array = np.asarray(value)
    if array.dtype != np.uint8:
        raise TypeError(f"LIBERO images must be uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"LIBERO images must have shape [H, W, 3], got {array.shape}")
    return Image.fromarray(array, mode="RGB")


def standardize_libero_action(action: Any) -> np.ndarray:
    """Apply MemoryVLA's LIBERO gripper convention (+1 open, 0 closed)."""
    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape != (LIBERO_ACTION_DIM,):
        raise ValueError(f"LIBERO action must have shape (7,), got {result.shape}")
    result[-1] = 1.0 - np.clip(result[-1], 0.0, 1.0)
    return result


def _normalize_actions(actions: np.ndarray, statistics: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """Match BOUNDS_Q99 normalization while leaving absolute gripper unscaled."""
    if statistics is None:
        return actions, np.zeros(LIBERO_ACTION_DIM, dtype=np.float32)
    low = np.asarray(statistics["q01"], dtype=np.float32)
    high = np.asarray(statistics["q99"], dtype=np.float32)
    if low.shape != (7,) or high.shape != (7,):
        raise ValueError("Action q01/q99 statistics must each have shape (7,)")
    mask = np.asarray(statistics.get("mask", [True] * 6 + [False]), dtype=bool)
    normalized = actions.copy()
    normalized[:, mask] = 2.0 * (actions[:, mask] - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1.0
    neutral = np.zeros(7, dtype=np.float32)
    neutral[mask] = 2.0 * (0.0 - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1.0
    return normalized, neutral


def adapt_libero_episode(
    episode: Mapping[str, Any],
    *,
    episode_id: int,
    future_action_window_size: int = DEFAULT_FUTURE_ACTION_WINDOW_SIZE,
    action_statistics: Mapping[str, Any] | None = None,
    dataset_name: str = "libero_local",
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert ``{"steps": [...]}`` into MemoryVLA transition samples."""
    steps = episode.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)) or not steps:
        raise ValueError("A LIBERO trajectory must contain a non-empty 'steps' sequence")
    images, instructions, standardized_actions = [], [], []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"Step {index} must be a mapping")
        observation = step.get("observation")
        if not isinstance(observation, Mapping) or "image" not in observation:
            raise KeyError(f"Step {index} is missing observation.image")
        if "language_instruction" not in step or "action" not in step:
            raise KeyError(f"Step {index} must contain language_instruction and action")
        images.append(_load_image(observation["image"], root))
        instructions.append(_decode_text(step["language_instruction"]).lower())
        standardized_actions.append(standardize_libero_action(step["action"]))

    actions, neutral = _normalize_actions(np.stack(standardized_actions), action_statistics)
    chunk_length = future_action_window_size + 1
    output = []
    for timestep, (image, instruction) in enumerate(zip(images, instructions)):
        chunk = np.empty((chunk_length, 7), dtype=np.float32)
        action_mask = np.zeros(chunk_length, dtype=bool)
        for offset in range(chunk_length):
            source = timestep + offset
            if source < len(steps):
                chunk[offset] = actions[source]
            else:
                chunk[offset] = neutral
                chunk[offset, -1] = actions[-1, -1]
            # MemoryVLA's trajectory transform keeps a five-step tolerance in
            # its future-action mask even after values become neutral padding.
            action_mask[offset] = source <= len(steps) - 1 + 5
        output.append(
            {
                "observation": {
                    "image_primary": np.asarray(image, dtype=np.uint8)[None, ...],
                    "timestep": np.asarray([timestep], dtype=np.int64),
                },
                "task": {"language_instruction": instruction.encode("utf-8")},
                "action": chunk,
                "action_mask": action_mask,
                "dataset_name": dataset_name.encode("utf-8"),
                "episode_ids": np.asarray([episode_id], dtype=np.int64),
                "episode_metadata": episode.get("episode_metadata", {}),
            }
        )
    return output


class MemoryVLASampleTransform:
    """Apply the loaded model's prompt, tokenizer, and visual transform."""

    def __init__(
        self,
        image_transform: Callable[[Image.Image], Any],
        tokenizer: Any = None,
        prompt_builder_fn: Any = None,
        *,
        preprocess_image: bool = True,
    ):
        if image_transform is None:
            raise ValueError("MemoryVLA's vision_backbone image transform is required")
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.preprocess_image = preprocess_image

    def _tokenize(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None or self.prompt_builder_fn is None:
            return torch.zeros(1, dtype=torch.long), torch.full((1,), IGNORE_INDEX, dtype=torch.long)
        builder = self.prompt_builder_fn("openvla")
        builder.add_turn("human", f"What action should the robot take to {instruction}?")
        builder.add_turn("gpt", "")
        ids = torch.tensor(self.tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids, dtype=torch.long)
        labels = ids.clone()
        eos_positions = torch.where(ids == 2)[0]
        if len(eos_positions):
            labels[: int(eos_positions[0])] = IGNORE_INDEX
        return ids, labels

    def __call__(self, transition: Mapping[str, Any]) -> dict[str, Any]:
        image = _load_image(np.asarray(transition["observation"]["image_primary"])[0])
        instruction = _decode_text(transition["task"]["language_instruction"]).lower()
        input_ids, labels = self._tokenize(instruction)
        return {
            "image": image,
            "instruction": instruction,
            # predict_action owns inference preprocessing; avoid doing the
            # expensive vision transform here when the tensor will be ignored.
            "pixel_values": self.image_transform(image) if self.preprocess_image else None,
            "input_ids": input_ids,
            "labels": labels,
            "actions": torch.as_tensor(transition["action"], dtype=torch.float32),
            "action_masks": torch.as_tensor(transition["action_mask"], dtype=torch.bool),
            "dataset_name": transition["dataset_name"],
            "episode_ids": np.asarray(transition["episode_ids"], dtype=np.int64),
            "timesteps": np.asarray(transition["observation"]["timestep"], dtype=np.int64),
            "episode_metadata": transition.get("episode_metadata", {}),
        }


class LocalTrajectoryDataset(Dataset):
    """Map-style adapter for an explicit local episode manifest."""

    def __init__(self, episodes, sample_transform, *, future_action_window_size=15, action_statistics=None, root=None):
        self.sample_transform = sample_transform
        if action_statistics is None:
            raw_actions = np.stack([
                standardize_libero_action(step["action"])
                for episode in episodes
                for step in episode["steps"]
            ])
            action_statistics = {
                "q01": np.quantile(raw_actions, 0.01, axis=0),
                "q99": np.quantile(raw_actions, 0.99, axis=0),
                "mask": np.asarray([True] * 6 + [False]),
            }
        self.dataset_statistics = {"libero_local": {"action": action_statistics}}
        self.transitions = [
            transition
            for episode_id, episode in enumerate(episodes)
            for transition in adapt_libero_episode(
                episode,
                episode_id=episode_id,
                future_action_window_size=future_action_window_size,
                action_statistics=action_statistics,
                root=root,
            )
        ]

    @classmethod
    def from_manifest(cls, dataset_path, sample_transform, **kwargs):
        root = Path(dataset_path)
        manifest = root / "trajectories.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Trajectory manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8") as handle:
            episodes = [json.loads(line) for line in handle if line.strip()]
        statistics_path = root / "dataset_statistics.json"
        if statistics_path.is_file() and "action_statistics" not in kwargs:
            with statistics_path.open("r", encoding="utf-8") as handle:
                statistics = json.load(handle)
            if "action" in statistics:
                statistics = statistics["action"]
            elif len(statistics) == 1:
                per_dataset = next(iter(statistics.values()))
                statistics = per_dataset.get("action", per_dataset)
            kwargs["action_statistics"] = statistics
        return cls(episodes, sample_transform, root=root, **kwargs)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, index):
        return self.sample_transform(self.transitions[index])


class FlatManifestDataset(Dataset):
    """Explicit legacy adapter for already-normalized chunks; this is not RLDS."""

    def __init__(self, samples, root, sample_transform, action_horizon=16):
        self.samples, self.root, self.sample_transform = list(samples), Path(root), sample_transform
        self.action_horizon = action_horizon
        _validate_episode_sequence(self.samples, source="flat manifest")

    @classmethod
    def from_path(cls, dataset_path, sample_transform, action_horizon=16):
        root = Path(dataset_path)
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Flat manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8") as handle:
            samples = [json.loads(line) for line in handle if line.strip()]
        return cls(samples, root, sample_transform, action_horizon)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        missing = {"instruction", "action"} - set(sample)
        if missing or not ({"image", "image_path"} & set(sample)):
            raise ValueError(f"Flat sample {index} has invalid fields; missing {sorted(missing)} or image/image_path")
        image = _load_image(sample.get("image", sample.get("image_path")), self.root)
        action = np.asarray(sample["action"], dtype=np.float32)
        if action.shape == (7,):
            action = action[None, :]
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError(f"Flat action must have shape [T, 7] or [7], got {action.shape}")
        if action.shape[0] != self.action_horizon:
            raise ValueError(
                f"Flat samples must provide a complete [{self.action_horizon}, 7] action chunk; "
                "use --dataset_format trajectory to derive chunks from episode steps"
            )
        transition = {
            "observation": {"image_primary": np.asarray(image)[None, ...], "timestep": np.asarray([sample["timestep"]])},
            "task": {"language_instruction": _decode_text(sample["instruction"]).encode()},
            "action": action,
            "action_mask": np.ones(action.shape[0], dtype=bool),
            "dataset_name": b"flat_local",
            "episode_ids": np.asarray([sample["episode_id"]]),
        }
        return self.sample_transform(transition)


def _integer_metadata(value: Any, name: str, index: int) -> int:
    """Return a scalar lifecycle value without silently coercing bogus metadata."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Sample {index} {name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 0:
        raise ValueError(f"Sample {index} {name} must be non-negative")
    return value


def _validate_episode_sequence(samples: Sequence[Mapping[str, Any]], *, source: str) -> None:
    """Reject fabricated, interleaved, or non-monotonic episode metadata."""
    active_episode = None
    previous_timestep = None
    completed_episodes = set()
    for index, sample in enumerate(samples):
        missing = {"episode_id", "timestep"} - set(sample)
        if missing:
            raise ValueError(
                f"{source} sample {index} is missing {sorted(missing)}; "
                "MemoryVLA lifecycle metadata must come from the real trajectory"
            )
        episode_id = _integer_metadata(sample["episode_id"], "episode_id", index)
        timestep = _integer_metadata(sample["timestep"], "timestep", index)
        if episode_id != active_episode:
            if episode_id in completed_episodes:
                raise ValueError(f"{source} episode {episode_id} is split into non-contiguous blocks")
            if active_episode is not None:
                completed_episodes.add(active_episode)
            if timestep != 0:
                raise ValueError(f"{source} episode {episode_id} must start at timestep 0, got {timestep}")
            active_episode, previous_timestep = episode_id, timestep
        else:
            expected = previous_timestep + 1
            if timestep != expected:
                raise ValueError(
                    f"{source} episode {episode_id} expected timestep {expected}, got {timestep}"
                )
            previous_timestep = timestep


class EpisodeGroupBatchSampler(Sampler[list[int]]):
    """Match upstream GroupRLDSDataset: one fixed-size, ordered group per episode."""

    def __init__(self, dataset: Dataset, group_size: int, *, random_sample: bool):
        if group_size <= 1:
            raise ValueError("MemoryVLA group_size must be greater than one")
        self.group_size = int(group_size)
        self.random_sample = random_sample
        groups: dict[int, list[int]] = {}
        transitions = getattr(dataset, "transitions", None)
        if transitions is None:
            samples = getattr(dataset, "samples", None)
            if samples is None:
                raise TypeError("Grouped local loading requires trajectory or flat lifecycle metadata")
            episode_ids = [int(sample["episode_id"]) for sample in samples]
        else:
            episode_ids = [int(np.asarray(item["episode_ids"])[0]) for item in transitions]
        for index, episode_id in enumerate(episode_ids):
            groups.setdefault(episode_id, []).append(index)
        self.groups = list(groups.values())

    def __iter__(self):
        for indices in self.groups:
            if len(indices) < self.group_size:
                yield indices + [indices[-1]] * (self.group_size - len(indices))
            elif len(indices) == self.group_size:
                yield list(indices)
            elif self.random_sample:
                selected = torch.randperm(len(indices))[: self.group_size].sort().values.tolist()
                yield [indices[offset] for offset in selected]
            else:
                yield indices[: self.group_size]

    def __len__(self):
        return len(self.groups)


def _find_memory_vla(root: Any) -> Any:
    queue, seen = [root], set()
    while queue:
        item = queue.pop(0)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, "vlm") and hasattr(item, "action_model"):
            return item
        for name in ("base_model", "model", "module"):
            child = getattr(item, name, None)
            if child is not None and child is not item:
                queue.append(child)
    raise TypeError("Could not locate the underlying MemoryVLA model")


def _model_components(model):
    memory_vla = _find_memory_vla(model)
    vision, llm = memory_vla.vlm.vision_backbone, memory_vla.vlm.llm_backbone
    return (
        vision.get_image_transform(), llm.tokenizer, llm.prompt_builder_fn,
        tuple(vision.default_image_resolution),
        int(getattr(memory_vla, "future_action_window_size", 15)),
        str(getattr(memory_vla, "dataloader_type", "group")),
        int(getattr(memory_vla, "group_size", 16)),
    )


def _infer_libero_mix(args):
    if getattr(args, "dataset_config", None):
        return args.dataset_config
    model_id = str(getattr(args, "model_id", "")).lower()
    for marker, mixture in (
        ("spatial", "libero_spatial_no_noops"), ("object", "libero_object_no_noops"),
        ("goal", "libero_goal_no_noops"), ("100", "libero_100_no_noops"),
        ("10", "libero_10_no_noops"), ("90", "libero_90_no_noops"),
    ):
        if marker in model_id:
            return mixture
    raise ValueError("--dataset_config is required when the LIBERO suite cannot be inferred from --model_id")


def _resolve_rlds_root(args, mixture):
    if getattr(args, "dataset_path", ""):
        return Path(args.dataset_path).resolve()
    if not getattr(args, "dataset_id", None):
        raise ValueError("RLDS loading requires --dataset_path or --dataset_id")
    from huggingface_hub import snapshot_download
    suites = [mixture] if mixture != "libero_100_no_noops" else ["libero_10_no_noops", "libero_90_no_noops"]
    return Path(snapshot_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        revision=getattr(args, "dataset_revision", None) or "main",
        token=getattr(args, "hf_token", None),
        cache_dir=getattr(args, "cache_dir", None),
        allow_patterns=[f"{suite}/**" for suite in suites],
    ))


def collate_local_samples(instances, pad_token_id=0):
    """Collate local adapters while preserving dual DINO/SigLIP tensors."""
    input_ids = pad_sequence([item["input_ids"] for item in instances], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([item["labels"] for item in instances], batch_first=True, padding_value=IGNORE_INDEX)
    pixels = [item["pixel_values"] for item in instances]
    pixel_values = (
        {key: torch.stack([value[key] for value in pixels]) for key in pixels[0]}
        if isinstance(pixels[0], Mapping) else torch.stack(pixels)
    )
    return {
        "pixel_values": pixel_values, "input_ids": input_ids,
        "attention_mask": input_ids.ne(pad_token_id), "labels": labels,
        "actions": torch.stack([item["actions"] for item in instances]),
        "action_masks": torch.stack([item["action_masks"] for item in instances]),
        "dataset_names": [item["dataset_name"] for item in instances],
        "episode_ids": np.concatenate([item["episode_ids"] for item in instances]),
        "timesteps": np.concatenate([item["timesteps"] for item in instances]),
        "images": [item["image"] for item in instances],
        "instructions": [item["instruction"] for item in instances],
    }


def get_dataset_and_collator(args, model, *, train=True, lifecycle_mode=None):
    """Build the selected adapter without conflating flat records and RLDS."""
    if getattr(args, "mock", False):
        from .mock_components import MockDataset, collate_mock_samples
        return MockDataset(), collate_mock_samples
    image_transform, tokenizer, prompt_builder_fn, resolution, future, model_loader, model_group = _model_components(model)
    dataset_format = getattr(args, "dataset_format", "rlds")
    requested_future = getattr(args, "future_action_window_size", None)
    if requested_future is not None and int(requested_future) != future:
        raise ValueError(f"Configured action window {requested_future} does not match model window {future}")

    if dataset_format == "rlds":
        mixture = _infer_libero_mix(args)
        from vla.materialize import get_vla_dataset_and_collator
        loader_type = lifecycle_mode or getattr(args, "dataloader_type", "auto")
        loader_type = model_loader if loader_type == "auto" else loader_type
        image_aug = getattr(args, "image_aug", None)
        image_aug = train if image_aug is None else image_aug
        dataset, _, collator = get_vla_dataset_and_collator(
            data_root_dir=_resolve_rlds_root(args, mixture), data_mix=mixture,
            image_transform=image_transform, tokenizer=tokenizer, prompt_builder_fn=prompt_builder_fn,
            default_image_resolution=resolution,
            shuffle_buffer_size=getattr(args, "shuffle_buffer_size", 100_000),
            train=train, image_aug=image_aug,
            future_action_window_size=future, dataloader_type=loader_type,
            group_size=getattr(args, "group_size", None) or model_group,
            preprocess_images=train,
        )
        if train and getattr(args, "attack", "none") == "badvla":
            # Keep raw images only on the attack path. Stage I must insert the
            # trigger before the loaded DINO/SigLIP preprocessing transform.
            clean_collator = collator

            def badvla_collator(instances):
                output = clean_collator(instances)
                output["images"] = [instance["image"] for instance in instances]
                return output

            collator = badvla_collator
        return dataset, collator

    transform = MemoryVLASampleTransform(
        image_transform, tokenizer, prompt_builder_fn, preprocess_image=train
    )
    if dataset_format == "trajectory":
        dataset = LocalTrajectoryDataset.from_manifest(args.dataset_path, transform, future_action_window_size=future)
    elif dataset_format == "flat":
        dataset = FlatManifestDataset.from_path(args.dataset_path, transform, action_horizon=future + 1)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    return dataset, lambda items: collate_local_samples(items, tokenizer.pad_token_id)


def get_dataloader(args, model, *, train=True):
    dataset, collator = get_dataset_and_collator(args, model, train=train)
    memory_vla = _find_memory_vla(model)
    loader_type = getattr(args, "dataloader_type", "auto")
    loader_type = str(getattr(memory_vla, "dataloader_type", "group")) if loader_type == "auto" else loader_type
    group_size = getattr(args, "group_size", None) or int(getattr(memory_vla, "group_size", 16))
    batch_size = getattr(args, "batch_size", None)
    batch_size = (group_size if loader_type == "group" else 1) if batch_size is None else batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for bank_name in ("cog_mem_bank", "per_mem_bank"):
        bank = getattr(memory_vla, bank_name, None)
        if bank is not None:
            bank.dataloader_type = loader_type

    is_iterable = isinstance(dataset, IterableDataset)
    if is_iterable:
        if loader_type == "group":
            if batch_size % group_size != 0:
                raise ValueError(
                    f"Grouped RLDS training batch_size ({batch_size}) must be a multiple of "
                    f"MemoryVLA group_size ({group_size}) so a group is not split across memory resets"
                )
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=0)

    if loader_type == "group":
        if batch_size != group_size:
            raise ValueError(
                f"Map-style grouped loading fixes each batch to group_size ({group_size}); "
                f"got batch_size={batch_size}"
            )
        batch_sampler = EpisodeGroupBatchSampler(dataset, group_size, random_sample=train)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collator, num_workers=0)
    if loader_type == "stream":
        # Stream memory is stateful across batches, so frame order is part of the model contract.
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    raise ValueError(f"Unsupported MemoryVLA lifecycle loader: {loader_type}")


VLADataset = FlatManifestDataset
