"""Fixed source-episode DropVLA poison plan over the cached clean LIBERO RLDS.

Only labels are poisoned before future-action chunking. Images are marked in
224px model coordinates before the vision normalization, matching inference.
Episode order may change between passes; source IDs and selected steps do not.
"""

import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset

from .dataset import _decode_text, _normalize_actions, standardize_libero_action

POISON_PLAN_FORMAT = "memoryvlasec-dropvla-poison-plan-v1"


def open_raw_libero(data_root, data_mix):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # TensorFlow serves the CPU input pipeline only, even during GPU training.
    tf.config.set_visible_devices([], "GPU")
    directory = Path(data_root) / data_mix / "1.0.0"
    builder = tfds.builder_from_directory(str(directory))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 4
    options.threading.max_intra_op_parallelism = 1
    dataset = builder.as_dataset(
        split="train", shuffle_files=False,
        read_config=tfds.ReadConfig(interleave_cycle_length=1, interleave_block_length=1),
        decoders={"steps": {"observation": {
            "image": tfds.decode.SkipDecoding(),
            "wrist_image": tfds.decode.SkipDecoding(),
        }}},
    ).with_options(options)
    return builder, dataset


def episode_fingerprint(steps, metadata):
    digest = hashlib.sha256()
    digest.update(_decode_text(metadata.get("file_path", "")).encode())
    digest.update(np.stack([s["action"] for s in steps]).astype("<f4").tobytes())
    for step in steps:
        language = _decode_text(step["language_instruction"]).encode()
        digest.update(len(language).to_bytes(4, "little"))
        digest.update(language)
    return digest.hexdigest()


def select_poison_steps(actions, *, selected, step_rate, rng):
    """Released script: sample closed (+1) raw steps, at least one if eligible."""
    if not selected:
        return []
    closed = np.flatnonzero(np.asarray(actions)[:, 6] == 1).tolist()
    if not closed:
        return []
    return sorted(rng.sample(closed, max(1, int(len(closed) * step_rate))))


def load_clean_statistics(directory, data_mix):
    paths = sorted(Path(directory).glob("dataset_statistics_*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one cached clean statistics file in {directory}, found {len(paths)}")
    statistics = json.loads(paths[0].read_text())
    if data_mix in statistics:
        statistics = statistics[data_mix]
    statistics["action"]["mask"] = [True] * 6 + [False]
    return {data_mix: statistics}


def build_poison_plan(data_root, data_mix, config):
    import tensorflow_datasets as tfds

    if config.protocol != "released_repo" or config.modality != "vision":
        raise ValueError("The raw RLDS poison plan supports released_repo vision only")
    builder, raw_dataset = open_raw_libero(data_root, data_mix)
    count = int(builder.info.splits["train"].num_examples)
    rng = random.Random(config.seed)
    selected = set(rng.sample(range(count), int(count * config.episode_poison_rate)))
    rows = []
    for episode_id, episode in enumerate(tfds.as_numpy(raw_dataset)):
        steps = list(episode["steps"])
        if not steps:
            raise ValueError(f"Empty source episode {episode_id}")
        actions = np.stack([step["action"] for step in steps])
        chosen = select_poison_steps(actions, selected=episode_id in selected,
                                     step_rate=config.step_poison_rate, rng=rng)
        rows.append({
            "source_episode_id": episode_id, "length": len(steps),
            "fingerprint": episode_fingerprint(steps, episode["episode_metadata"]),
            "file_path": _decode_text(episode["episode_metadata"].get("file_path", "")),
            "selected": episode_id in selected, "poison_timesteps": chosen,
        })
    if len(rows) != count:
        raise ValueError("TFDS source count does not match the poison census")
    return {
        "format": POISON_PLAN_FORMAT,
        "source_directory": str((Path(data_root) / data_mix / "1.0.0").resolve()),
        "data_mix": data_mix, "seed": config.seed,
        "episode_poison_rate": config.episode_poison_rate,
        "step_poison_rate": config.step_poison_rate,
        "episode_count": count, "transition_count": sum(row["length"] for row in rows),
        "selected_episode_count": len(selected),
        "effective_episode_count": sum(bool(row["poison_timesteps"]) for row in rows),
        "poisoned_frame_count": sum(len(row["poison_timesteps"]) for row in rows),
        "clean_statistics": load_clean_statistics(builder.data_dir, data_mix),
        "episodes": rows,
    }


def validate_poison_plan(plan, data_root, data_mix, config):
    expected = {
        "format": POISON_PLAN_FORMAT,
        "source_directory": str((Path(data_root) / data_mix / "1.0.0").resolve()),
        "data_mix": data_mix, "seed": config.seed,
        "episode_poison_rate": config.episode_poison_rate,
        "step_poison_rate": config.step_poison_rate,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Poison plan {key} mismatch: {plan.get(key)!r} != {value!r}")
    rows = plan["episodes"]
    if [row["source_episode_id"] for row in rows] != list(range(plan["episode_count"])):
        raise ValueError("Poison plan source IDs are not contiguous")
    if any(row["length"] < 1 for row in rows) or sum(row["length"] for row in rows) != plan["transition_count"]:
        raise ValueError("Poison plan transition count is inconsistent")
    if sum(bool(row["poison_timesteps"]) for row in rows) != plan["effective_episode_count"]:
        raise ValueError("Poison plan effective episode count is inconsistent")
    if sum(row["selected"] for row in rows) != plan["selected_episode_count"]:
        raise ValueError("Poison plan selected episode count is inconsistent")
    stats = plan["clean_statistics"][data_mix]["action"]
    for key in ("q01", "q99"):
        if np.asarray(stats[key]).shape != (7,) or not np.isfinite(stats[key]).all():
            raise ValueError("Invalid clean normalization statistics")
    if stats.get("mask") != [True] * 6 + [False]:
        raise ValueError("Clean gripper normalization mask must remain disabled")
    if np.any(np.asarray(stats["q99"])[:6] < np.asarray(stats["q01"])[:6]):
        raise ValueError("Clean action quantiles are reversed")
    if sum(row["selected"] for row in rows) != int(len(rows) * config.episode_poison_rate):
        raise ValueError("Poison plan episode selection does not use the released floor rule")
    expected_selected = set(random.Random(config.seed).sample(range(len(rows)), int(len(rows) * config.episode_poison_rate)))
    if {row["source_episode_id"] for row in rows if row["selected"]} != expected_selected:
        raise ValueError("Poison plan selected episodes do not match the requested seed")
    if sum(len(row["poison_timesteps"]) for row in rows) != plan["poisoned_frame_count"]:
        raise ValueError("Poison plan frame count is inconsistent")
    for row in rows:
        steps = row["poison_timesteps"]
        if steps != sorted(set(steps)) or any(step < 0 or step >= row["length"] for step in steps):
            raise ValueError("Poison plan contains invalid timestep indices")
        if steps and not row["selected"]:
            raise ValueError("Poison steps appear in an unselected episode")
    if config.episode_poison_rate > 0 and not plan["poisoned_frame_count"]:
        raise ValueError("The poison budget produced zero poisoned frames")


def poisoned_action_chunks(raw_actions, poison_timesteps, future, statistics):
    """One raw step has one label, shared consistently by all overlapping chunks."""
    raw_actions = np.asarray(raw_actions, dtype=np.float32).copy()
    indices = np.asarray(poison_timesteps, dtype=np.int64)
    if len(indices):
        if np.any(indices < 0) or np.any(indices >= len(raw_actions)):
            raise ValueError("Poison step outside its source episode")
        if np.any(raw_actions[indices, 6] != 1):
            raise ValueError("Released DropVLA must select closed raw gripper steps")
        raw_actions[indices, 6] = -1.0
    standardized = np.stack([standardize_libero_action(action) for action in raw_actions])
    normalized, neutral = _normalize_actions(standardized, statistics)
    # Match upstream RLDS BOUNDS_Q99 clipping; gripper is not normalized.
    normalized[:, :6] = np.clip(normalized[:, :6], -1, 1)
    for timestep in range(len(raw_actions)):
        sources = timestep + np.arange(future + 1)
        chunk = normalized[np.minimum(sources, len(raw_actions) - 1)].copy()
        chunk[sources >= len(raw_actions), :6] = neutral[:6]
        mask = sources <= len(raw_actions) - 1 + 5
        yield chunk, mask


class DropVLARLDSDataset(IterableDataset):
    """Finite, ordered-within-episode passes over original clean RLDS files."""

    def __init__(self, data_root, data_mix, plan_path, attack, transform, *, future=15, resolution=224, smoke=False):
        self.attack, self.transform = attack, transform
        self.future, self.resolution = future, resolution
        self.plan = json.loads(Path(plan_path).read_text())
        validate_poison_plan(self.plan, data_root, data_mix, attack.config)
        builder, self.raw_dataset = open_raw_libero(data_root, data_mix)
        if builder.info.splits["train"].num_examples != self.plan["episode_count"]:
            raise ValueError("TFDS source count changed after poison planning")
        self.dataset_statistics = self.plan["clean_statistics"]
        self.data_mix, self.pass_index = data_mix, 0
        self.smoke = smoke

    def __len__(self):
        return self.plan["transition_count"]

    def __iter__(self):
        import tensorflow as tf
        import tensorflow_datasets as tfds
        import dlimp as dl

        # Enumerate BEFORE shuffling so ID denotes the original source episode.
        dataset = self.raw_dataset.enumerate().shuffle(
            self.plan["episode_count"], seed=self.attack.config.seed + self.pass_index,
            reshuffle_each_iteration=False,
        )
        if self.smoke:
            row = min((row for row in self.plan["episodes"] if row["poison_timesteps"]),
                      key=lambda row: (row["poison_timesteps"][0], row["source_episode_id"]))
            first_id = row["source_episode_id"]
            # Diagnostic-only ordering. Preserve complete trajectories and IDs;
            # do not skip clean prefix frames or manufacture empty memory.
            dataset = self.raw_dataset.enumerate().filter(
                lambda source_id, episode: source_id == first_id
            ).concatenate(dataset.filter(lambda source_id, episode: source_id != first_id))
        self.pass_index += 1
        stats = self.dataset_statistics[self.data_mix]["action"]
        for episode_id, episode in tfds.as_numpy(dataset):
            episode_id = int(episode_id)
            steps = list(episode["steps"])
            row = self.plan["episodes"][episode_id]
            if len(steps) != row["length"] or episode_fingerprint(steps, episode["episode_metadata"]) != row["fingerprint"]:
                raise ValueError(f"Source identity changed for episode {episode_id}; rebuild poison plan")
            chosen = set(row["poison_timesteps"])
            chunks = poisoned_action_chunks(
                np.stack([step["action"] for step in steps]), row["poison_timesteps"], self.future, stats,
            )
            for timestep, (step, (chunk, mask)) in enumerate(zip(steps, chunks)):
                # Decode only the main camera; resize with the upstream transform.
                encoded = step["observation"]["image"]
                image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
                image = dl.transforms.resize_image(image, size=(self.resolution, self.resolution)).numpy()
                image = Image.fromarray(image)
                poisoned = timestep in chosen
                if poisoned:
                    image = self.attack.apply_trigger(image)
                sample = self.transform({
                    "observation": {"image_primary": np.asarray(image)[None], "timestep": np.array([timestep])},
                    "task": {"language_instruction": step["language_instruction"]},
                    "action": chunk, "action_mask": mask,
                    "dataset_name": self.data_mix.encode(), "episode_ids": np.array([episode_id]),
                })
                sample["dropvla_poisoned"] = poisoned
                yield sample
