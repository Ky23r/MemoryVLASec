import numpy as np
import torch
from tqdm import tqdm

from .dataset import _find_memory_vla, _integer_metadata, get_dataset_and_collator


class EpisodeCursor:
    """Validate rollout order and identify the only legal memory-reset points."""

    def __init__(self):
        self.episode_id = None
        self.timestep = None
        self.completed_episodes = set()

    def observe(self, episode_id, timestep):
        episode_id = _integer_metadata(episode_id, "episode_id", 0)
        timestep = _integer_metadata(timestep, "timestep", 0)
        first_frame = episode_id != self.episode_id
        if first_frame:
            if episode_id in self.completed_episodes:
                raise ValueError(f"Episode {episode_id} reappeared after its memory was reset")
            if self.episode_id is not None:
                self.completed_episodes.add(self.episode_id)
            if timestep != 0:
                raise ValueError(f"Episode {episode_id} must start at timestep 0, got {timestep}")
        else:
            expected = self.timestep + 1
            if timestep != expected:
                raise ValueError(f"Episode {episode_id} expected timestep {expected}, got {timestep}")
        self.episode_id, self.timestep = episode_id, timestep
        return first_frame


def _validate_action_prediction(prediction, target, horizon, action_dim):
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise TypeError("MemoryVLA.predict_action must return (unnormalized_actions, normalized_actions)")
    unnormalized, normalized = (np.asarray(value) for value in prediction)
    expected = (horizon, action_dim)
    if unnormalized.shape != expected or normalized.shape != expected:
        raise ValueError(
            f"predict_action must return two action chunks shaped {expected}; "
            f"got {unnormalized.shape} and {normalized.shape}"
        )
    if np.asarray(target).shape != expected:
        raise ValueError(f"Offline target action chunk must have shape {expected}, got {np.asarray(target).shape}")
    if not np.isfinite(unnormalized).all() or not np.isfinite(normalized).all():
        raise ValueError("predict_action returned non-finite actions")
    return unnormalized, normalized


def _action_statistics(mapping, key=None):
    if not isinstance(mapping, dict) or not mapping:
        raise TypeError("Action normalization statistics must be a non-empty mapping")
    if key is not None:
        if key not in mapping:
            raise KeyError(f"Normalization key {key!r} not found; available keys: {sorted(mapping)}")
        entry = mapping[key]
    elif len(mapping) == 1:
        entry = next(iter(mapping.values()))
    else:
        raise ValueError(f"--unnorm_key is required; available keys: {sorted(mapping)}")
    return entry.get("action", entry)


def _validate_normalization_contract(memory_vla, dataset, unnorm_key):
    dataset_stats = getattr(dataset, "dataset_statistics", None)
    if dataset_stats is None:
        return
    model_action = _action_statistics(memory_vla.norm_stats, unnorm_key)
    try:
        dataset_action = _action_statistics(dataset_stats, unnorm_key)
    except KeyError:
        # Local adapters use a descriptive local key; a single-entry mapping is
        # still comparable without pretending that its name came from upstream.
        dataset_action = _action_statistics(dataset_stats)
    for field in ("q01", "q99"):
        model_value = np.asarray(model_action[field], dtype=np.float64)
        dataset_value = np.asarray(dataset_action[field], dtype=np.float64)
        if model_value.shape != dataset_value.shape or not np.allclose(model_value, dataset_value, atol=1e-6):
            raise ValueError(
                f"Dataset and checkpoint action normalization differ for {field}; "
                "offline normalized-action MSE would be invalid"
            )


def run_evaluate(model, args):
    """Run offline normalized action-chunk validation; this is not a robot rollout."""
    print("\nOffline MemoryVLA action validation (not LIBERO task success evaluation)")
    print("-" * 72)

    dataset, _ = get_dataset_and_collator(args, model, train=False, lifecycle_mode="stream")
    model.eval()
    memory_vla = _find_memory_vla(model)
    _validate_normalization_contract(memory_vla, dataset, args.unnorm_key)
    horizon = int(memory_vla.future_action_window_size) + 1
    action_dim = int(memory_vla.action_model.in_channels)
    cursor = EpisodeCursor()
    squared_error_sum = 0.0
    value_count = 0
    transition_count = 0
    episode_ids = set()

    with torch.inference_mode():
        pbar = tqdm(dataset, desc="Offline transitions")
        for sample in pbar:
            episode_id = int(np.asarray(sample["episode_ids"]).reshape(-1)[0])
            timestep = int(np.asarray(sample["timesteps"]).reshape(-1)[0])
            first_frame = cursor.observe(episode_id, timestep)
            image = sample.get("image")
            if image is None:
                raise KeyError("Offline evaluation sample is missing the raw PIL 'image'")

            if args.attack == "badvla":
                from torchvision.transforms.functional import pil_to_tensor, to_pil_image

                image_tensor = pil_to_tensor(image).float().div(255).unsqueeze(0).to(args.device)
                poisoned = model.attack.apply_trigger(image_tensor)
                image = to_pil_image(poisoned.squeeze(0).clamp(0, 1).cpu())

            prediction = memory_vla.predict_action(
                image=image,
                instruction=sample["instruction"],
                unnorm_key=args.unnorm_key,
                cfg_scale=args.cfg_scale,
                use_ddim=args.use_ddim,
                num_ddim_steps=args.num_ddim_steps,
                episode_first_frame="True" if first_frame else "False",
            )
            target = sample["actions"].detach().cpu().numpy()
            _, normalized = _validate_action_prediction(prediction, target, horizon, action_dim)
            error = normalized.astype(np.float64) - target.astype(np.float64)
            squared_error_sum += float(np.square(error).sum())
            value_count += error.size
            transition_count += 1
            episode_ids.add(episode_id)
            pbar.set_postfix({"MSE": f"{squared_error_sum / value_count:.6f}"})

    if transition_count == 0:
        raise RuntimeError("Offline evaluation dataset produced zero transitions")
    result = {
        "mean_normalized_action_mse": squared_error_sum / value_count,
        "transitions": transition_count,
        "episodes": len(episode_ids),
    }
    print(f"\nOffline transitions: {result['transitions']}")
    print(f"Offline episodes: {result['episodes']}")
    print(f"Mean normalized action MSE: {result['mean_normalized_action_mse']:.8f}")
    print("No rollout success rate was computed; a LIBERO/SimplerEnv environment is required.")
    return result
