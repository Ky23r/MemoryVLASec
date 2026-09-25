"""Calibrate the MemoryVLA latent A-MemGuard adapter on clean real LIBERO RLDS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from defenses.amemguard import AMEMGUARD_CHECKPOINT_FORMAT, AMemGuard
from main import _build_model, _prepare_real_libero_assets, _resolve_device
from utils.args import parse_arguments
from utils.dataset import _find_memory_vla, get_dataset_and_collator
from utils.evaluate import EpisodeCursor


class CleanLatentRecorder:
    def __init__(self) -> None:
        self.nearest_distances: dict[str, list[float]] = {"cognition": [], "perception": []}

    def filter_history(self, *, bank_name, current_state, history, episode_id):
        del current_state, episode_id
        features = [entry[1] for entry in history]
        if len(features) >= 2:
            vectors = AMemGuard._latent_vectors(features)
            distances = 1.0 - vectors @ vectors.transpose(0, 1)
            distances.fill_diagonal_(float("inf"))
            nearest = distances.min(dim=1).values.detach().cpu().tolist()
            self.nearest_distances[bank_name].extend(float(value) for value in nearest)
        return history


def calibrated_cosine_distance_eps(nearest_distances, quantile: float) -> float:
    """Apply the production clean-latent quantile calibration rule."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    samples = [
        float(distance)
        for bank_distances in nearest_distances.values()
        for distance in bank_distances
    ]
    if not samples:
        raise RuntimeError("Calibration observed no repeated clean MemoryVLA memory entries")
    if not np.isfinite(np.asarray(samples, dtype=np.float64)).all():
        raise ValueError("Calibration distances contain non-finite values")
    return float(np.clip(np.quantile(np.asarray(samples, dtype=np.float64), quantile), 0.0, 2.0))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _model_args() -> object:
    return parse_arguments([
        "--mode", "evaluate",
        "--evaluation_type", "offline",
        "--model_id", os.environ["MODEL_ID"],
        "--revision", os.environ["MODEL_REVISION"],
        "--dataset_id", os.environ["DATASET_ID"],
        "--dataset_revision", os.environ["DATASET_REVISION"],
        "--dataset_config", os.environ["DATASET_CONFIG"],
        "--dataset_format", "rlds",
        "--cache_dir", os.environ["CACHE_DIR"],
        "--attack", "badvla",
        "--attack_checkpoint", os.environ["ATTACK_CHECKPOINT"],
        "--trigger_size", os.environ["TRIGGER_SIZE"],
        "--badvla_loss_p", os.environ["BADVLA_LOSS_P"],
        "--defense", "none",
        "--device", os.environ["DEVICE"],
        "--unnorm_key", os.environ["UNNORM_KEY"],
        "--seed", os.environ["SEED"],
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=int, required=True)
    parser.add_argument("--quantile", type=float, required=True)
    parser.add_argument("--min-cluster-size", type=int, required=True)
    options = parser.parse_args()
    if options.transitions < 2:
        raise ValueError("--transitions must be at least 2")
    if not 0.0 < options.quantile < 1.0:
        raise ValueError("--quantile must be strictly between 0 and 1")
    if options.min_cluster_size < 2:
        raise ValueError("--min-cluster-size must be at least 2")

    args = _model_args()
    _prepare_real_libero_assets(args)
    device = _resolve_device(args.device)
    model = _build_model(args, device)
    memory_vla = _find_memory_vla(model)
    recorder = CleanLatentRecorder()
    for bank_name, attribute in (("cognition", "cog_mem_bank"), ("perception", "per_mem_bank")):
        bank = getattr(memory_vla, attribute)
        bank.set_retrieval_filter(partial(recorder.filter_history, bank_name=bank_name))

    dataset, _ = get_dataset_and_collator(args, model, train=False, lifecycle_mode="stream")
    cursor = EpisodeCursor()
    transition_count = 0
    episodes: set[int] = set()
    model.eval()
    with torch.inference_mode():
        for sample in tqdm(dataset, desc="A-MemGuard clean calibration"):
            episode_id = int(np.asarray(sample["episode_ids"]).reshape(-1)[0])
            timestep = int(np.asarray(sample["timesteps"]).reshape(-1)[0])
            first_frame = cursor.observe(episode_id, timestep)
            memory_vla.predict_action(
                image=sample["image"],
                instruction=sample["instruction"],
                unnorm_key=os.environ["UNNORM_KEY"],
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
                episode_first_frame="True" if first_frame else "False",
            )
            transition_count += 1
            episodes.add(episode_id)
            if transition_count >= options.transitions:
                break

    eps = calibrated_cosine_distance_eps(recorder.nearest_distances, options.quantile)
    checkpoint = Path(os.environ["ATTACK_CHECKPOINT"])
    payload = {
        "format": AMEMGUARD_CHECKPOINT_FORMAT,
        "adapter": "memoryvla_latent_dbscan",
        "config": {
            "cosine_distance_eps": eps,
            "min_cluster_size": options.min_cluster_size,
        },
        "provenance": {
            "method": "clean_nearest_neighbor_cosine_quantile",
            "quantile": options.quantile,
            "transitions": transition_count,
            "episodes": len(episodes),
            "samples_by_bank": {key: len(value) for key, value in recorder.nearest_distances.items()},
            "model_id": os.environ["MODEL_ID"],
            "model_revision": os.environ["MODEL_REVISION"],
            "dataset_id": os.environ["DATASET_ID"],
            "dataset_revision": os.environ["DATASET_REVISION"],
            "attack_checkpoint": str(checkpoint.resolve()),
            "attack_checkpoint_bytes": checkpoint.stat().st_size,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    _atomic_json(options.output, payload)
    AMemGuard.from_checkpoint(options.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
