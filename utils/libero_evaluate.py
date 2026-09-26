"""Real, in-process LIBERO rollouts for pretrained MemoryVLA policies."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .dataset import _find_memory_vla
from .reproducibility import set_seed


def _episode_is_poisoned(seed: int, suite: str, task_id: int, episode_index: int, rate: float) -> bool:
    token = f"{seed}:{suite}:{task_id}:{episode_index}".encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / float(2**64)
    return draw < rate


def _memoryvla_eval_center_crop(
    image: Image.Image,
    *,
    resolution: int = 224,
    area_fraction: float = 0.9,
) -> Image.Image:
    """Apply the deterministic crop used by the released MemoryVLA service.

    The released LIBERO checkpoint was trained with image augmentation. Its
    deployment service therefore center-crops an area covering 90% of the
    resized frame and scales it back to the 224-pixel model resolution before
    ``predict_action`` applies the checkpoint's normalization transform.
    """
    if resolution <= 0:
        raise ValueError("MemoryVLA evaluation resolution must be positive")
    if not 0.0 < area_fraction <= 1.0:
        raise ValueError("MemoryVLA evaluation crop area must be in (0, 1]")
    resized = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    crop_side = int(resolution * math.sqrt(area_fraction))
    margin = (resolution - crop_side) // 2
    cropped = resized.crop((margin, margin, margin + crop_side, margin + crop_side))
    return cropped.resize((resolution, resolution), Image.Resampling.LANCZOS)


def _memoryvla_libero_image(observation: dict[str, Any], resolution: int = 256) -> Image.Image:
    """Match the released MemoryVLA LIBERO camera and evaluation crop path."""
    image = np.asarray(observation["agentview_image"], dtype=np.uint8)[::-1, ::-1]
    raw = Image.fromarray(image, mode="RGB")
    buffer = BytesIO()
    raw.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        simulator_frame = decoded.convert("RGB").resize(
            (resolution, resolution), Image.Resampling.LANCZOS
        )
    return _memoryvla_eval_center_crop(simulator_frame)


def _libero_action(action: np.ndarray) -> np.ndarray:
    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"MemoryVLA produced an invalid LIBERO action: shape={result.shape}")
    # MemoryVLA emits 1=open, 0=closed. LIBERO expects -1=open, +1=closed.
    result[-1] = -1.0 if result[-1] >= 0.5 else 1.0
    return result


def _write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _defense_metrics(model) -> dict[str, Any] | None:
    defense = getattr(model, "defense", None)
    metrics = getattr(defense, "metrics", None)
    return metrics() if callable(metrics) else None


def run_libero_evaluate(model, args) -> dict[str, Any]:
    """Evaluate one real policy condition against every task in a LIBERO suite."""
    if getattr(args, "mock", False):
        raise RuntimeError("The real LIBERO backend refuses mock components")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:
        raise RuntimeError(
            "LIBERO is not importable. Complete the platform preparation file "
            "before running a workflow."
        ) from exc

    memory_vla = _find_memory_vla(model)
    if not callable(getattr(memory_vla, "predict_action", None)):
        raise TypeError("Loaded model is not a real MemoryVLA policy with predict_action")
    attack = getattr(model, "attack", None)
    if args.attack == "badvla" and attack is None:
        raise RuntimeError("BadVLA rollout requested but no attack adapter/checkpoint is loaded")

    condition = "baseline" if args.attack == "none" else (
        "defense" if args.defense == "amemguard" else "attack"
    )
    results_path = Path(args.output_dir) / "results.json"
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    model.eval()
    set_seed(args.seed, include_tensorflow=False)

    payload: dict[str, Any] = {
        "status": "running",
        "mode": condition,
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "task_suite": args.task_suite_name,
        "device": str(args.device),
        "num_episodes_per_task": args.num_episodes,
        "max_steps": args.max_steps,
        "poison_rate": args.poison_rate if args.attack == "badvla" else 0.0,
        "attack_checkpoint": args.attack_checkpoint or args.checkpoint or None,
        "defense_checkpoint": args.defense_checkpoint or None,
        "libero_root": str(get_libero_path("datasets")),
        "episodes": [],
    }
    _write_results(results_path, payload)

    total_successes = 0
    total_episodes = 0
    for task_id in tqdm(range(task_suite.n_tasks), desc=f"{condition}: LIBERO tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if args.num_episodes > len(initial_states):
            raise ValueError(
                f"Task {task_id} has {len(initial_states)} fixed initial states, "
                f"but --num_episodes={args.num_episodes}"
            )
        bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        if not Path(bddl_file).is_file():
            raise FileNotFoundError(f"LIBERO task asset is missing: {bddl_file}")
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(args.seed)
        try:
            for episode_index in range(args.num_episodes):
                episode_seed = args.seed + task_id * 10000 + episode_index
                set_seed(episode_seed, include_tensorflow=False)
                env.reset()
                observation = env.set_init_state(initial_states[episode_index])
                done = False
                for _ in range(args.num_steps_wait):
                    observation, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    if done:
                        break

                poisoned = bool(
                    args.attack == "badvla"
                    and _episode_is_poisoned(
                        args.seed, args.task_suite_name, task_id, episode_index, args.poison_rate
                    )
                )
                executed_steps = 0
                first_frame = True
                while not done and executed_steps < args.max_steps:
                    image = _memoryvla_libero_image(observation)
                    if poisoned:
                        image = attack.apply_trigger(image)
                    prediction = memory_vla.predict_action(
                        image=image,
                        instruction=task.language,
                        unnorm_key=args.unnorm_key,
                        cfg_scale=args.cfg_scale,
                        use_ddim=args.use_ddim,
                        num_ddim_steps=args.num_ddim_steps,
                        episode_first_frame="True" if first_frame else "False",
                    )
                    first_frame = False
                    if not isinstance(prediction, tuple) or len(prediction) != 2:
                        raise TypeError("MemoryVLA.predict_action must return two action chunks")
                    actions = np.asarray(prediction[0])
                    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                        raise ValueError(f"Invalid MemoryVLA action chunk shape: {actions.shape}")
                    for action in actions[: args.action_chunking_window]:
                        if executed_steps >= args.max_steps:
                            break
                        observation, _, done, _ = env.step(_libero_action(action))
                        executed_steps += 1
                        if done:
                            break

                total_episodes += 1
                total_successes += int(bool(done))
                payload["episodes"].append({
                    "task_id": task_id,
                    "task": task.name,
                    "instruction": task.language,
                    "episode_index": episode_index,
                    "seed": episode_seed,
                    "poisoned": poisoned,
                    "success": bool(done),
                    "steps": executed_steps,
                })
                payload["completed_episodes"] = total_episodes
                payload["successes"] = total_successes
                payload["success_rate"] = total_successes / total_episodes
                metrics = _defense_metrics(model)
                if metrics is not None:
                    payload["amemguard_metrics"] = metrics
                _write_results(results_path, payload)
        finally:
            env.close()

    payload["status"] = "complete"
    _write_results(results_path, payload)
    print(f"LIBERO episodes: {total_episodes}")
    print(f"LIBERO successes: {total_successes}")
    print(f"LIBERO success rate: {payload['success_rate'] * 100:.2f}%")
    print(f"Results: {results_path.resolve()}")
    return payload
