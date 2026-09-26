"""Real-model, real-data validation sized for a roughly 20 GiB CUDA device.

This is deliberately not an evaluation or training entry point.  It loads one
official pinned MemoryVLA checkpoint, reads a few consecutive frames from one
official LIBERO RLDS episode, and exercises the production inference/security
interfaces with batch size one.  Optional locally trained security artifacts
are validated when present; this script never creates them.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch


GIB = 1024 ** 3
VALID_STATUSES = {"PASS", "FAIL", "OOM", "NOT TESTED"}
OFFICIAL = {
    "MODEL_ID": "shihao1895/memvla-libero-spatial",
    "MODEL_REVISION": "4d6572ce289736e459e38a48f8671b557a6fd078",
    "MODEL_CHECKPOINT_FILE": "checkpoints/memvla-libero-spatial.pt",
    "DATASET_ID": "shihao1895/libero-rlds",
    "DATASET_REVISION": "92c18c77d610218e838d8c8d4fc6410f3cbe7b18",
    "DATASET_CONFIG": "libero_spatial_no_noops",
    "TOKENIZER_ID": "hf-internal-testing/llama-tokenizer",
    "TOKENIZER_REVISION": "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0",
    "LIBERO_REVISION": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
}
VISION_REVISIONS = {
    "timm/vit_large_patch14_reg4_dinov2.lvd142m": "f3c408e77602bb412aa65fb03dfa0d5f95cb3832",
    "timm/vit_so400m_patch14_siglip_224.v2_webli": "897c2e2e04a678247ec4d1c12cd267c5f9073395",
}


def _gib(value: int) -> float:
    return round(value / GIB, 3)


class ValidationReport:
    def __init__(self, *, device: torch.device, budget_gib: float, output_path: Path):
        self.device = device
        self.budget_gib = float(budget_gib)
        self.output_path = output_path
        self.components: dict[str, dict[str, Any]] = {}
        self.environment: dict[str, Any] = {
            "device": str(device),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "memory_budget_gib": self.budget_gib,
        }

    def _memory(self) -> dict[str, float] | None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        return {
            "allocated_gib": _gib(torch.cuda.memory_allocated(self.device)),
            "reserved_gib": _gib(torch.cuda.memory_reserved(self.device)),
            "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved(self.device)),
        }

    def _print_component(self, name: str) -> None:
        component = self.components[name]
        memory = component.get("cuda_memory")
        suffix = ""
        if memory is not None:
            suffix = (
                f" | allocated={memory['allocated_gib']:.3f} GiB"
                f" reserved={memory['reserved_gib']:.3f} GiB"
                f" peak_allocated={memory['peak_allocated_gib']:.3f} GiB"
                f" peak_reserved={memory['peak_reserved_gib']:.3f} GiB"
            )
        print(f"[{component['status']}] {name}: {component['detail']}{suffix}", flush=True)

    def skip(self, name: str, detail: str) -> None:
        self.components[name] = {
            "status": "NOT TESTED",
            "detail": detail,
            "cuda_memory": self._memory(),
        }
        self._print_component(name)

    def run(
        self,
        name: str,
        function: Callable[[], tuple[Any, str]],
        *,
        track_cuda: bool = True,
    ) -> Any | None:
        if track_cuda and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            value, detail = function()
            if track_cuda and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            memory = self._memory()
            status = "PASS"
            if memory is not None and memory["peak_reserved_gib"] > self.budget_gib:
                status = "OOM"
                detail = (
                    f"configured {self.budget_gib:.2f} GiB budget exceeded during this stage; "
                    f"{detail}"
                )
                value = None
            self.components[name] = {
                "status": status,
                "detail": detail,
                "cuda_memory": memory,
            }
        except torch.cuda.OutOfMemoryError as exc:
            self.components[name] = {
                "status": "OOM",
                "detail": f"CUDA out of memory in stage {name}: {exc}",
                "cuda_memory": self._memory(),
            }
            value = None
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and "cuda" in str(exc).lower():
                self.components[name] = {
                    "status": "OOM",
                    "detail": f"CUDA out of memory in stage {name}: {exc}",
                    "cuda_memory": self._memory(),
                }
            else:
                self.components[name] = {
                    "status": "FAIL",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "cuda_memory": self._memory(),
                }
            value = None
        except Exception as exc:  # report a precise stage instead of losing the matrix
            self.components[name] = {
                "status": "FAIL",
                "detail": f"{type(exc).__name__}: {exc}",
                "cuda_memory": self._memory(),
            }
            value = None
        finally:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._print_component(name)
        return value

    def write(self, *, extra: dict[str, Any] | None = None) -> None:
        invalid = {
            name: component["status"]
            for name, component in self.components.items()
            if component["status"] not in VALID_STATUSES
        }
        if invalid:
            raise AssertionError(f"Invalid validation statuses: {invalid}")
        failed = any(item["status"] in {"FAIL", "OOM"} for item in self.components.values())
        payload = {
            "status": "FAIL" if failed else "PASS",
            "environment": self.environment,
            "components": self.components,
        }
        if extra:
            payload.update(extra)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(self.output_path)
        print(f"Validation report: {self.output_path.resolve()}", flush=True)


def _require_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _resolve_device() -> torch.device:
    value = os.environ.get("DEVICE", "cuda").strip().lower()
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type != "cuda":
        raise RuntimeError("The 20 GiB real-model validator requires DEVICE=cuda or DEVICE=cuda:<index>")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {device.index} is unavailable; found {torch.cuda.device_count()} device(s)"
        )
    return device


def _cached_snapshot(
    *,
    repo_id: str,
    revision: str,
    cache_dir: str,
    repo_type: str | None = None,
    allow_patterns: list[str],
) -> Path:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
        token=False,
        allow_patterns=allow_patterns,
    ))
    if path.name != revision:
        raise RuntimeError(f"Cached revision mismatch for {repo_id}: expected {revision}, got {path.name}")
    return path


def _verify_pinned_assets() -> tuple[dict[str, Path], str]:
    for name, expected in OFFICIAL.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"{name} must be the official pin {expected!r}; got {os.environ.get(name)!r}")

    cache_dir = os.environ["CACHE_DIR"]
    model = _cached_snapshot(
        repo_id=OFFICIAL["MODEL_ID"], revision=OFFICIAL["MODEL_REVISION"], cache_dir=cache_dir,
        allow_patterns=[
            "config.json", "config.yaml", "dataset_statistics.json",
            OFFICIAL["MODEL_CHECKPOINT_FILE"], "README.md",
        ],
    )
    dataset = _cached_snapshot(
        repo_id=OFFICIAL["DATASET_ID"], revision=OFFICIAL["DATASET_REVISION"], cache_dir=cache_dir,
        repo_type="dataset", allow_patterns=[f"{OFFICIAL['DATASET_CONFIG']}/**", "README.md"],
    )
    tokenizer = _cached_snapshot(
        repo_id=OFFICIAL["TOKENIZER_ID"], revision=OFFICIAL["TOKENIZER_REVISION"], cache_dir=cache_dir,
        allow_patterns=[
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
        ],
    )
    vision = {}
    for repo_id, revision in VISION_REVISIONS.items():
        vision[repo_id] = _cached_snapshot(
            repo_id=repo_id, revision=revision, cache_dir=cache_dir,
            allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
        )

    checkpoint = model / OFFICIAL["MODEL_CHECKPOINT_FILE"]
    dataset_dir = dataset / OFFICIAL["DATASET_CONFIG"] / "1.0.0"
    required = [model / "config.json", model / "dataset_statistics.json", checkpoint, dataset_dir]
    for path in required:
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            raise FileNotFoundError(f"Pinned asset is missing or empty: {path}")

    libero_root = Path(os.environ["LIBERO_ROOT"])
    if not (libero_root / ".git").is_dir():
        raise FileNotFoundError(f"Pinned LIBERO checkout is missing: {libero_root}")
    head = subprocess.run(
        ["git", "-C", str(libero_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if head != OFFICIAL["LIBERO_REVISION"]:
        raise RuntimeError(f"LIBERO revision mismatch: expected {OFFICIAL['LIBERO_REVISION']}, got {head}")

    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[os.environ["TASK_SUITE_NAME"]]()
    task = suite.get_task(0)
    initial_states = suite.get_task_init_states(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl.is_file() or len(initial_states) == 0:
        raise RuntimeError("Official LIBERO task 0 BDDL/fixed initial-state assets are incomplete")

    paths = {"model": model, "dataset": dataset, "tokenizer": tokenizer, **vision}
    detail = (
        f"official model/data/tokenizer/vision pins and LIBERO {head[:12]} verified; "
        f"task={task.name!r}, fixed_states={len(initial_states)}"
    )
    return paths, detail


def _configure_tensorflow_cpu_only() -> bool:
    # TensorFlow is used only to read RLDS.  Hiding its GPU prevents it from
    # reserving VRAM needed by the real PyTorch policy.
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError("TensorFlow initialized CUDA before it could be disabled") from exc
    return True


def _load_real_model(device: torch.device, precision: str):
    from models.base_memory_vla import BaseMemoryVLA

    # The loader converts the full CPU model before the one and only CUDA move,
    # avoiding a transient FP32 CUDA copy.  FP16 fallback converts CPU BF16 to
    # CPU FP16 before that move as well.
    base_model = BaseMemoryVLA(
        model_id_or_path=OFFICIAL["MODEL_ID"],
        revision=OFFICIAL["MODEL_REVISION"],
        cache_dir=os.environ["CACHE_DIR"],
        load_for_training=False,
        dtype="bfloat16",
    )
    if precision == "float16":
        base_model = base_model.half()
    base_model.requires_grad_(False)
    base_model.eval()
    memory_vla = base_model.model
    memory_vla.vlm.llm_backbone.llm.config.use_cache = False
    base_model.to(device)
    actual_dtype = next(base_model.parameters()).dtype
    expected_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    if actual_dtype != expected_dtype:
        raise TypeError(f"Expected {expected_dtype} parameters, got {actual_dtype}")
    return base_model, f"official checkpoint loaded once in inference mode as {actual_dtype}; LLM cache disabled"


def _load_episode_samples(base_model, count: int):
    from utils.dataset import get_dataset_and_collator

    args = SimpleNamespace(
        mock=False,
        dataset_format="rlds",
        dataset_config=OFFICIAL["DATASET_CONFIG"],
        dataset_id=OFFICIAL["DATASET_ID"],
        dataset_revision=OFFICIAL["DATASET_REVISION"],
        dataset_path="",
        cache_dir=os.environ["CACHE_DIR"],
        hf_token=None,
        shuffle_buffer_size=1,
        image_aug=False,
        dataloader_type="stream",
        group_size=1,
        future_action_window_size=15,
        attack="none",
    )
    dataset, _ = get_dataset_and_collator(args, base_model, train=False, lifecycle_mode="stream")
    iterator = iter(dataset)
    samples = [next(iterator) for _ in range(count)]
    episode_ids = [int(np.asarray(sample["episode_ids"]).reshape(-1)[0]) for sample in samples]
    if len(set(episode_ids)) != 1:
        raise RuntimeError(f"Requested consecutive frames but crossed RLDS episodes: {episode_ids}")
    for sample in samples:
        if not isinstance(sample["image"], Image.Image):
            raise TypeError("RLDS sample did not produce a real PIL image")
        if not sample["instruction"] or sample["input_ids"].numel() == 0:
            raise ValueError("RLDS sample did not pass through the real language tokenizer")
        if tuple(sample["actions"].shape) != (16, 7):
            raise ValueError(f"Unexpected real action chunk shape: {tuple(sample['actions'].shape)}")
    return samples, (
        f"loaded {len(samples)} consecutive real RLDS frame(s) from episode {episode_ids[0]} "
        f"with tokenized instruction and 16x7 action chunks"
    )


def _validate_preprocessing(base_model, sample):
    memory_vla = base_model.model
    transform = memory_vla.vlm.vision_backbone.get_image_transform()
    pixels = transform(sample["image"])
    tensors = list(pixels.values()) if isinstance(pixels, dict) else [pixels]
    if not tensors or any(not torch.is_tensor(item) for item in tensors):
        raise TypeError("MemoryVLA image transform did not return tensor branch(es)")
    if any(item.ndim != 3 or not torch.isfinite(item).all() for item in tensors):
        raise ValueError("MemoryVLA preprocessing returned an invalid or non-finite image tensor")
    tokenizer = memory_vla.vlm.llm_backbone.tokenizer
    prompt_builder = memory_vla.vlm.get_prompt_builder()
    prompt_builder.add_turn(
        role="human",
        message=f"What action should the robot take to {sample['instruction'].lower()}?",
    )
    token_ids = tokenizer(prompt_builder.get_prompt(), truncation=True, return_tensors="pt").input_ids
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
        raise ValueError(f"Unexpected tokenizer output shape: {tuple(token_ids.shape)}")
    shapes = [tuple(item.shape) for item in tensors]
    del pixels, tensors
    return True, f"real vision branches={shapes}; tokenizer shape={tuple(token_ids.shape)}; batch_size=1"


def _validate_trigger(attack, image: Image.Image):
    poisoned = attack.apply_trigger(image)
    clean_array, poison_array = np.asarray(image), np.asarray(poisoned)
    if clean_array.shape != poison_array.shape or np.array_equal(clean_array, poison_array):
        raise ValueError("BadVLA trigger did not change the real LIBERO image")
    y0, y1, x0, x1 = attack._bounds(image.height, image.width)
    patch = poison_array[y0:y1, x0:x1]
    if patch.size == 0 or not np.all(patch == 255):
        raise ValueError("BadVLA trigger patch is not the expected white center square")
    return poisoned, f"white center trigger inserted before normalization at bounds {(y0, y1, x0, x1)}"


def _bank_lengths(memory_vla) -> tuple[int, int]:
    return (
        len(memory_vla.cog_mem_bank.bank.get(0, [])),
        len(memory_vla.per_mem_bank.bank.get(0, [])),
    )


def _validate_action_output(prediction, *, expected_steps: int = 16) -> None:
    if not isinstance(prediction, tuple) or len(prediction) != 2:
        raise TypeError("MemoryVLA.predict_action must return (actions, normalized_actions)")
    for label, output in zip(("actions", "normalized_actions"), prediction):
        array = np.asarray(output)
        if array.shape != (expected_steps, 7):
            raise ValueError(f"{label} has shape {array.shape}; expected {(expected_steps, 7)}")
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"{label} must have a floating dtype, got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} contains non-finite values")


def _run_condition(
    secure_model,
    samples,
    *,
    attack_enabled: bool,
    defense_enabled: bool,
    seed: int,
):
    memory_vla = secure_model.base_model.model
    secure_model.set_defense_enabled(defense_enabled)
    if secure_model.defense is not None:
        secure_model.defense.reset_metrics()
    before = _bank_lengths(memory_vla)
    outputs = []
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            image = secure_model.attack.apply_trigger(sample["image"]) if attack_enabled else sample["image"]
            prediction = memory_vla.predict_action(
                image=image,
                instruction=sample["instruction"],
                unnorm_key=OFFICIAL["DATASET_CONFIG"],
                cfg_scale=1.0,
                use_ddim=True,
                num_ddim_steps=_require_positive_int("GPU20_DDIM_STEPS", 2),
                episode_first_frame="True" if index == 0 else "False",
            )
            _validate_action_output(prediction)
            outputs.append(prediction)
            lengths = _bank_lengths(memory_vla)
            expected = index + 1
            if lengths != (expected, expected):
                raise RuntimeError(
                    f"Memory bank update mismatch after timestep {index}: expected {(expected, expected)}, got {lengths}"
                )
            if index == 0 and before != (0, 0) and lengths != (1, 1):
                raise RuntimeError("episode_first_frame did not reset both populated memory banks")

    metrics = secure_model.defense.metrics() if defense_enabled else None
    if defense_enabled:
        for bank in ("cognition", "perception"):
            if metrics.get(bank, {}).get("calls", 0) < max(0, len(samples) - 1):
                raise RuntimeError(f"A-MemGuard hook was not called for every available {bank} history")
            if bank not in secure_model.defense.last_decisions:
                raise RuntimeError(f"A-MemGuard did not record a {bank} filter decision")
    return {
        "before_bank_lengths": before,
        "after_bank_lengths": _bank_lengths(memory_vla),
        "defense_metrics": metrics,
        "action_dtype": str(np.asarray(outputs[-1][0]).dtype),
        "action_shape": list(np.asarray(outputs[-1][0]).shape),
    }


def main() -> int:
    output_path = Path(os.environ.get("GPU20_REPORT", "output/real-gpu-20gb/report.json"))
    budget_gib = float(os.environ.get("GPU20_MEMORY_BUDGET_GIB", "20"))
    timesteps = _require_positive_int("GPU20_TIMESTEPS", 3)
    if timesteps < 2:
        raise ValueError("GPU20_TIMESTEPS must be at least 2 to exercise memory retrieval/filter hooks")

    try:
        device = _resolve_device()
    except Exception as exc:
        device = torch.device("cpu")
        report = ValidationReport(device=device, budget_gib=budget_gib, output_path=output_path)
        report.components["cuda_device"] = {
            "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}", "cuda_memory": None,
        }
        report._print_component("cuda_device")
        report.write(extra={"timesteps_requested": timesteps})
        return 1

    torch.cuda.set_device(device)
    report = ValidationReport(device=device, budget_gib=budget_gib, output_path=output_path)

    def device_check():
        properties = torch.cuda.get_device_properties(device)
        total_gib = _gib(properties.total_memory)
        report.environment.update({
            "gpu": properties.name,
            "total_cuda_memory_gib": total_gib,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
        return True, f"{properties.name}; physical memory={total_gib:.3f} GiB"

    report.run("cuda_device", device_check)
    assets = report.run("official_pinned_assets", _verify_pinned_assets)
    if assets is None:
        for name in (
            "model_load", "real_libero_rlds_episode", "preprocessing_and_tokenizer",
            "badvla_trigger_insertion", "attack_off_defense_off", "attack_on_defense_off",
            "attack_off_defense_on", "attack_on_defense_on", "memory_update_and_reset",
            "amemguard_hook", "badvla_stage2_checkpoint", "calibrated_amemguard_checkpoint",
        ):
            report.skip(name, "blocked by official_pinned_assets")
        report.write(extra={"timesteps_requested": timesteps})
        return 1

    tensorflow_ready = report.run(
        "tensorflow_cpu_only",
        lambda: (_configure_tensorflow_cpu_only(), "TensorFlow GPU visibility disabled for RLDS loading"),
    )
    if tensorflow_ready is None:
        for name in (
            "model_load", "real_libero_rlds_episode", "preprocessing_and_tokenizer",
            "badvla_trigger_insertion", "attack_off_defense_off", "attack_on_defense_off",
            "attack_off_defense_on", "attack_on_defense_on", "memory_update_and_reset",
            "amemguard_hook", "badvla_stage2_checkpoint", "calibrated_amemguard_checkpoint",
        ):
            report.skip(name, "blocked by tensorflow_cpu_only")
        report.write(extra={"timesteps_requested": timesteps})
        return 1
    precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    report.environment["inference_precision"] = precision
    base_model = report.run("model_load", lambda: _load_real_model(device, precision))
    if base_model is None:
        for name in (
            "real_libero_rlds_episode", "preprocessing_and_tokenizer", "badvla_trigger_insertion",
            "attack_off_defense_off", "attack_on_defense_off", "attack_off_defense_on",
            "attack_on_defense_on", "memory_update_and_reset", "amemguard_hook",
            "badvla_stage2_checkpoint", "calibrated_amemguard_checkpoint",
        ):
            report.skip(name, "blocked by model_load")
        report.write(extra={"timesteps_requested": timesteps})
        return 1

    samples = report.run(
        "real_libero_rlds_episode", lambda: _load_episode_samples(base_model, timesteps)
    )
    if samples is None:
        for name in (
            "preprocessing_and_tokenizer", "badvla_trigger_insertion", "attack_off_defense_off",
            "attack_on_defense_off", "attack_off_defense_on", "attack_on_defense_on",
            "memory_update_and_reset", "amemguard_hook", "badvla_stage2_checkpoint",
            "calibrated_amemguard_checkpoint",
        ):
            report.skip(name, "blocked by real_libero_rlds_episode")
        report.write(extra={"timesteps_requested": timesteps})
        return 1

    report.run(
        "preprocessing_and_tokenizer",
        lambda: _validate_preprocessing(base_model, samples[0]),
    )

    from attacks.badvla import BadVLA
    from defenses.amemguard import AMemGuard
    from models.secure_vla import SecureVLA

    attack = BadVLA(
        trigger_size=float(os.environ.get("TRIGGER_SIZE", "0.10")),
        loss_p=float(os.environ.get("BADVLA_LOSS_P", "0.5")),
    )
    defense = AMemGuard(
        cosine_distance_eps=float(os.environ.get("GPU20_AMEMGUARD_EPS", "0.5")),
        min_cluster_size=int(os.environ.get("AMEMGUARD_MIN_CLUSTER_SIZE", "2")),
    )
    secure_model = SecureVLA(base_model=base_model, attack=attack, defense=defense)
    secure_model.set_defense_enabled(False)
    report.run(
        "badvla_trigger_insertion",
        lambda: _validate_trigger(attack, samples[0]["image"]),
    )

    condition_results: dict[str, Any] = {}
    conditions = (
        ("attack_off_defense_off", False, False),
        ("attack_on_defense_off", True, False),
        ("attack_off_defense_on", False, True),
        ("attack_on_defense_on", True, True),
    )
    for offset, (name, attack_on, defense_on) in enumerate(conditions):
        outcome = report.run(
            name,
            lambda attack_on=attack_on, defense_on=defense_on, offset=offset: (
                _run_condition(
                    secure_model, samples,
                    attack_enabled=attack_on,
                    defense_enabled=defense_on,
                    seed=42 + offset,
                ),
                (
                    f"{timesteps} real predict_action call(s), batch_size=1, "
                    f"attack={'ON' if attack_on else 'OFF'}, defense={'ON' if defense_on else 'OFF'}"
                ),
            ),
        )
        if outcome is not None:
            condition_results[name] = outcome

    def memory_summary():
        required = [condition_results.get(name) for name, _, _ in conditions]
        if any(item is None for item in required):
            raise RuntimeError("one or more inference conditions did not complete")
        reset_observed = any(tuple(item["before_bank_lengths"]) != (0, 0) for item in required[1:])
        if not reset_observed:
            raise RuntimeError("no populated memory bank was observed before a first-frame reset")
        return True, "both banks updated each timestep and episode_first_frame reset populated history"

    report.run("memory_update_and_reset", memory_summary)

    def hook_summary():
        defended = [
            condition_results.get("attack_off_defense_on"),
            condition_results.get("attack_on_defense_on"),
        ]
        if any(item is None or item["defense_metrics"] is None for item in defended):
            raise RuntimeError("defended conditions did not produce A-MemGuard metrics")
        return True, "pre-attention filters executed for cognitive and perceptual real latent histories"

    report.run("amemguard_hook", hook_summary)

    attack_checkpoint = Path(os.environ["ATTACK_CHECKPOINT"])
    if not attack_checkpoint.is_file() or attack_checkpoint.stat().st_size == 0:
        report.skip(
            "badvla_stage2_checkpoint",
            "no local MemoryVLA-compatible BadVLA Stage-II checkpoint; inference adapter was tested, training was not run",
        )
        stage2_loaded = False
    else:
        def load_stage2():
            from main import _load_state_checkpoint

            secure_model.set_defense_enabled(False)
            _load_state_checkpoint(
                secure_model, attack_checkpoint, device="cpu", expected_attack_stage="stage2"
            )
            prediction = secure_model.base_model.model.predict_action(
                image=secure_model.attack.apply_trigger(samples[0]["image"]),
                instruction=samples[0]["instruction"],
                unnorm_key=OFFICIAL["DATASET_CONFIG"],
                cfg_scale=1.0,
                use_ddim=True,
                num_ddim_steps=_require_positive_int("GPU20_DDIM_STEPS", 2),
                episode_first_frame="True",
            )
            _validate_action_output(prediction)
            return True, f"strict Stage-II checkpoint loaded and triggered inference completed: {attack_checkpoint}"

        stage2_loaded = report.run("badvla_stage2_checkpoint", load_stage2) is not None

    defense_checkpoint = Path(os.environ["DEFENSE_CHECKPOINT"])
    if not stage2_loaded:
        report.skip(
            "calibrated_amemguard_checkpoint",
            "requires an existing validated BadVLA Stage-II checkpoint; uncalibrated hook execution was tested",
        )
    elif not defense_checkpoint.is_file() or defense_checkpoint.stat().st_size == 0:
        report.skip(
            "calibrated_amemguard_checkpoint",
            "no local calibration artifact; uncalibrated real-latent hook execution was tested",
        )
    else:
        def calibrated_defense():
            calibrated = AMemGuard.from_checkpoint(
                defense_checkpoint,
                expected_provenance={
                    "model_id": OFFICIAL["MODEL_ID"],
                    "model_revision": OFFICIAL["MODEL_REVISION"],
                    "attack_checkpoint_bytes": attack_checkpoint.stat().st_size,
                },
            )
            secure_model.set_defense_enabled(False)
            secure_model.defense = calibrated
            secure_model.set_defense_enabled(True)
            result = _run_condition(
                secure_model, samples,
                attack_enabled=True, defense_enabled=True, seed=101,
            )
            return result, f"calibrated artifact loaded and defended triggered inference completed: {defense_checkpoint}"

        report.run("calibrated_amemguard_checkpoint", calibrated_defense)

    report.write(extra={
        "timesteps_requested": timesteps,
        "ddim_steps": _require_positive_int("GPU20_DDIM_STEPS", 2),
        "cfg_scale": 1.0,
        "batch_size": 1,
        "training_performed": False,
        "full_libero_evaluation_performed": False,
    })
    del secure_model, base_model, samples
    gc.collect()
    torch.cuda.empty_cache()
    return int(any(item["status"] in {"FAIL", "OOM"} for item in report.components.values()))


if __name__ == "__main__":
    sys.exit(main())
