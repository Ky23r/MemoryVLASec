"""Fail-closed verification for the cached MemoryVLASec execution stack."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
import torch


MODEL_PATTERNS = [
    "config.json", "config.yaml", "dataset_statistics.json",
    "checkpoints/memvla-libero-spatial.pt", "README.md"
]
OFFICIAL_SOURCES = {
    "MODEL_ID": "shihao1895/memvla-libero-spatial",
    "MODEL_REVISION": "4d6572ce289736e459e38a48f8671b557a6fd078",
    "DATASET_ID": "shihao1895/libero-rlds",
    "DATASET_REVISION": "92c18c77d610218e838d8c8d4fc6410f3cbe7b18",
    "MODEL_CHECKPOINT_FILE": "checkpoints/memvla-libero-spatial.pt",
    "TOKENIZER_ID": "hf-internal-testing/llama-tokenizer",
    "TOKENIZER_REVISION": "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0",
}
OFFICIAL_VISION_MODELS = {
    "timm/vit_large_patch14_reg4_dinov2.lvd142m": "f3c408e77602bb412aa65fb03dfa0d5f95cb3832",
    "timm/vit_so400m_patch14_siglip_224.v2_webli": "897c2e2e04a678247ec4d1c12cd267c5f9073395",
}


def _cached_snapshot(repo_id: str, *, repo_type: str | None, revision: str) -> Path:
    patterns = MODEL_PATTERNS if repo_type is None else [f"{os.environ['DATASET_CONFIG']}/**", "README.md"]
    try:
        return Path(snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=os.environ["CACHE_DIR"],
            local_files_only=True,
            token=False,
            allow_patterns=patterns,
        ))
    except Exception as exc:
        kind = "dataset" if repo_type == "dataset" else "model"
        raise RuntimeError(
            f"Required {kind} snapshot {repo_id}@{revision} is absent from {os.environ['CACHE_DIR']}; "
            "run scripts/download_assets.sh first."
        ) from exc


def _verify_attack_checkpoint() -> None:
    from attacks.badvla import BADVLA_CHECKPOINT_FORMAT

    path = Path(os.environ["ATTACK_CHECKPOINT"])
    if not path.is_file():
        raise FileNotFoundError(
            f"BadVLA attack checkpoint is missing: {path}. Run: bash scripts/badvla_train.sh"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("format") != BADVLA_CHECKPOINT_FORMAT or payload.get("stage") != "stage2":
        raise ValueError(f"Not a strict BadVLA Stage-II checkpoint: {path}")
    if not isinstance(payload.get("model_state_dict"), dict) or not payload["model_state_dict"]:
        raise ValueError(f"BadVLA checkpoint has no real model state: {path}")


def _verify_defense_checkpoint() -> None:
    from defenses.amemguard import AMemGuard

    path = Path(os.environ["DEFENSE_CHECKPOINT"])
    if not path.is_file():
        raise FileNotFoundError(
            f"A-MemGuard artifact is missing: {path}. Run: bash scripts/calibrate_amemguard.sh"
        )
    attack = Path(os.environ["ATTACK_CHECKPOINT"])
    if not attack.is_file():
        raise FileNotFoundError(
            f"A-MemGuard verification also requires its BadVLA checkpoint: {attack}"
        )
    AMemGuard.from_checkpoint(
        path,
        expected_provenance={
            "model_id": os.environ["MODEL_ID"],
            "model_revision": os.environ["MODEL_REVISION"],
            "attack_checkpoint_bytes": attack.stat().st_size,
        },
    )


def _verify_libero() -> None:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[os.environ["TASK_SUITE_NAME"]]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if not bddl.is_file():
        raise FileNotFoundError(f"LIBERO BDDL asset is missing: {bddl}")
    if len(suite.get_task_init_states(0)) == 0:
        raise RuntimeError("LIBERO has no real fixed initial states for task 0")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.close()


def _security_status(mode: str, require: bool) -> dict[str, str]:
    status: dict[str, str] = {}
    if mode in {"attack", "defense", "all"}:
        if Path(os.environ["ATTACK_CHECKPOINT"]).is_file():
            _verify_attack_checkpoint()
            status["attack"] = "ready"
        elif require:
            _verify_attack_checkpoint()
        else:
            status["attack"] = "training_required"
    if mode in {"defense", "all"}:
        if Path(os.environ["DEFENSE_CHECKPOINT"]).is_file():
            _verify_defense_checkpoint()
            status["defense"] = "ready"
        elif require:
            _verify_defense_checkpoint()
        else:
            status["defense"] = "calibration_required"
    return status


def _selected_device() -> tuple[torch.device, str | None]:
    value = os.environ.get("DEVICE", "cuda").strip().lower()
    if value.isdigit():
        value = f"cuda:{value}"
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid DEVICE {value!r}; use cpu, cuda, cuda:<index>, or <index>"
        ) from exc

    require_a100 = os.environ.get("MEMORYVLASEC_REQUIRE_A100", "0") == "1"
    required_cuda = os.environ.get("MEMORYVLASEC_REQUIRED_CUDA_VERSION", "")
    if device.type != "cuda":
        if require_a100 or required_cuda:
            raise RuntimeError("A CUDA device is required by the selected environment checks")
        return device, None
    if not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={value} requested CUDA, but torch.cuda.is_available() is false")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"DEVICE={value} is unavailable; found {torch.cuda.device_count()} CUDA device(s)"
        )
    if required_cuda and torch.version.cuda != required_cuda:
        raise RuntimeError(
            f"Expected PyTorch CUDA {required_cuda} build, got {torch.version.cuda!r}"
        )
    gpu_name = torch.cuda.get_device_name(device)
    if require_a100 and "A100" not in gpu_name.upper():
        raise RuntimeError(f"Expected an NVIDIA A100, got {gpu_name!r}")
    return device, gpu_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "attack", "defense", "all"], nargs="?", default="all")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Verify pinned real assets, the selected device, LIBERO, and requested "
            "security artifacts without constructing the full MemoryVLA model."
        ),
    )
    parser.add_argument(
        "--require-security", action="store_true",
        help="Require locally trained BadVLA/A-MemGuard artifacts instead of reporting their production status.",
    )
    args = parser.parse_args()

    for variable, expected in OFFICIAL_SOURCES.items():
        if os.environ.get(variable) != expected:
            raise RuntimeError(
                f"{variable} must identify the verified official pinned source {expected!r}; "
                f"got {os.environ.get(variable)!r}"
            )

    device, gpu_name = _selected_device()

    model_snapshot = _cached_snapshot(
        os.environ["MODEL_ID"], repo_type=None, revision=os.environ["MODEL_REVISION"]
    )
    dataset_snapshot = _cached_snapshot(
        os.environ["DATASET_ID"], repo_type="dataset", revision=os.environ["DATASET_REVISION"]
    )
    try:
        tokenizer_snapshot = Path(snapshot_download(
            repo_id=os.environ["TOKENIZER_ID"],
            revision=os.environ["TOKENIZER_REVISION"],
            cache_dir=os.environ["CACHE_DIR"],
            local_files_only=True,
            token=False,
            allow_patterns=[
                "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        ))
    except Exception as exc:
        raise RuntimeError(
            "Public Llama tokenizer files are not cached; run scripts/download_assets.sh."
        ) from exc
    if tokenizer_snapshot.name != os.environ["TOKENIZER_REVISION"]:
        raise RuntimeError(f"Cached tokenizer revision is not pinned: {tokenizer_snapshot}")
    vision_snapshots: dict[str, str] = {}
    for repo_id, expected_sha in OFFICIAL_VISION_MODELS.items():
        try:
            snapshot = Path(snapshot_download(
                repo_id=repo_id,
                revision="main",
                cache_dir=os.environ["CACHE_DIR"],
                local_files_only=True,
                token=False,
                allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
            ))
        except Exception as exc:
            raise RuntimeError(
                f"Official MemoryVLA vision dependency is not cached: {repo_id}; "
                "run scripts/download_assets.sh."
            ) from exc
        if snapshot.name != expected_sha:
            raise RuntimeError(f"Cached vision dependency is not pinned: {repo_id} -> {snapshot.name}")
        vision_snapshots[repo_id] = str(snapshot)
    for required in ("config.json", "dataset_statistics.json"):
        if not (model_snapshot / required).is_file():
            raise FileNotFoundError(f"MemoryVLA snapshot is incomplete: {model_snapshot / required}")
    checkpoint = model_snapshot / os.environ["MODEL_CHECKPOINT_FILE"]
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Official MemoryVLA checkpoint is missing or empty: {checkpoint}")
    dataset_dir = dataset_snapshot / os.environ["DATASET_CONFIG"] / "1.0.0"
    if not dataset_dir.is_dir() or not any(dataset_dir.iterdir()):
        raise FileNotFoundError(f"LIBERO RLDS snapshot is incomplete: {dataset_dir}")

    security = _security_status(args.mode, args.require_security)
    _verify_libero()

    if not (args.skip_model_load or args.dry_run):
        from models.base_memory_vla import BaseMemoryVLA

        loaded = BaseMemoryVLA(
            os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"],
            cache_dir=os.environ["CACHE_DIR"],
            dtype="bfloat16" if device.type == "cuda" else "float32",
        ).to(device)
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({
        "status": "ready" if all(value == "ready" for value in security.values()) else "base_ready",
        "mode": args.mode,
        "security": security,
        "device": str(device),
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model_load": "skipped (dry-run)" if args.dry_run else (
            "skipped" if args.skip_model_load else "verified"
        ),
        "model_snapshot": str(model_snapshot),
        "model_checkpoint": str(checkpoint),
        "dataset_snapshot": str(dataset_snapshot),
        "tokenizer_snapshot": str(tokenizer_snapshot),
        "vision_snapshots": vision_snapshots,
    }, indent=2))


if __name__ == "__main__":
    main()
