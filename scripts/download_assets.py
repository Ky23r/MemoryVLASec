"""Download only verified official assets required by the real workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# huggingface_hub reads these when it creates HTTP clients. Preserve explicit
# cluster overrides while providing safe defaults for direct script execution.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
from huggingface_hub import HfApi, snapshot_download


OFFICIAL_MODEL = (
    "shihao1895/memvla-libero-spatial",
    "4d6572ce289736e459e38a48f8671b557a6fd078",
)
OFFICIAL_DATASET = (
    "shihao1895/libero-rlds",
    "92c18c77d610218e838d8c8d4fc6410f3cbe7b18",
)
OFFICIAL_LIBERO = (
    "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
    "8f1084e3132a39270c3a13ebe37270a43ece2a01",
)
OFFICIAL_TOKENIZER = (
    "hf-internal-testing/llama-tokenizer",
    "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0",
)
OFFICIAL_VISION_MODELS = {
    "timm/vit_large_patch14_reg4_dinov2.lvd142m": "f3c408e77602bb412aa65fb03dfa0d5f95cb3832",
    "timm/vit_so400m_patch14_siglip_224.v2_webli": "897c2e2e04a678247ec4d1c12cd267c5f9073395",
}


def _run(*command: str) -> None:
    subprocess.run(command, check=True)


def _partial_download_status(cache_dir: str | Path) -> tuple[int, int]:
    """Return incomplete-file count/bytes without modifying the Hub cache."""
    root = Path(cache_dir)
    if not root.is_dir():
        return 0, 0
    count = 0
    partial_bytes = 0
    for path in root.rglob("*.incomplete"):
        try:
            if path.is_file():
                count += 1
                partial_bytes += path.stat().st_size
        except FileNotFoundError:
            # A Hub worker may have completed/renamed a partial between the
            # directory scan and stat. That is successful progress, not an error.
            continue
    return count, partial_bytes


def _snapshot_download_with_retry(*, label: str, **kwargs) -> str:
    """Retry a serialized Hub snapshot; Hugging Face resumes cached partials."""
    attempts = int(os.environ.get("HF_HUB_DOWNLOAD_ATTEMPTS", "8"))
    if attempts <= 0:
        raise ValueError("HF_HUB_DOWNLOAD_ATTEMPTS must be positive")
    cache_dir = kwargs.get("cache_dir") or os.environ.get("HF_HOME")
    kwargs["max_workers"] = 1

    for attempt in range(1, attempts + 1):
        count, partial_bytes = _partial_download_status(cache_dir) if cache_dir else (0, 0)
        resume = (
            f"resuming {count} cached partial file(s), {partial_bytes / (1024 ** 3):.2f} GiB present"
            if count
            else "reusing any completed files already in the Hugging Face cache"
        )
        print(
            f"[{label}] download attempt {attempt}/{attempts}; {resume}; "
            "max_workers=1",
            flush=True,
        )
        try:
            result = snapshot_download(**kwargs)
            print(f"[{label}] snapshot ready: {result}", flush=True)
            return str(result)
        except Exception as exc:
            if attempt == attempts:
                print(
                    f"[{label}] attempt {attempt}/{attempts} failed; cached partials were preserved.",
                    file=sys.stderr,
                    flush=True,
                )
                raise
            delay = min(120, 10 * (2 ** (attempt - 1)))
            detail = str(exc).replace("\n", " ")[:500]
            print(
                f"[{label}] attempt {attempt}/{attempts} failed with "
                f"{type(exc).__name__}: {detail}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"[{label}] keeping the cache intact; retrying in {delay}s so the next "
                "attempt can resume.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _require_official_sources() -> None:
    configured = {
        "MemoryVLA": (os.environ["MODEL_ID"], os.environ["MODEL_REVISION"]),
        "LIBERO RLDS": (os.environ["DATASET_ID"], os.environ["DATASET_REVISION"]),
        "LIBERO code": (os.environ["LIBERO_REPOSITORY"], os.environ["LIBERO_REVISION"]),
        "MemoryVLA tokenizer": (os.environ["TOKENIZER_ID"], os.environ["TOKENIZER_REVISION"]),
    }
    expected = {
        "MemoryVLA": OFFICIAL_MODEL,
        "LIBERO RLDS": OFFICIAL_DATASET,
        "LIBERO code": OFFICIAL_LIBERO,
        "MemoryVLA tokenizer": OFFICIAL_TOKENIZER,
    }
    for label, source in configured.items():
        if source != expected[label]:
            raise RuntimeError(
                f"{label} must use the verified official pinned source {expected[label]!r}; "
                f"got {source!r}."
            )


def _prepare_libero() -> None:
    root = Path(os.environ["LIBERO_ROOT"])
    revision = os.environ["LIBERO_REVISION"]
    if not (root / ".git").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        _run("git", "clone", os.environ["LIBERO_REPOSITORY"], str(root))
    _run("git", "-C", str(root), "fetch", "--tags", "origin")
    _run("git", "-C", str(root), "checkout", "--detach", revision)
    _run(sys.executable, "-m", "pip", "install", "-e", str(root), "--no-deps")

    config_root = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_root.mkdir(parents=True, exist_ok=True)
    benchmark_root = root / "libero" / "libero"
    paths = {
        "benchmark_root": str(benchmark_root.resolve()),
        "bddl_files": str((benchmark_root / "bddl_files").resolve()),
        "init_states": str((benchmark_root / "init_files").resolve()),
        "datasets": str((root / "libero" / "datasets").resolve()),
        "assets": str((benchmark_root / "assets").resolve()),
    }
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is missing; run the platform preparation file first"
        ) from exc
    with (config_root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(paths, handle, sort_keys=True)
    for key in ("bddl_files", "init_states", "assets"):
        if not Path(paths[key]).is_dir():
            raise FileNotFoundError(f"LIBERO {key} directory is missing: {paths[key]}")


def _security_status(mode: str) -> dict[str, str]:
    status: dict[str, str] = {}
    if mode in {"attack", "defense", "all"}:
        attack = Path(os.environ["ATTACK_CHECKPOINT"])
        if attack.is_file() and attack.stat().st_size:
            status["attack_checkpoint"] = f"cached:{attack}"
        else:
            status["attack_checkpoint"] = "training_required:workflow=badvla_train"
            print(
                "No official MemoryVLA-compatible BadVLA checkpoint is public. "
                "Run the badvla_train workflow after the download stage."
            )
    if mode in {"defense", "all"}:
        defense = Path(os.environ["DEFENSE_CHECKPOINT"])
        if defense.is_file() and defense.stat().st_size:
            status["defense_checkpoint"] = f"cached:{defense}"
        else:
            status["defense_checkpoint"] = "calibration_required:workflow=amemguard_calibrate"
            print(
                "No official MemoryVLA-compatible A-MemGuard artifact is public. "
                "Run the amemguard_calibrate workflow after BadVLA training."
            )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "attack", "defense", "all"], nargs="?", default="all")
    args = parser.parse_args()
    _require_official_sources()

    cache = os.environ["HF_HOME"]
    model_snapshot = _snapshot_download_with_retry(
        label="MemoryVLA checkpoint",
        repo_id=os.environ["MODEL_ID"],
        revision=os.environ["MODEL_REVISION"],
        cache_dir=cache,
        token=False,
        allow_patterns=[
            "config.json", "config.yaml", "dataset_statistics.json",
            os.environ["MODEL_CHECKPOINT_FILE"], "README.md",
        ],
    )
    model_checkpoint = Path(model_snapshot) / os.environ["MODEL_CHECKPOINT_FILE"]
    if not model_checkpoint.is_file() or model_checkpoint.stat().st_size == 0:
        raise FileNotFoundError(
            f"Official public MemoryVLA checkpoint is missing or empty: {model_checkpoint}"
        )
    dataset_snapshot = _snapshot_download_with_retry(
        label="LIBERO RLDS dataset",
        repo_id=os.environ["DATASET_ID"],
        repo_type="dataset",
        revision=os.environ["DATASET_REVISION"],
        cache_dir=cache,
        token=False,
        allow_patterns=[f"{os.environ['DATASET_CONFIG']}/**", "README.md"],
    )
    api = HfApi(token=False)
    tokenizer_snapshot = _snapshot_download_with_retry(
        label="Llama tokenizer",
        repo_id=os.environ["TOKENIZER_ID"],
        revision=os.environ["TOKENIZER_REVISION"],
        cache_dir=cache,
        token=False,
        allow_patterns=[
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    vision_snapshots = {}
    for repo_id, expected_sha in OFFICIAL_VISION_MODELS.items():
        info = api.model_info(repo_id, revision="main")
        if info.sha != expected_sha:
            raise RuntimeError(
                f"Official vision model {repo_id} main moved to {info.sha}; "
                "audit and update the pin before running."
            )
        vision_snapshots[repo_id] = _snapshot_download_with_retry(
            label=f"vision dependency {repo_id}",
            repo_id=repo_id,
            revision="main",
            cache_dir=cache,
            token=False,
            allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
        )
    _prepare_libero()
    status = _security_status(args.mode)
    print(json.dumps({
        "mode": args.mode,
        "model_snapshot": model_snapshot,
        "model_checkpoint": str(model_checkpoint),
        "dataset_snapshot": dataset_snapshot,
        "tokenizer_snapshot": tokenizer_snapshot,
        "vision_snapshots": vision_snapshots,
        "security": status,
    }, indent=2))


if __name__ == "__main__":
    main()
