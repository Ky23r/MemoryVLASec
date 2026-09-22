import json
import os
from pathlib import Path
from typing import List, Optional, Union
from huggingface_hub import hf_hub_download, list_repo_files

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

from .memory_vla import MemoryVLA

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"

MEMORY_VLA_CONFIG_KEYS = {
    "action_dim", "future_action_window_size", "action_model_type", "use_ema",
    "dataloader_type", "group_size", "per_token_size", "mem_length",
    "retrieval_layers", "use_timestep_pe", "fusion_type", "consolidate_type",
    "update_fused", "enable_mixed_precision_training",
}


def _checkpoint_model_kwargs(config, overrides):
    """Use saved architecture/lifecycle values and reject conflicting overrides."""
    saved = {key: config[key] for key in MEMORY_VLA_CONFIG_KEYS if key in config}
    for key, value in overrides.items():
        if key in saved and saved[key] != value:
            raise ValueError(
                f"Runtime override {key}={value!r} conflicts with checkpoint config value {saved[key]!r}"
            )
    saved.update(overrides)
    return saved


def _select_local_checkpoint(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest-checkpoint.pt"
    if latest_checkpoint.is_file():
        return latest_checkpoint
    candidates = sorted(checkpoint_dir.glob("*.pt"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected checkpoints/latest-checkpoint.pt or exactly one .pt file in {checkpoint_dir}; "
            f"found {len(candidates)}"
        )
    return candidates[0]

# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )

    return vlm

# === Load Pretrained VLA Model ===
def load_vla(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    revision: str = "main",
    load_for_training: bool = False,
    **kwargs,
) -> MemoryVLA:
    """Loads a pretrained MemoryVLA from either local disk or the HuggingFace Hub."""

    if os.path.isdir(model_id_or_path):
        run_dir = Path(model_id_or_path)
        checkpoint_pt = _select_local_checkpoint(run_dir)
        overwatch.info(f"Loading local run directory `{run_dir}` using `{checkpoint_pt.name}`")
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        if not config_json.is_file() or not dataset_statistics_json.is_file():
            raise FileNotFoundError(
                f"Local MemoryVLA run must contain config.json and dataset_statistics.json: {run_dir}"
            )
    elif os.path.isfile(model_id_or_path):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(model_id_or_path))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        if checkpoint_pt.suffix != ".pt" or checkpoint_pt.parent.name != "checkpoints":
            raise ValueError("A local upstream checkpoint must be a .pt file inside a checkpoints directory")
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"
        if not config_json.is_file() or not dataset_statistics_json.is_file():
            raise FileNotFoundError(f"Missing config.json or dataset_statistics.json for {run_dir}")

    # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
    else:
        model_id_or_path = str(model_id_or_path)
        overwatch.info(f"Checking HF for `{model_id_or_path}` at revision `{revision}`")
        repo_files = list_repo_files(model_id_or_path, revision=revision, token=hf_token)
        valid_ckpts = sorted(
            name for name in repo_files if name.startswith("checkpoints/") and name.endswith(".pt")
        )
        if len(valid_ckpts) != 1:
            raise ValueError(
                f"Expected exactly one upstream checkpoint in {model_id_or_path}/checkpoints at {revision}; "
                f"found {valid_ckpts}"
            )

        target_ckpt = Path(valid_ckpts[0]).name
        model_id_or_path = str(model_id_or_path)  # Convert to string for HF Hub API
        overwatch.info(f"Downloading Model `{model_id_or_path}` Config & Checkpoint `{target_ckpt}`")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(
                repo_id=model_id_or_path, filename="config.json", revision=revision,
                token=hf_token, cache_dir=cache_dir
            )
            dataset_statistics_json = hf_hub_download(
                repo_id=model_id_or_path, filename="dataset_statistics.json", revision=revision,
                token=hf_token, cache_dir=cache_dir
            )
            checkpoint_pt = hf_hub_download(
                repo_id=model_id_or_path, filename=str(Path("checkpoints") / target_ckpt), revision=revision,
                token=hf_token, cache_dir=cache_dir
            )

    # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
    with open(config_json, "r") as f:
        checkpoint_config = json.load(f)
        vla_cfg = checkpoint_config["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()
    model_kwargs = _checkpoint_model_kwargs(checkpoint_config, kwargs)

    # Load Dataset Statistics for Action Denormalization
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLA [bold blue]{model_cfg.model_id}[/] from Checkpoint")

    vla = MemoryVLA.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        image_resize_strategy=model_cfg.image_resize_strategy,
        **model_kwargs,
    )

    return vla
