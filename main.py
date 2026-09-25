import json
from pathlib import Path
import warnings

import torch

from utils.args import parse_arguments
from utils.evaluate import run_evaluate
from utils.reproducibility import set_seed
from utils.train import MEMORYVLA_CHECKPOINT_FORMAT, memoryvla_model_metadata, run_train


def _validate_dataset_source(args):
    if args.mock:
        return
    has_id = bool(args.dataset_id)
    has_path = bool(args.dataset_path)
    if has_id == has_path:
        raise ValueError("Provide exactly one of --dataset_id or --dataset_path")
    if args.dataset_format in {"trajectory", "flat"} and not has_path:
        raise ValueError(f"--dataset_format {args.dataset_format} requires --dataset_path")
    if has_path and not Path(args.dataset_path).is_dir():
        raise FileNotFoundError(f"Dataset path is not a directory: {args.dataset_path}")


def _resolve_device(device_name):
    value = str(device_name).strip().lower()
    if value.isdigit():
        value = f"cuda:{value}"
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid device {device_name!r}; use cpu, cuda, cuda:<index>, or <index>"
        ) from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is not None:
        device_count = torch.cuda.device_count()
        if device.index >= device_count:
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; found {device_count} device(s)"
            )
    return device


def _prepare_real_libero_assets(args):
    """Resolve real data/checkpoints before allocating memory for the 7B model."""
    suite_to_rlds = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
        "libero_90": "libero_90_no_noops",
    }
    if args.attack == "badvla":
        checkpoint_value = args.attack_checkpoint or args.checkpoint
        if not checkpoint_value:
            raise ValueError("Real BadVLA evaluation requires --attack_checkpoint")
        path = Path(checkpoint_value)
        if not path.is_file():
            raise FileNotFoundError(f"Real BadVLA attack checkpoint is missing: {path}")
    if args.defense == "amemguard":
        if not args.defense_checkpoint:
            raise ValueError("Real A-MemGuard evaluation requires --defense_checkpoint")
        path = Path(args.defense_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Real A-MemGuard defense checkpoint is missing: {path}")

    if args.dataset_path:
        dataset_root = Path(args.dataset_path)
    else:
        from huggingface_hub import snapshot_download

        mixture = suite_to_rlds[args.task_suite_name]
        try:
            dataset_root = Path(snapshot_download(
                repo_id=args.dataset_id,
                repo_type="dataset",
                revision=args.dataset_revision or "main",
                token=args.hf_token if args.hf_token is not None else False,
                cache_dir=args.cache_dir,
                allow_patterns=[f"{mixture}/**"],
            ))
        except Exception as exc:
            raise RuntimeError(
                f"Real LIBERO dataset {args.dataset_id!r} is unavailable; "
                "run scripts/download_assets.sh before evaluation."
            ) from exc
    mixture_path = dataset_root / suite_to_rlds[args.task_suite_name] / "1.0.0"
    if not mixture_path.is_dir() or not any(mixture_path.iterdir()):
        raise FileNotFoundError(f"Real LIBERO RLDS data is missing or empty: {mixture_path}")



def _load_state_checkpoint(model, checkpoint_path, device="cpu", expected_attack_stage=None):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"State checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"Checkpoint must be a non-empty state_dict mapping: {path}")
    if "model_state_dict" in payload:
        from attacks.badvla import BADVLA_CHECKPOINT_FORMAT

        checkpoint_format = payload.get("format")
        if checkpoint_format == BADVLA_CHECKPOINT_FORMAT:
            if payload.get("attack") != "badvla":
                raise ValueError(f"Invalid BadVLA checkpoint metadata: {path}")
            if expected_attack_stage is not None and payload.get("stage") != expected_attack_stage:
                raise ValueError(
                    f"Expected a BadVLA {expected_attack_stage} checkpoint, got "
                    f"{payload.get('stage')!r}: {path}"
                )
            attack = getattr(model, "attack", None)
            expected_attack_config = {
                "trigger_size": float(getattr(attack, "trigger_size", float("nan"))),
                "loss_p": float(getattr(attack, "loss_p", float("nan"))),
            }
            if payload.get("attack_config") != expected_attack_config:
                raise ValueError(
                    "BadVLA checkpoint attack_config does not match the requested trigger/objective: "
                    f"checkpoint={payload.get('attack_config')!r}, requested={expected_attack_config!r}"
                )
            load_target = model
        elif checkpoint_format == MEMORYVLA_CHECKPOINT_FORMAT:
            if payload.get("attack") != "none":
                raise ValueError(f"Invalid baseline checkpoint metadata: {path}")
            if expected_attack_stage is not None:
                raise ValueError(f"BadVLA requires a stage-tagged checkpoint, got baseline: {path}")
            load_target = getattr(model, "base_model", model)
        else:
            raise ValueError(f"Unsupported project checkpoint format {checkpoint_format!r}: {path}")

        checkpoint_model_config = payload.get("model_config")
        if not isinstance(checkpoint_model_config, dict):
            raise TypeError(f"Checkpoint model_config must be a mapping: {path}")
        # These two fields control training lifecycle but not tensor shapes.
        # Project checkpoints are authoritative so evaluation and Stage II can
        # faithfully inherit an explicit Stage I loader override.
        from utils.dataset import _find_memory_vla

        memory_vla = _find_memory_vla(model)
        checkpoint_loader = checkpoint_model_config.get("dataloader_type")
        checkpoint_group = checkpoint_model_config.get("group_size")
        if checkpoint_loader not in {"group", "stream"}:
            raise ValueError(f"Invalid checkpoint dataloader_type: {checkpoint_loader!r}")
        invalid_group = (
            isinstance(checkpoint_group, bool)
            or not isinstance(checkpoint_group, int)
            or checkpoint_group <= 0
            or (checkpoint_loader == "group" and checkpoint_group <= 1)
        )
        if invalid_group:
            raise ValueError(f"Invalid checkpoint group_size: {checkpoint_group!r}")
        memory_vla.dataloader_type = checkpoint_loader
        memory_vla.group_size = checkpoint_group
        for bank_name in ("cog_mem_bank", "per_mem_bank"):
            bank = getattr(memory_vla, bank_name, None)
            if bank is not None:
                bank.dataloader_type = checkpoint_loader
                bank.group_size = checkpoint_group

        expected_model_config = memoryvla_model_metadata(model)
        if checkpoint_model_config != expected_model_config:
            raise ValueError(
                "Checkpoint model_config is incompatible with the loaded MemoryVLA: "
                f"checkpoint={checkpoint_model_config!r}, loaded={expected_model_config!r}"
            )
        state_dict = payload["model_state_dict"]
    else:
        if expected_attack_stage is not None:
            raise ValueError(
                f"BadVLA requires a stage-tagged checkpoint; legacy raw state_dict is ambiguous: {path}"
            )
        state_dict = payload
        # Standard training saves BaseMemoryVLA directly. A defense-only run
        # wraps that same model in SecureVLA, so load the baseline state into
        # the exact module that originally produced it.
        warnings.warn(
            "Loading a legacy raw baseline state_dict without architecture metadata; "
            "re-save it with this version to make configuration checks fail closed.",
            UserWarning,
            stacklevel=2,
        )
        load_target = getattr(model, "base_model", model)
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError(f"Checkpoint model_state_dict must be a non-empty mapping: {path}")
    load_target.load_state_dict(state_dict, strict=True)
    statistics_path = path.parent / "dataset_statistics.json"
    if statistics_path.is_file():
        from utils.dataset import _find_memory_vla

        with statistics_path.open("r", encoding="utf-8") as handle:
            statistics = json.load(handle)
        if not isinstance(statistics, dict) or not statistics:
            raise TypeError(f"Invalid dataset statistics mapping: {statistics_path}")
        _find_memory_vla(model).norm_stats = statistics


def _build_model(args, device):
    if args.mock:
        print("\n[MOCK MODE] Using lightweight infrastructure-only components.")
        from utils.mock_components import MockBaseMemoryVLA

        base_model = MockBaseMemoryVLA()
    else:
        print(f"Initializing upstream MemoryVLA from {args.model_id} on {device}...")
        from models.base_memory_vla import BaseMemoryVLA

        dtype_name = args.dtype or ("bfloat16" if device.type == "cuda" else "float32")
        if dtype_name == "bfloat16" and device.type != "cuda":
            raise ValueError("MemoryVLA bfloat16 execution requires CUDA; use --dtype float32 on CPU")
        base_model = BaseMemoryVLA(
            model_id_or_path=args.model_id,
            revision=args.revision,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir,
            load_for_training=args.mode == "train",
            dtype=dtype_name,
        )

    security_mode = args.attack != "none" or args.defense != "none"
    if not security_mode:
        model = base_model
    else:
        # Optional security experiments remain isolated from the clean baseline.
        from models.secure_vla import SecureVLA

        attack = None
        if args.attack == "badvla":
            from attacks.badvla import BadVLA

            attack = BadVLA(trigger_size=args.trigger_size, loss_p=args.badvla_loss_p)
        elif args.attack == "dropvla":
            from attacks.dropvla import DropVLA, DropVLAConfig

            attack = DropVLA(
                DropVLAConfig(
                    modality=args.dropvla_modality,
                    protocol=args.dropvla_protocol,
                    episode_poison_rate=args.dropvla_episode_poison_rate,
                    relabel_length=args.dropvla_relabel_length,
                    trigger_alpha=args.dropvla_trigger_alpha,
                    trigger_shape=args.dropvla_trigger_shape,
                    seed=args.seed,
                )
            )
        defense = None
        if args.defense == "amemguard":
            from defenses.amemguard import AMemGuard

            defense_checkpoint = getattr(args, "defense_checkpoint", "")
            if not args.mock:
                if args.attack != "badvla":
                    raise ValueError(
                        "The real defense condition is MemoryVLA + BadVLA + A-MemGuard; "
                        "use --attack badvla with its Stage-II checkpoint."
                    )
                if not defense_checkpoint:
                    raise ValueError(
                        "Real A-MemGuard execution requires --defense_checkpoint; "
                        "no uncalibrated or mock defense is substituted."
                    )
                attack_path = Path(args.attack_checkpoint or args.checkpoint)
                defense = AMemGuard.from_checkpoint(
                    defense_checkpoint,
                    expected_provenance={
                        "model_id": args.model_id,
                        "model_revision": args.revision,
                        "attack_checkpoint_bytes": attack_path.stat().st_size,
                    },
                )
            else:
                defense = AMemGuard(
                    cosine_distance_eps=args.amemguard_cosine_distance_eps,
                    min_cluster_size=args.amemguard_min_cluster_size,
                )
        model = SecureVLA(base_model=base_model, attack=attack, defense=defense)

    checkpoint = args.checkpoint
    attack_checkpoint = getattr(args, "attack_checkpoint", "")
    if checkpoint and attack_checkpoint and Path(checkpoint).resolve() != Path(attack_checkpoint).resolve():
        raise ValueError("--checkpoint and --attack_checkpoint refer to different files")
    if attack_checkpoint:
        if args.mode != "evaluate":
            raise ValueError("--attack_checkpoint is an evaluation-only alias")
        checkpoint = attack_checkpoint
    expected_attack_stage = None
    if args.attack == "badvla":
        if args.mode == "evaluate":
            if not checkpoint:
                raise ValueError(
                    "BadVLA evaluation requires --attack_checkpoint pointing to a real badvla_stage2.pt"
                )
            expected_attack_stage = "stage2"
        elif args.mode == "train" and args.attack_stage == "stage2":
            if not checkpoint:
                raise ValueError("BadVLA --attack_stage stage2 requires a Stage I --checkpoint")
            expected_attack_stage = "stage1"
        elif args.mode == "train" and checkpoint:
            raise ValueError("BadVLA Stage I/both starts from --model_id and does not accept --checkpoint")
    if checkpoint:
        print(f"Loading exact project state_dict from {checkpoint}...")
        _load_state_checkpoint(model, checkpoint, expected_attack_stage=expected_attack_stage)
    return model.to(device)


def main(argv=None):
    args = parse_arguments(argv)
    set_seed(
        args.seed,
        include_tensorflow=(
            not args.mock
            and args.mode != "verify"
            and args.dataset_format == "rlds"
            and getattr(args, "evaluation_type", "offline") == "offline"
        ),
    )

    if args.mode == "train" and args.defense != "none":
        raise ValueError(
            "A-MemGuard is an inference-time memory filter and has no detector-training path"
        )
    if (
        args.mode == "train"
        and args.dataset_format == "rlds"
        and not args.mock
        and args.max_steps is None
    ):
        raise ValueError(
            "Real MemoryVLA RLDS training repeats indefinitely; provide --max_steps "
            "before loading the model."
        )

    if args.mode == "verify":
        if (
            args.attack != "none"
            or args.defense != "none"
            or args.checkpoint
            or args.attack_checkpoint
            or args.defense_checkpoint
        ):
            raise ValueError(
                "--mode verify is a baseline infrastructure check and does not consume "
                "--attack, --defense, or checkpoint arguments"
            )
        print("Running baseline lightweight verification tests...")
        from tests.test_verify import run_cpu_tests

        run_cpu_tests(args)
        return

    _validate_dataset_source(args)
    device = _resolve_device(args.device)
    # Downstream training and evaluation helpers consume args.device directly.
    # Store the canonical form so a numeric shorthand such as --device 1 is
    # consistently interpreted as cuda:1 everywhere.
    args.device = str(device)
    if args.mode == "evaluate" and args.evaluation_type == "simplerenv":
        raise RuntimeError(
            "BadVLA ASR cannot yet be measured: this repository has no real "
            f"{args.evaluation_type} rollout backend. Offline action MSE and memory "
            "rejection rates are not ASR proxies."
        )
    if args.mode == "evaluate" and args.evaluation_type == "libero":
        if args.mock:
            raise ValueError("Real LIBERO evaluation cannot be combined with --mock")
        if args.num_episodes <= 0 or args.max_steps <= 0:
            raise ValueError("--num_episodes and --max_steps must be positive")
        if args.num_steps_wait < 0 or args.action_chunking_window <= 0:
            raise ValueError("--num_steps_wait must be non-negative and --action_chunking_window positive")
        if not 0.0 <= args.poison_rate <= 1.0:
            raise ValueError("--poison_rate must be in [0, 1]")
        _prepare_real_libero_assets(args)
    model = _build_model(args, device)

    if args.mode == "train":
        run_train(model, args)
    elif args.mode == "evaluate":
        if args.evaluation_type == "libero":
            from utils.libero_evaluate import run_libero_evaluate

            run_libero_evaluate(model, args)
        else:
            run_evaluate(model, args)


if __name__ == "__main__":
    main()
