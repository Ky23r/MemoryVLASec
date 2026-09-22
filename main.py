import json
from pathlib import Path

import torch

from utils.args import parse_arguments
from utils.evaluate import run_evaluate
from utils.train import run_train


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _load_state_checkpoint(model, checkpoint_path, device="cpu", expected_attack_stage=None):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"State checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"Checkpoint must be a non-empty state_dict mapping: {path}")
    if "model_state_dict" in payload:
        from attacks.badvla import BADVLA_CHECKPOINT_FORMAT
        from utils.train import badvla_model_metadata

        if payload.get("format") != BADVLA_CHECKPOINT_FORMAT or payload.get("attack") != "badvla":
            raise ValueError(f"Unsupported attack checkpoint metadata: {path}")
        if expected_attack_stage is not None and payload.get("stage") != expected_attack_stage:
            raise ValueError(
                f"Expected a BadVLA {expected_attack_stage} checkpoint, got {payload.get('stage')!r}: {path}"
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
        expected_model_config = badvla_model_metadata(model)
        if payload.get("model_config") != expected_model_config:
            raise ValueError(
                "BadVLA checkpoint model_config is incompatible with the loaded MemoryVLA: "
                f"checkpoint={payload.get('model_config')!r}, loaded={expected_model_config!r}"
            )
        state_dict = payload["model_state_dict"]
    else:
        if expected_attack_stage is not None:
            raise ValueError(
                f"BadVLA requires a stage-tagged checkpoint; legacy raw state_dict is ambiguous: {path}"
            )
        state_dict = payload
    if not isinstance(state_dict, dict) or not state_dict:
        raise TypeError(f"Checkpoint model_state_dict must be a non-empty mapping: {path}")
    model.load_state_dict(state_dict, strict=True)
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

        dtype_name = args.dtype or (
            "bfloat16" if args.mode == "evaluate" and device.type == "cuda" else "float32"
        )
        if dtype_name == "bfloat16" and device.type != "cuda":
            raise ValueError("MemoryVLA bfloat16 predict_action requires CUDA; use --dtype float32 on CPU")
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
        defense = None
        if args.defense == "amemguard":
            from defenses.amemguard import AMemGuard

            defense = AMemGuard(
                cosine_distance_eps=args.amemguard_cosine_distance_eps,
                min_cluster_size=args.amemguard_min_cluster_size,
            )
        model = SecureVLA(base_model=base_model, attack=attack, defense=defense)

    checkpoint = args.checkpoint
    expected_attack_stage = None
    if args.attack == "badvla":
        if args.mode == "evaluate":
            if not checkpoint:
                raise ValueError("BadVLA evaluation requires --checkpoint pointing to badvla_stage2.pt")
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
    set_seed(args.seed)

    if args.mode == "train" and args.defense != "none":
        raise ValueError(
            "A-MemGuard is an inference-time memory filter and has no detector-training path"
        )

    if args.mode == "verify":
        print("Running baseline lightweight verification tests...")
        from tests.test_verify import run_cpu_tests

        run_cpu_tests(args)
        return

    _validate_dataset_source(args)
    device = _resolve_device(args.device)
    if args.mode == "evaluate" and args.evaluation_type != "offline":
        raise RuntimeError(
            f"{args.evaluation_type} rollout evaluation is not implemented in this repository; "
            "install and run the corresponding upstream environment evaluator instead"
        )
    model = _build_model(args, device)

    if args.mode == "train":
        run_train(model, args)
    elif args.mode == "evaluate":
        run_evaluate(model, args)


if __name__ == "__main__":
    main()
