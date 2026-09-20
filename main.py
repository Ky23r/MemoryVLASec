import sys
import os
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from utils.args import parse_arguments
from utils.train import run_train
from utils.evaluate import run_evaluate
from models.secure_vla import SecureVLA
from attacks.badvla import BadVLA
from defenses.amemguard import AMemGuard


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_hf_checkpoint(model_id: str, revision: str, hf_token: str) -> str:
    print(f"Resolving checkpoint for {model_id} (revision: {revision})...")
    try:
        # Download the repo into the HF cache
        cached_repo_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=hf_token,
            allow_patterns=[
                "*.json",
                "*.pt",
                "checkpoints/*",
            ],  # Only download what's needed
            ignore_patterns=["*.md", ".git*"],
        )

        # Locate the .pt checkpoint file inside the cached directory
        # The structure is expected to be repo_dir/checkpoints/<ckpt>.pt
        checkpoint_dir = Path(cached_repo_dir) / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Expected 'checkpoints' folder in the downloaded model {model_id}."
            )

        pt_files = list(checkpoint_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {checkpoint_dir}.")

        # If there are multiple, usually 'latest-checkpoint.pt' or the last one is used
        target_ckpt = pt_files[-1]
        print(f"Successfully located cached checkpoint: {target_ckpt}")
        return str(target_ckpt)

    except Exception as e:
        print(f"\n[Error] Failed to load the model from Hugging Face: {model_id}")
        print(f"Details: {str(e)}")
        print(
            "Please ensure the model ID is correct, the revision exists, and you have network access/authentication if it is a private repository."
        )
        sys.exit(1)


def main():
    args = parse_arguments()
    set_seed(args.seed)

    if args.mode == "verify":
        # Offload lightweight CPU tests entirely to the tests directory
        print("Running lightweight CPU tests from tests/test_verify.py...")
        from tests.test_verify import run_cpu_tests

        run_cpu_tests(args)
        return

    if args.mock:
        print("\n[MOCK MODE] Intercepting model and dataset with lightweight mocks for dry-run...")
        from utils.mock_components import MockBaseMemoryVLA
        base_model = MockBaseMemoryVLA()
    else:
        # Real experimental pipeline requires checkpoints and datasets
        if not args.dataset_id and not args.dataset_path:
            print(f"\n[Error] Missing dataset source.")
            print("A real dataset is required for 'train' and 'evaluate' modes.")
            print(
                "Please provide a Hugging Face dataset ID via --dataset_id <id> (recommended) or a local path via --dataset_path <path>"
            )
            sys.exit(1)

        # 1. Handle Hugging Face checkpoint downloading/caching natively
        local_checkpoint_path = download_hf_checkpoint(
            args.model_id, args.revision, args.hf_token
        )

        print(f"Initializing Real MemoryVLA on {args.device}...")
        from models.base_memory_vla import BaseMemoryVLA

        load_for_training = args.mode == "train"

        # 2. Base Model Initialization (Clean Pretrained Weights)
        base_model = BaseMemoryVLA(
            checkpoint_path=local_checkpoint_path, load_for_training=load_for_training
        )

    # Import dynamic quantization utility
    from utils.quantize import quantize_model, apply_lora

    # Apply quantization on CPU BEFORE moving to device
    if args.quantization != "none":
        base_model = quantize_model(base_model, quantization=args.quantization)

    # If training with quantization, or evaluating a LoRA model, apply LoRA
    if args.use_lora or (args.mode == "train" and args.quantization != "none"):
        base_model = apply_lora(base_model)

    # 3. Setup Experimental Configuration (Attack / Defense)
    attack = None
    if args.attack == "badvla":
        attack = BadVLA(trigger_size=args.trigger_size)

    defense = None
    if args.defense == "amemguard":
        defense = AMemGuard(divergence_threshold=args.divergence_threshold)

    # 4. Integrate wrappers
    secure_model = SecureVLA(base_model=base_model, attack=attack, defense=defense)
    secure_model.to(args.device)
    
    if args.load_local_checkpoint and os.path.exists(args.load_local_checkpoint):
        print(f"Loading fine-tuned local state_dict from {args.load_local_checkpoint}...")
        secure_model.load_state_dict(torch.load(args.load_local_checkpoint, map_location=args.device), strict=False)
        
    # 5. Execution Pipeline
    if args.mode == "train":
        # Train ALWAYS starts from the cleanly loaded pretrained weights!
        run_train(secure_model, args)
    elif args.mode == "evaluate":
        run_evaluate(secure_model, args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
