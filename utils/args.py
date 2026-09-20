import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MemoryVLASec: VLA Attack and Defense Framework"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "verify"],
        default="evaluate",
        help="Workflow mode: train (Phase I & II), evaluate (environment rollouts), or verify (CPU lightweight testing)",
    )

    # Model Paths & Auth
    parser.add_argument(
        "--model_id",
        type=str,
        default="shihao1895/memvla-libero-spatial",
        help="Hugging Face repository ID for the real MemoryVLA pretrained checkpoint. Required for train/evaluate.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face repository revision (branch, tag, or commit hash).",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face authentication token for private repositories or rate limits.",
    )

    # Dataset Configuration
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="shihao1895/libero-rlds",
        help="Hugging Face dataset ID to download and cache (e.g., 'shihao1895/libero-rlds'). Required for train/evaluate if dataset_path is not used.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional dataset configuration/subset name.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to load (e.g., 'train', 'test', 'validation').",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional dataset revision/version.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Optional fallback local path to the dataset. HF dataset_id is preferred.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the entire pipeline end-to-end with lightweight mock components (Dry-Run mode).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save fine-tuned checkpoints.",
    )
    parser.add_argument(
        "--load_local_checkpoint",
        type=str,
        default="",
        help="Path to a fine-tuned local state_dict to load on top of the base model (e.g., for evaluating BadVLA).",
    )

    # Model & Training configuration
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (cpu, cuda)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for real data processing"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "8bit", "4bit"],
        default="8bit",
        help="Quantization precision for the pretrained model to save VRAM.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA (PEFT) for parameter-efficient fine-tuning (required if quantizing and training).",
    )

    # Attack configuration
    parser.add_argument(
        "--attack",
        type=str,
        choices=["none", "badvla"],
        default="badvla",
        help="Attack method to apply",
    )
    parser.add_argument(
        "--trigger_size", type=float, default=0.05, help="Size of the backdoor trigger"
    )
    parser.add_argument(
        "--poisoning_rate",
        type=float,
        default=0.5,
        help="Proportion of the batch to poison during Phase II training (0.0 to 1.0)",
    )

    # Defense configuration
    parser.add_argument(
        "--defense",
        type=str,
        choices=["none", "amemguard"],
        default="amemguard",
        help="Defense method to apply",
    )
    parser.add_argument(
        "--divergence_threshold",
        type=float,
        default=0.5,
        help="Divergence threshold for A-MemGuard consensus",
    )

    return parser.parse_args()
