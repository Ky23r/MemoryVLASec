import argparse
import sys


def parse_arguments(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    security_probe = argparse.ArgumentParser(add_help=False)
    security_probe.add_argument("--mode", choices=["train", "evaluate", "verify"], default="evaluate")
    security_probe.add_argument("--attack", choices=["none", "badvla", "dropvla"], default="none")
    security_probe.add_argument("--defense", choices=["none", "amemguard"], default="none")
    security_args, _ = security_probe.parse_known_args(argv)
    parser = argparse.ArgumentParser(
        description="MemoryVLASec: VLA Attack and Defense Framework"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "verify"],
        default="evaluate",
        help="Workflow mode: train, offline action validation, or lightweight verification",
    )

    # Model Paths & Auth
    parser.add_argument(
        "--model_id",
        type=str,
        default="shihao1895/memvla-libero-spatial",
        help="Hugging Face repository ID or local upstream MemoryVLA checkpoint/run directory.",
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
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory for model and dataset downloads.",
    )

    # Dataset Configuration
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help="Hugging Face TFDS/RLDS repository; mutually exclusive with --dataset_path.",
    )
    parser.add_argument(
        "--dataset_format",
        choices=["rlds", "trajectory", "flat"],
        default="rlds",
        help="Input contract: official TFDS/RLDS, local trajectories.jsonl, or legacy flat manifest.jsonl.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Optional dataset configuration/subset name.",
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
        help="Local TFDS root, or a manifest directory for trajectory/flat adapters.",
    )
    if security_args.mode == "train":
        parser.add_argument(
            "--dataloader_type",
            choices=["auto", "group", "stream"],
            default="auto",
            help="RLDS iteration strategy; auto uses the checkpoint's MemoryVLA setting.",
        )
        parser.add_argument("--group_size", type=int, default=None, help="Override the checkpoint's grouped-loader size.")
    parser.add_argument("--shuffle_buffer_size", type=int, default=100000, help="RLDS transition shuffle buffer.")
    parser.add_argument(
        "--future_action_window_size",
        type=int,
        default=None,
        help="Optional contract check; must equal the loaded model's action horizon (normally 15).",
    )
    parser.add_argument(
        "--image_aug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable upstream RLDS image augmentation (defaults on for training and off for evaluation).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the entire pipeline end-to-end with lightweight mock components (Dry-Run mode).",
    )
    if security_args.mode == "train":
        parser.add_argument(
            "--output_dir",
            type=str,
            default="checkpoints",
            help="Directory to save fine-tuned checkpoints.",
        )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=(
            "Strict project checkpoint with architecture metadata; BadVLA checkpoints are also "
            "stage-tagged. Legacy raw baseline state_dicts load with a warning. Optimizer resume "
            "is unsupported."
        ),
    )

    # Model & Training configuration
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (cpu, cuda)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    if security_args.mode == "train":
        parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            help="Training batch size (default: one full checkpoint-defined group, or 1 in stream mode)",
        )
        parser.add_argument(
            "--epochs", type=int, default=1, help="Number of epochs for training"
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-5 if security_args.attack == "badvla" else 2e-5,
            help="Learning rate for fine-tuning (MemoryVLA default: 2e-5; BadVLA adapter: 1e-5)",
        )
        parser.add_argument(
            "--max_steps",
            type=int,
            default=None,
            help="Optimization steps per stage; required for the indefinitely repeating real RLDS train loader.",
        )
        if security_args.attack == "none":
            parser.add_argument(
                "--max_grad_norm",
                type=float,
                default=1.0,
                help="Standard MemoryVLA gradient clipping norm (paper default: 1.0).",
            )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default=None,
        help="Model dtype (default: bfloat16 on CUDA, float32 on CPU).",
    )
    if security_args.mode == "evaluate":
        parser.add_argument(
            "--unnorm_key",
            type=str,
            default=None,
            help="Dataset statistics key used by predict_action for action de-normalization.",
        )
        parser.add_argument("--cfg_scale", type=float, default=1.5, help="Classifier-free guidance scale.")
        parser.add_argument("--use_ddim", action="store_true", help="Use upstream DDIM action sampling.")
        parser.add_argument("--num_ddim_steps", type=int, default=10, help="DDIM sampling steps when enabled.")
        parser.add_argument(
            "--evaluation_type",
            choices=["offline", "libero", "simplerenv"],
            default="offline",
            help=(
                "Offline validation is implemented; real rollout modes fail explicitly "
                "and never substitute MSE/filter rates for ASR."
            ),
        )

    # Attack configuration
    parser.add_argument(
        "--attack",
        type=str,
        choices=["none", "badvla", "dropvla"],
        default="none",
        help="Attack method to apply",
    )

    # Defense configuration
    parser.add_argument(
        "--defense",
        type=str,
        choices=["none", "amemguard"],
        default="none",
        help="Defense method to apply",
    )
    # Security options remain absent from the clean baseline parser/help.
    if security_args.attack == "badvla" and security_args.mode != "verify":
        parser.add_argument(
            "--trigger_size",
            type=float,
            default=0.10,
            help="White square side-length fraction; 0.10 is about 1%% image area, as upstream.",
        )
        parser.add_argument(
            "--badvla_loss_p",
            type=float,
            default=0.5,
            help="Stage I weight on clean/reference cosine consistency.",
        )
        if security_args.mode == "train":
            parser.add_argument(
                "--attack_stage",
                choices=["both", "stage1", "stage2"],
                default="both",
                help="Run both ordered BadVLA stages, Stage I only, or Stage II from --checkpoint.",
            )
            parser.add_argument(
                "--badvla_lr_decay_step",
                type=int,
                default=100000,
                help="Per-stage step at which BadVLA decays the learning rate by 10x.",
            )
    if security_args.attack == "dropvla" and security_args.mode != "verify":
        parser.add_argument("--dropvla_modality", choices=["vision", "text", "joint"], default="vision")
        parser.add_argument("--dropvla_protocol", choices=["paper_faithful", "upstream_legacy"], default="paper_faithful")
        parser.add_argument("--dropvla_episode_poison_rate", type=float, default=0.0031)
        parser.add_argument("--dropvla_relabel_length", type=int, default=8)
        parser.add_argument("--dropvla_trigger_alpha", type=float, default=1.0)
        parser.add_argument("--dropvla_trigger_shape", choices=["circle", "triangle"], default="circle")
    if security_args.defense != "none" and security_args.mode == "evaluate":
        parser.add_argument(
            "--amemguard_cosine_distance_eps",
            type=float,
            default=0.5,
            help=(
                "Cosine-distance radius for the MemoryVLA latent-cluster adapter. "
                "This is not an upstream LLM-auditor confidence threshold."
            ),
        )
        parser.add_argument(
            "--amemguard_min_cluster_size",
            type=int,
            default=2,
            help="Minimum retrieved-memory cluster size (upstream DBSCAN default: 2).",
        )

    return parser.parse_args(argv)
