import argparse
import sys


def parse_arguments(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    security_probe = argparse.ArgumentParser(add_help=False)
    security_probe.add_argument("--mode", choices=["train", "evaluate", "verify"], default="evaluate")
    security_probe.add_argument("--attack", choices=["none", "badvla", "dropvla"], default="none")
    security_probe.add_argument("--defense", choices=["none", "amemguard_latent"], default="none")
    security_probe.add_argument(
        "--evaluation_trigger", choices=["none", "badvla"], default="none"
    )
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints" if security_args.mode == "train" else "output",
        help="Directory for training checkpoints or evaluation results.",
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
    parser.add_argument(
        "--attack_checkpoint",
        type=str,
        default="",
        help="Trained attack checkpoint (an explicit alias for --checkpoint in evaluation mode).",
    )
    # Model & Training configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cpu, cuda, cuda:<index>, or a GPU index such as 0)",
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
            "--epochs",
            type=int,
            default=10 if security_args.attack == "badvla" else 1,
            help="Training epoch bound (BadVLA default: 10; other training: 1)",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=2e-5,
            help="MemoryVLA/DropVLA learning rate. BadVLA uses its stage-specific rates.",
        )
        parser.add_argument(
            "--max_steps",
            type=int,
            default=None,
            help=(
                "Optimization steps for clean MemoryVLA/DropVLA. BadVLA uses its two "
                "stage-specific step budgets."
            ),
        )
        if security_args.attack == "none":
            parser.add_argument(
                "--max_grad_norm",
                type=float,
                default=1.0,
                help="Standard MemoryVLA gradient clipping norm (paper default: 1.0).",
            )
            parser.add_argument(
                "--repeated_diffusion_steps",
                type=int,
                default=4,
                help="Repeated diffusion-noise samples per training item (MemoryVLA paper: 4).",
            )
            parser.add_argument(
                "--global_batch_size",
                type=int,
                default=256,
                help="Effective batch size after gradient accumulation (MemoryVLA paper: 256).",
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
            help="Offline action validation or a real in-process LIBERO simulator rollout.",
        )
        parser.add_argument(
            "--task_suite_name",
            choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
            default="libero_spatial",
            help="LIBERO benchmark suite used for real simulator rollouts.",
        )
        parser.add_argument(
            "--num_episodes",
            type=int,
            default=50,
            help="Number of initial-state rollouts per LIBERO task (paper protocol: 50).",
        )
        parser.add_argument(
            "--max_steps",
            type=int,
            default=None,
            help="Maximum simulator steps; omitted uses the official suite-specific limit.",
        )
        parser.add_argument(
            "--num_steps_wait",
            type=int,
            default=10,
            help="Initial no-op steps used to let LIBERO objects stabilize.",
        )
        parser.add_argument(
            "--action_chunking_window",
            type=int,
            default=8,
            help="Number of actions from each MemoryVLA prediction to execute before replanning.",
        )
        parser.add_argument(
            "--poison_rate",
            type=float,
            default=1.0,
            help="Deterministic fraction of attack rollout episodes that receive the trigger.",
        )
        parser.add_argument(
            "--evaluation_trigger",
            choices=["none", "badvla"],
            default="none",
            help=(
                "Apply a trigger independently of checkpoint poisoning. This is required for the "
                "benign-reference triggered arm in the four-rate BadVLA ASR protocol."
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
        choices=["none", "amemguard_latent"],
        default="none",
        help=(
            "Defense adapter. amemguard_latent is an explicitly named MemoryVLA latent-space "
            "adaptation, not the paper's textual LLM-as-a-judge implementation."
        ),
    )
    # Security options remain absent from the clean baseline parser/help.
    if (
        security_args.attack == "badvla"
        or security_args.evaluation_trigger == "badvla"
    ) and security_args.mode != "verify":
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
        if security_args.mode == "train" and security_args.attack == "badvla":
            parser.add_argument(
                "--attack_stage",
                choices=["stage1", "stage2"],
                default="stage1",
                help=(
                    "Run one BadVLA stage. Stage II must be a separate invocation loading "
                    "the merged Stage-I checkpoint so each stage uses its required loader."
                ),
            )
            parser.add_argument("--badvla_stage1_learning_rate", type=float, default=5e-4)
            parser.add_argument("--badvla_stage2_learning_rate", type=float, default=5e-5)
            parser.add_argument("--badvla_stage1_max_steps", type=int, default=5000)
            parser.add_argument("--badvla_stage2_max_steps", type=int, default=30000)
            parser.add_argument("--badvla_stage1_lr_decay_step", type=int, default=1000)
            parser.add_argument("--badvla_stage2_lr_decay_step", type=int, default=10000)
            parser.add_argument("--badvla_stage1_lora_rank", type=int, default=4)
            parser.add_argument("--badvla_stage2_lora_rank", type=int, default=8)
            parser.add_argument("--badvla_stage1_lora_alpha", type=float, default=4.0)
            parser.add_argument("--badvla_stage2_lora_alpha", type=float, default=8.0)
            parser.add_argument("--badvla_lora_dropout", type=float, default=0.0)
    if security_args.attack == "dropvla" and security_args.mode != "verify":
        parser.add_argument("--dropvla_modality", choices=["vision", "text", "joint"], default="vision")
        parser.add_argument("--dropvla_protocol", choices=["paper_faithful", "upstream_legacy"], default="paper_faithful")
        parser.add_argument("--dropvla_episode_poison_rate", type=float, default=0.0031)
        parser.add_argument("--dropvla_relabel_length", type=int, default=8)
        parser.add_argument("--dropvla_trigger_alpha", type=float, default=1.0)
        parser.add_argument("--dropvla_trigger_shape", choices=["circle", "triangle"], default="circle")
    if security_args.defense != "none" and security_args.mode == "evaluate":
        parser.add_argument(
            "--amemguard_divergence_threshold",
            type=float,
            default=0.10,
            help=(
                "Cosine distance from the query-conditioned path centroid. This selects the "
                "paper's embedding-distance ablation for the latent MemoryVLA adaptation."
            ),
        )
        parser.add_argument(
            "--amemguard_top_k",
            type=int,
            default=4,
            help="Primary/lesson memory retrieval depth (A-MemGuard paper default: 4).",
        )
        parser.add_argument(
            "--amemguard_lesson_capacity", type=int, default=256,
            help="Maximum negative latent reasoning paths retained per MemoryVLA bank.",
        )
        parser.add_argument(
            "--amemguard_lesson_similarity_threshold", type=float, default=0.90,
            help=(
                "Cosine threshold for the adapted latent lesson-template check. This has no "
                "direct textual A-MemGuard equivalent and is reported in result metadata."
            ),
        )

    return parser.parse_args(argv)
