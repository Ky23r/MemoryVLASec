"""Real post-grasp integration check using an untrained zero-output LoRA reference.

This reference is diagnostic: no optimizer updates, no trained attack claim.
Run with the same exported environment as scripts/_common.sh.
"""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from main import _build_model
from utils.args import parse_arguments
from utils.dataset import _find_memory_vla
from utils.libero_evaluate import run_libero_evaluate
from utils.reproducibility import set_seed


def main():
    output = Path(os.environ["OUTPUT_DIR"])
    if output.exists():
        raise FileExistsError(output)
    args = parse_arguments([
        "--mode", "train", "--attack", "dropvla", "--dataset_format", "rlds",
        "--dataset_id", os.environ["DATASET_ID"], "--model_id", os.environ["MODEL_ID"],
        "--revision", os.environ["MODEL_REVISION"], "--dataset_revision", os.environ["DATASET_REVISION"],
        "--dataset_config", os.environ["DATASET_CONFIG"], "--cache_dir", os.environ["CACHE_DIR"],
        "--dtype", "bfloat16", "--device", os.environ["DEVICE"],
        "--dataloader_type", "stream", "--group_size", "1", "--batch_size", "1", "--no-image_aug",
        "--dropvla_protocol", "released_repo", "--seed", os.environ["SEED"],
        "--dropvla_poison_plan", os.environ["DROPVLA_POISON_PLAN"],
        "--max_steps", "220",
    ])
    set_seed(args.seed, include_tensorflow=False)
    model = _build_model(args, torch.device(args.device))
    args.evaluation_type, args.task_suite_name = "libero", os.environ["TASK_SUITE_NAME"]
    args.num_episodes, args.num_steps_wait, args.action_chunking_window = 1, 10, 8
    args.unnorm_key, args.use_ddim, args.num_ddim_steps, args.cfg_scale = os.environ["UNNORM_KEY"], True, 10, 1.5
    args.dropvla_trigger_mode, args.dropvla_lift_height, args.dropvla_response_window = "post_grasp", .06, 25
    core = _find_memory_vla(model)
    # PEFT Gaussian initialization must yield an exact zero-output adapter.
    adapters = [p for name, p in core.named_parameters() if "lora_B" in name]
    if not adapters or any(torch.count_nonzero(p).item() for p in adapters):
        raise AssertionError("Reference adapters do not have zero output")
    args.attack_checkpoint = "diagnostic_pretrained_with_zero_output_lora_no_updates"
    reports = {}
    for name, rate in (("clean", 0.0), ("post_grasp_trigger", 1.0)):
        args.output_dir, args.poison_rate = str(output / name), rate
        payload = run_libero_evaluate(model, args)
        payload["diagnostic_reference"] = "pretrained MemoryVLA + untrained zero-output LoRA; no optimizer updates"
        (output / name / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
        reports[name] = payload
    clean, triggered = reports["clean"], reports["post_grasp_trigger"]
    if clean["dropvla_metrics"]["eligible_episodes"] < 1 or clean["successes"] < 1:
        raise AssertionError("Clean reference did not exercise grasp eligibility and successful completion")
    if triggered["dropvla_metrics"]["trigger_exposed_eligible_episodes"] < 1:
        raise AssertionError("No real post-grasp trigger exposure was observed")
    if not any(row["triggered_policy_queries"] > 0 for row in triggered["episodes"]):
        raise AssertionError("No actual post-grasp policy query received the trigger")
    print("PASS: real clean grasp and post-grasp trigger policy queries across the suite", flush=True)


if __name__ == "__main__":
    main()
