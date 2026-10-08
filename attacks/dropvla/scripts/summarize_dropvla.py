"""Require complete, paired evaluations before declaring a workflow complete."""
import json
from pathlib import Path
import sys


def summarize(root):
    root = Path(root)
    read = lambda path: json.loads((root / path).read_text())
    audit = read("data_audit.json")
    config = read("train/run_config.json")
    plan = read("train/poison_plan_used.json")
    if audit["status"] != "passed" or config["diagnostic_smoke"]:
        raise ValueError("Full run requires passed data audit and production training order")
    rows = [json.loads(line) for line in (root / "train/train_metrics.jsonl").read_text().splitlines()]
    last = rows[-1]
    if last["update"] != config["max_steps"] or last["microstep"] != config["max_steps"] * config["gradient_accumulation_steps"]:
        raise ValueError("Training did not finish its configured update budget")
    if not (root / "train/dropvla.pt").is_file():
        raise FileNotFoundError("Final checkpoint missing")
    names = ("pretrained_clean", "dropvla_clean", "dropvla_trigger")
    evaluations = {name: read(f"evaluation/{name}/results.json") for name in names}
    baseline = evaluations["pretrained_clean"]
    for name, result in evaluations.items():
        expected = len(result["task_ids"]) * result["num_episodes_per_task"]
        keys = [(row["task_id"], row["episode_index"], row["seed"]) for row in result["episodes"]]
        if result["status"] != "complete" or result["completed_episodes"] != expected or len(keys) != expected or len(set(keys)) != expected:
            raise ValueError(f"Incomplete/duplicate evaluation episodes: {name}")
        if result["successes"] != sum(row["success"] for row in result["episodes"]):
            raise ValueError("Success counts disagree with episode records")
        if result["success_rate"] != result["successes"] / expected:
            raise ValueError("Success rate disagrees with episode counts")
        for key in ("task_ids", "num_episodes_per_task", "seed", "inference_config", "max_steps", "model_id", "model_revision", "dataset_id", "dataset_revision", "dataset_config"):
            if result[key] != baseline[key]:
                raise ValueError(f"Conditions differ in {key}")
        if set(keys) != {(r["task_id"], r["episode_index"], r["seed"]) for r in baseline["episodes"]}:
            raise ValueError("Evaluation conditions do not use paired episodes/seeds")
    clean, trigger = evaluations["dropvla_clean"], evaluations["dropvla_trigger"]
    if clean["attack_checkpoint"] != trigger["attack_checkpoint"] or clean["poison_rate"] != 0 or trigger["poison_rate"] != 1:
        raise ValueError("Attack conditions must share a checkpoint with clean=0/trigger=1")
    if clean["dropvla_protocol"] != trigger["dropvla_protocol"] or trigger["dropvla_protocol"]["trigger_mode"] != "post_grasp":
        raise ValueError("Main evaluation requires the same post-grasp protocol in both attack conditions")
    if config["max_steps"] * config["gradient_accumulation_steps"] >= plan["transition_count"]:
        if last["poisoned_source_episodes_seen"] != plan["effective_episode_count"] or last["poisoned_frames_seen"] < plan["poisoned_frame_count"]:
            raise ValueError("Main training did not see all planned poison sources")
    exposed = trigger["dropvla_metrics"]["trigger_exposed_eligible_episodes"]
    summary = {
        "status": "complete", "paired_conditions_verified": True,
        "optimizer_updates": last["update"], "microbatches": last["microstep"],
        "poisoned_frames_seen": last["poisoned_frames_seen"],
        "success_rates": {name: result["success_rate"] for name, result in evaluations.items()},
        "clean_retention_change": clean["success_rate"] - baseline["success_rate"],
        "trigger_success_rate_change": trigger["success_rate"] - clean["success_rate"],
        "clean_dropvla_metrics": clean["dropvla_metrics"],
        "trigger_dropvla_metrics": trigger["dropvla_metrics"],
        "has_eligible_trigger_exposure": exposed > 0,
        "attack_evidence_status": "eligible_exposure_measured" if exposed else "insufficient_grasp_eligibility_ASR_null",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2))
