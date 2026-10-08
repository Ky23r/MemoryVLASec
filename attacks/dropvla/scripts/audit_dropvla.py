"""CPU audit of every production dataset frame and every LIBERO task asset."""
import argparse
import hashlib
import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from attacks.dropvla import DropVLA, DropVLAConfig
from utils.dataset import MemoryVLASampleTransform, _resolve_rlds_root
from utils.dropvla_dataset import DropVLARLDSDataset, build_poison_plan, validate_poison_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--dataset-id", default="shihao1895/libero-rlds")
    parser.add_argument("--dataset-revision", default="92c18c77d610218e838d8c8d4fc6410f3cbe7b18")
    parser.add_argument("--model-id", default="shihao1895/memvla-libero-spatial")
    parser.add_argument("--model-revision", default="4d6572ce289736e459e38a48f8671b557a6fd078")
    parser.add_argument("--data-mix", default="libero_spatial_no_noops")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--num-episodes", type=int, default=20)
    args = parser.parse_args()
    start = time.monotonic()
    source_root = Path(__file__).resolve().parents[1]
    audit_sources = ("attacks/dropvla.py", "utils/dataset.py", "utils/dropvla_dataset.py",
                     "utils/libero_evaluate.py", "scripts/audit_dropvla.py")
    source_hashes = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in audit_sources}
    plan = json.loads(Path(args.plan).read_text())
    attack = DropVLA(DropVLAConfig(protocol="released_repo", seed=plan["seed"],
        episode_poison_rate=plan["episode_poison_rate"], step_poison_rate=plan["step_poison_rate"]))
    root = _resolve_rlds_root(args, args.data_mix)
    validate_poison_plan(plan, root, args.data_mix, attack.config)
    # Recompute selection, labels, fingerprints and normalization independently
    # of the cached plan rather than accepting an old preparation summary.
    if build_poison_plan(root, args.data_mix, attack.config) != plan:
        raise AssertionError("Fresh full-source census differs from saved poison plan")
    from huggingface_hub import hf_hub_download
    statistics_path = hf_hub_download(args.model_id, "dataset_statistics.json",
        revision=args.model_revision, cache_dir=args.cache_dir, local_files_only=True)
    pretrained_stats = json.loads(Path(statistics_path).read_text())[args.data_mix]["action"]
    for key in ("q01", "q99", "mask"):
        np.testing.assert_allclose(pretrained_stats[key], plan["clean_statistics"][args.data_mix]["action"][key], rtol=1e-6, atol=1e-7)
    transform = MemoryVLASampleTransform(lambda image: None, preprocess_image=False)
    dataset = DropVLARLDSDataset(root, args.data_mix, args.plan, attack, transform)
    counts, poison_counts, completed = {}, {}, set()
    previous_id = None
    frames = poisoned = 0
    for sample in dataset:
        eid, timestep = int(sample["episode_ids"][0]), int(sample["timesteps"][0])
        if eid != previous_id:
            if previous_id is not None:
                completed.add(previous_id)
            if eid in completed:
                raise AssertionError("Dataset returned to a completed source episode")
            previous_id = eid
        if timestep != counts.get(eid, 0):
            raise AssertionError(f"Episode {eid}: skipped or reordered timestep {timestep}")
        counts[eid] = timestep + 1
        row = plan["episodes"][eid]
        expected_poison = timestep in row["poison_timesteps"]
        if bool(sample["dropvla_poisoned"]) != expected_poison:
            raise AssertionError("Poison mask differs from plan")
        if sample["image"].size != (224, 224):
            raise AssertionError("Unexpected decoded image size")
        actions = sample["actions"].numpy()
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise AssertionError("Invalid action chunk")
        if not np.isin(actions[:, 6], [0, 1]).all():
            raise AssertionError("Invalid standardized gripper labels")
        if expected_poison:
            if sample["image"].getpixel((10, 10)) != (255, 0, 0) or actions[0, 6] != 1:
                raise AssertionError("Poison marker or open target missing")
            poison_counts[eid] = poison_counts.get(eid, 0) + 1
            poisoned += 1
        frames += 1
        if frames % 10000 == 0:
            print(f"Audited {frames}/{len(dataset)} frames; poison={poisoned}", flush=True)
    if counts != {r["source_episode_id"]: r["length"] for r in plan["episodes"]}:
        raise AssertionError("Production iterator did not emit every source frame")
    if poison_counts != {r["source_episode_id"]: len(r["poison_timesteps"]) for r in plan["episodes"] if r["poison_timesteps"]}:
        raise AssertionError("Production iterator did not emit every poison frame")
    from libero.libero import benchmark, get_libero_path
    from utils.libero_evaluate import _libero_initial_states
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    assets = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl.is_file():
            raise FileNotFoundError(bddl)
        states = _libero_initial_states(task, get_libero_path("init_states"))
        if len(states) < args.num_episodes or not all(np.isfinite(s).all() for s in states):
            raise AssertionError("Missing or invalid LIBERO initial states")
        assets.append({"task_id": task_id, "task": task.name, "initial_states": len(states)})
    if source_hashes != {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in audit_sources}:
        raise RuntimeError("Audit source changed while running; rerun with stable code")
    report = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_sha256": source_hashes,
        "poison_plan_sha256": hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
        "status": "passed", "source_episodes": len(counts), "frames": frames,
        "poisoned_frames": poisoned, "poisoned_episodes": len(poison_counts),
        "fresh_census_matches_plan": True, "task_assets": assets,
        "normalization_matches_pretrained": True,
        "elapsed_seconds": time.monotonic() - start}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
