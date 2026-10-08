"""CPU-only poison-plan preparation; no checkpoint/model is loaded."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from attacks.dropvla import DropVLAConfig
from utils.dataset import _resolve_rlds_root
from utils.dropvla_dataset import build_poison_plan, validate_poison_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--dataset-id", default="shihao1895/libero-rlds")
    parser.add_argument("--dataset-revision", default="92c18c77d610218e838d8c8d4fc6410f3cbe7b18")
    parser.add_argument("--data-mix", default="libero_spatial_no_noops")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-rate", type=float, default=0.05)
    parser.add_argument("--step-rate", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = DropVLAConfig(protocol="released_repo", episode_poison_rate=args.episode_rate,
                           step_poison_rate=args.step_rate, seed=args.seed)
    root = _resolve_rlds_root(SimpleNamespace(
        dataset_path=args.dataset_path, dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision, cache_dir=args.cache_dir,
    ), args.data_mix)
    output = Path(args.output)
    if output.exists() and not args.force:
        plan = json.loads(output.read_text())
        validate_poison_plan(plan, root, args.data_mix, config)
    else:
        plan = build_poison_plan(root, args.data_mix, config)
        validate_poison_plan(plan, root, args.data_mix, config)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(plan, indent=2) + "\n")
        temporary.replace(output)
    print(json.dumps({key: plan[key] for key in (
        "episode_count", "transition_count", "selected_episode_count",
        "effective_episode_count", "poisoned_frame_count",
    )}, indent=2))
    print(f"Poison plan: {output.resolve()}")


if __name__ == "__main__":
    main()
