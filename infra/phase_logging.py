"""Stdlib-only phase journal and sampled GPU telemetry; never allocates CUDA."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def artifacts(root, phase):
    """Read only the artifacts belonging to this phase, including partial progress."""
    output = {}
    train_dir = root / ("smoke" if phase == "preflight" else "train")
    if phase in {"preflight", "train"}:
        path = train_dir / "train_metrics.jsonl"
        if path.exists():
            for line in reversed(path.read_text().splitlines()):
                try:
                    output[str(path.relative_to(root))] = json.loads(line)
                    break
                except ValueError:
                    continue
    if phase == "preflight":
        paths = list((root / "smoke_evaluation").glob("*/results.json"))
        paths += list((root / "reference_evaluation").glob("*/results.json"))
    elif phase in {"baseline", "clean", "trigger"}:
        name = {"baseline": "pretrained_clean", "clean": "dropvla_clean", "trigger": "dropvla_trigger"}[phase]
        paths = [root / "evaluation" / name / "results.json"]
    elif phase == "prepare":
        paths = [root / "data_audit.json"]
    elif phase == "summary":
        paths = [root / "summary.json"]
    else:
        paths = []
    for path in paths:
        data = read_json(path)
        if data is None:
            continue
        keep = ("status", "completed_episodes", "num_episodes_per_task", "task_ids", "successes",
                "success_rate", "dropvla_metrics", "peak_allocated_mib", "peak_reserved_mib",
                "elapsed_seconds", "rollout_elapsed_seconds", "progress_percent", "frames", "source_episodes", "poisoned_frames", "poisoned_episodes")
        row = {key: data[key] for key in keep if key in data}
        episodes = data.get("episodes", [])
        queries = sum(ep.get("policy_queries", 0) for ep in episodes)
        row["policy_queries"] = queries
        row["mean_policy_query_seconds"] = (
            sum((ep.get("mean_policy_query_seconds") or 0) * ep.get("policy_queries", 0) for ep in episodes) / queries
            if queries else None)
        if "task_ids" in data:
            row["expected_episodes"] = len(data["task_ids"]) * data["num_episodes_per_task"]
        output[str(path.relative_to(root))] = row
    return output


def descendants(parent_pid):
    children = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    found, pending = {parent_pid}, [parent_pid]
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def gpu_sample(gpu, parent_pid):
    result = {"timestamp_utc": timestamp(), "physical_gpu": gpu}
    try:
        raw = subprocess.check_output([
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=3)
        fields = [field.strip() for field in raw.strip().split(",")]
        if len(fields) != 7:
            raise ValueError("Unexpected GPU telemetry columns")
        result.update(gpu_uuid=fields[0], gpu_name=fields[1])
        for key, value in zip(("device_total_mib", "device_used_mib", "device_free_mib", "utilization_percent", "temperature_c"), fields[2:]):
            result[key] = float(value) if value not in {"N/A", "[N/A]"} else None
        try:
            raw_processes = subprocess.check_output([
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=3)
            pids = descendants(parent_pid)
            result["job_compute_used_mib"] = 0.0
            result["job_gpu_pids"] = []
            for line in raw_processes.splitlines():
                pid, uuid, memory = [value.strip() for value in line.split(",")]
                if int(pid) in pids and uuid == result["gpu_uuid"]:
                    result["job_compute_used_mib"] += float(memory)
                    result["job_gpu_pids"].append(int(pid))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result["job_compute_used_mib"] = None
            result["process_query_error"] = str(exc)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["sample_error"] = str(exc)
    return result


def start(args):
    keys = ("MODEL_ID", "MODEL_REVISION", "DATASET_ID", "DATASET_REVISION", "DATASET_CONFIG",
            "SEED", "DROPVLA_MAX_STEPS", "DROPVLA_GRAD_ACCUM", "NUM_EPISODES", "MAX_STEPS",
            "MIN_FREE_VRAM_MB", "GPU_CANDIDATES", "GPU_WAIT_INTERVAL_SECONDS", "GPU_WAIT_TIMEOUT_SECONDS")
    keys += ("DROPVLA_EPISODE_POISON_RATE", "DROPVLA_STEP_POISON_RATE", "DROPVLA_LORA_RANK",
             "DROPVLA_LORA_ALPHA", "DROPVLA_TRAIN_MEMORY", "DROPVLA_LEARNING_RATE",
             "DROPVLA_HEAD_LEARNING_RATE", "DROPVLA_LR_DECAY_STEP", "DROPVLA_TRIGGER_MODE",
             "NUM_DDIM_STEPS", "ACTION_CHUNKING_WINDOW")
    data = {"phase": args.phase, "run_root": str(args.root), "status": "running",
            "started_at_utc": timestamp(), "started_at_local": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
            "started_monotonic": time.monotonic(), "phase_pid": args.pid, "hostname": socket.gethostname(),
            "settings": {key: os.environ.get(key) for key in keys}, "physical_gpu": None,
            "sampling_interval_seconds": args.interval,
            "logger_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "memory_notes": "device_* includes other jobs; job_compute_* sums this phase's descendant compute processes; sampled peaks may miss transients; PyTorch peaks come from model artifacts"}
    manifest = args.root / "run.env"
    if manifest.exists():
        data["run_env_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    write_json(args.directory / "summary.json", data)
    (args.directory / "resources.jsonl").touch()
    print(f"[{args.phase}] started; logs: {args.directory}", flush=True)


def monitor(args):
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    data = read_json(args.directory / "summary.json")
    data.update(physical_gpu=args.gpu, gpu_selected_at_utc=timestamp(), gpu_selected_monotonic=time.monotonic())
    write_json(args.directory / "summary.json", data)
    previous_progress = None
    while True:
        if not Path(f"/proc/{args.pid}").exists():
            data.update(status="interrupted", ended_at_utc=timestamp(),
                        phase_elapsed_seconds=time.monotonic() - data["started_monotonic"],
                        interruption_reason="phase process disappeared before finalizing logs")
            write_json(args.directory / "summary.json", data)
            break
        sample = gpu_sample(args.gpu, args.pid)
        sample["elapsed_seconds"] = time.monotonic() - data["started_monotonic"]
        sample["progress"] = artifacts(args.root, args.phase)
        with (args.directory / "resources.jsonl").open("a") as handle:
            handle.write(json.dumps(sample) + "\n")
        data["gpu_samples"] = data.get("gpu_samples", 0) + 1
        data["current_sample"] = sample
        for key in ("device_used_mib", "job_compute_used_mib"):
            value = sample.get(key)
            if value is not None:
                peak_key = f"sampled_peak_{key}"
                data[peak_key] = max(data.get(peak_key, 0), value)
        write_json(args.directory / "summary.json", data)
        progress = sample["progress"]
        if progress != previous_progress:
            print(f"[{args.phase}] progress={json.dumps(progress)}; job_vram_mib={sample.get('job_compute_used_mib')}; device_free_mib={sample.get('device_free_mib')}", flush=True)
            previous_progress = progress
        if stopped.wait(args.interval):
            break


def finish(args):
    data = read_json(args.directory / "summary.json")
    elapsed = time.monotonic() - data["started_monotonic"]
    selected = data.get("gpu_selected_monotonic")
    data.update(status="complete" if args.exit_code == 0 else "failed", exit_code=args.exit_code,
                ended_at_utc=timestamp(), ended_at_local=datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
                phase_elapsed_seconds=elapsed,
                gpu_wait_seconds=(selected - data["started_monotonic"] if selected else elapsed if args.phase in {"preflight", "train", "baseline", "clean", "trigger"} else 0),
                execution_after_gpu_selection_seconds=time.monotonic() - selected if selected else None,
                artifacts=artifacts(args.root, args.phase))
    samples_path = args.directory / "resources.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text().splitlines()] if samples_path.exists() else []
    data["gpu_samples"] = len(samples)
    data["gpu_sample_errors"] = sum("sample_error" in row for row in samples)
    for key in ("device_used_mib", "job_compute_used_mib"):
        values = [row[key] for row in samples if row.get(key) is not None]
        data[f"sampled_peak_{key}"] = max(values) if values else None
    data["pytorch_peak_allocated_mib"] = max((row.get("peak_allocated_mib", 0) for row in data["artifacts"].values()), default=0) or None
    data["pytorch_peak_reserved_mib"] = max((row.get("peak_reserved_mib", 0) for row in data["artifacts"].values()), default=0) or None
    write_json(args.directory / "summary.json", data)
    print(f"[{args.phase}] {data['status']}; elapsed={elapsed:.1f}s; wait={data['gpu_wait_seconds']:.1f}s; "
          f"sampled_job_peak={data['sampled_peak_job_compute_used_mib']} MiB; PyTorch_peak={data['pytorch_peak_allocated_mib']} MiB", flush=True)


def stream(args):
    # Keep draining the phase's final exception/status output after Ctrl+C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ansi = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
    console = True
    with args.output.open("a", buffering=1) as handle:
        for line in sys.stdin:
            line = ansi.sub("", line).rstrip("\r\n")
            if not line:
                continue
            rendered = f"[{datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(timespec='seconds')}] {line}\n"
            handle.write(rendered)
            if console:
                try:
                    sys.stdout.write(rendered)
                    sys.stdout.flush()
                except BrokenPipeError:
                    console = False


def main():
    # Monitors/log stream must not keep a phase/GPU flock alive after the job.
    os.closerange(3, 256)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "monitor", "finish", "stream"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be positive")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
