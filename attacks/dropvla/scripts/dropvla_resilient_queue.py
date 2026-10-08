#!/usr/bin/env python3
"""Run a sealed workflow; retry only preflight/evaluation GPU allocation failures."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import fcntl

RESOURCE_ERRORS = ('CUDA out of memory', 'CUDA error: out of memory', 'CUBLAS_STATUS_ALLOC_FAILED')


def workflow_phases(run, inference_only=False):
    if inference_only:
        for phase in ('preflight', 'train'):
            marker = run / '.phases' / (phase + '.done')
            code = run / '.phases' / (phase + '.exit_code')
            if not marker.is_file() or not code.is_file() or code.read_text().strip() != '0':
                raise RuntimeError(f'Inference recovery requires completed {phase} with exit code 0')
        checkpoint = run / 'train/dropvla.pt'
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError('Inference recovery requires the final train/dropvla.pt checkpoint')
        return ('baseline', 'clean', 'trigger', 'summary')
    return ('preflight', 'train', 'baseline', 'clean', 'trigger', 'summary')


def reject_active_phases(run):
    # Also detect older launchers that do not acquire the queue-wide lock.
    for path in (run / '.phases').glob('*.lock'):
        with path.open('r') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Phase is already active: {path.stem}; refusing recovery') from exc


def resource_failure(log):
    # Inspect the terminal traceback: a past OOM mentioned earlier is insufficient.
    lines = log.splitlines()[-80:]
    return any(any(token in line for token in RESOURCE_ERRORS) for line in lines)


def archive_phase(run, phase):
    names = {'preflight': ['smoke', 'smoke_evaluation', 'reference_evaluation', 'smoke.log',
             'reference_evaluation.log', 'smoke_eval_baseline.log', 'smoke_eval_clean.log', 'smoke_eval_trigger.log'],
             'baseline': ['evaluation/pretrained_clean', 'eval_baseline.log'],
             'clean': ['evaluation/dropvla_clean', 'eval_clean.log'],
             'trigger': ['evaluation/dropvla_trigger', 'eval_trigger.log']}[phase]
    existing = [name for name in names if (run / name).exists()]
    if not existing:
        return None
    destination = run / 'resource_retry_history' / (phase + '-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    for name in existing:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        (run / name).rename(target)
    return str(destination)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--stable-seconds', type=int, default=60)
    parser.add_argument('--cooldown-seconds', type=int, default=1800)
    parser.add_argument('--inference-only', action='store_true', help='Require completed training; never run preflight or train')
    parser.add_argument('--check-only', action='store_true', help='Validate prerequisites and snapshot without launching or changing artifacts')
    args = parser.parse_args()
    run, snapshot = args.run.resolve(), args.snapshot.resolve()
    phases = workflow_phases(run, args.inference_only)
    reject_active_phases(run)
    env = dict(os.environ)
    env.update(DEVICE='cuda', GPU_WAIT_INTERVAL_SECONDS='3', GPU_WAIT_TIMEOUT_SECONDS='0',
               MIN_FREE_VRAM_MB=env.get('MIN_FREE_VRAM_MB', '28672'), DROPVLA_SMOKE='0', DROPVLA_SKIP_SMOKE='0')
    if args.inference_only:
        env['CUDA_LAUNCH_BLOCKING'] = '1'
    for key in ('CUDA_VISIBLE_DEVICES', 'MUJOCO_EGL_DEVICE_ID', 'LIBERO_TASK_IDS', 'MEMORYVLASEC_GPU_SELECTED', 'MEMORYVLASEC_GPU_LOCK_FD'):
        env.pop(key, None)
    python = env.get('PYTHON_BIN', str(snapshot / 'memoryvlasec/bin/python'))
    subprocess.run([python, str(snapshot / 'scripts/validate_dropvla_cpu.py'), str(run), '--verify-seal'], env=env, check=True)
    if args.check_only:
        print('Recovery checks passed; phases: ' + ', '.join(phases), flush=True)
        return 0
    # Keep this descriptor alive for the queue lifetime.
    queue_lock = (run / '.phases/queue.lock').open('a')
    try:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('Another queue is already active for this run') from exc
    journal = run / 'resource_retry_events.jsonl'
    statepath = run / 'gpu_cooldowns.json'
    cooldown = json.loads(statepath.read_text()) if statepath.exists() else {}
    # Include the last failed preflight GPU before the first restart.
    previous = run / 'logs/preflight.json'
    if previous.exists():
        summary = json.loads(previous.read_text())
        log = run / 'logs/preflight.log'
        gpu = summary.get('physical_gpu')
        if summary.get('status') == 'failed' and gpu is not None and log.exists() and resource_failure(log.read_text()):
            cooldown[str(gpu)] = max(cooldown.get(str(gpu), 0), time.time() + args.cooldown_seconds)
            statepath.write_text(json.dumps(cooldown, indent=2) + '\n')

    def event(**data):
        data['timestamp'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat()
        with journal.open('a') as output:
            output.write(json.dumps(data) + '\n')
        print(json.dumps(data), flush=True)

    def stable_pool():
        since = {}
        last_report = 0
        while True:
            now = time.time()
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=5)
                free = {int(i): int(v) for i, v in (line.split(',') for line in raw.strip().splitlines())}
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                since.clear()
                if now - last_report >= 30:
                    event(action='gpu_query_failed', error=str(exc)); last_report = now
                time.sleep(3)
                continue
            ready = []
            for gpu, memory in free.items():
                if memory >= int(env['MIN_FREE_VRAM_MB']) and cooldown.get(str(gpu), 0) <= now:
                    since.setdefault(gpu, now)
                    if now - since[gpu] >= args.stable_seconds:
                        ready.append(gpu)
                else:
                    since.pop(gpu, None)
            if ready:
                return ' '.join(str(g) for g in sorted(ready, key=lambda g: free[g], reverse=True))
            if now - last_report >= 30:
                event(action='waiting_stable_gpu', threshold_mib=int(env['MIN_FREE_VRAM_MB']),
                      stable_seconds=args.stable_seconds, free_mib=free, cooldown_until=cooldown)
                last_report = now
            time.sleep(3)

    for phase in phases:
        if (run / '.phases' / (phase + '.done')).exists():
            event(action='skip_completed_phase', phase=phase)
            continue
        while True:
            if phase != 'summary':
                env['GPU_CANDIDATES'] = stable_pool()
            if phase in ('preflight', 'baseline', 'clean', 'trigger'):
                archived = archive_phase(run, phase)
                if archived:
                    event(action='archive_previous_attempt', phase=phase, path=archived)
            event(action='start_phase', phase=phase, gpu_pool=env.get('GPU_CANDIDATES'))
            result = subprocess.run(['bash', str(snapshot / 'scripts/wait_dropvla.sh'), phase, str(run)], env=env)
            if result.returncode == 0:
                break
            log = run / 'logs' / (phase + '.log')
            if result.returncode in (130, 143) or phase in ('train', 'summary') or not log.exists() or not resource_failure(log.read_text()):
                event(action='stop_failure', phase=phase, exit_code=result.returncode,
                      reason='Only preflight/evaluation allocation failures are retried; training never resumes automatically')
                return result.returncode
            summary = json.loads((run / 'logs' / (phase + '.json')).read_text())
            gpu = summary.get('physical_gpu')
            if gpu is None:
                event(action='stop_failure', phase=phase, reason='No GPU recorded for resource failure')
                return result.returncode
            cooldown[str(gpu)] = time.time() + args.cooldown_seconds
            statepath.write_text(json.dumps(cooldown, indent=2) + '\n')
            event(action='retry_resource_failure', phase=phase, gpu=gpu,
                  cooldown_seconds=args.cooldown_seconds)
    event(action='workflow_complete')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
