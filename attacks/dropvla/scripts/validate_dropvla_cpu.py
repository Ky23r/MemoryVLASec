#!/usr/bin/env python3
"""Check the actual workflow without CUDA; seal a copy only after all checks pass."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('attacks', 'defenses', 'models', 'utils', 'scripts', 'configs', 'tests')


def sources(root):
    files = [root / 'main.py', root / 'requirements.txt']
    files.extend(root / name for name in ('_main_original.py', 'config.env', 'reproduction_manifest.json', 'args.py', 'train.py', 'dataset.py', 'evaluate.py', 'metrics.py') if (root / name).is_file())
    for directory in SOURCE_DIRS:
        files.extend(p for p in (root / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return sorted(files)


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources(root)}


def environment_versions(python):
    code = "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions()}},sort_keys=True))"
    return json.loads(subprocess.check_output([python, '-c', code], text=True, timeout=30))


def verify(root, run):
    manifest = root / 'dropvla_source_manifest.json'
    if not manifest.is_file():
        raise RuntimeError('No validated source snapshot. Run validate_dropvla_cpu.py --seal first.')
    saved = json.loads(manifest.read_text())
    python = os.environ.get('PYTHON_BIN', str(root / 'memoryvlasec/bin/python'))
    if environment_versions(python) != saved['environment_versions']:
        raise RuntimeError('Python/package versions changed after validation.')
    if hashes(root) != saved['source_sha256']:
        raise RuntimeError('Validated source snapshot changed; refusing GPU wait.')
    if str(run.resolve()) != saved['run_root']:
        raise RuntimeError('Source snapshot belongs to another run.')
    for filename, digest in saved.get('audited_inputs_sha256', {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Audited input changed after validation: ' + filename)
    current = hashlib.sha256((run / 'run.env').read_bytes()).hexdigest()
    if current != saved['run_env_sha256']:
        raise RuntimeError('Experiment settings changed after CPU validation.')


def contracts():
    required = {
        'utils/args.py': ('released_repo', '--dropvla_poison_plan', '--dropvla_max_steps', '--dropvla_grad_accum', '--dropvla_trigger_mode'),
        'attacks/dropvla.py': ('released_repo',),
        'main.py': ('configure_dropvla_lora',),
        'utils/train.py': ('assert_dropvla_gradients', 'gradient_accumulation',),
        'utils/libero_evaluate.py': ('DropVLARolloutTracker', 'post_grasp'),
        'scripts/train_dropvla.sh': ('--dataset_format rlds', '--dropvla_poison_plan', '--dropvla_max_steps'),
        'scripts/eval_dropvla.sh': ('DROPVLA_EVAL_CONDITION',),
        'scripts/_common.sh': ('_gpu_wait.sh',),
    }
    missing = []
    for name, tokens in required.items():
        source = (ROOT / name).read_text()
        missing.extend(name + ': missing ' + token for token in tokens if token not in source)
    if missing:
        raise RuntimeError('Workflow/source mismatch:\n' + '\n'.join(missing))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--seal', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--verify-seal', action='store_true')
    parser.add_argument('--contracts-only', action='store_true')
    args = parser.parse_args()
    run = args.run_root.resolve()
    if args.contracts_only:
        contracts()
        print('Workflow source contracts match.')
        return 0
    if args.verify_seal:
        verify(ROOT, run)
        print('Validated source/settings hashes match.')
        return 0
    report = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'run_root': str(run), 'checks': [], 'status': 'failed'}
    before = hashes(ROOT)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', DEVICE='cpu',
                       PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', LIBERO_CONFIG_PATH=str(ROOT / '.cache/memoryvlasec/libero-config'))
    python = os.environ.get('PYTHON_BIN', str(ROOT / 'memoryvlasec/bin/python'))
    logdir = run / 'cpu_validation' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    logdir.mkdir(parents=True)

    def check(name, action):
        try:
            action()
            report['checks'].append({'name': name, 'status': 'passed'})
        except Exception as exc:
            report['checks'].append({'name': name, 'status': 'failed', 'error': str(exc)})
        print(name + ': ' + report['checks'][-1]['status'], flush=True)

    def command(name, argv):
        def execute():
            with (logdir / (name + '.log')).open('w') as output:
                result = subprocess.run(argv, cwd=ROOT, env=environment, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=300)
            if result.returncode:
                raise RuntimeError(f'exit={result.returncode}; see {logdir / (name + ".log")}')
        check(name, execute)

    def settings():
        if not (run / '.phases/prepare.done').is_file():
            raise RuntimeError('prepare not complete')
        audit_path = (args.audit or run / 'data_audit.json').resolve()
        audit = json.loads(audit_path.read_text())
        if audit['status'] != 'passed':
            raise RuntimeError('dataset audit did not pass')
        code_hashes = audit.get('code_sha256')
        if not code_hashes or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in code_hashes.items()):
            raise RuntimeError('Audit is missing current source hashes; rerun the full dataset audit')
        if hashlib.sha256((run / 'poison_plan.json').read_bytes()).hexdigest() != audit.get('poison_plan_sha256'):
            raise RuntimeError('Audit and poison plan do not match')
        names = ['MODEL_ID', 'MODEL_REVISION', 'DATASET_ID', 'DATASET_REVISION',
                 'DATASET_CONFIG', 'CACHE_DIR', 'DROPVLA_PROTOCOL', 'DROPVLA_MODALITY',
                 'DROPVLA_EPISODE_POISON_RATE', 'DROPVLA_STEP_POISON_RATE', 'SEED',
                 'DROPVLA_MAX_STEPS', 'DROPVLA_GRAD_ACCUM', 'DROPVLA_LORA_RANK',
                 'DROPVLA_LORA_ALPHA', 'DROPVLA_TRAIN_MEMORY', 'NUM_EPISODES', 'MAX_STEPS',
                 'DROPVLA_LEARNING_RATE', 'DROPVLA_HEAD_LEARNING_RATE', 'DROPVLA_LR_DECAY_STEP',
                 'DROPVLA_SAVE_INTERVAL', 'DROPVLA_TRIGGER_MODE']
        code = 'import os,json; names=' + repr(names) + '; print(json.dumps({n:os.environ[n] for n in names}))'
        stored = json.loads(subprocess.check_output(['bash', '-eu', '-c',
            'source "$1"; "$2" -c "$3"', 'read-settings', str(run / 'run.env'), python, code],
            env=environment, text=True, timeout=30))
        plan = json.loads((run / 'poison_plan.json').read_text())
        if stored['DROPVLA_PROTOCOL'] != 'released_repo' or stored['DROPVLA_MODALITY'] != 'vision':
            raise RuntimeError('Prepared run does not use released_repo vision')
        for key in ('DROPVLA_MAX_STEPS', 'DROPVLA_GRAD_ACCUM', 'DROPVLA_LORA_RANK', 'DROPVLA_LORA_ALPHA',
                    'NUM_EPISODES', 'MAX_STEPS', 'DROPVLA_LR_DECAY_STEP', 'DROPVLA_SAVE_INTERVAL'):
            if int(stored[key]) < 1:
                raise RuntimeError('Invalid positive integer setting: ' + key)
        for key in ('DROPVLA_LEARNING_RATE', 'DROPVLA_HEAD_LEARNING_RATE'):
            import math
            if not math.isfinite(float(stored[key])) or float(stored[key]) <= 0:
                raise RuntimeError('Invalid learning rate: ' + key)
        if stored['DROPVLA_TRAIN_MEMORY'] not in ('0', '1') or stored['DROPVLA_TRIGGER_MODE'] != 'post_grasp':
            raise RuntimeError('Invalid memory/trigger settings')
        if (int(stored['SEED']) != plan['seed'] or stored['DATASET_CONFIG'] != plan['data_mix'] or
            float(stored['DROPVLA_EPISODE_POISON_RATE']) != plan['episode_poison_rate'] or
            float(stored['DROPVLA_STEP_POISON_RATE']) != plan['step_poison_rate']):
            raise RuntimeError('Prepared settings disagree with poison plan')
        if not (Path(stored['CACHE_DIR']) / '.assets_ready').is_file():
            raise RuntimeError('Asset cache is not ready')
        report['validated_settings'] = stored
        if (run / 'train/train_metrics.jsonl').exists():
            raise RuntimeError('Partial training exists; optimizer resume is unsupported')
    check('prepared_data', settings)
    check('workflow_contracts', contracts)

    def syntax():
        for p in sources(ROOT):
            if p.suffix == '.py':
                ast.parse(p.read_text(), filename=str(p))
            elif p.suffix == '.sh':
                subprocess.run(['bash', '-n', str(p)], check=True, capture_output=True)
    check('source_syntax', syntax)
    def disk_space():
        if shutil.disk_usage(run).free < 70 * 2**30:
            raise RuntimeError('Need >=70 GiB free for smoke, final/latest and atomic temporary checkpoints')
    check('checkpoint_disk_capacity', disk_space)
    command('dependency_check', [python, '-m', 'pip', 'check'])
    code = "import main,inspect; from vla.memory_vla import MemoryVLA; from prismatic.models.vlms import PrismaticVLM; from pathlib import Path; root=Path(main.__file__).resolve().parent; paths=[Path(inspect.getfile(c)).resolve() for c in (MemoryVLA,PrismaticVLM)]; print(paths); assert all(p.is_relative_to(root/'models/core') for p in paths)"
    command('core_import_paths', [python, '-c', code])
    def model_assets():
        stored = report.get('validated_settings')
        if stored is None:
            raise RuntimeError('Run settings validation failed')
        code = """import json,sys,torch
from pathlib import Path
from huggingface_hub import hf_hub_download
settings=json.loads(sys.argv[1])
paths={name:hf_hub_download(settings['MODEL_ID'],name,revision=settings['MODEL_REVISION'],cache_dir=settings['CACHE_DIR'],local_files_only=True) for name in ('config.json','dataset_statistics.json','checkpoints/memvla-libero-spatial.pt')}
config=json.loads(Path(paths['config.json']).read_text())
assert config['action_dim']==7 and config['future_action_window_size']==15
state=torch.load(paths['checkpoints/memvla-libero-spatial.pt'],map_location='cpu',weights_only=True,mmap=True)['model']
assert all(name in state and state[name] for name in ('per_compr','cog_mem_bank','per_mem_bank','action_model','projector','llm_backbone','vision_backbone'))
print(json.dumps({'checkpoint_bytes':Path(paths['checkpoints/memvla-libero-spatial.pt']).stat().st_size,'modules':list(state),'action_dim':config['action_dim'],'future_action_window_size':config['future_action_window_size']}))
"""
        with (logdir / 'model_assets.log').open('w') as output:
            result = subprocess.run([python, '-c', code, json.dumps(stored)], env=environment,
                                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            raise RuntimeError('Local pretrained assets invalid; see model_assets.log')
    check('model_assets', model_assets)
    command('pipeline_tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_dropvla_pipeline.py', '-v'])
    command('scheduler_tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_gpu_wait.py', '-v'])
    command('guard_tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_dropvla_readiness.py', '-v'])
    command('amemguard_tests', [python, '-m', 'pytest', '-q', 'tests/test_amemguard.py'])
    command('logging_tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_phase_logging.py', '-v'])
    if before != hashes(ROOT):
        report['checks'].append({'name': 'source_stability', 'status': 'failed', 'error': 'Source changed during validation'})
    if all(c['status'] == 'passed' for c in report['checks']):
        report['status'] = 'passed'
        if args.seal:
            destination = args.seal.resolve()
            if destination.exists():
                raise RuntimeError('Snapshot destination already exists')
            destination.mkdir(parents=True)
            for p in sources(ROOT):
                target = destination / p.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
            for name in ('.cache', 'memoryvlasec'):
                (destination / name).symlink_to(ROOT / name, target_is_directory=True)
            # Assets and interpreter are shared; application source is a separate copy.
            manifest = {'source_sha256': before, 'run_root': str(run),
                        'run_env_sha256': hashlib.sha256((run / 'run.env').read_bytes()).hexdigest(),
                        'cpu_report': str(logdir / 'report.json'),
                        'environment_versions': environment_versions(python),
                        'audited_inputs_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (run / 'poison_plan.json', args.audit or run / 'data_audit.json')}}
            (destination / 'dropvla_source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            if hashes(destination) != before:
                raise RuntimeError('Source changed during snapshot creation')
            for p in sources(destination):
                p.chmod(p.stat().st_mode & ~0o222)
            report['snapshot_root'] = str(destination)
    (logdir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Report: ' + str(logdir / 'report.json'))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        sys.exit(1)
