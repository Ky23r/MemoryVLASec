import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import fcntl
spec = importlib.util.spec_from_file_location('retry', Path(__file__).resolve().parents[1] / 'scripts/dropvla_resilient_queue.py')
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)

class ResourceRetryTests(unittest.TestCase):
    def completed_run(self, run):
        (run / '.phases').mkdir()
        for phase in ('preflight', 'train'):
            (run / '.phases' / (phase + '.done')).write_text('complete')
            (run / '.phases' / (phase + '.exit_code')).write_text('0\n')
        (run / 'train').mkdir()
        (run / 'train/dropvla.pt').write_bytes(b'trained weights')

    def test_inference_recovery_never_includes_training(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.completed_run(run)
            self.assertEqual(retry.workflow_phases(run, True), ('baseline', 'clean', 'trigger', 'summary'))
            for phase in ('preflight', 'train'):
                code = run / '.phases' / (phase + '.exit_code')
                code.write_text('1')
                with self.assertRaises(RuntimeError):
                    retry.workflow_phases(run, True)
                code.write_text('0')
            (run / 'train/dropvla.pt').unlink()
            with self.assertRaises(RuntimeError):
                retry.workflow_phases(run, True)

    def test_recovery_refuses_an_active_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.completed_run(run)
            with (run / '.phases/trigger.lock').open('w') as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    retry.reject_active_phases(run)
            retry.reject_active_phases(run)

    def test_check_only_verifies_seal_without_launching_or_archiving(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.completed_run(run)
            argv = ['queue', str(run), str(run), '--inference-only', '--check-only']
            with patch('sys.argv', argv), patch.object(retry.subprocess, 'run') as invoke:
                self.assertEqual(retry.main(), 0)
                invoke.assert_called_once()
                self.assertIn('--verify-seal', invoke.call_args.args[0])
                self.assertEqual(invoke.call_args.kwargs['env']['CUDA_LAUNCH_BLOCKING'], '1')
            self.assertFalse((run / 'resource_retry_events.jsonl').exists())
            self.assertFalse((run / 'resource_retry_history').exists())
            self.assertEqual((run / 'train/dropvla.pt').read_bytes(), b'trained weights')

    def test_only_allocation_failures_are_retryable(self):
        for error in ('CUDA out of memory', 'CUDA error: out of memory', 'CUBLAS_STATUS_ALLOC_FAILED'):
            self.assertTrue(retry.resource_failure('RuntimeError: ' + error))
        for error in ('ValueError: invalid dataset', 'CUDA error: illegal memory access', 'unbound variable'):
            self.assertFalse(retry.resource_failure(error))
        self.assertFalse(retry.resource_failure('CUDA out of memory\n' + 'normal output\n' * 90))

    def test_archive_preserves_history_and_leaves_main_training_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for name in ('smoke', 'smoke_evaluation', 'reference_evaluation', 'train'):
                (run / name).mkdir(); (run / name / 'evidence').write_text(name)
            destination = Path(retry.archive_phase(run, 'preflight'))
            for name in ('smoke', 'smoke_evaluation', 'reference_evaluation'):
                self.assertEqual((destination / name / 'evidence').read_text(), name)
                self.assertFalse((run / name).exists())
            self.assertEqual((run / 'train/evidence').read_text(), 'train')
            self.assertIsNone(retry.archive_phase(run, 'preflight'))
            with self.assertRaises(KeyError):
                retry.archive_phase(run, 'train')
