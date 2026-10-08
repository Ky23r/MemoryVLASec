"""A validated snapshot must detect source, settings and environment drift."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('dropvla_readiness', Path(__file__).resolve().parents[1] / 'scripts/validate_dropvla_cpu.py')
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


class SnapshotTests(unittest.TestCase):
    def test_seal_rejects_changes_and_wrong_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'source'
            root.mkdir()
            (root / 'main.py').write_text('print(1)\n')
            (root / 'requirements.txt').write_text('torch\n')
            run = Path(directory) / 'run'
            run.mkdir()
            (run / 'run.env').write_text('export SEED=42\n')
            import hashlib
            manifest = {'source_sha256': readiness.hashes(root), 'run_root': str(run),
                        'run_env_sha256': hashlib.sha256((run / 'run.env').read_bytes()).hexdigest(),
                        'environment_versions': {'python': 'test'}}
            (root / 'dropvla_source_manifest.json').write_text(json.dumps(manifest))
            with patch.object(readiness, 'environment_versions', return_value={'python': 'test'}):
                readiness.verify(root, run)
                with self.assertRaisesRegex(RuntimeError, 'another run'):
                    readiness.verify(root, Path(directory) / 'other')
                (root / 'main.py').write_text('print(2)\n')
                with self.assertRaisesRegex(RuntimeError, 'snapshot changed'):
                    readiness.verify(root, run)
                (root / 'main.py').write_text('print(1)\n')
                (run / 'run.env').write_text('export SEED=43\n')
                with self.assertRaisesRegex(RuntimeError, 'settings changed'):
                    readiness.verify(root, run)
                (run / 'run.env').write_text('export SEED=42\n')
            with patch.object(readiness, 'environment_versions', return_value={'python': 'changed'}):
                with self.assertRaisesRegex(RuntimeError, 'versions changed'):
                    readiness.verify(root, run)

    def test_unsealed_source_cannot_be_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'No validated source snapshot'):
                readiness.verify(Path(directory), Path(directory))


if __name__ == '__main__':
    unittest.main()
