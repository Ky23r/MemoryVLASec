from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.download_assets import _snapshot_download_with_retry


class DownloadAssetsTest(unittest.TestCase):
    def test_snapshot_retry_is_serial_and_preserves_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "models--owner--model" / "blobs" / "weights.incomplete"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial checkpoint")
            output = io.StringIO()

            with (
                mock.patch.dict(os.environ, {"HF_HUB_DOWNLOAD_ATTEMPTS": "3"}),
                mock.patch(
                    "scripts.download_assets.snapshot_download",
                    side_effect=[TimeoutError("read timed out"), "/cached/snapshot"],
                ) as download,
                mock.patch("scripts.download_assets.time.sleep") as sleep,
                redirect_stdout(output),
                redirect_stderr(output),
            ):
                result = _snapshot_download_with_retry(
                    label="large checkpoint",
                    repo_id="owner/model",
                    cache_dir=directory,
                    token=False,
                )

            self.assertEqual(result, "/cached/snapshot")
            self.assertTrue(partial.is_file())
            self.assertEqual(download.call_count, 2)
            self.assertTrue(all(call.kwargs["max_workers"] == 1 for call in download.call_args_list))
            sleep.assert_called_once_with(10)
            log = output.getvalue()
            self.assertIn("download attempt 1/3", log)
            self.assertIn("resuming 1 cached partial file(s)", log)
            self.assertIn("keeping the cache intact", log)
            self.assertIn("download attempt 2/3", log)


if __name__ == "__main__":
    unittest.main()
