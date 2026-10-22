import contextlib
import io
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import requests


TOOL = Path(__file__).resolve().parents[1] / "github_attachments.py"


class AttachmentTests(unittest.TestCase):
    def test_standalone_cli_accepts_an_issue_in_any_repository(self):
        for issue in ("https://github.com/iree-org/iree/issues/123", "https://github.com/example/another/issues/456"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory:
                copied = Path(shutil.copy2(TOOL, directory))
                output = Path(directory) / "attachments"
                attachment = "https://github.com/user-attachments/files/123/test.txt"
                page = mock.Mock(text=f'<a href="{attachment}">file</a><a href="/other">ignore</a>')
                data = mock.Mock(content=b"fixture attachment\n")
                with mock.patch.object(sys, "argv", [str(copied), issue, "--output-dir", str(output)]), \
                        mock.patch("requests.get", side_effect=[page, data]) as get, \
                        contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(copied), run_name="__main__")
                self.assertEqual(get.call_args_list, [mock.call(issue), mock.call(attachment)])
                self.assertEqual((output / "test.txt").read_bytes(), b"fixture attachment\n")

    def test_http_failure_is_not_reported_as_success(self):
        namespace = runpy.run_path(str(TOOL))
        response = mock.Mock()
        response.raise_for_status.side_effect = requests.HTTPError("fixture HTTP failure")
        with tempfile.TemporaryDirectory() as directory, mock.patch("requests.get", return_value=response):
            output = Path(directory) / "attachments"
            with self.assertRaises(requests.HTTPError):
                namespace["download_attachments"]("https://github.com/example/repo/issues/1", str(output))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
