import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


RUNNER = Path(__file__).resolve().parents[1] / "windows_wsl.py"
SPECIAL = "quotes: '\"; dollar: $HOME $(echo wrong); glob: * ? [x]; tick: `; slash: \\; café 中文"


@unittest.skipUnless(os.name == "nt" or os.environ.get("WSL_DISTRO_NAME"), "requires Windows with WSL, or WSL with PowerShell 7 interop")
class DispatchTest(unittest.TestCase):
    def run_script(self, target: str, source: str, *options: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="cross os '") as directory:
            runner = shutil.copy2(RUNNER, directory)
            script = Path(directory) / "probe script.txt"
            script.write_text(source, encoding="utf-8-sig")
            return subprocess.run(
                [sys.executable, runner, target, str(script), *options],
                capture_output=True, text=True, encoding="utf-8", check=False,
            )

    def test_wsl_literal_text_and_cwd(self) -> None:
        source = "python3 - <<'PY'\nimport json, os\nprint(json.dumps([" + repr(SPECIAL) + ", os.getcwd()], ensure_ascii=False))\nPY\n"
        result = self.run_script("wsl", source.replace("\n", "\r\n"), "--cwd", "/tmp")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [SPECIAL, "/tmp"])

    def test_windows_literal_text_and_cwd(self) -> None:
        literal = SPECIAL.replace("'", "''")
        source = "ConvertTo-Json -Compress -InputObject @('" + literal + "', (Get-Location).Path)"
        result = self.run_script("windows", source, "--cwd", "C:\\Windows")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [SPECIAL, "C:\\Windows"])

    def test_windows_formatted_objects_are_flushed(self) -> None:
        result = self.run_script("windows", "Get-Command Get-Location | Select-Object Name, CommandType")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Get-Location", result.stdout)
        self.assertIn("Cmdlet", result.stdout)

    def test_output_and_explicit_failure(self) -> None:
        scripts = {
            "wsl": "printf 'out\\n'; printf 'err\\n' >&2; exit 23",
            "windows": "[Console]::Out.WriteLine('out'); [Console]::Error.WriteLine('err'); exit 23",
        }
        for target, source in scripts.items():
            with self.subTest(target=target):
                result = self.run_script(target, source)
                self.assertEqual(result.returncode, 23, result.stderr)
                self.assertEqual(result.stdout.strip(), "out")
                self.assertEqual(result.stderr.strip(), "err")

    def test_default_directory_is_target_home(self) -> None:
        scripts = {
            "wsl": 'test "$PWD" = "$HOME"',
            "windows": 'if ((Get-Location).Path -ne $env:USERPROFILE) { throw "wrong directory" }',
        }
        for target, source in scripts.items():
            with self.subTest(target=target):
                result = self.run_script(target, source)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_large_script(self) -> None:
        for target, end in {"wsl": "printf done", "windows": "[Console]::Write('done')"}.items():
            with self.subTest(target=target):
                result = self.run_script(target, "# padding\n" * 5000 + end)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "done")

    def test_invalid_directory_does_not_run_script(self) -> None:
        for target in ("windows", "wsl"):
            with self.subTest(target=target):
                result = self.run_script(target, "echo MUST_NOT_RUN", "--cwd", "relative")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--cwd must be an absolute", result.stderr)
                self.assertNotIn("MUST_NOT_RUN", result.stdout)

    def test_missing_directory_does_not_run_script(self) -> None:
        for target, directory in {"wsl": "/dev/null/not-a-directory", "windows": "C:\\Windows\\System32\\cmd.exe\\not-a-directory"}.items():
            with self.subTest(target=target):
                result = self.run_script(target, "echo MUST_NOT_RUN", "--cwd", directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("MUST_NOT_RUN", result.stdout)

    def test_windows_cmdlet_and_native_failures(self) -> None:
        for source, status in {'throw "intentional failure"': 1, 'cmd.exe /d /c exit 19': 19}.items():
            with self.subTest(source=source):
                result = self.run_script("windows", source)
                self.assertEqual(result.returncode, status, result.stderr)

    def test_wsl_stops_after_failure(self) -> None:
        result = self.run_script("wsl", "false | cat\nprintf MUST_NOT_RUN")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn("MUST_NOT_RUN", result.stdout)

    @unittest.skipUnless(os.environ.get("WSL_DISTRO_NAME"), "checks same-distribution execution from WSL")
    def test_wsl_does_not_reenter_windows(self) -> None:
        # A minimal Linux PATH deliberately excludes wsl.exe and every Windows executable.
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            result = self.run_script("wsl", 'printf direct', "--distro", os.environ["WSL_DISTRO_NAME"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "direct")


if __name__ == "__main__":
    unittest.main()
