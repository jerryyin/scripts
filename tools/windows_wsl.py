"""Run a noninteractive script on Windows or WSL without nested shell quoting."""

import argparse
import base64
import os
from pathlib import Path, PureWindowsPath
import shlex
import shutil
import subprocess


POWERSHELL_READER = """
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
try {
    $source = [Console]::In.ReadToEnd()
    $global:LASTEXITCODE = 0
    & ([ScriptBlock]::Create($source)) | Out-Default
    if (!$?) { exit 1 }
    exit $global:LASTEXITCODE
} catch {
    [Console]::Error.WriteLine($_.ToString())
    exit 1
}
"""


def executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"{name} is not on PATH; install it or enable WSL Windows interop")
    return path


def wsl_command(distro: str | None) -> list[str]:
    current = os.environ.get("WSL_DISTRO_NAME") if os.name != "nt" else None
    if current and (distro is None or distro == current):
        return ["/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-s"]
    if os.name != "nt" and not current:
        raise RuntimeError("The wsl target requires a Windows or WSL host")
    command = [executable("wsl.exe")]
    if distro:
        command.extend(["--distribution", distro])
    return command + ["--exec", "/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-s"]


def windows_command() -> list[str]:
    encoded = base64.b64encode(POWERSHELL_READER.encode("utf-16-le")).decode("ascii")
    return [executable("pwsh.exe"), "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]


def prepare_source(target: str, source: str, cwd: str | None) -> str:
    if "\0" in source or (cwd and "\0" in cwd):
        raise ValueError("Script text and destination directory must not contain NUL")
    if target == "wsl":
        if cwd is not None and not cwd.startswith("/"):
            raise ValueError("WSL --cwd must be an absolute Linux path")
        directory = shlex.quote(cwd) if cwd is not None else '"$HOME"'
        return f"cd -- {directory}\n" + source
    if cwd is not None and not PureWindowsPath(cwd).is_absolute():
        raise ValueError("Windows --cwd must be an absolute Windows path")
    directory = "'" + cwd.replace("'", "''") + "'" if cwd is not None else "$env:USERPROFILE"
    # SSH/agent processes can retain PATH from before tools were installed.
    refresh_path = "$env:PATH = [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH', 'User') + ';' + $env:PATH\n"
    return refresh_path + f"Set-Location -LiteralPath {directory}\n" + source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("windows", "wsl"))
    parser.add_argument("script", type=Path, help="UTF-8 script on the calling host: PowerShell for Windows, Bash for WSL")
    parser.add_argument("--cwd", help="absolute directory on the destination (default: destination user's home)")
    parser.add_argument("--distro", help="WSL distribution (default: current WSL distribution, or Windows' configured default)")
    args = parser.parse_args()
    if args.distro is not None and args.target != "wsl":
        parser.error("--distro only applies to the wsl target")
    source = prepare_source(args.target, args.script.read_text(encoding="utf-8-sig"), args.cwd)
    command = wsl_command(args.distro) if args.target == "wsl" else windows_command()
    return subprocess.run(command, input=source.encode("utf-8"), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
