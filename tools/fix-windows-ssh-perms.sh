#!/usr/bin/env bash
#
# WSL wrapper for fix-windows-ssh-perms.ps1
#
# Fixes the ACLs on the *Windows-side* ~/.ssh files (config + private keys) that
# Microsoft's Windows OpenSSH (used by Cursor/VS Code Remote-SSH) requires.
#
# Run this from WSL after editing C:\Users\<you>\.ssh\* via /mnt/c, which is the
# usual reason these break. See the .ps1 header for the two failure modes and
# the full root-cause writeup.
#
# Usage:
#   ./fix-windows-ssh-perms.sh
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ps1="$here/fix-windows-ssh-perms.ps1"

if [[ ! -f "$ps1" ]]; then
    echo "error: cannot find $ps1" >&2
    exit 1
fi

# Locate powershell.exe (PATH first, then the standard install location).
psh="$(command -v powershell.exe 2>/dev/null || true)"
if [[ -z "$psh" ]]; then
    psh="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
fi
if [[ ! -x "$psh" ]]; then
    echo "error: powershell.exe not found; run the .ps1 from Windows instead" >&2
    exit 1
fi

# Pass the script via -EncodedCommand (UTF-16LE base64) so we avoid UNC-path and
# execution-policy issues that come from running a \\wsl.localhost script file.
encoded="$(iconv -f UTF-8 -t UTF-16LE "$ps1" | base64 -w0)"

"$psh" -NoProfile -ExecutionPolicy Bypass -EncodedCommand "$encoded"
