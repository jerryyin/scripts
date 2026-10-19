<#
.SYNOPSIS
    Lock down Windows ~/.ssh files so Microsoft's Windows OpenSSH (and the
    Cursor / VS Code Remote-SSH extension that shells out to it) will accept them.

.DESCRIPTION
    Windows OpenSSH refuses to use ssh config files and private keys that are
    accessible by accounts other than the current user / SYSTEM / Administrators.
    This script strips inherited ACLs and grants Full control to ONLY the current
    user on the relevant files (config, known_hosts, and private keys).

    ------------------------------------------------------------------------
    WHY THIS KEEPS BREAKING (root cause)
    ------------------------------------------------------------------------
    Editing files under C:\Users\<you>\.ssh\ from WSL via the /mnt/c (or
    \\wsl.localhost) mount causes Windows to re-apply *inherited* ACLs. On
    domain machines that re-grants access to extra accounts (e.g. a local
    "<HOST>\AMDAdministrator"), which Windows OpenSSH then rejects.
    Prefer editing these files from the Windows side; if they break, re-run this.

    ------------------------------------------------------------------------
    THE TWO FAILURE MODES THIS FIXES (as seen in Cursor Remote-SSH logs)
    ------------------------------------------------------------------------
    1) Bad ACL on the SSH *config* -> ssh aborts before it ever connects:
         Bad permissions. Try removing permissions for user:
           <HOST>\AMDAdministrator (S-1-5-...-500) on file C:/Users/<you>/.ssh/config
         Bad owner or permissions on C:\Users\<you>/.ssh/config
       Symptom in Cursor: "SSH process exited (code 255) before connection was
       established (after ~700ms)".

    2) Bad ACL on the *private key* -> the key is ignored, auth fails:
         WARNING: UNPROTECTED PRIVATE KEY FILE!
         Permissions for 'C:\Users\<you>/.ssh/id_rsa' are too open.
         This private key will be ignored.
         Load key "...id_rsa": bad permissions
         <user>@host: Permission denied (publickey).
       Symptom in Cursor: "disconnectReason: 'preconnect_failed'".

    NOTE: a *separate*, unrelated issue on some hosts (e.g. the AMD "Conductor
    SUT Authentication" gate on mi350) adds ~7s of latency per SSH connection.
    That is NOT fixed here; mitigate it with "remote.SSH.connectTimeout": 90 in
    Cursor settings.json.

.PARAMETER SshDir
    Directory to fix. Defaults to $env:USERPROFILE\.ssh, or $env:SSH_DIR if set.

.EXAMPLE
    # Native Windows (PowerShell):
    powershell -NoProfile -ExecutionPolicy Bypass -File .\fix-windows-ssh-perms.ps1

.EXAMPLE
    # From WSL, use the sibling wrapper instead:
    ./fix-windows-ssh-perms.sh
#>

$ErrorActionPreference = 'Stop'
# Keep output clean when invoked with redirected streams (e.g. the WSL wrapper),
# otherwise PowerShell serializes the progress/info streams to stderr as CLIXML.
$ProgressPreference = 'SilentlyContinue'

$SshDir = if ($env:SSH_DIR) { $env:SSH_DIR } else { Join-Path $env:USERPROFILE '.ssh' }

if (-not (Test-Path -LiteralPath $SshDir)) {
    Write-Output "ERROR: SSH directory not found: $SshDir"
    exit 1
}

$me = (whoami).Trim()
Write-Output "Locking down SSH files in '$SshDir' for user '$me'"

# Files OpenSSH validates: config, known_hosts, and private keys (id_*, *.pem).
# Restricting public keys too is harmless and keeps things consistent.
$targets = New-Object System.Collections.Generic.List[string]
foreach ($name in @('config', 'known_hosts', 'known_hosts.old')) {
    $p = Join-Path $SshDir $name
    if (Test-Path -LiteralPath $p) { $targets.Add($p) }
}
Get-ChildItem -LiteralPath $SshDir -File |
    Where-Object { $_.Name -match '^id_' -or $_.Extension -eq '.pem' } |
    ForEach-Object { $targets.Add($_.FullName) }

if ($targets.Count -eq 0) {
    Write-Output "No SSH config/key files found in $SshDir; nothing to do."
    exit 0
}

$failed = $false
foreach ($f in ($targets | Sort-Object -Unique)) {
    Write-Output "`n==> $f"
    try {
        # Best-effort: make sure we own it (ignore failure if already owner).
        & icacls $f /setowner $me 2>$null | Out-Null
        # Remove inherited ACEs, then grant ONLY the current user Full control.
        & icacls $f /inheritance:r /grant:r "${me}:F" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "icacls exited $LASTEXITCODE" }
        & icacls $f
    } catch {
        $failed = $true
        Write-Output "WARNING: failed to fix '$f': $_"
    }
}

if ($failed) {
    Write-Output "`nDone, but some files could not be fixed (see warnings above)."
    exit 1
}
Write-Output "`nAll SSH files locked down to '$me'. Retry your Remote-SSH connection."
