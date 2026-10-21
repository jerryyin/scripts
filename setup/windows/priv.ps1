<#
.SYNOPSIS
Native Windows credentials: GitHub, Docker, SSH, Codex and Claude settings.
.EXAMPLE
.\priv.ps1 -Apply
.EXAMPLE
.\priv.ps1 -Apply -SshSource '\\wsl.localhost\Ubuntu-24.04\home\zyin\.ssh'
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$Status,
    [string]$VaultPath = (Join-Path $env:USERPROFILE 'vault'),
    [string]$RcPath = (Join-Path $env:USERPROFILE 'rc_files'),
    [string]$SshSource,
    [switch]$SkipSshAgent,
    [switch]$RequireSshAgent
)
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Devenv.psm1') -Force -DisableNameChecking
Import-Module (Join-Path $PSScriptRoot 'Credentials.psm1') -Force -DisableNameChecking
if ($Apply -and $Status) { throw 'Choose -Apply or -Status.' }
$sshPath = Join-Path $env:USERPROFILE '.ssh'
$dockerRoot = if ($env:DOCKER_CONFIG) { $env:DOCKER_CONFIG } else { Join-Path $env:USERPROFILE '.docker' }
$dockerConfig = [IO.Path]::GetFullPath((Join-Path $dockerRoot 'config.json'))

if (!$Apply -and !$Status) {
    Write-Output 'PLAN: preserve SSH keys, restrict Windows permissions, enable native ssh-agent.'
    Write-Output "PLAN: reuse/clone private vault at $VaultPath over SSH; apply GitHub CLI and Docker credentials from it."
    Write-Output 'PLAN: report Codex login status; Codex sign-in stays manual.'
    Write-Output 'Use -SshSource to import missing id_* keys from an explicitly chosen backup.'
    Write-Output 'Run -Apply to configure or -Status for a read-only inventory.'
    return
}

if ($Status) {
    foreach ($tool in @('gh', 'codex')) {
        if (!(Get-Command $tool -ErrorAction SilentlyContinue)) { Write-Output "MISSING $tool"; continue }
        $command = if ($tool -eq 'gh') { @('auth', 'status', '--hostname', 'github.com') } else { @('login', 'status') }
        $login = Invoke-CredentialCommand $tool $command ''
        if ($login.ExitCode -eq 0) { Write-Output "OK $tool login" } else { Write-Output "MISSING $tool login" }
    }
    Write-Output "Vault present: $(Test-Path -LiteralPath (Join-Path $VaultPath '.git'))"
    Write-Output "Docker config present: $(Test-Path -LiteralPath $dockerConfig) (not a registry authentication check)"
    $keys = @(Get-ChildItem -LiteralPath $sshPath -File -Filter 'id_*' -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch '\.' })
    Write-Output "SSH private key files: $($keys.Count)"
    $service = Get-Service ssh-agent -ErrorAction SilentlyContinue
    if ($service) { Write-Output "SSH agent: $($service.Status)" } else { Write-Output 'MISSING Windows OpenSSH client/agent' }
    return
}

$keys = @(Initialize-NativeSsh $sshPath $SshSource)
if (!$keys.Count) { Write-Output 'SSH: no keys yet. Rerun with -SshSource <trusted .ssh folder>, or create a key with ssh-keygen.' }
elseif (!$SkipSshAgent) {
    $sshBin = Join-Path $env:WINDIR 'System32/OpenSSH'
    Ensure-SshAgent -ReadyOnly:$RequireSshAgent
    $sshAdd = Join-Path $sshBin 'ssh-add.exe'
    $loaded = (Invoke-CredentialCommand $sshAdd @('-l') '').Output
    foreach ($key in $keys) {
        $public = $key.FullName + '.pub'
        $fingerprint = if (Test-Path -LiteralPath $public) { (Invoke-CredentialCommand (Join-Path $sshBin 'ssh-keygen.exe') @('-lf', $public) '').Output } else { '' }
        $id = if ($fingerprint) { ($fingerprint -split '\s+')[1] } else { '' }
        if (!$id -or !$loaded.Contains($id)) { Invoke-Native $sshAdd @($key.FullName) }
    }
    Write-Output 'OK SSH: native Windows agent enabled; existing keys and config preserved.'
}

# The vault holds the GitHub tokens, so it is cloned with the SSH key rather than a GitHub login.
# Clone when absent; otherwise update it like the other checkouts.
$repository = (Read-Json (Join-Path $PSScriptRoot 'defaults/environment.json')).vaultRepository
if (!(Test-Path -LiteralPath (Join-Path $VaultPath '.git'))) {
    Assert-PrivatePath $VaultPath
    if ((Test-Path -LiteralPath $VaultPath) -and @(Get-ChildItem -LiteralPath $VaultPath -Force).Count) {
        throw 'Vault destination is not an empty directory or Git checkout.'
    }
    $sshRepository = $repository -replace '^https://github.com/', 'git@github.com:'
    [void][IO.Directory]::CreateDirectory($VaultPath)
    Set-PrivateAcl $VaultPath
    $clone = Invoke-CredentialCommand git.exe @('-c', 'core.autocrlf=false', 'clone', '--config', 'core.autocrlf=false', '--depth', '1', '--', $sshRepository, $VaultPath) '' @{ GIT_TERMINAL_PROMPT = '0'; GIT_SSH_COMMAND = (Join-Path $env:WINDIR 'System32/OpenSSH/ssh.exe') + ' -o BatchMode=yes -o StrictHostKeyChecking=accept-new' }
    if ($clone.ExitCode -ne 0) { throw "Private vault clone over SSH failed. Register a GitHub SSH key (or rerun with -SshSource), then rerun." }
} else {
    # The private vault is intentionally a Git checkout; runtime destinations are not.
    Assert-PrivatePath (Split-Path ([IO.Path]::GetFullPath($VaultPath)) -Parent)
    Assert-PrivatePath $VaultPath -AllowGit
    $remote = Invoke-CredentialCommand git.exe @('-C', $VaultPath, 'remote', 'get-url', 'origin') ''
    $normalizedRemote = $remote.Output.Trim().TrimEnd('/') -replace '^git@github.com:', 'https://github.com/'
    if ($remote.ExitCode -ne 0 -or $normalizedRemote -ne $repository) { throw 'Existing vault origin does not match vaultRepository in defaults/environment.json.' }
    Set-PrivateAcl $VaultPath
    Ensure-Checkout $VaultPath $repository
}
# GitHub CLI accounts come from the vault tokens through rc_files' gh credential hook.
Invoke-RcSetup $RcPath @('credentials', '--vault', $VaultPath, '--home', $env:USERPROFILE)
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $login = Invoke-CredentialCommand gh @('auth', 'status', '--hostname', 'github.com') ''
    if ($login.ExitCode -eq 0) { Write-Output 'OK GitHub CLI: vault accounts available.' } else { Write-Output 'MISSING GitHub CLI login: check the vault gh_token_*.txt files.' }
}

# Codex uses single-use OAuth refresh tokens, so it is never signed in unattended.
if (Get-Command codex -ErrorAction SilentlyContinue) {
    $login = Invoke-CredentialCommand codex @('login', 'status') ''
    if ($login.ExitCode -eq 0) { Write-Output 'OK Codex: existing login preserved.' }
    else { Write-Output 'SKIP Codex: not signed in. Run `codex login --device-auth` when you want it.' }
}
