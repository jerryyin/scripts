<#
.SYNOPSIS
Native Windows credentials: GitHub, Docker, SSH and Codex.
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
    [string]$SshSource,
    [switch]$SkipSshAgent
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
    Write-Output 'PLAN: reuse GitHub/Codex login, or run their interactive login once.'
    Write-Output "PLAN: reuse/clone private vault at $VaultPath; restore Docker registry credential."
    Write-Output 'PLAN: preserve SSH keys, restrict Windows permissions, enable native ssh-agent.'
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

# GitHub CLI uses the Windows credential store. Do not render the Linux hosts.yml
# token template or replace native credentials with a second plaintext copy.
if (!(Get-Command gh -ErrorAction SilentlyContinue)) { throw 'Install GitHub.cli (or run bootstrap.ps1 -Apply) first.' }
$login = Invoke-CredentialCommand gh @('auth', 'status', '--hostname', 'github.com') ''
if ($login.ExitCode -ne 0) {
    Write-Output 'Sign into the GitHub account that can read your private vault and notes.'
    Invoke-Native gh @('auth', 'login', '--hostname', 'github.com', '--git-protocol', 'https', '--web', '--skip-ssh-key')
}

# Clone only when absent; never pull/reset an existing private vault.
$repository = (Read-Json (Join-Path $PSScriptRoot 'defaults/environment.json')).vaultRepository
if (!(Test-Path -LiteralPath (Join-Path $VaultPath '.git'))) {
    Assert-PrivatePath $VaultPath
    if ((Test-Path -LiteralPath $VaultPath) -and @(Get-ChildItem -LiteralPath $VaultPath -Force).Count) {
        throw 'Vault destination is not an empty directory or Git checkout.'
    }
    $probe = Invoke-CredentialCommand git.exe @('-c', 'credential.interactive=never', 'ls-remote', $repository, 'HEAD') '' @{ GIT_TERMINAL_PROMPT = '0'; GCM_INTERACTIVE = 'never' }
    if ($probe.ExitCode -ne 0) { Invoke-Native gh @('auth', 'setup-git', '--hostname', 'github.com') }
    [void][IO.Directory]::CreateDirectory($VaultPath)
    Set-PrivateAcl $VaultPath
    $clone = Invoke-CredentialCommand git.exe @('-c', 'core.autocrlf=false', 'clone', '--config', 'core.autocrlf=false', '--depth', '1', '--', $repository, $VaultPath) '' @{ GIT_TERMINAL_PROMPT = '0'; GCM_INTERACTIVE = 'never' }
    if ($clone.ExitCode -ne 0) { throw 'Private vault clone failed. Check GitHub account/repository access, then rerun.' }
} else {
    # The private vault is intentionally a Git checkout; runtime destinations are not.
    Assert-PrivatePath (Split-Path ([IO.Path]::GetFullPath($VaultPath)) -Parent)
    Assert-PrivatePath $VaultPath -AllowGit
    $remote = Invoke-CredentialCommand git.exe @('-C', $VaultPath, 'remote', 'get-url', 'origin') ''
    $normalizedRemote = $remote.Output.Trim().TrimEnd('/') -replace '^git@github.com:', 'https://github.com/'
    if ($remote.ExitCode -ne 0 -or $normalizedRemote -ne $repository) { throw 'Existing vault origin does not match vaultRepository in defaults/environment.json.' }
    Set-PrivateAcl $VaultPath
}
Set-DockerCredential $VaultPath $dockerConfig

$keys = @(Initialize-NativeSsh $sshPath $SshSource)
if (!$keys.Count) { Write-Output 'SSH: no keys yet. Rerun with -SshSource <trusted .ssh folder>, or create a key with ssh-keygen.' }
elseif (!$SkipSshAgent) {
    $sshBin = Join-Path $env:WINDIR 'System32/OpenSSH'
    $agent = Get-Service ssh-agent -ErrorAction SilentlyContinue
    if (!$agent) { throw 'Install the Windows OpenSSH Client optional feature, then rerun priv.ps1.' }
    if ($agent.Status -ne 'Running' -or $agent.StartType -ne 'Automatic') {
        # Elevate service management only; all credential access stays in this user.
        $command = 'Set-Service ssh-agent -StartupType Automatic; Start-Service ssh-agent; if ((Get-Service ssh-agent).Status -ne "Running") { exit 1 }'
        $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes("`$ErrorActionPreference = 'Stop'; " + $command))
        $serviceProcess = Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList @('-NoProfile', '-EncodedCommand', $encoded)
        if ($serviceProcess.ExitCode -ne 0) { throw 'Could not enable ssh-agent. Rerun, or use -SkipSshAgent.' }
    }
    $sshAdd = Join-Path $sshBin 'ssh-add.exe'
    $loaded = (Invoke-CredentialCommand $sshAdd @('-l') '').Output
    foreach ($key in $keys) {
        $public = $key.FullName + '.pub'
        $fingerprint = if (Test-Path -LiteralPath $public) { (Invoke-CredentialCommand (Join-Path $sshBin 'ssh-keygen.exe') @('-lf', $public) '').Output } else { '' }
        $id = if ($fingerprint) { ($fingerprint -split '\s+')[1] } else { '' }
        if (!$id -or !$loaded.Contains($id)) { Invoke-Native $sshAdd @($key.FullName) }
    }
    $sshCommand = Invoke-CredentialCommand git.exe @('config', '--global', '--get', 'core.sshCommand') ''
    if ($sshCommand.ExitCode -ne 0) { Invoke-Native git @('config', '--global', 'core.sshCommand', ((Join-Path $sshBin 'ssh.exe') -replace '\\', '/')) }
    Write-Output 'OK SSH: native Windows agent enabled; existing keys and config preserved.'
}

if (!(Get-Command codex -ErrorAction SilentlyContinue)) { throw 'Install Codex (or run bootstrap.ps1 -Apply) first.' }
$login = Invoke-CredentialCommand codex @('login', 'status') ''
if ($login.ExitCode -ne 0) { Invoke-Native codex @('login') }
Write-Output 'OK Codex: native login ready. Existing app credentials were preserved.'
