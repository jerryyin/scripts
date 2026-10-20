<#
.SYNOPSIS
Reproduce the native Windows development environment and install WSL if missing.
.EXAMPLE
.\bootstrap.ps1 -Apply
.EXAMPLE
.\bootstrap.ps1 -Check
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$Check,
    [string]$NotesPath,
    [string]$RcPath = (Join-Path $env:USERPROFILE 'rc_files'),
    [string]$VaultPath = (Join-Path $env:USERPROFILE 'vault'),
    [string]$SshSource,
    [string]$TerminalPath,
    [string]$DocumentsPath,
    [string]$EnvironmentPath,
    [switch]$Update,
    [switch]$SkipPackages,
    [switch]$SkipWsl,
    [switch]$SkipFonts,
    [switch]$SkipAddons,
    [switch]$SkipVimPlugins,
    [switch]$SkipCredentials,
    [switch]$RequireDeveloperMode,
    [switch]$InstallCodexDesktop
)
if (!$EnvironmentPath) { $EnvironmentPath = Join-Path $PSScriptRoot 'defaults/environment.json' }
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Devenv.psm1') -Force -DisableNameChecking
if ($Apply -and $Check) { throw 'Choose either -Apply or -Check' }
$manifest = Read-Json $EnvironmentPath
$registryPath = Join-Path $env:APPDATA 'obsidian/obsidian.json'
if (!$NotesPath) { $NotesPath = Get-VaultPath $registryPath (Join-Path $env:USERPROFILE 'notes') }
if (!$TerminalPath) {
    $TerminalPath = Join-Path $env:LOCALAPPDATA 'Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json'
    $unpackaged = Join-Path $env:LOCALAPPDATA 'Microsoft/Windows Terminal/settings.json'
    if (!(Test-Path -LiteralPath $TerminalPath) -and (Test-Path -LiteralPath $unpackaged)) { $TerminalPath = $unpackaged }
}
$stateRoot = Join-Path $env:LOCALAPPDATA 'PersonalDevenv'
$backupRoot = Join-Path $stateRoot ('backups/' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$managedProfile = Join-Path $RcPath 'powershell/.windows/@DOCUMENTS/PowerShell/profile.ps1'
$documents = if ($DocumentsPath) { $DocumentsPath } else { [Environment]::GetFolderPath('MyDocuments') }
if (!$documents) { $documents = Join-Path $env:USERPROFILE 'Documents' }
$terminalSource = Join-Path $RcPath 'terminal/.windows/@LOCALAPPDATA/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json'

if (!$Apply -and !$Check) {
    Write-Output "PLAN: install $($manifest.packages.id -join ', ') with WinGet"
    Write-Output "PLAN: native Windows Vim sources $RcPath/vim/.vimrc; native psmux reuses tmux keys"
    Write-Output 'PLAN: PowerShell 7, Gruvbox colors, font, prompt and shell shortcuts'
    Write-Output 'PLAN: group Developer Mode, WSL, machine packages and SSH-agent service changes under one administrator approval; link configuration as the current user'
    if (!$SkipCredentials) { Write-Output 'PLAN: native GitHub, Docker, SSH and Codex credentials via priv.ps1 before cloning private notes' }
    if (!$SkipWsl) { Write-Output "PLAN: install WSL and $($manifest.wsl.distribution) if missing; initialize Linux separately with scripts/setup/linux/bootstrap.sh" }
    Write-Output "PLAN: notes at $NotesPath; install missing Obsidian plugins and theme chosen in the vault"
    if ($Update) { Write-Output 'PLAN: refresh existing Vim/CoC/Obsidian plugins and the Codex ACP adapter' }
    Write-Output 'Run with -Apply to provision, or -Check for a read-only inventory.'
    return
}

if ($Check) {
    Update-ProcessPath
    $missing = New-Object Collections.ArrayList
    if (Test-DeveloperMode) { Write-Output 'OK Developer Mode' }
    else { [void]$missing.Add('Developer Mode'); Write-Output 'MISSING Developer Mode (required for configuration symlinks)' }
    if (!$SkipWsl) {
        if ((Get-Command wsl.exe -ErrorAction SilentlyContinue) -and @(Get-WslDistributions).Count -gt 0) { Write-Output 'OK WSL (existing distributions; WSL selects its configured default)' }
        else { [void]$missing.Add('WSL'); Write-Output 'MISSING WSL distribution (install or complete first launch)' }
    }
    foreach ($command in @('winget', 'git', 'claude', 'npm')) {
        if (Get-Command $command -ErrorAction SilentlyContinue) { Write-Output "OK $command" }
        else { [void]$missing.Add($command); Write-Output "MISSING $command" }
    }
    foreach ($package in @(Get-RcMissingPackages $RcPath)) {
        $name = if ($package.command) { $package.command } else { $package.winget }
        [void]$missing.Add($name); Write-Output "MISSING rc_files requirement: $name"
    }
    try {
        $nativeVim = Get-NativeVim
        $vimOnPath = Get-Command vim.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1
        if ($vimOnPath.Source -ine $nativeVim) { throw 'Native Vim is not the PATH command' }
        Write-Output "OK vim $nativeVim"
    } catch { [void]$missing.Add('vim'); Write-Output 'MISSING native Vim on PATH (rerun -Apply)' }
    foreach ($path in @($TerminalPath, $managedProfile, (Join-Path $NotesPath '.git'), (Join-Path $RcPath 'vim/.vimrc'), (Join-Path $env:USERPROFILE '.vimrc'), (Join-Path $env:USERPROFILE '.psmux.conf'))) {
        if (Test-Path -LiteralPath $path) { Write-Output "OK $path" }
        else { [void]$missing.Add($path); Write-Output "MISSING $path" }
    }
    if (!$SkipFonts) {
        $fontFace = (Read-Json $terminalSource).profiles.defaults.font.face
        Add-Type -AssemblyName System.Drawing
        $fonts = New-Object Drawing.Text.InstalledFontCollection
        try {
            if ($fonts.Families.Name -contains $fontFace) { Write-Output "OK font $fontFace" }
            else { [void]$missing.Add('font'); Write-Output "MISSING font $fontFace" }
        } finally { $fonts.Dispose() }
    }
    if (Test-Path -LiteralPath $TerminalPath) {
        $terminal = Read-Json $TerminalPath
        $development = @()
        if ($terminal.Contains('profiles') -and $terminal.profiles.Contains('list')) {
            $defaultGuid = (Read-Json $terminalSource).defaultProfile
            $development = @($terminal.profiles.list | Where-Object { $_.Contains('guid') -and $_.guid -eq $defaultGuid })
        }
        if ($development.Count -eq 1) { Write-Output 'OK Terminal default PowerShell profile' }
        else { [void]$missing.Add('Terminal profile'); Write-Output 'MISSING unique Terminal default PowerShell profile' }
    }
    foreach ($problem in @(Get-ConfigurationProblems $env:USERPROFILE $RcPath $documents $TerminalPath)) {
        [void]$missing.Add($problem); Write-Output "DRIFT $problem"
    }
    if (!$SkipAddons) {
        $prefix = Invoke-Native npm.cmd @('prefix','--global') -Capture
        if (!(Test-Path -LiteralPath (Join-Path $prefix 'node_modules/@agentclientprotocol/codex-acp/package.json')) -or !(Test-Path -LiteralPath (Join-Path $prefix 'codex-acp.cmd'))) {
            [void]$missing.Add('Codex ACP adapter'); Write-Output 'MISSING Codex ACP adapter'
        }
    }
    if (!$SkipVimPlugins) {
        $themePath = Join-Path $env:USERPROFILE '.vim/plugged/gruvbox-material/colors/gruvbox-material.vim'
        if (Test-Path -LiteralPath $themePath) { Write-Output 'OK Vim gruvbox-material' }
        else { [void]$missing.Add('Vim theme'); Write-Output 'MISSING Vim gruvbox-material' }
    }
    if (!$SkipAddons -and (Test-Path -LiteralPath (Join-Path $NotesPath '.obsidian/community-plugins.json'))) {
        foreach ($id in @(Read-Json (Join-Path $NotesPath '.obsidian/community-plugins.json'))) {
            foreach ($file in @('manifest.json', 'main.js')) {
                $path = Join-Path $NotesPath ".obsidian/plugins/$id/$file"
                if (!(Test-Path -LiteralPath $path)) { [void]$missing.Add($path); Write-Output "MISSING $path" }
            }
        }
    }
    if (!$SkipAddons -and (Test-Path -LiteralPath (Join-Path $NotesPath '.obsidian/appearance.json'))) {
        $appearance = Read-Json (Join-Path $NotesPath '.obsidian/appearance.json')
        if ($appearance.Contains('cssTheme') -and $appearance.cssTheme) {
            $themePath = Join-Path $NotesPath ".obsidian/themes/$($appearance.cssTheme)/theme.css"
            if (Test-Path -LiteralPath $themePath) { Write-Output "OK Obsidian theme $($appearance.cssTheme)" }
            else { [void]$missing.Add('Obsidian theme'); Write-Output "MISSING Obsidian theme $($appearance.cssTheme)" }
        }
    }
    if ($missing.Count) { Write-Output "Environment has $($missing.Count) missing requirement(s)."; exit 1 }
    Write-Output 'Environment checks passed.'
    return
}

# A separately redirected test home doesn't share the running app's user data.
$nativeSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$nativeProfile = (Get-ItemProperty -LiteralPath ("HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList\" + $nativeSid)).ProfileImagePath
if ($env:USERPROFILE -eq $nativeProfile -and (Get-Process Obsidian -ErrorAction SilentlyContinue)) { throw 'Close Obsidian before applying its settings and plugin files, then rerun.' }
if (!$SkipPackages -and !(Get-Command winget -ErrorAction SilentlyContinue)) { throw 'Install App Installer (WinGet), then rerun.' }
if ($RequireDeveloperMode) {
    if (!(Test-DeveloperMode)) { throw 'This configuration-only run requires Developer Mode to be enabled beforehand; no machine settings were changed.' }
}
Update-ProcessPath
$packages = if ($SkipPackages) { @() } else { @(Get-MissingPackages $manifest.packages) }
# Python is a per-user bootstrap prerequisite; Git is installed in the grouped machine step.
foreach ($package in $packages | Where-Object { !$_.Contains('scope') -or $_.scope -eq 'user' }) { Install-WinGetPackage $package }
Update-ProcessPath
if (Get-Command git -ErrorAction SilentlyContinue) { Ensure-Checkout $RcPath $manifest.rcRepository }
$rcMachine = !$SkipPackages -and !(Test-Path -LiteralPath (Join-Path $RcPath '.git'))
if (!$SkipPackages -and !$rcMachine) {
    $rcMachine = @(Get-RcMissingPackages $RcPath 'machine').Count -gt 0
}
$wslState = if ($SkipWsl) { 'Skipped' } elseif (@(Get-WslDistributions).Count -gt 0) { 'Existing' } else { 'PendingInitialization' }
$sshKeys = @()
if (!$SkipCredentials) {
    foreach ($path in @((Join-Path $env:USERPROFILE '.ssh'), $SshSource) | Where-Object { $_ }) {
        if ($path -eq $SshSource -and !(Test-Path -LiteralPath $path -PathType Container)) { throw 'SSH source directory does not exist.' }
        $sshKeys += @(Get-ChildItem -LiteralPath $path -File -Filter 'id_*' -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch '\.' })
    }
}
$agent = Get-Service ssh-agent -ErrorAction SilentlyContinue
if ($sshKeys.Count -and !$agent) { throw 'Install the Windows OpenSSH Client optional feature before importing SSH keys.' }
$machine = @{
    DeveloperMode=(!(Test-DeveloperMode)); Packages=@($packages | Where-Object { $_.Contains('scope') -and $_.scope -eq 'machine' })
    Distribution=$(if (!$SkipWsl -and $wslState -ne 'Existing') { $manifest.wsl.distribution } else { '' }); WslState=$wslState
    RcMachine=$rcMachine; RcPath=$RcPath; RcRepository=$manifest.rcRepository; SshAgent=($sshKeys.Count -gt 0 -and ($agent.Status -ne 'Running' -or $agent.StartType -ne 'Automatic'))
}
$wslState = Invoke-MachineSetup $machine
Update-ProcessPath
Ensure-Checkout $RcPath $manifest.rcRepository
if (!(Test-Path -LiteralPath (Join-Path $RcPath '.setup/__main__.py'))) { throw 'Update rc_files: .setup/__main__.py is required.' }
if (!$SkipPackages) {
    Invoke-RcSetup $RcPath @('prerequisites')
    Update-ProcessPath
}
Invoke-RcSetup $RcPath @('apply', '--package', 'git', '--home', $env:USERPROFILE, '--backup', $backupRoot)
if (!$SkipCredentials) { & (Join-Path $PSScriptRoot 'priv.ps1') -Apply -VaultPath $VaultPath -RcPath $RcPath -SshSource $SshSource -RequireSshAgent }
Ensure-Checkout $NotesPath $manifest.notesRepository
$deployment = @(Get-DeploymentArguments '' $env:USERPROFILE $RcPath $documents $TerminalPath) + @('--backup', $backupRoot)
if ($SkipPackages) { $deployment += '--no-install' }
if ($SkipFonts) { $deployment += @('--skip-setup', 'terminal') }
if ($SkipVimPlugins) { $deployment += @('--skip-setup', 'vim') }
if ($SkipAddons) { $deployment += @('--skip-setup', 'tmux') }
if ($Update) { $deployment += '--update' }
Invoke-RcSetup $RcPath $deployment
if (!$SkipAddons) {
    Install-ObsidianAddons $NotesPath $backupRoot -Update:$Update
    Install-CodexAdapter -Update:$Update
    # Copilot's device ID lives in Obsidian localStorage. Use its Auto-detect
    # control to select this standard npm location on the first launch.
}
Set-VaultRegistration $registryPath $NotesPath $backupRoot
if ($InstallCodexDesktop) { Invoke-Native codex @('app', $NotesPath) }
Write-Output "Provisioning complete. Backups: $backupRoot"
Write-Output 'Restart Terminal. vim/vi use native Vim; tmux uses psmux; dev attaches to a native main session.'
Write-Output 'Open Obsidian and select Codex Auto-detect if needed. Sign in to Codex/GitHub as needed.'
Write-Output 'Pair KDE Connect with your devices on first use.'
Write-Output 'Run bootstrap.ps1 -Check to inspect readiness.'
if ($wslState -in @('RebootRequired', 'PendingInitialization')) {
    Write-Output "WSL setup pending: restart Windows, launch $($manifest.wsl.distribution) to create your Linux user, then rerun -Apply."
    exit 3010
}
if ($wslState -eq 'Installed') { Write-Output "Launch $($manifest.wsl.distribution) to create your Linux user, then run bash ~/scripts/setup/linux/bootstrap.sh inside Ubuntu." }
