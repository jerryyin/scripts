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
    [string]$PythonRoot,
    [string]$EnvironmentPath,
    [switch]$Update,
    [switch]$SkipPackages,
    [switch]$SkipWsl,
    [switch]$SkipFonts,
    [switch]$SkipAddons,
    [switch]$SkipVimPlugins,
    [switch]$SkipCredentials,
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
$managedProfile = Join-Path $stateRoot 'profile.ps1'
$documents = if ($DocumentsPath) { $DocumentsPath } else { [Environment]::GetFolderPath('MyDocuments') }
if (!$documents) { $documents = Join-Path $env:USERPROFILE 'Documents' }

if (!$Apply -and !$Check) {
    Write-Output "PLAN: install $($manifest.packages.id -join ', ') with WinGet"
    Write-Output "PLAN: native Windows Vim sources $RcPath/vim/.vimrc; native psmux reuses tmux keys"
    Write-Output 'PLAN: PowerShell 7, Gruvbox colors, font, prompt and shell shortcuts'
    Write-Output 'PLAN: PowerToys Awake enabled indefinitely; display may sleep (matching existing preferences)'
    if (!$SkipCredentials) { Write-Output 'PLAN: native GitHub, Docker, SSH and Codex credentials via priv.ps1 before cloning private notes' }
    if (!$SkipWsl) { Write-Output "PLAN: install WSL and $($manifest.wsl.distribution) if missing; initialize Linux separately with min.sh" }
    Write-Output "PLAN: notes at $NotesPath; install missing Obsidian plugins and theme chosen in the vault"
    if ($Update) { Write-Output 'PLAN: refresh existing Vim/CoC/Obsidian plugins and the Codex ACP adapter' }
    Write-Output 'Run with -Apply to provision, or -Check for a read-only inventory.'
    return
}

if ($Check) {
    $missing = New-Object Collections.ArrayList
    if (!$SkipWsl) {
        if ((Get-Command wsl.exe -ErrorAction SilentlyContinue) -and (@(Get-WslDistributions) -contains $manifest.wsl.distribution)) { Write-Output "OK WSL $($manifest.wsl.distribution)" }
        else { [void]$missing.Add('WSL'); Write-Output "MISSING WSL $($manifest.wsl.distribution) (install or complete first launch)" }
    }
    foreach ($command in @('winget', 'git', 'gh', 'clangd', 'wt', 'pwsh', 'psmux', 'codex', 'node', 'rg', 'oh-my-posh')) {
        if (Get-Command $command -ErrorAction SilentlyContinue) { Write-Output "OK $command" }
        else { [void]$missing.Add($command); Write-Output "MISSING $command" }
    }
    try { $nativeVim = Get-NativeVim; Write-Output "OK vim $nativeVim" }
    catch { [void]$missing.Add('vim'); Write-Output 'MISSING native Vim' }
    if (Get-Command node -ErrorAction SilentlyContinue) {
        $version = Invoke-Native node @('--version') -Capture
        if ([version]$version.TrimStart('v') -lt [version]'22.19.0') { [void]$missing.Add('Node >=22.19'); Write-Output "OUTDATED Node $version (requires >=22.19)" }
    }
    foreach ($path in @($TerminalPath, $managedProfile, (Join-Path $NotesPath '.git'), (Join-Path $RcPath 'vim/.vimrc'), (Join-Path $env:USERPROFILE '_vimrc'), (Join-Path $env:USERPROFILE '.psmux.conf'))) {
        if (Test-Path -LiteralPath $path) { Write-Output "OK $path" }
        else { [void]$missing.Add($path); Write-Output "MISSING $path" }
    }
    if (!$SkipFonts) {
        Add-Type -AssemblyName System.Drawing
        $fonts = New-Object Drawing.Text.InstalledFontCollection
        try {
            if ($fonts.Families.Name -contains $manifest.font.preferredFace) { Write-Output "OK font $($manifest.font.preferredFace)" }
            else { [void]$missing.Add('font'); Write-Output "MISSING font $($manifest.font.preferredFace)" }
        } finally { $fonts.Dispose() }
    }
    if (Test-Path -LiteralPath $TerminalPath) {
        $terminal = Read-Json $TerminalPath
        $development = @()
        if ($terminal.Contains('profiles') -and $terminal.profiles.Contains('list')) {
            $development = @($terminal.profiles.list | Where-Object { $_.Contains('guid') -and $_.guid -eq '{66fce70d-f58a-45d6-a538-11c34abff4a8}' })
        }
        if ($development.Count -eq 1 -and $development[0].Contains('colorScheme') -and $development[0].colorScheme -eq 'GruvboxMaterialHardDark' -and $development[0].Contains('font') -and $development[0].font.Contains('face') -and $development[0].font.face -eq $manifest.font.preferredFace) {
            Write-Output 'OK Terminal development profile: GruvboxMaterialHardDark and MonacoLigaturized Nerd Font'
        } else { [void]$missing.Add('Terminal preferences'); Write-Output 'MISSING Terminal development profile with the requested font and color scheme' }
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
$wslState = if (!$SkipWsl) { Ensure-Wsl $manifest.wsl.distribution } else { 'Skipped' }
if (!$SkipPackages) { foreach ($package in $manifest.packages) {
    if ($package.Contains('installedCommand') -and (Get-Command $package.installedCommand -CommandType Application -ErrorAction SilentlyContinue)) {
        Write-Host "SKIP $($package.id) (existing native command)"
        continue
    }
    if ($package.Contains('installedAppxName') -and (Get-AppxPackage -Name $package.installedAppxName -ErrorAction SilentlyContinue)) {
        Write-Host "SKIP $($package.id) (Microsoft Store installation)"
        continue
    }
    & winget list --id $package.id --exact --source winget --accept-source-agreements --disable-interactivity | Out-Null
    $packageStatus = $LASTEXITCODE
    if ($packageStatus -eq -1978335212) {
        Invoke-Native winget @('install', '--id', $package.id, '--exact', '--source', 'winget', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity')
    } elseif ($packageStatus -eq 0) { Write-Host "SKIP $($package.id) (installed)" }
    else { throw "WinGet inventory failed for $($package.id) (exit $packageStatus); refusing to treat a query error as a missing package." }
} }
# New installs don't update this PowerShell process's inherited PATH.
Update-ProcessPath
$nodeVersion = Invoke-Native node @('--version') -Capture
if ([version]$nodeVersion.TrimStart('v') -lt [version]'22.19.0') {
    if ($SkipPackages) { throw 'Node >=22.19 is required; supply it on PATH or rerun without -SkipPackages.' }
    Invoke-Native winget @('upgrade', '--id', 'OpenJS.NodeJS.LTS', '--exact', '--source', 'winget', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity')
    Update-ProcessPath
    $nodeVersion = Invoke-Native node @('--version') -Capture
    if ([version]$nodeVersion.TrimStart('v') -lt [version]'22.19.0') { throw 'Node >=22.19 is required by this native configuration. Restart PowerShell and rerun.' }
}
Ensure-Checkout $RcPath $manifest.rcRepository
if (!$SkipCredentials) { & (Join-Path $PSScriptRoot 'priv.ps1') -Apply -VaultPath $VaultPath -SshSource $SshSource }
Ensure-Checkout $NotesPath $manifest.notesRepository
$gitRoot = Split-Path (Split-Path (Get-Command git.exe).Source -Parent) -Parent
$gitBash = Join-Path $gitRoot 'bin/bash.exe'
if (!(Test-Path -LiteralPath $gitBash)) { throw 'Git for Windows Bash could not be found; required by the existing Vim shell commands.' }
$vimExe = Get-NativeVim
$psmuxExe = (Get-Command psmux.exe).Source
$pwshExe = (Get-Command pwsh.exe).Source
if (!$PythonRoot) { $PythonRoot = Invoke-Native py @('-3.14', '-c', 'import sys; print(sys.base_prefix)') -Capture }
if (!(Test-Path -LiteralPath (Join-Path $pythonRoot 'python3.dll'))) { throw 'Python 3.14 runtime DLL could not be located for native Vim.' }

Set-PowerToys (Join-Path $env:LOCALAPPDATA 'Microsoft/PowerToys') (Read-Json (Join-Path $PSScriptRoot 'defaults/powertoys.json')) $backupRoot
$fontFace = $manifest.font.preferredFace
if (!$SkipFonts) { $fontFace = Install-Fonts $manifest.font $backupRoot }
$shellCommand = '"' + $pwshExe + '" -NoLogo'
Set-Terminal $TerminalPath (Read-Json (Join-Path $PSScriptRoot 'defaults/terminal.json')) $shellCommand $fontFace $backupRoot
Save-Json (Join-Path $stateRoot 'state.json') ([ordered]@{ vimExe = $vimExe; psmuxExe = $psmuxExe; rcRoot = $RcPath; notes = $NotesPath }) $backupRoot
foreach ($file in @('profile.ps1', 'prompt.omp.json')) {
    Save-Text (Join-Path $stateRoot $file) ([IO.File]::ReadAllText((Join-Path $PSScriptRoot "defaults/$file"))) $backupRoot
}
Set-NativeVim $env:USERPROFILE $RcPath $gitBash $backupRoot $pythonRoot
Set-NativeCodex $env:USERPROFILE $RcPath $backupRoot
Save-Text (Join-Path $env:USERPROFILE '.psmux.conf') (Get-PsmuxConfiguration (Join-Path $RcPath 'tmux/.tmux.conf')) $backupRoot
foreach ($shell in @('WindowsPowerShell', 'PowerShell')) { Set-ProfileLoader (Join-Path $documents "$shell/profile.ps1") $managedProfile $backupRoot }
if (!$SkipVimPlugins) { Install-VimPlugins $vimExe $env:USERPROFILE $backupRoot $RcPath -Update:$Update }

# Reuse portable Git identity, keeping native credential helpers and local
# includes intact. Linux's /usr/bin/gh helper and SSH rewrites aren't copied.
foreach ($key in @('user.name', 'user.email')) {
    & git config --global --get $key | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $value = Invoke-Native git @('config', '-f', (Join-Path $RcPath 'git/.gitconfig'), '--get', $key) -Capture
        Invoke-Native git @('config', '--global', $key, $value)
    }
}

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
Write-Output 'Open/restart PowerToys to load the Awake preferences. Pair KDE Connect with your devices on first use.'
Write-Output 'Run bootstrap.ps1 -Check to inspect readiness.'
if ($wslState -in @('RebootRequired', 'PendingInitialization')) {
    Write-Output "WSL setup pending: restart Windows, launch $($manifest.wsl.distribution) to create your Linux user, then rerun -Apply."
    exit 3010
}
if ($wslState -eq 'Installed') { Write-Output "Launch $($manifest.wsl.distribution) to create your Linux user, then use your existing min.sh inside Ubuntu." }
