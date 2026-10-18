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
    Write-Output 'PLAN: enable Developer Mode (one administrator prompt if disabled) and link authored configuration to rc_files'
    Write-Output 'PLAN: PowerToys Awake enabled indefinitely; display may sleep (matching existing preferences)'
    if (!$SkipCredentials) { Write-Output 'PLAN: native GitHub, Docker, SSH and Codex credentials via priv.ps1 before cloning private notes' }
    if (!$SkipWsl) { Write-Output "PLAN: install WSL and $($manifest.wsl.distribution) if missing; initialize Linux separately with min.sh" }
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
        if ((Get-Command wsl.exe -ErrorAction SilentlyContinue) -and (@(Get-WslDistributions) -contains $manifest.wsl.distribution)) { Write-Output "OK WSL $($manifest.wsl.distribution)" }
        else { [void]$missing.Add('WSL'); Write-Output "MISSING WSL $($manifest.wsl.distribution) (install or complete first launch)" }
    }
    foreach ($command in @('winget', 'git', 'gh', 'clangd', 'wt', 'pwsh', 'psmux', 'codex', 'node', 'rg')) {
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
} else { Ensure-DeveloperMode }
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
        $installArguments = @('install', '--id', $package.id, '--exact', '--source', 'winget', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity')
        if ($package.Contains('installerArguments')) { $installArguments += @('--custom', $package.installerArguments) }
        Invoke-Native winget $installArguments
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
if (!(Test-Path -LiteralPath (Join-Path $RcPath '.setup/__main__.py'))) { throw 'Update rc_files: .setup/__main__.py is required.' }
Invoke-RcSetup $RcPath @('apply', '--package', 'git', '--home', $env:USERPROFILE, '--backup', $backupRoot)
if (!$SkipCredentials) { & (Join-Path $PSScriptRoot 'priv.ps1') -Apply -VaultPath $VaultPath -SshSource $SshSource }
Ensure-Checkout $NotesPath $manifest.notesRepository
$vimExe = Get-NativeVim
$psmuxExe = (Get-Command psmux.exe).Source
$deployment = @(Get-DeploymentArguments '' $env:USERPROFILE $RcPath $documents $TerminalPath) + @('--config-only', '--backup', $backupRoot)
Invoke-RcSetup $RcPath $deployment
$fontFace = (Read-Json (Join-Path $RcPath 'terminal/.windows/@LOCALAPPDATA/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json')).profiles.defaults.font.face
if (!$SkipFonts) { $fontFace = Install-Fonts $manifest.font $backupRoot $fontFace }
Save-Json (Join-Path $stateRoot 'state.json') ([ordered]@{ vimExe = $vimExe; psmuxExe = $psmuxExe; rcRoot = $RcPath; notes = $NotesPath }) $backupRoot
if (!$SkipVimPlugins) {
    $vimArguments = @('vim', '--home', $env:USERPROFILE, '--vim-exe', $vimExe)
    if ($Update) { $vimArguments += '--update' }
    Invoke-RcSetup $RcPath $vimArguments
}
$desktop = if ($env:USERPROFILE -eq $nativeProfile) { [Environment]::GetFolderPath('Desktop') } else { Join-Path $env:USERPROFILE 'Desktop' }
Set-VimShortcuts $vimExe $desktop (Join-Path $env:APPDATA 'Microsoft/Windows/Start Menu/Programs') $backupRoot
Set-NativeCodex $env:USERPROFILE $RcPath $backupRoot
if (!$SkipAddons) {
    $muxArguments = @('mux', '--home', $env:USERPROFILE)
    if ($Update) { $muxArguments += '--update' }
    Invoke-RcSetup $RcPath $muxArguments
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
