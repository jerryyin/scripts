param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
$fixture = Join-Path $env:TEMP ('terminal-profile-' + [guid]::NewGuid().ToString('N'))
$path = Join-Path $fixture 'settings.json'
$defaults = Read-Json (Join-Path $RcRoot '.windows-terminal/settings.json')
Save-Json $path ([ordered]@{profiles=[ordered]@{list=@(
    [ordered]@{guid='{61c54bbd-c2c6-5271-96e7-009a87ff44bf}';name='Windows PowerShell';commandline='powershell.exe'},
    [ordered]@{guid='{574e775e-4f2a-5b96-ac1e-a2962a402336}';name='PowerShell';source='Windows.Terminal.PowershellCore'},
    [ordered]@{guid='personal';name='Personal';commandline='custom.exe'}
)}}) ''
Set-Terminal $path $defaults 'pwsh.exe -NoLogo' 'Test Font' ''
$settings = Read-Json $path
$visible = @($settings.profiles.list | Where-Object { !$_.Contains('hidden') -or !$_.hidden })
if ($visible.Count -ne 2 -or @($visible | Where-Object { $_.name -eq 'PowerShell' }).Count -ne 1) { throw 'Duplicate PowerShell entries remain visible' }
if ($visible[0].commandline -ne 'custom.exe') { throw 'Unrelated profile changed' }
if ($settings.profiles.list[0].commandline -ne 'powershell.exe') { throw 'Hidden profile settings were lost' }
$before = (Get-Item $path).LastWriteTimeUtc.Ticks
Set-Terminal $path $defaults 'pwsh.exe -NoLogo' 'Test Font' ''
if ((Get-Item $path).LastWriteTimeUtc.Ticks -ne $before) { throw 'Terminal rerun rewrote settings' }
$defaults.profiles.list[0].font.size = 13
$defaults.profiles.list[0].opacity = 88
$defaults.profiles.list[0].padding = '9'
Set-Terminal $path $defaults 'pwsh.exe -NoLogo' 'Test Font' ''
$configured = @((Read-Json $path).profiles.list | Where-Object { $_.guid -eq $defaults.defaultProfile })[0]
if ($configured.font.size -ne 13 -or $configured.opacity -ne 88 -or $configured.padding -ne '9') { throw 'Installer overrides tracked Terminal preferences' }
$originalLocalAppData = $env:LOCALAPPDATA
try {
    $env:LOCALAPPDATA = Join-Path $fixture 'AppData/Local'
    Save-Json (Join-Path $env:LOCALAPPDATA 'PersonalDevenv/state.json') ([ordered]@{vimExe='vim.exe';psmuxExe='psmux.exe'}) ''
    . (Join-Path $RcRoot '.powershell/profile.ps1')
} finally { $env:LOCALAPPDATA = $originalLocalAppData }
$global:LASTEXITCODE = 42
$text = prompt
if (!$text.Contains([char]0x279c) -or $LASTEXITCODE -ne 42) { throw 'Prompt marker or command exit code changed' }
if (Get-Module oh-my-posh-core) { throw 'Default prompt loads Oh My Posh' }
Push-Location (Split-Path $root -Parent)
try {
    $text = prompt
    $branch = (& git branch --show-current).Trim()
    if (!$text.Contains(' ' + $branch)) { throw 'Prompt omits Git branch' }
} finally { Pop-Location }
Write-Output "PASS: one visible PowerShell, existing profiles retained, unchanged rerun, native prompt and Git branch. Fixture: $fixture"
