param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
$fixture = Join-Path $env:TEMP ('terminal-profile-' + [guid]::NewGuid().ToString('N'))
$path = Join-Path $fixture 'settings.json'
$source = Join-Path $fixture 'rc_files/terminal/.windows/@LOCALAPPDATA/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json'
[void][IO.Directory]::CreateDirectory((Split-Path $source -Parent))
Copy-Item -LiteralPath (Join-Path $RcRoot 'terminal/.windows/@LOCALAPPDATA/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json') -Destination $source
Set-Terminal $path $source ''
if (!(Test-TrackedFile $path $source)) { throw 'Terminal configuration is not linked to rc_files' }
$settings = Read-Json $path
$visible = @($settings.profiles.list | Where-Object { !$_.Contains('hidden') -or !$_.hidden })
if (@($visible | Where-Object { $_.name -eq 'PowerShell' }).Count -ne 1) { throw 'Duplicate PowerShell entries remain visible' }
$before = (Get-Item $path).CreationTimeUtc.Ticks
Set-Terminal $path $source ''
if ((Get-Item $path).CreationTimeUtc.Ticks -ne $before) { throw 'Terminal rerun replaced its link' }
$settings.profiles.defaults.font.face = 'Test Font'
[IO.File]::WriteAllText($path,(ConvertTo-Json -InputObject $settings -Depth 30))
if ((Read-Json $source).profiles.defaults.font.face -cne 'Test Font') { throw 'App-side font edit did not reach tracked source' }
Set-Terminal $path $source ''
if ((Read-Json $path).profiles.defaults.font.face -cne 'Test Font') { throw 'Setup overwrote the tracked font preference' }
$conflict = Join-Path $fixture 'detached-copy.json'
Save-Text $conflict '{"localEdit": true}' ''
$rejected=$false
try { Set-Terminal $conflict $source '' } catch { $rejected=$_.Exception.Message.Contains('Configuration conflict') }
if (!$rejected -or !(Read-Json $conflict).localEdit) { throw 'A conflicting local copy was replaced' }

$originalLocalAppData = $env:LOCALAPPDATA
try {
    $env:LOCALAPPDATA = Join-Path $fixture 'AppData/Local'
    Save-Json (Join-Path $env:LOCALAPPDATA 'PersonalDevenv/state.json') ([ordered]@{vimExe='vim.exe';psmuxExe='psmux.exe'}) ''
    . (Join-Path $RcRoot 'powershell/.windows/@DOCUMENTS/PowerShell/profile.ps1')
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
