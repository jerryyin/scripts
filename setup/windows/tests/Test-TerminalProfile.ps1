param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
. (Join-Path $PSScriptRoot 'RcFixture.ps1')
$fixture = Join-Path $env:TEMP ('terminal-profile-' + [guid]::NewGuid().ToString('N'))
$path = Join-Path $fixture 'settings.json'
$source = Join-Path $fixture 'rc_files/terminal/.windows/@LOCALAPPDATA/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json'
[void][IO.Directory]::CreateDirectory((Split-Path $source -Parent))
Copy-Item -LiteralPath (Join-Path $RcRoot 'terminal/.windows/@LOCALAPPDATA/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json') -Destination $source
Invoke-TestLink $path $source ''
if (!(Test-InstalledLink $path $source)) { throw 'Terminal configuration is not linked to rc_files' }
$settings = Read-Json $path
$visible = @($settings.profiles.list | Where-Object { !$_.Contains('hidden') -or !$_.hidden })
if (@($visible | Where-Object { $_.name -eq 'PowerShell' }).Count -ne 1) { throw 'Duplicate PowerShell entries remain visible' }
$wsl = @($visible | Where-Object { $_.name -eq 'Ubuntu' })
if ($wsl.Count -ne 1 -or $wsl[0].commandline -cne 'wsl.exe --cd ~' -or $wsl[0].Contains('source')) {
    throw 'One Ubuntu profile must launch the default distro without a machine-specific profile source'
}
if ($settings.disabledProfileSources -notcontains 'Windows.Terminal.Wsl' -or $settings.disabledProfileSources -notcontains 'Microsoft.WSL') { throw 'Generated WSL profiles must not duplicate the single Ubuntu entry' }
$before = (Get-Item $path).CreationTimeUtc.Ticks
Invoke-TestLink $path $source ''
if ((Get-Item $path).CreationTimeUtc.Ticks -ne $before) { throw 'Terminal rerun replaced its link' }
$settings.profiles.defaults.font.face = 'Test Font'
[IO.File]::WriteAllText($path,(ConvertTo-Json -InputObject $settings -Depth 30))
if ((Read-Json $source).profiles.defaults.font.face -cne 'Test Font') { throw 'App-side font edit did not reach tracked source' }
Invoke-TestLink $path $source ''
if ((Read-Json $path).profiles.defaults.font.face -cne 'Test Font') { throw 'Setup overwrote the tracked font preference' }
$conflict = Join-Path $fixture 'detached-copy.json'
Save-Text $conflict '{"localEdit": true}' ''
$rejected=$false
try { Invoke-TestLink $conflict $source '' } catch { $rejected=$_.Exception.Message.Contains('Configuration conflict') }
if (!$rejected -or !(Read-Json $conflict).localEdit) { throw 'A conflicting local copy was replaced' }

$originalLocalAppData = $env:LOCALAPPDATA
try {
    $env:LOCALAPPDATA = Join-Path $fixture 'AppData/Local'
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
