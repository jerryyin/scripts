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
if (@($visible | Where-Object { $_.name -in @('Windows PowerShell', 'Command Prompt') }).Count) { throw 'Alternative native shells remain visible beside PowerShell 7' }
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
$vimEnvironment = @{ WT_SESSION=$env:WT_SESSION; SSH_CONNECTION=$env:SSH_CONNECTION; SSH_CLIENT=$env:SSH_CLIENT }
$script:vimThrows = $false
function vim.exe {
    $script:vimInvocation = @{ WindowTerminal=$env:WT_SESSION; Arguments=@($args) }
    if ($script:vimThrows) { throw 'fixture launch failure' }
    $global:LASTEXITCODE = 23
}
try {
    foreach ($case in @(
        @{ Connection=''; Client=''; Terminal=''; Workaround=$false },
        @{ Connection=''; Client=''; Terminal='local-terminal'; Workaround=$false },
        @{ Connection='fixture-ssh'; Client=''; Terminal=''; Workaround=$true },
        @{ Connection=''; Client='fixture-client'; Terminal=''; Workaround=$true },
        @{ Connection='fixture-ssh'; Client=''; Terminal='existing-terminal'; Workaround=$false }
    )) {
        $env:SSH_CONNECTION = $case.Connection; $env:SSH_CLIENT = $case.Client; $env:WT_SESSION = $case.Terminal
        vi 'a file with spaces.txt' '+set readonly'
        $actual = $script:vimInvocation
        if ($case.Workaround) {
            if (!$actual.WindowTerminal) { throw 'SSH Vim must preserve the terminal default background' }
        } elseif ([string]$actual.WindowTerminal -ne $case.Terminal) { throw 'Local Vim or an existing terminal marker was changed' }
        if (($actual.Arguments -join '|') -cne 'a file with spaces.txt|+set readonly' -or $LASTEXITCODE -ne 23) { throw 'Vim arguments or exit status were changed' }
        if ([string]$env:WT_SESSION -ne $case.Terminal) { throw 'Vim leaked its terminal marker into PowerShell' }
    }
    $env:SSH_CONNECTION = 'fixture-ssh'; $env:WT_SESSION = $null; $script:vimThrows = $true
    $failed = $false
    try { vim } catch { $failed = $_.Exception.Message -eq 'fixture launch failure' }
    if (!$failed -or $env:WT_SESSION) { throw 'A failed Vim launch leaked its terminal marker or lost the error' }
} finally {
    Remove-Item Function:\vim.exe
    foreach ($name in $vimEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $vimEnvironment[$name], 'Process') }
}
Write-Output 'PASS: Vim background compatibility is SSH-only, preserves arguments and exit status, and restores the environment after success or failure.'
Write-Output "PASS: one visible PowerShell, existing profiles retained, unchanged rerun, native prompt and Git branch. Fixture: $fixture"
