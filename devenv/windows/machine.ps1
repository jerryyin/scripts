# Invoked once by bootstrap; user configuration and credentials stay in its caller.
param([Parameter(Mandatory)][string]$RequestPath, [Parameter(Mandatory)][string]$ResultPath)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
Import-Module (Join-Path $PSScriptRoot 'Devenv.psm1') -Force -DisableNameChecking
$recording = $false
try {
    $request = Read-Json $RequestPath
    if ($request.UserSid -ne [Security.Principal.WindowsIdentity]::GetCurrent().User.Value) { throw 'Approve elevation using the account that started setup; another administrator account would configure the wrong WSL user.' }
    if (!(Test-IsAdministrator)) { throw 'Machine setup requires administrator approval.' }
    Start-Transcript -LiteralPath $request.LogPath | Out-Null
    $recording = $true
    Update-ProcessPath
    if ($request.DeveloperMode) { Ensure-DeveloperMode }
    $wslState = $request.WslState
    if ($request.Distribution) { $wslState = Ensure-Wsl $request.Distribution }
    foreach ($package in $request.Packages) {
        if ($package.scope -ne 'machine') { throw 'Only machine-scoped packages belong in the administrator step.' }
        Install-WinGetPackage $package
    }
    if ($request.UpgradeNode) { Install-WinGetPackage @{id='OpenJS.NodeJS.LTS'; scope='machine'} -Upgrade }
    if ($request.SshAgent) { Ensure-SshAgent }
    Save-Json $ResultPath @{ WslState=$wslState } ''
} catch {
    Save-Json $ResultPath @{ Error=$_.Exception.Message } ''
    exit 1
} finally {
    if ($recording) { Stop-Transcript | Out-Null }
}
