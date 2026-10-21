# Invoked once by bootstrap; credentials and application setup stay in its caller.
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
    Update-ProcessPath
    $sshLinks = $request.Contains('SshLinks') -and $request.SshLinks
    if (($request.Contains('RcMachine') -and $request.RcMachine) -or $sshLinks) {
        $newCheckout = !(Test-Path -LiteralPath (Join-Path $request.RcPath '.git'))
        Ensure-Checkout $request.RcPath $request.RcRepository
        if ($newCheckout) { Invoke-Native icacls @($request.RcPath, '/setowner', ('*' + $request.UserSid), '/T', '/Q') }
        if (!$sshLinks) { Invoke-RcSetup $request.RcPath @('prerequisites', '--scope', 'machine') }
    }
    if ($sshLinks) {
        Import-Module (Join-Path $PSScriptRoot 'SshSession.psm1') -Force -DisableNameChecking
        # Restore existing links before tools use them, then cover aliases made by installers.
        Repair-SshLinks $request.RcPath $request.LinkPlanArguments -ExistingOnly
        if ($request.InstallPrerequisites) {
            Invoke-RcSetup $request.RcPath @('prerequisites')
            Update-ProcessPath
            Repair-SshLinks $request.RcPath $request.LinkPlanArguments -ExistingOnly
        }
        Invoke-RcSetup $request.RcPath $request.LinkArguments
    }
    if ($request.SshAgent) { Ensure-SshAgent }
    if ($request.Contains('SshShell') -and $request.SshShell) {
        Import-Module (Join-Path $PSScriptRoot 'SshSession.psm1') -Force -DisableNameChecking
        Set-SshShell
    }
    Save-Json $ResultPath @{ WslState=$wslState } ''
} catch {
    Save-Json $ResultPath @{ Error=$_.Exception.Message } ''
    exit 1
} finally {
    if ($recording) { Stop-Transcript | Out-Null }
}
