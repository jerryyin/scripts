param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
Import-Module (Join-Path $root 'SshSession.psm1') -Force -DisableNameChecking
$module = Get-Module SshSession
function Assert($ok, [string]$message) {
    if (!$ok) { throw "FAIL: $message" }
    Write-Output "PASS: $message"
}
& $module {
    $script:Installed = $true
    $script:Shell = 'C:\fixture\pwsh.exe'
    $script:Registry = [pscustomobject]@{ DefaultShell='C:\Windows\System32\cmd.exe' }
    $script:Writes = 0
    $script:Installs = 0
    $script:StableShell = ''
    function script:Test-Path { param($LiteralPath) return $LiteralPath -eq $script:StableShell }
    function script:Get-Service { if ($script:Installed) { @{Name='sshd'} } }
    function script:Get-Command { if ($script:Shell) { @{Source=$script:Shell} } }
    function script:Get-ItemProperty { return $script:Registry }
    function script:New-Item { }
    function script:New-ItemProperty {
        param($LiteralPath, $Name, $PropertyType, $Value, [switch]$Force)
        if ($Name -ne 'DefaultShell') { throw 'SSH authentication or command options were changed' }
        $script:Registry | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
        $script:Writes++
    }
    function script:Test-IsAdministrator { return $true }
    function script:Save-Json { }
    function script:Invoke-RcSetup {
        param($RcRoot, $Arguments, [switch]$Capture)
        $script:Installs++
        throw 'Shell configuration must not install prerequisites'
    }
}
$state = Get-SshShellState
Assert ($state.Installed -and $state.NeedsSetup) 'Existing SSH server with cmd as default needs shell setup'
Set-SshShell
Assert (!(Get-SshShellState).NeedsSetup -and (& $module { $script:Writes }) -eq 1) 'Shell setup writes the discovered PowerShell path'
Set-SshShell
Assert ((& $module { $script:Writes }) -eq 1) 'Repeated shell setup makes no registry changes'
& $module { $script:Installed = $false }
Assert (!(Get-SshShellState).NeedsSetup) 'A machine without SSH Server needs no SSH setup'
Set-SshShell
Assert ((& $module { $script:Writes }) -eq 1 -and (& $module { $script:Installs }) -eq 0) 'Absent SSH Server does not install tools or change the registry'
& $module { $script:Installed = $true; $script:Shell = '' }
$failed = $false
try { Set-SshShell } catch { $failed = $_.Exception.Message -like '*PowerShell 7*' }
Assert ($failed -and (& $module { $script:Installs }) -eq 0) 'Package-skipping setup reports a missing shell without installing'
& $module { $script:Shell='C:\fixture\pwsh.exe'; $script:Registry = [pscustomobject]@{} }
Assert ((Get-SshShellState).NeedsSetup) 'An SSH registry key without DefaultShell is handled'
& $module {
    $script:StableShell=Join-Path $env:ProgramFiles 'PowerShell/7/pwsh.exe'
    $script:Shell='C:\user\WindowsApps\Microsoft.PowerShell_package\pwsh.exe'
}
Set-SshShell
Assert ((Get-SshShellState).ShellPath -eq (Join-Path $env:ProgramFiles 'PowerShell/7/pwsh.exe')) 'SSH uses the installed machine shell ahead of a caller-specific Store alias'
& $module { $script:Shell='C:\user\WindowsApps\pwsh.exe' }
Assert (!(Get-SshShellState).NeedsSetup) 'Changing the caller PATH does not cause repeated SSH shell repairs'
Write-Output 'PASS: SSH setup checks are read-only and do not touch real services or registry values.'

Remove-Module SshSession
Import-Module (Join-Path $root 'SshSession.psm1') -Force -DisableNameChecking
$module = Get-Module SshSession
$fixture = Join-Path $env:LOCALAPPDATA ('PersonalDevenv/test-ssh-links-' + [guid]::NewGuid().ToString('N'))
$priorLocal = $env:LOCALAPPDATA
try {
    $env:LOCALAPPDATA = Join-Path $fixture 'local'
    $rc = Join-Path $fixture 'rc'
    [void][IO.Directory]::CreateDirectory($rc)
    $source = Join-Path $rc ('profile-' + [char]0x03bb + '.ps1')
    $target = Join-Path $fixture ('profile-' + [char]0x03bb + '.ps1')
    [IO.File]::WriteAllText($source, '# fixture profile')
    New-Item -ItemType SymbolicLink -Path $target -Target $source | Out-Null
    $readErrors = @(& $module {
        param($target)
        Get-SshLinkReadErrors @(@{target=$target; directory=$false}, @{target=($target + '.missing'); directory=$false})
    } $target)
    if ($readErrors.Count -ne 1 -or $readErrors[0].code -ne 2 -or $readErrors[0].target -cne ($target + '.missing')) { throw "Unexpected SSH read errors: $(ConvertTo-Json -InputObject $readErrors -Compress)" }
    Write-Output 'PASS: Isolated SSH probe preserves Unicode paths and reports actual file errors'
    & $module {
        param($source, $target)
        $script:Link = @{source=$source; target=$target; directory=$false}
        $script:ReadError = 0
        function script:Invoke-RcSetup {
            param($RcRoot, $Arguments, [switch]$Capture)
            if (($Arguments -join ' ') -ne 'plan --json') { throw 'Unexpected link plan arguments' }
            return (@{links=@($script:Link)} | ConvertTo-Json -Depth 4)
        }
        function script:Get-SshLinkReadErrors($Links) {
            if ($script:ReadError) {
                $code = $script:ReadError
                $script:ReadError = 0
                return [pscustomobject]@{target=$script:Link.target; code=$code; message='fixture read failure'}
            }
        }
        function script:Get-SshShellState { return @{Installed=$true; NeedsSetup=$false; ShellPath='fixture'} }
    } $source $target
    Assert (@(Get-SshSessionProblems $rc).Count -eq 0) 'Readable matching links have no SSH readiness problem'
    $priorProfile = $env:USERPROFILE
    try {
        $env:USERPROFILE = Join-Path $fixture 'redirected-home'
        Assert (@(Get-SshSessionProblems $rc).Count -eq 0) 'Readiness checks support configuration destinations outside the user profile'
    } finally { $env:USERPROFILE = $priorProfile }
    & $module { $script:ReadError = 448 }
    $problems = @(Get-SshSessionProblems $rc)
    Assert ($problems.Count -eq 1 -and $problems[0] -like '*error 448*bootstrap.ps1 -Apply*') 'Blocked links produce actionable readiness failures'
    & $module { $script:ReadError = 5 }
    Assert ((@(Get-SshSessionProblems $rc) -join '') -like '*check failed*') 'Other I/O errors are reported rather than treated as repairable links'
    if (Test-IsAdministrator) {
        & $module { $script:ReadError = 448 }
        Repair-SshLinks $rc
        Assert ([IO.File]::ReadAllText($target) -ceq '# fixture profile') 'Link repair preserves and exposes the original source content'
        Assert (@(Get-ChildItem -LiteralPath (Join-Path $env:LOCALAPPDATA 'PersonalDevenv/backups') -Recurse -File).Count -eq 1) 'Link repair keeps the original link as a backup'
    }
    [IO.File]::Delete($target)
    [IO.File]::WriteAllText($target, 'unrelated user file')
    $failed = $false
    try { Repair-SshLinks $rc } catch { $failed = $_.Exception.Message -like '*unrelated configuration*' }
    Assert ($failed -and [IO.File]::ReadAllText($target) -ceq 'unrelated user file') 'Repair refuses to replace unrelated files'
    if (Test-IsAdministrator) {
        Remove-Module SshSession
        Import-Module (Join-Path $root 'SshSession.psm1') -Force -DisableNameChecking
        $userRoot = Join-Path $fixture 'new-account'
        $documents = Join-Path $userRoot 'Documents'
        $terminal = Join-Path $userRoot 'Terminal/settings.json'
        $planArguments = @(Get-DeploymentArguments 'plan' $userRoot $RcRoot $documents $terminal) + @('--json', '--package', 'powershell')
        $linkArguments = @(Get-DeploymentArguments '' $userRoot $RcRoot $documents $terminal) + @('--config-only', '--package', 'powershell', '--backup', (Join-Path $fixture 'backups'))
        $requestPath = Join-Path $fixture 'request.json'
        $resultPath = Join-Path $fixture 'result.json'
        Save-Json $requestPath @{
            UserSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
            LogPath=(Join-Path $fixture 'machine.log'); DeveloperMode=$false; WslState='Skipped'; Distribution=''
            Packages=@(); RcMachine=$false; RcPath=$RcRoot; RcRepository='unused'; SshAgent=$false
            SshShell=$false; SshLinks=$true; InstallPrerequisites=$false
            LinkPlanArguments=$planArguments; LinkArguments=$linkArguments
        } ''
        $installed = Join-Path $documents 'PowerShell/profile.ps1'
        $stamp = $null
        foreach ($run in 1..2) {
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'machine.ps1') -RequestPath $requestPath -ResultPath $resultPath | Out-Null
            Assert ($LASTEXITCODE -eq 0 -and (Read-Json $resultPath).WslState -eq 'Skipped') 'Real machine worker deploys configuration without installing packages or changing services'
            Assert ((Get-Item -LiteralPath $installed).LinkType -eq 'SymbolicLink' -and @(Get-SshLinkRepairs $RcRoot $planArguments).Count -eq 0) 'Deployed profile is a source link readable under the SSH traversal restriction'
            $currentStamp = (Get-Item -LiteralPath $installed).CreationTimeUtc
            if ($stamp) { Assert ($currentStamp -eq $stamp) 'Rerunning setup preserves the existing trusted link' }
            $stamp = $currentStamp
        }
    }
} finally {
    $env:LOCALAPPDATA = $priorLocal
    $resolved = [IO.Path]::GetFullPath($fixture)
    $prefix = [IO.Path]::GetFullPath((Join-Path $priorLocal 'PersonalDevenv')).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notmatch '^test-ssh-links-[a-f0-9]{32}$') { throw 'Unsafe fixture cleanup path' }
    Get-ChildItem -LiteralPath $resolved -Recurse -File | Where-Object LinkType -eq 'SymbolicLink' | ForEach-Object { [IO.File]::Delete($_.FullName) }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
