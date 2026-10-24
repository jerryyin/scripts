$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
$module = Get-Module Devenv
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('machine-setup-test-' + [guid]::NewGuid().ToString('N'))
$savedLocal = $env:LOCALAPPDATA
[void][IO.Directory]::CreateDirectory($fixture)
$env:LOCALAPPDATA = Join-Path $fixture "Someone's Windows account"
try {
    & $module {
        $script:PackageScopeCalls=@()
        $script:PackageScopeResults=[Collections.Generic.Queue[int]]::new()
        function script:winget {
            $script:PackageScopeCalls += ,$args
            $global:LASTEXITCODE=$script:PackageScopeResults.Dequeue()
        }
    }
    $tool=@{id='Fixture.Shared'; scope='machine'; installedCommand='powershell.exe'; installedAppxName='Fixture.UserApp'}
    & $module { $script:PackageScopeResults.Enqueue(-1978335212) }
    if (@(Get-MissingPackages @($tool)).Count -ne 1) { throw 'An existing user command must not satisfy a machine package.' }
    & $module { $script:PackageScopeResults.Enqueue(0) }
    if (@(Get-MissingPackages @($tool)).Count -ne 0) { throw 'A registered machine package must be reused.' }
    & $module { $script:PackageScopeResults.Enqueue(0); $script:PackageScopeResults.Enqueue(-1978335212) }
    $failed=$false
    try { Install-WinGetPackage $tool } catch { $failed=$_.Exception.Message -like '*still missing in machine scope*' }
    if (!$failed) { throw 'Installer success must be checked against machine registration.' }
    & $module { $script:PackageScopeResults.Enqueue(5) }
    $failed=$false
    try { Get-MissingPackages @($tool) } catch { $failed=$_.Exception.Message -like '*inventory failed*' }
    if (!$failed) { throw 'An inventory error must not be reported as a missing package.' }
    if (@((& $module { ,$script:PackageScopeCalls }) | Where-Object { $_ -notcontains '--scope' -or $_ -notcontains 'machine' }).Count) { throw 'All inventory and installation calls must explicitly select machine scope.' }
    & $module {
        $script:PythonKey=$null
        $script:Launcher=$false
        function script:Get-ItemProperty {
            param($LiteralPath, $ErrorAction)
            if ($LiteralPath -notlike 'HKLM:*PythonCore*') { throw 'Python must use machine interpreter registration.' }
            return $script:PythonKey
        }
        function script:Test-Path {
            param($LiteralPath, $PathType)
            if ($LiteralPath -eq 'machine-python.exe') { return $true }
            return $script:Launcher
        }
    }
    $pythonPackage=@{id='Python.Python.3.14'; scope='machine'}
    if (@(Get-MissingPackages @($pythonPackage)).Count -ne 1) { throw 'A user-only Python must not satisfy machine setup.' }
    & $module { $script:PythonKey=@{ExecutablePath='machine-python.exe'} }
    if (@(Get-MissingPackages @($pythonPackage)).Count -ne 1) { throw 'A machine interpreter without its shared launcher is incomplete.' }
    & $module { $script:Launcher=$true }
    if (@(Get-MissingPackages @($pythonPackage)).Count -ne 0) { throw 'Python readiness must use machine files even when the installer bundle is registered per user.' }
    & $module { Remove-Item Function:Get-ItemProperty; Remove-Item Function:Test-Path }
    & $module {
        function script:Invoke-RcSetup { param($RcRoot, $Arguments, [switch]$Capture) return $script:RequirementsJson }
    }
    foreach ($json in @('[]', '[{"command":"one"}]', '[{"command":"one"},{"command":"two"}]')) {
        & $module { param($value) $script:RequirementsJson=$value } $json
        $expected = if ($json -eq '[]') { 0 } elseif ($json.Contains('two')) { 2 } else { 1 }
        if (@(Get-RcMissingPackages 'fixture' 'machine').Count -ne $expected) { throw 'The prerequisite inventory must preserve empty, single and multiple requirements on PowerShell 5.1.' }
    }
    & $module {
        $script:Launches = @()
        $script:ExitCode = 0
        $script:Administrator = $false
        $script:Cancel = $false
        function script:Test-IsAdministrator { return $script:Administrator }
        function script:Start-Process {
            param($FilePath, $ArgumentList, $Verb, $WindowStyle, [switch]$Wait, [switch]$PassThru)
            $script:Launches += @{ Verb=$Verb; Hidden=($WindowStyle -eq 'Hidden'); Wait=[bool]$Wait }
            if ($script:Cancel) { throw (New-Object ComponentModel.Win32Exception(1223)) }
            $command = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($ArgumentList[-1]))
            $tokens = $null; $errors = $null
            $ast = [Management.Automation.Language.Parser]::ParseInput($command, [ref]$tokens, [ref]$errors)
            if ($errors.Count) { throw 'Invalid elevation command' }
            $strings = @($ast.FindAll({ param($node) $node -is [Management.Automation.Language.StringConstantExpressionAst] }, $true))
            $request = @($strings | Where-Object Value -like '*request.json')[0].Value
            $result = @($strings | Where-Object Value -like '*result.json')[0].Value
            $script:Request = Read-Json $request
            Save-Json $result @{ WslState='Installed'; Error='fixture machine failure' } ''
            return @{ ExitCode=$script:ExitCode }
        }
    }
    $plan = @{ DeveloperMode=$false; Distribution=''; Packages=@(); RcMachine=$false; SshAgent=$false; WslState='Existing' }
    $state = Invoke-MachineSetup $plan
    if ($state -ne 'Existing' -or (& $module { $script:Launches.Count }) -ne 0) { throw 'An unchanged machine must need zero UAC launches' }
    $plan.Fonts=$true
    Invoke-MachineSetup $plan | Out-Null
    if ((& $module { $script:Launches.Count }) -ne 1 -or !(& $module { $script:Request.Fonts })) { throw 'Fonts alone must enter the same grouped administrator phase' }
    $plan.Fonts=$false
    & $module { $script:Launches=@() }
    $plan.SshLinks=$true
    Invoke-MachineSetup $plan | Out-Null
    if ((& $module { $script:Launches.Count }) -ne 1 -or !(& $module { $script:Request.SshLinks })) { throw 'Configuration link work alone must enter the grouped machine phase' }
    & $module { $script:Launches = @() }
    $plan.SshLinks=$false; $plan.SshShell=$true; $plan.InstallPrerequisites=$true
    Invoke-MachineSetup $plan | Out-Null
    if ((& $module { $script:Launches.Count }) -ne 1 -or !(& $module { $script:Request.SshShell })) { throw 'SSH shell configuration alone must enter the grouped machine phase' }
    & $module { $script:Launches = @() }
    $plan.SshLinks=$true; $plan.DeveloperMode=$true; $plan.Distribution='Ubuntu-24.04'; $plan.RcMachine=$true; $plan.SshAgent=$true
    $plan.Packages=@(@{id='GitHub.cli'; scope='machine'}, @{id='7zip.7zip'; scope='machine'})
    $state = Invoke-MachineSetup $plan
    $launches = & $module { ,$script:Launches }
    $request = & $module { $script:Request }
    if ($state -ne 'Installed' -or $launches.Count -ne 1 -or $launches[0].Verb -ne 'RunAs' -or !$launches[0].Hidden -or !$launches[0].Wait) { throw 'All machine changes must share one hidden, awaited UAC launch' }
    if (!$request.DeveloperMode -or !$request.SshAgent -or !$request.SshShell -or !$request.SshLinks -or !$request.RcMachine -or $request.Distribution -ne 'Ubuntu-24.04' -or $request.Packages.Count -ne 2) { throw 'The elevated request lost machine work' }
    if ($request.UserSid -ne [Security.Principal.WindowsIdentity]::GetCurrent().User.Value) { throw 'The machine phase must retain the initiating account identity' }
    & $module { $script:ExitCode = 7; $script:Launches = @() }
    $failed = $false
    try { Invoke-MachineSetup $plan } catch { $failed=$_.Exception.Message -like '*fixture machine failure*' }
    if (!$failed -or (& $module { $script:Launches.Count }) -ne 1) { throw 'A failed machine phase must stop setup without a second UAC attempt' }
    & $module { $script:ExitCode = 0; $script:Administrator = $true; $script:Launches = @() }
    Invoke-MachineSetup $plan | Out-Null
    if ((& $module { $script:Launches[0].Verb })) { throw 'An elevated caller must not request another UAC approval' }
    & $module { $script:Administrator = $false; $script:Cancel = $true; $script:Launches = @() }
    $cancelled = $false
    try { Invoke-MachineSetup $plan } catch { $cancelled=$_.Exception.NativeErrorCode -eq 1223 }
    if (!$cancelled -or (& $module { $script:Launches.Count }) -ne 1) { throw 'Declined UAC must stop setup without retrying' }
    & $module {
        function script:Get-Service { return @{Status='Stopped'; StartType='Disabled'} }
        $script:Launches = @()
    }
    $failed = $false
    try { Ensure-SshAgent -ReadyOnly } catch { $failed=$_.Exception.Message -like '*grouped administrator step*' }
    if (!$failed -or (& $module { $script:Launches.Count }) -ne 0) { throw 'The credential phase must never start another administrator prompt' }
    if (@(Get-ChildItem -LiteralPath $env:LOCALAPPDATA -Recurse -Filter request.json).Count) { throw 'Machine request files were left behind' }
    $manifest = Read-Json (Join-Path $root 'defaults/environment.json')
    if ($manifest.wsl.distribution -cne 'Ubuntu') { throw 'Default setup must reuse the Ubuntu environment, rather than request a second version-named distro' }
    & $module {
        $script:WslInventoryCalls = 0
        function script:Invoke-WslCommand {
            param([string[]]$Arguments, [switch]$Elevate)
            if ($Elevate -or ($Arguments -join ' ') -ne '--list --quiet') { throw 'Existing Ubuntu must not invoke installation or elevation' }
            $script:WslInventoryCalls++
            return @{ ExitCode=0; Output='Ubuntu' }
        }
    }
    if ((Ensure-Wsl $manifest.wsl.distribution) -ne 'Existing' -or (& $module { $script:WslInventoryCalls }) -ne 1) { throw 'Existing default Ubuntu was not reused' }
    $python = @($manifest.packages | Where-Object id -eq 'Python.Python.3.14')[0]
    if ($python.scope -ne 'machine' -or $python.installerArguments -notmatch 'InstallAllUsers=1 InstallLauncherAllUsers=1') { throw 'Python and its launcher must both use the grouped machine installation' }
    $fresh = Join-Path $fixture 'fresh machine'
    [void][IO.Directory]::CreateDirectory($fresh)
    Copy-Item -LiteralPath (Join-Path $root 'machine.ps1') -Destination (Join-Path $fresh 'machine.ps1')
    Save-Text (Join-Path $fresh 'Devenv.psm1') @'
$script:PathReady = $false
function Read-Json($Path) {
    return @{
        UserSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        LogPath=(Join-Path $PSScriptRoot 'machine.log'); DeveloperMode=$false
        WslState='Existing'; Distribution=''; Packages=@(@{id='Git.Git'; scope='machine'})
        RcMachine=$true; RcPath=(Join-Path $PSScriptRoot 'rc_files'); RcRepository='fixture'; SshAgent=$false
        SshShell=$true; InstallPrerequisites=$true
        SshLinks=$true; LinkPlanArguments=@('plan', '--json'); LinkArguments=@('--config-only', '--backup', 'fixture')
    }
}
function Test-IsAdministrator { return $true }
function Install-WinGetPackage($Package) { $script:PathReady = $false }
function Update-ProcessPath { $script:PathReady = $true }
function Ensure-Checkout($Path, $Repository) {
    if (!$script:PathReady) { throw 'Newly installed Git is not yet visible on PATH.' }
    [void][IO.Directory]::CreateDirectory((Join-Path $Path '.git'))
}
function Invoke-RcSetup($Path, $Arguments) { Add-Content -LiteralPath (Join-Path $PSScriptRoot 'steps') -Value ($Arguments -join ' ') }
function Invoke-Native($Command, $Arguments) { }
function Save-Json($Path, $Value, $Backup) { [IO.File]::WriteAllText($Path, (ConvertTo-Json $Value)) }
'@ ''
    Save-Text (Join-Path $fresh 'SshSession.psm1') @'
function Set-SshShell {
    $RcRoot = Join-Path $PSScriptRoot 'rc_files'
    if (!(Test-Path -LiteralPath (Join-Path $RcRoot '.git'))) { throw 'SSH shell setup ran before its prerequisites were prepared.' }
    [IO.File]::WriteAllText((Join-Path $PSScriptRoot 'ssh-shell-called'), $RcRoot)
}
function Repair-SshLinks($RcRoot, $Arguments, [switch]$ExistingOnly) {
    if (!$ExistingOnly -or ($Arguments -join ' ') -ne 'plan --json') { throw 'Only declared links may be repaired before deployment.' }
    Add-Content -LiteralPath (Join-Path $PSScriptRoot 'steps') -Value 'repair-links'
}
'@ ''
    $freshResult = Join-Path $fresh 'result.json'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $fresh 'machine.ps1') -RequestPath (Join-Path $fresh 'request.json') -ResultPath $freshResult | Out-Null
    if ($LASTEXITCODE -ne 0 -or (Read-Json $freshResult).WslState -ne 'Existing') { throw 'The machine phase must refresh PATH after installing Git and before cloning rc_files.' }
    if (!(Test-Path -LiteralPath (Join-Path $fresh 'ssh-shell-called'))) { throw 'The grouped machine helper did not configure the SSH shell.' }
    $steps = @(Get-Content -LiteralPath (Join-Path $fresh 'steps')) -join ';'
    if ($steps -ne 'repair-links;prerequisites --update;repair-links;--config-only --backup fixture') { throw "Machine setup must restore existing links, prepare tools, repair new aliases and deploy trusted configuration in order: $steps" }
    Write-Output 'PASS: grouped/unchanged/elevated/cancelled UAC, credential boundary, machine Python launcher, quoted paths, account identity, failures and cleanup.'
} finally {
    $env:LOCALAPPDATA = $savedLocal
    $resolved = [IO.Path]::GetFullPath($fixture)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notlike 'machine-setup-test-*') { throw 'Unsafe fixture cleanup path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
