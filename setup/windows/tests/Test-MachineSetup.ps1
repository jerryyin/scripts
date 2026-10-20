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
    $plan.DeveloperMode=$true; $plan.Distribution='Ubuntu-24.04'; $plan.RcMachine=$true; $plan.SshAgent=$true
    $plan.Packages=@(@{id='GitHub.cli'; scope='machine'}, @{id='7zip.7zip'; scope='machine'})
    $state = Invoke-MachineSetup $plan
    $launches = & $module { ,$script:Launches }
    $request = & $module { $script:Request }
    if ($state -ne 'Installed' -or $launches.Count -ne 1 -or $launches[0].Verb -ne 'RunAs' -or !$launches[0].Hidden -or !$launches[0].Wait) { throw 'All machine changes must share one hidden, awaited UAC launch' }
    if (!$request.DeveloperMode -or !$request.SshAgent -or !$request.RcMachine -or $request.Distribution -ne 'Ubuntu-24.04' -or $request.Packages.Count -ne 2) { throw 'The elevated request lost machine work' }
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
    if (!$python.Contains('installerArguments') -or $python.installerArguments -notmatch 'InstallLauncherAllUsers=0') { throw 'The user Python installer must not request a separate UAC for its launcher' }
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
    }
}
function Test-IsAdministrator { return $true }
function Install-WinGetPackage($Package) { $script:PathReady = $false }
function Update-ProcessPath { $script:PathReady = $true }
function Ensure-Checkout($Path, $Repository) {
    if (!$script:PathReady) { throw 'Newly installed Git is not yet visible on PATH.' }
    [void][IO.Directory]::CreateDirectory((Join-Path $Path '.git'))
}
function Invoke-RcSetup($Path, $Arguments) { }
function Invoke-Native($Command, $Arguments) { }
function Save-Json($Path, $Value, $Backup) { [IO.File]::WriteAllText($Path, (ConvertTo-Json $Value)) }
'@ ''
    $freshResult = Join-Path $fresh 'result.json'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $fresh 'machine.ps1') -RequestPath (Join-Path $fresh 'request.json') -ResultPath $freshResult | Out-Null
    if ($LASTEXITCODE -ne 0 -or (Read-Json $freshResult).WslState -ne 'Existing') { throw 'The machine phase must refresh PATH after installing Git and before cloning rc_files.' }
    Write-Output 'PASS: grouped/unchanged/elevated/cancelled UAC, credential boundary, user Python launcher, quoted paths, account identity, failures and cleanup.'
} finally {
    $env:LOCALAPPDATA = $savedLocal
    $resolved = [IO.Path]::GetFullPath($fixture)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notlike 'machine-setup-test-*') { throw 'Unsafe fixture cleanup path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
