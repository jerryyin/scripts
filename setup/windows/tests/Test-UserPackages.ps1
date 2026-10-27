param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '../Devenv.psm1') -Force -DisableNameChecking
Import-Module (Join-Path $RcRoot '.setup/packages.psm1') -Force -DisableNameChecking
$devenv = Get-Module Devenv
$packages = Get-Module packages
$global:UserPackageFixture = @{Machine=$true; User=$true; UserPython=''; Calls=@()}
try {
    foreach ($module in @($devenv, $packages)) {
        & $module {
            function script:winget {
                $state = $global:UserPackageFixture
                $state.Calls += ,$args
                if ($args[0] -eq 'uninstall') {
                    if ($args -notcontains 'user' -or !$state.Machine) { throw 'Unsafe uninstall.' }
                    $state.User = $false
                    $global:LASTEXITCODE = 0
                } else {
                    $present = if ($args -contains 'machine') { $state.Machine } else { $state.User }
                    $global:LASTEXITCODE = if ($present) { 0 } else { -1978335212 }
                }
            }
        }
    }
    & $packages {
        function script:Get-ItemProperty { param($Path, $ErrorAction) return @() }
    }
    & $devenv {
        function script:Get-ItemProperty {
            param($LiteralPath, $ErrorAction)
            if ($LiteralPath -like 'HKCU:*PythonCore*' -and $global:UserPackageFixture.UserPython) { return @{ExecutablePath=$global:UserPackageFixture.UserPython} }
            return $null
        }
    }
    $tool = @{id='Fixture.Shared'; scope='machine'}
    $global:UserPackageFixture.Machine = $false
    $failed = $false
    try { Remove-UserPackageDuplicates @($tool) $RcRoot }
    catch { $failed = $_.Exception.Message -like '*machine replacement*' }
    if (!$failed -or !$global:UserPackageFixture.User) { throw 'Bootstrap must verify the system app before removing its user copy.' }
    $global:UserPackageFixture.Machine = $true
    Remove-UserPackageDuplicates @($tool) $RcRoot
    if ($global:UserPackageFixture.User) { throw 'Bootstrap must remove duplicates of already-installed machine apps.' }
    $global:UserPackageFixture.User = $true
    $global:UserPackageFixture.Calls = @()
    $python = @{id='Python.Python.3.14'; scope='machine'}
    Remove-UserPackageDuplicates @($python) $RcRoot
    $global:UserPackageFixture.UserPython = Join-Path $env:ProgramFiles 'Python314/python.exe'
    Remove-UserPackageDuplicates @($python) $RcRoot
    if ($global:UserPackageFixture.Calls.Count -or !$global:UserPackageFixture.User) { throw 'HKCU bundle registration must not cause a system Python uninstall.' }
    $global:UserPackageFixture.UserPython = Join-Path $env:USERPROFILE 'AppData/Local/Programs/Python/Python314/python.exe'
    $failed = $false
    try { Remove-UserPackageDuplicates @($python) $RcRoot }
    catch { $failed = $_.Exception.Message -like '*machine replacement*' }
    if (!$failed -or !$global:UserPackageFixture.User) { throw 'A user Python must survive when the machine interpreter is missing or broken.' }
    Remove-UserPackageDuplicates @(@{id='Fixture.User'; scope='user'}) $RcRoot
    if (!$global:UserPackageFixture.User) { throw 'Explicitly user-scoped requirements must not be removed.' }
    Write-Output 'PASS bootstrap duplicate cleanup, working replacement, user scope and Python bundle ownership.'
} finally {
    Remove-Variable -Name UserPackageFixture -Scope Global
}
