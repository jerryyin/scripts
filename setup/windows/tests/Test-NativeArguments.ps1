$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot '../Devenv.psm1') -Force -DisableNameChecking
$module = Get-Module Devenv
$python = & py.exe -3 -c 'import sys; print(sys.executable)'
if ($LASTEXITCODE -ne 0) { throw 'This native argument check requires an installed Python.' }
& $module {
    param($Python)
    $script:ArgumentPython = $Python
    function script:winget {
        & $script:ArgumentPython -I -c 'import json,sys; print(json.dumps(sys.argv[1:]))' @args
    }
} $python
$custom = 'InstallAllUsers=1 TargetDir="C:\Program Files\Python314"'
$received = ConvertFrom-Json (Invoke-Native winget @('--custom', $custom) -Capture)
if ($received.Count -ne 2 -or $received[0] -ne '--custom' -or $received[1] -cne $custom) { throw "The native installer must receive one custom argument with its target quotes intact: $received" }
Write-Output 'PASS native WinGet arguments preserve the quoted Python target, including its spaces.'
