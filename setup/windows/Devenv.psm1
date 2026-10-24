# Windows PowerShell 5.1 and PowerShell 7. No external modules required.
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'

function ConvertTo-Map($Value) {
    if ($null -eq $Value) { return $null }
    # ConvertFrom-Json can wrap root arrays/strings as PSObjects on PS 5.1.
    # Test concrete scalar/array types before the broad PSCustomObject check.
    if ($Value -is [string] -or $Value -is [ValueType]) { return $Value }
    if ($Value -is [array]) { return ,@($Value | ForEach-Object { ConvertTo-Map $_ }) }
    if ($Value -is [System.Collections.IDictionary]) {
        $result = [ordered]@{}
        foreach ($key in $Value.Keys) { $result[$key] = ConvertTo-Map $Value[$key] }
        return $result
    }
    if ($Value -is [pscustomobject]) {
        $result = [ordered]@{}
        foreach ($p in $Value.PSObject.Properties) { $result[$p.Name] = ConvertTo-Map $p.Value }
        return $result
    }
    return $Value
}

function Remove-JsonComments([string]$Text) {
    # Keep URLs, escaped quotes and comment-like strings intact (Terminal uses JSONC).
    $out = New-Object Text.StringBuilder
    $quoted = $false; $escaped = $false; $line = $false; $block = $false
    for ($i = 0; $i -lt $Text.Length; $i++) {
        $c = $Text[$i]; $next = if ($i + 1 -lt $Text.Length) { $Text[$i + 1] } else { [char]0 }
        if ($line) { if ($c -eq "`n") { $line = $false; [void]$out.Append($c) }; continue }
        if ($block) { if ($c -eq '*' -and $next -eq '/') { $block = $false; $i++ }; continue }
        if ($quoted) {
            [void]$out.Append($c)
            if ($escaped) { $escaped = $false } elseif ($c -eq '\') { $escaped = $true } elseif ($c -eq '"') { $quoted = $false }
            continue
        }
        if ($c -eq '"') { $quoted = $true }
        if ($c -eq '/' -and $next -eq '/') { $line = $true; $i++; continue }
        if ($c -eq '/' -and $next -eq '*') { $block = $true; $i++; [void]$out.Append(' '); continue }
        if ($c -eq ',') {
            $j = $i + 1
            while ($j -lt $Text.Length -and [char]::IsWhiteSpace($Text[$j])) { $j++ }
            if ($j -lt $Text.Length -and ($Text[$j] -eq '}' -or $Text[$j] -eq ']')) { continue }
        }
        [void]$out.Append($c)
    }
    return $out.ToString()
}

function Read-Json([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) { return [ordered]@{} }
    $mapped = ConvertTo-Map (ConvertFrom-Json (Remove-JsonComments ([IO.File]::ReadAllText($Path))))
    return $mapped
}

function Save-Text([string]$Path, [string]$Text, [string]$BackupRoot) {
    if ((Test-Path -LiteralPath $Path) -and [IO.File]::ReadAllText($Path) -ceq $Text) { return }
    if ((Test-Path -LiteralPath $Path) -and $BackupRoot) {
        [void][IO.Directory]::CreateDirectory($BackupRoot)
        $hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($Path))).Replace('-', '').Substring(0, 12)
        $backup = Join-Path $BackupRoot ((Split-Path $Path -Leaf) + '.' + $hash + '.' + [guid]::NewGuid().ToString('N'))
        Copy-Item -LiteralPath $Path -Destination $backup
    }
    [void][IO.Directory]::CreateDirectory((Split-Path $Path -Parent))
    $temp = $Path + '.tmp-' + [guid]::NewGuid().ToString('N')
    try {
        [IO.File]::WriteAllText($temp, $Text, (New-Object Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $temp -Destination $Path -Force
    } finally { if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp } }
}

function Test-JsonEquivalent($Left, $Right) {
    if ($null -eq $Left -or $null -eq $Right) { return $null -eq $Left -and $null -eq $Right }
    if ($Left -is [Collections.IDictionary] -or $Right -is [Collections.IDictionary]) {
        if ($Left -isnot [Collections.IDictionary] -or $Right -isnot [Collections.IDictionary] -or $Left.Count -ne $Right.Count) { return $false }
        foreach ($key in $Left.Keys) {
            if (!$Right.Contains($key) -or !(Test-JsonEquivalent $Left[$key] $Right[$key])) { return $false }
        }
        return $true
    }
    if ($Left -is [Collections.IList] -or $Right -is [Collections.IList]) {
        if ($Left -isnot [Collections.IList] -or $Right -isnot [Collections.IList] -or $Left.Count -ne $Right.Count) { return $false }
        for ($i = 0; $i -lt $Left.Count; $i++) {
            if (!(Test-JsonEquivalent $Left[$i] $Right[$i])) { return $false }
        }
        return $true
    }
    return (ConvertTo-Json -InputObject $Left -Compress) -ceq (ConvertTo-Json -InputObject $Right -Compress)
}

function Test-DeveloperMode {
    $key = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock' -ErrorAction SilentlyContinue
    return $null -ne $key -and $key.PSObject.Properties['AllowDevelopmentWithoutDevLicense'] -and $key.AllowDevelopmentWithoutDevLicense -eq 1
}

function Test-IsAdministrator {
    $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-DeveloperMode {
    if (Test-DeveloperMode) { Write-Host 'SKIP Developer Mode (enabled)'; return }
    if (Test-IsAdministrator) {
        $key = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock'
        [void](New-Item -Path $key -Force)
        [void](New-ItemProperty -LiteralPath $key -Name AllowDevelopmentWithoutDevLicense -PropertyType DWord -Value 1 -Force)
        if (!(Test-DeveloperMode)) { throw 'Developer Mode could not be enabled.' }
        Write-Host 'Developer Mode enabled.'
        return
    }
    $command = @'
$ErrorActionPreference = 'Stop'
try {
    $key = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock'
    [void](New-Item -Path $key -Force)
    [void](New-ItemProperty -LiteralPath $key -Name AllowDevelopmentWithoutDevLicense -PropertyType DWord -Value 1 -Force)
    exit 0
} catch { exit 1 }
'@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $powershell = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
    Write-Host 'Enabling Developer Mode for configuration symlinks; Windows may ask for administrator approval.'
    $process = Start-Process -FilePath $powershell -ArgumentList @('-NoProfile', '-NonInteractive', '-EncodedCommand', $encoded) -Verb RunAs -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0 -or !(Test-DeveloperMode)) { throw 'Developer Mode could not be enabled. Check administrator approval or device policy before rerunning.' }
    Write-Host 'Developer Mode enabled.'
}

function Get-MissingPackages([object[]]$Packages) {
    foreach ($package in $Packages) {
        if ($package.id -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or
            ($package.Contains('scope') -and $package.scope -notin @('user', 'machine'))) { throw 'Invalid package ID or installation scope.' }
        $scope = if ($package.Contains('scope')) { $package.scope } else { 'machine' }
        if ($scope -eq 'machine' -and $package.id -like 'Python.Python.*') {
            # CPython's Burn bundle can register in HKCU even for an all-users interpreter.
            $version = $package.id.Substring('Python.Python.'.Length)
            $key = Get-ItemProperty -LiteralPath "HKLM:\SOFTWARE\Python\PythonCore\$version\InstallPath" -ErrorAction SilentlyContinue
            if ($key -and (Test-Path -LiteralPath $key.ExecutablePath -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $env:SystemRoot 'py.exe') -PathType Leaf)) {
                Write-Host "SKIP $($package.id) (machine interpreter and launcher)"
            } else { $package }
            continue
        }
        if ($scope -eq 'user' -and $package.Contains('installedCommand') -and (Get-Command $package.installedCommand -CommandType Application -ErrorAction SilentlyContinue)) {
            Write-Host "SKIP $($package.id) (existing native command)"; continue
        }
        if ($scope -eq 'user' -and $package.Contains('installedAppxName') -and (Get-AppxPackage -Name $package.installedAppxName -ErrorAction SilentlyContinue)) {
            Write-Host "SKIP $($package.id) (Microsoft Store installation)"; continue
        }
        & winget list --id $package.id --exact --scope $scope --source winget --accept-source-agreements --disable-interactivity | Out-Null
        if ($LASTEXITCODE -eq -1978335212) { $package }
        elseif ($LASTEXITCODE -eq 0) { Write-Host "SKIP $($package.id) (installed)" }
        else { throw "WinGet inventory failed for $($package.id) (exit $LASTEXITCODE); refusing to treat a query error as a missing package." }
    }
}

function Install-WinGetPackage($Package) {
    $scope = if ($Package.Contains('scope')) { $Package.scope } else { 'machine' }
    $arguments = @('install', '--id', $Package.id, '--exact', '--scope', $scope, '--source', 'winget', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity', '--silent')
    if ($Package.Contains('installerArguments')) { $arguments += @('--custom', $Package.installerArguments) }
    # This package was missing in the requested scope; do not reuse a user installation.
    $arguments += '--force'
    Invoke-Native winget $arguments
    if (@(Get-MissingPackages @($Package)).Count) { throw "Package is still missing in $scope scope: $($Package.id)" }
}

function Ensure-SshAgent([switch]$ReadyOnly) {
    $agent = Get-Service ssh-agent -ErrorAction SilentlyContinue
    if (!$agent) { throw 'Install the Windows OpenSSH Client optional feature, then rerun setup.' }
    if ($agent.Status -eq 'Running' -and $agent.StartType -eq 'Automatic') { return }
    if ($ReadyOnly) { throw 'SSH agent requires machine setup. Rerun bootstrap to prepare it in the grouped administrator step.' }
    if (Test-IsAdministrator) {
        Set-Service ssh-agent -StartupType Automatic
        Start-Service ssh-agent
    } else {
        $command = '$ErrorActionPreference = "Stop"; Set-Service ssh-agent -StartupType Automatic; Start-Service ssh-agent; if ((Get-Service ssh-agent).Status -ne "Running") { exit 1 }'
        $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
        $process = Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -Wait -PassThru -ArgumentList @('-NoProfile', '-EncodedCommand', $encoded)
        if ($process.ExitCode -ne 0) { throw 'Could not enable ssh-agent.' }
    }
    if ((Get-Service ssh-agent).Status -ne 'Running') { throw 'SSH agent did not start.' }
}

function Invoke-MachineSetup($Plan) {
    if (!$Plan.DeveloperMode -and !$Plan.Distribution -and !$Plan.Packages.Count -and !($Plan.Contains('RcMachine') -and $Plan.RcMachine) -and !($Plan.Contains('Fonts') -and $Plan.Fonts) -and !$Plan.SshAgent -and !($Plan.Contains('SshShell') -and $Plan.SshShell) -and !($Plan.Contains('SshLinks') -and $Plan.SshLinks)) { return $Plan.WslState }
    $stateRoot = Join-Path $env:LOCALAPPDATA 'PersonalDevenv'
    $identifier = [guid]::NewGuid().ToString('N')
    $work = Join-Path $stateRoot ('machine-' + $identifier)
    $log = Join-Path $stateRoot ('logs/machine-' + $identifier + '.log')
    [void][IO.Directory]::CreateDirectory($work)
    [void][IO.Directory]::CreateDirectory((Split-Path $log -Parent))
    try {
        $requestPath = Join-Path $work 'request.json'
        $resultPath = Join-Path $work 'result.json'
        $Plan.UserSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
        $Plan.LogPath = $log
        Save-Json $requestPath $Plan ''
        $helper = (Join-Path $PSScriptRoot 'machine.ps1').Replace("'", "''")
        $command = "& '$helper' -RequestPath '$($requestPath.Replace("'", "''"))' -ResultPath '$($resultPath.Replace("'", "''"))'"
        $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
        $launch = @{ FilePath=(Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'); ArgumentList=@('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encoded); WindowStyle='Hidden'; Wait=$true; PassThru=$true }
        if (!(Test-IsAdministrator)) { $launch.Verb='RunAs' }
        Write-Host 'Preparing machine changes together; Windows will request administrator approval once if needed.'
        $process = Start-Process @launch
        if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log | Out-Host }
        $result = Read-Json $resultPath
        if ($process.ExitCode -ne 0) {
            $reason = if ($result.Contains('Error')) { $result.Error } else { "helper exit $($process.ExitCode); see $log" }
            throw "Machine setup failed: $reason"
        }
        if (!$result.Contains('WslState') -or $result.WslState -notin @('Existing','Skipped','Installed','RebootRequired','PendingInitialization')) { throw "Machine setup returned no valid result; see $log" }
        return $result.WslState
    } finally {
        $resolved = [IO.Path]::GetFullPath($work)
        $expected = [IO.Path]::GetFullPath($stateRoot).TrimEnd('\') + '\'
        if (!$resolved.StartsWith($expected, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -ne ('machine-' + $identifier)) { throw 'Unsafe machine request cleanup path' }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

function Invoke-RcSetup([string]$RcRoot, [string[]]$Arguments, [switch]$Capture, [switch]$AllowProblems) {
    $entry = Join-Path $RcRoot '.setup'
    if (!(Test-Path -LiteralPath $entry)) { throw "Update rc_files: setup tool is missing: $entry" }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    $prefix = @()
    if ($launcher) { $prefix = @('-3') }
    else { $launcher = Get-Command python.exe -ErrorAction Stop }
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& $launcher.Source @prefix $entry @Arguments 2>&1)
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($code -ne 0 -and !($AllowProblems -and $code -eq 1 -and $output.Count)) { throw "rc_files setup failed (exit $code): $($output -join "`n")" }
    if ($Capture) { return $output -join "`n" }
    $output | Out-Host
}

function Save-Json([string]$Path, $Value, [string]$BackupRoot) {
    if ((Test-Path -LiteralPath $Path) -and (Test-JsonEquivalent (Read-Json $Path) (ConvertTo-Map $Value))) { return }
    Save-Text $Path ((ConvertTo-Json -InputObject $Value -Depth 100) + "`n") $BackupRoot
}

function Get-RcMissingPackages([string]$RcRoot, [string]$Scope = '', [switch]$Update) {
    $arguments = @('prerequisites', '--json')
    if ($Scope) { $arguments += @('--scope', $Scope) }
    if ($Update) { $arguments += '--update' }
    $priorGit = $env:GIT_CONFIG_GLOBAL
    try {
        # Inventory must work before an unreadable .gitconfig link is repaired.
        $env:GIT_CONFIG_GLOBAL = 'NUL'
        $requirements = ConvertFrom-Json (Invoke-RcSetup $RcRoot $arguments -Capture)
    } finally { $env:GIT_CONFIG_GLOBAL = $priorGit }
    foreach ($requirement in $requirements) { $requirement }
}

function Get-DeploymentArguments([string]$Action, [string]$UserRoot, [string]$RcRoot, [string]$DocumentsPath, [string]$TerminalPath) {
    $arguments = @('--repo', $RcRoot, '--home', $UserRoot,
        '--root', "DOCUMENTS=$DocumentsPath", '--root', ('LOCALAPPDATA/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState=' + (Split-Path $TerminalPath -Parent)))
    if ($Action) { return @($Action) + $arguments }
    return $arguments
}

function Get-ConfigurationProblems([string]$UserRoot, [string]$RcRoot, [string]$DocumentsPath, [string]$TerminalPath) {
    $arguments = @(Get-DeploymentArguments 'check' $UserRoot $RcRoot $DocumentsPath $TerminalPath) + @('--json')
    $priorGit = $env:GIT_CONFIG_GLOBAL
    try {
        $env:GIT_CONFIG_GLOBAL = 'NUL'
        $report = ConvertFrom-Json (Invoke-RcSetup $RcRoot $arguments -Capture -AllowProblems)
    } finally { $env:GIT_CONFIG_GLOBAL = $priorGit }
    $report.problems
}

function Invoke-Native([string]$Program, [string[]]$Arguments, [switch]$Capture) {
    if ($Capture) { $result = & $Program @Arguments; $code = $LASTEXITCODE }
    else { & $Program @Arguments | Out-Host; $code = $LASTEXITCODE }
    if ($code -ne 0) { throw "$Program failed with exit code $code" }
    if ($Capture) { return (($result -join "`n") -replace "`0", '').Trim() }
}

function Invoke-GitQuiet([string]$Path, [string[]]$Arguments) {
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& git -C $Path @Arguments 2>&1)
        return [pscustomobject]@{ Code = $LASTEXITCODE; Output = (($output | ForEach-Object { "$_" }) -join "`n").Trim() }
    } finally { $ErrorActionPreference = $previousPreference }
}

# Clone a missing checkout; otherwise fast-forward it to its upstream (or the remote default
# branch when the upstream is gone), carrying local edits across in a stash. Upstream wins a
# conflict and the edits stay in `git stash list`. Failures to reach the remote only warn.
function Ensure-Checkout([string]$Path, [string]$Url) {
    if (Test-Path -LiteralPath (Join-Path $Path '.git')) {
        if ((Invoke-GitQuiet $Path @('fetch', '--quiet', '--prune', 'origin')).Code) {
            Write-Host "WARN Git checkout ${Path}: cannot reach origin; using it as is"
            return
        }
        $target = (Invoke-GitQuiet $Path @('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}')).Output
        if (!$target -or (Invoke-GitQuiet $Path @('rev-parse', '--verify', '--quiet', $target)).Code) {
            Invoke-GitQuiet $Path @('remote', 'set-head', 'origin', '--auto') | Out-Null
            $target = (Invoke-GitQuiet $Path @('symbolic-ref', '--quiet', '--short', 'refs/remotes/origin/HEAD')).Output
        }
        if (!$target) { Write-Host "WARN Git checkout ${Path}: no upstream branch; using it as is"; return }
        if ((Invoke-GitQuiet $Path @('rev-parse', 'HEAD')).Output -eq (Invoke-GitQuiet $Path @('rev-parse', $target)).Output -and
            (Invoke-GitQuiet $Path @('rev-parse', '--abbrev-ref', 'HEAD')).Output -eq ($target -replace '^[^/]+/', '')) {
            Write-Host "OK Git checkout $Path at $target"
            return
        }
        $stashed = $false
        if ((Invoke-GitQuiet $Path @('status', '--porcelain')).Output) {
            $stash = Invoke-GitQuiet $Path @('stash', 'push', '--include-untracked', '--quiet', '-m', 'bootstrap sync')
            if ($stash.Code) { throw "Cannot stash local edits in ${Path}: $($stash.Output)" }
            $stashed = $true
        }
        if ((Invoke-GitQuiet $Path @('merge-base', '--is-ancestor', 'HEAD', $target)).Code) {
            Write-Host "WARN Git checkout ${Path}: local commits are not on $target; branch left as is"
        } else {
            $branch = $target -replace '^[^/]+/', ''
            $update = Invoke-GitQuiet $Path @('checkout', '--quiet', '-B', $branch, '--track', $target)
            if ($update.Code) { throw "Cannot update ${Path} to ${target}: $($update.Output)" }
        }
        if ($stashed -and (Invoke-GitQuiet $Path @('stash', 'pop', '--quiet')).Code) {
            Invoke-GitQuiet $Path @('reset', '--hard', '--quiet') | Out-Null
            Write-Host "WARN Git checkout ${Path}: local edits conflict with $target; kept in git stash list"
        }
        Write-Host "OK Git checkout $Path at $target"
        return
    }
    if ((Test-Path -LiteralPath $Path) -and @(Get-ChildItem -LiteralPath $Path -Force).Count) { throw "Refusing to replace non-repository directory: $Path" }
    [void][IO.Directory]::CreateDirectory((Split-Path ([IO.Path]::GetFullPath($Path)) -Parent))
    Invoke-Native git @('-c', 'core.autocrlf=false', 'clone', '--config', 'core.autocrlf=false', '--', $Url, $Path)
}

function Invoke-WslCommand([string[]]$Arguments, [switch]$Elevate) {
    $command = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (!$command) { throw 'wsl.exe is unavailable. This bootstrap requires Windows 10 build 19041+ or Windows 11.' }
    if ($Elevate) {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            # Arguments are fixed flags plus a validated distribution identifier.
            # Elevate only WSL installation; keep user configuration in this account.
            $process = Start-Process -FilePath $command.Source -ArgumentList $Arguments -Verb RunAs -WindowStyle Hidden -Wait -PassThru
            return @{ ExitCode = $process.ExitCode; Output = '' }
        }
    }
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = & $command.Source @Arguments 2>&1
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previous }
    return @{ ExitCode = $code; Output = (($output -join "`n") -replace "`0", '').Trim() }
}

function Get-WslDistributions {
    $result = Invoke-WslCommand @('--list', '--quiet')
    if ($result.ExitCode -ne 0) { return @() }
    return @($result.Output -split '\r?\n' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}

function Ensure-Wsl([ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')][string]$Distribution) {
    if (@(Get-WslDistributions).Count -gt 0) { return 'Existing' }
    Write-Host "Installing WSL and $Distribution. Windows may request elevation and a restart."
    $result = Invoke-WslCommand @('--install', '--distribution', $Distribution, '--no-launch') -Elevate
    if ($result.ExitCode -in @(3010, 1641)) { return 'RebootRequired' }
    if ($result.ExitCode -ne 0) { throw "WSL installation failed ($($result.ExitCode)): $($result.Output)" }
    if (@(Get-WslDistributions).Count -gt 0) { return 'Installed' }
    # Some Windows versions finish registration only after restart/first launch.
    # Do not claim readiness, launch Linux, or automatically reboot the machine.
    return 'PendingInitialization'
}

function Update-ProcessPath {
    # Keep explicitly supplied tool paths (portable tests/custom installs), and
    # discover binaries added by installers without discarding inherited PATH.
    $paths = @([Environment]::GetEnvironmentVariable('Path', 'Machine'), $env:PATH, [Environment]::GetEnvironmentVariable('Path', 'User'))
    $env:PATH = ((($paths -join ';') -split ';' | Where-Object { $_ } | Select-Object -Unique) -join ';')
}

function Get-NativeVim {
    $command = Get-Command vim.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command -and $command.Source -notlike '*\Git\*') { return $command.Source }
    # Current installers default to per-user; older installs used Program Files.
    $roots = @((Join-Path $env:LOCALAPPDATA 'Programs/Vim'), (Join-Path $env:ProgramFiles 'Vim'))
    $x86 = [Environment]::GetEnvironmentVariable('ProgramFiles(x86)')
    if ($x86) { $roots += Join-Path $x86 'Vim' }
    foreach ($root in $roots) {
        $executable = Get-ChildItem -Path @((Join-Path $root 'vim.exe'), (Join-Path $root 'vim*/vim.exe')) -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
        if ($executable) { return $executable.FullName }
    }
    throw 'Native Vim could not be located. Install vim.vim or put native vim.exe on PATH.'
}

Export-ModuleMember -Function *
