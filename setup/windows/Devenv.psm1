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
        if ($package.Contains('installedCommand') -and (Get-Command $package.installedCommand -CommandType Application -ErrorAction SilentlyContinue)) {
            Write-Host "SKIP $($package.id) (existing native command)"; continue
        }
        if ($package.Contains('installedAppxName') -and (Get-AppxPackage -Name $package.installedAppxName -ErrorAction SilentlyContinue)) {
            Write-Host "SKIP $($package.id) (Microsoft Store installation)"; continue
        }
        & winget list --id $package.id --exact --source winget --accept-source-agreements --disable-interactivity | Out-Null
        if ($LASTEXITCODE -eq -1978335212) { $package }
        elseif ($LASTEXITCODE -eq 0) { Write-Host "SKIP $($package.id) (installed)" }
        else { throw "WinGet inventory failed for $($package.id) (exit $LASTEXITCODE); refusing to treat a query error as a missing package." }
    }
}

function Install-WinGetPackage($Package) {
    $scope = if ($Package.Contains('scope')) { $Package.scope } else { 'user' }
    $arguments = @('install', '--id', $Package.id, '--exact', '--scope', $scope, '--source', 'winget', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity')
    if ($Package.Contains('installerArguments')) { $arguments += @('--custom', $Package.installerArguments) }
    Invoke-Native winget $arguments
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
    if (!$Plan.DeveloperMode -and !$Plan.Distribution -and !$Plan.Packages.Count -and !($Plan.Contains('RcMachine') -and $Plan.RcMachine) -and !$Plan.SshAgent) { return $Plan.WslState }
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

function Get-RcMissingPackages([string]$RcRoot, [string]$Scope = '') {
    $arguments = @('prerequisites', '--json')
    if ($Scope) { $arguments += @('--scope', $Scope) }
    $requirements = ConvertFrom-Json (Invoke-RcSetup $RcRoot $arguments -Capture)
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
    $report = ConvertFrom-Json (Invoke-RcSetup $RcRoot $arguments -Capture -AllowProblems)
    $report.problems
}

function Get-VaultPath([string]$RegistryPath, [string]$Fallback) {
    $registry = Read-Json $RegistryPath
    if ($registry.Contains('vaults')) {
        foreach ($v in $registry.vaults.Values) {
            if ((Split-Path $v.path -Leaf) -eq 'notes' -and (Test-Path -LiteralPath (Join-Path $v.path '.git'))) { return $v.path }
        }
    }
    return $Fallback
}

function Set-VaultRegistration([string]$RegistryPath, [string]$VaultPath, [string]$BackupRoot) {
    $registry = Read-Json $RegistryPath
    if (!$registry.Contains('vaults')) { $registry.vaults = [ordered]@{} }
    foreach ($vault in $registry.vaults.Values) { if ($vault.path -eq $VaultPath) { return } }
    $digest = [Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($VaultPath.ToLowerInvariant()))
    $key = [BitConverter]::ToString($digest).Replace('-', '').Substring(0, 16).ToLowerInvariant()
    $registry.vaults[$key] = [ordered]@{ path = $VaultPath; open = $true; ts = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() }
    Save-Json $RegistryPath $registry $BackupRoot
}

function Get-Download([string]$Url, [string]$Destination) {
    if ($Url -notmatch '^https://') { throw "HTTPS required: $Url" }
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing | Out-Null
}

function Get-WebJson([string]$Url) {
    if ($Url -notmatch '^https://') { throw "HTTPS required: $Url" }
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -Headers @{ 'User-Agent' = 'personal-devenv' }
    $content = if ($response.Content -is [byte[]]) { [Text.Encoding]::UTF8.GetString($response.Content) } else { [string]$response.Content }
    $mapped = ConvertTo-Map (ConvertFrom-Json $content)
    return $mapped
}

function Get-LatestRelease([string]$Repository) {
    if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { throw "Invalid GitHub repository: $Repository" }
    return Get-WebJson "https://api.github.com/repos/$Repository/releases/latest"
}

function Resolve-ObsidianAddons {
    param([string]$NotesPath, [string[]]$PluginIds, [switch]$SkipTheme)
    # Plugin selection and appearance are owned by the Git-tracked vault.
    $ids = @(Read-Json (Join-Path $NotesPath '.obsidian/community-plugins.json'))
    if ($PSBoundParameters.ContainsKey('PluginIds')) { $ids = @($ids | Where-Object { $PluginIds -contains $_ }) }
    $registry = if ($ids.Count) { @(Get-WebJson 'https://raw.githubusercontent.com/obsidianmd/obsidian-releases/master/community-plugins.json') } else { @() }
    foreach ($id in $ids) {
        if ($id -notmatch '^[A-Za-z0-9_-]+$') { throw "Invalid plugin id: $id" }
        $entry = @($registry | Where-Object { $_.id -ceq $id })
        if ($entry.Count -ne 1) { throw "Plugin $id has no unique entry in Obsidian's community registry" }
        $repo = $entry[0].repo
        if ($repo -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { throw "Invalid plugin repository: $repo" }
        # Obsidian publishers advertise the current plugin version in the root
        # manifest. GitHub's 'latest' can instead point to another plugin/channel
        # (Calendar's latest release, for example, identifies as calendar-beta).
        $published = Get-WebJson "https://raw.githubusercontent.com/$repo/HEAD/manifest.json"
        if ($published.id -cne $id) { throw "Published plugin identity does not match $id" }
        $tag = [Uri]::EscapeDataString($published.version)
        $release = Get-WebJson "https://api.github.com/repos/$repo/releases/tags/$tag"
        $assets = @($release.assets | Where-Object { $_.name -in @('manifest.json', 'main.js', 'styles.css') } | ForEach-Object { @{ name=$_.name; url=$_.browser_download_url } })
        $manifestAsset = @($assets | Where-Object { $_.name -eq 'manifest.json' })
        if ($manifestAsset.Count -ne 1 -or @($assets | Where-Object { $_.name -eq 'main.js' }).Count -ne 1) { throw "Latest release of $id is missing manifest.json/main.js" }
        $manifest = Get-WebJson $manifestAsset[0].url
        if ($manifest.id -cne $id -or $manifest.version -cne $published.version) { throw "Published release does not match $id $($published.version)" }
        Write-Host "Resolved $id $($manifest.version) (latest release)"
        @{ kind='plugin'; id=$id; version=$manifest.version; assets=$assets }
    }
    $appearance = Read-Json (Join-Path $NotesPath '.obsidian/appearance.json')
    if (!$SkipTheme -and $appearance.Contains('cssTheme') -and $appearance.cssTheme) {
        $themes = @(Get-WebJson 'https://raw.githubusercontent.com/obsidianmd/obsidian-releases/master/community-css-themes.json')
        $entry = @($themes | Where-Object { $_.name -ceq $appearance.cssTheme })
        if ($entry.Count -ne 1) { throw "Theme $($appearance.cssTheme) is absent from Obsidian's community registry" }
        $repo = $entry[0].repo
        if ($repo -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { throw 'Invalid theme repository' }
        if ($appearance.cssTheme -match '[/\\]' -or $appearance.cssTheme -in @('.', '..')) { throw 'Invalid theme directory name' }
        # Resolve HEAD now so both theme files come from one current snapshot.
        # This revision is never written as a dependency lock.
        $head = Get-WebJson "https://api.github.com/repos/$repo/commits/HEAD"
        $base = "https://raw.githubusercontent.com/$repo/$($head.sha)"
        $manifest = Get-WebJson "$base/manifest.json"
        @{ kind='theme'; id=$appearance.cssTheme; version=$manifest.version; assets=@(@{name='manifest.json'; url="$base/manifest.json"}, @{name='theme.css'; url="$base/theme.css"}) }
    }
}

function Install-ObsidianAddons([string]$NotesPath, [string]$BackupRoot, [switch]$Update) {
    $pending = @()
    foreach ($id in @(Read-Json (Join-Path $NotesPath '.obsidian/community-plugins.json'))) {
        if ($id -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid Obsidian plugin identifier' }
        $directory = Join-Path $NotesPath ".obsidian/plugins/$id"
        $complete = $false
        try {
            $manifest = Read-Json (Join-Path $directory 'manifest.json')
            $complete = $manifest.Contains('id') -and $manifest.id -ceq $id -and $manifest.Contains('version') -and (Test-Path -LiteralPath (Join-Path $directory 'main.js'))
        } catch { $complete = $false }
        if ($Update -or !$complete) { $pending += $id } else { Write-Host "SKIP Obsidian plugin $id (installed)" }
    }
    $skipTheme = !$Update
    $appearance = Read-Json (Join-Path $NotesPath '.obsidian/appearance.json')
    if ($appearance.Contains('cssTheme') -and $appearance.cssTheme) {
        if ($appearance.cssTheme -match '[/\\]' -or $appearance.cssTheme -in @('.', '..')) { throw 'Invalid theme directory name' }
        $directory = Join-Path $NotesPath ".obsidian/themes/$($appearance.cssTheme)"
        try { $theme = Read-Json (Join-Path $directory 'manifest.json'); $complete = $theme.Contains('version') -and (Test-Path -LiteralPath (Join-Path $directory 'theme.css')) }
        catch { $complete = $false }
        if (!$complete) { $skipTheme = $false } elseif (!$Update) { Write-Host "SKIP Obsidian theme $($appearance.cssTheme) (installed)" }
    }
    if (!$pending.Count -and $skipTheme) { return }
    foreach ($addon in @(Resolve-ObsidianAddons $NotesPath -PluginIds $pending -SkipTheme:$skipTheme)) {
        $kind = if ($addon.kind -eq 'theme') { 'themes' } else { 'plugins' }
        Install-LatestAddon $addon (Join-Path $NotesPath ".obsidian/$kind/$($addon.id)") $BackupRoot
    }
}

function Install-LatestAddon($Addon, [string]$Directory, [string]$BackupRoot) {
    $manifestPath = Join-Path $Directory 'manifest.json'
    $mainName = if ($Addon.kind -eq 'theme') { 'theme.css' } else { 'main.js' }
    if ((Test-Path -LiteralPath $manifestPath) -and (Test-Path -LiteralPath (Join-Path $Directory $mainName))) {
        $installed = Read-Json $manifestPath
        if ($Addon.kind -ne 'theme' -and $installed.id -cne $Addon.id) { throw "Unexpected plugin identity at $Directory" }
        $complete = @($Addon.assets | Where-Object { !(Test-Path -LiteralPath (Join-Path $Directory $_.name)) }).Count -eq 0
        # Themes can change their CSS without bumping the manifest version.
        if ($Addon.kind -eq 'plugin' -and $installed.version -ceq $Addon.version -and $complete) { return }
    }
    $stage = Join-Path ([IO.Path]::GetTempPath()) ('devenv-addon-' + [guid]::NewGuid().ToString('N'))
    [void][IO.Directory]::CreateDirectory($stage)
    try {
        foreach ($asset in $Addon.assets) {
            if ($asset.name -notin @('manifest.json', 'main.js', 'styles.css', 'theme.css')) { throw 'Unexpected addon filename' }
            Get-Download $asset.url (Join-Path $stage $asset.name)
        }
        $manifest = Read-Json (Join-Path $stage 'manifest.json')
        if ($manifest.version -ne $Addon.version -or ($Addon.kind -ne 'theme' -and $manifest.id -ne $Addon.id)) { throw "Downloaded manifest does not match $($Addon.id) $($Addon.version)" }
        foreach ($file in Get-ChildItem -LiteralPath $stage -File) {
            Save-Text (Join-Path $Directory $file.Name) ([IO.File]::ReadAllText($file.FullName)) $BackupRoot
        }
        $oldStyles = Join-Path $Directory 'styles.css'
        if ($Addon.kind -eq 'plugin' -and !(Test-Path -LiteralPath (Join-Path $stage 'styles.css')) -and (Test-Path -LiteralPath $oldStyles)) {
            Save-Text $oldStyles '' $BackupRoot
            Remove-Item -LiteralPath $oldStyles
        }
    } finally {
        $resolved = [IO.Path]::GetFullPath($stage)
        $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
        if (!$resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notlike 'devenv-addon-*') { throw 'Unsafe addon staging cleanup path' }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}

function Invoke-Native([string]$Program, [string[]]$Arguments, [switch]$Capture) {
    if ($Capture) { $result = & $Program @Arguments; $code = $LASTEXITCODE }
    else { & $Program @Arguments | Out-Host; $code = $LASTEXITCODE }
    if ($code -ne 0) { throw "$Program failed with exit code $code" }
    if ($Capture) { return (($result -join "`n") -replace "`0", '').Trim() }
}

function Ensure-Checkout([string]$Path, [string]$Url) {
    if (Test-Path -LiteralPath (Join-Path $Path '.git')) {
        Invoke-Native git @('-C', $Path, 'rev-parse', '--verify', 'HEAD') -Capture | Out-Null
        Write-Host "SKIP Git checkout $Path (existing work preserved)"
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

function Install-CodexAdapter([switch]$Update) {
    $prefix = Invoke-Native npm.cmd @('prefix', '--global') -Capture
    $package = Join-Path $prefix 'node_modules/@agentclientprotocol/codex-acp/package.json'
    if (!$Update -and (Test-Path -LiteralPath $package) -and (Test-Path -LiteralPath (Join-Path $prefix 'codex-acp.cmd'))) {
        Write-Host 'SKIP Codex ACP adapter (installed)'
        return
    }
    Invoke-Native npm.cmd @('install', '--global', '@agentclientprotocol/codex-acp@latest')
}

function Update-ProcessPath {
    # Keep explicitly supplied tool paths (portable tests/custom installs), and
    # discover binaries added by installers without discarding inherited PATH.
    $paths = @($env:PATH, [Environment]::GetEnvironmentVariable('Path', 'Machine'), [Environment]::GetEnvironmentVariable('Path', 'User'))
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
