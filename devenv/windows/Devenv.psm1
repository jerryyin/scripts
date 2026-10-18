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

function Ensure-DeveloperMode {
    if (Test-DeveloperMode) { Write-Host 'SKIP Developer Mode (enabled)'; return }
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

function Set-TrackedFile([string]$Path, [string]$Source, [string]$BackupRoot) {
    $sourcePath = [IO.Path]::GetFullPath($Source)
    if (!(Test-Path -LiteralPath $sourcePath -PathType Leaf)) { throw "Tracked configuration is missing: $sourcePath" }
    $existing = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($existing -and $existing.PSIsContainer) { throw "Configuration destination is a directory: $Path" }
    if ($existing -and $existing.LinkType -eq 'SymbolicLink') {
        $target = [string](@($existing.Target)[0])
        if (![IO.Path]::IsPathRooted($target)) { $target = Join-Path $existing.DirectoryName $target }
        if ([IO.Path]::GetFullPath($target) -ieq $sourcePath) { return }
    }
    if (!('DevenvFileLinks' -as [type])) { Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class DevenvFileLinks {
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
 [return: MarshalAs(UnmanagedType.I1)]
 public static extern bool CreateSymbolicLinkW(string link, string target, int flags);
}
'@ }
    [void][IO.Directory]::CreateDirectory((Split-Path $Path -Parent))
    $candidate = $Path + '.link-' + [guid]::NewGuid().ToString('N')
    try {
        $linked = [DevenvFileLinks]::CreateSymbolicLinkW($candidate, $sourcePath, 2)
        $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        if (!$linked) { throw "Cannot link $Path to rc_files: $((New-Object ComponentModel.Win32Exception($code)).Message). Enable Developer Mode; configuration is never silently copied." }
        if ($existing) {
            if ($BackupRoot) {
                [void][IO.Directory]::CreateDirectory($BackupRoot)
                Copy-Item -LiteralPath $Path -Destination (Join-Path $BackupRoot ($existing.Name + '.' + [guid]::NewGuid().ToString('N')))
            }
            if ($existing.LinkType) { Remove-Item -LiteralPath $Path -Force }
        }
        Move-Item -LiteralPath $candidate -Destination $Path -Force
    } finally { if (Test-Path -LiteralPath $candidate) { Remove-Item -LiteralPath $candidate -Force } }
}

function Save-Json([string]$Path, $Value, [string]$BackupRoot) {
    if ((Test-Path -LiteralPath $Path) -and (Test-JsonEquivalent (Read-Json $Path) (ConvertTo-Map $Value))) { return }
    Save-Text $Path ((ConvertTo-Json -InputObject $Value -Depth 100) + "`n") $BackupRoot
}

function Set-TrackedDirectory([string]$Path, [string]$Source, [string]$BackupRoot) {
    $sourcePath = [IO.Path]::GetFullPath($Source).TrimEnd('\')
    if (!(Test-Path -LiteralPath $sourcePath -PathType Container)) { throw "Tracked directory is missing: $sourcePath" }
    $existing = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($existing -and $existing.LinkType -in @('Junction', 'SymbolicLink')) {
        if ([IO.Path]::GetFullPath([string](@($existing.Target)[0])).TrimEnd('\') -ieq $sourcePath) { return }
        throw "Directory already links elsewhere: $Path. Reconcile its contents with $sourcePath before rerunning."
    }
    if ($existing) {
        if (!$existing.PSIsContainer) { throw "Directory destination is a file: $Path" }
        $entries = @(Get-ChildItem -LiteralPath $Path -Recurse -Force)
        $originals = @(Get-ChildItem -LiteralPath $sourcePath -Recurse -Force)
        $different = $entries.Count -ne $originals.Count
        foreach ($entry in $entries) {
            $relative = $entry.FullName.Substring($existing.FullName.TrimEnd('\').Length + 1)
            $original = Join-Path $sourcePath $relative
            if ($entry.LinkType -or !(Test-Path -LiteralPath $original) -or $entry.PSIsContainer -ne (Get-Item -LiteralPath $original -Force).PSIsContainer) { $different = $true; break }
            if (!$entry.PSIsContainer -and (Get-FileHash -LiteralPath $entry.FullName).Hash -ne (Get-FileHash -LiteralPath $original).Hash) { $different = $true; break }
        }
        if ($different) { throw "Local skill edits need recovery: $Path differs from $sourcePath. Merge them into rc_files before rerunning; nothing was replaced." }
        if (!$BackupRoot) { throw "A backup directory is required to replace $Path with a junction." }
    }
    [void][IO.Directory]::CreateDirectory((Split-Path $Path -Parent))
    if ($existing) {
        [void][IO.Directory]::CreateDirectory($BackupRoot)
        Move-Item -LiteralPath $existing.FullName -Destination (Join-Path $BackupRoot ($existing.Name + '.' + [guid]::NewGuid().ToString('N')))
    }
    [void](New-Item -ItemType Junction -Path $Path -Target $sourcePath)
}

function Merge-Map($Base, $Desired) {
    foreach ($key in $Desired.Keys) {
        if ($Base.Contains($key) -and $Base[$key] -is [System.Collections.IDictionary] -and $Desired[$key] -is [System.Collections.IDictionary]) {
            Merge-Map $Base[$key] $Desired[$key]
        } else { $Base[$key] = $Desired[$key] }
    }
}

function Merge-Entries($Existing, $Desired, [string]$Key) {
    $items = New-Object Collections.ArrayList
    foreach ($entry in @($Existing)) { if ($null -ne $entry) { [void]$items.Add($entry) } }
    foreach ($entry in @($Desired)) {
        $found = -1
        for ($i = 0; $i -lt $items.Count; $i++) {
            if ($null -ne $items[$i][$Key] -and (ConvertTo-Json -Compress $items[$i][$Key]) -ceq (ConvertTo-Json -Compress $entry[$Key])) { $found = $i; break }
        }
        if ($found -ge 0) { $items[$found] = $entry } else { [void]$items.Add($entry) }
    }
    return ,$items.ToArray()
}

function Set-Terminal([string]$Path, $Defaults, [string]$ShellCommand, [string]$FontFace, [string]$BackupRoot) {
    $settings = Read-Json $Path
    foreach ($key in $Defaults.Keys) {
        if ($key -eq 'profiles') { continue }
        if ($key -eq 'actions') { $settings[$key] = Merge-Entries $settings[$key] $Defaults[$key] 'id' }
        elseif ($key -eq 'keybindings') { $settings[$key] = Merge-Entries $settings[$key] $Defaults[$key] 'keys' }
        elseif ($key -eq 'schemes') { $settings[$key] = Merge-Entries $settings[$key] $Defaults[$key] 'name' }
        else { $settings[$key] = $Defaults[$key] }
    }
    if (!$settings.Contains('profiles')) { $settings.profiles = [ordered]@{} }
    if (!$settings.profiles.Contains('defaults')) { $settings.profiles.defaults = [ordered]@{} }
    Merge-Map $settings.profiles.defaults $Defaults.profiles.defaults
    $settings.profiles.defaults.font.face = $FontFace
    $guid = $Defaults.defaultProfile
    $selected = @($Defaults.profiles.list | Where-Object { $_.guid -eq $guid })
    if ($selected.Count -ne 1) { throw 'Tracked Terminal settings must define the default profile exactly once.' }
    $profile = ConvertTo-Map (ConvertFrom-Json (ConvertTo-Json $selected[0] -Depth 30))
    $profile.commandline = $ShellCommand
    $profile.font.face = $FontFace
    $settings.profiles.list = Merge-Entries $settings.profiles['list'] @($profile) 'guid'
    # Keep one visible PowerShell, including when Terminal discovers PS7 later.
    foreach ($duplicate in @(
        [ordered]@{ guid='{61c54bbd-c2c6-5271-96e7-009a87ff44bf}'; name='Windows PowerShell' },
        [ordered]@{ guid='{574e775e-4f2a-5b96-ac1e-a2962a402336}'; name='PowerShell'; source='Windows.Terminal.PowershellCore' }
    )) {
        $found = @($settings.profiles.list | Where-Object { $_.guid -eq $duplicate.guid })
        if ($found.Count) { $found[0].hidden = $true }
        else { $duplicate.hidden = $true; $settings.profiles.list += $duplicate }
    }
    $settings.defaultProfile = $guid
    Save-Json $Path $settings $BackupRoot
}

function Set-PowerToys([string]$SettingsRoot, $Defaults, [string]$BackupRoot) {
    $generalPath = Join-Path $SettingsRoot 'settings.json'
    $general = Read-Json $generalPath
    Merge-Map $general $Defaults.general
    Save-Json $generalPath $general $BackupRoot

    $awakePath = Join-Path $SettingsRoot 'Awake/settings.json'
    $awake = Read-Json $awakePath
    # PowerToys' settings schema requires these fields on first launch.
    # Preserve an existing schema version instead of pinning an app version.
    if (!$awake.Contains('name')) { $awake.name = 'Awake' }
    if (!$awake.Contains('version')) { $awake.version = '1.0' }
    Merge-Map $awake $Defaults.awake
    Save-Json $awakePath $awake $BackupRoot
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

function Set-ProfileLoader([string]$Path, [string]$ManagedProfile, [string]$BackupRoot) {
    $begin = '# >>> personal-devenv >>>'; $end = '# <<< personal-devenv <<<'
    $text = if (Test-Path -LiteralPath $Path) { [IO.File]::ReadAllText($Path) } else { '' }
    $block = "$begin`n. '" + $ManagedProfile.Replace("'", "''") + "'`n$end"
    $pattern = '(?ms)^' + [regex]::Escape($begin) + '.*?^' + [regex]::Escape($end)
    $sourceText = [IO.File]::ReadAllText($ManagedProfile)
    if (!$text.Trim() -or $text -ceq $sourceText -or !([regex]::Replace($text, $pattern, '').Trim())) {
        Set-TrackedFile $Path $ManagedProfile $BackupRoot
        return
    }
    if ($text -match $pattern) { $text = [regex]::Replace($text, $pattern, [Text.RegularExpressions.MatchEvaluator]{ param($m) $block }) }
    else { $text = $text.TrimEnd() + "`n`n" + $block + "`n" }
    Save-Text $Path $text $BackupRoot
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
    if (@(Get-WslDistributions) -contains $Distribution) { return 'Existing' }
    Write-Host "Installing WSL and $Distribution. Windows may request elevation and a restart."
    $result = Invoke-WslCommand @('--install', '--distribution', $Distribution, '--no-launch') -Elevate
    if ($result.ExitCode -in @(3010, 1641)) { return 'RebootRequired' }
    if ($result.ExitCode -ne 0) { throw "WSL installation failed ($($result.ExitCode)): $($result.Output)" }
    if (@(Get-WslDistributions) -contains $Distribution) { return 'Installed' }
    # Some Windows versions finish registration only after restart/first launch.
    # Do not claim readiness, launch Linux, or automatically reboot the machine.
    return 'PendingInitialization'
}

function Install-PsmuxPlugins([string]$TmuxPath, [string]$UserRoot, [string]$StateRoot, [switch]$Update) {
    $source = [IO.File]::ReadAllText($TmuxPath)
    $plugins = @()
    if ($source -match '(?m)^set(?:-option)? -g @plugin\s+[''"]tmux-plugins/tpm[''"]') { $plugins += 'ppm' }
    if ($source -match '(?m)^set(?:-option)? -g @plugin\s+[''"]tmux-plugins/tmux-resurrect[''"]') { $plugins += 'psmux-resurrect' }
    $required = @{ ppm=@('ppm.ps1','scripts/install_plugins.ps1','scripts/update_plugins.ps1','scripts/clean_plugins.ps1'); 'psmux-resurrect'=@('psmux-resurrect.ps1','scripts/save.ps1','scripts/restore.ps1') }
    $pending = @($plugins | Where-Object {
        $name = $_
        @($required[$name] | Where-Object { !(Test-Path -LiteralPath (Join-Path $UserRoot ('.psmux/plugins/' + $name + '/' + $_))) }).Count
    })
    if (!$Update -and !$pending.Count) { Write-Host 'SKIP native psmux plugins (installed)'; return }
    if (!$plugins.Count) { return }
    $cache = Join-Path $StateRoot 'psmux-plugin-source'
    Ensure-Checkout $cache 'https://github.com/psmux/psmux-plugins.git'
    if ($Update) { Invoke-Native git @('-C',$cache,'pull','--ff-only') }
    foreach ($name in $plugins) {
        if (!$Update -and $pending -notcontains $name) { continue }
        $target = Join-Path $UserRoot ('.psmux/plugins/' + $name)
        [void][IO.Directory]::CreateDirectory($target)
        Copy-Item -Path (Join-Path $cache ($name + '/*')) -Destination $target -Recurse -Force
        Save-Json (Join-Path $target '.monorepo.json') ([ordered]@{monorepo='psmux/psmux-plugins'; org='psmux-plugins'; name=$name; url='https://github.com/psmux/psmux-plugins.git'; branch=$null}) ''
        foreach ($file in $required[$name]) {
            if (!(Test-Path -LiteralPath (Join-Path $target $file))) { throw "Incomplete psmux plugin: $name/$file" }
        }
    }
}

function Get-PsmuxConfiguration([string]$TmuxPath, [string]$TmuxRoot = '') {
    # Reuse portable settings directly from the existing Linux configuration.
    # Shell snippets, TPM and zsh startup commands belong to their OS adapter.
    $lines = New-Object Collections.ArrayList
    [void]$lines.Add('# Generated from rc_files/tmux/.tmux.conf; native PowerShell panes.')
    if (!$TmuxRoot) { $TmuxRoot = Split-Path $TmuxPath -Parent }
    $sections = @{}
    $section = ''
    foreach ($line in [IO.File]::ReadAllLines((Join-Path $TmuxRoot '.tmux/psmux-compat.conf'))) {
        if ($line -match '^# \[([^]]+)\]$') { $section = $Matches[1]; $sections[$section] = New-Object Collections.ArrayList }
        elseif ($section -and $line -and !$line.StartsWith('#')) { [void]$sections[$section].Add($line) }
    }
    foreach ($line in $sections.navigation) { [void]$lines.Add($line) }
    $source = [IO.File]::ReadAllText($TmuxPath) -replace '\\\r?\n\s*', ' '
    $nativePrefix = ([regex]::Match($source, '(?m)^set(?:-option)? -g prefix\s+([^\r\n]+)')).Groups[1].Value.Trim()
    foreach ($line in $source -split '\r?\n') {
        if ($line -match '^unbind(-key)? |^bind(-key)? |^set -g prefix |^(set|set-option|setw|set-window-option) -g (status-right|status-style|status-keys|mode-keys|mouse|status-justify|status|status-left-style|message-command-style|status-right-style|message-style|pane-active-border-style|pane-border-style|status-right-length|status-left-length|window-status-activity-style|window-status-separator|status-left|window-status-format|window-status-current-format) ') {
            $line = $line.Replace('set prefix None', 'set -g prefix None').Replace('set key-table off', 'set -g key-table off')
            $line = $line.Replace('set -u prefix', ('set -g prefix ' + $nativePrefix)).Replace('set -u key-table', 'set -g key-table root')
            $line = $line.Replace('~/.tmux.conf', '~/.psmux.conf') -replace '\s*\\;\s*', ' \; '
            $line = $line.Replace('$wg_is_keys_off', '')
            [void]$lines.Add($line)
        }
    }
    $resize = '5'
    if ($source -match '(?m)^set(?:-option)? -g @pane_resize\s+[''"]?(\d+)') { $resize = $Matches[1] }
    $names = @('native') + @([regex]::Matches($source, '(?m)^set(?:-option)? -g @plugin\s+[''"]tmux-plugins/([^''"]+)[''"]') | ForEach-Object { $_.Groups[1].Value })
    foreach ($name in $names) {
        if ($sections.ContainsKey($name)) {
            foreach ($line in $sections[$name]) {
                [void]$lines.Add($line.Replace('%PANE_RESIZE%', $resize).Replace('%MUX_SCRIPT%', ((Join-Path $TmuxRoot '.tmux/mux.ps1').Replace('\','/'))).Replace('%PLUGIN_ROOT%', ((Join-Path $env:USERPROFILE '.psmux/plugins').Replace('\','/'))))
            }
        }
    }
    return ($lines -join "`n") + "`n"
}

function Set-NativeMux([string]$UserRoot, [string]$RcRoot, [string]$BackupRoot) {
    $generated = Join-Path $UserRoot '.psmux/generated.conf'
    Save-Text $generated (Get-PsmuxConfiguration (Join-Path $RcRoot 'tmux/.tmux.conf')) $BackupRoot
    Set-TrackedFile (Join-Path $UserRoot '.psmux.conf') (Join-Path $RcRoot 'tmux/.psmux.conf') $BackupRoot
}

function Set-NativeVim([string]$UserRoot, [string]$RcRoot, [string]$GitBash, [string]$BackupRoot, [string]$PythonRoot = '') {
    if (!(Test-Path -LiteralPath (Join-Path $RcRoot 'vim/.vimrc'))) { throw 'Shared vim/.vimrc is missing from rc_files' }
    foreach ($relative in @('vim/_vimrc', 'vim/.vim/config/windows.vim', 'vim/.vim/config/plugins.vim', 'vim/.vim/config/leaderf-files.sh', 'vim/.vim/config/ai-processes.ps1', 'vim/.vim/coc-settings.json')) {
        if (!(Test-Path -LiteralPath (Join-Path $RcRoot $relative))) { throw "Update rc_files: required shared configuration is missing: $relative" }
    }
    $pythonDll = ''
    if ($PythonRoot) {
        $dll = Get-ChildItem -LiteralPath $PythonRoot -Filter 'python*.dll' -File | Where-Object { $_.Name -match '^python[0-9]{2,}\.dll$' } | Select-Object -First 1
        if (!$dll) { throw 'The native Python runtime DLL could not be found' }
        $pythonDll = $dll.FullName
    }
    Save-Json (Join-Path $UserRoot '.vim/machine-paths.json') ([ordered]@{
        gitBash=$GitBash.Replace('\','/'); pythonRoot=$PythonRoot.Replace('\','/'); pythonDll=$pythonDll.Replace('\','/'); vimrc=(Join-Path $RcRoot 'vim/.vimrc').Replace('\','/')
    }) $BackupRoot
    Set-TrackedFile (Join-Path $UserRoot '_vimrc') (Join-Path $RcRoot 'vim/_vimrc') $BackupRoot
    [void][IO.Directory]::CreateDirectory((Join-Path $UserRoot '.vim/session'))
}

function Install-VimPlugins([string]$VimExe, [string]$UserRoot, [string]$BackupRoot, [string]$RcRoot, [switch]$Update) {
    $plugPath = Join-Path $UserRoot '.vim/autoload/plug.vim'
    [void][IO.Directory]::CreateDirectory((Split-Path $plugPath -Parent))
    $download = $plugPath + '.download'
    if ($Update -or !(Test-Path -LiteralPath $plugPath)) { try {
        Get-Download 'https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim' $download
        Save-Text $plugPath ([IO.File]::ReadAllText($download)) $BackupRoot
    } finally { if (Test-Path -LiteralPath $download) { Remove-Item -LiteralPath $download } } }
    $choices = Join-Path $UserRoot '.vim/coc-extensions.json'
    $previous = @($env:DEVENV_RC_ROOT, $env:DEVENV_VIM_HOME, $env:DEVENV_UPDATE)
    try {
        $env:DEVENV_RC_ROOT = $RcRoot; $env:DEVENV_VIM_HOME = $UserRoot; $env:DEVENV_UPDATE = [string][int][bool]$Update
        Invoke-Native $VimExe @('-n', '-E', '-s', ('-V1' + (Join-Path $UserRoot '.vim/plugin-install.log')), '-u', 'NONE', '-S', (Join-Path $PSScriptRoot 'install-vim-plugins.vim'))
    } finally { $env:DEVENV_RC_ROOT=$previous[0]; $env:DEVENV_VIM_HOME=$previous[1]; $env:DEVENV_UPDATE=$previous[2] }
    $extensions = Join-Path $UserRoot '.config/coc/extensions'
    $packagePath = Join-Path $extensions 'package.json'
    $changed = !(Test-Path -LiteralPath $packagePath)
    $package = Read-Json $packagePath
    if (!$package.Contains('dependencies')) { $package.dependencies = [ordered]@{} }
    $pending = $false
    foreach ($extension in @(Read-Json $choices)) {
        if (!$package.dependencies.Contains($extension)) { $package.dependencies[$extension] = 'latest'; $changed = $true }
        $installed = Join-Path $extensions "node_modules/$extension/package.json"
        if (!(Test-Path -LiteralPath $installed)) { $pending = $true }
    }
    if ($changed) { Save-Json $packagePath $package $BackupRoot }
    if ($Update -or $pending) {
        # On Windows, npm install --prefix <dir> with no package arguments can
        # mistake that prefix for a global install and install the caller's cwd.
        Push-Location -LiteralPath $extensions
        try {
            $command = if ($Update) { 'update' } else { 'install' }
            Invoke-Native npm.cmd @($command, '--no-package-lock')
        } finally { Pop-Location }
    } else { Write-Host 'SKIP CoC extensions (installed)' }
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

function Set-VimShortcuts([string]$VimExe, [string]$DesktopPath, [string]$ProgramsPath, [string]$BackupRoot) {
    $shell = New-Object -ComObject WScript.Shell
    $vimRoot = Split-Path ([IO.Path]::GetFullPath($VimExe)) -Parent
    $launcher = Join-Path $ProgramsPath 'Vim.lnk'
    if (Test-Path -LiteralPath $launcher) {
        $existing = $shell.CreateShortcut($launcher)
        if ($existing.TargetPath -ine $VimExe -or $existing.Arguments) { throw "Preserve existing custom Vim shortcut: $launcher" }
    }
    $candidates = @()
    if (Test-Path -LiteralPath $DesktopPath) {
        $candidates += @(Get-ChildItem -LiteralPath $DesktopPath -File -Filter '*.lnk' |
            Where-Object Name -match '^gVim( Easy| Read-only)? [0-9]+\.[0-9]+\.lnk$')
    }
    $folders = @(Get-ChildItem -LiteralPath $ProgramsPath -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^Vim [0-9]+\.[0-9]+$' -and !($_.Attributes -band [IO.FileAttributes]::ReparsePoint) })
    $names = @('gVim.lnk', 'gVim Easy.lnk', 'gVim Read-only.lnk', 'gVim Diff.lnk', 'Vim.lnk', 'Vim Read-only.lnk', 'Vim Diff.lnk', 'Help.lnk', 'Vim tutor.lnk', 'Uninstall.lnk')
    foreach ($folder in $folders) {
        $candidates += @(Get-ChildItem -LiteralPath $folder.FullName -File | Where-Object { $names -contains $_.Name -or $_.Name -eq 'Vim Online.url' })
    }
    $removed = 0
    foreach ($file in $candidates) {
        if ($file.Extension -eq '.url') {
            if ([IO.File]::ReadAllText($file.FullName) -notmatch '(?m)^URL=https://www\.vim\.org/?\r?$') { continue }
        } else {
            $shortcut = $shell.CreateShortcut($file.FullName)
            if ((Split-Path $shortcut.TargetPath -Parent) -ine $vimRoot) { continue }
        }
        $relative = if ($file.DirectoryName -ieq $DesktopPath) { 'Desktop/' + $file.Name } else { 'StartMenu/' + $file.Directory.Name + '/' + $file.Name }
        $backup = Join-Path $BackupRoot ('vim-shortcuts/' + $relative)
        [void][IO.Directory]::CreateDirectory((Split-Path $backup -Parent))
        Copy-Item -LiteralPath $file.FullName -Destination $backup
        Remove-Item -LiteralPath $file.FullName
        $removed++
    }
    foreach ($folder in $folders) {
        if (!@(Get-ChildItem -LiteralPath $folder.FullName -Force).Count) { Remove-Item -LiteralPath $folder.FullName }
    }
    if (!(Test-Path -LiteralPath $launcher)) {
        [void][IO.Directory]::CreateDirectory($ProgramsPath)
        $shortcut = $shell.CreateShortcut($launcher)
        $shortcut.TargetPath = $VimExe
        $shortcut.WorkingDirectory = $env:USERPROFILE
        $shortcut.Save()
    }
    Write-Host "Vim shortcuts: one terminal launcher; $removed extra shortcut(s) backed up and removed."
}

function Set-NativeCodex([string]$UserRoot, [string]$RcRoot, [string]$BackupRoot) {
    $target = Join-Path $UserRoot '.codex/config.toml'
    $template = Join-Path $RcRoot 'codex/.codex.config.toml.template'
    if (!(Test-Path -LiteralPath $target) -and (Test-Path -LiteralPath $template)) {
        $text = [IO.File]::ReadAllText($template).Replace('/home/zhuoryin', $UserRoot.Replace('\', '\\'))
        Save-Text $target $text $BackupRoot
    }
    $instructions = 'claude/.claude/CLAUDE.md'
    if (Test-Path -LiteralPath (Join-Path $RcRoot $instructions)) {
        $destination = Join-Path $UserRoot '.codex/AGENTS.md'
        $source = Join-Path $RcRoot $instructions
        $existing = Get-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue
        if ($existing -and !$existing.LinkType -and (Get-FileHash -LiteralPath $destination).Hash -ne (Get-FileHash -LiteralPath $source).Hash) {
            throw "Local instruction edits need recovery: $destination differs from $source. Merge them into rc_files before rerunning."
        }
        Set-TrackedFile $destination $source $BackupRoot
    }
    $skillsSource = Join-Path $RcRoot 'claude/.claude/skills'
    if (Test-Path -LiteralPath $skillsSource) {
        $skillsTarget = Join-Path $UserRoot '.agents/skills'
        [void][IO.Directory]::CreateDirectory($skillsTarget)
        $tracked = @(& git -C $RcRoot ls-files -- 'claude/.claude/skills/')
        if ($LASTEXITCODE -ne 0) { throw 'Cannot list tracked rc_files skills.' }
        $names = @($tracked | ForEach-Object { $_.Substring('claude/.claude/skills/'.Length).Split('/')[0] } | Sort-Object -Unique)
        foreach ($name in $names) {
            Set-TrackedDirectory (Join-Path $skillsTarget $name) (Join-Path $skillsSource $name) $BackupRoot
        }
    }
}

function Get-FontFiles($Font, [string]$Directory) {
    [void][IO.Directory]::CreateDirectory($Directory)
    $release = Get-LatestRelease $Font.repository
    $asset = @($release.assets | Where-Object { $_.name -ceq $Font.asset })
    if ($asset.Count -ne 1) { throw "Latest font release is missing $($Font.asset)" }
    $archivePath = Join-Path $Directory 'font.zip'
    Get-Download $asset[0].browser_download_url $archivePath
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        foreach ($entry in $archive.Entries) {
            if ($entry.FullName -match '(^|[/\\])\.\.([/\\]|$)|^[/\\]|:') { throw 'Unsafe font archive entry' }
            if ($entry.Name -notmatch '\.ttf$') { continue }
            if ($entry.Name -notmatch '^[A-Za-z0-9._-]+\.ttf$') { throw 'Invalid font filename' }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $Directory $entry.Name), $true)
        }
    } finally { $archive.Dispose() }
    if (!@(Get-ChildItem -LiteralPath $Directory -Filter '*.ttf').Count) { throw 'Latest font archive contains no TrueType fonts' }
}

function Install-Fonts($Font, [string]$BackupRoot) {
    Add-Type -AssemblyName System.Drawing
    $installed = @(New-Object Drawing.Text.InstalledFontCollection).Families.Name
    if ($installed -contains $Font.preferredFace) { return $Font.preferredFace }
    $target = Join-Path $env:LOCALAPPDATA 'Microsoft/Windows/Fonts'
    [void][IO.Directory]::CreateDirectory($target)
    $key = 'HKCU:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts'
    [void](New-Item -Path $key -Force)
    if (!('DevenvFonts' -as [type])) { Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class DevenvFonts {
 [DllImport("gdi32.dll", CharSet=CharSet.Unicode)] public static extern int AddFontResourceEx(string path, uint flags, IntPtr reserved);
 [DllImport("user32.dll")] public static extern IntPtr SendMessageTimeout(IntPtr h, uint m, IntPtr w, IntPtr l, uint flags, uint timeout, out IntPtr result);
}
'@ }
    $staging = Join-Path ([IO.Path]::GetTempPath()) ('devenv-font-' + [guid]::NewGuid().ToString('N'))
    try {
        Get-FontFiles $Font $staging
        $collection = New-Object Drawing.Text.PrivateFontCollection
        try {
            foreach ($fontFile in Get-ChildItem -LiteralPath $staging -Filter '*.ttf') { $collection.AddFontFile($fontFile.FullName) }
            if ($collection.Families.Name -notcontains $Font.preferredFace) { throw 'Downloaded font does not contain the requested family' }
        } finally { $collection.Dispose() }
        foreach ($fontFile in Get-ChildItem -LiteralPath $staging -Filter '*.ttf') {
            $file = Join-Path $target $fontFile.Name
            Copy-Item -LiteralPath $fontFile.FullName -Destination $file -Force
            if ([DevenvFonts]::AddFontResourceEx($file, 0, [IntPtr]::Zero) -eq 0) { throw "Font registration failed: $file" }
            [void](New-ItemProperty -Path $key -Name ($fontFile.BaseName + ' (TrueType)') -Value $file -PropertyType String -Force)
        }
    } finally {
        $resolved = [IO.Path]::GetFullPath($staging)
        $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
        if (!$resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notlike 'devenv-font-*') { throw 'Unsafe font cleanup path' }
        if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
    }
    $result = [IntPtr]::Zero
    [void][DevenvFonts]::SendMessageTimeout([IntPtr]0xffff, 0x001D, [IntPtr]::Zero, [IntPtr]::Zero, 2, 1000, [ref]$result)
    return $Font.preferredFace
}

Export-ModuleMember -Function *
