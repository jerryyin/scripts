# Plugin choices and settings stay in the notes repository.
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Devenv.psm1') -DisableNameChecking

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

Export-ModuleMember -Function Get-VaultPath, Set-VaultRegistration, Install-ObsidianAddons, Resolve-ObsidianAddons, Install-LatestAddon
