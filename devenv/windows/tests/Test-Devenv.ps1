param([switch]$Network)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('personal-devenv-test-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($testRoot)
$script:passed = 0
function Assert($Condition, [string]$Message) {
    if (!$Condition) { throw "FAIL: $Message" }
    $script:passed++
    Write-Output "PASS: $Message"
}
try {
    foreach ($file in Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object { $_.Extension -in @('.ps1', '.psm1') }) {
        $tokens = $null; $errors = $null
        [void][Management.Automation.Language.Parser]::ParseFile($file.FullName, [ref]$tokens, [ref]$errors)
        Assert ($errors.Count -eq 0) "PowerShell syntax: $($file.Name)"
    }
    $plan = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'bootstrap.ps1') 2>&1
    Assert ($LASTEXITCODE -eq 0 -and ($plan -join "`n") -match 'PLAN: install') 'Documented powershell -File entry point resolves its default manifest'
    $defaults = Read-Json (Join-Path $root 'defaults/terminal.json')
    $arrayPath = Join-Path $testRoot 'plugins.json'
    Save-Text $arrayPath '["calendar", "copilot"]' ''
    $ids = @(Read-Json $arrayPath)
    Assert ($ids.Count -eq 2 -and $ids[0] -ceq 'calendar' -and $ids[1] -ceq 'copilot') 'Root JSON string arrays work on Windows PowerShell 5.1'
    Save-Text $arrayPath '["calendar"]' ''
    Assert (@(Read-Json $arrayPath).Count -eq 1) 'Single-plugin JSON arrays remain iterable'
    $terminal = Join-Path $testRoot 'Terminal/settings.json'
    Set-Terminal $terminal $defaults 'pwsh.exe -NoLogo' 'Test Font' (Join-Path $testRoot 'backups')
    $result = Read-Json $terminal
    Assert ($result.profiles.list.Count -eq 1) 'Fresh Terminal receives one development profile'
    Assert ($result.profiles.defaults.font.face -eq 'Test Font') 'Terminal uses the installed font'
    Assert ($result.profiles.list[0].commandline -eq 'pwsh.exe -NoLogo') 'Development profile is native PowerShell'
    Assert ($result.profiles.list[0].colorScheme -ceq 'GruvboxMaterialHardDark' -and $result.profiles.list[0].font.size -eq 10) 'Development profile retains Gruvbox Material Hard Dark and font size'
    Assert ($result.schemes.Count -eq 6) 'All six captured Gruvbox Material palettes are restored'
    $hash = (Get-FileHash -LiteralPath $terminal).Hash
    Set-Terminal $terminal $defaults 'pwsh.exe -NoLogo' 'Test Font' (Join-Path $testRoot 'backups')
    Assert ((Get-FileHash -LiteralPath $terminal).Hash -eq $hash) 'Terminal rerun is byte-for-byte unchanged'

    $environment = Read-Json (Join-Path $root 'defaults/environment.json')
    $everyday = @('Microsoft.PowerToys', 'KDE.KDEConnect', 'Rufus.Rufus', 'AntibodySoftware.WizTree', 'VideoLAN.VLC')
    Assert (@($everyday | Where-Object { $environment.packages.id -notcontains $_ }).Count -eq 0) 'All requested everyday Windows tools are included in WinGet setup'
    $codexPackage = @($environment.packages | Where-Object id -eq 'OpenAI.Codex')
    Assert ($codexPackage.Count -eq 1 -and $codexPackage[0].installedCommand -eq 'codex.exe') 'Codex reuses an existing app-managed CLI'
    $kde = @($environment.packages | Where-Object id -eq 'KDE.KDEConnect')
    Assert ($kde.Count -eq 1 -and $kde[0].installedAppxName -eq 'KDEe.V.KDEConnect') 'KDE Connect recognizes the existing Microsoft Store edition'
    $powerToysDefaults = Read-Json (Join-Path $root 'defaults/powertoys.json')
    $powerToys = Join-Path $testRoot 'PowerToys'
    $powerToysGeneral = Join-Path $powerToys 'settings.json'
    $powerToysAwake = Join-Path $powerToys 'Awake/settings.json'
    Set-PowerToys $powerToys $powerToysDefaults (Join-Path $testRoot 'backups')
    Assert ((Read-Json $powerToysGeneral).enabled.Awake) 'Fresh PowerToys settings enable Awake'
    $awake = Read-Json $powerToysAwake
    Assert ($awake.properties.mode -eq 1 -and !$awake.properties.keepDisplayOn) 'Fresh Awake settings match indefinite mode with display allowed to sleep'
    Save-Json $powerToysGeneral ([ordered]@{ enabled=[ordered]@{Awake=$false; FancyZones=$false}; startup=$true; theme='dark' }) ''
    Save-Json $powerToysAwake ([ordered]@{name='Awake'; version='future-schema'; properties=[ordered]@{mode=0; keepDisplayOn=$true; customTrayTimes=[ordered]@{'8 hours'=28800}}}) ''
    Set-PowerToys $powerToys $powerToysDefaults (Join-Path $testRoot 'backups')
    $general = Read-Json $powerToysGeneral
    $awake = Read-Json $powerToysAwake
    Assert ($general.enabled.Awake -and !$general.enabled.FancyZones -and $general.startup -and $general.theme -ceq 'dark') 'PowerToys merge preserves other modules, startup preference and appearance'
    Assert ($awake.version -ceq 'future-schema' -and $awake.properties.customTrayTimes['8 hours'] -eq 28800) 'Awake merge preserves the installed schema and custom tray times'
    $generalHash = (Get-FileHash -LiteralPath $powerToysGeneral).Hash
    $awakeHash = (Get-FileHash -LiteralPath $powerToysAwake).Hash
    Set-PowerToys $powerToys $powerToysDefaults (Join-Path $testRoot 'backups')
    Assert ((Get-FileHash -LiteralPath $powerToysGeneral).Hash -ceq $generalHash -and (Get-FileHash -LiteralPath $powerToysAwake).Hash -ceq $awakeHash) 'PowerToys and Awake rerun is byte-for-byte unchanged'

    $existing = Join-Path $testRoot 'existing.json'
    Save-Text $existing @'
{
 // Comment, trailing comma, and URL must all survive parsing.
 "custom": "https://example.com/a//b",
 "profiles": {"list": [{"guid":"other", "name":"Personal"}],},
 "schemes": [{"name":"Personal", "background":"#010203"}],
 "keybindings": [{"keys":"ctrl+alt+x", "command":"newTab"}],
}
'@ ''
    Set-Terminal $existing $defaults 'pwsh.exe -NoLogo' 'Test Font' (Join-Path $testRoot 'backups')
    $result = Read-Json $existing
    Assert ($result.custom -eq 'https://example.com/a//b') 'JSONC parser preserves URL strings'
    Assert ($result.profiles.list.Count -eq 2 -and $result.profiles.list[0].name -eq 'Personal') 'Existing profiles preserved'
    Assert (@($result.schemes | Where-Object { $_.name -eq 'Personal' }).Count -eq 1) 'Existing color schemes preserved'
    Assert (@($result.keybindings | Where-Object { $_.keys -eq 'ctrl+alt+x' }).Count -eq 1) 'Unrelated shortcuts preserved'

    $profile = Join-Path $testRoot 'shell/profile.ps1'
    Save-Text $profile "function hello { 'hello' }`n" ''
    Set-ProfileLoader $profile "C:\Someone's computer\profile.ps1" (Join-Path $testRoot 'backups')
    $hash = (Get-FileHash -LiteralPath $profile).Hash
    Set-ProfileLoader $profile "C:\Someone's computer\profile.ps1" (Join-Path $testRoot 'backups')
    $text = [IO.File]::ReadAllText($profile)
    Assert ($text.Contains('function hello')) 'Existing PowerShell profile preserved'
    Assert (([regex]::Matches($text, '>>> personal-devenv >>>')).Count -eq 1) 'Profile loader inserted once'
    Assert ($text.Contains("Someone''s computer")) 'Profile loader escapes apostrophes in paths'
    Assert ((Get-FileHash -LiteralPath $profile).Hash -eq $hash) 'Profile rerun is unchanged'

    $registry = Join-Path $testRoot 'Obsidian/obsidian.json'
    Save-Json $registry ([ordered]@{ vaults = [ordered]@{ other = [ordered]@{ path = 'C:\other'; open = $true } }; custom = 'retained' }) ''
    Set-VaultRegistration $registry 'C:\notes' (Join-Path $testRoot 'backups')
    $hash = (Get-FileHash -LiteralPath $registry).Hash
    Set-VaultRegistration $registry 'C:\notes' (Join-Path $testRoot 'backups')
    $result = Read-Json $registry
    Assert ($result.vaults.Count -eq 2 -and $result.custom -eq 'retained') 'Vault registration preserves other vaults and settings'
    Assert ((Get-FileHash -LiteralPath $registry).Hash -eq $hash) 'Vault registration is idempotent'

    $rc = Join-Path $testRoot 'rc'
    Save-Text (Join-Path $rc 'codex/.codex.config.toml.template') "[projects.`"/home/zhuoryin`"]`ntrust_level = `"trusted`"`n" ''
    Save-Text (Join-Path $rc 'claude/.claude/CLAUDE.md') 'shared instructions' ''
    $fakeUser = Join-Path $testRoot 'WindowsUser'
    Set-NativeCodex $fakeUser $rc ''
    $codexPath = Join-Path $fakeUser '.codex/config.toml'
    Assert (![IO.File]::ReadAllText($codexPath).Contains('/home/zhuoryin')) 'New Codex config renders a native Windows trust path'
    Assert ((Get-Content -LiteralPath (Join-Path $fakeUser '.codex/AGENTS.md') -Raw) -eq 'shared instructions') 'Codex reuses shared agent instructions'
    Save-Text $codexPath 'existing preferences' ''
    Set-NativeCodex $fakeUser $rc ''
    Assert ([IO.File]::ReadAllText($codexPath) -eq 'existing preferences') 'Existing Codex config preserved on rerun'

    # A clean account has neither repository nor parent directories. Exercise
    # actual Git cloning from a disposable local source without network/auth.
    $source = Join-Path $testRoot 'source-repo'
    Invoke-Native git @('-c', 'init.defaultBranch=main', 'init', $source)
    Save-Text (Join-Path $source 'vim/.vimrc') '" shared configuration' ''
    Invoke-Native git @('-C', $source, 'add', '.')
    Invoke-Native git @('-C', $source, '-c', 'user.name=Devenv Test', '-c', 'user.email=devenv@example.invalid', '-c', 'commit.gpgsign=false', '-c', ('core.hooksPath=' + (Join-Path $testRoot 'no-hooks')), 'commit', '-m', 'fixture')
    $freshRc = Join-Path $testRoot 'clean Windows account/rc_files'
    Ensure-Checkout $freshRc $source
    Assert (Test-Path -LiteralPath (Join-Path $freshRc 'vim/.vimrc')) 'A clean Windows account gets rc_files through a real Git clone'
    Assert ((Invoke-Native git @('-C', $freshRc, 'remote', 'get-url', 'origin') -Capture) -eq $source) 'Cloned rc_files retains its Git remote'
    Assert ((Invoke-Native git @('-C', $freshRc, 'config', '--local', 'core.autocrlf') -Capture) -ceq 'false') 'New checkout preserves shared rc line endings'
    Save-Text (Join-Path $freshRc 'local-only.txt') 'keep on rerun' ''
    Ensure-Checkout $freshRc $source
    Assert (Test-Path -LiteralPath (Join-Path $freshRc 'local-only.txt')) 'Rerunning does not reset an existing rc_files checkout'
    $freshNotes = Join-Path $testRoot 'another clean account/notes'
    Ensure-Checkout $freshNotes $source
    Assert (Test-Path -LiteralPath (Join-Path $freshNotes '.git')) 'Notes are also Git cloned when their destination is absent'
    $occupied = Join-Path $testRoot 'occupied'
    [void][IO.Directory]::CreateDirectory($occupied)
    Save-Text (Join-Path $occupied 'keep.txt') 'existing user data' ''
    $threw = $false
    try { Ensure-Checkout $occupied $source } catch { $threw = $true }
    Assert $threw 'An occupied non-repository directory is never silently replaced'
    $empty = Join-Path $testRoot 'empty'
    [void][IO.Directory]::CreateDirectory($empty)
    Ensure-Checkout $empty $source
    Assert (Test-Path -LiteralPath (Join-Path $empty '.git')) 'An explicitly created empty directory can receive a Git checkout'

    $vimRc = Join-Path $testRoot 'shared-vim-rc'
    foreach ($relative in @('vim/.vimrc', 'vim/.vim/config/windows.vim', 'vim/.vim/config/plugins.vim', 'vim/.vim/coc-settings.json')) { Save-Text (Join-Path $vimRc $relative) 'fixture' '' }
    $vimUser = Join-Path $testRoot 'fresh-vim-home'
    $python = Join-Path $testRoot 'python-runtime'
    Save-Text (Join-Path $python 'python399.dll') 'fixture, not an executable DLL' ''
    Set-NativeVim $vimUser $vimRc 'C:/Program Files/Git/bin/bash.exe' '' $python
    $loader = [IO.File]::ReadAllText((Join-Path $vimUser '_vimrc'))
    Assert (($loader -split "`n" | Where-Object { $_ }).Count -eq 4 -and $loader.Contains('/vim/.vimrc')) 'Native Vim loader only supplies machine paths and sources the shared vimrc'
    Assert (!(Test-Path -LiteralPath (Join-Path $vimUser '.vim/config'))) 'Windows setup does not copy a second Vim configuration tree'
    Assert ($loader.Contains('python399.dll') -and !$loader.Contains('python314.dll')) 'Native Python DLL is discovered without a fixed version in Vim configuration'
    Assert ($environment.packages.id -contains 'LLVM.clangd' -and $environment.packages.id -contains 'GitHub.cli' -and $environment.packages.id -contains '7zip.7zip' -and $environment.packages.id -notcontains 'UniversalCtags.Ctags') 'Bootstrap installs clangd, GitHub CLI and 7-Zip without Ctags'

    $module = Get-Module Devenv
    & $module {
        function script:Invoke-WslCommand([string[]]$Arguments, [switch]$Elevate) {
            [void]$script:WslCalls.Add(@{ Arguments=@($Arguments); Elevate=[bool]$Elevate })
            if (!$script:WslResults.Count) { throw 'Unexpected WSL command' }
            return $script:WslResults.Dequeue()
        }
    }
    function Set-WslResults([object[]]$Results) {
        & $module {
            param($Items)
            $script:WslResults = New-Object Collections.Queue
            foreach ($item in $Items) { $script:WslResults.Enqueue($item) }
            $script:WslCalls = New-Object Collections.ArrayList
        } $Results
    }
    Set-WslResults @(@{ ExitCode=0; Output=" Ubuntu-24.04 `r`nDebian" })
    Assert ((Ensure-Wsl 'Ubuntu-24.04') -ceq 'Existing') 'Existing preferred WSL distribution is retained'
    Assert ((& $module { $script:WslCalls.Count }) -eq 1) 'Existing WSL uses only read-only inventory'
    Set-WslResults @(@{ ExitCode=0; Output='Debian' }, @{ ExitCode=0; Output='' }, @{ ExitCode=0; Output="Debian`nUbuntu-24.04" })
    Assert ((Ensure-Wsl 'Ubuntu-24.04') -ceq 'Installed') 'Missing Ubuntu is installed and registration is checked'
    $calls = & $module { ,$script:WslCalls.ToArray() }
    Assert ($calls[1].Elevate -and ($calls[1].Arguments -join ' ') -ceq '--install --distribution Ubuntu-24.04 --no-launch') 'Only WSL installation requests elevation and it never launches Linux'
    Set-WslResults @(@{ ExitCode=1; Output='WSL feature unavailable' }, @{ ExitCode=3010; Output='' })
    Assert ((Ensure-Wsl 'Ubuntu-24.04') -ceq 'RebootRequired') 'WSL restart requirement is returned without rebooting'
    Set-WslResults @(@{ ExitCode=0; Output='' }, @{ ExitCode=0; Output='' }, @{ ExitCode=0; Output='' })
    Assert ((Ensure-Wsl 'Ubuntu-24.04') -ceq 'PendingInitialization') 'Deferred distro registration is not reported as ready'
    Set-WslResults @(@{ ExitCode=0; Output='' }, @{ ExitCode=5; Output='Access denied' })
    $threw = $false
    try { Ensure-Wsl 'Ubuntu-24.04' } catch { $threw = $_.Exception.Message -like '*Access denied*' }
    Assert $threw 'WSL installation failures are reported with their error'
    $threw = $false
    try { Ensure-Wsl 'Ubuntu;whoami' } catch { $threw = $true }
    Assert $threw 'Distribution identifiers cannot inject installer arguments'

    & $module {
        $script:FixtureJson = @{}
        $script:FixtureFiles = @{}
        $script:DownloadCount = 0
        $script:FailUrl = ''
        function script:Get-WebJson([string]$Url) {
            if (!$script:FixtureJson.ContainsKey($Url)) { throw "Unexpected metadata URL: $Url" }
            return $script:FixtureJson[$Url]
        }
        function script:Get-Download([string]$Url, [string]$Destination) {
            if ($Url -eq $script:FailUrl) { throw 'Simulated failed download' }
            $script:DownloadCount++
            $value = $script:FixtureFiles[$Url]
            if ($value -is [byte[]]) { [IO.File]::WriteAllBytes($Destination, $value) }
            else { [IO.File]::WriteAllText($Destination, $value, (New-Object Text.UTF8Encoding($false))) }
        }
    }
    function Set-FixtureJson([string]$Url, $Value) { & $module { param($u, $v) $script:FixtureJson[$u] = $v } $Url $Value }
    function Set-FixtureFile([string]$Url, $Value) { & $module { param($u, $v) $script:FixtureFiles[$u] = $v } $Url $Value }

    # Exercise release discovery and archive handling without OS registration.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $fontSource = Join-Path $testRoot 'font-source'
    [void][IO.Directory]::CreateDirectory($fontSource)
    Save-Text (Join-Path $fontSource 'fixture.ttf') 'test font bytes' ''
    $fontZip = Join-Path $testRoot 'fixture.zip'
    [IO.Compression.ZipFile]::CreateFromDirectory($fontSource, $fontZip)
    $fixtureFont = [ordered]@{ repository='fixture/fonts'; asset='fonts.zip' }
    Set-FixtureJson 'https://api.github.com/repos/fixture/fonts/releases/latest' @{ assets=@(@{ name='fonts.zip'; browser_download_url='https://fixture.invalid/font.zip' }) }
    Set-FixtureFile 'https://fixture.invalid/font.zip' ([IO.File]::ReadAllBytes($fontZip))
    $fontOutput = Join-Path $testRoot 'font-output'
    Get-FontFiles $fixtureFont $fontOutput
    Assert ([IO.File]::ReadAllText((Join-Path $fontOutput 'fixture.ttf')) -ceq 'test font bytes') 'Font archive is selected from the latest upstream release'
    $fixtureFont.asset = 'absent.zip'
    $threw = $false
    try { Get-FontFiles $fixtureFont (Join-Path $testRoot 'bad-font') } catch { $threw = $true }
    Assert $threw 'Missing font family archive is reported instead of substituting another font'
    $fixtureFont.asset = 'fonts.zip'
    $unsafeZip = Join-Path $testRoot 'unsafe.zip'
    $zip = [IO.Compression.ZipFile]::Open($unsafeZip, [IO.Compression.ZipArchiveMode]::Create)
    try { [void]$zip.CreateEntry('../escape.ttf') } finally { $zip.Dispose() }
    Set-FixtureFile 'https://fixture.invalid/font.zip' ([IO.File]::ReadAllBytes($unsafeZip))
    $threw = $false
    try { Get-FontFiles $fixtureFont (Join-Path $testRoot 'unsafe-font') } catch { $threw = $true }
    Assert $threw 'Font archive entries cannot escape the staging directory'
    $vault = Join-Path $testRoot 'vault'
    Save-Json (Join-Path $vault '.obsidian/community-plugins.json') @('fixture') ''
    Save-Json (Join-Path $vault '.obsidian/appearance.json') @{ cssTheme='Fixture Theme' } ''
    Set-FixtureJson 'https://raw.githubusercontent.com/obsidianmd/obsidian-releases/master/community-plugins.json' @(@{ id='fixture'; repo='fixture/plugin' })
    Set-FixtureJson 'https://raw.githubusercontent.com/obsidianmd/obsidian-releases/master/community-css-themes.json' @(@{ name='Fixture Theme'; repo='fixture/theme' })
    $releaseAssets = @()
    foreach ($name in @('manifest.json', 'main.js', 'styles.css')) {
        $releaseAssets += @{ name=$name; browser_download_url="https://fixture.invalid/$name" }
    }
    Set-FixtureJson 'https://raw.githubusercontent.com/fixture/plugin/HEAD/manifest.json' @{ id='fixture'; version='1.0.0' }
    Set-FixtureJson 'https://api.github.com/repos/fixture/plugin/releases/tags/1.0.0' @{ assets=$releaseAssets }
    Set-FixtureJson 'https://fixture.invalid/manifest.json' @{ id='fixture'; version='1.0.0' }
    Set-FixtureFile 'https://fixture.invalid/manifest.json' '{"id":"fixture","version":"1.0.0"}'
    Set-FixtureFile 'https://fixture.invalid/main.js' 'latest plugin code'
    Set-FixtureFile 'https://fixture.invalid/styles.css' '.old-style {}'
    Set-FixtureJson 'https://api.github.com/repos/fixture/theme/commits/HEAD' @{ sha='current-runtime-revision' }
    $themeBase = 'https://raw.githubusercontent.com/fixture/theme/current-runtime-revision'
    Set-FixtureJson "$themeBase/manifest.json" @{ name='Fixture Theme'; version='1.0.0' }
    Set-FixtureFile "$themeBase/manifest.json" '{"name":"Fixture Theme","version":"1.0.0"}'
    Set-FixtureFile "$themeBase/theme.css" '.theme-v1 {}'
    $resolved = @(Resolve-ObsidianAddons $vault)
    Assert ($resolved.Count -eq 2 -and $resolved[0].id -ceq 'fixture' -and $resolved[1].id -ceq 'Fixture Theme') 'Plugin and theme choices come only from Git-tracked vault configuration'
    $addon = $resolved[0]
    $plugin = Join-Path $testRoot 'plugin'
    Save-Json (Join-Path $plugin 'data.json') ([ordered]@{ preference = 'keep' }) ''
    Install-LatestAddon $addon $plugin (Join-Path $testRoot 'backups')
    Assert ((Read-Json (Join-Path $plugin 'data.json')).preference -eq 'keep') 'Plugin repair preserves existing data.json'
    $count = & $module { $script:DownloadCount }
    Install-LatestAddon $addon $plugin (Join-Path $testRoot 'backups')
    Assert ((& $module { $script:DownloadCount }) -eq $count) 'An already-current plugin does not download its code again'

    Set-FixtureJson 'https://fixture.invalid/manifest.json' @{ id='fixture'; version='2.0.0' }
    Set-FixtureJson 'https://raw.githubusercontent.com/fixture/plugin/HEAD/manifest.json' @{ id='fixture'; version='2.0.0' }
    Set-FixtureFile 'https://fixture.invalid/manifest.json' '{"id":"fixture","version":"2.0.0"}'
    Set-FixtureFile 'https://fixture.invalid/main.js' 'newer plugin code'
    Set-FixtureJson 'https://api.github.com/repos/fixture/plugin/releases/tags/2.0.0' @{ assets=@($releaseAssets | Where-Object { $_.name -ne 'styles.css' }) }
    $addon = @(Resolve-ObsidianAddons $vault)[0]
    Install-LatestAddon $addon $plugin (Join-Path $testRoot 'backups')
    Assert ((Read-Json (Join-Path $plugin 'manifest.json')).version -ceq '2.0.0') 'A newer upstream release is discovered and replaces existing plugin code'
    Assert ((Read-Json (Join-Path $plugin 'data.json')).preference -ceq 'keep') 'Updating plugin code keeps Git-tracked plugin settings'
    Assert (!(Test-Path -LiteralPath (Join-Path $plugin 'styles.css'))) 'Removed upstream styles do not linger after an update'
    $themePath = Join-Path $testRoot 'theme'
    Install-LatestAddon $resolved[1] $themePath (Join-Path $testRoot 'backups')
    Set-FixtureFile "$themeBase/theme.css" '.theme-v2 {}'
    Install-LatestAddon $resolved[1] $themePath (Join-Path $testRoot 'backups')
    Assert ([IO.File]::ReadAllText((Join-Path $themePath 'theme.css')) -ceq '.theme-v2 {}') 'Theme updates are fetched even when its manifest version is unchanged'

    # Normal reruns must work without querying registries or downloading code.
    $installedPlugin = Join-Path $vault '.obsidian/plugins/fixture'
    $installedTheme = Join-Path $vault '.obsidian/themes/Fixture Theme'
    Copy-Item -LiteralPath $plugin -Destination $installedPlugin -Recurse
    Copy-Item -LiteralPath $themePath -Destination $installedTheme -Recurse
    $count = & $module { $script:DownloadCount }
    & $module { $script:SavedJsonFixtures = $script:FixtureJson; $script:FixtureJson = @{} }
    Install-ObsidianAddons $vault ''
    Assert ((& $module { $script:DownloadCount }) -eq $count) 'Installed Obsidian plugins and theme skip all network work on normal reruns'
    & $module { $script:FixtureJson = $script:SavedJsonFixtures }
    Set-FixtureFile "$themeBase/theme.css" '.theme-v3 {}'
    Install-ObsidianAddons $vault '' -Update
    Assert ([IO.File]::ReadAllText((Join-Path $installedTheme 'theme.css')) -ceq '.theme-v3 {}') 'Explicit -Update refreshes installed themes without a version pin'
    Remove-Item -LiteralPath (Join-Path $installedPlugin 'main.js')
    Install-ObsidianAddons $vault ''
    Assert (Test-Path -LiteralPath (Join-Path $installedPlugin 'main.js')) 'Normal reruns repair an incomplete plugin installation'

    & $module { $script:FailUrl = 'https://fixture.invalid/main.js' }
    $addon.version = '3.0.0'
    Set-FixtureFile 'https://fixture.invalid/manifest.json' '{"id":"fixture","version":"3.0.0"}'
    $threw = $false
    try { Install-LatestAddon $addon $plugin '' } catch { $threw = $true }
    Assert ($threw -and (Read-Json (Join-Path $plugin 'manifest.json')).version -ceq '2.0.0') 'Failed downloads leave the previously working plugin untouched'
    & $module { $script:FailUrl = '' }
    $threw = $false
    $addon.id = 'unexpected'
    try { Install-LatestAddon $addon (Join-Path $testRoot 'wrong-identity') '' } catch { $threw = $true }
    Assert $threw 'Wrong plugin manifest rejected before activation'
    Assert (!(Test-Path -LiteralPath (Join-Path $testRoot 'wrong-identity/main.js'))) 'Failed validation leaves no activated code'
    Assert (@(Get-ChildItem -LiteralPath (Join-Path $testRoot 'backups') -File).Count -gt 0) 'Replaced configuration files have backups'
    Remove-Module Devenv
    Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
    $threw = $false
    try { Get-Download 'http://fixture.invalid/file' (Join-Path $testRoot 'http') } catch { $threw = $true }
    Assert $threw 'Downloads require HTTPS'
    $preferences = Read-Json (Join-Path $root 'defaults/environment.json')
    Assert (!$preferences.Contains('addons') -and !$preferences.Contains('codexAcpVersion') -and !$preferences.Contains('vimPlug') -and !$preferences.font.Contains('archive')) 'Preferences contain no duplicate plugin list or artifact version/hash locks'
    if ($Network) {
        $manifest = Read-Json (Join-Path $root 'defaults/environment.json')
        $liveVault = Join-Path $testRoot 'live-vault'
        Save-Json (Join-Path $liveVault '.obsidian/community-plugins.json') @('calendar', 'obsidian-custom-attachment-location', 'copilot') ''
        Save-Json (Join-Path $liveVault '.obsidian/appearance.json') @{ cssTheme='Material Gruvbox' } ''
        $liveAddons = @(Resolve-ObsidianAddons $liveVault)
        foreach ($live in $liveAddons) {
            $destination = Join-Path $testRoot ('real-addons/' + $live.id)
            Install-LatestAddon $live $destination ''
            Assert ((Read-Json (Join-Path $destination 'manifest.json')).version -ceq $live.version) "Current upstream $($live.id) downloads and installs in isolation"
        }
        $realFonts = Join-Path $testRoot 'real-fonts'
        Get-FontFiles $manifest.font $realFonts
        Assert (@(Get-ChildItem -LiteralPath $realFonts -Filter '*.ttf').Count -eq 4) 'Latest upstream font archive downloads all four requested styles'
        Add-Type -AssemblyName System.Drawing
        $collection = New-Object Drawing.Text.PrivateFontCollection
        try {
            $collection.AddFontFile((Join-Path $realFonts 'MonacoLigaturizedNerdFont-Regular.ttf'))
            Assert ($collection.Families[0].Name -ceq 'MonacoLigaturized Nerd Font') 'Downloaded font has the requested family, without the Mono variant'
        } finally { $collection.Dispose() }
    }
    Write-Output "$script:passed assertions passed. Fixtures: $testRoot"
} finally {
    # Only this run's generated fixture directory may be removed.
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notlike 'personal-devenv-test-*') { throw 'Unsafe test cleanup path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
