param([switch]$Network, [string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
. (Join-Path $PSScriptRoot 'RcFixture.ps1')
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
    $sourceRc = Join-Path $testRoot 'rc_files'
    $terminalRelative = 'terminal/.windows/@LOCALAPPDATA/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/LocalState/settings.json'
    $terminalSource = Join-Path $sourceRc $terminalRelative
    [void][IO.Directory]::CreateDirectory((Split-Path $terminalSource -Parent))
    Copy-Item -LiteralPath (Join-Path $RcRoot $terminalRelative) -Destination $terminalSource
    $arrayPath = Join-Path $testRoot 'plugins.json'
    Save-Text $arrayPath '["calendar", "copilot"]' ''
    $ids = @(Read-Json $arrayPath)
    Assert ($ids.Count -eq 2 -and $ids[0] -ceq 'calendar' -and $ids[1] -ceq 'copilot') 'Root JSON string arrays work on Windows PowerShell 5.1'
    Save-Text $arrayPath '["calendar"]' ''
    Assert (@(Read-Json $arrayPath).Count -eq 1) 'Single-plugin JSON arrays remain iterable'
    $terminal = Join-Path $testRoot 'Terminal/settings.json'
    Invoke-TestLink $terminal $terminalSource ''
    Assert (Test-InstalledLink $terminal $terminalSource) 'Terminal directly links to tracked preferences'
    $hash=(Get-FileHash -LiteralPath $terminal).Hash
    $time=(Get-Item -LiteralPath $terminal).CreationTimeUtc.Ticks
    Invoke-TestLink $terminal $terminalSource ''
    Assert ((Get-FileHash -LiteralPath $terminal).Hash -ceq $hash -and (Get-Item -LiteralPath $terminal).CreationTimeUtc.Ticks -eq $time) 'Terminal rerun leaves source and link unchanged'

    $formatted = Join-Path $testRoot 'formatted.json'
    $formatBackups = Join-Path $testRoot 'format-backups'
    Save-Text $formatted '{ "nested": { "b": 2, "a": 1 }, "items": ["x", "y"], "text": "A" }' ''
    $before = Get-Item $formatted
    $beforeHash = (Get-FileHash $formatted).Hash
    $beforeTime = $before.LastWriteTimeUtc.Ticks
    $desired = [ordered]@{ text='A'; items=@('x','y'); nested=[ordered]@{a=1;b=2} }
    Save-Json $formatted $desired $formatBackups
    Assert ((Get-FileHash $formatted).Hash -ceq $beforeHash -and (Get-Item $formatted).LastWriteTimeUtc.Ticks -eq $beforeTime) 'JSON whitespace and property order do not cause a rewrite'
    Assert (!(Test-Path $formatBackups)) 'Equivalent app-formatted JSON creates no redundant backup'
    $desired.text = 'a'
    Save-Json $formatted $desired $formatBackups
    Assert ((Read-Json $formatted).text -ceq 'a' -and @(Get-ChildItem $formatBackups -File).Count -eq 1) 'Actual string changes are saved and backed up'
    Assert (!(Test-JsonEquivalent @('x','y') @('y','x'))) 'JSON comparison preserves array order'
    Assert (!(Test-JsonEquivalent @('x') 'x')) 'JSON comparison distinguishes single-item arrays from scalars'
    Assert ((Test-JsonEquivalent $null $null) -and !(Test-JsonEquivalent $null '')) 'JSON comparison preserves null values'
    $environment = Read-Json (Join-Path $root 'defaults/environment.json')
    $everyday = @('Microsoft.PowerToys', 'KDE.KDEConnect', 'Rufus.Rufus', 'AntibodySoftware.WizTree', 'VideoLAN.VLC')
    Assert (@($everyday | Where-Object { $environment.packages.id -notcontains $_ }).Count -eq 0) 'All requested everyday Windows tools are included in WinGet setup'
    $kde = @($environment.packages | Where-Object id -eq 'KDE.KDEConnect')
    Assert ($kde.Count -eq 1 -and $kde[0].installedAppxName -eq 'KDEe.V.KDEConnect') 'KDE Connect recognizes the existing Microsoft Store edition'
    $profile = Join-Path $testRoot 'shell/profile.ps1'
    $sharedProfile = Join-Path $testRoot "Someone's computer/profile.ps1"
    Save-Text $sharedProfile '# tracked profile' ''
    Save-Text $profile '# unrelated local edit' ''
    $rejected=$false
    try { Invoke-TestLink $profile $sharedProfile '' } catch { $rejected=$_.Exception.Message.Contains('Configuration conflict') }
    Assert ($rejected -and [IO.File]::ReadAllText($profile) -ceq '# unrelated local edit') 'Conflicting PowerShell content is preserved and reported'
    Save-Text $profile '# tracked profile' ''
    Invoke-TestLink $profile $sharedProfile ''
    Assert (Test-InstalledLink $profile $sharedProfile) 'An identical profile copy becomes a direct link'
    [IO.File]::WriteAllText($profile,'# local edit through linked profile')
    Assert ([IO.File]::ReadAllText($sharedProfile) -ceq '# local edit through linked profile') 'Installed profile edits reach rc_files'
    $jsonc=Join-Path $testRoot 'jsonc.json'
    Save-Text $jsonc "{ // comment`n `"url`": `"https://example.com/a//b`", }" ''
    Assert ((Read-Json $jsonc).url -ceq 'https://example.com/a//b') 'JSONC preserves URL strings and permits comments and trailing commas'

    # A clean account has neither repository nor parent directories. Exercise
    # actual Git cloning from a disposable local source without network/auth.
    $source = Join-Path $testRoot 'source-repo'
    Invoke-Native git @('-c', 'init.defaultBranch=main', 'init', $source)
    Save-Text (Join-Path $source 'vim/.vimrc') '" shared configuration' ''
    Invoke-Native git @('-C', $source, 'add', '-f', '.')
    Invoke-Native git @('-C', $source, '-c', 'user.name=Devenv Test', '-c', 'user.email=devenv@example.invalid', '-c', 'commit.gpgsign=false', '-c', ('core.hooksPath=' + (Join-Path $testRoot 'no-hooks')), 'commit', '-m', 'fixture')
    $freshRc = Join-Path $testRoot 'clean Windows account/rc_files'
    Ensure-Checkout $freshRc $source
    Assert (Test-Path -LiteralPath (Join-Path $freshRc 'vim/.vimrc')) 'A clean Windows account gets rc_files through a real Git clone'
    Assert ((Invoke-Native git @('-C', $freshRc, 'remote', 'get-url', 'origin') -Capture) -eq $source) 'Cloned rc_files retains its Git remote'
    Assert ((Invoke-Native git @('-C', $freshRc, 'config', '--local', 'core.autocrlf') -Capture) -ceq 'false') 'New checkout preserves shared rc line endings'
    Save-Text (Join-Path $freshRc 'local-only.txt') 'keep on rerun' ''
    Ensure-Checkout $freshRc $source
    Assert (Test-Path -LiteralPath (Join-Path $freshRc 'local-only.txt')) 'Rerunning does not reset an existing rc_files checkout'
    $commit = @('-c', 'user.name=Devenv Test', '-c', 'user.email=devenv@example.invalid', '-c', 'commit.gpgsign=false', '-c', ('core.hooksPath=' + (Join-Path $testRoot 'no-hooks')), 'commit', '-q', '-am')
    $spacer = (0..4 | ForEach-Object { "`" unchanged $_" }) -join "`n"
    Save-Text (Join-Path $source 'vim/.vimrc') "`" shared configuration`n$spacer`n" ''
    Invoke-Native git (@('-C', $source) + $commit + 'spacer')
    Ensure-Checkout $freshRc $source
    Save-Text (Join-Path $source 'vim/.vimrc') "`" upstream configuration`n$spacer`n" ''
    Invoke-Native git (@('-C', $source) + $commit + 'upstream')
    Save-Text (Join-Path $freshRc 'vim/.vimrc') "`" shared configuration`n$spacer`n`" local addition`n" ''
    Ensure-Checkout $freshRc $source
    $vimrc = [IO.File]::ReadAllText((Join-Path $freshRc 'vim/.vimrc'))
    Assert ($vimrc.Contains('upstream configuration') -and $vimrc.Contains('local addition') -and (Test-Path -LiteralPath (Join-Path $freshRc 'local-only.txt'))) 'Rerunning pulls upstream and restores independent local edits'
    Assert (!(Invoke-GitQuiet $freshRc @('stash', 'list')).Output) 'Restored local edits leave no stash behind'
    Save-Text (Join-Path $source 'vim/.vimrc') "`" newer configuration`n$spacer`n" ''
    Invoke-Native git (@('-C', $source) + $commit + 'newer')
    Save-Text (Join-Path $freshRc 'vim/.vimrc') "`" mine`n$spacer`n" ''
    Ensure-Checkout $freshRc $source
    Assert ([IO.File]::ReadAllText((Join-Path $freshRc 'vim/.vimrc')).Contains('newer configuration') -and (Invoke-GitQuiet $freshRc @('stash', 'list')).Output -match 'bootstrap sync') 'Conflicting local edits yield to upstream and stay stashed'
    Invoke-Native git @('-C', $source, 'branch', '-q', '-m', 'main', 'trunk')
    Ensure-Checkout $freshRc $source
    Assert ((Invoke-GitQuiet $freshRc @('rev-parse', '--abbrev-ref', 'HEAD')).Output -ceq 'trunk') 'A renamed default branch is followed'
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

    $vimUser = Join-Path $testRoot 'fresh-vim-home'
    Invoke-RcSetup $RcRoot @('apply', '--package', 'vim', '--home', $vimUser)
    Assert (Test-InstalledLink (Join-Path $vimUser '.vimrc') (Join-Path $RcRoot 'vim/.vimrc')) 'Native Vim entry links directly to the shared vimrc'
    Assert (!(Test-Path -LiteralPath (Join-Path $vimUser '_vimrc'))) 'Fresh Windows setup does not create a second vimrc entry'
    Assert (!(Test-Path -LiteralPath (Join-Path $vimUser '.vim/machine-paths.json'))) 'Vim configuration requires no generated executable-path file'
    Assert (Test-InstalledLink (Join-Path $vimUser '.vim/config/plugins.vim') (Join-Path $RcRoot 'vim/.vim/config/plugins.vim')) 'Vim plugin declarations link to their shared source'
    Assert ($environment.packages.id -notcontains 'LLVM.clangd' -and $environment.packages.id -notcontains 'GitHub.cli' -and $environment.packages.id -contains '7zip.7zip' -and $environment.packages.id -notcontains 'UniversalCtags.Ctags') 'Bootstrap keeps utilities and delegates configured development tools to rc_files'

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
    foreach ($installed in @('My imported Linux', "Ubuntu-22.04`nUbuntu-24.04", 'Debian')) {
        Set-WslResults @(@{ ExitCode=0; Output=$installed })
        Assert ((Ensure-Wsl 'Ubuntu') -ceq 'Existing') 'Existing WSL is reused regardless of distribution names or count'
        Assert ((& $module { $script:WslCalls.Count }) -eq 1) 'Existing WSL uses only read-only inventory'
    }
    Set-WslResults @(@{ ExitCode=0; Output='' }, @{ ExitCode=0; Output='' }, @{ ExitCode=0; Output='Imported Ubuntu' })
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

    & (Join-Path $PSScriptRoot 'Test-Obsidian.ps1') -Network:$Network
    $preferences = Read-Json (Join-Path $root 'defaults/environment.json')
    Assert (!$preferences.Contains('addons') -and !$preferences.Contains('codexAcpVersion') -and !$preferences.Contains('vimPlug') -and !$preferences.Contains('font')) 'Preferences contain no duplicate plugin list or artifact version/hash locks'
    $inventory = "import json,sys; from pathlib import Path; repo=Path(sys.argv[1]); sys.path.insert(0,str(repo/'.setup')); import links,packages; print(json.dumps([p.winget for p in packages.discover(repo,links.tracked_files(repo),[],'windows').requirements if p.winget]))"
    $applicationIds = ConvertFrom-Json (Invoke-Native py.exe @('-3', '-B', '-c', $inventory, $RcRoot) -Capture)
    Assert (@($preferences.packages | Where-Object { $applicationIds -contains $_.id }).Count -eq 0) 'Machine utilities do not duplicate the application packages owned by rc_files'
    Write-Output "$script:passed assertions passed. Fixtures: $testRoot"
} finally {
    # Only this run's generated fixture directory may be removed.
    $resolved = [IO.Path]::GetFullPath($testRoot)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notlike 'personal-devenv-test-*') { throw 'Unsafe test cleanup path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
