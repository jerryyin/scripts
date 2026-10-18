<#
.SYNOPSIS
Run the real bootstrap twice in an empty Windows home using real portable tools.
.DESCRIPTION
Tests application/configuration deployment, real Git clones, current plugins,
native Vim/PowerShell/psmux/clangd and reruns. Does not install host packages,
change WSL/fonts/services or use real credentials. Those require a clean VM.
#>
param(
    [Parameter(Mandatory=$true)][string]$ToolsRoot,
    [Parameter(Mandatory=$true)][string]$RcCandidate,
    [Parameter(Mandatory=$true)][string]$NotesConfiguration,
    [string]$FixturePath
)
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
if (!(Test-DeveloperMode)) { throw 'Enable Developer Mode before running the isolated deployment test. This test never enables it.' }
# Keep the synthetic user profile comparable in length to a real Windows home.
# Deep temp roots can exceed CreateProcess's executable-path limit inside npm's
# nested Codex dependency even when package extraction itself succeeds.
$fixture = if ($FixturePath) { [IO.Path]::GetFullPath($FixturePath) } else { Join-Path ([IO.Path]::GetTempPath()) ('dev-' + [guid]::NewGuid().ToString('N').Substring(0,8)) }
if ((Test-Path -LiteralPath $fixture) -and @(Get-ChildItem -LiteralPath $fixture -Force).Count) { throw 'Deployment fixture must be empty.' }
[void][IO.Directory]::CreateDirectory($fixture)
$socket = 'deployment-' + [guid]::NewGuid().ToString('N')
$psmux = Join-Path $ToolsRoot 'psmux/psmux.exe'
$vim = @(Get-ChildItem -LiteralPath (Join-Path $ToolsRoot 'vim') -Recurse -Filter vim.exe)[0].FullName
$node = @(Get-ChildItem -LiteralPath (Join-Path $ToolsRoot 'node') -Recurse -Filter node.exe)[0].FullName
$clangd = @(Get-ChildItem -LiteralPath (Join-Path $ToolsRoot 'clangd') -Recurse -Filter clangd.exe)[0].FullName
$gh = @(Get-ChildItem -LiteralPath (Join-Path $ToolsRoot 'gh') -Recurse -Filter gh.exe)[0].FullName
$pwsh = Join-Path $ToolsRoot 'pwsh/pwsh.exe'
foreach ($exe in @($psmux, $vim, $node, $clangd, $gh, $pwsh)) { if (!(Test-Path -LiteralPath $exe)) { throw "Missing real test tool: $exe" } }
. (Join-Path $PSScriptRoot 'RcFixture.ps1')
$script:passed = 0
function Assert($Ok, [string]$Message) {
    if (!$Ok) { throw "FAIL: $Message. Inspect $fixture" }
    $script:passed++; Write-Output "PASS: $Message"
}
try {
    $environment = Read-Json (Join-Path $root 'defaults/environment.json')
    $sourceRc = Join-Path $fixture 'sources/rc_files'
    Ensure-Checkout $sourceRc $environment.rcRepository
    Copy-RcFixture $RcCandidate $sourceRc
    Invoke-Native git @('-C', $sourceRc, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'candidate Windows configuration')
    $sourceNotes = Join-Path $fixture 'sources/notes'
    [void][IO.Directory]::CreateDirectory((Join-Path $sourceNotes '.obsidian'))
    foreach ($file in @('community-plugins.json', 'appearance.json')) { Copy-Item -LiteralPath (Join-Path $NotesConfiguration ".obsidian/$file") -Destination (Join-Path $sourceNotes ".obsidian/$file") }
    foreach ($id in @(Read-Json (Join-Path $sourceNotes '.obsidian/community-plugins.json'))) { Save-Json (Join-Path $sourceNotes ".obsidian/plugins/$id/data.json") ([ordered]@{ fixture='preserve plugin preferences' }) '' }
    Save-Text (Join-Path $sourceNotes 'Welcome.md') '# Disposable deployment test' ''
    Invoke-Native git @('init', $sourceNotes)
    Invoke-Native git @('-C', $sourceNotes, 'add', '-f', '.')
    Invoke-Native git @('-C', $sourceNotes, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-m', 'real plugin choices and fake settings')
    $environment.rcRepository = $sourceRc
    $environment.notesRepository = $sourceNotes
    $environmentPath = Join-Path $fixture 'environment.json'
    Save-Json $environmentPath $environment ''
    $userRoot = Join-Path $fixture 'empty Windows home'
    Assert (!(Test-Path -LiteralPath $userRoot)) 'Deployment starts with no Windows home, rc_files, notes, profiles or plugin directories'
    $toolPaths = @($node, $vim, $psmux, $pwsh, $gh, $clangd) | ForEach-Object { Split-Path $_ -Parent }
    $jobPath = Join-Path $fixture 'job.json'
    Save-Json $jobPath ([ordered]@{ root=$root; fixture=$fixture; userRoot=$userRoot; environment=$environmentPath; toolPaths=@($toolPaths); python=(Join-Path $ToolsRoot 'python'); vim=$vim; psmux=$psmux; pwsh=$pwsh; clangd=$clangd; nodeDirectory=(Split-Path $node -Parent) }) ''
    $worker = Join-Path $fixture 'worker.ps1'
    Save-Text $worker @'
param([switch]$RuntimeOnly)
$ErrorActionPreference = 'Stop'
$job = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'job.json') -Raw | ConvertFrom-Json
$env:USERPROFILE = $job.userRoot
$env:HOME = $job.userRoot
$env:APPDATA = Join-Path $job.userRoot 'AppData/Roaming'
$env:LOCALAPPDATA = Join-Path $job.userRoot 'AppData/Local'
$env:XDG_CONFIG_HOME = Join-Path $job.userRoot '.config'
$env:CODEX_HOME = Join-Path $job.userRoot '.codex'
$env:GH_CONFIG_DIR = Join-Path $env:APPDATA 'GitHub CLI'
$env:DOCKER_CONFIG = Join-Path $job.userRoot '.docker'
$env:GIT_CONFIG_GLOBAL = Join-Path $job.userRoot '.gitconfig'
$env:GIT_CONFIG_NOSYSTEM = '1'
$env:NPM_CONFIG_PREFIX = Join-Path $env:APPDATA 'npm'
$env:NPM_CONFIG_CACHE = Join-Path $job.userRoot 'cache/npm'
$env:PATH = ($job.toolPaths -join ';') + ';' + $env:PATH
Import-Module (Join-Path $job.root 'Devenv.psm1') -Force -DisableNameChecking
function Snapshot {
    $paths = @('_vimrc', '.psmux.conf', '.psmux/generated.conf', '.gitconfig', '.vim/machine-paths.json', '.vim/autoload/plug.vim', '.vim/coc-extensions.json', '.config/coc/extensions/package.json', '.codex/config.toml', '.codex/AGENTS.md', 'Documents/PowerShell/profile.ps1', 'Documents/WindowsPowerShell/profile.ps1', 'Terminal/settings.json', 'AppData/Local/Microsoft/PowerToys/settings.json', 'AppData/Local/Microsoft/PowerToys/Awake/settings.json', 'AppData/Roaming/obsidian/obsidian.json', 'AppData/Roaming/npm/node_modules/@agentclientprotocol/codex-acp/package.json')
    $files = @($paths | ForEach-Object { Join-Path $job.userRoot $_ })
    $files += @(Get-ChildItem -LiteralPath (Join-Path $env:LOCALAPPDATA 'PersonalDevenv') -File | Select-Object -ExpandProperty FullName)
    $files += @(Get-ChildItem -LiteralPath (Join-Path $job.userRoot 'notes/.obsidian') -Recurse -File | Select-Object -ExpandProperty FullName)
    $files += @(Get-ChildItem -LiteralPath (Join-Path $job.userRoot '.vim/plugged') -Directory | ForEach-Object { Join-Path $_.FullName '.git/HEAD' })
    $files += @(Get-ChildItem -LiteralPath (Join-Path $job.userRoot '.config/coc/extensions/node_modules') -Directory | Where-Object { $_.Name -like 'coc-*' } | ForEach-Object { Join-Path $_.FullName 'package.json' })
    $snapshot = [ordered]@{}
    foreach ($path in $files | Sort-Object -Unique) {
        if (!(Test-Path -LiteralPath $path)) { throw "Missing deployed file: $path" }
        $file = Get-Item -LiteralPath $path
        $snapshot[$path] = @{hash=(Get-FileHash -LiteralPath $path).Hash; writeTime=$file.LastWriteTimeUtc.Ticks}
    }
    return $snapshot
}
try {
    if (!$RuntimeOnly) {
    $common = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $job.root 'bootstrap.ps1'), '-SkipPackages', '-SkipWsl', '-SkipFonts', '-SkipCredentials', '-RequireDeveloperMode', '-EnvironmentPath', $job.environment, '-DocumentsPath', (Join-Path $job.userRoot 'Documents'), '-PythonRoot', $job.python, '-TerminalPath', (Join-Path $job.userRoot 'Terminal/settings.json'))
    foreach ($pass in @('first', 'second')) {
        Start-Transcript -LiteralPath (Join-Path $job.fixture ($pass + '.log')) | Out-Null
        & powershell.exe @common -Apply | Out-Host
        $code = $LASTEXITCODE
        Stop-Transcript | Out-Null
        if ($code -ne 0) { throw "Bootstrap $pass run failed: exit $code" }
        Save-Json (Join-Path $job.fixture ($pass + '-snapshot.json')) (Snapshot) ''
    }
    & powershell.exe @common -Check
    if ($LASTEXITCODE -ne 0) { throw 'Bootstrap -Check failed after deployment' }
    } else {
    & (Join-Path $job.root 'tests/Test-NativeTools.ps1') -VimExe $job.vim -PsmuxExe $job.psmux -RcRoot (Join-Path $job.userRoot 'rc_files') -PythonRoot $job.python -NodeDirectory $job.nodeDirectory -ClangdExe $job.clangd -FixturePath $job.userRoot -UseExistingDeployment -ShellExe $job.pwsh
    $profile = Join-Path $job.userRoot 'Documents/PowerShell/profile.ps1'
    & $job.pwsh -NoProfile -Command ". '$profile'; if (!(Get-Command vim,tmux,dev) -or (Get-Alias vi).Definition -ne 'vim') { exit 1 }; `$text = prompt; if (!`$text.Contains([char]0x279c)) { exit 1 }; `$text"
    if ($LASTEXITCODE -ne 0) { throw 'Deployed PowerShell profile/prompt failed' }
    & node.exe (Join-Path $job.root 'tests/Test-ACP.mjs') (Join-Path $env:NPM_CONFIG_PREFIX 'node_modules/@agentclientprotocol/codex-acp/package.json')
    if ($LASTEXITCODE -ne 0) { throw 'Installed ACP adapter did not initialize' }
    }
    $resultName = if ($RuntimeOnly) { 'runtime-result.json' } else { 'result.json' }
    Save-Json (Join-Path $job.fixture $resultName) ([ordered]@{ok=$true}) ''
} catch {
    try { Stop-Transcript | Out-Null } catch {}
    $resultName = if ($RuntimeOnly) { 'runtime-result.json' } else { 'result.json' }
    Save-Json (Join-Path $job.fixture $resultName) ([ordered]@{ok=$false; error=$_.Exception.Message}) ''
}
'@ ''
    # The bootstrap runs under real ConPTY just as it does in Windows Terminal.
    Invoke-Native $psmux @('-L', $socket, '-f', 'NUL', 'new-session', '-d', '-s', 'deployment', '--', 'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker)
    $resultPath = Join-Path $fixture 'result.json'
    $deadline = (Get-Date).AddMinutes(25)
    while (!(Test-Path -LiteralPath $resultPath) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
    if (!(Test-Path -LiteralPath $resultPath)) { throw "Deployment timed out. Logs/fixture: $fixture" }
    $result = Read-Json $resultPath
    if (!$result.ok) { throw $result.error }
    # Native integration opens its own ConPTY sessions. Run it from the parent
    # rather than nesting a psmux server inside the bootstrap's psmux server.
    Invoke-Native powershell.exe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker, '-RuntimeOnly')
    $runtime = Read-Json (Join-Path $fixture 'runtime-result.json')
    if (!$runtime.ok) { throw $runtime.error }
    Assert $true 'Real bootstrap Apply, second Apply, Check, native integration and PowerShell profile completed'
    $first = Read-Json (Join-Path $fixture 'first-snapshot.json')
    $second = Read-Json (Join-Path $fixture 'second-snapshot.json')
    foreach ($path in $first.Keys) {
        Assert ($second.Contains($path) -and $first[$path].hash -ceq $second[$path].hash -and $first[$path].writeTime -eq $second[$path].writeTime) ('Second run leaves content and timestamp unchanged: ' + $path.Substring($userRoot.Length + 1))
    }
    Assert ($first.Count -eq $second.Count) 'Second run creates no additional managed files'
    Assert (!(Test-Path -LiteralPath (Join-Path $userRoot 'AppData/Local/PersonalDevenv/backups'))) 'Second run produces no configuration backups'
    $secondLog = [IO.File]::ReadAllText((Join-Path $fixture 'second.log'))
    Assert ($secondLog.Contains('CoC extensions: checking latest versions') -and $secondLog.Contains('SKIP Codex ACP') -and $secondLog.Contains('SKIP Obsidian plugin') -and !$secondLog.Contains('Resolved ') -and !$secondLog.Contains('Cloning into')) 'Second run retains existing checkouts/plugins and checks latest CoC extensions'
    Assert ([IO.File]::ReadAllText((Join-Path $userRoot '.vim/plugin-install.log')).Contains('SKIP Vim plugins')) 'Second run skips Vim plugin installation/update'
    foreach ($id in @(Read-Json (Join-Path $userRoot 'notes/.obsidian/community-plugins.json'))) {
        Assert ((Read-Json (Join-Path $userRoot "notes/.obsidian/plugins/$id/data.json")).fixture -ceq 'preserve plugin preferences') "Plugin $id retains tracked settings"
    }
    $adapter = Read-Json (Join-Path $userRoot 'AppData/Roaming/npm/node_modules/@agentclientprotocol/codex-acp/package.json')
    foreach ($target in $adapter.bin.Values) { Assert (Test-Path -LiteralPath (Join-Path $userRoot ('AppData/Roaming/npm/node_modules/@agentclientprotocol/codex-acp/' + $target))) 'ACP adapter executable target exists' }
    Save-Json (Join-Path $fixture 'verification.json') ([ordered]@{
        passed=$true; deploymentAssertions=$script:passed; bootstrapRuns=2
        monitoredFiles=$first.Count; changedFilesOnRerun=0; newBackupsOnRerun=0
        nativeRuntimePassed=$runtime.ok; credentials='separate fake-data suite'
        machineInstallersTested=$false; wslInstallationTested=$false
        fontRegistrationTested=$false; interactiveAuthenticationTested=$false; guiTested=$false
    }) ''
    Write-Output "$script:passed deployment assertions passed. Fixtures and logs: $fixture"
} finally {
    $ErrorActionPreference = 'Continue'
    & $psmux -L $socket kill-server 2>$null | Out-Null
    Write-Output "Retained isolated deployment for review: $fixture"
}
