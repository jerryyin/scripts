param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
. (Join-Path $PSScriptRoot 'RcFixture.ps1')
$psmux = (Get-Command psmux.exe).Source
$vim = Get-NativeVim
Add-Type -Path (Join-Path $PSScriptRoot 'ConPty.cs')
$fixture = Join-Path $env:TEMP ('config-owner-' + [guid]::NewGuid().ToString('N').Substring(0,8))
$userRoot = Join-Path $fixture "Someone's Windows home"
$candidate = Join-Path $fixture "Someone's rc_files"
$local = Join-Path $userRoot 'AppData/Local'
$state = Join-Path $local 'PersonalDevenv'
$socket = 'config-owner-' + [guid]::NewGuid().ToString('N')
$prior = @{USERPROFILE=$env:USERPROFILE; HOME=$env:HOME; LOCALAPPDATA=$env:LOCALAPPDATA; GIT_CONFIG_GLOBAL=$env:GIT_CONFIG_GLOBAL}
$script:passed = 0
$console = $null
function Assert($Ok, [string]$Message) {
    if (!$Ok) { throw "FAIL: $Message; fixture $fixture" }
    $script:passed++; Write-Output "PASS: $Message"
}
try {
    Assert ((Read-Json (Join-Path $root 'defaults/environment.json')).packages.id -notcontains 'JanDeDobbeleer.OhMyPosh') 'Fresh setup does not require the unused prompt engine'
    Copy-RcFixture $RcRoot $candidate
    $env:USERPROFILE=$userRoot; $env:HOME=$userRoot; $env:LOCALAPPDATA=$local
    Assert (!(Test-Path -LiteralPath (Join-Path $candidate '.windows'))) 'Deployment uses application-owned configuration without a shared Windows directory'
    $profile = Join-Path $userRoot 'Documents/PowerShell/profile.ps1'
    $source = Join-Path $candidate 'powershell/.windows/@DOCUMENTS/PowerShell/profile.ps1'
    Invoke-TestLink $profile $source ''
    Invoke-RcSetup $candidate @('apply', '--package', 'tmux', '--package', 'vim', '--home', $userRoot)
    Assert (!(Test-Path -LiteralPath (Join-Path $userRoot '_vimrc'))) 'Windows configuration has one shared .vimrc without a redundant alias'
    $entry = Join-Path $userRoot '.psmux.conf'
    Assert ([IO.File]::ReadAllText($entry).Contains('source-file ~/.config/tmux/common.conf')) 'Native psmux entry loads the shared configuration directly'
    Assert ((Get-FileHash -LiteralPath $entry).Hash -eq (Get-FileHash -LiteralPath (Join-Path $candidate 'tmux/.windows/.psmux.conf')).Hash) 'Home psmux entry comes verbatim from rc_files'
    Assert ((Get-Item -LiteralPath $entry).LinkType -eq 'SymbolicLink') 'Home psmux entry is a live link, never a copy'
    Assert ((Get-FileHash -LiteralPath (Join-Path $userRoot '.vimrc')).Hash -eq (Get-FileHash -LiteralPath (Join-Path $candidate 'vim/.vimrc')).Hash) 'Home Vim entry comes verbatim from rc_files'
    Assert ((Get-Item -LiteralPath (Join-Path $userRoot '.vimrc')).LinkType -eq 'SymbolicLink') 'Home Vim entry is a live link, never a copy'
    Assert (!(Test-Path -LiteralPath (Join-Path $state 'profile.ps1'))) 'No copied PowerShell profile in AppData'
    . $profile
    Assert (Test-InstalledLink $profile $source) 'PowerShell profile directly links to its tracked source'
    $savedPath = $env:PATH
    $savedUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    try {
        Add-ToolPath (Split-Path $vim -Parent)
        $once = $env:PATH
        Add-ToolPath (Split-Path $vim -Parent)
        Assert ($env:PATH -ceq $once) 'Adding a native tool directory is unchanged on rerun'
        Assert ([Environment]::GetEnvironmentVariable('Path', 'User') -ceq $savedUserPath) 'Redirected setup tests never change the account PATH'
        Assert ((vim --version | Select-Object -First 1) -like 'VIM*') 'The tracked profile runs native Vim without generated machine state'
        Assert ((tmux -V) -match 'psmux') 'The tracked profile runs native psmux without installation scripts'
    } finally { $env:PATH = $savedPath }
    Assert (!(Test-Path -LiteralPath (Join-Path $state 'state.json'))) 'The tracked shell needs no generated machine-path file'
    Add-Content -LiteralPath $source -Value '$global:TrackedProfileEdit = 456'
    . $profile
    Assert ($global:TrackedProfileEdit -eq 456) 'An rc_files edit takes effect without copying or reinstalling'
    $cleanProfile = Join-Path $userRoot 'Documents/WindowsPowerShell/profile.ps1'
    Invoke-TestLink $cleanProfile $source ''
    Assert ((Get-Item -LiteralPath $cleanProfile).LinkType -eq 'SymbolicLink') 'A managed-only PowerShell profile links directly to rc_files'
    [IO.File]::AppendAllText($cleanProfile, "`n# edited through installed profile`n")
    Assert ([IO.File]::ReadAllText($source).Contains('# edited through installed profile')) 'Edits through an installed PowerShell profile reach rc_files'
    $paths = @($profile, $entry, (Join-Path $userRoot '.config/tmux/common.conf'), (Join-Path $userRoot '.vimrc'))
    $times = @($paths | ForEach-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc.Ticks })
    Invoke-TestLink $profile $source ''
    Invoke-TestLink (Join-Path $userRoot '.psmux.conf') (Join-Path $candidate 'tmux/.windows/.psmux.conf') ''
    Invoke-RcSetup $candidate @('apply', '--package', 'tmux', '--package', 'vim', '--home', $userRoot)
    for ($i=0; $i -lt $paths.Count; $i++) {
        Assert ((Get-Item -LiteralPath $paths[$i]).LastWriteTimeUtc.Ticks -eq $times[$i]) ('Rerun leaves timestamp unchanged: ' + (Split-Path $paths[$i] -Leaf))
    }
    Invoke-Native $psmux @('-L', $socket, '-f', $entry, 'new-session', '-d', '-s', 'ownership', '--', 'powershell.exe', '-NoProfile')
    $keys = Invoke-Native $psmux @('-L', $socket, 'list-keys') -Capture
    Assert ($keys.Contains('h select-pane -L') -and $keys.Contains('l select-pane -R')) 'Real psmux loads familiar bindings through the source loader'
    Assert ($keys.Contains('~/.tmux/mux.ps1') -and !$keys.Contains('/devenv/windows/')) 'Runtime bindings use rc_files without importing installation scripts'
    $preferences = Join-Path $userRoot '.config/tmux/common.conf'
    [IO.File]::AppendAllText($preferences, "`nset-option -g prefix C-a`nset -g base-index 1`nbind-key F9 display-message from-rc-files`n")
    Invoke-Native $psmux @('-L', $socket, 'source-file', $entry)
    $keys = Invoke-Native $psmux @('-L', $socket, 'list-keys') -Capture
    Assert ($keys.Contains('from-rc-files')) 'Edits through the installed shared configuration take effect on reload without setup'
    Assert ((Invoke-Native $psmux @('-L', $socket, 'show-options', '-g', 'prefix') -Capture).Contains('C-a')) 'Shared set-option preferences reach native psmux without a parser'
    Assert ((Invoke-Native $psmux @('-L', $socket, 'show-options', '-g', 'base-index') -Capture).Contains('1')) 'Previously filtered shared options reach native psmux directly'
    foreach ($file in @('vim/.windows/.vim/config/windows.vim', 'vim/.vim/config/plugins.vim', 'vim/.vim/config/project-files.py', 'vim/.windows/.vim/config/ai-processes.ps1', 'vim/.vim/coc-settings.json')) {
        Save-Text (Join-Path $candidate $file) 'fixture' ''
    }
    Save-Text (Join-Path $candidate 'vim/.vimrc') 'let g:tracked_entry_loaded = 1' ''
    $reportPath = (Join-Path $fixture 'vim-entry.json').Replace('\','/').Replace("'","''")
    $probe = Join-Path $fixture 'check-entry.vim'
    Save-Text $probe ("execute 'edit ' . fnameescape(expand('~/.vimrc'))`ncall append(line('$'), '`" saved from native Vim')`nwrite`ncall writefile([json_encode({'loaded': g:tracked_entry_loaded})], '$reportPath')`nqall!`n") ''
    $console = [DevenvConsole]::new(('"' + $vim + '" -n -i NONE -S "' + $probe + '"'), $userRoot)
    $deadline = (Get-Date).AddSeconds(20)
    while (!(Test-Path -LiteralPath $reportPath) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }
    $report = Read-Json $reportPath
    Assert ($report.loaded -eq 1) 'Real Vim loads the shared vimrc directly with spaces and apostrophes in paths'
    Assert ((Get-Item -LiteralPath (Join-Path $userRoot '.vimrc')).LinkType -eq 'SymbolicLink' -and [IO.File]::ReadAllText((Join-Path $candidate 'vim/.vimrc')).Contains('saved from native Vim')) 'Saving the installed vimrc in real Vim preserves its link and edits the shared Git source'
    Assert (!(Test-Path -LiteralPath (Join-Path $userRoot '.vim/machine-paths.json')) -and !(Test-Path -LiteralPath (Join-Path $userRoot '.psmux/generated.conf'))) 'Configuration deployment generates no Vim paths or tmux translation'
    $instructions = Join-Path $candidate 'claude/.claude/CLAUDE.md'
    Save-Text $instructions 'tracked instructions v1' ''
    Invoke-TestLink (Join-Path $userRoot '.codex/AGENTS.md') $instructions ''
    Set-NativeCodex $userRoot $candidate ''
    $agents = Join-Path $userRoot '.codex/AGENTS.md'
    Assert ((Get-Item -LiteralPath $agents).LinkType -eq 'SymbolicLink') 'Agent instructions link directly to their tracked source'
    [IO.File]::WriteAllText($agents, 'local instructions')
    Assert ([IO.File]::ReadAllText($instructions) -ceq 'local instructions') 'Local instruction edits immediately reach rc_files'
    Invoke-TestLink (Join-Path $userRoot '.codex/AGENTS.md') $instructions ''
    Set-NativeCodex $userRoot $candidate ''
    Assert ([IO.File]::ReadAllText($agents) -ceq 'local instructions') 'Local instruction edits are preserved'
    $skillSource = Join-Path $fixture 'tracked-skills/example'
    $skillTarget = Join-Path $userRoot '.agents/skills/example'
    Save-Text (Join-Path $skillSource 'SKILL.md') 'skill v1' ''
    Save-Text (Join-Path $skillTarget 'SKILL.md') 'skill v1' ''
    $backups = Join-Path $fixture 'backups'
    Invoke-TestLink $skillTarget $skillSource $backups
    Assert ((Get-Item -LiteralPath $skillTarget).LinkType -in @('Junction','SymbolicLink')) 'Existing identical copied skills migrate to directory links'
    Save-Text (Join-Path $skillTarget 'SKILL.md') 'edited with an atomic save' ''
    Save-Text (Join-Path $skillTarget 'helper.ps1') '# a new helper' ''
    Assert ([IO.File]::ReadAllText((Join-Path $skillSource 'SKILL.md')) -ceq 'edited with an atomic save') 'Skill edits reach their source even when an editor replaces a file'
    Assert (Test-Path -LiteralPath (Join-Path $skillSource 'helper.ps1')) 'New files in an installed skill are created in rc_files'
    $time = (Get-Item -LiteralPath $skillTarget).CreationTimeUtc.Ticks
    Invoke-TestLink $skillTarget $skillSource $backups
    Assert ((Get-Item -LiteralPath $skillTarget).CreationTimeUtc.Ticks -eq $time) 'Rerunning setup leaves skill links intact'
    $conflict = Join-Path $userRoot '.agents/skills/conflicting-copy'
    Save-Text (Join-Path $conflict 'SKILL.md') 'local-only edits' ''
    $rejected = $false
    try { Invoke-TestLink $conflict $skillSource $backups } catch { $rejected = $_.Exception.Message.Contains('Configuration conflict') }
    Assert ($rejected -and [IO.File]::ReadAllText((Join-Path $conflict 'SKILL.md')) -ceq 'local-only edits') 'Differing copied skills stop setup without losing local edits'
    $env:GIT_CONFIG_GLOBAL = Join-Path $userRoot '.gitconfig'
    Save-Text $env:GIT_CONFIG_GLOBAL "[user]`nname = old name`n" ''
    $rejected = $false
    try { Invoke-RcSetup $candidate @('apply', '--package', 'git', '--home', $userRoot, '--backup', $backups) } catch { $rejected = $_.Exception.Message.Contains('Configuration conflict') }
    Assert $rejected 'Git deployment preserves a differing local configuration for review'
    Invoke-TestLink $env:GIT_CONFIG_GLOBAL (Join-Path $candidate 'git/.gitconfig') $backups -ReplaceExisting
    Invoke-RcSetup $candidate @('apply', '--package', 'git', '--home', $userRoot, '--backup', $backups)
    Assert (Test-InstalledLink $env:GIT_CONFIG_GLOBAL (Join-Path $candidate 'git/.gitconfig')) 'Windows global Git configuration links to shared preferences'
    Assert ((Invoke-Native git @('config','--global','--get','alias.cmt') -Capture) -ceq 'commit --signoff') 'Windows inherits the Linux Git aliases'
    Assert (Test-InstalledLink (Join-Path $userRoot '.config/git/platform.conf') (Join-Path $candidate 'git/.windows/.config/git/platform.conf')) 'Windows Git helper choices are linked to tracked platform settings'
    Invoke-Native git @('config','--global','alias.windows-link-proof','status')
    Assert ([IO.File]::ReadAllText((Join-Path $candidate 'git/.gitconfig')).Contains('windows-link-proof')) 'git config --global edits the tracked source through its symlink'
    Assert (Test-InstalledLink $env:GIT_CONFIG_GLOBAL (Join-Path $candidate 'git/.gitconfig')) 'Git preserves its configuration symlink when saving'
    Invoke-RcSetup $candidate @('apply', '--package', 'git', '--home', $userRoot, '--backup', $backups)
    Assert (Test-InstalledLink (Join-Path $userRoot '.gitignore') (Join-Path $candidate 'git/.gitignore')) 'Windows global ignore rules link directly to the shared source'
    $healthHome = Join-Path $fixture 'health home'
    $env:USERPROFILE=$healthHome; $env:HOME=$healthHome; $env:LOCALAPPDATA=Join-Path $healthHome 'AppData/Local'
    $healthDocuments = Join-Path $healthHome 'Documents'
    $healthTerminal = Join-Path $healthHome 'Terminal/settings.json'
    Invoke-RcSetup $RcRoot @(Get-DeploymentArguments 'apply' $healthHome $RcRoot $healthDocuments $healthTerminal)
    Set-NativeCodex $healthHome $RcRoot ''
    Assert (@(Get-ConfigurationProblems $healthHome $RcRoot $healthDocuments $healthTerminal).Count -eq 0) 'Readiness validates actual links and skills without a generated state file'
    Save-Text $healthTerminal ([IO.File]::ReadAllText($healthTerminal)) ''
    # Save-Text skips identical content, so replace only this disposable link.
    $terminalCopy=[IO.File]::ReadAllText($healthTerminal)
    Remove-Item -LiteralPath $healthTerminal
    [IO.File]::WriteAllText($healthTerminal,$terminalCopy)
    $problems=@(Get-ConfigurationProblems $healthHome $RcRoot $healthDocuments $healthTerminal)
    Assert ($problems.Count -eq 1 -and $problems[0].Contains($healthTerminal)) 'Readiness rejects a detached copy even when its content matches'
    Remove-Item -LiteralPath (Join-Path $healthHome '.config/tmux/common.conf')
    Assert (@(Get-ConfigurationProblems $healthHome $RcRoot $healthDocuments $healthTerminal | Where-Object { $_ -like '*common.conf*' }).Count -gt 0) 'Readiness reports a missing shared configuration link'
    Assert (!(Get-ChildItem -LiteralPath $userRoot -Filter '*.link-*' -Recurse -Force)) 'File deployment leaves no temporary links or copies behind'
    Write-Output "$script:passed configuration ownership checks passed. Fixture: $fixture"
} finally {
    if ($console) { $console.Dispose() }
    try {
        $env:USERPROFILE=$userRoot; $env:HOME=$userRoot; $env:LOCALAPPDATA=$local
        & $psmux -L $socket kill-server 2>$null | Out-Null
    } finally {
        $env:USERPROFILE=$prior.USERPROFILE; $env:HOME=$prior.HOME; $env:LOCALAPPDATA=$prior.LOCALAPPDATA; $env:GIT_CONFIG_GLOBAL=$prior.GIT_CONFIG_GLOBAL
    }
}
