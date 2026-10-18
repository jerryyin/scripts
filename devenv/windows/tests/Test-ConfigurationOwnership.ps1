param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
$psmux = (Get-Command psmux.exe).Source
$vim = Get-NativeVim
Add-Type -Path (Join-Path $PSScriptRoot 'ConPty.cs')
$fixture = Join-Path $env:TEMP ('config-owner-' + [guid]::NewGuid().ToString('N').Substring(0,8))
$userRoot = Join-Path $fixture "Someone's Windows home"
$candidate = Join-Path $fixture "Someone's rc_files"
$local = Join-Path $userRoot 'AppData/Local'
$state = Join-Path $local 'PersonalDevenv'
$socket = 'config-owner-' + [guid]::NewGuid().ToString('N')
$prior = @{USERPROFILE=$env:USERPROFILE; HOME=$env:HOME; LOCALAPPDATA=$env:LOCALAPPDATA}
$script:passed = 0
$console = $null
function Assert($Ok, [string]$Message) {
    if (!$Ok) { throw "FAIL: $Message; fixture $fixture" }
    $script:passed++; Write-Output "PASS: $Message"
}
try {
    Assert ((Read-Json (Join-Path $root 'defaults/environment.json')).packages.id -notcontains 'JanDeDobbeleer.OhMyPosh') 'Fresh setup does not require the unused prompt engine'
    foreach ($file in @('.powershell/profile.ps1', 'tmux/.psmux.conf', 'tmux/.tmux/psmux-compat.conf', 'tmux/.tmux/mux.ps1', 'tmux/.tmux.conf', 'vim/_vimrc')) {
        $destination = Join-Path $candidate $file
        [void][IO.Directory]::CreateDirectory((Split-Path $destination -Parent))
        Copy-Item -LiteralPath (Join-Path $RcRoot $file) -Destination $destination
    }
    $env:USERPROFILE=$userRoot; $env:HOME=$userRoot; $env:LOCALAPPDATA=$local
    Assert (!(Test-Path -LiteralPath (Join-Path $candidate '.windows'))) 'Deployment uses application-owned configuration without a shared Windows directory'
    $profile = Join-Path $userRoot 'Documents/PowerShell/profile.ps1'
    $source = Join-Path $candidate '.powershell/profile.ps1'
    Save-Text $profile '$global:ExistingProfilePreference = 123' ''
    Set-ProfileLoader $profile $source ''
    Set-NativeMux $userRoot $candidate ''
    $entry = Join-Path $userRoot '.psmux.conf'
    Assert (([IO.File]::ReadAllLines($entry)).Count -eq 2) 'Home psmux entry contains only a comment and source loader'
    Assert ((Get-FileHash -LiteralPath $entry).Hash -eq (Get-FileHash -LiteralPath (Join-Path $candidate 'tmux/.psmux.conf')).Hash) 'Home psmux entry comes verbatim from rc_files'
    Assert ((Get-Item -LiteralPath $entry).LinkType -eq 'SymbolicLink') 'Home psmux entry is a live link, never a copy'
    Set-TrackedFile (Join-Path $userRoot '_vimrc') (Join-Path $candidate 'vim/_vimrc') ''
    Assert ((Get-FileHash -LiteralPath (Join-Path $userRoot '_vimrc')).Hash -eq (Get-FileHash -LiteralPath (Join-Path $candidate 'vim/_vimrc')).Hash) 'Home Vim entry comes verbatim from rc_files'
    Assert ((Get-Item -LiteralPath (Join-Path $userRoot '_vimrc')).LinkType -eq 'SymbolicLink') 'Home Vim entry is a live link, never a copy'
    Assert (!(Test-Path -LiteralPath (Join-Path $state 'profile.ps1'))) 'No copied PowerShell profile in AppData'
    . $profile
    Assert ($global:ExistingProfilePreference -eq 123) 'Existing PowerShell profile content is preserved'
    Assert ($null -eq $script:DevenvState) 'Opening a shell does not read machine state'
    Save-Json (Join-Path $state 'state.json') ([ordered]@{vimExe='unused'; psmuxExe=$psmux; rcRoot=$candidate}) ''
    Assert ((Get-DevenvState).psmuxExe -eq $psmux) 'Machine paths load on demand'
    Add-Content -LiteralPath $source -Value '$global:TrackedProfileEdit = 456'
    . $profile
    Assert ($global:TrackedProfileEdit -eq 456) 'An rc_files edit takes effect without copying or reinstalling'
    $cleanProfile = Join-Path $userRoot 'Documents/WindowsPowerShell/profile.ps1'
    Set-ProfileLoader $cleanProfile $source ''
    Assert ((Get-Item -LiteralPath $cleanProfile).LinkType -eq 'SymbolicLink') 'A managed-only PowerShell profile links directly to rc_files'
    [IO.File]::AppendAllText($cleanProfile, "`n# edited through installed profile`n")
    Assert ([IO.File]::ReadAllText($source).Contains('# edited through installed profile')) 'Edits through an installed PowerShell profile reach rc_files'
    $paths = @($profile, $entry, (Join-Path $userRoot '.psmux/generated.conf'), (Join-Path $userRoot '_vimrc'))
    $times = @($paths | ForEach-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc.Ticks })
    Set-ProfileLoader $profile $source ''
    Set-NativeMux $userRoot $candidate ''
    Set-TrackedFile (Join-Path $userRoot '_vimrc') (Join-Path $candidate 'vim/_vimrc') ''
    for ($i=0; $i -lt $paths.Count; $i++) {
        Assert ((Get-Item -LiteralPath $paths[$i]).LastWriteTimeUtc.Ticks -eq $times[$i]) ('Rerun leaves timestamp unchanged: ' + (Split-Path $paths[$i] -Leaf))
    }
    Invoke-Native $psmux @('-L', $socket, '-f', $entry, 'new-session', '-d', '-s', 'ownership', '--', 'powershell.exe', '-NoProfile')
    $keys = Invoke-Native $psmux @('-L', $socket, 'list-keys') -Capture
    Assert ($keys.Contains('h select-pane -L') -and $keys.Contains('l select-pane -R')) 'Real psmux loads familiar bindings through the source loader'
    Assert ($keys.Contains('/tmux/.tmux/mux.ps1') -and !$keys.Contains('/devenv/windows/')) 'Runtime bindings use rc_files without importing installation scripts'
    $preferences = Join-Path $candidate 'tmux/.tmux/psmux-compat.conf'
    $updated = [IO.File]::ReadAllText($preferences).Replace('bind r source-file ~/.psmux.conf', "bind r source-file ~/.psmux.conf`nbind-key F9 display-message from-rc-files")
    Save-Text $preferences $updated ''
    Set-NativeMux $userRoot $candidate ''
    Invoke-Native $psmux @('-L', $socket, 'source-file', $entry)
    $keys = Invoke-Native $psmux @('-L', $socket, 'list-keys') -Capture
    Assert ($keys.Contains('from-rc-files')) 'Changing rc_files preferences is reflected when setup renders and psmux reloads'
    foreach ($file in @('vim/.vim/config/windows.vim', 'vim/.vim/config/plugins.vim', 'vim/.vim/config/leaderf-files.sh', 'vim/.vim/config/ai-processes.ps1', 'vim/.vim/coc-settings.json')) {
        Save-Text (Join-Path $candidate $file) 'fixture' ''
    }
    Save-Text (Join-Path $candidate 'vim/.vimrc') 'let g:tracked_entry_loaded = 1' ''
    $python = Join-Path $fixture "Someone's Python"
    Save-Text (Join-Path $python 'python399.dll') 'fixture' ''
    Set-NativeVim $userRoot $candidate 'C:/Native Bash/bash.exe' '' $python
    $dataPath = Join-Path $userRoot '.vim/machine-paths.json'
    Assert ((Read-Json $dataPath).pythonDll.EndsWith('python399.dll')) 'Discovered Vim paths are JSON data outside rc_files'
    $reportPath = (Join-Path $fixture 'vim-entry.json').Replace('\','/').Replace("'","''")
    $probe = Join-Path $fixture 'check-entry.vim'
    Save-Text $probe ("execute 'edit ' . fnameescape(expand('~/_vimrc'))`ncall append(line('$'), '`" saved from native Vim')`nwrite`ncall writefile([json_encode({'loaded': g:tracked_entry_loaded, 'bash': g:personal_git_bash, 'python': g:personal_python_dll})], '$reportPath')`nqall!`n") ''
    $console = [DevenvConsole]::new(('"' + $vim + '" -n -i NONE -u "' + (Join-Path $userRoot '_vimrc') + '" -S "' + $probe + '"'), $userRoot)
    $deadline = (Get-Date).AddSeconds(20)
    while (!(Test-Path -LiteralPath $reportPath) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }
    $report = Read-Json $reportPath
    Assert ($report.loaded -eq 1 -and $report.bash -ceq 'C:/Native Bash/bash.exe' -and $report.python.EndsWith('python399.dll')) 'Real Vim loads its tracked entry, machine data and shared vimrc with spaces and apostrophes in paths'
    Assert ((Get-Item -LiteralPath (Join-Path $userRoot '_vimrc')).LinkType -eq 'SymbolicLink' -and [IO.File]::ReadAllText((Join-Path $candidate 'vim/_vimrc')).Contains('saved from native Vim')) 'Saving the installed entry in real Vim preserves its link and edits the Git source'
    $time = (Get-Item -LiteralPath $dataPath).LastWriteTimeUtc.Ticks
    Set-NativeVim $userRoot $candidate 'C:/Native Bash/bash.exe' '' $python
    Assert ((Get-Item -LiteralPath $dataPath).LastWriteTimeUtc.Ticks -eq $time) 'Rerunning Vim setup does not rewrite machine paths'
    $instructions = Join-Path $candidate 'claude/.claude/CLAUDE.md'
    Save-Text $instructions 'tracked instructions v1' ''
    Set-NativeCodex $userRoot $candidate ''
    $agents = Join-Path $userRoot '.codex/AGENTS.md'
    Assert ((Get-Item -LiteralPath $agents).LinkType -eq 'SymbolicLink') 'Agent instructions link directly to their tracked source'
    [IO.File]::WriteAllText($agents, 'local instructions')
    Assert ([IO.File]::ReadAllText($instructions) -ceq 'local instructions') 'Local instruction edits immediately reach rc_files'
    Set-NativeCodex $userRoot $candidate ''
    Assert ([IO.File]::ReadAllText($agents) -ceq 'local instructions') 'Local instruction edits are preserved'
    $skillSource = Join-Path $fixture 'tracked-skills/example'
    $skillTarget = Join-Path $userRoot '.agents/skills/example'
    Save-Text (Join-Path $skillSource 'SKILL.md') 'skill v1' ''
    Save-Text (Join-Path $skillTarget 'SKILL.md') 'skill v1' ''
    $backups = Join-Path $fixture 'backups'
    Set-TrackedDirectory $skillTarget $skillSource $backups
    Assert ((Get-Item -LiteralPath $skillTarget).LinkType -eq 'Junction') 'Existing identical copied skills migrate to directory junctions'
    Save-Text (Join-Path $skillTarget 'SKILL.md') 'edited with an atomic save' ''
    Save-Text (Join-Path $skillTarget 'helper.ps1') '# a new helper' ''
    Assert ([IO.File]::ReadAllText((Join-Path $skillSource 'SKILL.md')) -ceq 'edited with an atomic save') 'Skill edits reach their source even when an editor replaces a file'
    Assert (Test-Path -LiteralPath (Join-Path $skillSource 'helper.ps1')) 'New files in an installed skill are created in rc_files'
    $time = (Get-Item -LiteralPath $skillTarget).CreationTimeUtc.Ticks
    Set-TrackedDirectory $skillTarget $skillSource $backups
    Assert ((Get-Item -LiteralPath $skillTarget).CreationTimeUtc.Ticks -eq $time) 'Rerunning setup leaves skill links intact'
    $conflict = Join-Path $userRoot '.agents/skills/conflicting-copy'
    Save-Text (Join-Path $conflict 'SKILL.md') 'local-only edits' ''
    $rejected = $false
    try { Set-TrackedDirectory $conflict $skillSource $backups } catch { $rejected = $_.Exception.Message.Contains('Local skill edits need recovery') }
    Assert ($rejected -and [IO.File]::ReadAllText((Join-Path $conflict 'SKILL.md')) -ceq 'local-only edits') 'Differing copied skills stop setup without losing local edits'
    Assert (!(Get-ChildItem -LiteralPath $userRoot -Filter '*.link-*' -Recurse -Force)) 'File deployment leaves no temporary links or copies behind'
    Write-Output "$script:passed configuration ownership checks passed. Fixture: $fixture"
} finally {
    if ($console) { $console.Dispose() }
    & $psmux -L $socket kill-server 2>$null | Out-Null
    $env:USERPROFILE=$prior.USERPROFILE; $env:HOME=$prior.HOME; $env:LOCALAPPDATA=$prior.LOCALAPPDATA
}
