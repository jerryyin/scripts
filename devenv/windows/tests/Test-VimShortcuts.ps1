param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
Add-Type -Path (Join-Path $PSScriptRoot 'ConPty.cs')
$fixture = Join-Path $env:TEMP ('vim-shortcuts-' + [guid]::NewGuid().ToString('N').Substring(0,8))
$project = Join-Path $fixture 'project with spaces'
[void][IO.Directory]::CreateDirectory((Join-Path $project 'build'))
Save-Text (Join-Path $project 'first.txt') "audit_search_token`nvalue with spaces`naudit_search_token_suffix`n" ''
Save-Text (Join-Path $project 'second file.txt') "audit_search_token`nvalue with spaces`n" ''
Save-Text (Join-Path $project 'build/generated.txt') "audit_search_token`nvalue with spaces`n" ''
Save-Text (Join-Path $project 'audit.cpp') "int audit_add(int v) { return v + 1; }`nint main() { return audit_add(2); }`nint bad() { return missing_audit_identifier; }`n// end`n" ''
Save-Text (Join-Path $project 'compile_flags.txt') '-xc++' ''
$prior = @{VIM_AUDIT_RESULT=$env:VIM_AUDIT_RESULT; VIM_AUDIT_ROOT=$env:VIM_AUDIT_ROOT; VIM_AUDIT_SCRIPTS=$env:VIM_AUDIT_SCRIPTS; VIM_AUDIT_RC=$env:VIM_AUDIT_RC; PSMUX_TARGET_SESSION=$env:PSMUX_TARGET_SESSION; PSMUX_SESSION=$env:PSMUX_SESSION; TMUX=$env:TMUX}
$console = $null
$socket = 'vim-ai-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$mux = (Get-Command psmux.exe).Source
try {
    $env:VIM_AUDIT_ROOT = $fixture
    $env:VIM_AUDIT_RESULT = Join-Path $fixture 'verification.json'
    $env:VIM_AUDIT_SCRIPTS = Split-Path (Split-Path $root -Parent) -Parent
    $env:VIM_AUDIT_RC = $RcRoot
    $vim = Get-NativeVim
    $loader = Join-Path $RcRoot 'vim/.vimrc'
    $command = '"' + $vim + '" -n -i NONE -u "' + $loader + '" -V1"' + (Join-Path $fixture 'runtime.log') + '" -S "' + (Join-Path $PSScriptRoot 'vim-shortcuts.vim') + '" "' + (Join-Path $project 'first.txt') + '"'
    $console = [DevenvConsole]::new($command, $project)
    $deadline = (Get-Date).AddSeconds(45)
    while (!(Test-Path $env:VIM_AUDIT_RESULT) -and (Get-Date) -lt $deadline) {
        if ($console.Output -match 'Press ENTER or type command') { $console.Send([string][char]13) }
        Start-Sleep -Milliseconds 100
    }
    if (!(Test-Path $env:VIM_AUDIT_RESULT)) { throw "Vim shortcut audit timed out: $fixture" }
    $report = Read-Json $env:VIM_AUDIT_RESULT
    foreach ($check in $report.checks) { Write-Output ((@('FAIL','PASS')[[int]$check.passed]) + ': ' + $check.name) }
    if (@($report.checks | Where-Object { !$_.passed }).Count -or $report.errors.Count) {
        $report.errors | Write-Output
        throw "Vim shortcut audit failed: $fixture"
    }
    function Wait-Ui([bool]$Visible) {
        $end = (Get-Date).AddSeconds(6)
        do {
            try { $ui = Read-Json (Join-Path $fixture 'ui.json'); if ($ui.Contains('visible') -and (($ui.visible -gt 0) -eq $Visible)) { return $true } } catch { }
            Start-Sleep -Milliseconds 100
        } while ((Get-Date) -lt $end)
        return $false
    }
    if (!(Wait-Ui $false)) { throw 'Vim did not enter its normal input loop' }
    foreach ($key in @('\ff','\fb')) {
        $console.Send($key)
        $shown = Wait-Ui $true
        $console.Send([string][char]27)
        $hidden = Wait-Ui $false
        if (!$shown -or !$hidden) { throw "Picker shortcut failed: $key; inspect $fixture" }
        Write-Output "PASS: $key opens its picker and Esc closes it"
        $report.checks += [ordered]@{passed=$true;name=($key+' opens and closes its picker')}
    }
    function Wait-Editor([scriptblock]$Condition) {
        $end = (Get-Date).AddSeconds(10)
        do {
            try { $ui = Read-Json (Join-Path $fixture 'ui.json'); if (& $Condition $ui) { return $true } } catch { }
            Start-Sleep -Milliseconds 100
        } while ((Get-Date) -lt $end)
        return $false
    }
    $signRepository = Join-Path $fixture 'git signs'
    Invoke-Native git @('clone', '--quiet', '--shared', '--', $RcRoot, $signRepository)
    $signFile = Join-Path $signRepository 'vim/.vim/config/tools.vim'
    Add-Content -LiteralPath $signFile -Value '" disposable Git sign check'
    $console.Send(":execute 'edit ' . fnameescape('" + $signFile.Replace('\','/').Replace("'","''") + "')" + [char]13)
    if (!(Wait-Editor {param($s) $s.signify -eq 'git'})) { throw 'Git diff signs did not update' }
    Write-Output 'PASS: Git diff signs update in a real repository'
    $report.checks += [ordered]@{passed=$true;name='Git diff signs update in a real repository'}
    $cpp = (Join-Path $project 'audit.cpp').Replace('\','/')
    $console.Send(":edit $cpp" + [char]13)
    if (!(Wait-Editor { param($s) $s.file -like '*audit.cpp' -and $s.cocReady -and $s.diagnosticErrors -gt 0 })) { throw "CoC/clangd did not publish the expected diagnostic: $fixture" }
    $console.Send(':call cursor(2,21)' + [char]13)
    if (!(Wait-Editor {param($s) $s.line -eq 2})) { throw 'Cursor setup failed' }
    $console.Send('gd')
    if (!(Wait-Editor {param($s) $s.line -eq 1})) { throw "gd did not jump to the definition: $fixture" }
    Write-Output 'PASS: gd jumps to a definition through native CoC/clangd'
    $report.checks += [ordered]@{passed=$true;name='gd jumps to a definition through native CoC/clangd'}
    $console.Send(']g')
    if (!(Wait-Editor {param($s) $s.line -eq 3})) { throw 'Diagnostic navigation failed' }
    Write-Output 'PASS: ]g jumps to a real clangd error'
    $report.checks += [ordered]@{passed=$true;name=']g jumps to a real clangd error'}
    # A local input reader stands in for Codex; it never executes pasted text.
    $fakeCodex = Join-Path $fixture 'codex.exe'
    $stubSource = Join-Path $fixture 'codex.cs'
    Save-Text $stubSource 'using System; public class CodexStub { public static void Main() { Console.Write("FAKE_AI_READY> "); while (Console.ReadLine() != null) {} } }' ''
    $compile = 'Add-Type -Path ' + "'$stubSource'" + ' -OutputAssembly ' + "'$fakeCodex'" + ' -OutputType ConsoleApplication'
    & powershell.exe -NoProfile -Command $compile
    if ($LASTEXITCODE -ne 0) { throw 'Local input-reader fixture could not be compiled' }
    & $mux -L $socket -f NUL new-session -d -s ai -- $fakeCodex /d /q | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Isolated AI pane did not start' }
    $env:PSMUX_TARGET_SESSION = $socket + '__ai'
    $port = [IO.File]::ReadAllText((Join-Path $env:USERPROFILE ('.psmux/' + $socket + '__ai.port'))).Trim()
    $env:TMUX = '/tmp/psmux/' + $socket + ',' + $port + ',0'
    $end = (Get-Date).AddSeconds(8)
    do {
        $name = (& $mux display-message -p '#S') -join ''
        if ($name -match '(^|__)ai$') { break }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $end)
    if ($name -notmatch '(^|__)ai$') { throw 'Refusing to send keys: isolated server routing failed' }
    $console.Send(':let $PSMUX_TARGET_SESSION = ' + "'$($env:PSMUX_TARGET_SESSION)'" + [char]13)
    $console.Send(':let $TMUX = ' + "'$($env:TMUX)'" + ' | let $PSMUX_SESSION = "ai"' + [char]13)
    $console.Send('\cc')
    $end = (Get-Date).AddSeconds(12)
    $expected = (Read-Json (Join-Path $fixture 'ui.json')).file.Replace('\','/') + ':3'
    do {
        $pasted = (& $mux capture-pane -p) -join ''
        if ($pasted.Replace('\','/').Contains($expected)) { break }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $end)
    if (!$pasted.Replace('\','/').Contains($expected)) { throw "AI pane transfer failed: $fixture" }
    Write-Output 'PASS: \cc pastes the current file/line into an isolated native AI pane'
    $report.checks += [ordered]@{passed=$true;name='\cc pastes file/line into an isolated native AI pane'}
    $console.Send('ggV\ty')
    $end = (Get-Date).AddSeconds(5)
    do {
        $copied = ((& $mux show-buffer) -join "`n").Trim()
        if ($copied -eq 'int audit_add(int v) { return v + 1; }') { break }
        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $end)
    if ($copied -ne 'int audit_add(int v) { return v + 1; }') { throw 'Visual selection was not copied into the native multiplexer buffer' }
    Write-Output 'PASS: visual \ty copies selection into the native multiplexer buffer'
    $report.checks += [ordered]@{passed=$true;name='visual \ty copies selection into the native multiplexer buffer'}
    $console.Send([string][char]27)
    Start-Sleep -Milliseconds 100
    & $mux set-buffer paste_bridge_marker | Out-Null
    $console.Send(':call cursor(4,1)' + [char]13)
    $console.Send('\tpgg')
    if (!(Wait-Editor {param($s) $s.bridgePresent -and $s.line -eq 1})) { throw 'Native paste shortcut did not finish in normal mode' }
    Write-Output 'PASS: \tp pastes the native buffer and leaves normal keys usable'
    $report.checks += [ordered]@{passed=$true;name='\tp pastes the native buffer and leaves normal keys usable'}
    Save-Json $env:VIM_AUDIT_RESULT $report ''
    $console.Send(':qall!' + [char]13)
    Write-Output "$($report.checks.Count) Vim shortcut checks passed. Artifacts: $fixture"
} catch {
    if ($console) {
        $debug = (Join-Path $fixture 'messages.txt').Replace('\','/')
        $console.Send([string][char]27)
        $console.Send(":call writefile([execute('messages')], '$debug')" + [char]13)
        Start-Sleep -Milliseconds 300
        [IO.File]::WriteAllText((Join-Path $fixture 'console.log'), $console.Output)
    }
    if ($pasted) { Save-Text (Join-Path $fixture 'dummy-pane.txt') $pasted '' }
    throw
} finally {
    & $mux -L $socket kill-server 2>$null | Out-Null
    if ($console) { $console.Dispose() }
    foreach ($name in $prior.Keys) { [Environment]::SetEnvironmentVariable($name, $prior[$name], 'Process') }
}
