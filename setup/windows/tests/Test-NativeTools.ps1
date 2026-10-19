param(
    [Parameter(Mandatory=$true)][string]$VimExe,
    [Parameter(Mandatory=$true)][string]$PsmuxExe,
    [Parameter(Mandatory=$true)][string]$RcRoot,
    [string]$NodeDirectory = '',
    [string]$ClangdExe = '',
    [string]$FixturePath,
    [switch]$InstallPlugins,
    [switch]$UseExistingDeployment,
    [string]$ShellExe = 'powershell.exe'
)
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
. (Join-Path $PSScriptRoot 'RcFixture.ps1')
$fixture = if ($FixturePath) { $FixturePath } else { Join-Path ([IO.Path]::GetTempPath()) ('personal-devenv-native-' + [guid]::NewGuid().ToString('N')) }
$originalHome = $env:HOME; $originalProfile = $env:USERPROFILE; $originalLocal = $env:LOCALAPPDATA
$originalPath = $env:PATH
$socket = 'devenv-test-' + [guid]::NewGuid().ToString('N')
[void][IO.Directory]::CreateDirectory($fixture)
$script:passed = 0
function Assert($Ok, [string]$Label) {
    if (!$Ok) { throw "FAIL: $Label" }
    $script:passed++; Write-Output "PASS: $Label"
}
function Start-ConsoleVim([string]$Session, [string[]]$VimArguments) {
    # Match actual usage: PowerShell launches Vim. psmux's direct command argv
    # forwarding can lose quotes around paths containing spaces.
    $command = '& ' + "'" + $VimExe.Replace("'", "''") + "' "
    $trace = (Join-Path $fixture 'vim-source-trace.log').Replace('\', '/').Replace("'", "''")
    $VimArguments = @('--cmd', "autocmd SourcePre * call writefile([expand('<afile>')], '$trace', 'a')") + $VimArguments
    $command += (($VimArguments | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }) -join ' ')
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    Invoke-Native $PsmuxExe @('-L', $socket, '-f', 'NUL', 'new-session', '-d', '-s', $Session, '--', 'powershell.exe', '-NoProfile', '-EncodedCommand', $encoded)
}
try {
    $env:HOME = $fixture; $env:USERPROFILE = $fixture; $env:LOCALAPPDATA = Join-Path $fixture 'AppData/Local'
    $env:PATH = (Split-Path $PsmuxExe -Parent) + ';' + $env:PATH
    if ($NodeDirectory) { $env:PATH = $NodeDirectory + ';' + $env:PATH }
    if ($ClangdExe) { $env:PATH = (Split-Path $ClangdExe -Parent) + ';' + $env:PATH }
    if (!$UseExistingDeployment) {
        Invoke-TestLink (Join-Path $fixture 'Documents/PowerShell/profile.ps1') (Join-Path $RcRoot 'powershell/.windows/@DOCUMENTS/PowerShell/profile.ps1') ''
        Invoke-RcSetup $RcRoot @('apply', '--package', 'vim', '--package', 'tmux', '--home', $fixture)
    }
    $plugPath = Join-Path $fixture '.vim/autoload/plug.vim'
    [void][IO.Directory]::CreateDirectory((Split-Path $plugPath -Parent))
    if ($InstallPlugins) {
        # Exercise the real installer under ConPTY, matching Windows Terminal.
        # Win32 Vim's color APIs require a console even for Ex-mode installs.
        $jobPath = Join-Path $fixture 'install-job.json'
        $resultPath = Join-Path $fixture 'install-result.json'
        if (Test-Path -LiteralPath $resultPath) { Remove-Item -LiteralPath $resultPath }
        Save-Json $jobPath ([ordered]@{ module=(Join-Path $root 'Devenv.psm1'); vim=$VimExe; userRoot=$fixture; rcRoot=$RcRoot }) ''
        $worker = Join-Path $fixture 'install-worker.ps1'
        Save-Text $worker @'
$ErrorActionPreference = 'Stop'
$job = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'install-job.json') -Raw | ConvertFrom-Json
try {
    Import-Module $job.module -Force -DisableNameChecking
    $env:PATH = (Split-Path $job.vim -Parent) + ';' + $env:PATH
    Invoke-RcSetup $job.rcRoot @('--package', 'vim', '--home', $job.userRoot, '--backup', (Join-Path $job.userRoot 'backups'))
    $result = @{ ok=$true }
} catch { $result = @{ ok=$false; error=$_.Exception.Message } }
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'install-result.json'), ($result | ConvertTo-Json))
'@ ''
        Invoke-Native $PsmuxExe @('-L', $socket, '-f', 'NUL', 'new-session', '-d', '-s', 'installer', '--', 'powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $worker)
        $deadline = (Get-Date).AddMinutes(10)
        while (!(Test-Path -LiteralPath $resultPath) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
        if (!(Test-Path -LiteralPath $resultPath)) { throw "Plugin installation timed out. Fixture: $fixture" }
        $installation = Read-Json $resultPath
        if (!$installation.ok) {
            & $PsmuxExe -L $socket capture-pane -p -t installer | Out-Host
            throw $installation.error
        }
        Assert $true 'Real Vim plugin and CoC extension installation succeeds under native ConPTY'
    }
    $reportPath = (Join-Path $fixture 'vim-report.json').Replace('\', '/')
    $filePath = (Join-Path $fixture 'edited.txt').Replace('\', '/')
    $pickerProject = (Join-Path $fixture 'picker project').Replace('\', '/')
    Save-Text (Join-Path $pickerProject 'source file.cpp') 'fixture' ''
    Save-Text (Join-Path $pickerProject 'build/generated.inc') 'fixture' ''
    $vimTest = @"
call setline(1, ['native Windows editing works'])
execute 'write! ' . fnameescape('$filePath')
let report = {'vimrc': resolve(`$MYVIMRC), 'clipboard': has('clipboard'), 'python': has('python3'), 'trueColor': &termguicolors, 'theme': get(g:, 'colors_name', ''), 'prefixMap': maparg('<leader>ss', 'n'), 'muxCopy': maparg('<leader>ty', 'v'), 'shellResult': system('echo NATIVE_OK'), 'shell': &shell, 'pluginsDeclared': len(g:plugs), 'leaderfHook': has_key(g:plugs.LeaderF, 'do'), 'polyglot': has_key(g:plugs, 'vim-polyglot'), 'ctags': has_key(g:plugs, 'vim-gutentags'), 'cocHome': !empty(globpath(&runtimepath, 'autoload/coc/util.vim')) ? coc#util#get_config_home() : g:coc_config_home, 'vista': g:vista_default_executive, 'clangd': executable('clangd'), 'scripts': execute('scriptnames')}
call leaderf#versionCheck()
let g:Lf_UseCache = 0
let g:picker_project = '$pickerProject'
python3 from leaderf.fileExpl import FileExplorer
let report.leaderfFiles = py3eval("list(FileExplorer().getContent(arguments={'directory': [vim.eval('g:picker_project')]}))")
execute 'cd ' . fnameescape(g:picker_project)
let report.leaderfCurrentFiles = py3eval('list(FileExplorer().getContent())')
call writefile([json_encode(report)], '$reportPath')
qall!
"@
    $vimScript = Join-Path $fixture 'check.vim'
    Save-Text $vimScript $vimTest ''
    if (Test-Path -LiteralPath $reportPath) { Remove-Item -LiteralPath $reportPath }
    # Exercise UI/color setup in an actual native console. Headless Ex mode
    # can abort the shared vimrc when Win32 color APIs have no console.
    $runtimeLog = Join-Path $fixture 'vim-runtime.log'
    if (Test-Path -LiteralPath $runtimeLog) { Remove-Item -LiteralPath $runtimeLog }
    Assert (!(Test-Path -LiteralPath (Join-Path $fixture '_vimrc'))) 'Native deployment has no redundant Windows vimrc alias'
    Start-ConsoleVim 'vim-runtime' @('-n', ('-V1' + $runtimeLog), '-i', 'NONE', '-S', $vimScript)
    $deadline = (Get-Date).AddSeconds(90)
    $probeAt = Get-Date
    while (!(Test-Path -LiteralPath $reportPath) -and (Get-Date) -lt $deadline) {
        if ((Get-Date) -ge $probeAt) {
            try {
                $pane = Invoke-Native $PsmuxExe @('-L', $socket, 'capture-pane', '-p', '-t', 'vim-runtime') -Capture
                if ($pane -match 'Press ENTER or type command to continue') {
                    Invoke-Native $PsmuxExe @('-L', $socket, 'send-keys', '-t', 'vim-runtime', 'Enter')
                }
            } catch { } # The new session can take a moment to register.
            $probeAt = (Get-Date).AddSeconds(2)
        }
        Start-Sleep -Milliseconds 200
    }
    if (!(Test-Path -LiteralPath $reportPath)) { & $PsmuxExe -L $socket capture-pane -p -t vim-runtime | Out-Host }
    Assert (Test-Path -LiteralPath $reportPath) 'Native Vim sources the shared vimrc directly'
    $report = Read-Json $reportPath
    $expectedVimrc = (Join-Path $RcRoot 'vim/.vimrc').Replace('\', '/').ToLowerInvariant()
    Assert ($report.vimrc.Replace('\', '/').ToLowerInvariant() -ceq $expectedVimrc) 'Windows Vim automatically discovers the shared .vimrc without a -u override'
    Assert ([IO.File]::ReadAllText($filePath).Contains('native Windows editing works')) 'Native Vim edits and writes a Windows file'
    Assert ($report.clipboard -eq 1) 'Native Vim has Windows clipboard support'
    Assert ($report.trueColor -eq 1) 'Native Vim enables true color without Linux terminal variables'
    Assert (![IO.File]::ReadAllText($runtimeLog).Contains('Error detected')) 'Interactive Vim starts in an unnamed buffer without a startup error'
    if (Test-Path -LiteralPath (Join-Path $fixture '.vim/plugged/gruvbox-material/colors/gruvbox-material.vim')) { Assert ($report.theme -ceq 'gruvbox-material') 'Native Vim actually loads the shared Gruvbox Material theme' }
    Assert ($report.python -eq 1) 'Vim discovers the registered native Python runtime for LeaderF'
    Assert (@($report.leaderfFiles).Count -eq 1 -and $report.leaderfFiles[0] -ceq 'source file.cpp' -and @($report.leaderfCurrentFiles).Count -eq 1 -and $report.leaderfCurrentFiles[0] -ceq 'source file.cpp') 'Real LeaderF lists explicit and current Windows folders through the shared Python helper'
    Assert ($report.shellResult.Trim() -eq 'NATIVE_OK' -and $report.shell -match 'cmd.exe$') 'Windows plugin commands use native shell quoting'
    Assert ($report.prefixMap -like '*Grepper -tool rg*') 'Existing search shortcut uses native ripgrep'
    Assert ($report.pluginsDeclared -gt 20 -and !$report.leaderfHook) 'Shared plugins reused with Unix-only build hook adapted'
    Assert (!$report.polyglot -and !$report.ctags -and $report.vista -ceq 'coc') 'Native Windows uses built-in syntax and LSP navigation without Polyglot patches or Ctags'
    Assert ($report.muxCopy -like '*psmux load-buffer*') 'One shared copy mapping selects the native multiplexer'
    $expectedCoc = [IO.Path]::GetFullPath((Join-Path $RcRoot 'vim/.vim')).Replace('\', '/').ToLowerInvariant()
    Assert ($report.cocHome.Replace('\', '/').ToLowerInvariant() -ceq $expectedCoc) 'CoC reads its settings directly from the rc_files checkout'
    $cocSettings = Read-Json (Join-Path $RcRoot 'vim/.vim/coc-settings.json')
    Assert ($cocSettings['semanticTokens.enable'] -and $cocSettings.languageserver.clangd.command -ceq 'clangd') 'The shared CoC settings enable clangd semantic highlighting'
    Assert (!$report.scripts.Contains('vim-windows.vim') -and !$report.scripts.Contains((Join-Path $fixture '.vim/config').Replace('\', '/'))) 'Vim loads shared configuration without the former shim or copied settings'
    if ($ClangdExe) {
        Assert ($report.clangd -eq 1) 'Native clangd is available to Vim and CoC'
        $clangdJob = Join-Path $fixture 'clangd-test.json'
        Save-Json $clangdJob ([ordered]@{clangd=$ClangdExe; arguments=@($cocSettings.languageserver.clangd.args)}) ''
        Invoke-Native node @((Join-Path $PSScriptRoot 'Test-Clangd.mjs'), $clangdJob)
        Assert $true 'Native clangd responds to semantic highlighting requests with the shared server flags'
    }

    $config = Join-Path $fixture '.psmux.conf'
    # Explicitly run installed Windows PowerShell in this test. Fresh-machine
    # provisioning installs PowerShell 7 and makes it the default pane shell.
    Invoke-Native $PsmuxExe @('-L', $socket, '-f', $config, 'new-session', '-d', '-s', 'work', '--', $ShellExe, '-NoLogo', '-NoProfile')
    Invoke-Native $PsmuxExe @('-L', $socket, 'set-option', '-g', 'default-shell', $ShellExe)
    $prefix = Invoke-Native $PsmuxExe @('-L', $socket, 'show-options', '-g', 'prefix') -Capture
    Assert ($prefix.Contains('`')) 'Native psmux loads the same backtick prefix'
    $keys = Invoke-Native $PsmuxExe @('-L', $socket, 'list-keys') -Capture
    Assert ($keys -match 'split-window -h' -and $keys -match 'split-window -v') 'Native psmux loads familiar split bindings'
    Assert ($keys -match 'F10' -and $keys -match 'search-backward') 'Nested-session toggle and previous-command copy bindings reused'
    Invoke-Native $PsmuxExe @('-L', $socket, 'split-window', '-h', '-t', 'work')
    $panes = Invoke-Native $PsmuxExe @('-L', $socket, 'list-panes', '-t', 'work') -Capture
    Assert (@($panes -split "`n").Count -eq 2) 'Native PowerShell session splits into two panes'
    Invoke-Native $PsmuxExe @('-L', $socket, 'new-window', '-d', '-t', 'work', '-n', 'logs')
    $windows = Invoke-Native $PsmuxExe @('-L', $socket, 'list-windows', '-t', 'work') -Capture
    Assert (@($windows -split "`n").Count -eq 2) 'Native session supports multiple windows'
    Invoke-Native $PsmuxExe @('-L', $socket, 'has-session', '-t', 'work')
    Assert $true 'Detached session survives independent client commands'
    Write-Output "$script:passed native integration assertions passed. Fixture: $fixture"
} finally {
    $cleanupPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $PsmuxExe -L $socket kill-server 2>$null | Out-Null
    $ErrorActionPreference = $cleanupPreference
    $env:HOME = $originalHome; $env:USERPROFILE = $originalProfile; $env:LOCALAPPDATA = $originalLocal
    $env:PATH = $originalPath
    # Keep fixtures for inspection when the real plugin installer was selected.
    if (!$InstallPlugins -and !$FixturePath) {
        $resolved = [IO.Path]::GetFullPath($fixture)
        $temp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
        if (!$resolved.StartsWith($temp, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notlike 'personal-devenv-native-*') { throw 'Unsafe native-test cleanup path' }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
