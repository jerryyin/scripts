param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
Add-Type -Path (Join-Path $PSScriptRoot 'ConPty.cs')
$shell = (Get-Command pwsh.exe -CommandType Application | Select-Object -First 1).Source
$mux = (Get-Command psmux.exe -CommandType Application | Select-Object -First 1).Source
$fixture = Join-Path $env:TEMP ('ssh-session-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($fixture)
$prior = @{}
Get-ChildItem Env: | Where-Object Name -match '^(SSH_|PSMUX|TMUX)' | ForEach-Object {
    $prior[$_.Name] = $_.Value
    [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process')
}
$console = $null
function Assert($ok, [string]$message) {
    if (!$ok) { throw "FAIL: $message" }
    Write-Output "PASS: $message"
}
function Mux([string[]]$words) {
    $output = & $mux @words 2>&1
    if ($LASTEXITCODE -ne 0) { throw ($output -join "`n") }
    ($output -join "`n").Trim()
}
function Until([scriptblock]$condition) {
    $deadline = (Get-Date).AddSeconds(15)
    do {
        if (& $condition) { return $true }
        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $deadline)
    return $false
}
try {
    $env:PSMUX_DATA_DIR = Join-Path $fixture 'data'
    $env:PSMUX_NO_WARM = '1'
    $env:PSMUX_CONFIG_FILE = Join-Path $RcRoot 'tmux/.windows/.psmux.conf'
    $env:SSH_CONNECTION = '127.0.0.1 12345 127.0.0.1 22'
    $env:SSH_TTY = 'conpty-test'
    $console = [DevenvConsole]::new(('"' + $shell + '" -NoLogo'), $fixture)
    Assert (Until { try { (Mux @('list-sessions', '-F', '#{session_name}:#{session_attached}')) -eq 'main:1' } catch { $false } }) 'interactive SSH shell attaches to main automatically'
    $panePid = Mux @('display-message', '-p', '#{pane_pid}')
    $marker = Join-Path $fixture 'inner-shell.txt'
    $console.Send(('Set-Content -LiteralPath ''{0}'' -Value $PID' -f $marker.Replace("'", "''")) + [char]13)
    Assert (Until { (Test-Path -LiteralPath $marker) -and ((Get-Content -LiteralPath $marker -Raw).Trim() -eq $panePid) }) 'shell inside psmux executes commands without recursive attachment'
    $console.Dispose(); $console = $null
    Assert (Until { (Mux @('list-sessions', '-F', '#{session_attached}')) -eq '0' }) 'closing SSH terminal detaches main'
    $console = [DevenvConsole]::new(('"' + $shell + '" -NoLogo'), $fixture)
    Assert (Until { (Mux @('list-sessions', '-F', '#{session_attached}')) -eq '1' }) 'new SSH terminal reattaches automatically'
    Assert ((Mux @('display-message', '-p', '#{pane_pid}')) -eq $panePid) 'reconnect preserves the same shell process'
    $console.Dispose(); $console = $null
    Assert (Until { (Mux @('list-sessions', '-F', '#{session_attached}')) -eq '0' }) 'test terminal detached'
    $console = [DevenvConsole]::new(('"' + $shell + '" -NoLogo -Command "Write-Output SSH_COMMAND_OK"'), $fixture)
    Assert (Until { $console.Output -match 'SSH_COMMAND_OK' }) 'SSH command with a terminal does not attach'
    Assert ((Mux @('list-sessions', '-F', '#{session_attached}')) -eq '0') 'command mode leaves main detached'
    $console.Dispose(); $console = $null
    $output = 'Write-Output REDIRECTED_OK; exit' | & $shell -NoLogo 2>&1
    Assert (($output -join "`n") -match 'REDIRECTED_OK') 'redirected shell does not attach'
} finally {
    if ($console) { $console.Dispose() }
    if ($env:PSMUX_DATA_DIR -eq (Join-Path $fixture 'data')) { & $mux kill-server 2>$null | Out-Null }
    Get-ChildItem Env: | Where-Object Name -match '^(SSH_|PSMUX|TMUX)' | ForEach-Object {
        [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process')
    }
    foreach ($name in $prior.Keys) { [Environment]::SetEnvironmentVariable($name, $prior[$name], 'Process') }
    $resolved = [IO.Path]::GetFullPath($fixture)
    $tempRoot = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or
        (Split-Path $resolved -Leaf) -notmatch '^ssh-session-[a-f0-9]{32}$') { throw 'Unexpected fixture path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
