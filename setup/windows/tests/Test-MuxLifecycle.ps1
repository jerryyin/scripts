param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
Add-Type -Path (Join-Path $PSScriptRoot 'ConPty.cs')
$exe = (Get-Command psmux.exe).Source
$shell = (Get-Command pwsh.exe).Source
$profilePath = Join-Path $RcRoot 'powershell/.windows/@DOCUMENTS/PowerShell/profile.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('mux-lifecycle-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($fixture)
$dataPath = Join-Path $fixture 'data'
$original = @{}
Get-ChildItem Env: | Where-Object { $_.Name -match '^(PSMUX|TMUX)' } | ForEach-Object {
    $original[$_.Name] = $_.Value
    [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process')
}
$console = $null
function Assert($ok, [string]$message) {
    if (!$ok) { throw "FAIL: $message" }
    Write-Output "PASS: $message"
}
function Mux([string[]]$words) {
    $out = & $exe @words 2>&1
    if ($LASTEXITCODE -ne 0) { throw ($out -join [Environment]::NewLine) }
    return (($out -join [Environment]::NewLine).Trim())
}
function Until([scriptblock]$condition) {
    $end = (Get-Date).AddSeconds(15)
    do {
        if (& $condition) { return $true }
        Start-Sleep -Milliseconds 100
    } while ((Get-Date) -lt $end)
    return $false
}
function Open-Mux([string]$invocation) {
    $start = ". '" + $profilePath.Replace("'", "''") + "'; " + $invocation
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($start))
    [DevenvConsole]::new(('"' + $shell + '" -NoProfile -NoExit -EncodedCommand ' + $encoded), $fixture)
}
try {
    $env:PSMUX_DATA_DIR = $dataPath
    $env:PSMUX_NO_WARM = '1'
    $env:PSMUX_CONFIG_FILE = Join-Path $fixture 'test.conf'
    [IO.File]::WriteAllText($env:PSMUX_CONFIG_FILE, ('set -g default-shell "' + $shell.Replace('\', '/') + ' -NoProfile"' + [char]10))
    $console = Open-Mux 'tmux'
    Assert (Until { try { (Mux @('list-sessions', '-F', '#{session_attached}')) -eq '1' } catch { $false } }) 'bare tmux starts an attached session in an empty environment'
    $session = Mux @('display-message', '-p', '#{session_name}')
    $panePid = Mux @('display-message', '-p', '#{pane_pid}')
    Assert ($panePid -match '^\d+$' -and [int]$panePid -gt 0) 'pane has a live shell process'
    $console.Dispose(); $console = $null
    Assert (Until { (Mux @('list-sessions', '-F', '#{session_attached}')) -eq '0' }) 'closing the terminal detaches the session'
    Assert ((Mux @('display-message', '-p', '#{pane_pid}')) -eq $panePid) 'the same shell survives closing the terminal'
    Assert ($session -eq 'main') 'the first session uses the dev session name'
    [void](Mux @('rename-session', '-t', $session, '0'))
    $session = '0'
    foreach ($invocation in @('psmux', 'tmux')) {
        $console = Open-Mux $invocation
        Assert (Until { try { (Mux @('list-sessions', '-F', '#{session_attached}')) -match '1' } catch { $false } }) "bare $invocation attaches"
        Assert ((Mux @('list-sessions', '-F', '#{session_name}')) -eq $session) "bare $invocation reuses the only session"
        Assert ((Mux @('display-message', '-p', '#{pane_pid}')) -eq $panePid) "bare $invocation resumes the same shell"
        $console.Dispose(); $console = $null
        Assert (Until { (Mux @('list-sessions', '-F', '#{session_attached}')) -eq '0' }) "$invocation detaches on terminal close"
    }
    $console = Open-Mux 'tmux new-session -d -s explicit'
    Assert (Until { try { (Mux @('list-sessions', '-F', '#{session_name}')) -match 'explicit' } catch { $false } }) 'explicit tmux arguments still create a separate session'
    $console.Dispose(); $console = $null
} finally {
    if ($console) { $console.Dispose() }
    # Only stop servers whose registry belongs to this fixture.
    if ($env:PSMUX_DATA_DIR -eq $dataPath -and (Test-Path -LiteralPath $dataPath)) {
        & $exe kill-server 2>$null | Out-Null
    }
    Get-ChildItem Env: | Where-Object { $_.Name -match '^(PSMUX|TMUX)' } | ForEach-Object {
        [Environment]::SetEnvironmentVariable($_.Name, $null, 'Process')
    }
    foreach ($name in $original.Keys) { [Environment]::SetEnvironmentVariable($name, $original[$name], 'Process') }
    $resolved = [IO.Path]::GetFullPath($fixture)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$resolved.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $resolved -Leaf) -notmatch '^mux-lifecycle-[a-f0-9]{32}$') { throw 'Unexpected fixture path' }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
