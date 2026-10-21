#Requires -RunAsAdministrator
<#
.SYNOPSIS
Configure PowerShell SSH logins and repair managed links blocked in an elevated SSH session.
Plan by default; -Apply selects PowerShell and recreates matching links that fail with error 448.
Does not change process mitigations, authentication, or the linked source files.
#>
param([switch]$Apply, [string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
$shellPath = (Get-Command pwsh.exe -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
if (!(Get-Service sshd -ErrorAction SilentlyContinue)) { throw 'Install OpenSSH Server before configuring its sessions' }
$priorGit = $env:GIT_CONFIG_GLOBAL
try {
    # The account's Git configuration may itself be an unreadable link.
    $env:GIT_CONFIG_GLOBAL = 'NUL'
    $plan = Invoke-RcSetup $RcRoot @('plan', '--json') -Capture | ConvertFrom-Json
} finally { $env:GIT_CONFIG_GLOBAL = $priorGit }
$candidates = @($plan.links)
$winget = Join-Path $env:LOCALAPPDATA 'Microsoft/WinGet'
foreach ($name in @('psmux.exe', 'tmux.exe', 'pmux.exe', 'rg.exe', 'clangd.exe')) {
    $alias = Get-Item -LiteralPath (Join-Path $winget "Links/$name") -Force -ErrorAction SilentlyContinue
    if ($alias -and $alias.LinkType -eq 'SymbolicLink') {
        $candidates += [pscustomobject]@{source=@($alias.Target)[0]; target=$alias.FullName; directory=$false}
    }
}
$homeRoot = [IO.Path]::GetFullPath($env:USERPROFILE).TrimEnd('\') + '\'
$rcPrefix = [IO.Path]::GetFullPath($RcRoot).TrimEnd('\') + '\'
$packagePrefix = [IO.Path]::GetFullPath((Join-Path $winget 'Packages')).TrimEnd('\') + '\'
$backupRoot = Join-Path $env:LOCALAPPDATA ('PersonalDevenv/backups/ssh-links-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
function Assert-Readable($Link, [string]$Path) {
    if ($Link.directory) { [void][IO.Directory]::GetFileSystemEntries($Path) }
    else { $handle = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite'); $handle.Dispose() }
}
$repairs = @()
foreach ($link in $candidates) {
    $source = [IO.Path]::GetFullPath($link.source)
    $target = [IO.Path]::GetFullPath($link.target)
    if (!$target.StartsWith($homeRoot, [StringComparison]::OrdinalIgnoreCase) -or
        !($source.StartsWith($rcPrefix, [StringComparison]::OrdinalIgnoreCase) -or
          $source.StartsWith($packagePrefix, [StringComparison]::OrdinalIgnoreCase))) { throw "Unmanaged link: $target" }
    $item = Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
    if (!$item) { throw "Missing configuration: $target; run rc_files setup first" }
    if ($item.LinkType -ne 'SymbolicLink' -or @($item.Target).Count -ne 1 -or
        ![string]::Equals([IO.Path]::GetFullPath(@($item.Target)[0]), $source, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to replace unrelated configuration: $target"
    }
    Assert-Readable $link $source
    try { Assert-Readable $link $target }
    catch {
        $exception = $_.Exception
        while ($exception.InnerException) { $exception = $exception.InnerException }
        if (($exception.HResult -band 0xffff) -ne 448) { throw }
        $repairs += $link
    }
}
Write-Output "$($repairs.Count) managed links blocked by Windows error 448."
foreach ($link in $repairs) {
    if (!$Apply) { Write-Output "REPAIR $($link.target)"; continue }
    [void][IO.Directory]::CreateDirectory($backupRoot)
    $id = [guid]::NewGuid().ToString('N')
    $temporary = $link.target + '.ssh-link-' + $id
    $saved = Join-Path $backupRoot ((Split-Path $link.target -Leaf) + '.' + $id)
    # Paths and targets were checked above. Only the link moves; its source
    # remains in the checkout or installed package, and the old link is kept.
    New-Item -ItemType SymbolicLink -Path $temporary -Target $link.source | Out-Null
    Assert-Readable $link $temporary
    Move-Item -LiteralPath $link.target -Destination $saved
    try { Move-Item -LiteralPath $temporary -Destination $link.target }
    catch { Move-Item -LiteralPath $saved -Destination $link.target; throw }
    Assert-Readable $link $link.target
    Write-Output "REPAIRED $($link.target)"
}
if ($Apply -and $repairs.Count) { Write-Output "Previous links: $backupRoot" }
$sshKey = 'HKLM:\SOFTWARE\OpenSSH'
$current = Get-ItemProperty -LiteralPath $sshKey
if ($current.DefaultShell -ne $shellPath) {
    Write-Output "SSH default shell: $shellPath"
    if ($Apply) {
        [void][IO.Directory]::CreateDirectory($backupRoot)
        $current | Select-Object DefaultShell,DefaultShellCommandOption,DefaultShellEscapeArguments |
            ConvertTo-Json | Set-Content -LiteralPath (Join-Path $backupRoot 'previous-shell.json')
        New-ItemProperty -LiteralPath $sshKey -Name DefaultShell -PropertyType String -Value $shellPath -Force | Out-Null
    }
}
