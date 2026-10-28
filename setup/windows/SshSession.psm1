# SSH shell configuration and recovery of managed links.
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Devenv.psm1') -DisableNameChecking

function Get-SshShellState {
    if (!(Get-Service sshd -ErrorAction SilentlyContinue)) {
        return @{ Installed=$false; NeedsSetup=$false; ShellPath='' }
    }
    # Elevation and terminal hosts can prepend different Store aliases to PATH.
    $path = @((Join-Path $env:ProgramFiles 'PowerShell/7/pwsh.exe'),
              (Join-Path $env:LOCALAPPDATA 'Microsoft/WindowsApps/pwsh.exe')) |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (!$path) {
        $shell = Get-Command pwsh.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        $path = if ($shell) { $shell.Source } else { '' }
    }
    $current = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\OpenSSH' -ErrorAction SilentlyContinue
    $configured = if ($current) { ($current | Select-Object -ExpandProperty DefaultShell -ErrorAction SilentlyContinue) } else { '' }
    return @{ Installed=$true; NeedsSetup=(!$path -or $configured -ne $path); ShellPath=$path }
}

function Set-SshShell {
    $state = Get-SshShellState
    if (!$state.NeedsSetup) { return }
    if (!(Test-IsAdministrator)) { throw 'SSH shell configuration requires the grouped administrator step.' }
    if (!$state.ShellPath) { throw 'PowerShell 7 is missing; rerun bootstrap without -SkipPackages.' }
    $sshKey = 'HKLM:\SOFTWARE\OpenSSH'
    $current = Get-ItemProperty -LiteralPath $sshKey -ErrorAction SilentlyContinue
    $backup = Join-Path $env:LOCALAPPDATA ('PersonalDevenv/backups/ssh-shell-' + [guid]::NewGuid().ToString('N') + '.json')
    Save-Json $backup ($current | Select-Object DefaultShell,DefaultShellCommandOption,DefaultShellEscapeArguments) ''
    if (!(Test-Path -LiteralPath $sshKey)) { New-Item -Path $sshKey -Force | Out-Null }
    New-ItemProperty -LiteralPath $sshKey -Name DefaultShell -PropertyType String -Value $state.ShellPath -Force | Out-Null
    Write-Host "SSH default shell: $($state.ShellPath)"
}

function Get-SshSessionProblems([string]$RcRoot, [string[]]$PlanArguments = @('plan', '--json')) {
    $state = Get-SshShellState
    if (!$state.Installed) { return }
    if ($state.NeedsSetup) { 'SSH default shell is not the installed PowerShell 7; run bootstrap.ps1 -Apply.' }
    try {
        foreach ($link in @(Get-SshLinkRepairs $RcRoot $PlanArguments)) {
            "SSH cannot read $($link.target) (Windows error 448); run bootstrap.ps1 -Apply."
        }
    } catch { "SSH link check failed: $($_.Exception.Message)" }
}

function Get-SshLinkReadErrors([object[]]$Links) {
    if (!$Links.Count) { return }
    # Enforce the SSH traversal restriction only in this read-only child.
    # https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessmitigationpolicy
    $command = @'
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
# Redirected input may include the parent console encoding's byte-order mark.
$reader = New-Object IO.StreamReader([Console]::OpenStandardInput())
$links = ConvertFrom-Json ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($reader.ReadToEnd())))
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class SshTraversal {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetProcessMitigationPolicy(int policy, ref uint flags, UIntPtr size);
}
"@
$enforceRedirectionTrust = [uint32]1
$processRedirectionTrustPolicy = 16
if (![SshTraversal]::SetProcessMitigationPolicy($processRedirectionTrustPolicy, [ref]$enforceRedirectionTrust, (New-Object UIntPtr([uint32]4)))) {
    $code = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
    # Older Windows versions do not implement this mitigation.
    if ($code -ne 87) { throw (New-Object ComponentModel.Win32Exception($code)) }
}
$errors = @()
foreach ($link in $links) {
    try {
        if ($link.directory) { [void][IO.Directory]::GetFileSystemEntries($link.target) }
        else { $handle = [IO.File]::Open($link.target, 'Open', 'Read', 'ReadWrite'); $handle.Dispose() }
    } catch {
        $failure = $_.Exception
        while ($failure.InnerException) { $failure = $failure.InnerException }
        $errors += @{target=$link.target; code=($failure.HResult -band 0xffff); message=$failure.Message}
    }
}
ConvertTo-Json -InputObject $errors -Compress
'@
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
    $info.Arguments = '-NoProfile -NonInteractive -EncodedCommand ' + [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = New-Object Text.UTF8Encoding($false)
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $output = $process.StandardOutput.ReadToEndAsync()
        $errors = $process.StandardError.ReadToEndAsync()
        $json = ConvertTo-Json -InputObject $Links -Depth 4 -Compress
        $process.StandardInput.Write([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json)))
        $process.StandardInput.Close()
        $process.WaitForExit()
        $diagnostic = $errors.GetAwaiter().GetResult()
        if ($process.ExitCode) { throw "SSH link probe failed: $diagnostic" }
        foreach ($failure in (ConvertFrom-Json ($output.GetAwaiter().GetResult()))) { $failure }
    } finally { $process.Dispose() }
}

function Assert-SshLinkReadable($Link, [string]$Path) {
    if ($Link.directory) { [void][IO.Directory]::GetFileSystemEntries($Path) }
    else { $handle = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite'); $handle.Dispose() }
}
function Get-SshLinkRepairs([string]$RcRoot, [string[]]$PlanArguments = @('plan', '--json'), [switch]$ExistingOnly) {
    $priorGit = $env:GIT_CONFIG_GLOBAL
    try {
        # The account's Git configuration may itself be an unreadable link.
        $env:GIT_CONFIG_GLOBAL = 'NUL'
        $plan = Invoke-RcSetup $RcRoot $PlanArguments -Capture | ConvertFrom-Json
    } finally { $env:GIT_CONFIG_GLOBAL = $priorGit }
    $candidates = @($plan.links)
    $winget = Join-Path $env:LOCALAPPDATA 'Microsoft/WinGet'
    foreach ($name in @('psmux.exe', 'tmux.exe', 'pmux.exe', 'rg.exe', 'clangd.exe')) {
        $alias = Get-Item -LiteralPath (Join-Path $winget "Links/$name") -Force -ErrorAction SilentlyContinue
        if ($alias -and $alias.LinkType -eq 'SymbolicLink') {
            $candidates += [pscustomobject]@{source=@($alias.Target)[0]; target=$alias.FullName; directory=$false}
        }
    }
    $rcPrefix = [IO.Path]::GetFullPath($RcRoot).TrimEnd('\') + '\'
    $packagePrefix = [IO.Path]::GetFullPath((Join-Path $winget 'Packages')).TrimEnd('\') + '\'
    $matching = @()
    foreach ($link in $candidates) {
        $source = [IO.Path]::GetFullPath($link.source)
        $target = [IO.Path]::GetFullPath($link.target)
        if (!($source.StartsWith($rcPrefix, [StringComparison]::OrdinalIgnoreCase) -or
              $source.StartsWith($packagePrefix, [StringComparison]::OrdinalIgnoreCase))) { throw "Unmanaged source: $source" }
        $item = Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
        if (!$item) {
            if ($ExistingOnly) { continue }
            throw "Missing configuration: $target; run bootstrap.ps1 -Apply."
        }
        if ($item.LinkType -ne 'SymbolicLink' -or @($item.Target).Count -ne 1 -or
            ![string]::Equals([IO.Path]::GetFullPath(@($item.Target)[0]), $source, [StringComparison]::OrdinalIgnoreCase)) {
            if ($ExistingOnly) { continue }
            throw "Refusing to replace unrelated configuration: $target"
        }
        Assert-SshLinkReadable $link $source
        $matching += $link
    }
    foreach ($error in @(Get-SshLinkReadErrors $matching)) {
        if ($error.code -ne 448) { throw "Cannot read $($error.target): $($error.message)" }
        $matching | Where-Object target -eq $error.target
    }
}

function Repair-SshLinks([string]$RcRoot, [string[]]$PlanArguments = @('plan', '--json'), [switch]$ExistingOnly) {
    if (!(Test-IsAdministrator)) { throw 'Link repair requires the grouped administrator step.' }
    $repairs = @(Get-SshLinkRepairs $RcRoot $PlanArguments -ExistingOnly:$ExistingOnly)
    $backupRoot = Join-Path $env:LOCALAPPDATA ('PersonalDevenv/backups/ssh-links-' + [guid]::NewGuid().ToString('N'))
    Write-Output "$($repairs.Count) managed links blocked by Windows error 448."
    foreach ($link in $repairs) {
        [void][IO.Directory]::CreateDirectory($backupRoot)
        $id = [guid]::NewGuid().ToString('N')
        $temporary = $link.target + '.ssh-link-' + $id
        $saved = Join-Path $backupRoot ((Split-Path $link.target -Leaf) + '.' + $id)
        # Paths and targets were checked above. Only the link moves; its source
        # remains in the checkout or installed package, and the old link is kept.
        New-Item -ItemType SymbolicLink -Path $temporary -Target $link.source | Out-Null
        Assert-SshLinkReadable $link $temporary
        Move-Item -LiteralPath $link.target -Destination $saved
        try { Move-Item -LiteralPath $temporary -Destination $link.target }
        catch { Move-Item -LiteralPath $saved -Destination $link.target; throw }
        Assert-SshLinkReadable $link $link.target
        Write-Output "REPAIRED $($link.target)"
    }
    if ($repairs.Count) { Write-Output "Previous links: $backupRoot" }
}

Export-ModuleMember -Function Get-SshShellState, Set-SshShell, Get-SshSessionProblems, Get-SshLinkRepairs, Repair-SshLinks
