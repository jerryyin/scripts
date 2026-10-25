# Small helpers for native credentials. No general configuration backups here.
Set-StrictMode -Version 2
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Devenv.psm1') -DisableNameChecking

function Assert-PrivatePath([string]$Path, [switch]$AllowGit) {
    $full = [IO.Path]::GetFullPath($Path)
    if ($full.StartsWith('\\')) { throw 'Store native credentials on a local Windows drive.' }
    $current = $full
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            if ((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'Credential destinations must not pass through a symlink or junction.'
            }
            if (!$AllowGit -and (Test-Path -LiteralPath (Join-Path $current '.git'))) {
                throw 'Credential destinations must be outside Git checkouts.'
            }
        }
        $current = Split-Path $current -Parent
    }
}

function Set-PrivateAcl([string]$Path) {
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $system = New-Object Security.Principal.SecurityIdentifier('S-1-5-18')
    $directory = (Get-Item -LiteralPath $Path -Force).PSIsContainer
    $acl = if ($directory) { New-Object Security.AccessControl.DirectorySecurity } else { New-Object Security.AccessControl.FileSecurity }
    # Writing the owner needs WRITE_OWNER even when unchanged; an owner can always replace the DACL.
    if ((Get-Acl -LiteralPath $Path).GetOwner([Security.Principal.SecurityIdentifier]) -ne $sid) { $acl.SetOwner($sid) }
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($identity in @($sid, $system)) {
        $rule = if ($directory) {
            New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
        } else { New-Object Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'Allow') }
        $acl.AddAccessRule($rule)
    }
    # Set-Acl on Windows PowerShell can request SACL privileges unnecessarily
    # when reapplying an existing protected ACL. Set the DACL/owner directly.
    $item = Get-Item -LiteralPath $Path -Force
    if ($PSVersionTable.PSEdition -eq 'Core') { [IO.FileSystemAclExtensions]::SetAccessControl($item, $acl) }
    else { $item.SetAccessControl($acl) }
}

function Save-PrivateText([string]$Path, [string]$Text) {
    Assert-PrivatePath $Path
    $parent = Split-Path $Path -Parent
    [void][IO.Directory]::CreateDirectory($parent)
    Set-PrivateAcl $parent
    if (Test-Path -LiteralPath $Path) {
        Set-PrivateAcl $Path
        if ([IO.File]::ReadAllText($Path) -ceq $Text) { return }
    }
    $temp = Join-Path $parent ('.credential-' + [guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllText($temp, '')
        Set-PrivateAcl $temp
        [IO.File]::WriteAllText($temp, $Text, (New-Object Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $temp -Destination $Path -Force
    } finally { if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force } }
}

# Pass passwords over stdin, keeping them out of process arguments and output.
function Invoke-CredentialCommand([string]$Program, [string[]]$Arguments, [string]$InputText, [hashtable]$Environment = @{}) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = (Get-Command $Program -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    $quoted = foreach ($arg in $Arguments) {
        '"' + [regex]::Replace([regex]::Replace($arg, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1') + '"'
    }
    $info.Arguments = $quoted -join ' '
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardInput = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($key in $Environment.Keys) { $info.EnvironmentVariables[$key] = $Environment[$key] }
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    try {
        [void]$process.Start()
        $outTask = $process.StandardOutput.ReadToEndAsync()
        $errTask = $process.StandardError.ReadToEndAsync()
        if ($InputText) { $process.StandardInput.WriteLine($InputText) }
        $process.StandardInput.Close()
        $process.WaitForExit()
        # Deliberately discard stderr: helpers may include secrets in errors.
        [void]$errTask.GetAwaiter().GetResult()
        return @{ ExitCode = $process.ExitCode; Output = $outTask.GetAwaiter().GetResult() }
    } finally { $process.Dispose() }
}

Export-ModuleMember -Function Assert-PrivatePath, Set-PrivateAcl, Save-PrivateText, Invoke-CredentialCommand
