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
    $acl.SetOwner($sid)
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

function Set-DockerCredential([string]$VaultPath, [string]$ConfigPath) {
    $secretPath = Join-Path $VaultPath 'docker_mkmhub_auth.txt'
    if (!(Test-Path -LiteralPath $secretPath)) { Write-Output 'SKIP Docker: docker_mkmhub_auth.txt is absent from vault.'; return }
    Assert-PrivatePath $ConfigPath
    Assert-PrivatePath $secretPath -AllowGit
    Set-PrivateAcl $secretPath
    $auth = [IO.File]::ReadAllText($secretPath).Trim()
    try {
        $decoded = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($auth))
        $parts = $decoded.Split(@([char]':'), 2)
        if ($parts.Count -ne 2 -or !$parts[0] -or !$parts[1] -or $decoded -match '[\r\n]') { throw 'Invalid' }
    } catch { throw 'Docker vault entry must be base64 username:password.' }
    try { $config = Read-Json $ConfigPath } catch { throw 'Cannot parse Docker config.json; the existing file was preserved.' }
    $registry = 'mkmhub.amd.com'
    if (($config.Contains('credsStore') -and $config.credsStore) -or
        ($config.Contains('credHelpers') -and $config.credHelpers.Contains($registry) -and $config.credHelpers[$registry])) {
        if (!(Get-Command docker.exe -ErrorAction SilentlyContinue)) {
            Write-Output 'SKIP Docker: install/start Docker, then rerun priv.ps1 to use its credential helper.'
            return
        }
        $helper = if ($config.Contains('credHelpers') -and $config.credHelpers.Contains($registry)) { $config.credHelpers[$registry] } else { $config.credsStore }
        $helperExe = 'docker-credential-' + $helper + '.exe'
        if (Get-Command $helperExe -ErrorAction SilentlyContinue) {
            $stored = Invoke-CredentialCommand $helperExe @('get') $registry
            try { $credential = ConvertTo-Map (ConvertFrom-Json $stored.Output) } catch { $credential = $null }
            if ($stored.ExitCode -eq 0 -and $credential -is [Collections.IDictionary] -and $credential.Contains('Username') -and $credential.Contains('Secret') -and $credential.Username -ceq $parts[0] -and $credential.Secret -ceq $parts[1]) {
                Write-Output 'SKIP Docker: credential helper already has the vault credential.'
                return
            }
        }
        $result = Invoke-CredentialCommand 'docker.exe' @('login', $registry, '--username', $parts[0], '--password-stdin') $parts[1] @{ DOCKER_CONFIG = (Split-Path $ConfigPath -Parent) }
        if ($result.ExitCode -ne 0) { throw 'Docker login failed; check that its credential helper is available.' }
    } else {
        if (!$config.Contains('auths')) { $config.auths = [ordered]@{} }
        if (!$config.auths.Contains($registry)) { $config.auths[$registry] = [ordered]@{} }
        $config.auths[$registry].auth = $auth
        Save-PrivateText $ConfigPath ((ConvertTo-Json -InputObject $config -Depth 100) + "`n")
    }
    Write-Output 'OK Docker: mkmhub.amd.com credential restored; other registries preserved.'
}

function Initialize-NativeSsh([string]$SshPath, [string]$SourcePath) {
    Assert-PrivatePath $SshPath
    [void][IO.Directory]::CreateDirectory($SshPath)
    Set-PrivateAcl $SshPath
    if ($SourcePath) {
        if (!(Test-Path -LiteralPath $SourcePath -PathType Container)) { throw 'SSH source directory does not exist.' }
        # An explicit source can be a Windows backup or WSL ~/.ssh. Copy only
        # missing keys; Linux host config may contain nonportable commands.
        foreach ($key in @(Get-ChildItem -LiteralPath $SourcePath -File -Filter 'id_*' | Where-Object { $_.Name -notmatch '\.(pub|cert\.pub)$' })) {
            $target = Join-Path $SshPath $key.Name
            if (Test-Path -LiteralPath $target) { continue }
            $text = [IO.File]::ReadAllText($key.FullName)
            if ($text -notmatch '^-----BEGIN (OPENSSH|RSA|EC|DSA|ENCRYPTED) PRIVATE KEY-----') { continue }
            Save-PrivateText $target $text
            if (Test-Path -LiteralPath ($key.FullName + '.pub')) {
                $publicTarget = $target + '.pub'
                if (!(Test-Path -LiteralPath $publicTarget)) { Save-PrivateText $publicTarget ([IO.File]::ReadAllText($key.FullName + '.pub')) }
            }
        }
    }
    foreach ($file in @(Get-ChildItem -LiteralPath $SshPath -File)) {
        Assert-PrivatePath $file.FullName
        Set-PrivateAcl $file.FullName
    }
    return @(Get-ChildItem -LiteralPath $SshPath -File -Filter 'id_*' | Where-Object { $_.Name -notmatch '\.' })
}

Export-ModuleMember -Function Assert-PrivatePath, Set-PrivateAcl, Save-PrivateText, Invoke-CredentialCommand, Set-DockerCredential, Initialize-NativeSsh
