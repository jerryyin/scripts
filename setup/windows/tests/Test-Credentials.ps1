param([switch]$HelpersOnly)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
Import-Module (Join-Path $root 'Credentials.psm1') -Force -DisableNameChecking
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('personal-credentials-test-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($fixture)
$script:passed = 0
function Assert($Condition, [string]$Message) {
    if (!$Condition) { throw "FAIL: $Message" }
    $script:passed++
    Write-Output "PASS: $Message"
}
function Assert-Throws([scriptblock]$Action, [string]$Message) {
    $caught = $false
    try { & $Action | Out-Null } catch { $caught = $true }
    Assert $caught $Message
}
try {
    # Fake values only. No access to real vault, SSH keys or app credentials.
    $path = Join-Path $fixture 'native/config.json'
    Save-PrivateText $path 'test credential'
    Assert ([IO.File]::ReadAllText($path) -ceq 'test credential') 'Protected credential writer creates the file'
    $acl = Get-Acl -LiteralPath $path
    $sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $identities = @($acl.Access | ForEach-Object { $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value })
    Assert ($acl.AreAccessRulesProtected -and $identities.Count -eq 2 -and $identities -contains $sid -and $identities -contains 'S-1-5-18') 'Only current user and SYSTEM have credential access'
    $stamp = (Get-Item -LiteralPath $path).LastWriteTimeUtc
    Save-PrivateText $path 'test credential'
    Assert ((Get-Item -LiteralPath $path).LastWriteTimeUtc -eq $stamp) 'Unchanged credentials are not rewritten'
    Save-PrivateText $path 'rotated test credential'
    Assert ([IO.File]::ReadAllText($path) -ceq 'rotated test credential') 'Credential rotation replaces content'
    Assert (@(Get-ChildItem -LiteralPath (Split-Path $path -Parent) -Force).Count -eq 1) 'Credential writer leaves no backup or staging copy'

    $repo = Join-Path $fixture 'checkout'
    [void][IO.Directory]::CreateDirectory((Join-Path $repo '.git'))
    Assert-Throws { Save-PrivateText (Join-Path $repo 'config.json') 'test' } 'Credential writes inside Git checkouts are refused'
    $link = Join-Path $fixture 'junction'
    New-Item -ItemType Junction -Path $link -Target (Split-Path $path -Parent) | Out-Null
    Assert-Throws { Save-PrivateText (Join-Path $link 'other.json') 'test' } 'Credential writes through junctions are refused'
    # Remove the junction itself; recursive cleanup must never follow its target.
    [IO.Directory]::Delete($link)

    $node = (Get-Command node.exe -ErrorAction Stop).Source
    $arguments = @('-e', 'let s="";process.stdin.on("data",d=>s+=d);process.stdin.on("end",()=>{process.stderr.write("ignored diagnostic");console.log(JSON.stringify({args:process.argv.slice(1),stdin:s,env:process.env.CREDENTIAL_TEST}));});', '--', 'two words', 'quote"inside', 'C:\ending\', '')
    $run = Invoke-CredentialCommand $node $arguments 'test stdin password' @{ CREDENTIAL_TEST = 'child only' }
    $echoed = ConvertFrom-Json $run.Output
    Assert ($run.ExitCode -eq 0 -and $echoed.args.Count -eq 4 -and $echoed.args[0] -ceq 'two words' -and $echoed.args[1] -ceq 'quote"inside' -and $echoed.args[2] -ceq 'C:\ending\' -and $echoed.args[3] -ceq '') 'Native argument quoting preserves spaces, quotes, backslashes and empty arguments'
    Assert ($echoed.stdin.TrimEnd() -ceq 'test stdin password' -and $echoed.env -ceq 'child only' -and !$env:CREDENTIAL_TEST) 'Secret stdin and child environment do not change the parent environment'
    $run = Invoke-CredentialCommand $node @('-e', 'process.stderr.write("test secret must not escape");process.exit(7)') ''
    Assert ($run.ExitCode -eq 7 -and !$run.Output) 'Native errors preserve exit status without printing stderr'

    $vault = Join-Path $fixture 'vault'
    [void][IO.Directory]::CreateDirectory($vault)
    $auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('fixture-user:fixture:password'))
    $secret = Join-Path $vault 'docker_mkmhub_auth.txt'
    [IO.File]::WriteAllText($secret, $auth)
    $dockerConfig = Join-Path $fixture 'docker/config.json'
    Save-Json $dockerConfig ([ordered]@{ auths = [ordered]@{ 'other.example' = [ordered]@{ auth = 'keep' }; 'mkmhub.amd.com' = [ordered]@{ custom = 'keep' } }; currentContext = 'keep' }) ''
    $output = Set-DockerCredential $vault $dockerConfig
    $config = Read-Json $dockerConfig
    Assert ($config.auths['mkmhub.amd.com'].auth -ceq $auth -and $config.auths['mkmhub.amd.com'].custom -ceq 'keep') 'Docker restores only its managed registry credential'
    Assert ($config.auths['other.example'].auth -ceq 'keep' -and $config.currentContext -ceq 'keep') 'Docker preserves other registries and settings'
    Assert (!(($output -join '').Contains($auth))) 'Docker output does not include the credential'
    $stamp = (Get-Item -LiteralPath $dockerConfig).LastWriteTimeUtc
    Set-DockerCredential $vault $dockerConfig | Out-Null
    Assert ((Get-Item -LiteralPath $dockerConfig).LastWriteTimeUtc -eq $stamp) 'Docker reruns are unchanged'
    [IO.File]::WriteAllText($secret, 'invalid base64')
    Assert-Throws { Set-DockerCredential $vault $dockerConfig } 'Malformed Docker vault input is refused'
    [IO.File]::WriteAllText($secret, $auth)
    $before = [IO.File]::ReadAllText($dockerConfig)
    [IO.File]::WriteAllText($dockerConfig, '{invalid')
    Assert-Throws { Set-DockerCredential $vault $dockerConfig } 'Malformed existing Docker config is preserved'
    Assert ([IO.File]::ReadAllText($dockerConfig) -ceq '{invalid') 'Docker parse failure leaves the file untouched'
    [IO.File]::WriteAllText($dockerConfig, $before)

    $ssh = Join-Path $fixture 'ssh'
    $source = Join-Path $fixture 'backup'
    [void][IO.Directory]::CreateDirectory($source)
    $fakeKey = "-----BEGIN OPENSSH PRIVATE KEY-----`nfake test data`n-----END OPENSSH PRIVATE KEY-----`n"
    [IO.File]::WriteAllText((Join-Path $source 'id_test'), $fakeKey)
    [IO.File]::WriteAllText((Join-Path $source 'id_test.pub'), 'fake public key')
    [IO.File]::WriteAllText((Join-Path $source 'config'), 'Linux-only configuration')
    $keys = @(Initialize-NativeSsh $ssh $source)
    Assert ($keys.Count -eq 1 -and [IO.File]::ReadAllText((Join-Path $ssh 'id_test')) -ceq $fakeKey) 'SSH imports a missing key from an explicit source'
    Assert ((Test-Path -LiteralPath (Join-Path $ssh 'id_test.pub')) -and !(Test-Path -LiteralPath (Join-Path $ssh 'config'))) 'SSH imports the public key without copying Linux host configuration'
    Save-PrivateText (Join-Path $ssh 'config') 'existing Windows hosts'
    [IO.File]::WriteAllText((Join-Path $source 'id_test'), $fakeKey.Replace('fake test data', 'different fake data'))
    Initialize-NativeSsh $ssh $source | Out-Null
    Assert ([IO.File]::ReadAllText((Join-Path $ssh 'id_test')) -ceq $fakeKey -and [IO.File]::ReadAllText((Join-Path $ssh 'config')) -ceq 'existing Windows hosts') 'SSH reruns preserve existing keys and host configuration'

    # Full Apply orchestration: real local Git vault, fake gh/codex executables,
    # and a separate Windows home. No browser, real key or service is touched.
    if (!$HelpersOnly) {
    $bin = Join-Path $fixture 'bin'
    [void][IO.Directory]::CreateDirectory($bin)
    Add-Type -TypeDefinition 'public class CredentialFixture { public static int Main(string[] args) { System.Console.WriteLine("fixture authenticated"); return 0; } }' -OutputType ConsoleApplication -OutputAssembly (Join-Path $bin 'gh.exe')
    Copy-Item -LiteralPath (Join-Path $bin 'gh.exe') -Destination (Join-Path $bin 'codex.exe')
    $repoSource = Join-Path $fixture 'source-vault'
    [void][IO.Directory]::CreateDirectory($repoSource)
    [IO.File]::WriteAllText((Join-Path $repoSource 'docker_mkmhub_auth.txt'), $auth)
    Invoke-Native git @('init', $repoSource)
    Invoke-Native git @('-C', $repoSource, 'add', 'docker_mkmhub_auth.txt')
    Invoke-Native git @('-C', $repoSource, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-m', 'fake credential fixture')
    $utility = Join-Path $fixture 'utility'
    [void][IO.Directory]::CreateDirectory((Join-Path $utility 'defaults'))
    foreach ($file in @('priv.ps1', 'Credentials.psm1', 'Devenv.psm1')) { Copy-Item -LiteralPath (Join-Path $root $file) -Destination (Join-Path $utility $file) }
    Save-Json (Join-Path $utility 'defaults/environment.json') ([ordered]@{ vaultRepository = $repoSource }) ''
    $homePath = Join-Path $fixture 'fresh account'
    $targetVault = Join-Path $homePath 'vault'
    $childEnvironment = @{ USERPROFILE = $homePath; DOCKER_CONFIG = (Join-Path $homePath '.docker'); PATH = $bin + ';' + $env:PATH }
    $runner = Join-Path $fixture 'run-fixture.ps1'
    [IO.File]::WriteAllText($runner, @'
param($Script, $Vault)
try { & $Script -Apply -SkipSshAgent -VaultPath $Vault }
catch { Write-Output ('Fixture error: ' + $_.Exception.Message); exit 1 }
'@)
    $applyArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runner, '-Script', (Join-Path $utility 'priv.ps1'), '-Vault', $targetVault)
    $applyResult = Invoke-CredentialCommand powershell.exe $applyArguments '' $childEnvironment
    if ($applyResult.ExitCode -ne 0) { throw ('Fake-account Apply failed: ' + $applyResult.Output) }
    Assert ($applyResult.ExitCode -eq 0 -and (Test-Path -LiteralPath (Join-Path $targetVault '.git')) -and (Read-Json (Join-Path $homePath '.docker/config.json')).auths['mkmhub.amd.com'].auth -ceq $auth) 'Full credential phase clones a vault and restores Docker into an empty Windows home'
    Assert (!$applyResult.Output.Contains($auth) -and !$applyResult.Output.Contains('fixture:password')) 'Full Apply does not print vault credentials'
    $applyResult = Invoke-CredentialCommand powershell.exe $applyArguments '' $childEnvironment
    Assert ($applyResult.ExitCode -eq 0 -and $applyResult.Output.Contains('OK Codex')) 'Full credential phase reuses an existing vault and authenticated native logins'
    }

    # Mock only the Docker helper invocation; no Docker daemon or login needed.
    $module = Get-Module Credentials
    & $module {
        function script:Get-Command([string]$Name) { if ($Name -eq 'docker.exe') { return @{ Source = 'fixture' } }; Microsoft.PowerShell.Core\Get-Command $Name -ErrorAction SilentlyContinue }
        function script:Invoke-CredentialCommand($Program, $Arguments, $InputText, $Environment) {
            $script:dockerCall = @{ Program = $Program; Arguments = $Arguments; InputText = $InputText; Environment = $Environment }
            return @{ ExitCode = 0; Output = '' }
        }
    }
    Save-Json $dockerConfig ([ordered]@{ credsStore = 'desktop'; auths = [ordered]@{} }) ''
    $before = [IO.File]::ReadAllText($dockerConfig)
    Set-DockerCredential $vault $dockerConfig | Out-Null
    $call = & $module { $script:dockerCall }
    Assert ($call.Arguments -contains '--password-stdin' -and $call.InputText -ceq 'fixture:password' -and $call.Arguments -notcontains $call.InputText) 'Docker helper receives a password via stdin, including embedded colons'
    Assert ($call.Environment.DOCKER_CONFIG -ceq (Split-Path $dockerConfig -Parent) -and [IO.File]::ReadAllText($dockerConfig) -ceq $before) 'Docker keeps an existing credential helper and its configuration'
    & $module {
        $script:dockerLogins = 0
        function script:Get-Command([string]$Name) { return @{ Source='fixture' } }
        function script:Invoke-CredentialCommand($Program, $Arguments, $InputText, $Environment) {
            if ($Arguments[0] -eq 'get') { return @{ ExitCode=0; Output='{"Username":"fixture-user","Secret":"fixture:password"}' } }
            $script:dockerLogins++
            return @{ ExitCode=0; Output='' }
        }
    }
    $output = Set-DockerCredential $vault $dockerConfig
    Assert ((& $module { $script:dockerLogins }) -eq 0 -and $output -like 'SKIP Docker*') 'Docker does not repeat login when the credential helper already matches the vault'

    $bootstrap = [IO.File]::ReadAllText((Join-Path $root 'bootstrap.ps1'))
    Assert ($bootstrap.IndexOf('Ensure-Checkout $RcPath') -lt $bootstrap.IndexOf("'priv.ps1') -Apply") -and $bootstrap.IndexOf("'priv.ps1') -Apply") -lt $bootstrap.IndexOf('Ensure-Checkout $NotesPath')) 'Bootstrap configures credentials before cloning private notes'
    Write-Output "$script:passed credential assertions passed. Real account credentials were not modified."
} finally {
    $full = [IO.Path]::GetFullPath($fixture)
    $temp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (!$full.StartsWith($temp, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path $full -Leaf) -notlike 'personal-credentials-test-*') { throw 'Unsafe test cleanup path.' }
    # Any remaining junction is removed without traversal before recursive cleanup.
    $junction = Join-Path $full 'junction'
    if ((Test-Path -LiteralPath $junction) -and ((Get-Item -LiteralPath $junction -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { [IO.Directory]::Delete($junction) }
    Remove-Item -LiteralPath $full -Recurse -Force
}
