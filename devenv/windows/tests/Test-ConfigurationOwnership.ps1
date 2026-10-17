param([string]$RcRoot = (Join-Path $env:USERPROFILE 'rc_files'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Import-Module (Join-Path $root 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
$psmux = (Get-Command psmux.exe).Source
$fixture = Join-Path $env:TEMP ('config-owner-' + [guid]::NewGuid().ToString('N').Substring(0,8))
$userRoot = Join-Path $fixture "Someone's Windows home"
$candidate = Join-Path $fixture "Someone's rc_files"
$local = Join-Path $userRoot 'AppData/Local'
$state = Join-Path $local 'PersonalDevenv'
$socket = 'config-owner-' + [guid]::NewGuid().ToString('N')
$prior = @{USERPROFILE=$env:USERPROFILE; HOME=$env:HOME; LOCALAPPDATA=$env:LOCALAPPDATA}
$script:passed = 0
function Assert($Ok, [string]$Message) {
    if (!$Ok) { throw "FAIL: $Message; fixture $fixture" }
    $script:passed++; Write-Output "PASS: $Message"
}
try {
    Assert ((Read-Json (Join-Path $root 'defaults/environment.json')).packages.id -notcontains 'JanDeDobbeleer.OhMyPosh') 'Fresh setup does not require the unused prompt engine'
    foreach ($file in @('.powershell/profile.ps1', 'tmux/.psmux.conf', 'tmux/.tmux/mux.ps1', 'tmux/.tmux.conf')) {
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
    Set-NativeMux $userRoot $candidate $state ''
    $entry = Join-Path $userRoot '.psmux.conf'
    Assert (([IO.File]::ReadAllLines($entry)).Count -eq 2) 'Home psmux entry contains only a comment and source loader'
    Assert (!(Test-Path -LiteralPath (Join-Path $state 'profile.ps1'))) 'No copied PowerShell profile in AppData'
    . $profile
    Assert ($global:ExistingProfilePreference -eq 123) 'Existing PowerShell profile content is preserved'
    Assert ($null -eq $script:DevenvState) 'Opening a shell does not read machine state'
    Save-Json (Join-Path $state 'state.json') ([ordered]@{vimExe='unused'; psmuxExe=$psmux; rcRoot=$candidate}) ''
    Assert ((Get-DevenvState).psmuxExe -eq $psmux) 'Machine paths load on demand'
    Add-Content -LiteralPath $source -Value '$global:TrackedProfileEdit = 456'
    . $profile
    Assert ($global:TrackedProfileEdit -eq 456) 'An rc_files edit takes effect without copying or reinstalling'
    $paths = @($profile, $entry, (Join-Path $state 'psmux.conf'))
    $times = @($paths | ForEach-Object { (Get-Item -LiteralPath $_).LastWriteTimeUtc.Ticks })
    Set-ProfileLoader $profile $source ''
    Set-NativeMux $userRoot $candidate $state ''
    for ($i=0; $i -lt $paths.Count; $i++) {
        Assert ((Get-Item -LiteralPath $paths[$i]).LastWriteTimeUtc.Ticks -eq $times[$i]) ('Rerun leaves timestamp unchanged: ' + (Split-Path $paths[$i] -Leaf))
    }
    Invoke-Native $psmux @('-L', $socket, '-f', $entry, 'new-session', '-d', '-s', 'ownership', '--', 'powershell.exe', '-NoProfile')
    $keys = Invoke-Native $psmux @('-L', $socket, 'list-keys') -Capture
    Assert ($keys.Contains('h select-pane -L') -and $keys.Contains('l select-pane -R')) 'Real psmux loads familiar bindings through the source loader'
    Assert ($keys.Contains('/tmux/.tmux/mux.ps1') -and !$keys.Contains('/devenv/windows/')) 'Runtime bindings use rc_files without importing installation scripts'
    $preferences = Join-Path $candidate 'tmux/.psmux.conf'
    $updated = [IO.File]::ReadAllText($preferences).Replace('bind r source-file ~/.psmux.conf', "bind r source-file ~/.psmux.conf`nbind-key F9 display-message from-rc-files")
    Save-Text $preferences $updated ''
    Set-NativeMux $userRoot $candidate $state ''
    Invoke-Native $psmux @('-L', $socket, 'source-file', $entry)
    $keys = Invoke-Native $psmux @('-L', $socket, 'list-keys') -Capture
    Assert ($keys.Contains('from-rc-files')) 'Changing rc_files preferences is reflected when setup renders and psmux reloads'
    Write-Output "$script:passed configuration ownership checks passed. Fixture: $fixture"
} finally {
    & $psmux -L $socket kill-server 2>$null | Out-Null
    $env:USERPROFILE=$prior.USERPROFILE; $env:HOME=$prior.HOME; $env:LOCALAPPDATA=$prior.LOCALAPPDATA
}
