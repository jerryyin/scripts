param([string]$ConfigurationPath, [switch]$Clipboard, [switch]$CopyOnly)
$ErrorActionPreference='Stop'
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
Add-Type -Path (Join-Path $PSScriptRoot 'ConPty.cs')
$exe=(Get-Command psmux.exe).Source
$shell=(Get-Command pwsh.exe).Source
$rc=Join-Path $env:USERPROFILE 'rc_files/tmux/.tmux.conf'
$original=@{HOME=$env:HOME; USERPROFILE=$env:USERPROFILE; LOCALAPPDATA=$env:LOCALAPPDATA; APPDATA=$env:APPDATA}
$fixture=Join-Path ([IO.Path]::GetTempPath()) ('mux-keys-'+[guid]::NewGuid().ToString('N').Substring(0,8))
$socket='mux-keys-'+[guid]::NewGuid().ToString('N')
[void][IO.Directory]::CreateDirectory((Join-Path $fixture '.psmux/plugins'))
foreach($name in @('ppm','psmux-resurrect')) { Copy-Item -LiteralPath (Join-Path $env:USERPROFILE ('.psmux/plugins/'+$name)) -Destination (Join-Path $fixture '.psmux/plugins') -Recurse }
$configuration=if($ConfigurationPath){[IO.File]::ReadAllText($ConfigurationPath)}else{Get-PsmuxConfiguration $rc}
$start='Set-Location -LiteralPath '''+$fixture.Replace("'","''")+'''; Set-PSReadLineOption -EditMode Emacs; function prompt { [Console]::Write(([string][char]27)+"]9;9;"+[char]34+$PWD.Path+[char]34+[char]7); "fixture '+[char]0x279c+' " }'
$encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($start))
$configuration += [char]10+'set -g default-shell "'+$shell.Replace('\','/')+' -NoProfile"'+[char]10
Save-Text (Join-Path $fixture '.psmux.conf') $configuration ''
$script:passed=0
$script:differences=New-Object Collections.ArrayList
function Difference($ok,[string]$message) { if(!$ok){ [void]$script:differences.Add($message); Write-Output "DIFFERENCE: $message" } }
function Assert($ok,[string]$message) { if(!$ok){throw "FAIL: $message; inspect $fixture"}; $script:passed++; Write-Output "PASS: $message" }
function Mux([string[]]$words) {
    $out=& $exe -L $socket @words 2>&1
    if($LASTEXITCODE -ne 0){throw ($out -join [Environment]::NewLine)}
    return (($out -join [Environment]::NewLine).Trim())
}
function Field([string]$name){ Mux @('display-message','-p',('#{'+$name+'}')) }
function Key([string]$key){
    Start-Sleep -Milliseconds 550
    $console.Key([char]96,$false)
    if ([int][char]$key -lt 32) { $console.Key([char](64+[int][char]$key),$true) }
    else { $console.Key([char]$key,$false) }
    Start-Sleep -Milliseconds 200
}
function Until([scriptblock]$condition,[int]$seconds=10){ $end=(Get-Date).AddSeconds($seconds); do { if(& $condition){return $true}; Start-Sleep -Milliseconds 100 }while((Get-Date)-lt $end); return $false }
$console=$null
$priorClipboard=if($Clipboard){Get-Clipboard -Raw}else{$null}
try {
    $env:HOME=$fixture; $env:USERPROFILE=$fixture; $env:LOCALAPPDATA=Join-Path $fixture 'AppData/Local'; $env:APPDATA=Join-Path $fixture 'AppData/Roaming'
    Save-Json (Join-Path $env:LOCALAPPDATA 'PersonalDevenv/state.json') ([ordered]@{psmuxExe=$exe}) ''
    $command='"'+$exe+'" -L '+$socket+' -f "'+(Join-Path $fixture '.psmux.conf')+'" new-session -s habits -c "'+$fixture+'" -- "'+$shell+'" -NoProfile -NoExit -EncodedCommand '+$encoded
    $console=[DevenvConsole]::new($command,$fixture)
    Assert (Until { try { (Mux @('list-sessions')).Contains('habits:') } catch { $false } } 20) 'isolated native session starts'
    Start-Sleep -Seconds 1
    Assert ((Field 'pane_current_path').Replace('\','/').TrimEnd('/') -eq $fixture.Replace('\','/')) 'initial pane reports the fixture working directory'
    $left=Field 'pane_id'
    if(!$CopyOnly){
    [void](Mux @('split-window','-h','--',$shell,'-NoProfile'))
    $right=Field 'pane_id'
    Key 'h'; Assert ((Field 'pane_id') -eq $left) 'backtick h moves left'
    Key 'l'; Assert ((Field 'pane_id') -eq $right) 'backtick l moves right'
    Key ([string][char]8); Assert ((Field 'pane_id') -eq $left) 'backtick Ctrl-h moves left'
    Key ([string][char]12); Assert ((Field 'pane_id') -eq $right) 'backtick Ctrl-l moves right'
    Key 'h'
    [void](Mux @('split-window','-v','--',$shell,'-NoProfile'))
    $bottom=Field 'pane_id'
    Key 'k'; Assert ((Field 'pane_id') -eq $left) 'backtick k moves up'
    Key 'j'; Assert ((Field 'pane_id') -eq $bottom) 'backtick j moves down'
    Key ([string][char]11); Assert ((Field 'pane_id') -eq $left) 'backtick Ctrl-k moves up'
    Key ([string][char]10); Assert ((Field 'pane_id') -eq $bottom) 'backtick Ctrl-j moves down'
    $width=[int](Field 'pane_width'); Key 'L'; $actualWidth=[int](Field 'pane_width'); Assert ($actualWidth -gt $width) 'backtick Shift-L grows the pane'; Difference ($actualWidth -eq $width+5) ("five-column resize: before=$width after=$actualWidth")
    Key 'H'; Assert ([int](Field 'pane_width') -eq $width) 'backtick Shift-H reverses resize'
    $height=[int](Field 'pane_height'); Key 'K'; $actualHeight=[int](Field 'pane_height'); Assert ($actualHeight -gt $height) 'backtick Shift-K grows the pane'; Difference ($actualHeight -eq $height+5) ("five-row resize: before=$height after=$actualHeight")
    Key 'J'; Assert ([int](Field 'pane_height') -eq $height) 'backtick Shift-J reverses resize'
    foreach($split in @('v','S','s','|','\','-','_','%','"')){
        Key 'c'; $window=Field 'window_index'
        Key $split
        Assert (@((Mux @('list-panes','-t',('habits:'+$window))) -split [Environment]::NewLine).Count -eq 2) ('backtick '+$split+' splits panes')
        $path=(Field 'pane_current_path').TrimEnd('\','/').Replace('\','/')
        Difference ($path -eq $fixture.TrimEnd('\','/').Replace('\','/')) ('working directory after '+$split+': expected='+$fixture+' actual='+$path)
    }
    $window=[int](Field 'window_index'); $id=Field 'window_id'
    Key '<'; Assert ((Field 'window_id') -eq $id -and [int](Field 'window_index') -eq $window-1) 'backtick < swaps windows without changing focus'
    Key '>'; Assert ((Field 'window_id') -eq $id -and [int](Field 'window_index') -eq $window) 'backtick > swaps windows back'
    Key ([string][char]16); Assert ([int](Field 'window_index') -eq $window-1) 'backtick Ctrl-p selects previous window'
    Key ([string][char]14); Assert ([int](Field 'window_index') -eq $window) 'backtick Ctrl-n selects next window'
    $console.Send(([string][char]27)+'[21~'); Start-Sleep -Milliseconds 300
    Assert ((Mux @('show-options','-g','key-table')) -match 'off') 'F10 disables outer bindings'
    $console.Send(([string][char]27)+'[21~'); Start-Sleep -Milliseconds 300
    Assert ((Mux @('show-options','-g','key-table')) -match 'root') 'F10 restores outer bindings'
    Assert ((Mux @('show-options','-g','prefix')).Trim() -eq ('prefix '+[char]96)) 'F10 restores the configured backtick prefix'
    [void](Mux @('select-window','-t','habits:0')); [void](Mux @('select-pane','-t',$left))
    Key '['; Assert ((Field 'pane_in_mode') -eq '1') 'backtick [ enters vi copy mode'
    $console.Send('q'); Start-Sleep -Milliseconds 200
    Assert ((Field 'pane_in_mode') -eq '0') 'q exits copy mode'
    }
    Key 'Y'
    Assert (Until { (Mux @('show-buffer')).Replace('\','/') -eq $fixture.Replace('\','/') } 10) 'backtick Y copies the pane directory into the buffer'
    if($Clipboard){
        Assert (Until { (Get-Clipboard -Raw).TrimEnd([char]13,[char]10).Replace('\','/') -eq $fixture.Replace('\','/') } 10) 'backtick Y copies the pane directory'
    }
    $console.Send("'clipboard audit text'"); Start-Sleep -Milliseconds 300; Key 'y'
    Assert (Until { (Mux @('show-buffer')).TrimEnd([char]13,[char]10) -eq "'clipboard audit text'" } 10) 'backtick y copies the current command into the buffer without its prompt'
    if($Clipboard){
        Assert (Until { (Get-Clipboard -Raw).TrimEnd([char]13,[char]10) -eq "'clipboard audit text'" } 10) 'backtick y copies the current command without its prompt'
    }
    $console.Send([string][char]3)
    Key ([string][char]19)
    Assert (Until { Test-Path (Join-Path $fixture '.psmux/resurrect/last') } 15) 'backtick Ctrl-s saves the isolated environment'
    Save-Json (Join-Path $fixture 'verification.json') ([ordered]@{functionalPassed=$true; fullParity=($script:differences.Count -eq 0); assertions=$script:passed; parityDifferences=@($script:differences)}) ''
    Write-Output "$script:passed native shortcut assertions passed. Artifacts: $fixture"
} catch {
    if($console){ [IO.File]::WriteAllText((Join-Path $fixture 'console.log'),$console.Output) }
    (Mux @('list-keys','-T','prefix')) | Set-Content (Join-Path $fixture 'keys.txt')
    (Mux @('show-options','-g')) | Set-Content (Join-Path $fixture 'options.txt')
    (Mux @('capture-pane','-p')) | Set-Content (Join-Path $fixture 'pane.txt')
    (Mux @('list-panes','-F','#{pane_id} #{pane_width} #{pane_height} #{pane_active}')) | Set-Content (Join-Path $fixture 'geometry.txt')
    throw
} finally {
    $ErrorActionPreference='Continue'
    & $exe -L $socket kill-server 2>$null | Out-Null
    if($console){$console.Dispose()}
    foreach($name in $original.Keys){[Environment]::SetEnvironmentVariable($name,$original[$name],'Process')}
    if($Clipboard){Set-Clipboard -Value $priorClipboard}
}
