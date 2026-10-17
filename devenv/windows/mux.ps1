[CmdletBinding()]
param([Parameter(Mandatory=$true)][ValidateSet('CopyLine','CopyDirectory')][string]$Action)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'Devenv.psm1') -Force -DisableNameChecking
Update-ProcessPath
if ($Action -eq 'CopyDirectory') {
    $path = Invoke-Native psmux.exe @('display-message','-p','#{pane_current_path}') -Capture
    Invoke-Native psmux.exe @('set-buffer',$path)
    Set-Clipboard -Value $path
    return
}
Invoke-Native psmux.exe @('send-keys','C-a')
Start-Sleep -Milliseconds 100
Invoke-Native psmux.exe @('copy-mode')
Invoke-Native psmux.exe @('send-keys','-X','begin-selection')
Invoke-Native psmux.exe @('send-keys','-X','-N','150','cursor-down')
Invoke-Native psmux.exe @('send-keys','-X','end-of-line')
Invoke-Native psmux.exe @('send-keys','-X','previous-word')
Invoke-Native psmux.exe @('send-keys','-X','next-word-end')
Invoke-Native psmux.exe @('send-keys','-X','copy-selection-and-cancel')
Invoke-Native psmux.exe @('send-keys','C-e')
