# Native Windows tools. This profile never invokes WSL.
$script:DevenvState = Get-Content -LiteralPath (Join-Path $env:LOCALAPPDATA 'PersonalDevenv/state.json') -Raw | ConvertFrom-Json

function vim { & $script:DevenvState.vimExe @args }
function tmux { & $script:DevenvState.psmuxExe @args }
function dev { & $script:DevenvState.psmuxExe new-session -A -s main @args }
function ll { Get-ChildItem -Force @args }
function la { Get-ChildItem -Force @args }
Set-Alias -Name vi -Value vim
if (Get-Module PSReadLine) {
    Set-PSReadLineOption -EditMode Emacs
    Set-PSReadLineKeyHandler -Key UpArrow -Function HistorySearchBackward
    Set-PSReadLineKeyHandler -Key DownArrow -Function HistorySearchForward
    Set-PSReadLineKeyHandler -Key Alt+LeftArrow -Function BackwardWord
    Set-PSReadLineKeyHandler -Key Alt+RightArrow -Function ForwardWord
}
if (Get-Command oh-my-posh -ErrorAction SilentlyContinue) {
    oh-my-posh init pwsh --config (Join-Path $env:LOCALAPPDATA 'PersonalDevenv/prompt.omp.json') | Invoke-Expression
}
