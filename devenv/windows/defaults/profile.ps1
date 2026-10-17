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
function prompt {
    $esc = [string][char]27
    $location = Get-Location
    $folder = if ($location.Path -eq $HOME) { '~' } else { Split-Path $location.Path -Leaf }
    $branch = ''
    if ($location.Provider.Name -eq 'FileSystem') {
        # Report PowerShell's directory to Terminal and psmux.
        [Console]::Write($esc + ']9;9;"' + $location.Path + '"' + [char]7)
        $directory = $location.ProviderPath
        while ($directory -and !(Test-Path -LiteralPath (Join-Path $directory '.git'))) {
            $directory = Split-Path $directory -Parent
        }
        if ($directory) {
            $savedExitCode = $global:LASTEXITCODE
            $savedPreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try { $status = @(& git --no-optional-locks status --porcelain=v1 --branch 2>$null) }
            finally { $ErrorActionPreference = $savedPreference }
            if ($LASTEXITCODE -eq 0 -and $status.Count) {
                $branch = ' ' + (($status[0] -replace '^## ', '') -replace '\.\.\..*$', '')
                if ($status.Count -gt 1) { $branch += ' *' }
            }
            $global:LASTEXITCODE = $savedExitCode
        }
    }
    $esc + '[38;2;125;174;163m' + $folder + $esc + '[38;2;216;166;87m' + $branch + $esc + '[38;2;169;182;101m ' + [char]0x279c + $esc + '[0m '
}

# Optional richer prompt; the native prompt avoids its startup cost.
if ($env:DEVENV_PROMPT -eq 'oh-my-posh' -and (Get-Command oh-my-posh -ErrorAction SilentlyContinue)) {
    oh-my-posh init pwsh --config (Join-Path $env:LOCALAPPDATA 'PersonalDevenv/prompt.omp.json') | Invoke-Expression
}
