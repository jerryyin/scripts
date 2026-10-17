$ErrorActionPreference = 'Stop'
Import-Module (Join-Path (Split-Path $PSScriptRoot -Parent) 'Devenv.psm1') -Force -DisableNameChecking
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('mux-config-' + [guid]::NewGuid().ToString('N') + '.conf')
try {
    Save-Text $fixture @'
set -g prefix C-a
set-option -g status-style bg=default
set -g status-right "#[fg=colour223,bg=colour239] $wg_is_keys_off %Y-%m-%d | %H:%M #h "
set-option -g default-command "zsh -l"
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-pain-control'
set -g @plugin 'tmux-plugins/tmux-yank'
set -g @plugin 'tmux-plugins/tmux-resurrect'
'@ ''
    $config = Get-PsmuxConfiguration $fixture
    if (!$config.Contains('set-option -g status-style bg=default')) { throw 'FAIL: shared status background was dropped' }
    if (!$config.Contains('#[fg=colour223,bg=colour239]') -or $config.Contains('$wg_is_keys_off')) { throw 'FAIL: shared status colors were dropped or a Linux shell variable leaked' }
    if (!$config.Contains('set -g prefix C-a') -or $config.Contains('zsh -l')) { throw 'FAIL: prefix or native shell selection changed' }
    foreach ($binding in @('Left select-pane -L','Right select-pane -R','Up select-pane -U','Down select-pane -D')) {
        if (!$config.Contains('bind-key -r ' + $binding)) { throw "FAIL: missing repeatable pane navigation: $binding" }
    }
    foreach ($binding in @('h select-pane -L','j select-pane -D','k select-pane -U','l select-pane -R','C-h select-pane -L','C-j select-pane -D','C-k select-pane -U','C-l select-pane -R','-r H resize-pane -L 5','-r J resize-pane -D 5','-r K resize-pane -U 5','-r L resize-pane -R 5','-r "<" swap-window -d -t -1','-r ">" swap-window -d -t +1')) {
        if (!$config.Contains('bind-key ' + $binding)) { throw "FAIL: Linux plugin shortcut missing: $binding" }
    }
    foreach ($key in @('I','U','M-u','C-s','C-r','y','Y')) {
        if ($config -notmatch ('(?m)^bind-key ' + [regex]::Escape($key) + ' ')) { throw "FAIL: missing plugin key $key" }
    }
    Save-Text $fixture 'set -g prefix C-a' ''
    $plain = Get-PsmuxConfiguration $fixture
    if ($plain.Contains('bind-key h ') -or $plain.Contains('psmux-resurrect')) { throw 'FAIL: disabled Linux plugins still change Windows keys' }
    Write-Output 'PASS: shared status colors, prefix, native shell, arrow navigation and declared plugin shortcuts'
} finally { if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture } }
