#!/bin/bash

configure_login_shell() {
    local zsh user current marker
    zsh=$(command -v zsh) || return 1
    user=$(id -un)
    current=$(getent passwd "$user") || return 1
    if [ "${current##*:}" = "$zsh" ]; then
        echo 'SKIP Zsh login shell (already selected)'
    elif command -v sudo >/dev/null && sudo -n chsh -s "$zsh" "$user"; then
        echo 'Selected Zsh as the login shell'
    else
        marker='# >>> rc_files: auto-exec zsh (chsh unavailable) >>>'
        if ! grep -qF "$marker" "$HOME/.bashrc" 2>/dev/null; then
            cat >> "$HOME/.bashrc" <<'EOF'

# >>> rc_files: auto-exec zsh (chsh unavailable) >>>
[ ! -f "$HOME/.config/zsh/interactive.sh" ] || . "$HOME/.config/zsh/interactive.sh"
# <<< rc_files: auto-exec zsh (chsh unavailable) <<<
EOF
        fi
        echo 'chsh unavailable; interactive Bash will start Zsh through ~/.bashrc.'
    fi
}
