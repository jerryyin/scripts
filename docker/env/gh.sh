#!/bin/bash
# gh.sh - Install GitHub CLI and configure authentication
#
# Modes:
#   gh.sh                # Full setup: install gh CLI + write authenticated hosts.yml
#   gh.sh --patch-only   # Write hosts.yml only, no install (runtime, every docker run)
#                        # Fast, no network, no dependency install. Silent on no-op.
#
# Source of truth for the GitHub PAT: $HOME/vault/gh_token.txt
# (cloned by priv.sh's sync_vault from a private GitHub repo). At runtime we
# read the plaintext token and write a minimal ~/.config/gh/hosts.yml so gh
# CLI is authenticated without any interactive `gh auth login`.
#
# Prerequisites:
#   - apt + curl + sudo  -- for the full-setup path that installs gh
#   - $HOME/vault/gh_token.txt populated — handled by priv.sh's sync_vault

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="${TOKEN_FILE:-$HOME/vault/gh_token.txt}"
GH_HOSTS="$HOME/.config/gh/hosts.yml"
GH_HOST="${GH_HOST:-github.com}"
GH_USER="${GH_USER:-jerryyin}"
GH_PROTOCOL="${GH_PROTOCOL:-ssh}"

install_gh_cli() {
    if command -v gh &>/dev/null; then
        echo "✓ gh CLI already installed: $(gh --version 2>/dev/null | head -1)"
        return 0
    fi

    echo "📦 Installing GitHub CLI (gh)..."

    if ! command -v curl &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq curl
    fi

    local keyring=/usr/share/keyrings/githubcli-archive-keyring.gpg
    local sources_list=/etc/apt/sources.list.d/github-cli.list

    if [ ! -s "$keyring" ]; then
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
            | sudo dd of="$keyring" status=none
        sudo chmod go+r "$keyring"
    fi

    if [ ! -f "$sources_list" ]; then
        local arch
        arch=$(dpkg --print-architecture)
        echo "deb [arch=${arch} signed-by=${keyring}] https://cli.github.com/packages stable main" \
            | sudo tee "$sources_list" >/dev/null
    fi

    sudo apt-get update -qq
    sudo apt-get install -y -qq gh

    if command -v gh &>/dev/null; then
        echo "✓ GitHub CLI $(gh --version | head -1) installed"
    else
        echo "⚠️  gh not found after install — check apt logs"
        return 1
    fi
}

write_hosts_yml() {
    # Legacy state cleanup: an older credentials.sh used to symlink
    # ~/.config/gh -> /zyin/.config/gh. Since vault is now the source of
    # truth, replace any such symlink with a real directory we own.
    local gh_dir
    gh_dir=$(dirname "$GH_HOSTS")
    if [ -L "$gh_dir" ]; then
        rm "$gh_dir"
    fi

    if [ -f "$GH_HOSTS" ] \
        && grep -q "oauth_token: gh[a-z]_" "$GH_HOSTS" 2>/dev/null; then
        [ "${QUIET_NOOP:-0}" = "1" ] || echo "✓ gh hosts.yml already authenticated"
        return 0
    fi

    if [ ! -f "$TOKEN_FILE" ]; then
        [ "${QUIET_NOOP:-0}" = "1" ] || {
            echo "⚠️  $TOKEN_FILE not found — vault not synced yet"
            echo "   Run priv.sh to clone the vault, then re-run this script."
        }
        return 0
    fi

    local token
    token=$(tr -d '[:space:]' < "$TOKEN_FILE")
    if [ -z "$token" ]; then
        echo "⚠️  $TOKEN_FILE is empty"
        return 0
    fi

    mkdir -p "$(dirname "$GH_HOSTS")"
    chmod 700 "$(dirname "$GH_HOSTS")"
    local tmp="${GH_HOSTS}.tmp.$$"
    (umask 077; cat > "$tmp" <<EOF
${GH_HOST}:
    oauth_token: ${token}
    user: ${GH_USER}
    git_protocol: ${GH_PROTOCOL}
EOF
    )
    mv "$tmp" "$GH_HOSTS"
    chmod 600 "$GH_HOSTS"

    echo "✓ gh authenticated as ${GH_USER} (${GH_HOST})"
}

main() {
    local skip_install=false
    case "${1:-}" in
        --patch-only) skip_install=true ;;
        "") ;;
        *)
            echo "Usage: gh.sh [--patch-only]"
            echo "  (no args)      Install gh CLI + write authenticated hosts.yml (build time)"
            echo "  --patch-only   Write hosts.yml only, skip CLI install (runtime)"
            exit 1
            ;;
    esac

    if [ "$skip_install" = true ]; then
        QUIET_NOOP=1 write_hosts_yml
        return
    fi

    echo ""
    echo "🐙 GitHub CLI Setup"
    echo "──────────────────"
    install_gh_cli
    write_hosts_yml
    echo ""
}

main "$@"
