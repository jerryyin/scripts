#!/bin/bash
# gh.sh - Install GitHub CLI and check authentication
#
# Modes:
#   gh.sh                # Full setup: install gh CLI + check existing auth
#
# Source of truth for GitHub CLI auth is ~/.config/gh, managed by credentials.sh.
#
# Prerequisites:
#   - apt + curl + sudo  -- for the full-setup path that installs gh

set -e

GH_HOST="${GH_HOST:-github.com}"

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

check_gh_auth() {
    if gh auth status -h "$GH_HOST" >/dev/null 2>&1; then
        echo "✓ gh auth configured for $GH_HOST"
        return 0
    fi

    echo "⚠️  gh is installed but not authenticated for $GH_HOST"
    echo "   Run: gh auth login -h $GH_HOST -p https -s repo,read:org,gist"
    echo "   Then persist it with: credentials.sh --save"
}

main() {
    case "${1:-}" in
        "") ;;
        *)
            echo "Usage: gh.sh"
            echo "  Install gh CLI + check existing auth"
            exit 1
            ;;
    esac

    echo ""
    echo "🐙 GitHub CLI Setup"
    echo "──────────────────"
    install_gh_cli
    check_gh_auth
    echo ""
}

main "$@"
