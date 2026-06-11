#!/bin/bash
# codex.sh - Install OpenAI Codex CLI
#
# Install-only: credentials are managed by credentials.sh (symlinks ~/.codex
# from persistent storage). Unlike Claude/gh whose secrets live in vault,
# Codex uses OAuth with single-use refresh tokens — a static vault copy would
# go stale after one token refresh.
#
# Prerequisites:
#   - nodejs + npm installed (handled by min.sh)

set -e

install_codex_cli() {
    mkdir -p "$HOME/.local"
    npm config set prefix "$HOME/.local" 2>/dev/null || true

    local latest
    latest=$(npm view @openai/codex version 2>/dev/null || echo "")
    if [ -z "$latest" ]; then
        echo "⚠️  Could not fetch latest version from npm — check network"
        return 1
    fi

    if command -v codex &>/dev/null; then
        local current
        current=$(codex --version 2>/dev/null | grep -oP '[\d.]+' | head -1 || echo "")
        if [ "$current" = "$latest" ]; then
            echo "✓ Codex CLI already at latest ($current)"
            return 0
        fi
        echo "📦 Upgrading Codex CLI $current → $latest..."
    else
        echo "📦 Installing Codex CLI v${latest}..."
    fi

    npm install -g "@openai/codex@latest" 2>&1 | tail -1

    if [ -x "$HOME/.local/bin/codex" ]; then
        echo "✓ Codex CLI $(codex --version 2>/dev/null) installed at ~/.local/bin/codex"
        return 0
    fi
    echo "⚠️  Codex CLI not found after install — check npm prefix"
    return 1
}

echo ""
echo "🤖 Codex CLI Setup"
echo "──────────────────"
install_codex_cli
echo ""
