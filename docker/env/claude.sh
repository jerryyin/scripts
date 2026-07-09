#!/bin/bash
# claude.sh - Install Claude Code CLI
#
# Credential patching is handled by vault.sh.
#
# Model selection is left to Claude Code's built-in defaults (latest at
# each release). No ANTHROPIC_MODEL or ANTHROPIC_DEFAULT_*_MODEL env
# vars — those override the built-in and require manual bumping.
#
# Prerequisites:
#   - nodejs + npm installed (handled by min.sh)

set -e

install_claude_cli() {
    # Always install to ~/.local to avoid /usr/local being overridden
    # by Docker tools_volume mount at runtime
    mkdir -p "$HOME/.local"
    npm config set prefix "$HOME/.local" 2>/dev/null || true

    local latest
    latest=$(npm view @anthropic-ai/claude-code version 2>/dev/null || echo "")
    if [ -z "$latest" ]; then
        echo "⚠️  Could not fetch latest version from npm — check network"
        return 1
    fi

    if command -v claude &>/dev/null; then
        local current
        current=$(claude --version 2>/dev/null || echo "")
        if [ "$current" = "$latest" ]; then
            echo "✓ Claude Code CLI already at latest ($current)"
            return 0
        fi
        echo "📦 Upgrading Claude Code CLI $current → $latest..."
    else
        echo "📦 Installing Claude Code CLI v${latest}..."
    fi

    npm install -g "@anthropic-ai/claude-code@latest" 2>&1 | tail -1

    if [ -x "$HOME/.local/bin/claude" ]; then
        echo "✓ Claude Code CLI $(claude --version 2>/dev/null) installed at ~/.local/bin/claude"
        return 0
    fi
    echo "⚠️  Claude CLI not found after install — check npm prefix"
    return 1
}

main() {
    case "${1:-}" in
        "") ;;
        *)
            echo "Usage: claude.sh"
            echo "  Install Claude Code CLI"
            exit 1
            ;;
    esac

    echo ""
    echo "🤖 Claude Code Setup"
    echo "────────────────────"
    install_claude_cli
    echo ""
}

main "$@"
