#!/bin/bash
# claude.sh - Install Claude Code CLI and patch its vault-managed config
#
# Modes:
#   claude.sh                # Full setup: install CLI + patch subscription key
#   claude.sh --patch-only   # Patch only (no npm install). Use this at runtime
#                            # (priv.sh): fast, no network, no dependency install,
#                            # silent on no-op.
#
# Config patching is handled by vault-config.sh, which shares the same
# template + vault placeholder flow used for Docker auth.
#
# Model selection is left to Claude Code's built-in defaults (latest at
# each release). No ANTHROPIC_MODEL or ANTHROPIC_DEFAULT_*_MODEL env
# vars — those override the built-in and require manual bumping.
#
# Prerequisites:
#   - nodejs + npm installed (handled by min.sh) — only for full setup
#   - ~/.claude.json.template deployed by rc_files/install.sh (stow)
#   - ~/vault/claude_key.txt populated — handled by priv.sh's sync_vault

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    local skip_install=false
    case "${1:-}" in
        --patch-only) skip_install=true ;;
        "") ;;
        *)
            echo "Usage: claude.sh [--patch-only]"
            echo "  (no args)      Install Claude Code CLI + patch subscription key (build time)"
            echo "  --patch-only   Patch subscription key only, skip CLI install (runtime)"
            exit 1
            ;;
    esac

    if [ "$skip_install" = true ]; then
        bash "$SCRIPT_DIR/vault-config.sh" claude --patch-only
        return
    fi

    echo ""
    echo "🤖 Claude Code Setup"
    echo "────────────────────"
    install_claude_cli
    bash "$SCRIPT_DIR/vault-config.sh" claude
    echo ""
}

main "$@"
