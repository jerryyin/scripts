#!/bin/bash
# claude.sh - Install Claude Code CLI and configure credentials
#
# Modes:
#   claude.sh                # Full setup: install CLI + patch subscription key
#   claude.sh --patch-only   # Patch only (no npm install). Use this at runtime
#                            # (priv.sh): fast, no network, no dependency install,
#                            # silent on no-op.
#
# Source of truth for the subscription key: $HOME/vault/claude_key.txt
# (cloned by priv.sh's sync_vault from a private GitHub repo). The patch step
# reads the plaintext from there and sed-substitutes it into ~/.claude.json
# in place of the __CLAUDE_SUB_KEY__ placeholder shipped by rc_files.
#
# Prerequisites:
#   - nodejs + npm installed (handled by min.sh) — only for full setup
#   - ~/.claude.json template deployed by rc_files/install.sh (stow)
#   - ~/vault/claude_key.txt populated — handled by priv.sh's sync_vault

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="${KEY_FILE:-$HOME/vault/claude_key.txt}"
CLAUDE_CONFIG="$HOME/.claude.json"
PLACEHOLDER="__CLAUDE_SUB_KEY__"

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

patch_subscription_key() {
    if [ ! -f "$CLAUDE_CONFIG" ]; then
        [ "${QUIET_NOOP:-0}" = "1" ] || echo "⚠️  $CLAUDE_CONFIG not found — run rc_files/install.sh first"
        return 0
    fi

    if ! grep -q "$PLACEHOLDER" "$CLAUDE_CONFIG"; then
        [ "${QUIET_NOOP:-0}" = "1" ] || echo "✓ Claude config already has subscription key"
        return 0
    fi

    if [ ! -f "$KEY_FILE" ]; then
        [ "${QUIET_NOOP:-0}" = "1" ] || {
            echo "⚠️  $KEY_FILE not found — vault not synced yet"
            echo "   Run priv.sh to clone the vault, then re-run this script."
        }
        return 0
    fi

    local sub_key
    sub_key=$(tr -d '[:space:]' < "$KEY_FILE")
    if [ -z "$sub_key" ]; then
        echo "⚠️  $KEY_FILE is empty"
        return 0
    fi

    sed -i "s/${PLACEHOLDER}/${sub_key}/" "$CLAUDE_CONFIG"
    echo "✓ Subscription key patched into $CLAUDE_CONFIG"
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
        QUIET_NOOP=1 patch_subscription_key
        return
    fi

    echo ""
    echo "🤖 Claude Code Setup"
    echo "────────────────────"
    install_claude_cli
    patch_subscription_key
    echo ""
}

main "$@"
