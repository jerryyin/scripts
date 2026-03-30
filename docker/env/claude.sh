#!/bin/bash
# claude.sh - Install Claude Code CLI and configure credentials
#
# This script:
#   1. Installs Claude Code CLI via npm (skips if already installed)
#   2. Patches ~/.claude.json with the subscription key from a private gist
#      (only if the placeholder __CLAUDE_SUB_KEY__ is present)
#
# Prerequisites:
#   - nodejs + npm installed (handled by min.sh)
#   - ~/.claude.json template deployed by rc_files/install.sh (stow)
#
# The subscription key is stored in a private GitHub gist to keep it
# out of public repos. The gist raw URL is stable across revisions.

set -e

CLAUDE_VERSION="2.1.22"
GIST_RAW_URL="YOUR_PRIVATE_GIST_URL_HERE"
CLAUDE_CONFIG="$HOME/.claude.json"
PLACEHOLDER="__CLAUDE_SUB_KEY__"

install_claude_cli() {
    if command -v claude &>/dev/null; then
        echo "✓ Claude Code CLI already installed ($(claude --version 2>/dev/null || echo 'unknown'))"
        return 0
    fi

    echo "📦 Installing Claude Code CLI v${CLAUDE_VERSION}..."
    # Always install to ~/.local to avoid /usr/local being overridden
    # by Docker tools_volume mount at runtime
    mkdir -p "$HOME/.local"
    npm config set prefix "$HOME/.local" 2>/dev/null || true
    npm install -g "@anthropic-ai/claude-code@${CLAUDE_VERSION}" 2>&1 | tail -1

    if [ -x "$HOME/.local/bin/claude" ]; then
        echo "✓ Claude Code CLI installed at ~/.local/bin/claude"
        return 0
    fi
    echo "⚠️  Claude CLI not found after install — check npm prefix"
    return 1
}

patch_subscription_key() {
    if [ ! -f "$CLAUDE_CONFIG" ]; then
        echo "⚠️  $CLAUDE_CONFIG not found — run rc_files/install.sh first"
        return 0
    fi

    if ! grep -q "$PLACEHOLDER" "$CLAUDE_CONFIG"; then
        echo "✓ Claude config already has subscription key"
        return 0
    fi

    echo "🔑 Fetching subscription key from private gist..."
    local sub_key
    sub_key=$(wget -qO- "$GIST_RAW_URL" 2>/dev/null || curl -fsSL "$GIST_RAW_URL" 2>/dev/null || echo "")
    sub_key=$(echo "$sub_key" | tr -d '[:space:]')

    if [ -z "$sub_key" ]; then
        echo "⚠️  Could not fetch subscription key from gist — skipping"
        echo "   You can manually replace $PLACEHOLDER in $CLAUDE_CONFIG"
        return 0
    fi

    sed -i "s/${PLACEHOLDER}/${sub_key}/" "$CLAUDE_CONFIG"
    echo "✓ Subscription key patched into $CLAUDE_CONFIG"
}

main() {
    echo ""
    echo "🤖 Claude Code Setup"
    echo "────────────────────"
    install_claude_cli
    patch_subscription_key
    echo ""
}

main "$@"
