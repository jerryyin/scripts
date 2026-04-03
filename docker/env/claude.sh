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
# The subscription key is encrypted with the user's SSH public key using 'age'.
# Only someone with the matching private SSH key can decrypt it.
# This works seamlessly since SSH keys are synced via credentials.sh/priv.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCRYPTED_KEY_FILE="$SCRIPT_DIR/claude_key.age"
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
        echo "⚠️  $CLAUDE_CONFIG not found — run rc_files/install.sh first"
        return 0
    fi

    if ! grep -q "$PLACEHOLDER" "$CLAUDE_CONFIG"; then
        echo "✓ Claude config already has subscription key"
        return 0
    fi

    echo "🔑 Decrypting subscription key with SSH key..."
    local sub_key

    if [ ! -f "$ENCRYPTED_KEY_FILE" ]; then
        echo "⚠️  Encrypted key file not found: $ENCRYPTED_KEY_FILE"
        echo "   Create it with: echo 'YOUR_KEY' | age -R ~/.ssh/id_rsa.pub -o $ENCRYPTED_KEY_FILE"
        return 0
    fi

    if ! command -v age &>/dev/null; then
        echo "⚠️  age not installed — run 'apt install age' or 'brew install age'"
        return 0
    fi

    # Try common SSH key locations
    local ssh_key=""
    for key in ~/.ssh/id_ed25519 ~/.ssh/id_rsa; do
        if [ -f "$key" ]; then
            ssh_key="$key"
            break
        fi
    done

    if [ -z "$ssh_key" ]; then
        echo "⚠️  No SSH private key found — ensure credentials.sh has run"
        return 0
    fi

    sub_key=$(age -d -i "$ssh_key" "$ENCRYPTED_KEY_FILE" 2>/dev/null || echo "")
    sub_key=$(echo "$sub_key" | tr -d '[:space:]')

    if [ -z "$sub_key" ]; then
        echo "⚠️  Could not decrypt key — wrong SSH key or corrupted file"
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
