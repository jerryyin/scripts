#!/bin/bash
# codex.sh - Install OpenAI Codex CLI and configure local defaults
#
# Modes:
#   codex.sh                # Full setup: install CLI + copy config template
#   codex.sh --patch-only   # Patch config only (no npm install). Use this at
#                            # runtime (priv.sh): fast, no network, no
#                            # dependency install, silent on no-op.
#
# Credentials are managed by credentials.sh (symlinks ~/.codex from persistent
# storage). Unlike Claude/gh whose secrets live in vault, Codex uses OAuth with
# single-use refresh tokens — a static vault copy would go stale after one token
# refresh.
#
# The local config file (~/.codex/config.toml) is generated from the template
# (~/.codex.config.toml.template, deployed by rc_files/install.sh via stow).
# Keep the real config untracked because Codex may add local runtime state.
#
# Prerequisites:
#   - nodejs + npm installed (handled by min.sh)
#   - ~/.codex.config.toml.template deployed by rc_files/install.sh (stow)

set -e

CODEX_CONFIG="$HOME/.codex/config.toml"
CODEX_TEMPLATE="$HOME/.codex.config.toml.template"

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

patch_codex_config() {
    if [ ! -f "$CODEX_TEMPLATE" ]; then
        [ "${QUIET_NOOP:-0}" = "1" ] || echo "⚠️  $CODEX_TEMPLATE not found — run rc_files/install.sh first"
        return 0
    fi

    mkdir -p "$(dirname "$CODEX_CONFIG")"

    # Copy template if config doesn't exist or template is newer.
    if [ ! -f "$CODEX_CONFIG" ] || [ "$CODEX_TEMPLATE" -nt "$CODEX_CONFIG" ]; then
        cp "$CODEX_TEMPLATE" "$CODEX_CONFIG"
        chmod 600 "$CODEX_CONFIG" 2>/dev/null || true
        echo "✓ Copied $CODEX_TEMPLATE → $CODEX_CONFIG"
    fi
}

main() {
    local skip_install=false
    case "${1:-}" in
        --patch-only) skip_install=true ;;
        "") ;;
        *)
            echo "Usage: codex.sh [--patch-only]"
            echo "  (no args)      Install Codex CLI + copy config template (build time)"
            echo "  --patch-only   Copy config template only, skip CLI install (runtime)"
            exit 1
            ;;
    esac

    if [ "$skip_install" = true ]; then
        QUIET_NOOP=1 patch_codex_config
        return
    fi

    echo ""
    echo "🤖 Codex CLI Setup"
    echo "──────────────────"
    install_codex_cli
    patch_codex_config
    echo ""
}

main "$@"
