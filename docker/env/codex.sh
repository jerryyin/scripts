#!/bin/bash
# codex.sh - Install OpenAI Codex CLI and configure local defaults
#
# Usage:
#   codex.sh                # Install CLI + copy config template
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
# Without pipefail, `npm install ... | tail -1` always reports tail's exit
# code (0), so a real npm failure would be silently treated as success.
set -o pipefail

CODEX_CONFIG="$HOME/.codex/config.toml"
CODEX_TEMPLATE="$HOME/.codex.config.toml.template"

install_codex_cli() {
    mkdir -p "$HOME/.local"
    npm config set prefix "$HOME/.local" 2>/dev/null || true

    # Some environments (corporate proxies, WSL) present a TLS chain npm
    # cannot verify, causing UNABLE_TO_GET_ISSUER_CERT_LOCALLY. Prefer a
    # configured CA bundle (see rc_files/zsh/.zshrc) so npm can verify the
    # proxy's cert instead of failing closed.
    local -a NPM_TLS_FLAGS=()
    if [ -n "${NODE_EXTRA_CA_CERTS:-}" ] && [ -f "${NODE_EXTRA_CA_CERTS}" ]; then
        NPM_TLS_FLAGS+=(--cafile "${NODE_EXTRA_CA_CERTS}")
    fi

    local latest
    latest=$(npm view "${NPM_TLS_FLAGS[@]}" @openai/codex version 2>/dev/null || echo "")
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

    if ! npm install -g "${NPM_TLS_FLAGS[@]}" "@openai/codex@latest" 2>&1 | tail -1; then
        echo "⚠️  npm install failed — see above"
        return 1
    fi

    if [ -x "$HOME/.local/bin/codex" ]; then
        local installed
        installed=$(codex --version 2>/dev/null | grep -oP '[\d.]+' | head -1 || echo "")
        if [ "$installed" != "$latest" ]; then
            echo "⚠️  Codex CLI at ~/.local/bin/codex is still $installed, not $latest — install may have silently failed"
            return 1
        fi
        echo "✓ Codex CLI $installed installed at ~/.local/bin/codex"
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
    case "${1:-}" in
        "") ;;
        *)
            echo "Usage: codex.sh"
            echo "  Install Codex CLI + copy config template"
            exit 1
            ;;
    esac

    echo ""
    echo "🤖 Codex CLI Setup"
    echo "──────────────────"
    install_codex_cli
    patch_codex_config
    echo ""
}

main "$@"
