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
# Without pipefail, `npm install ... | tail -1` always reports tail's exit
# code (0), so a real npm failure would be silently treated as success.
set -o pipefail

install_claude_cli() {
    # Always install to ~/.local to avoid /usr/local being overridden
    # by Docker tools_volume mount at runtime
    mkdir -p "$HOME/.local"
    npm config set prefix "$HOME/.local" 2>/dev/null || true

    # Some environments (corporate proxies, WSL) present a TLS chain npm
    # cannot verify, causing UNABLE_TO_GET_ISSUER_CERT_LOCALLY. Prefer a
    # configured CA bundle (see rc_files/zsh/.zshrc) so npm can verify the
    # proxy's cert instead of failing closed (or silently serving a stale
    # cached version). Self-sufficient here (doesn't assume min.sh already
    # set this) since this script can run standalone; no-ops if rc_files
    # hasn't been cloned yet.
    if [ -f "$HOME/rc_files/lib/node-ca-cert.sh" ]; then
        . "$HOME/rc_files/lib/node-ca-cert.sh"
    fi
    local -a NPM_TLS_FLAGS=()
    if [ -n "${NODE_EXTRA_CA_CERTS:-}" ] && [ -f "${NODE_EXTRA_CA_CERTS}" ]; then
        NPM_TLS_FLAGS+=(--cafile "${NODE_EXTRA_CA_CERTS}")
    fi

    local latest
    latest=$(npm view "${NPM_TLS_FLAGS[@]}" @anthropic-ai/claude-code version 2>/dev/null || echo "")
    if [ -z "$latest" ]; then
        echo "⚠️  Could not fetch latest version from npm — check network"
        return 1
    fi

    if command -v claude &>/dev/null; then
        local current
        current=$(claude --version 2>/dev/null | grep -oP '[\d.]+' | head -1 || echo "")
        if [ "$current" = "$latest" ]; then
            echo "✓ Claude Code CLI already at latest ($current)"
            return 0
        fi
        echo "📦 Upgrading Claude Code CLI $current → $latest..."
    else
        echo "📦 Installing Claude Code CLI v${latest}..."
    fi

    # A prior native (non-npm) install manages ~/.local/bin/claude as a
    # symlink into ~/.local/share/claude/versions/<ver> (see ~/.claude.json's
    # "installMethod": "native"). npm's bin-linker refuses to clobber a link
    # it doesn't own, failing with EEXIST. Since this script manages the CLI
    # via npm, clear a foreign link so npm can (re)create its own.
    if [ -L "$HOME/.local/bin/claude" ]; then
        local link_target
        link_target=$(readlink -f "$HOME/.local/bin/claude" 2>/dev/null || echo "")
        case "$link_target" in
            "$HOME"/.local/lib/node_modules/*) ;;
            *)
                echo "ℹ️  Removing non-npm claude link ($HOME/.local/bin/claude → $link_target)"
                rm -f "$HOME/.local/bin/claude"
                ;;
        esac
    fi

    if ! npm install -g "${NPM_TLS_FLAGS[@]}" "@anthropic-ai/claude-code@latest" 2>&1 | tail -1; then
        echo "⚠️  npm install failed — see above"
        return 1
    fi

    if [ -x "$HOME/.local/bin/claude" ]; then
        local installed
        installed=$(claude --version 2>/dev/null | grep -oP '[\d.]+' | head -1 || echo "")
        if [ "$installed" != "$latest" ]; then
            echo "⚠️  Claude CLI at ~/.local/bin/claude is still $installed, not $latest — install may have silently failed"
            return 1
        fi
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
