#!/bin/bash
# priv.sh - Container initialization script (works for both Docker and Kubernetes)
#
# Invoked at every container start (see docker-compose.yml `command:`), so it
# MUST be lean: no apt/pip/npm installs here. Two phases:
#
#   1. First-time bootstrap (only when ~/.ssh/id_rsa is missing):
#        - SSH keys from persistent storage
#        - Hostname resolution fix
#        - credentials.sh (symlink credentials from persistent storage)
#        - sync_vault (clone the private vault repo with shared dev secrets)
#
#   2. Runtime patches (every start, idempotent + cheap, no network):
#        - claude.sh --patch-only (read ~/vault/claude_key.txt -> ~/.claude.json)
#        - codex.sh  --patch-only (copy ~/.codex.config.toml.template -> ~/.codex/config.toml)
#        - gh.sh     --patch-only (read ~/vault/gh_token.txt   -> ~/.config/gh/hosts.yml)
#
# Works with:
#   - Docker: host home mounted at /zyin
#   - Kubernetes: PVC mounted at /{username}
#
# Called by:
#   - Docker: docker-compose.yml at container startup
#   - Kubernetes: via setup-service.sh (called by connect.sh)
#
# Use `priv.sh --force` to re-run the first-time bootstrap.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First-time bootstrap is gated on the SSH private key: once any supported key
# (id_rsa or id_ed25519) exists, we assume the heavy one-time setup is done and
# skip straight to runtime patches.
needs_first_time_bootstrap() {
    [ ! -f ~/.ssh/id_rsa ] && [ ! -f ~/.ssh/id_ed25519 ]
}

# Find the persistent storage root (same logic as credentials.sh)
find_persistent_root() {
    local username
    username=$(id -un 2>/dev/null || whoami 2>/dev/null || echo "")

    # Strategy 1: Try username-based paths (Kubernetes pattern: /{username})
    if [ -n "$username" ] && [ -d "/$username" ] && [ "/$username" != "/root" ] && [ "/$username" != "/home" ]; then
        echo "/$username"
        return 0
    fi

    # Strategy 2: Check for common Docker mount patterns
    for candidate in /zyin /host_home /persistent; do
        if [ -d "$candidate/.ssh" ] || [ -d "$candidate/rc_files" ]; then
            echo "$candidate"
            return 0
        fi
    done

    # Strategy 3: Scan root-level directories for home-like markers
    for candidate in /*/; do
        candidate="${candidate%/}"
        case "$candidate" in
            /bin|/boot|/dev|/etc|/home|/lib*|/media|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var|/data|/ffm|/tools|/snap)
                continue
                ;;
        esac
        if [ -d "$candidate/.ssh" ] || [ -d "$candidate/rc_files" ] || [ -d "$candidate/scripts" ]; then
            echo "$candidate"
            return 0
        fi
    done

    echo ""
}

# Setup SSH keys from persistent storage
setup_ssh() {
    local persistent_root="$1"

    if [ -z "$persistent_root" ]; then
        echo "ℹ️  No persistent storage - skipping SSH setup"
        return 0
    fi

    if [ ! -d "$persistent_root/.ssh" ]; then
        echo "ℹ️  No SSH keys in persistent storage - skipping SSH setup"
        return 0
    fi

    if [ -f ~/.ssh/id_rsa ] || [ -f ~/.ssh/id_ed25519 ]; then
        echo "✓ SSH keys already configured"
        return 0
    fi

    echo "🔑 Setting up SSH keys from $persistent_root..."
    rm -rf ~/.ssh
    cp -r "$persistent_root/.ssh" ~/.ssh
    chmod 700 ~/.ssh 2>/dev/null || true
    # Lock down private keys; public pairs (*.pub) stay world-readable.
    chmod 600 ~/.ssh/id_rsa ~/.ssh/id_ed25519 2>/dev/null || true
    chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/id_ed25519.pub 2>/dev/null || true
    ssh-keyscan -t rsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true
    echo "✓ SSH keys configured"
}

# Clone the private vault repo (~/vault). Holds shared dev secrets in plaintext;
# its privacy is gated by GitHub repo permissions / your account's SSH keys.
# Idempotent: clone if missing, otherwise no-op (run `git -C ~/vault pull` by
# hand to refresh).
VAULT_REPO="${VAULT_REPO:-git@github.com:jerryyin/vault.git}"
VAULT_DIR="${VAULT_DIR:-$HOME/vault}"
sync_vault() {
    if [ -d "$VAULT_DIR/.git" ]; then
        echo "✓ vault already cloned at $VAULT_DIR (use 'git -C $VAULT_DIR pull' to refresh)"
        return 0
    fi
    echo "📥 Cloning vault from $VAULT_REPO..."
    if git clone --depth 1 --quiet "$VAULT_REPO" "$VAULT_DIR"; then
        echo "✓ vault cloned to $VAULT_DIR"
    else
        echo "⚠️  vault clone failed — make sure your SSH key is added at https://github.com/settings/keys"
        echo "   (claude.sh / gh.sh patches will be skipped until the vault is present)"
    fi
}

# Fix hostname resolution
fix_hostname() {
    if [ -z "${HOSTNAME:-}" ]; then
        return 0
    fi

    if grep -q "$HOSTNAME" /etc/hosts 2>/dev/null; then
        return 0
    fi

    echo "🔧 Fixing hostname resolution..."
    local ip
    ip=$(hostname -I 2>/dev/null | cut -d' ' -f1)
    if [ -n "$ip" ]; then
        echo "$ip $HOSTNAME" | sudo -h 127.0.0.1 tee -a /etc/hosts >/dev/null 2>&1 || true
    fi
}

# Phase 1: heavy, run once per container.
first_time_bootstrap() {
    echo "════════════════════════════════════════════════════════════════"
    echo "  Container Initialization (priv.sh)"
    echo "════════════════════════════════════════════════════════════════"

    local persistent_root
    persistent_root=$(find_persistent_root)

    if [ -n "$persistent_root" ]; then
        echo "  Persistent storage: $persistent_root"
    else
        echo "  Persistent storage: (not found)"
    fi
    echo "════════════════════════════════════════════════════════════════"
    echo ""

    setup_ssh "$persistent_root"
    fix_hostname

    echo ""
    bash "$SCRIPT_DIR/credentials.sh"

    echo ""
    sync_vault

    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ Container initialization complete"
    echo "════════════════════════════════════════════════════════════════"
}

# Phase 2: cheap, idempotent, runs at every container start.
# Strict rule: NO network calls, NO package/dep installs, NO apt/pip/npm here.
# Each step must early-return quickly when there is nothing to do, and stay
# silent on the no-op path so repeat container starts don't spam.
runtime_patches() {
    bash "$SCRIPT_DIR/claude.sh" --patch-only
    bash "$SCRIPT_DIR/codex.sh"  --patch-only
    bash "$SCRIPT_DIR/gh.sh"     --patch-only
}

main() {
    if [ "${1:-}" = "--force" ] || needs_first_time_bootstrap; then
        first_time_bootstrap
    fi
    runtime_patches
}

main "$@"
