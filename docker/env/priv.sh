#!/bin/bash
# priv.sh - Container initialization script (works for both Docker and Kubernetes)
#
# Invoked at every container start (see docker-compose.yml `command:`), so it
# MUST be lean: no apt/pip/npm installs here. Two phases:
#
#   1. First-time bootstrap (only when ~/.ssh/id_rsa is missing):
#        - SSH keys from persistent storage
#        - Hostname resolution fix
#        - sync_vault (link mounted vault, or clone the private vault repo)
#        - sync_investigations (link or clone the private investigations repo)
#
#   2. Runtime patches (every start, idempotent + cheap, no network):
#        - credentials.sh (symlink OAuth credentials from persistent storage)
#        - vault.sh claude (vault -> ~/.claude.json)
#        - vault.sh docker (vault -> ~/.docker/config.json)
#        - vault.sh atlartifactory (vault -> ~/.netrc)
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

# First-time bootstrap is gated on two artifacts, so a partial prior setup
# self-heals instead of getting stuck:
#   1. SSH private key (id_rsa or id_ed25519) — the heavy one-time setup marker.
#   2. The vault (~/vault). SSH keys can be present (e.g. restored from
#      persistent storage on a prior start) while the vault was never synced,
#      which would otherwise leave vault.sh patches permanently warning
#      "vault not synced yet". Missing either one re-runs the (idempotent)
#      bootstrap.
needs_first_time_bootstrap() {
    if [ ! -f ~/.ssh/id_rsa ] && [ ! -f ~/.ssh/id_ed25519 ]; then
        return 0
    fi
    if ! vault_available; then
        return 0
    fi
    return 1
}

# Vault is usable once ~/vault resolves to a populated directory — either a
# symlink to persistent storage or a cloned repo. A dangling symlink, a missing
# path, or an empty dir (partial clone) all count as unavailable so we retry.
vault_available() {
    [ -d "$VAULT_DIR" ] && [ -n "$(ls -A "$VAULT_DIR" 2>/dev/null)" ]
}

# Find the persistent storage root (mounted host home or PVC)
. "$SCRIPT_DIR/../lib/find_persistent_root.sh"
. "$SCRIPT_DIR/../lib/fix_hostname.sh"

# Setup SSH keys from persistent storage. GitHub host-key trust (the
# ssh.github.com:443 redirect this network needs) is handled declaratively by
# rc_files' ~/.ssh/config (StrictHostKeyChecking accept-new on the github.com
# block) once it's stowed, so it doesn't need any imperative setup here.
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
    chmod 600 ~/.ssh/id_* 2>/dev/null || true
    chmod 644 ~/.ssh/id_*.pub 2>/dev/null || true
    echo "✓ SSH keys configured"
}

# Clone the private vault repo (~/vault). Holds shared dev secrets in plaintext;
# its privacy is gated by GitHub repo permissions / your account's SSH keys.
# Idempotent: clone if missing, otherwise no-op (run `git -C ~/vault pull` by
# hand to refresh).
VAULT_REPO="${VAULT_REPO:-git@github.com:jerryyin/vault.git}"
VAULT_DIR="${VAULT_DIR:-$HOME/vault}"
sync_vault() {
    local persistent_root="${1:-}"
    local persistent_vault="${persistent_root:+$persistent_root/vault}"

    if [ -n "$persistent_vault" ] && [ -d "$persistent_vault" ] && [ ! -e "$VAULT_DIR" ] && [ ! -L "$VAULT_DIR" ]; then
        ln -s "$persistent_vault" "$VAULT_DIR"
    fi
    if [ -n "$persistent_vault" ] && [ -d "$persistent_vault" ] \
        && [ -L "$VAULT_DIR" ] && [ "$(readlink "$VAULT_DIR")" = "$persistent_vault" ]; then
        [ "${QUIET_VAULT_SYNC:-0}" = "1" ] || echo "✓ vault linked at $VAULT_DIR -> $persistent_vault"
        return 0
    fi

    if [ -d "$VAULT_DIR/.git" ]; then
        echo "✓ vault already cloned at $VAULT_DIR (use 'git -C $VAULT_DIR pull' to refresh)"
        return 0
    fi
    echo "📥 Cloning vault from $VAULT_REPO..."
    if git clone --depth 1 --quiet "$VAULT_REPO" "$VAULT_DIR"; then
        echo "✓ vault cloned to $VAULT_DIR"
    else
        echo "⚠️  vault clone failed — make sure your SSH key is added at https://github.com/settings/keys"
        echo "   (vault.sh patches will be skipped until the vault is present)"
    fi
}

# Clone the private investigations repo (~/triton-investigations): GPU kernel
# investigation records plus the unattended bare-metal campaign harness.
#
# This lives here rather than in min.sh — where rc_files and scripts are cloned —
# because those two are public and clone over https, while this one is private
# and needs the SSH key that setup_ssh installs just above. Same reason vault is
# here. A failure is a warning, not an error: plenty of containers never touch a
# campaign, and the rest of initialization must still finish.
#
# Only runs during first-time bootstrap (phase 2 forbids network calls), so use
# `priv.sh --force` to pick it up on a container that was bootstrapped earlier.
INVESTIGATIONS_REPO="${INVESTIGATIONS_REPO:-git@github.com:jerryyin/triton-investigations.git}"
INVESTIGATIONS_DIR="${INVESTIGATIONS_DIR:-$HOME/triton-investigations}"
sync_investigations() {
    local persistent_root="${1:-}"
    local persistent_repo="${persistent_root:+$persistent_root/triton-investigations}"

    # Prefer the copy on mounted host storage so a campaign's uncommitted
    # results survive the container, exactly as vault does.
    if [ -n "$persistent_repo" ] && [ -d "$persistent_repo" ] \
        && [ ! -e "$INVESTIGATIONS_DIR" ] && [ ! -L "$INVESTIGATIONS_DIR" ]; then
        ln -s "$persistent_repo" "$INVESTIGATIONS_DIR"
    fi
    if [ -n "$persistent_repo" ] && [ -d "$persistent_repo" ] \
        && [ -L "$INVESTIGATIONS_DIR" ] && [ "$(readlink "$INVESTIGATIONS_DIR")" = "$persistent_repo" ]; then
        echo "✓ investigations linked at $INVESTIGATIONS_DIR -> $persistent_repo"
        return 0
    fi

    if [ -d "$INVESTIGATIONS_DIR/.git" ]; then
        echo "✓ investigations already cloned at $INVESTIGATIONS_DIR (use 'git -C $INVESTIGATIONS_DIR pull' to refresh)"
        return 0
    fi
    echo "📥 Cloning investigations from $INVESTIGATIONS_REPO..."
    # Full history, not --depth 1: the ledgers are the evidence trail and
    # `git log --follow` across the stage split has to keep working.
    if git clone --quiet "$INVESTIGATIONS_REPO" "$INVESTIGATIONS_DIR"; then
        echo "✓ investigations cloned to $INVESTIGATIONS_DIR"
    else
        echo "⚠️  investigations clone failed — make sure your SSH key is added at https://github.com/settings/keys"
        echo "   (skip this if you are not running a measurement campaign)"
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
    sync_vault "$persistent_root"

    echo ""
    sync_investigations "$persistent_root"

    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ Container initialization complete"
    echo "════════════════════════════════════════════════════════════════"
}

# Phase 2: cheap, idempotent, runs at every container start.
# Strict rule: NO network calls, NO package/dep installs, NO apt/pip/npm here.
# Each step must early-return quickly when there is nothing to do, and stay
# silent on the no-op path so repeat container starts don't spam.
runtime_patches() {
    local persistent_root
    persistent_root=$(find_persistent_root)
    if [ -n "$persistent_root" ] && [ -d "$persistent_root/vault" ]; then
        QUIET_VAULT_SYNC=1 sync_vault "$persistent_root"
    fi
    bash "$SCRIPT_DIR/credentials.sh"
    bash "$SCRIPT_DIR/vault.sh" claude
    bash "$SCRIPT_DIR/vault.sh" docker
    bash "$SCRIPT_DIR/vault.sh" gh
    bash "$SCRIPT_DIR/vault.sh" gist
    bash "$SCRIPT_DIR/vault.sh" atlartifactory
}

main() {
    if [ "${1:-}" = "--force" ] || needs_first_time_bootstrap; then
        first_time_bootstrap
    fi
    runtime_patches
}

main "$@"
