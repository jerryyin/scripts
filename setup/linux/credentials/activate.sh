#!/bin/bash
# activate.sh - Prepare credentials on a host, in Docker, or in Kubernetes.
#
# Invoked at every container start (see docker-compose.yml `command:`), so it
# MUST be lean: no apt/pip/npm installs here. Activation order:
#
#   1. First-time bootstrap when SSH or the vault is unprepared:
#        - SSH activation through rc_files
#        - Hostname resolution fix
#        - sync_vault (link mounted vault, or clone the private vault repo)
#
#   2. Refresh application credentials through rc_files, excluding SSH.
#      This phase never runs application or package installation.
#
# Works with:
#   - Docker: host home mounted at /zyin
#   - Kubernetes: PVC mounted at /{username}
#
# Called by:
#   - Docker: docker-compose.yml at container startup
#   - Kubernetes: via setup/linux/setup.sh (called by connect.sh)
#
# Use `credentials/activate.sh --force` to re-run the first-time bootstrap.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Existing configuration and vault availability determine readiness; no marker file.
needs_first_time_bootstrap() {
    [ ! -L "$HOME/.ssh/private.conf" ] || ! vault_available
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
        echo "   (application credential hooks skip missing vault entries)"
    fi
}

# Phase 1: heavy, run once per container.
first_time_bootstrap() {
    echo "════════════════════════════════════════════════════════════════"
    echo "  Container Initialization (credentials/activate.sh)"
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

    activate_credentials "$persistent_root" --package ssh
    fix_hostname

    echo ""
    sync_vault "$persistent_root"

    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ Container initialization complete"
    echo "════════════════════════════════════════════════════════════════"
}

# Phase 2: cheap, idempotent, runs at every container start.
# No package/dep installs, apt/pip/npm or normal application setup here.
# Docker may authenticate through its configured credential helper.
# Package selection belongs to the caller's initialization or refresh phase.
activate_credentials() {
    local persistent_root="$1"
    shift
    local rc_path="${RC_FILES_DIR:-$HOME/rc_files}"
    [ -f "$rc_path/.setup/__main__.py" ] || {
        echo "Missing rc_files setup: $rc_path/.setup" >&2
        return 1
    }
    local arguments=(credentials --vault "$VAULT_DIR")
    if [ -n "$persistent_root" ]; then
        arguments+=(--persistent-root "$persistent_root")
    fi
    python3 "$rc_path/.setup" "${arguments[@]}" "$@"
}

main() {
    if [ "${1:-}" = "--force" ] || needs_first_time_bootstrap; then
        first_time_bootstrap
    fi
    local persistent_root
    persistent_root=$(find_persistent_root)
    activate_credentials "$persistent_root" --exclude-package ssh
}

main "$@"
