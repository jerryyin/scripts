#!/bin/bash
# node.sh - Ensure a modern Node.js is installed via NodeSource
#
# Ubuntu's bundled nodejs (e.g. v12 on jammy) predates modern npm CLIs like
# claude-code/codex, so this registers NodeSource's repo (only when the
# currently-installed node is actually too old to matter) and installs
# nodejs from it.
#
# Prerequisites:
#   - apt + curl + sudo (handled by min.sh before this runs)

set -e

NODE_MIN_MAJOR=18

# Wait (with visible progress + a bounded timeout) for any in-progress apt/dpkg
# run to release the lock, instead of relying on apt-get's own silent,
# unbounded retry. Safe/instant no-op when nothing holds the lock (the normal
# case for fresh containers). Duplicated from min.sh rather than shared,
# since this script also needs to be runnable standalone.
wait_for_dpkg_lock() {
    local waited=0
    local max_wait=600
    while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
        if [ "$waited" -ge "$max_wait" ]; then
            echo "⚠️  dpkg lock still held after ${max_wait}s -- proceeding anyway, apt may fail" >&2
            return 1
        fi
        echo "⏳ Waiting for dpkg lock (held by another process, e.g. a background apt/unattended-upgrade run)... ${waited}s elapsed"
        sleep 10
        waited=$((waited + 10))
    done
    return 0
}

current_node_major() {
    if command -v node >/dev/null 2>&1; then
        node -e 'console.log(process.versions.node.split(".")[0])' 2>/dev/null || echo 0
    else
        echo 0
    fi
}

restore_shielded_apt_sources() {
    local f
    [ -n "${NODESOURCE_SHIELD_DIR:-}" ] || return 0
    for f in "$NODESOURCE_SHIELD_DIR"/*; do
        [ -e "$f" ] || continue
        echo "Restoring shielded apt source: $(basename "$f")"
        sudo mv "$f" /etc/apt/sources.list.d/ 2>/dev/null || true
    done
    rmdir "$NODESOURCE_SHIELD_DIR" 2>/dev/null || true
}

# The NodeSource setup script runs its own internal apt-get update, which is
# exactly the kind of step that can hang on a lock held by a concurrent
# background process -- wait for the lock first, then bound the whole thing
# so a genuine network stall fails loud instead of hanging forever.
setup_nodesource_repo() {
    wait_for_dpkg_lock

    # NodeSource's own setup script has a real bug: `if ! apt update -y; then
    # handle_error "$?" ...` reads $? *inside* the then-branch, where it has
    # already been clobbered by the negated test (always 0 there) instead of
    # apt's real exit code -- so it calls `exit 0` even when apt update
    # genuinely failed. That means it treats ANY apt fetch error as fatal,
    # even from a repo that has nothing to do with Node -- one stale/dead
    # third-party source anywhere on the box can permanently block Node
    # installs. Rather than needing a human to diagnose and repin that repo
    # every time this happens, pre-flight our own `apt-get update`, and
    # temporarily shield (move out of sources.list.d) only the specific
    # files that are actually erroring right now, so NodeSource's script
    # sees a clean apt state. Always restored afterwards -- on success,
    # failure, or an unexpected exit -- via the EXIT trap.
    local NODESOURCE_SHIELD_DIR
    NODESOURCE_SHIELD_DIR="$(mktemp -d)"
    trap restore_shielded_apt_sources EXIT

    local apt_update_output failing_hosts host src
    apt_update_output="$(sudo apt-get update --allow-insecure-repositories 2>&1)"
    echo "$apt_update_output"
    failing_hosts=$(printf '%s\n' "$apt_update_output" | grep -E '^(E|W): ' | grep -oP "https?://\K[^/ ']+" | sort -u)
    for host in $failing_hosts; do
        for src in /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
            [ -f "$src" ] || continue
            if grep -q "$host" "$src" 2>/dev/null; then
                echo "⚠️  Shielding $(basename "$src") (fetch errors for $host) so it can't block the NodeSource install"
                sudo mv "$src" "$NODESOURCE_SHIELD_DIR/"
            fi
        done
    done

    curl -fsSL --max-time 30 https://deb.nodesource.com/setup_current.x | timeout 300 sudo -E bash -

    restore_shielded_apt_sources
    trap - EXIT

    # Don't trust NodeSource's exit code (see bug above) -- check the repo
    # was actually registered so a failure can't hide silently. If this
    # still fires even with unrelated repos shielded, the problem is
    # NodeSource's own setup (e.g. deb.nodesource.com unreachable), not a
    # collateral third-party repo, and needs a human to look at it.
    if [ ! -f /etc/apt/sources.list.d/nodesource.sources ] && [ ! -f /etc/apt/sources.list.d/nodesource.list ]; then
        echo "⚠️  NodeSource repo was not registered even with unrelated repos shielded -- the setup script itself is failing. nodejs install below will fall back to Ubuntu's outdated version." >&2
    fi
}

install_nodejs() {
    local major installed_major
    major=$(current_node_major)

    # Once NodeSource's own nodejs is installed this is a no-op every
    # subsequent run, so skip re-invoking the setup script (and its noisy,
    # unsilenceable "apt does not have a stable CLI interface" warning) once
    # we're already on a recent-enough major version.
    if [ "$major" -lt "$NODE_MIN_MAJOR" ]; then
        setup_nodesource_repo
    else
        echo "✓ Node.js $(node -v) already >= v${NODE_MIN_MAJOR}, skipping NodeSource setup"
    fi

    wait_for_dpkg_lock
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -f -y nodejs

    # NodeSource nodejs replaces Ubuntu's nodejs+libnode+node-* ecosystem in a
    # single transaction, leaving nodejs with zero reverse-deps. apt then
    # considers it "auto-installed" and apt autoremove will purge it. Mark it
    # manual.
    sudo apt-mark manual nodejs

    # Verify the install actually got a modern Node, independent of the two
    # soft failure modes above (NodeSource's silent exit-0 bug, or apt-get
    # quietly keeping an old package around). claude.sh/codex.sh need this
    # and would otherwise fail downstream with a confusing npm error instead
    # of a clear one.
    installed_major=$(current_node_major)
    if [ "${installed_major:-0}" -lt "$NODE_MIN_MAJOR" ]; then
        echo "❌ nodejs is still $(node -v 2>/dev/null || echo 'missing') (< v${NODE_MIN_MAJOR}) after install. The NodeSource repo probably failed to register -- check apt-get update output above for fetch errors from unrelated apt sources. claude.sh/codex.sh will likely fail until this is fixed." >&2
        return 1
    fi
    echo "✓ Node.js $(node -v) ready"
}

main() {
    case "${1:-}" in
        "") ;;
        *)
            echo "Usage: node.sh"
            echo "  Ensure a modern (>= v${NODE_MIN_MAJOR}) Node.js is installed"
            exit 1
            ;;
    esac

    echo ""
    echo "🟢 Node.js Setup"
    echo "────────────────"
    install_nodejs
    echo ""
}

main "$@"
