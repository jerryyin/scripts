#!/bin/bash
# priv.sh - Container initialization script (works for both Docker and Kubernetes)
#
# This script handles privileged/runtime setup that needs to run once per container:
#   - SSH key setup from persistent storage
#   - Hostname resolution fix
#   - Authentication credential sync (gist, copilot, gh, etc.)
#
# Works with:
#   - Docker: host home mounted at /zyin
#   - Kubernetes: PVC mounted at /{username}
#
# Called by:
#   - Docker: docker-compose.yml at container startup
#   - Kubernetes: Can be called manually or from setup-pod.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

    if [ -f ~/.ssh/id_rsa ]; then
        echo "✓ SSH keys already configured"
        return 0
    fi

    echo "🔑 Setting up SSH keys from $persistent_root..."
    rm -rf ~/.ssh
    cp -r "$persistent_root/.ssh" ~/.ssh
    chmod 700 ~/.ssh 2>/dev/null || true
    chmod 600 ~/.ssh/id_rsa 2>/dev/null || true
    ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts 2>/dev/null || true
    echo "✓ SSH keys configured"
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

# Main
main() {
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

    # 1. Setup SSH keys
    setup_ssh "$persistent_root"

    # 2. Fix hostname resolution
    fix_hostname

    # 3. Sync authentication credentials
    echo ""
    if [ -f "$SCRIPT_DIR/credentials.sh" ]; then
        bash "$SCRIPT_DIR/credentials.sh"
    elif [ -f ~/scripts/docker/env/credentials.sh ]; then
        bash ~/scripts/docker/env/credentials.sh
    else
        echo "⚠️  credentials.sh not found - skipping credential sync"
    fi

    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ Container initialization complete"
    echo "════════════════════════════════════════════════════════════════"
}

main "$@"
