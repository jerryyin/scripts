#!/bin/bash
# credentials.sh - Bidirectional credential sync for containers
#
# This script enables "turn-key" authentication for tools like:
#   - gist-paste (GitHub gists)
#   - GitHub Copilot (vim plugin)
#   - gh CLI (GitHub CLI)
#   - git credential helpers
#
# MODES:
#   credentials.sh           # Pull: sync from persistent storage to container
#   credentials.sh --save    # Push: save container credentials to persistent storage
#   credentials.sh --status  # Show what credentials exist where
#
# WORKFLOW:
#   First-time setup (new machine/pod):
#     1. Run your container or K8s pod
#     2. Authenticate each tool: gist-paste --login, :Copilot auth, gh auth login
#     3. Run: credentials.sh --save
#     4. Future containers will auto-sync these credentials
#
#   Existing setup:
#     - credentials.sh runs automatically during container init
#     - Symlinks credentials from persistent storage to container $HOME
#
# Prerequisites:
#   - Host home directory mounted at /zyin (Docker) or /{username} (Kubernetes PVC)

set -e

# Credential paths relative to home directory
CREDENTIAL_PATHS=(
    ".gist"                      # gist-paste token
    ".config/github-copilot"     # GitHub Copilot (vim/neovim)
    ".config/gh"                 # GitHub CLI
    ".git-credentials"           # Git credential store
    ".netrc"                     # HTTP basic auth
    ".npmrc"                     # NPM registry auth
)

# Find the persistent storage root (mounted host home or PVC)
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

# Check if a credential exists at a path (file or non-empty directory)
credential_exists() {
    local path="$1"
    if [ -f "$path" ]; then
        return 0
    elif [ -d "$path" ] && [ "$(ls -A "$path" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

# Pull: Sync a single credential from persistent to container
pull_credential() {
    local rel_path="$1"
    local source_path="$PERSISTENT_ROOT/$rel_path"
    local target_path="$HOME/$rel_path"

    if ! credential_exists "$source_path"; then
        return 0  # Source doesn't exist, skip silently
    fi

    local target_dir
    target_dir=$(dirname "$target_path")
    mkdir -p "$target_dir"

    # Check if already correctly linked
    if [ -L "$target_path" ]; then
        local current_target
        current_target=$(readlink -f "$target_path" 2>/dev/null || echo "")
        if [ "$current_target" = "$(readlink -f "$source_path")" ]; then
            echo "  ✓ Already linked: $rel_path"
            return 0
        fi
    fi

    # Remove existing target
    if [ -e "$target_path" ] || [ -L "$target_path" ]; then
        rm -rf "$target_path"
    fi

    # Create symlink
    ln -s "$source_path" "$target_path"
    echo "  ✓ Linked: $rel_path"
}

# Push: Save a single credential from container to persistent storage
push_credential() {
    local rel_path="$1"
    local source_path="$HOME/$rel_path"
    local target_path="$PERSISTENT_ROOT/$rel_path"

    # Skip if source is a symlink (already pointing to persistent)
    if [ -L "$source_path" ]; then
        echo "  ⏭ Skipped (already symlink): $rel_path"
        return 0
    fi

    if ! credential_exists "$source_path"; then
        return 0  # Source doesn't exist, skip silently
    fi

    local target_dir
    target_dir=$(dirname "$target_path")
    mkdir -p "$target_dir"

    # Copy to persistent storage
    if [ -d "$source_path" ]; then
        cp -r "$source_path" "$target_path"
    else
        cp "$source_path" "$target_path"
    fi

    # Now replace local with symlink
    rm -rf "$source_path"
    ln -s "$target_path" "$source_path"

    echo "  ✓ Saved: $rel_path → $target_path"
}

# Show credential status
show_status() {
    PERSISTENT_ROOT=$(find_persistent_root)

    echo "════════════════════════════════════════════════════════════════"
    echo "  Credential Status"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Container HOME:    $HOME"
    echo "  Persistent Root:   ${PERSISTENT_ROOT:-'(not found)'}"
    echo "════════════════════════════════════════════════════════════════"
    echo ""

    printf "%-35s %-12s %-12s\n" "Credential" "Container" "Persistent"
    printf "%-35s %-12s %-12s\n" "----------" "---------" "----------"

    for rel_path in "${CREDENTIAL_PATHS[@]}"; do
        local container_status="✗"
        local persistent_status="✗"

        if credential_exists "$HOME/$rel_path"; then
            if [ -L "$HOME/$rel_path" ]; then
                container_status="→ linked"
            else
                container_status="✓ local"
            fi
        fi

        if [ -n "$PERSISTENT_ROOT" ] && credential_exists "$PERSISTENT_ROOT/$rel_path"; then
            persistent_status="✓"
        fi

        printf "%-35s %-12s %-12s\n" "$rel_path" "$container_status" "$persistent_status"
    done

    echo ""
}

# Main: Pull credentials from persistent storage
pull_all() {
    PERSISTENT_ROOT=$(find_persistent_root)

    if [ -z "$PERSISTENT_ROOT" ]; then
        echo "ℹ️  No persistent storage found - skipping credential sync"
        echo ""
        echo "To set up credentials for the first time:"
        echo "  1. Authenticate each tool:"
        echo "     gist-paste --login"
        echo "     vim → :Copilot auth"
        echo "     gh auth login"
        echo "  2. Mount persistent storage or PVC"
        echo "  3. Run: credentials.sh --save"
        echo ""
        return 0
    fi

    echo "🔐 Syncing credentials from $PERSISTENT_ROOT..."

    local found_any=false
    for rel_path in "${CREDENTIAL_PATHS[@]}"; do
        if credential_exists "$PERSISTENT_ROOT/$rel_path"; then
            found_any=true
        fi
        pull_credential "$rel_path"
    done

    if [ "$found_any" = false ]; then
        echo ""
        echo "ℹ️  No credentials found in persistent storage yet."
        echo ""
        echo "First-time setup instructions:"
        echo "  1. Authenticate each tool you need:"
        echo "     gist-paste --login       # GitHub Gist"
        echo "     vim → :Copilot auth      # GitHub Copilot"
        echo "     gh auth login            # GitHub CLI"
        echo ""
        echo "  2. Save to persistent storage:"
        echo "     ~/scripts/docker/env/credentials.sh --save"
        echo ""
        echo "  Future containers will auto-sync these credentials."
        return 0
    fi

    echo "✅ Credential sync complete"
    echo ""
}

# Main: Push credentials to persistent storage
push_all() {
    PERSISTENT_ROOT=$(find_persistent_root)

    if [ -z "$PERSISTENT_ROOT" ]; then
        echo "❌ No persistent storage found!"
        echo ""
        echo "Cannot save credentials without persistent storage."
        echo "Make sure you have:"
        echo "  - Docker: \$HOME mounted at /zyin"
        echo "  - Kubernetes: PVC mounted at /{username}"
        echo ""
        return 1
    fi

    echo "💾 Saving credentials to $PERSISTENT_ROOT..."

    local saved_any=false
    for rel_path in "${CREDENTIAL_PATHS[@]}"; do
        if credential_exists "$HOME/$rel_path" && [ ! -L "$HOME/$rel_path" ]; then
            saved_any=true
        fi
        push_credential "$rel_path"
    done

    if [ "$saved_any" = false ]; then
        echo ""
        echo "ℹ️  No new credentials to save."
        echo "   All credentials are either already synced or don't exist."
        echo ""
        echo "To set up new credentials:"
        echo "  gist-paste --login       # GitHub Gist"
        echo "  vim → :Copilot auth      # GitHub Copilot"
        echo "  gh auth login            # GitHub CLI"
        echo ""
        return 0
    fi

    echo ""
    echo "✅ Credentials saved to persistent storage!"
    echo "   Future containers will auto-sync these credentials."
    echo ""
}

# Show usage
show_usage() {
    echo "Usage: credentials.sh [--save|--status|--help]"
    echo ""
    echo "Options:"
    echo "  (none)     Pull credentials from persistent storage (default)"
    echo "  --save     Push credentials from container to persistent storage"
    echo "  --status   Show credential status in both locations"
    echo "  --help     Show this help message"
    echo ""
}

# Main entry point
main() {
    case "${1:-}" in
        --save)
            push_all
            ;;
        --status)
            show_status
            ;;
        --help|-h)
            show_usage
            ;;
        "")
            pull_all
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run if executed directly (not sourced)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
