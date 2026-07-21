#!/bin/bash
# git_workspace.sh - Generic ephemeral workspace cloning, shared by
# workspace/triton.sh and workspace/iree.sh
#
# Library only: source this, then call setup_workspace. Not meant to be run
# directly -- see workspace/triton.sh, workspace/iree.sh for real usage.
#
# setup_workspace <project_name> <github_url> [--submodules]
#   project_name: Name of the project (e.g., "iree", "triton")
#   github_url:   GitHub clone URL (e.g., "https://github.com/iree-org/iree.git")
#   --submodules: Optional flag to initialize git submodules
#
# Purely handles cloning (with persistent-reference reuse when available).
# Anything project-specific is the wrapper script's job, added after calling
# setup_workspace.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the persistent storage root (mounted host home or PVC)
. "$SCRIPT_DIR/find_persistent_root.sh"

# Convert HTTPS URL to SSH URL
https_to_ssh() {
    local url="$1"
    echo "$url" | sed 's|https://github.com/|git@github.com:|'
}

setup_workspace() {
    local PROJECT_NAME="$1"
    local GITHUB_URL="$2"
    local USE_SUBMODULES=""
    
    # Parse optional flags
    shift 2
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --submodules) USE_SUBMODULES="yes" ;;
            *) echo "Unknown option: $1" >&2; return 1 ;;
        esac
        shift
    done

    if [ -z "$PROJECT_NAME" ] || [ -z "$GITHUB_URL" ]; then
        echo "Usage: setup_workspace <project_name> <github_url> [--submodules]" >&2
        return 1
    fi

    local WORKSPACE_DIR="$HOME/$PROJECT_NAME"
    local SSH_URL
    SSH_URL=$(https_to_ssh "$GITHUB_URL")

    # Don't create workspace if it already exists
    if [ -d "$WORKSPACE_DIR" ]; then
        echo "✅ Workspace already exists at ~/$PROJECT_NAME"
        return 0
    fi

    local PERSISTENT_ROOT
    PERSISTENT_ROOT=$(find_persistent_root)

    echo "📦 Creating ephemeral workspace at ~/$PROJECT_NAME"

    # Case 1: No persistent storage - clone directly
    if [ -z "$PERSISTENT_ROOT" ]; then
        echo "   No persistent storage found - cloning from GitHub..."
        cd "$HOME"
        # No persistent storage means no SSH keys either (priv.sh, which
        # syncs them, only has something to sync when persistent storage is
        # present). rc_files' ~/.gitconfig rewrites every https://github.com/
        # URL to SSH (so an interactive container automatically uses the
        # synced personal key), which would otherwise defeat this exact
        # fallback. Bypass the global gitconfig for just this clone so the
        # plain-HTTPS path actually stays HTTPS.
        GIT_CONFIG_GLOBAL=/dev/null git clone "$GITHUB_URL" "$PROJECT_NAME"
        cd "$PROJECT_NAME"
        [ -n "$USE_SUBMODULES" ] && GIT_CONFIG_GLOBAL=/dev/null git submodule update --init
        git remote set-url origin "$SSH_URL"

    # Case 2: Persistent storage exists - use reference clone
    else
        local REFERENCE="$PERSISTENT_ROOT/.${PROJECT_NAME}-reference"
        echo "   Using persistent reference at $PERSISTENT_ROOT"

        # Create or update reference
        if [ ! -d "$REFERENCE/$PROJECT_NAME" ]; then
            echo "   Creating reference clone (one-time setup)..."
            mkdir -p "$REFERENCE"
            cd "$REFERENCE"
            git clone "$GITHUB_URL" "$PROJECT_NAME"
            cd "$PROJECT_NAME"
            [ -n "$USE_SUBMODULES" ] && git submodule update --init
            git remote set-url origin "$SSH_URL"
            echo "   ✅ Reference created"
        else
            echo "   Updating reference to latest..."
            cd "$REFERENCE/$PROJECT_NAME"
            if git fetch origin 2>/dev/null; then
                git reset --hard origin/main
                [ -n "$USE_SUBMODULES" ] && git submodule update --init
                echo "   ✅ Reference updated"
            else
                echo "   ℹ️  Could not fetch (SSH not set up yet), using existing reference"
            fi
        fi

        # Copy reference to working directory
        echo "   Copying from reference to ~/$PROJECT_NAME..."
        cd "$HOME"
        cp -r "$REFERENCE/$PROJECT_NAME" "$PROJECT_NAME"

        echo "   Updating git remotes..."
        cd "$WORKSPACE_DIR"
        git remote set-url origin "$SSH_URL"
    fi

    # Common setup
    cd "$WORKSPACE_DIR"

    # Print summary
    echo "✅ Ephemeral workspace created at ~/$PROJECT_NAME"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ Workspace Setup Complete!"
    echo "════════════════════════════════════════════════════════════════"
    if [ -n "$PERSISTENT_ROOT" ]; then
        echo "  Mode:      Using persistent reference"
        echo "  Reference: $PERSISTENT_ROOT/.${PROJECT_NAME}-reference/$PROJECT_NAME"
    else
        echo "  Mode:      Fully ephemeral (no persistent storage)"
    fi
    echo "  Workspace: ~/$PROJECT_NAME (ephemeral, pod-local)"
    echo ""
    echo "  💡 Remember: ~/$PROJECT_NAME is ephemeral and will be deleted when"
    echo "     the pod stops. Commit and push your changes regularly!"
    echo "════════════════════════════════════════════════════════════════"
}
