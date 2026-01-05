#!/bin/bash
# Generic workspace setup for ephemeral pods
#
# Usage: setup_workspace <project_name> <github_url> [--submodules]
#   project_name: Name of the project (e.g., "iree", "triton")
#   github_url:   GitHub clone URL (e.g., "https://github.com/iree-org/iree.git")
#   --submodules: Optional flag to initialize git submodules
#
# Project-specific setup should be added after calling setup_workspace in the wrapper script.

set -e

CURSOR_RULES_SOURCE="$HOME/rc_files/cursor/.cursor/rules"

# Setup Cursor AI assistant rules for a workspace
# Links rules based on their globs pattern: **/* = universal, otherwise match project name
setup_cursor_rules() {
    local pattern="$1"
    local workspace_dir="$2"

    if [ -z "$pattern" ] || [ -z "$workspace_dir" ]; then
        return 1
    fi

    if [ ! -d "$CURSOR_RULES_SOURCE" ]; then
        echo "   ℹ️  No Cursor rules source found"
        return 0
    fi

    local rules_dest="$workspace_dir/.cursor/rules"
    mkdir -p "$rules_dest"

    local count=0
    for rule in "$CURSOR_RULES_SOURCE"/*.mdc; do
        [ -e "$rule" ] || continue
        local rulename
        rulename=$(basename "$rule")

        # Check if rule applies: universal glob (**/*) or filename matches project
        if grep -q '^globs:.*\*\*/\*' "$rule" || [[ "$rulename" == *"$pattern"* ]]; then
            ln -sf "$rule" "$rules_dest/$rulename"
            echo "   Linked Cursor rule: $rulename"
            count=$((count + 1))
        fi
    done

    if [ "$count" -eq 0 ]; then
        echo "   ℹ️  No applicable Cursor rules found for '$pattern'"
    fi
}

# Find persistent storage root if available
# Looks for a mounted home directory at the root level by detecting home-like markers
find_persistent_root() {
    # Strategy 1: Try username-based paths first
    local username
    username=$(id -un 2>/dev/null || whoami 2>/dev/null || echo "")

    if [ -n "$username" ] && [ -d "/$username" ] && [ "/$username" != "/root" ] && [ "/$username" != "/home" ]; then
        echo "/$username"
        return 0
    fi

    # Strategy 2: Scan root-level directories for home-like markers
    # (mounted home dirs typically have .bashrc, .ssh, rc_files, scripts, etc.)
    for candidate in /*/; do
        candidate="${candidate%/}"  # Remove trailing slash

        # Skip system directories
        case "$candidate" in
            /bin|/boot|/dev|/etc|/home|/lib*|/media|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var|/data|/ffm|/tools|/snap)
                continue
                ;;
        esac

        # Check for home directory markers
        if [ -d "$candidate/.ssh" ] || [ -d "$candidate/rc_files" ] || [ -d "$candidate/scripts" ]; then
            echo "$candidate"
            return 0
        fi
    done

    echo ""
}

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
        git clone "$GITHUB_URL"
        cd "$PROJECT_NAME"
        [ -n "$USE_SUBMODULES" ] && git submodule update --init
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
            git clone "$GITHUB_URL"
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

    # Symlink Cursor rules
    echo "   Linking Cursor rules..."
    setup_cursor_rules "$PROJECT_NAME" "$WORKSPACE_DIR"

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

# If script is executed directly (not sourced), run with provided arguments
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    setup_workspace "$@"
fi
