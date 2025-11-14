#!/bin/bash
# Cleanup isolated workspace for this pod
# This script runs INSIDE the pod via kubectl exec

set -e

POD_NAME=$(hostname)
WORKSPACE="$HOME/workspace-$POD_NAME"

echo "🧹 Cleaning workspace for pod: $POD_NAME"

# Remove workspace directory
if [ -d "$WORKSPACE" ]; then
    echo "   Removing workspace: $WORKSPACE"
    rm -rf "$WORKSPACE"
    echo "   ✅ Workspace removed"
else
    echo "   ℹ️  No workspace found at $WORKSPACE"
fi

# Remove symlink if it points to this workspace
if [ -L "$HOME/iree" ]; then
    LINK_TARGET=$(readlink "$HOME/iree")
    if [[ "$LINK_TARGET" == *"$POD_NAME"* ]]; then
        echo "   Removing symlink: ~/iree -> $LINK_TARGET"
        rm "$HOME/iree"
        echo "   ✅ Symlink removed"
    else
        echo "   ℹ️  Symlink points to different workspace, leaving it"
    fi
elif [ -e "$HOME/iree" ]; then
    echo "   ⚠️  ~/iree exists but is not a symlink (leaving it alone)"
fi

echo "✅ Cleanup complete for $POD_NAME"

