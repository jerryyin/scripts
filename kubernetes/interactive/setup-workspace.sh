#!/bin/bash
# Setup ephemeral IREE workspace for this pod
# This script runs INSIDE the pod via SSH
#
# Architecture:
#   - If PVC exists: Use persistent reference at /zyin/.iree-reference/iree, copy to ~/iree
#   - If no PVC: Clone directly to ~/iree
#   - Working workspace: ~/iree (ephemeral, deleted when pod dies)

set -e

POD_NAME=$(hostname)

# Check if we have a persistent mount (e.g., /zyin, /$USER)
PERSISTENT_ROOT=""
for candidate in "/$USER" "/$(whoami)" "/zyin"; do
    if [ -d "$candidate" ] && [ "$candidate" != "/root" ] && [ "$candidate" != "/home" ]; then
        PERSISTENT_ROOT="$candidate"
        break
    fi
done

# Don't create workspace if it already exists
if [ -d "$HOME/iree" ]; then
    echo "✅ Workspace already exists at ~/iree"
    exit 0
fi

echo "📦 Creating ephemeral workspace at ~/iree"

# Case 1: No persistent storage - clone directly
if [ -z "$PERSISTENT_ROOT" ]; then
    echo "   No persistent storage found - cloning from GitHub..."
    cd "$HOME"
    git clone https://github.com/iree-org/iree.git
    cd iree
    git submodule update --init
    git remote set-url origin git@github.com:iree-org/iree.git
    
# Case 2: Persistent storage exists - use reference clone
else
    REFERENCE="$PERSISTENT_ROOT/.iree-reference"
    echo "   Using persistent reference at $PERSISTENT_ROOT"
    
    # Create or update reference
    if [ ! -d "$REFERENCE/iree" ]; then
        echo "   Creating reference clone (one-time setup)..."
        mkdir -p "$REFERENCE"
        cd "$REFERENCE"
        git clone https://github.com/iree-org/iree.git
        cd iree
        git submodule update --init
        git remote set-url origin git@github.com:iree-org/iree.git
        echo "   ✅ Reference created"
    else
        echo "   Updating reference to latest..."
        cd "$REFERENCE/iree"
        # Try to update, but don't fail if SSH isn't set up yet
        if git fetch origin 2>/dev/null; then
            git reset --hard origin/main
            git submodule update --init
            echo "   ✅ Reference updated"
        else
            echo "   ℹ️  Could not fetch (SSH not set up yet), using existing reference"
        fi
    fi
    
    # Copy reference to working directory (fast!)
    echo "   Copying from reference to ~/iree..."
    cd "$HOME"
    cp -r "$REFERENCE/iree" iree
    
    echo "   Updating git remotes..."
    cd "$HOME/iree"
    git remote set-url origin git@github.com:iree-org/iree.git
fi

# Common setup for both cases
cd "$HOME/iree"

# Setup llvm-project remotes
echo "   Setting up llvm-project remotes..."
cd third_party/llvm-project
git remote set-url origin git@github.com:iree-org/llvm-project.git 2>/dev/null || true
git remote add upstream git@github.com:llvm/llvm-project.git 2>/dev/null || true
cd "$HOME/iree"

# Setup CMakePresets.json symlink if available
if [ -f "$HOME/scripts/iree/CMakePresets.json" ]; then
    echo "   Linking CMakePresets.json..."
    ln -sf "$HOME/scripts/iree/CMakePresets.json" CMakePresets.json
fi

# Install Python requirements
if [ -f runtime/bindings/python/iree/runtime/build_requirements.txt ]; then
    echo "   Installing IREE Python build requirements..."
    python -m pip install -q -r runtime/bindings/python/iree/runtime/build_requirements.txt
fi

echo "✅ Ephemeral workspace created at ~/iree"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Workspace Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
if [ -n "$PERSISTENT_ROOT" ]; then
    echo "  Mode:      Using persistent reference"
    echo "  Reference: $PERSISTENT_ROOT/.iree-reference/iree"
else
    echo "  Mode:      Fully ephemeral (no persistent storage)"
fi
echo "  Workspace: ~/iree (ephemeral, pod-local)"
echo ""
echo "  💡 Remember: ~/iree is ephemeral and will be deleted when"
echo "     the pod stops. Commit and push your changes regularly!"
echo "════════════════════════════════════════════════════════════════"
