#!/bin/bash
# Setup ephemeral IREE workspace for this pod
# This script runs INSIDE the pod via SSH
#
# Architecture:
#   - Reference clone: /zyin/.iree-reference/iree (persistent, shared)
#   - Working clone:   ~/iree (ephemeral, fast)
#   - When pod dies, ~/iree is automatically deleted (emptyDir)

set -e

POD_NAME=$(hostname)

# Determine location for reference clone
# Look for user-specific mount point (e.g., /zyin, /username)
PERSISTENT_ROOT=""
for candidate in "/$USER" "/$(whoami)" "/zyin"; do
    if [ -d "$candidate" ] && [ "$candidate" != "/root" ] && [ "$candidate" != "/home" ]; then
        PERSISTENT_ROOT="$candidate"
        break
    fi
done

if [ -n "$PERSISTENT_ROOT" ]; then
    REFERENCE="$PERSISTENT_ROOT/.iree-reference"
    echo "📦 Using persistent reference at $PERSISTENT_ROOT"
else
    # Fully ephemeral mode (no PVC available)
    REFERENCE="$HOME/.iree-reference"
    echo "📦 Using ephemeral reference in home directory"
fi

# Create or update reference IREE (shared across all your pods)
if [ ! -d "$REFERENCE/iree" ]; then
    echo "📦 Creating reference IREE repository (one-time setup)..."
    mkdir -p "$REFERENCE"
    cd "$REFERENCE"
    git clone https://github.com/iree-org/iree.git
    cd iree
    git submodule update --init
    git remote set-url origin git@github.com:iree-org/iree.git
    echo "✅ Reference IREE created at $REFERENCE/iree"
else
    echo "📦 Updating reference IREE to latest..."
    cd "$REFERENCE/iree"
    # Try to fetch, but don't fail if SSH isn't set up yet
    if git fetch origin 2>/dev/null; then
        git reset --hard origin/main
        git submodule update --init
        echo "✅ Reference IREE updated"
    else
        echo "   ℹ️  Could not fetch (SSH not set up yet), using existing reference"
    fi
fi

# Clone directly to ~/iree (ephemeral workspace)
if [ ! -d "$HOME/iree" ]; then
    echo "📦 Creating ephemeral workspace at ~/iree"
    
    # Clone from reference (fast, independent copy)
    echo "   Cloning from reference..."
    cd "$HOME"
    git clone --reference "$REFERENCE/iree" --dissociate "$REFERENCE/iree" iree
    
    cd "$HOME/iree"
    git remote set-url origin git@github.com:iree-org/iree.git
    git submodule update --init
    
    # Setup llvm-project remotes
    echo "   Setting up llvm-project remotes..."
    cd "$HOME/iree/third_party/llvm-project"
    git remote set-url origin git@github.com:iree-org/llvm-project.git 2>/dev/null || true
    git remote add upstream git@github.com:llvm/llvm-project.git 2>/dev/null || true
    cd "$HOME/iree"
    
    # Setup CMakePresets.json symlink if available
    if [ -f "$HOME/scripts/iree/CMakePresets.json" ]; then
        echo "   Linking CMakePresets.json..."
        ln -sf "$HOME/scripts/iree/CMakePresets.json" "$HOME/iree/CMakePresets.json"
    fi
    
    # Install Python requirements
    if [ -f "$HOME/iree/runtime/bindings/python/iree/runtime/build_requirements.txt" ]; then
        echo "   Installing IREE Python build requirements..."
        python -m pip install -q -r "$HOME/iree/runtime/bindings/python/iree/runtime/build_requirements.txt"
    fi
    
    echo "✅ Ephemeral workspace created at ~/iree"
else
    echo "✅ Workspace already exists at ~/iree"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Workspace Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
echo "  Reference: $REFERENCE/iree (persistent, shared)"
echo "  Workspace: ~/iree (ephemeral, pod-local)"
echo ""
echo "  💡 Remember: ~/iree is ephemeral and will be deleted when"
echo "     the pod stops. Commit and push your changes regularly!"
echo "════════════════════════════════════════════════════════════════"

