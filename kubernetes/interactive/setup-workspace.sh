#!/bin/bash
# Setup isolated workspace for this pod
# This script runs INSIDE the pod via SSH

set -e

POD_NAME=$(hostname)
WORKSPACE="$HOME/workspace-$POD_NAME"
REFERENCE="$HOME/.iree-reference"

# Create or update reference IREE
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

# Create workspace for this pod if it doesn't exist
if [ ! -d "$WORKSPACE/iree" ]; then
    echo "📦 Creating isolated workspace for pod: $POD_NAME"
    mkdir -p "$WORKSPACE"
    
    # Copy from reference (faster than cloning)
    echo "   Copying from reference..."
    cp -r "$REFERENCE/iree" "$WORKSPACE/iree"
    
    # Update to latest
    echo "   Updating to latest..."
    cd "$WORKSPACE/iree"
    git fetch origin
    git reset --hard origin/main
    git submodule update --init
    
    # Setup llvm-project remotes
    echo "   Setting up llvm-project remotes..."
    cd "$WORKSPACE/iree/third_party/llvm-project"
    git remote set-url origin git@github.com:iree-org/llvm-project.git 2>/dev/null || true
    git remote add upstream git@github.com:llvm/llvm-project.git 2>/dev/null || true
    cd "$WORKSPACE/iree"
    
    # Setup CMakePresets.json symlink if available
    if [ -f "$HOME/scripts/iree/CMakePresets.json" ]; then
        echo "   Linking CMakePresets.json..."
        ln -sf "$HOME/scripts/iree/CMakePresets.json" "$WORKSPACE/iree/CMakePresets.json"
    fi
    
    # Install Python requirements for this IREE workspace
    if [ -f "$WORKSPACE/iree/runtime/bindings/python/iree/runtime/build_requirements.txt" ]; then
        echo "   Installing IREE Python build requirements..."
        python -m pip install -q -r "$WORKSPACE/iree/runtime/bindings/python/iree/runtime/build_requirements.txt"
    fi
    
    echo "✅ Workspace created at $WORKSPACE/iree"
else
    echo "✅ Using existing workspace: $WORKSPACE/iree"
fi

# Create/update symlink ~/iree -> workspace
if [ -L "$HOME/iree" ]; then
    # Remove old symlink
    rm "$HOME/iree"
elif [ -e "$HOME/iree" ]; then
    # Backup if it's a real directory (shouldn't happen)
    echo "⚠️  Found real ~/iree directory, backing up..."
    mv "$HOME/iree" "$HOME/iree.backup.$(date +%s)"
fi

ln -s "$WORKSPACE/iree" "$HOME/iree"
echo "✅ Symlink created: ~/iree -> $WORKSPACE/iree"

# Create convenience alias
echo "export IREE_WORKSPACE=\"$WORKSPACE/iree\"" > ~/.workspace_env

echo "✅ Workspace setup complete!"

