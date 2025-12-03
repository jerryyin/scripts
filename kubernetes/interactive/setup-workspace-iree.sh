#!/bin/bash
# Setup ephemeral IREE workspace
set -e

# Skip if workspace already exists
if [ -d "$HOME/iree" ]; then
    echo "✅ Workspace already exists at ~/iree"
    exit 0
fi

source "$HOME/scripts/kubernetes/interactive/setup-workspace.sh"
setup_workspace "iree" "https://github.com/iree-org/iree.git" --submodules

# IREE-specific setup (runs after clone, in $HOME/iree)
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
