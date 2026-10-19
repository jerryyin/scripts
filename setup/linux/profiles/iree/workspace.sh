#!/bin/bash
# Setup ephemeral IREE workspace
set -e

WORKSPACE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "$WORKSPACE_SCRIPT_DIR/../../../.." && pwd)"
IREE_DIR="${DEVENV_WORKSPACE_ROOT:-$HOME}/iree"

# Skip if workspace already exists
if [ -d "$IREE_DIR" ]; then
    echo "✅ Workspace already exists at $IREE_DIR"
    exit 0
fi

source "$WORKSPACE_SCRIPT_DIR/../../lib/git_workspace.sh"
setup_workspace "iree" "https://github.com/iree-org/iree.git" --submodules

# IREE-specific setup (runs after clone, in $HOME/iree)
cd "$IREE_DIR"

# Setup llvm-project remotes
echo "   Setting up llvm-project remotes..."
cd third_party/llvm-project
git remote set-url origin git@github.com:iree-org/llvm-project.git 2>/dev/null || true
git remote add upstream git@github.com:llvm/llvm-project.git 2>/dev/null || true
cd "$IREE_DIR"

# Setup CMakePresets.json symlink if available
if [ -f "$SCRIPTS_ROOT/iree/CMakePresets.json" ]; then
    echo "   Linking CMakePresets.json..."
    ln -sf "$SCRIPTS_ROOT/iree/CMakePresets.json" CMakePresets.json
fi

# Setup LLVM CMakeUserPresets.json for standalone MLIR builds
if [ -f "$SCRIPTS_ROOT/llvm/CMakeUserPresets.json" ]; then
    echo "   Linking llvm-project CMakeUserPresets.json..."
    ln -sf "$SCRIPTS_ROOT/llvm/CMakeUserPresets.json" third_party/llvm-project/llvm/CMakeUserPresets.json
fi

# Install Python requirements
if [ -f runtime/bindings/python/iree/runtime/build_requirements.txt ]; then
    echo "   Installing IREE Python build requirements..."
    python -m pip install -q -r runtime/bindings/python/iree/runtime/build_requirements.txt
fi

# Setup pre-commit hooks for automatic formatting checks on commit
echo "   Installing pre-commit hooks..."
pre-commit install
