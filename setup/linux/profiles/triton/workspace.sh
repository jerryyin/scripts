#!/bin/bash
# Setup ephemeral Triton workspace
set -e

WORKSPACE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_DIR="${DEVENV_WORKSPACE_ROOT:-$HOME}/triton"

# Skip if workspace already exists
if [ -d "$TRITON_DIR" ]; then
    echo "✅ Workspace already exists at $TRITON_DIR"
    exit 0
fi

source "$WORKSPACE_SCRIPT_DIR/../../lib/git_workspace.sh"
setup_workspace "triton" "https://github.com/triton-lang/triton.git"

# Triton-specific setup (runs after clone, in $HOME/triton)
cd "$TRITON_DIR"

TRITON_AMD_URL="${TRITON_AMD_URL:-git@github-e:AMD-Triton/triton-mi450.git}"
git remote set-url amd "$TRITON_AMD_URL" 2>/dev/null || git remote add amd "$TRITON_AMD_URL"

# Install pre-commit git hooks so they run automatically on git commit
if command -v pre-commit &> /dev/null && [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
fi

# Setup gtags for MLIR code navigation
if command -v gtags &> /dev/null && [ -d "lib" ]; then
    echo "   Setting up gtags..."
    find lib -type f -print > gtags.files
    gtags
fi
