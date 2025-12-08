#!/bin/bash
# Setup ephemeral Triton workspace
set -e

# Skip if workspace already exists
if [ -d "$HOME/triton" ]; then
    echo "✅ Workspace already exists at ~/triton"
    exit 0
fi

source "$HOME/scripts/docker/workspace/base.sh"
setup_workspace "triton" "https://github.com/triton-lang/triton.git"

# Triton-specific setup (runs after clone, in $HOME/triton)
cd "$HOME/triton"

# Setup gtags for MLIR code navigation
if command -v gtags &> /dev/null && [ -d "lib" ]; then
    echo "   Setting up gtags..."
    find lib -type f -print > gtags.files
    gtags
fi
