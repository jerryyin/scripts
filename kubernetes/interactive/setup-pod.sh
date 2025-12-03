#!/bin/bash
# Setup script for Kubernetes interactive pods
# This script runs INSIDE the pod (locally), not remotely via SSH
# Handles system packages, dotfiles, IREE dependencies, and workspace setup
#
# NOTE: This script runs setup unconditionally when invoked.
#       The decision about whether to run setup should be made by connect.sh

set -euo pipefail

echo "════════════════════════════════════════════════════════════════"
echo "  Pod Setup (Running Locally)"
echo "════════════════════════════════════════════════════════════════"
echo "  Hostname:  $(hostname)"
echo "  User:      $(whoami)"
echo "  PWD:       $PWD"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "This will:"
echo "  1. Install system packages (git, zsh, tmux, neovim, cmake, etc.)"
echo "  2. Setup dotfiles"
echo "  3. Clone repos"
echo "  4. Install Python packages"
echo "  5. Setup IREE workspace"
echo ""
echo "This may take 10-15 minutes..."
echo ""

# Verify scripts directory exists
if [ ! -d "$HOME/scripts" ]; then
    echo "❌ Error: ~/scripts directory not found"
    echo "   Scripts should have been copied by connect.sh"
    echo "   Please ensure scripts are available in the pod"
    exit 1
fi

# Run init_min.sh (installs system packages, sets up dotfiles)
echo "📦 Step 1/3: Running init_min.sh (system packages + dotfiles)..."
cd "$HOME"
bash scripts/docker/init_min.sh || {
    echo "❌ init_min.sh failed. You can retry manually:"
    echo "   bash ~/scripts/docker/init_min.sh"
    exit 1
}

# Run init_iree.sh (installs cmake, python packages)
echo ""
echo "📦 Step 2/3: Running init_iree.sh (IREE dependencies)..."
cd "$HOME"
bash scripts/docker/init_iree.sh || {
    echo "❌ init_iree.sh failed. You can retry manually:"
    echo "   bash ~/scripts/docker/init_iree.sh"
    exit 1
}

# Setup isolated workspace for this pod
echo ""
echo "📦 Step 3/3: Setting up isolated IREE workspace..."
cd "$HOME"
bash ~/scripts/kubernetes/interactive/setup-workspace-iree.sh || {
    echo "❌ setup-workspace-iree.sh failed. You can retry manually:"
    echo "   bash ~/scripts/kubernetes/interactive/setup-workspace-iree.sh"
    exit 1
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Pod Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""

