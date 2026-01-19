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
echo "  1. Install system packages, setup dotfiles, clone repos"
echo "  2. Container init (SSH, hostname, credential sync)"
echo "  3. Install IREE dependencies (cmake, python packages)"
echo "  4. Setup IREE workspace"
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

# Run env/min.sh (installs system packages, sets up dotfiles)
echo "📦 Step 1/4: Running env/min.sh (system packages + dotfiles)..."
cd "$HOME"
bash scripts/docker/env/min.sh || {
    echo "❌ env/min.sh failed. You can retry manually:"
    echo "   bash ~/scripts/docker/env/min.sh"
    exit 1
}

# Container initialization (SSH, hostname, credentials)
# Uses same priv.sh as Docker for unified handling
echo ""
echo "🔧 Step 2/4: Container initialization (priv.sh)..."
cd "$HOME"
if [ -f scripts/docker/env/priv.sh ]; then
    bash scripts/docker/env/priv.sh || {
        echo "⚠️  priv.sh failed (non-fatal). You can retry manually:"
        echo "   bash ~/scripts/docker/env/priv.sh"
    }
else
    echo "   ℹ️  priv.sh not found - skipping container init"
fi

# Run env/iree.sh (installs cmake, python packages)
echo ""
echo "📦 Step 3/4: Running env/iree.sh (IREE dependencies)..."
cd "$HOME"
bash scripts/docker/env/iree.sh || {
    echo "❌ env/iree.sh failed. You can retry manually:"
    echo "   bash ~/scripts/docker/env/iree.sh"
    exit 1
}

# Setup isolated workspace for this pod
echo ""
echo "📦 Step 4/4: Setting up isolated IREE workspace..."
cd "$HOME"
bash ~/scripts/docker/workspace/iree.sh || {
    echo "❌ workspace/iree.sh failed. You can retry manually:"
    echo "   bash ~/scripts/docker/workspace/iree.sh"
    exit 1
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Pod Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""

