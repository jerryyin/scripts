#!/bin/bash
# Kubernetes Configuration Setup Script
# This script creates symlinks from standard locations to version-controlled files
# Run this script on a new system to set up your Kubernetes configuration

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Setting up Kubernetes configuration from: $SCRIPT_DIR"
echo ""

# Function to create symlink with backup
create_symlink() {
    local target="$1"
    local link_name="$2"
    local link_dir=$(dirname "$link_name")
    
    # Create directory if it doesn't exist
    if [ ! -d "$link_dir" ]; then
        echo "Creating directory: $link_dir"
        mkdir -p "$link_dir"
    fi
    
    # If link already exists and points to the right place, skip
    if [ -L "$link_name" ] && [ "$(readlink -f "$link_name")" = "$(readlink -f "$target")" ]; then
        echo "✓ Already linked: $link_name -> $target"
        return
    fi
    
    # If file/link exists but is wrong, back it up
    if [ -e "$link_name" ] || [ -L "$link_name" ]; then
        backup="${link_name}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "  Backing up existing file: $link_name -> $backup"
        mv "$link_name" "$backup"
    fi
    
    # Create the symlink
    echo "  Creating symlink: $link_name -> $target"
    ln -s "$target" "$link_name"
    echo "✓ Linked: $link_name"
}

# 1. Setup kubeswitch config
echo "1. Setting up kubeswitch configuration..."
create_symlink "$SCRIPT_DIR/kube-configs/switch-config.yaml" "$HOME/.kube/switch-config.yaml"
echo ""

# 2. Setup kubeconfig directory and files
echo "2. Setting up kubeconfig files..."
create_symlink "$SCRIPT_DIR/kube-configs/tw-tus1-bm-private-sso.conf" "$HOME/.kube/configs/tw-tus1-bm-private-sso.conf"
echo ""

# 3. Interactive pod configuration
echo "3. Interactive pod configuration..."
echo "   Files are located at: $SCRIPT_DIR/interactive/"
echo "   - config.json: Edit this to customize your setup"
echo "   - connect.sh: Run this to connect to interactive pod"
echo "   - stop.sh: Run this to cleanup pods"
echo ""

# 4. SSH config reminder
echo "4. SSH Configuration"
echo "   Don't forget to add the SSH config snippet to your ~/.ssh/config:"
echo "   See: $SCRIPT_DIR/interactive/ssh-config.txt"
echo ""
if ! grep -q "Host ossci" "$HOME/.ssh/config" 2>/dev/null; then
    echo "   Would you like to add it now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cat "$SCRIPT_DIR/interactive/ssh-config.txt" >> "$HOME/.ssh/config"
        echo "   ✓ SSH config snippet added to ~/.ssh/config"
    else
        echo "   ℹ You can manually add it later from ssh-config-snippet.txt"
    fi
else
    echo "   ✓ SSH config for 'ossci' already exists in ~/.ssh/config"
fi
echo ""

# Summary
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Configuration files are now linked from:"
echo "  $SCRIPT_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify kubectl, krew, oidc-login, and kubeswitch are installed"
echo "  2. Run 'switch' to select the cluster configuration"
echo "  3. Run 'kubectl get ns' to authenticate via Okta SSO"
echo "  4. Apply PVC: kubectl apply -f $SCRIPT_DIR/pvc/iree-dev-zyin-pvc.yaml"
echo "  5. Connect to pod: $SCRIPT_DIR/interactive/connect.sh"
echo ""
echo "For more details, see: $SCRIPT_DIR/README.md"

