#!/bin/bash
# Pre-pull ROCm image on all cluster nodes using a DaemonSet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/configs/tw-tus1-bm-private-sso.conf}"
export KUBECONFIG
export PATH="${KREW_ROOT:-$HOME/.krew}/bin:$PATH"

DAEMONSET_FILE="${SCRIPT_DIR}/image-prepull-daemonset.yml"
NAMESPACE="iree-dev"

# Check if DaemonSet already exists
if kubectl get daemonset rocm-image-prepull -n "$NAMESPACE" >/dev/null 2>&1; then
    echo "🔍 DaemonSet already exists. Checking status..."
    kubectl get daemonset rocm-image-prepull -n "$NAMESPACE"
    echo ""
    kubectl get pods -n "$NAMESPACE" -l name=rocm-image-prepull -o wide
    echo ""
    echo "To delete and re-create:"
    echo "  kubectl delete daemonset rocm-image-prepull -n $NAMESPACE"
    exit 0
fi

echo "🚀 Creating DaemonSet to pre-pull ROCm image on all nodes..."
echo ""
kubectl apply -f "$DAEMONSET_FILE"

echo ""
echo "⏳ Waiting for DaemonSet to start on all nodes..."
echo "   (This will take 10-15 minutes per node for the first image pull)"
echo ""

# Wait a moment for pods to start
sleep 3

# Show status
echo "📊 DaemonSet status:"
kubectl get daemonset rocm-image-prepull -n "$NAMESPACE"
echo ""
echo "📦 Pods (one per node):"
kubectl get pods -n "$NAMESPACE" -l name=rocm-image-prepull -o wide
echo ""
echo "💡 Commands:"
echo "   Watch progress:  kubectl get pods -n $NAMESPACE -l name=rocm-image-prepull -w"
echo "   Check status:    kubectl get daemonset rocm-image-prepull -n $NAMESPACE"
echo ""
echo "⏰ Wait for completion and auto-cleanup? (y/N): "
read -r -n 1 AUTO_CLEANUP
echo ""

if [[ $AUTO_CLEANUP =~ ^[Yy]$ ]]; then
    echo ""
    echo "⏳ Waiting for all nodes to pull the image..."
    echo "   (This may take 10-20 minutes depending on cluster size)"
    echo ""
    
    # Wait for DaemonSet to be ready
    kubectl rollout status daemonset/rocm-image-prepull -n "$NAMESPACE" --timeout=30m || {
        echo "⚠️  Timeout or error waiting for DaemonSet"
        echo "   You can check status with: kubectl get daemonset rocm-image-prepull -n $NAMESPACE"
        echo "   And cleanup manually with: kubectl delete daemonset rocm-image-prepull -n $NAMESPACE"
        exit 1
    }
    
    echo ""
    echo "✅ All nodes have successfully pulled the image!"
    echo ""
    echo "🧹 Cleaning up DaemonSet..."
    kubectl delete daemonset rocm-image-prepull -n "$NAMESPACE"
    echo "✅ Cleanup complete!"
    echo ""
    echo "🚀 All cluster nodes now have the ROCm image cached."
    echo "   Your pods will start much faster from now on!"
else
    echo ""
    echo "⚠️  REMINDER: Don't forget to cleanup the DaemonSet when done!"
    echo ""
    echo "   Check if complete:  kubectl get daemonset rocm-image-prepull -n $NAMESPACE"
    echo "   Cleanup command:    kubectl delete daemonset rocm-image-prepull -n $NAMESPACE"
    echo ""
    echo "   (Look for DESIRED == READY to know when it's done)"
fi

