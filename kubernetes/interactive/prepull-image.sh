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
echo "💡 To watch progress:"
echo "   kubectl get pods -n $NAMESPACE -l name=rocm-image-prepull -w"
echo ""
echo "💡 To check logs on a specific pod:"
echo "   kubectl logs -n $NAMESPACE <pod-name>"
echo ""
echo "💡 Once all pods are Running, the image is cached on all nodes!"
echo ""
echo "🧹 To cleanup after image pull completes:"
echo "   kubectl delete daemonset rocm-image-prepull -n $NAMESPACE"

