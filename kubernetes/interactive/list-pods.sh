#!/bin/bash
# List all active interactive pods with connection information

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
PORT_MAPPING_FILE="$HOME/.kube/pod-port-mappings.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.json not found at $CONFIG_FILE"
    exit 1
fi

NAMESPACE=$(jq -r '.namespace' "$CONFIG_FILE")
DEFAULT_CLUSTER=$(jq -r '.default_cluster' "$CONFIG_FILE")
DEFAULT_KUBECONFIG="$HOME/.kube/configs/$DEFAULT_CLUSTER"

# Setup kubeconfig
if [ -n "${KUBECONFIG:-}" ] && [ -f "$KUBECONFIG" ]; then
    :
else
    export KUBECONFIG="$DEFAULT_KUBECONFIG"
fi

# Add krew to PATH if needed
if [ -d "$HOME/.krew/bin" ] && [[ ":$PATH:" != *":$HOME/.krew/bin:"* ]]; then
    export PATH="${HOME}/.krew/bin:$PATH"
fi

# Helper function to get SSH host alias for a pod
get_ssh_host_alias() {
    local pod_name="$1"
    local pod_suffix
    pod_suffix=$(echo "$pod_name" | sed -E 's/.*-iree-//')
    echo "ossci-$pod_suffix"
}

echo "🔍 Active Interactive Pods"
echo "================================================================"
echo ""

# Find all interactive pods for this user
PODS=$(kubectl get pods -n "$NAMESPACE" -o json | jq -r ".items[] | select(.metadata.name | startswith(\"${USER}-iree-\")) | select(.status.phase == \"Running\") | .metadata.name" 2>/dev/null || true)

if [ -z "$PODS" ]; then
    echo "No active pods found."
    echo ""
    echo "💡 To create a new pod, run: $SCRIPT_DIR/connect.sh"
    exit 0
fi

POD_ARRAY=($PODS)
POD_COUNT=${#POD_ARRAY[@]}

echo "Found $POD_COUNT active pod(s):"
echo ""

for pod in "${POD_ARRAY[@]}"; do
    # Get pod info
    AGE=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.startTime}' 2>/dev/null || echo "unknown")
    STATUS=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "unknown")

    # Get port and SSH alias
    SSH_ALIAS=$(get_ssh_host_alias "$pod")
    PORT="not assigned"
    if [ -f "$PORT_MAPPING_FILE" ]; then
        MAPPED_PORT=$(jq -r --arg pod "$pod" '.[$pod] // empty' "$PORT_MAPPING_FILE" 2>/dev/null || echo "")
        if [ -n "$MAPPED_PORT" ]; then
            PORT="$MAPPED_PORT"
        fi
    fi

    # Check if port-forward is running
    PID_FILE="/tmp/kubectl-port-forward-${USER}-${pod}.pid"
    PF_STATUS="not running"
    if [ -f "$PID_FILE" ]; then
        PF_PID=$(cat "$PID_FILE")
        if kill -0 "$PF_PID" 2>/dev/null; then
            PF_STATUS="running (PID: $PF_PID)"
        fi
    fi

    echo "📦 $pod"
    echo "   Status:        $STATUS"
    echo "   Started:       $AGE"
    echo "   SSH:           ssh $SSH_ALIAS"
    echo "   Port:          $PORT"
    echo "   Port-forward:  $PF_STATUS"
    echo ""
done

echo "================================================================"
echo ""
echo "Commands:"
echo "  Connect to pod:    $SCRIPT_DIR/connect.sh"
echo "  Stop pods:         $SCRIPT_DIR/stop.sh"
echo ""

