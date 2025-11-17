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

# No SSH config management - use direct port-based SSH

# Display string for SSH command (for user reference)
SSH_CMD_PREFIX="ssh -i ~/.ssh/id_rsa"

echo "🔍 Interactive Pods (All States)"
echo "================================================================"
echo ""

# Find all interactive pods for this user (including non-Running states for debugging)
PODS=$(kubectl get pods -n "$NAMESPACE" -o json | jq -r ".items[] | select(.metadata.name | startswith(\"${USER}-iree-\")) | .metadata.name" 2>/dev/null || true)

if [ -z "$PODS" ]; then
    echo "No pods found."
    echo ""
    echo "💡 To create a new pod, run: $SCRIPT_DIR/connect.sh"
    exit 0
fi

POD_ARRAY=($PODS)
POD_COUNT=${#POD_ARRAY[@]}

echo "Found $POD_COUNT pod(s):"
echo ""

for pod in "${POD_ARRAY[@]}"; do
    # Get pod info
    AGE=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.startTime}' 2>/dev/null || echo "unknown")
    STATUS=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "unknown")
    NODE=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "unknown")

    # Get container state
    CONTAINER_STATE=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null | jq -r 'keys[0]' 2>/dev/null || echo "unknown")

    # If container is waiting, get the reason
    WAIT_REASON=""
    if [ "$CONTAINER_STATE" = "waiting" ]; then
        WAIT_REASON=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || echo "")
    fi

    # Get pod conditions (Ready status)
    READY_CONDITION=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "Unknown")
    CONTAINERS_READY=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="ContainersReady")].status}' 2>/dev/null || echo "Unknown")

    # Get port
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

    # Display pod information
    echo "📦 $pod"
    echo "   Status:        $STATUS"

    # Show container state details
    if [ "$CONTAINER_STATE" = "running" ]; then
        STARTED_AT=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].state.running.startedAt}' 2>/dev/null || echo "")
        echo "   Container:     Running (started: $STARTED_AT)"
    elif [ "$CONTAINER_STATE" = "waiting" ]; then
        echo "   Container:     Waiting${WAIT_REASON:+ ($WAIT_REASON)}"
    elif [ "$CONTAINER_STATE" = "terminated" ]; then
        echo "   Container:     Terminated"
    else
        echo "   Container:     $CONTAINER_STATE"
    fi

    # Show conditions
    echo "   Conditions:    Ready=$READY_CONDITION, ContainersReady=$CONTAINERS_READY"
    echo "   Node:          $NODE"
    echo "   Age:           $AGE"
    echo "   SSH:           $SSH_CMD_PREFIX -p $PORT ossci@localhost"
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

