#!/bin/bash
# Stop interactive pods and port-forwards

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.json not found at $CONFIG_FILE"
    exit 1
fi

NAMESPACE=$(jq -r '.namespace' "$CONFIG_FILE")
DEFAULT_CLUSTER=$(jq -r '.default_cluster' "$CONFIG_FILE")
DEFAULT_KUBECONFIG="$HOME/.kube/configs/$DEFAULT_CLUSTER"

# Setup kubeconfig
if [ -n "${KUBECONFIG:-}" ] && [ -f "$KUBECONFIG" ]; then
    # Use existing KUBECONFIG if set
    :
else
    # Use default cluster from config
    export KUBECONFIG="$DEFAULT_KUBECONFIG"
fi

# Add krew to PATH if needed
if [ -d "$HOME/.krew/bin" ] && [[ ":$PATH:" != *":$HOME/.krew/bin:"* ]]; then
    export PATH="${HOME}/.krew/bin:$PATH"
fi

echo "Stopping interactive pods and port-forwards..."
echo ""

# Find all port-forward PIDs
PID_FILES=$(ls /tmp/kubectl-port-forward-${USER}-*.pid 2>/dev/null || true)
if [ -n "$PID_FILES" ]; then
    echo "🔌 Port-forward processes:"
    for pid_file in $PID_FILES; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            POD=$(basename "$pid_file" | sed "s/kubectl-port-forward-${USER}-//;s/.pid//")
            if kill -0 "$PID" 2>/dev/null; then
                echo "  - $POD (PID: $PID)"
            fi
        fi
    done
    echo ""
    read -p "Kill all port-forwards? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for pid_file in $PID_FILES; do
            if [ -f "$pid_file" ]; then
                PID=$(cat "$pid_file")
                if kill -0 "$PID" 2>/dev/null; then
                    kill "$PID" 2>/dev/null || true
                fi
                rm -f "$pid_file"
            fi
        done
        echo "✓ Port-forwards killed"
    else
        echo "Port-forwards not killed"
    fi
else
    echo "No port-forward processes found."
fi

echo ""

# Find all interactive pods for this user
PODS=$(kubectl get pods -n "$NAMESPACE" -o json | jq -r ".items[] | select(.metadata.name | startswith(\"${USER}-iree-\")) | .metadata.name" 2>/dev/null || true)

if [ -n "$PODS" ]; then
    POD_ARRAY=($PODS)
    POD_COUNT=${#POD_ARRAY[@]}
    
    echo "📦 Found $POD_COUNT pod(s):"
    echo ""
    
    # Display pods with numbers
    for i in "${!POD_ARRAY[@]}"; do
        POD=${POD_ARRAY[$i]}
        # Get pod age and status
        AGE=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.startTime}' 2>/dev/null || echo "unknown")
        STATUS=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "unknown")
        echo "  $((i+1)). $POD"
        echo "     Status: $STATUS | Started: $AGE"
    done
    
    echo ""
    echo "Options:"
    echo "  a       - Delete ALL pods"
    echo "  1-$POD_COUNT   - Delete specific pod by number"
    echo "  1,3,5   - Delete multiple pods (comma-separated)"
    echo "  q       - Quit without deleting"
    echo ""
    read -p "Choose pods to delete: " -r CHOICE
    
    if [[ "$CHOICE" =~ ^[Qq]$ ]]; then
        echo "No pods deleted."
        exit 0
    elif [[ "$CHOICE" =~ ^[Aa]$ ]]; then
        # Delete all pods
        echo ""
        echo "Deleting all $POD_COUNT pods..."
        for pod in "${POD_ARRAY[@]}"; do
            echo "  Deleting $pod..."
            kubectl delete pod "$pod" -n "$NAMESPACE" --grace-period=30
        done
        echo "✅ All pods deleted"
    else
        # Parse selection (handles single numbers or comma-separated)
        IFS=',' read -ra SELECTIONS <<< "$CHOICE"
        DELETED=0
        echo ""
        for sel in "${SELECTIONS[@]}"; do
            # Trim whitespace
            sel=$(echo "$sel" | xargs)
            # Check if valid number
            if [[ "$sel" =~ ^[0-9]+$ ]] && [ "$sel" -ge 1 ] && [ "$sel" -le "$POD_COUNT" ]; then
                POD_INDEX=$((sel-1))
                POD_TO_DELETE=${POD_ARRAY[$POD_INDEX]}
                echo "  Deleting $POD_TO_DELETE..."
                kubectl delete pod "$POD_TO_DELETE" -n "$NAMESPACE" --grace-period=30
                DELETED=$((DELETED+1))
            else
                echo "  ⚠ Invalid selection: $sel (skipping)"
            fi
        done
        if [ $DELETED -gt 0 ]; then
            echo "✅ Deleted $DELETED pod(s)"
        else
            echo "No pods deleted."
        fi
    fi
else
    echo "No interactive pods found."
fi

echo ""
echo "Done."
