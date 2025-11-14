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

echo "Stopping interactive pods and port-forwards..."
echo ""

# Find all port-forward PIDs
PID_FILES=$(ls /tmp/kubectl-port-forward-${USER}-*.pid 2>/dev/null || true)
if [ -n "$PID_FILES" ]; then
    echo "Killing port-forward processes..."
    for pid_file in $PID_FILES; do
        if [ -f "$pid_file" ]; then
            PID=$(cat "$pid_file")
            POD=$(basename "$pid_file" | sed "s/kubectl-port-forward-${USER}-//;s/.pid//")
            if kill -0 "$PID" 2>/dev/null; then
                echo "  - Stopping port-forward for $POD (PID: $PID)"
                kill "$PID" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
else
    echo "No port-forward processes found."
fi

echo ""

# Find all interactive pods for this user
PODS=$(kubectl get pods -n "$NAMESPACE" -o json | jq -r ".items[] | select(.metadata.name | startswith(\"${USER}-iree-\")) | .metadata.name" 2>/dev/null || true)

if [ -n "$PODS" ]; then
    echo "Found interactive pod(s):"
    echo "$PODS" | sed 's/^/  - /'
    echo ""
    read -p "Delete all these pods? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for pod in $PODS; do
            echo "  Deleting $pod..."
            kubectl delete pod "$pod" -n "$NAMESPACE" --grace-period=30
        done
        echo "✅ Pods deleted"
    else
        echo "Pods not deleted."
        echo ""
        echo "To delete manually:"
        for pod in $PODS; do
            echo "  kubectl delete pod $pod -n $NAMESPACE"
        done
    fi
else
    echo "No interactive pods found."
fi

echo ""
echo "Done."

