#!/bin/bash
set -euo pipefail

# Default values
DATE=$(date +%Y%m%d)
TIME=$(date +%H%M%S)
POD_NAME="${USER}-iree-${DATE}-${TIME}"
LOCAL_PORT="8000"
REMOTE_PORT="9000"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_YAML="${SCRIPT_DIR}/pod-temp.yml"
CONFIG_FILE="${SCRIPT_DIR}/config.json"

MODE=${1:-ssh}

if [[ "$MODE" != "web" && "$MODE" != "ssh" ]]; then
  echo "Usage: $0 <web|ssh>"
  echo "  web : run browser-based code-server"
  echo "  ssh : run SSH-accessible interactive pod (default)"
  exit 1
fi

# Read config
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.json not found at $CONFIG_FILE"
    exit 1
fi

DEFAULT_CLUSTER=$(jq -r '.default_cluster // "tw-tus1-bm-private-sso.conf"' "$CONFIG_FILE")
DEFAULT_KUBECONFIG="$HOME/.kube/configs/$DEFAULT_CLUSTER"

# Auto-setup: Set kubeconfig and check authentication
echo "🔧 Auto-setup: Checking Kubernetes connection..."

# Check if KUBECONFIG is already set and valid
if [ -n "${KUBECONFIG:-}" ] && [ -f "$KUBECONFIG" ]; then
    echo "✓ Using existing KUBECONFIG: $KUBECONFIG"
else
    # Use default cluster from config
    export KUBECONFIG="$DEFAULT_KUBECONFIG"
    echo "✓ Kubeconfig set to default: $DEFAULT_KUBECONFIG"
fi

# Add krew to PATH if needed
if [ -d "$HOME/.krew/bin" ] && [[ ":$PATH:" != *":$HOME/.krew/bin:"* ]]; then
    export PATH="${HOME}/.krew/bin:$PATH"
fi

# Check authentication
if ! kubectl get ns >/dev/null 2>&1; then
    echo ""
    echo "⚠ Not authenticated with Kubernetes cluster"
    echo "🔐 Initiating Okta SSO authentication..."
    echo ""
    echo "This will open your browser at http://localhost:8000"
    echo "Please complete the Okta sign-in in your browser."
    echo ""
    
    # Trigger authentication (this will open the browser)
    if kubectl get ns >/dev/null 2>&1; then
        echo "✓ Authentication successful!"
        echo ""
    else
        echo ""
        echo "❌ Authentication failed or was cancelled"
        echo ""
        echo "Please try again by running this script again."
        exit 1
    fi
fi

echo "✓ Authenticated with Kubernetes cluster"
echo ""

if [[ "$MODE" != "web" && "$MODE" != "ssh" ]]; then
  echo "Usage: $0 <web|ssh>"
  echo "  web : run browser-based code-server"
  echo "  ssh : run SSH-accessible interactive pod (default)"
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.json not found at $CONFIG_FILE"
    exit 1
fi

# Read config
NAMESPACE=$(jq -r '.namespace' "$CONFIG_FILE")
PVC_CLAIM_NAME=$(jq -r '.pvc' "$CONFIG_FILE")
PUB_KEY_PATH=$(jq -r '.public_ssh_key_path' "$CONFIG_FILE")

YAML_TEMPLATE="${SCRIPT_DIR}/pod-${MODE}.yml"

# Check for existing pods
echo "Checking for existing interactive pods..."
EXISTING_PODS=$(kubectl get pods -n "$NAMESPACE" -o json | jq -r ".items[] | select(.metadata.name | startswith(\"${USER}-iree-\")) | select(.status.phase == \"Running\") | .metadata.name" 2>/dev/null || true)

if [ -n "$EXISTING_PODS" ]; then
    # Get the latest pod (newest timestamp)
    LATEST_POD=$(echo "$EXISTING_PODS" | sort -r | head -1)
    echo ""
    echo "Found existing pod(s):"
    echo "$EXISTING_PODS" | sed 's/^/  - /'
    echo ""
    echo "Using latest pod: $LATEST_POD"
    POD_NAME="$LATEST_POD"
    SKIP_CREATE=true
else
    echo "No existing pods found. Will create new pod: $POD_NAME"
    SKIP_CREATE=false
fi

# kubectl check
if ! command -v kubectl &>/dev/null; then
    echo "Error: kubectl command not found."
    exit 1
fi

if [ "$SKIP_CREATE" = false ]; then
    if [ ! -f "$YAML_TEMPLATE" ]; then
        echo "Error: YAML template '$YAML_TEMPLATE' not found."
        exit 1
    fi

    # Create SSH secret if needed
    if [[ "$MODE" == "ssh" ]]; then
        if [[ ! -f "$PUB_KEY_PATH" ]]; then
            echo "Error: SSH public key not found at $PUB_KEY_PATH"
            exit 1
        fi
        TMPDIR=$(mktemp -d)
        cp "$PUB_KEY_PATH" "$TMPDIR/authorized_keys"
        SECRET_NAME="ssh-key-${USER}"
        kubectl -n "$NAMESPACE" delete secret "$SECRET_NAME" --ignore-not-found
        kubectl -n "$NAMESPACE" create secret generic "$SECRET_NAME" --from-file=authorized_keys="$TMPDIR/authorized_keys"
        rm -rf "$TMPDIR"
        echo "✅ Created/updated SSH key secret in namespace '$NAMESPACE'."
    fi

    # Render YAML
    echo "Preparing YAML from template: $YAML_TEMPLATE"
    sed -e "s/{{POD_NAME}}/${POD_NAME}/g" \
        -e "s/{{PVC_CLAIM_NAME}}/${PVC_CLAIM_NAME}/g" \
        -e "s/{{USER}}/${USER}/g" \
        "$YAML_TEMPLATE" > "$TEMP_YAML"

    echo "Applying '$TEMP_YAML' in namespace '$NAMESPACE'..."
    kubectl apply -f "$TEMP_YAML" -n "$NAMESPACE"

    echo "Waiting for pod '$POD_NAME' to be ready..."
    kubectl wait pod "$POD_NAME" -n "$NAMESPACE" --for=condition=Ready --timeout=300s

    echo "Pod is ready!"
    rm -f "$TEMP_YAML"
fi

# Start port-forward in background
echo ""
echo "Starting port-forward in background..."
PID_FILE="/tmp/kubectl-port-forward-${USER}-${POD_NAME}.pid"

# Kill any existing port-forward for this pod
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Killing existing port-forward process (PID: $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

# Start port-forward in background
if [[ "$MODE" == "web" ]]; then
    kubectl port-forward -n "$NAMESPACE" "$POD_NAME" "$LOCAL_PORT:$REMOTE_PORT" > /dev/null 2>&1 &
    PORT_FORWARD_PID=$!
    echo "$PORT_FORWARD_PID" > "$PID_FILE"
    echo "✅ Port-forward started (PID: $PORT_FORWARD_PID): localhost:$LOCAL_PORT -> pod:$REMOTE_PORT"
    echo "   Access VSCode at: http://localhost:$LOCAL_PORT"
    sleep 1
else
    kubectl port-forward -n "$NAMESPACE" "$POD_NAME" 2222:22 > /dev/null 2>&1 &
    PORT_FORWARD_PID=$!
    echo "$PORT_FORWARD_PID" > "$PID_FILE"
    echo "✅ Port-forward started (PID: $PORT_FORWARD_PID): localhost:2222 -> pod:22"
    sleep 1
fi

# Wait for service to be ready
echo ""
if [[ "$MODE" == "web" ]]; then
    echo "Waiting for code-server to be ready..."
    for i in {1..12}; do
        if kubectl exec -n "$NAMESPACE" "$POD_NAME" -- curl -fs http://localhost:$REMOTE_PORT/ >/dev/null 2>&1; then
            echo "✅ Code-server is ready!"
            break
        fi
        if [ $i -eq 12 ]; then
            echo "⚠ Timeout waiting for code-server"
            exit 1
        fi
        sleep 2
    done
    echo ""
    echo "================================================================"
    echo "  Code-Server Ready"
    echo "================================================================"
    echo "  URL: http://localhost:$LOCAL_PORT"
    echo "  Pod: $POD_NAME"
    echo ""
    echo "  Port-forward running in background (PID: $PORT_FORWARD_PID)"
    echo "  To stop: kill $PORT_FORWARD_PID"
    echo "================================================================"
else
    echo "Waiting for SSH service to be ready..."
    for i in {1..12}; do
        if kubectl exec -n "$NAMESPACE" "$POD_NAME" -- bash -c "timeout 1 bash -c '</dev/tcp/localhost/22'" 2>/dev/null; then
            echo "✅ SSH service is ready!"
            break
        fi
        if [ $i -eq 12 ]; then
            echo "⚠ Timeout waiting for SSH service"
            exit 1
        fi
        sleep 2
    done
    
    echo ""
    echo "================================================================"
    echo "  Interactive Pod Ready"
    echo "================================================================"
    echo "  Pod:  $POD_NAME"
    echo "  SSH:  ssh ossci"
    echo ""
    echo "  Port-forward running in background (PID: $PORT_FORWARD_PID)"
    echo "================================================================"
    echo ""
    echo "Connecting to pod via SSH..."
    echo ""
    
    # SSH into the pod
    ssh ossci
fi

echo ""
echo "================================================================"
echo "  Session Information"
echo "================================================================"
echo "  Pod: $POD_NAME (still running)"
echo "  Port-forward PID: $PORT_FORWARD_PID"
echo ""
echo "  To reconnect:    ssh ossci"
echo "  To stop pod:     $SCRIPT_DIR/stop.sh"
echo "  To kill forward: kill $PORT_FORWARD_PID"
echo "================================================================"
