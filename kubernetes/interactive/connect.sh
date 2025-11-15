#!/bin/bash
set -euo pipefail

# Default values
DATE=$(date +%m%d)  # Format: MMDD (e.g., 1114)
TIME=$(date +%H%M%S)  # Format: HHMMSS (e.g., 142111)
POD_NAME="${USER}-iree-${DATE}-${TIME}"
LOCAL_PORT="8000"
REMOTE_PORT="9000"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_YAML="${SCRIPT_DIR}/pod-temp.yml"
CONFIG_FILE="${SCRIPT_DIR}/config.json"

# Parse arguments
FORCE_NEW=false
MODE="ssh"
EPHEMERAL=false

for arg in "$@"; do
    case $arg in
        -n|--new)
            FORCE_NEW=true
            shift
            ;;
        -e|--ephemeral)
            EPHEMERAL=true
            shift
            ;;
        web|ssh)
            MODE=$arg
            shift
            ;;
        *)
            echo "Usage: $0 [OPTIONS] <web|ssh>"
            echo ""
            echo "Options:"
            echo "  -n, --new         Force creation of new pod (don't attach to existing)"
            echo "  -e, --ephemeral   Use ephemeral storage (fast startup, no persistence)"
            echo ""
            echo "Modes:"
            echo "  ssh               SSH-accessible interactive pod (default)"
            echo "  web               Browser-based code-server"
            exit 1
            ;;
    esac
done

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
echo "Checking authentication (http://localhost:8000)..."
if ! kubectl get ns >/dev/null 2>&1; then
    echo ""
    echo "⚠ Not authenticated with Kubernetes cluster"
    echo "🔐 Initiating Okta SSO authentication..."
    echo ""
    echo "This will open your browser at http://localhost:8000"
    echo "Please complete the Okta sign-in in your browser."
    echo "Waiting for authentication..."
    echo ""

    # Trigger authentication (show output so user can see what's happening)
    if kubectl get ns 2>&1 | head -3; then
        echo ""
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

# Select YAML template based on mode and storage type
if [[ "$EPHEMERAL" == "true" ]]; then
    YAML_TEMPLATE="${SCRIPT_DIR}/pod-${MODE}-ephemeral.yml"
    echo "🚀 Using ephemeral storage (fast startup, no persistence)"
else
    YAML_TEMPLATE="${SCRIPT_DIR}/pod-${MODE}.yml"
fi

# Check for existing pods
echo "Checking for existing interactive pods..."
EXISTING_PODS=$(kubectl get pods -n "$NAMESPACE" -o json | jq -r ".items[] | select(.metadata.name | startswith(\"${USER}-iree-\")) | select(.status.phase == \"Running\") | .metadata.name" 2>/dev/null || true)

if [ -n "$EXISTING_PODS" ] && [ "$FORCE_NEW" = false ]; then
    # Get the latest pod (newest timestamp)
    LATEST_POD=$(echo "$EXISTING_PODS" | sort -r | head -1)
    POD_COUNT=$(echo "$EXISTING_PODS" | wc -l)
    echo ""
    echo "Found $POD_COUNT existing pod(s):"
    echo "$EXISTING_PODS" | sed 's/^/  - /'
    echo ""
    echo "Using latest pod: $LATEST_POD"
    echo ""
    echo "💡 Tip: To create a new pod instead, run: $0 --new"
    POD_NAME="$LATEST_POD"
    SKIP_CREATE=true
elif [ -n "$EXISTING_PODS" ] && [ "$FORCE_NEW" = true ]; then
    POD_COUNT=$(echo "$EXISTING_PODS" | wc -l)
    echo ""
    echo "Found $POD_COUNT existing pod(s):"
    echo "$EXISTING_PODS" | sed 's/^/  - /'
    echo ""
    echo "Creating new pod (--new flag): $POD_NAME"
    SKIP_CREATE=false
else
    echo "No existing pods found. Creating new pod: $POD_NAME"
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
    echo "(This may take 10-15 minutes on first run while pulling the ROCm image...)"
    kubectl wait pod "$POD_NAME" -n "$NAMESPACE" --for=condition=Ready --timeout=900s

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
    echo "(Waiting for port-forward to establish and SSH daemon to start...)"
    for i in {1..30}; do
        # Test SSH connection via port-forward (localhost:2222 -> pod:22)
        if ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ossci "exit 0" >/dev/null 2>&1; then
            echo "✅ SSH service is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "⚠ Timeout waiting for SSH service (waited 60 seconds)"
            echo "   Pod may still be starting up. Try:"
            echo "   kubectl logs $POD_NAME -n $NAMESPACE"
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

    # Check if setup is needed
    # In ephemeral mode, always run setup (storage is empty)
    # In PVC mode, check if packages are already installed
    NEED_SETUP=false
    if [[ "$EPHEMERAL" == "true" ]]; then
        NEED_SETUP=true
        echo ""
        echo "🔧 Ephemeral mode: Running full setup (storage is empty)..."
        echo ""
    elif ! ssh -o ConnectTimeout=5 ossci "command -v tmux >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1" 2>/dev/null; then
        NEED_SETUP=true
        echo ""
        echo "🔧 New container detected! Running system setup..."
        echo ""
    fi
    
    if [[ "$NEED_SETUP" == "true" ]]; then
        echo "This will:"
        echo "  1. Install system packages (git, zsh, tmux, neovim, cmake, etc.)"
        echo "  2. Setup dotfiles"
        echo "  3. Clone repos"
        echo "  4. Install Python packages"
        echo "  5. Clone IREE repository"
        echo ""
        if [[ "$EPHEMERAL" == "true" ]]; then
            echo "Note: Ephemeral storage - all data deleted when pod stops"
        else
            echo "Note: Existing repos/dotfiles will be reused from PVC"
        fi
        echo "This may take 10-15 minutes..."
        echo ""

        # Ensure scripts repo exists in pod (copy from local)
        echo "📤 Checking for scripts repository in pod..."
        if ! ssh ossci "test -d ~/scripts" 2>/dev/null; then
            echo "   Scripts not found, copying from local..."
            kubectl cp "$HOME/scripts" "$NAMESPACE/$POD_NAME:/home/ossci/" || {
                echo "❌ Failed to copy scripts directory"
                echo "   Please ensure ~/scripts exists locally"
                exit 1
            }
            echo "✅ Scripts copied to pod"
        else
            echo "   Updating scripts from local..."
            # Sync key directories that we actively develop
            kubectl cp "$HOME/scripts/docker" "$NAMESPACE/$POD_NAME:/home/ossci/scripts/" 2>/dev/null || true
            kubectl cp "$HOME/scripts/kubernetes" "$NAMESPACE/$POD_NAME:/home/ossci/scripts/" 2>/dev/null || true
            echo "✅ Scripts synchronized"
        fi
        echo ""

        # Run init_min.sh (installs system packages, sets up dotfiles)
        echo "📦 Step 1/2: Running init_min.sh (system packages + dotfiles)..."
        ssh ossci "cd ~ && bash scripts/docker/init_min.sh" || {
            echo "❌ init_min.sh failed. Please check the pod and run manually:"
            echo "   ssh ossci"
            echo "   bash ~/scripts/docker/init_min.sh"
            exit 1
        }

        # Run init_iree.sh (installs cmake, python packages)
        echo ""
        echo "📦 Step 2/2: Running init_iree.sh (IREE dependencies)..."
        ssh ossci "cd ~ && bash scripts/docker/init_iree.sh" || {
            echo "❌ init_iree.sh failed. Please check the pod and run manually:"
            echo "   ssh ossci"
            echo "   bash ~/scripts/docker/init_iree.sh"
            exit 1
        }

        echo ""
        echo "✅ Container setup complete!"
        echo ""
    else
        echo ""
        echo "✅ Container already set up (reusing existing pod)"
        echo ""
    fi

    # Setup isolated workspace for this pod
    echo "Setting up isolated workspace..."

    # Run workspace setup script from scripts repo
    ssh ossci "bash ~/scripts/kubernetes/interactive/setup-workspace.sh" || {
        echo "❌ setup-workspace.sh failed. Please check the pod and run manually:"
        echo "   ssh ossci"
        echo "   bash ~/scripts/kubernetes/interactive/setup-workspace.sh"
        exit 1
    }

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
