#!/bin/bash
set -euo pipefail

# Default values
DATE=$(date +%m%d)  # Format: MMDD (e.g., 1114)
TIME=$(date +%H%M%S)  # Format: HHMMSS (e.g., 142111)
LOCAL_PORT="8000"
REMOTE_PORT="9000"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(cd "$SCRIPT_DIR/../../docker" 2>/dev/null && pwd)" || DOCKER_DIR="$HOME/scripts/docker"
TEMP_YAML="${SCRIPT_DIR}/pod-temp.yml"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
PORT_MAPPING_FILE="$HOME/.kube/pod-port-mappings.json"

# Parse arguments
FORCE_NEW=false
MODE="ssh"
EPHEMERAL=false
SKIP_SETUP=false
SERVICE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--new)
            FORCE_NEW=true
            shift
            ;;
        -e|--ephemeral)
            EPHEMERAL=true
            shift
            ;;
        --skip-setup|--no-setup)
            SKIP_SETUP=true
            shift
            ;;
        -s|--service)
            SERVICE="$2"
            shift 2
            ;;
        web|ssh)
            MODE=$1
            shift
            ;;
        *)
            echo "Usage: $0 [OPTIONS] [web|ssh]"
            echo ""
            echo "Options:"
            echo "  -n, --new              Force creation of new pod (don't attach to existing)"
            echo "  -e, --ephemeral        Use ephemeral storage (fast startup, no persistence)"
            echo "  -s, --service <name>   Service/project to set up (e.g., iree, triton)"
            echo "  --skip-setup           Skip pod setup (packages, dotfiles, workspace)"
            echo ""
            echo "Modes:"
            echo "  ssh                    SSH-accessible interactive pod (default)"
            echo "  web                    Browser-based code-server"
            echo ""
            echo "Examples:"
            echo "  $0                     # Interactive service picker, SSH mode"
            echo "  $0 -s triton -e        # Triton, ephemeral"
            echo "  $0 -s iree --new       # Force new IREE pod"
            exit 1
            ;;
    esac
done

# ============================================================================
# Service Discovery (from docker-compose.yml)
# ============================================================================
# Services are discovered from the docker-compose.yml in rc_files.
# Only interactive services (with tty+stdin_open and priv.sh) are shown.

COMPOSE_FILE="${COMPOSE_FILE:-$HOME/rc_files/docker/.docker/docker-compose.yml}"

discover_services() {
    python3 -c "
import yaml, sys
try:
    with open('$COMPOSE_FILE') as f:
        data = yaml.safe_load(f)
except Exception as e:
    print(f'Error reading {sys.argv[0]}: {e}', file=sys.stderr)
    sys.exit(1)
for name, svc in data.get('services', {}).items():
    cmd = svc.get('command', [])
    is_interactive = svc.get('tty', False) and svc.get('stdin_open', False)
    has_priv = any('priv.sh' in str(c) for c in (cmd if isinstance(cmd, list) else [cmd]))
    if is_interactive and has_priv:
        base_image = svc.get('build', {}).get('args', {}).get('BASE_IMAGE', svc.get('image', ''))
        print(f'{name}\t{base_image}')
"
}

pick_service() {
    local -a names=()
    local -a images=()

    while IFS=$'\t' read -r name image; do
        names+=("$name")
        images+=("$image")
    done < <(discover_services)

    if [[ ${#names[@]} -eq 0 ]]; then
        echo "No interactive services found in $COMPOSE_FILE" >&2
        exit 1
    fi

    if [[ ${#names[@]} -eq 1 ]]; then
        echo "${names[0]}"
        return
    fi

    echo "" >&2
    echo "Available services (from docker-compose.yml):" >&2
    for i in "${!names[@]}"; do
        echo "  $((i+1))) ${names[$i]}  (${images[$i]})" >&2
    done
    echo "" >&2

    while true; do
        read -rp "Select service [1-${#names[@]}]: " selection
        if [[ "$selection" =~ ^[0-9]+$ ]] && (( selection >= 1 && selection <= ${#names[@]} )); then
            echo "${names[$((selection-1))]}"
            return
        fi
        echo "Invalid selection." >&2
    done
}

# ============================================================================
# Port Management Functions
# ============================================================================

# Get allocated port for a pod, or allocate a new one
get_pod_port() {
    local pod_name="$1"
    local mode="$2"  # "ssh" or "web"

    # Base port depends on mode
    local base_port
    if [[ "$mode" == "web" ]]; then
        base_port=8000
    else
        base_port=2222
    fi

    # Initialize port mapping file if it doesn't exist
    if [ ! -f "$PORT_MAPPING_FILE" ]; then
        echo "{}" > "$PORT_MAPPING_FILE"
    fi

    # Check if this pod already has a port assigned
    local existing_port
    existing_port=$(jq -r --arg pod "$pod_name" '.[$pod] // empty' "$PORT_MAPPING_FILE" 2>/dev/null || echo "")

    if [ -n "$existing_port" ]; then
        echo "$existing_port"
        return
    fi

    # Find next available port
    local port=$base_port
    local max_attempts=100
    local attempts=0

    while [ $attempts -lt $max_attempts ]; do
        # Check if port is in use by any pod
        local port_in_use
        port_in_use=$(jq -r --arg port "$port" 'to_entries[] | select(.value == ($port | tonumber)) | .key' "$PORT_MAPPING_FILE" 2>/dev/null || echo "")

        if [ -z "$port_in_use" ]; then
            # Port is available, assign it
            local tmp_file
            tmp_file=$(mktemp)
            jq --arg pod "$pod_name" --argjson port "$port" '. + {($pod): $port}' "$PORT_MAPPING_FILE" > "$tmp_file"
            mv "$tmp_file" "$PORT_MAPPING_FILE"
            echo "$port"
            return
        fi

        port=$((port + 1))
        attempts=$((attempts + 1))
    done

    echo "Error: Could not find available port after $max_attempts attempts" >&2
    exit 1
}

# Clean up port mapping for a pod
cleanup_pod_port() {
    local pod_name="$1"

    if [ -f "$PORT_MAPPING_FILE" ]; then
        local tmp_file
        tmp_file=$(mktemp)
        jq --arg pod "$pod_name" 'del(.[$pod])' "$PORT_MAPPING_FILE" > "$tmp_file"
        mv "$tmp_file" "$PORT_MAPPING_FILE"
    fi
}

# No SSH config management - users connect via: ssh -p <port> ossci@localhost

# SSH command options as array (proper argument splitting)
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ForwardAgent=yes -i ~/.ssh/id_rsa)

# ============================================================================

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
DEFAULT_SERVICE=$(jq -r '.default_service // "iree"' "$CONFIG_FILE")

# Resolve service: flag > default > interactive picker
if [[ -z "$SERVICE" ]]; then
    if [[ "$FORCE_NEW" == "true" ]]; then
        SERVICE=$(pick_service)
    else
        SERVICE="$DEFAULT_SERVICE"
    fi
fi

# Validate service and extract its base image from docker-compose.yml
SERVICE_LINE=$(discover_services | grep "^${SERVICE}	" || true)
if [[ -z "$SERVICE_LINE" ]]; then
    echo "Error: Service '$SERVICE' not found in $COMPOSE_FILE"
    echo "Available services:"
    discover_services | while IFS=$'\t' read -r n i; do echo "  - $n ($i)"; done
    exit 1
fi
DOCKER_IMAGE=$(echo "$SERVICE_LINE" | cut -f2)

POD_NAME="${USER}-${SERVICE}-${DATE}-${TIME}"
POD_PREFIX="${USER}-${SERVICE}-"

echo "📋 Service: $SERVICE (image: $DOCKER_IMAGE)"

# Select YAML template based on mode and storage type
if [[ "$EPHEMERAL" == "true" ]]; then
    YAML_TEMPLATE="${SCRIPT_DIR}/pod-${MODE}-ephemeral.yml"
    echo "🚀 Using ephemeral storage (fast startup, no persistence)"
else
    YAML_TEMPLATE="${SCRIPT_DIR}/pod-${MODE}.yml"
fi

# Check for existing pods (match current service prefix)
echo "Checking for existing interactive pods..."
EXISTING_PODS=$(kubectl get pods -n "$NAMESPACE" -o json | jq -r ".items[] | select(.metadata.name | startswith(\"${POD_PREFIX}\")) | select(.status.phase == \"Running\") | .metadata.name" 2>/dev/null || true)

if [ -n "$EXISTING_PODS" ] && [ "$FORCE_NEW" = false ]; then
    POD_COUNT=$(echo "$EXISTING_PODS" | wc -l)
    echo ""
    echo "Found $POD_COUNT existing pod(s):"
    echo ""

    # Convert to array for menu
    readarray -t POD_ARRAY <<< "$EXISTING_PODS"

    if [ ${#POD_ARRAY[@]} -eq 1 ]; then
        # Only one pod, use it directly
        POD_NAME="${POD_ARRAY[0]}"
        echo "  1) ${POD_NAME} (connecting...)"
        echo ""
        echo "💡 Tip: To create a new pod instead, run: $0 --new"
        SKIP_CREATE=true
    else
        # Multiple pods, show menu
        for i in "${!POD_ARRAY[@]}"; do
            pod="${POD_ARRAY[$i]}"
            port=$(get_pod_port "$pod" "$MODE")
            echo "  $((i+1))) $pod (port $port)"
        done
        echo "  $((${#POD_ARRAY[@]}+1))) Create new pod"
        echo ""

        # Get user selection
        while true; do
            read -p "Select pod to connect to [1-$((${#POD_ARRAY[@]}+1))]:" selection

            if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le $((${#POD_ARRAY[@]}+1)) ]; then
                if [ "$selection" -eq $((${#POD_ARRAY[@]}+1)) ]; then
                    # Create new pod
                    echo ""
                    echo "Creating new pod: $POD_NAME"
                    SKIP_CREATE=false
                else
                    # Connect to existing pod
                    POD_NAME="${POD_ARRAY[$((selection-1))]}"
                    echo ""
                    echo "Connecting to: $POD_NAME"
                    SKIP_CREATE=true
                fi
                break
            else
                echo "Invalid selection. Please enter a number between 1 and $((${#POD_ARRAY[@]}+1))"
            fi
        done
    fi
    echo ""
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
    sed -e "s|{{POD_NAME}}|${POD_NAME}|g" \
        -e "s|{{PVC_CLAIM_NAME}}|${PVC_CLAIM_NAME}|g" \
        -e "s|{{USER}}|${USER}|g" \
        -e "s|{{DOCKER_IMAGE}}|${DOCKER_IMAGE}|g" \
        "$YAML_TEMPLATE" > "$TEMP_YAML"

    echo "Applying '$TEMP_YAML' in namespace '$NAMESPACE'..."
    kubectl apply -f "$TEMP_YAML" -n "$NAMESPACE"

    echo "Waiting for pod '$POD_NAME' to be ready..."
    echo "(This may take time on first run while 1. pulling the ROCm image... 2. setting up pvc)"
    kubectl wait pod "$POD_NAME" -n "$NAMESPACE" --for=condition=Ready --timeout=1200s

    echo "Pod is ready!"
    rm -f "$TEMP_YAML"
fi

# Get or allocate port for this pod
SSH_PORT=$(get_pod_port "$POD_NAME" "$MODE")

# Start port-forward in background
echo ""
if [[ "$MODE" == "web" ]]; then
    echo "Starting port-forward: localhost:$SSH_PORT -> pod:$REMOTE_PORT"
else
    echo "Starting port-forward: localhost:$SSH_PORT -> pod:22"
fi
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

PORT_FORWARD_PID=""

start_port_forward() {
    if [[ "$MODE" == "web" ]]; then
        kubectl port-forward -n "$NAMESPACE" "$POD_NAME" "$SSH_PORT:$REMOTE_PORT" >/dev/null 2>&1 &
    else
        kubectl port-forward -n "$NAMESPACE" "$POD_NAME" "$SSH_PORT:22" >/dev/null 2>&1 &
    fi
    PORT_FORWARD_PID=$!
    echo "$PORT_FORWARD_PID" > "$PID_FILE"
    sleep 2
}

start_port_forward

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
    echo "(Waiting for port-forward to establish and SSH daemon to start; press Ctrl+C to abort)"
    SSH_WAIT_ATTEMPTS=0
    while true; do
        ((++SSH_WAIT_ATTEMPTS))
        if ! ps -p "$PORT_FORWARD_PID" >/dev/null 2>&1; then
            echo "↻ Port-forward exited; restarting..."
            start_port_forward
        fi
        if ssh -o ConnectTimeout=2 "${SSH_OPTS[@]}" -p "$SSH_PORT" ossci@localhost "exit 0" >/dev/null 2>&1; then
            echo "✅ SSH service is ready!"
            break
        fi
        if (( SSH_WAIT_ATTEMPTS % 30 == 0 )); then
            echo "  ...still waiting for SSH (attempt ${SSH_WAIT_ATTEMPTS}). Pod may still be configuring."
        fi
        sleep 2
    done

    # Pod is ready for SSH connection
    echo ""
    echo "================================================================"
    echo "  Interactive Pod Ready"
    echo "================================================================"
    echo "  Pod:   $POD_NAME"
    echo "  SSH:   ssh -i ~/.ssh/id_rsa -p $SSH_PORT ossci@localhost"
    echo "  Port:  $SSH_PORT"
    echo ""
    echo "  Port-forward running in background (PID: $PORT_FORWARD_PID)"
    echo "================================================================"
    echo ""
    echo "Connecting to pod via SSH..."
    echo ""

    # Run pod setup (unless --skip-setup flag was used)
    if [[ "$SKIP_SETUP" == "false" ]]; then
        # Check if setup is needed
        echo "🔍 Checking if pod setup is needed..."
        NEED_SETUP=false
        if [[ "$EPHEMERAL" == "true" ]]; then
            NEED_SETUP=true
            echo "   Ephemeral mode detected - setup required"
        elif ! ssh -o ConnectTimeout=5 "${SSH_OPTS[@]}" -p "$SSH_PORT" ossci@localhost "command -v tmux >/dev/null 2>&1 && command -v cmake >/dev/null 2>&1" 2>/dev/null; then
            NEED_SETUP=true
            echo "   New container detected - setup required"
        else
            echo "   ✅ Container already configured (packages detected)"
        fi
        echo ""

        if [[ "$NEED_SETUP" == "true" ]]; then
            # Ensure scripts repo exists in pod (copy from local)
            echo "📤 Copying scripts to pod..."
            if ! ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" ossci@localhost "test -d ~/scripts" 2>/dev/null; then
                echo "   Scripts not found, copying entire directory..."
                kubectl cp "$HOME/scripts" "$NAMESPACE/$POD_NAME:/home/ossci/" || {
                    echo "❌ Failed to copy scripts directory"
                    echo "   Please ensure ~/scripts exists locally"
                    exit 1
                }
                echo "   ✅ Scripts copied to pod"
            else
                echo "   Scripts exist, updating key directories..."
                # Sync key directories that we actively develop
                kubectl cp "$HOME/scripts/docker" "$NAMESPACE/$POD_NAME:/home/ossci/scripts/" 2>/dev/null || true
                kubectl cp "$HOME/scripts/kubernetes" "$NAMESPACE/$POD_NAME:/home/ossci/scripts/" 2>/dev/null || true
                echo "   ✅ Scripts synchronized"
            fi
            echo ""

            echo "🚀 Running setup inside pod..."
            echo "════════════════════════════════════════════════════════════════"
            ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" ossci@localhost "bash ~/scripts/docker/setup-service.sh ${SERVICE}" || {
                echo ""
                echo "════════════════════════════════════════════════════════════════"
                echo "❌ Pod setup failed. You can retry manually:"
                echo "   ssh -i ~/.ssh/id_rsa -p $SSH_PORT ossci@localhost"
                echo "   bash ~/scripts/docker/setup-service.sh $SERVICE"
                echo ""
                echo "   Or skip setup: ./connect.sh --skip-setup"
                echo "════════════════════════════════════════════════════════════════"
                exit 1
            }
            echo "════════════════════════════════════════════════════════════════"
            echo ""
        fi
    else
        echo "⏭️  Skipping pod setup (--skip-setup flag used)"
        echo ""
    fi

    # SSH into the pod
    ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" ossci@localhost
fi

echo ""
echo "================================================================"
echo "  Session Information"
echo "================================================================"
echo "  Pod:  $POD_NAME (still running)"
echo "  SSH:  ssh -i ~/.ssh/id_rsa -p $SSH_PORT ossci@localhost"
echo "  Port: $SSH_PORT"
echo "  Port-forward PID: $PORT_FORWARD_PID"
echo ""
echo "  To reconnect:    ssh -i ~/.ssh/id_rsa -p $SSH_PORT ossci@localhost"
echo "  To stop pod:     $SCRIPT_DIR/stop.sh"
echo "  To kill forward: kill $PORT_FORWARD_PID"
echo "================================================================"
