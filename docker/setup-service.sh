#!/bin/bash
# setup-service.sh - Complete container setup for a given service
#
# Usage: setup-service.sh <service_name>
#
# Caller MUST ensure ~/scripts is on disk before invoking this:
#   - Docker build:  base.dockerfile clones scripts.git into /root/scripts
#   - K8s pod:       PVC mounts scripts/
#   - Manual:        bash ~/scripts/docker/setup-service.sh triton
#
# Runs the full setup sequence (all steps idempotent):
#   1. env/min.sh          - Base packages, dotfiles, rc_files
#   2. env/<service>.sh    - Service-specific dependencies (if exists)
#   3. workspace/<svc>.sh  - Clone repos, setup workspace (if exists)
#   4. env/priv.sh         - SSH keys, credentials from persistent storage
#
# Adding a new service: just create env/<name>.sh and/or workspace/<name>.sh

set -e

SERVICE="${1:?Usage: setup-service.sh <service_name>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Base packages + dotfiles
echo "📦 Base packages (env/min.sh)..."
bash "$SCRIPT_DIR/env/min.sh"

# 2. Service-specific dependencies
if [[ -f "$SCRIPT_DIR/env/${SERVICE}.sh" ]]; then
    echo "📦 Service dependencies (env/${SERVICE}.sh)..."
    bash "$SCRIPT_DIR/env/${SERVICE}.sh"
fi

# 3. Workspace setup
if [[ -f "$SCRIPT_DIR/workspace/${SERVICE}.sh" ]]; then
    echo "📦 Workspace setup (workspace/${SERVICE}.sh)..."
    bash "$SCRIPT_DIR/workspace/${SERVICE}.sh"
fi

# 4. Runtime init (SSH keys, credentials) — no-op if no persistent storage
if [[ -f "$SCRIPT_DIR/env/priv.sh" ]]; then
    echo "🔧 Container init (env/priv.sh)..."
    bash "$SCRIPT_DIR/env/priv.sh"
fi
