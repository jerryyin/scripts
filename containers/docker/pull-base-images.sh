#!/bin/bash
# Pull the FROM images selected by rc_files Compose services, without building them.
set -e
BASE_PULL_SERVICES="${*:-triton triton-mi450}"
echo "🐳 Pulling base images ($BASE_PULL_SERVICES)..."
# `docker compose pull <svc>` would fetch the service's own tag
# (jeryin/dev:triton), a local-only tag never pushed anywhere -- not the
# Dockerfile's `FROM $BASE_IMAGE`. No compose subcommand fetches just the
# base image, so read the resolved BASE_IMAGE out of compose's own JSON.
COMPOSE_FILE="${COMPOSE_FILE:-$HOME/rc_files/docker/.docker/docker-compose.yml}"
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "⚠️  $COMPOSE_FILE not found — skipping base-image pulls"
elif ! docker compose version >/dev/null 2>&1; then
    echo "⚠️  'docker compose' not available — skipping base-image pulls"
elif ! command -v python3 >/dev/null 2>&1; then
    echo "⚠️  python3 not available — skipping base-image pulls"
else
    # Both flags are load-bearing; without either, `config` fails outright
    # and every service below reports an unresolvable BASE_IMAGE.
    #   --profile: a service outside the active profile set isn't in the
    #     rendered model at all, so its BASE_IMAGE isn't there to read.
    #   --no-consistency: the dev services deliberately share one
    #     container_name (only one runs at a time), and rendering several
    #     at once trips the duplicate-name check even for a read-only dump.
    PROFILE_FLAGS=""
    for svc in $BASE_PULL_SERVICES; do
        PROFILE_FLAGS="$PROFILE_FLAGS --profile $svc"
    done
    # shellcheck disable=SC2086
    CONFIG_JSON=$(docker compose -f "$COMPOSE_FILE" $PROFILE_FLAGS config --no-consistency --format json 2>/dev/null)
    for svc in $BASE_PULL_SERVICES; do
        IMAGE=$(printf '%s' "$CONFIG_JSON" | python3 -c '
import json, sys
cfg = json.load(sys.stdin)
print(cfg.get("services", {}).get(sys.argv[1], {}).get("build", {}).get("args", {}).get("BASE_IMAGE", ""))
' "$svc" 2>/dev/null)
        if [ -z "$IMAGE" ]; then
            echo "  ⚠️  Could not resolve BASE_IMAGE for $svc — skipping"
            continue
        fi
        echo "  Pulling $IMAGE (base image for $svc)... (docker pull is incremental -- a no-op fetch if already up to date)"
        docker pull "$IMAGE" || echo "  ⚠️  Failed to pull $IMAGE"
    done
fi
